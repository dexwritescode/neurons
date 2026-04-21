#include "llama_model.h"
#include "model_loader.h"
#include "sampler.h"
#include "../core/compute_backend.h"
#include <algorithm>
#include <cmath>
#include <sstream>

namespace compute {

// ── Private constructor ───────────────────────────────────────────────────────

LlamaModel::LlamaModel(
    ModelConfig                             config,
    SimpleBpeTokenizer                      tokenizer,
    std::unordered_map<std::string, Tensor> weights,
    ComputeBackend*                         backend)
    : config_(std::move(config))
    , tokenizer_(std::move(tokenizer))
    , weights_(std::move(weights))
    , backend_(backend)
    , tool_family_(detect_tool_family(tokenizer_, config_))
{}

// ── Factory ───────────────────────────────────────────────────────────────────

Result<LlamaModel> LlamaModel::from_model_dir(
    const std::filesystem::path& model_dir,
    ComputeBackend*              backend)
{
    if (!backend) {
        return std::unexpected(Error{ErrorCode::InvalidArgument, "Backend cannot be null"});
    }

    auto model_result = ModelLoader::load_model(model_dir, backend);
    if (!model_result) return std::unexpected(model_result.error());

    auto& [config, weights] = *model_result;

    auto tokenizer_result = SimpleBpeTokenizer::from_model_dir(model_dir);
    if (!tokenizer_result) return std::unexpected(tokenizer_result.error());

    if (!config.is_valid()) {
        return std::unexpected(Error{ErrorCode::InvalidModel, "Invalid model configuration"});
    }
    if (!config.is_supported_architecture()) {
        return std::unexpected(Error{ErrorCode::InvalidModel,
            "Unsupported model architecture: " + config.model_type});
    }

    return LlamaModel(
        std::move(config),
        std::move(*tokenizer_result),
        std::move(weights),
        backend);
}

// ── Weight lookup ─────────────────────────────────────────────────────────────

Result<Tensor> LlamaModel::get_weight(const std::string& name) const {
    auto it = weights_.find(name);
    if (it == weights_.end()) {
        return std::unexpected(Error{ErrorCode::TensorNotFound, "Weight not found: " + name});
    }
    return it->second;
}

// ── Linear projection (quantized or unquantized) ─────────────────────────────

Result<Tensor> LlamaModel::linear(const Tensor& input, const std::string& weight_key) {
    auto w = get_weight(weight_key + ".weight");
    if (!w) return std::unexpected(w.error());

    auto s_it = weights_.find(weight_key + ".scales");
    if (s_it != weights_.end()) {
        // Quantized path (mlx-community int4/int8): use quantized_matmul
        const int gs   = config_.quantization ? config_.quantization->group_size : 64;
        const int bits = config_.quantization ? config_.quantization->bits : 4;
        auto b_it = weights_.find(weight_key + ".biases");
        const Tensor* biases = (b_it != weights_.end()) ? &b_it->second : nullptr;
        return backend_->quantized_matmul(input, *w, s_it->second, biases, true, gs, bits);
    }

    // Unquantized path (fp16/bf16): weight is stored [out, in] — transpose before matmul
    auto w_t = backend_->swapaxes(*w, 0, 1);
    if (!w_t) return std::unexpected(w_t.error());
    return backend_->matmul(input, *w_t);
}

// ── Embedding ─────────────────────────────────────────────────────────────────

Result<Tensor> LlamaModel::embedding(const std::vector<int>& token_ids) {
    if (token_ids.empty()) {
        return std::unexpected(Error{ErrorCode::InvalidInput, "Empty token_ids"});
    }

    // Dequantize and cache the embedding table on first use.
    if (!dequantized_embed_tokens_.has_value()) {
        auto w = get_weight("model.embed_tokens.weight");
        if (!w) return std::unexpected(w.error());

        auto scales = get_weight("model.embed_tokens.scales");
        if (scales) {
            auto biases = get_weight("model.embed_tokens.biases");
            if (!biases) return std::unexpected(biases.error());
            const int gs   = config_.quantization ? config_.quantization->group_size : 64;
            const int bits = config_.quantization ? config_.quantization->bits : 4;
            auto deq = backend_->dequantize(*w, *scales, *biases, gs, bits);
            if (!deq) return std::unexpected(deq.error());
            dequantized_embed_tokens_ = std::move(*deq);
        } else {
            dequantized_embed_tokens_ = std::move(*w);
        }
    }
    const auto& embed_weight = *dequantized_embed_tokens_;

    if (embed_weight.shape().size() != 2) {
        return std::unexpected(Error{ErrorCode::InvalidModel,
            "Embedding weight must be 2D"});
    }

    const size_t vocab_size  = embed_weight.shape()[0];
    const size_t hidden_size = embed_weight.shape()[1];

    for (int id : token_ids) {
        if (id < 0 || static_cast<size_t>(id) >= vocab_size) {
            return std::unexpected(Error{ErrorCode::InvalidInput,
                "Token ID " + std::to_string(id) + " out of range [0, " +
                std::to_string(vocab_size) + ")"});
        }
    }

    std::vector<Tensor> rows;
    rows.reserve(token_ids.size());
    for (int id : token_ids) {
        auto row = backend_->slice(embed_weight, id, id + 1, 0);
        if (!row) return std::unexpected(row.error());
        auto vec = backend_->reshape(*row, {1, hidden_size});
        if (!vec) return std::unexpected(vec.error());
        rows.push_back(*vec);
    }

    auto result = backend_->concatenate(rows, 0);
    if (!result) return std::unexpected(result.error());

    const auto& shape = result->shape();
    if (shape.size() != 2 || shape[0] != token_ids.size() || shape[1] != hidden_size) {
        return std::unexpected(Error{ErrorCode::ComputeError, "Embedding shape mismatch"});
    }
    return *result;
}

// ── RMSNorm ───────────────────────────────────────────────────────────────────

Result<Tensor> LlamaModel::rms_norm(const Tensor& input, const Tensor& weight, float eps) {
    return backend_->rms_norm(input, weight, eps);
}

// ── Attention layer ───────────────────────────────────────────────────────────

// Public no-cache shim used by tests
Result<Tensor> LlamaModel::attention_layer(const Tensor& input, int layer_idx) {
    return attention_layer(input, layer_idx, 0, nullptr);
}

Result<Tensor> LlamaModel::attention_layer(
    const Tensor& input,
    int           layer_idx,
    int           position_offset,
    LayerKVCache* cache)
{
    if (input.shape().size() != 2) {
        return std::unexpected(Error{ErrorCode::InvalidInput,
            "Attention input must be 2D [seq_len, hidden_size]"});
    }

    const size_t seq_len     = input.shape()[0];
    const size_t hidden_size = input.shape()[1];
    const size_t n_heads     = config_.num_attention_heads;
    const size_t n_kv_heads  = config_.num_key_value_heads;
    const size_t head_dim    = hidden_size / n_heads;
    const float  scale       = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const std::string prefix = "model.layers." + std::to_string(layer_idx) + ".self_attn.";

    // ── QKV projections ───────────────────────────────────────────────────────
    // linear() dispatches to quantized_matmul or matmul based on whether
    // {proj}.scales exists — works for both int4 and fp16/bf16 weights.
    // Separately, Qwen2 adds a learned attention bias (*q_proj.bias*) after the
    // projection. We always probe for it — absent in Llama/Mistral, so no-op there.
    auto queries_flat = linear(input, prefix + "q_proj");
    if (!queries_flat) return std::unexpected(queries_flat.error());
    {
        auto it = weights_.find(prefix + "q_proj.bias");
        if (it != weights_.end()) {
            queries_flat = backend_->add(*queries_flat, it->second);
            if (!queries_flat) return std::unexpected(queries_flat.error());
        }
    }

    auto keys_flat = linear(input, prefix + "k_proj");
    if (!keys_flat) return std::unexpected(keys_flat.error());
    {
        auto it = weights_.find(prefix + "k_proj.bias");
        if (it != weights_.end()) {
            keys_flat = backend_->add(*keys_flat, it->second);
            if (!keys_flat) return std::unexpected(keys_flat.error());
        }
    }

    auto values_flat = linear(input, prefix + "v_proj");
    if (!values_flat) return std::unexpected(values_flat.error());
    {
        auto it = weights_.find(prefix + "v_proj.bias");
        if (it != weights_.end()) {
            values_flat = backend_->add(*values_flat, it->second);
            if (!values_flat) return std::unexpected(values_flat.error());
        }
    }

    // ── Reshape → [heads, seq, head_dim] ──────────────────────────────────────
    auto q3 = backend_->reshape(*queries_flat, {seq_len, n_heads,    head_dim});
    if (!q3) return std::unexpected(q3.error());
    auto qt = backend_->swapaxes(*q3, 0, 1);
    if (!qt) return std::unexpected(qt.error());

    auto k3 = backend_->reshape(*keys_flat,   {seq_len, n_kv_heads, head_dim});
    if (!k3) return std::unexpected(k3.error());
    auto kt = backend_->swapaxes(*k3, 0, 1);
    if (!kt) return std::unexpected(kt.error());

    auto v3 = backend_->reshape(*values_flat, {seq_len, n_kv_heads, head_dim});
    if (!v3) return std::unexpected(v3.error());
    auto vt = backend_->swapaxes(*v3, 0, 1);
    if (!vt) return std::unexpected(vt.error());

    // ── RoPE ──────────────────────────────────────────────────────────────────
    auto q_rope = backend_->rope(*qt, head_dim, config_.rope_theta, position_offset);
    if (!q_rope) return std::unexpected(q_rope.error());
    auto k_rope = backend_->rope(*kt, head_dim, config_.rope_theta, position_offset);
    if (!k_rope) return std::unexpected(k_rope.error());

    // ── KV cache update ───────────────────────────────────────────────────────
    Tensor full_k = *k_rope;
    Tensor full_v = *vt;
    std::string attn_mask = "causal";

    if (cache) {
        if (!cache->valid) {
            cache->keys   = *k_rope;
            cache->values = *vt;
            cache->valid  = true;
        } else {
            auto cat_k = backend_->concatenate({*cache->keys,   *k_rope}, 1);
            if (!cat_k) return std::unexpected(cat_k.error());
            auto cat_v = backend_->concatenate({*cache->values, *vt},    1);
            if (!cat_v) return std::unexpected(cat_v.error());
            cache->keys   = *cat_k;
            cache->values = *cat_v;
            full_k = *cat_k;
            full_v = *cat_v;
        }
        if (seq_len == 1) attn_mask = "";
    }

    // ── SDPA ──────────────────────────────────────────────────────────────────
    auto attn_out = backend_->scaled_dot_product_attention(*q_rope, full_k, full_v, scale, attn_mask);
    if (!attn_out) return std::unexpected(attn_out.error());

    // ── Reshape output → [seq, hidden_size] ───────────────────────────────────
    auto attn_t    = backend_->swapaxes(*attn_out, 0, 1);
    if (!attn_t) return std::unexpected(attn_t.error());
    auto attn_flat = backend_->reshape(*attn_t, {seq_len, hidden_size});
    if (!attn_flat) return std::unexpected(attn_flat.error());

    // ── Output projection ─────────────────────────────────────────────────────
    return linear(*attn_flat, prefix + "o_proj");
}

// ── MLP layer ─────────────────────────────────────────────────────────────────

Result<Tensor> LlamaModel::mlp_layer(const Tensor& input, int layer_idx) {
    const std::string prefix = "model.layers." + std::to_string(layer_idx) + ".mlp.";

    auto gate = linear(input, prefix + "gate_proj");
    if (!gate) return std::unexpected(gate.error());

    auto up = linear(input, prefix + "up_proj");
    if (!up) return std::unexpected(up.error());

    auto activated = backend_->silu(*gate);
    if (!activated) return std::unexpected(activated.error());
    auto hidden = backend_->multiply(*activated, *up);
    if (!hidden) return std::unexpected(hidden.error());

    return linear(*hidden, prefix + "down_proj");
}

// ── Transformer block ─────────────────────────────────────────────────────────

Result<Tensor> LlamaModel::transformer_block(
    const Tensor& input, int layer_idx,
    int position_offset, LayerKVCache* cache)
{
    const std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";

    auto pre_norm_w = get_weight(prefix + "input_layernorm.weight");
    if (!pre_norm_w) return std::unexpected(pre_norm_w.error());
    auto normed = rms_norm(input, *pre_norm_w, config_.rms_norm_eps);
    if (!normed) return std::unexpected(normed.error());

    auto attn_out = attention_layer(*normed, layer_idx, position_offset, cache);
    if (!attn_out) return std::unexpected(attn_out.error());

    auto residual1 = backend_->add(input, *attn_out);
    if (!residual1) return std::unexpected(residual1.error());

    auto post_norm_w = get_weight(prefix + "post_attention_layernorm.weight");
    if (!post_norm_w) return std::unexpected(post_norm_w.error());
    auto normed2 = rms_norm(*residual1, *post_norm_w, config_.rms_norm_eps);
    if (!normed2) return std::unexpected(normed2.error());

    auto mlp_out = mlp_layer(*normed2, layer_idx);
    if (!mlp_out) return std::unexpected(mlp_out.error());

    return backend_->add(*residual1, *mlp_out);
}

// ── Internal forward ─────────────────────────────────────────────────────────

Result<std::vector<float>> LlamaModel::forward_impl(
    const std::vector<int>&    input_ids,
    int                        position_offset,
    std::vector<LayerKVCache>* cache_vec)
{
    auto hidden = embedding(input_ids);
    if (!hidden) return std::unexpected(hidden.error());

    for (int i = 0; i < static_cast<int>(config_.num_hidden_layers); ++i) {
        LayerKVCache* layer_cache = cache_vec ? &(*cache_vec)[i] : nullptr;
        auto block_out = transformer_block(*hidden, i, position_offset, layer_cache);
        if (!block_out) return std::unexpected(block_out.error());
        hidden = std::move(block_out);
    }

    auto norm_w = get_weight("model.norm.weight");
    if (!norm_w) return std::unexpected(norm_w.error());
    auto normed = rms_norm(*hidden, *norm_w, config_.rms_norm_eps);
    if (!normed) return std::unexpected(normed.error());

    // ── LM head projection ───────────────────────────────────────────────────
    // When tie_word_embeddings=true (e.g. Llama-3), there is no separate lm_head —
    // the model reuses the embedding table transposed. Fall back to the dequantized
    // embed_tokens weight and an unquantized matmul in that case.
    Result<Tensor> logits_tensor{std::unexpected(Error{ErrorCode::TensorNotFound, "unset"})};
    if (weights_.count("lm_head.weight")) {
        logits_tensor = linear(*normed, "lm_head");
    } else if (config_.tie_word_embeddings) {
        // Ensure embedding table is dequantized (embedding() is normally called first,
        // but forward_impl may be reached via forward_logits which doesn't go through it).
        if (!dequantized_embed_tokens_.has_value()) {
            auto ew = get_weight("model.embed_tokens.weight");
            if (!ew) return std::unexpected(ew.error());
            auto es = get_weight("model.embed_tokens.scales");
            if (es) {
                const int gs   = config_.quantization ? config_.quantization->group_size : 64;
                const int bits = config_.quantization ? config_.quantization->bits : 4;
                auto eb = get_weight("model.embed_tokens.biases");
                if (!eb) return std::unexpected(eb.error());
                auto deq = backend_->dequantize(*ew, *es, *eb, gs, bits);
                if (!deq) return std::unexpected(deq.error());
                dequantized_embed_tokens_ = std::move(*deq);
            } else {
                dequantized_embed_tokens_ = std::move(*ew);
            }
        }
        // normed: [seq, hidden] × embed_tokens^T: [hidden, vocab] → [seq, vocab]
        auto embed_t = backend_->swapaxes(*dequantized_embed_tokens_, 0, 1);
        if (!embed_t) return std::unexpected(embed_t.error());
        logits_tensor = backend_->matmul(*normed, *embed_t);
    } else {
        return std::unexpected(Error{ErrorCode::TensorNotFound,
            "lm_head.weight not found and tie_word_embeddings is false"});
    }
    if (!logits_tensor) return std::unexpected(logits_tensor.error());

    const size_t seq_len    = input_ids.size();
    const size_t vocab_size = config_.vocab_size;

    auto last = backend_->slice(*logits_tensor,
                                static_cast<int>(seq_len - 1),
                                static_cast<int>(seq_len), 0);
    if (!last) return std::unexpected(last.error());
    auto flat = backend_->reshape(*last, {vocab_size});
    if (!flat) return std::unexpected(flat.error());

    std::vector<float> result(vocab_size);
    auto extract = backend_->extract(*flat, result);
    if (!extract) return std::unexpected(extract.error());
    return result;
}

// ── KV-cache public API ───────────────────────────────────────────────────────

void LlamaModel::reset_cache() {
    kv_cache_.clear();
    cache_position_ = 0;
}

Result<std::vector<float>> LlamaModel::prefill(const std::vector<int>& prompt_ids) {
    if (prompt_ids.empty()) {
        return std::unexpected(Error{ErrorCode::InvalidInput, "prefill: prompt_ids cannot be empty"});
    }
    reset_cache();
    kv_cache_.assign(config_.num_hidden_layers, LayerKVCache{});
    auto result = forward_impl(prompt_ids, 0, &kv_cache_);
    if (result) cache_position_ = prompt_ids.size();
    return result;
}

Result<std::vector<float>> LlamaModel::decode(int token_id) {
    if (cache_position_ == 0) {
        return std::unexpected(Error{ErrorCode::InvalidInput,
            "decode: must call prefill() before decode()"});
    }
    auto result = forward_impl({token_id}, static_cast<int>(cache_position_), &kv_cache_);
    if (result) ++cache_position_;
    return result;
}

// ── No-cache forward (test interface) ────────────────────────────────────────

Result<std::vector<float>> LlamaModel::forward(const std::vector<int>& input_ids) {
    return forward_impl(input_ids, 0, nullptr);
}

Result<Tensor> LlamaModel::forward_logits(const std::vector<int>& input_ids) {
    auto hidden = embedding(input_ids);
    if (!hidden) return std::unexpected(hidden.error());

    for (int i = 0; i < static_cast<int>(config_.num_hidden_layers); ++i) {
        auto block_out = transformer_block(*hidden, i, 0, nullptr);
        if (!block_out) return std::unexpected(block_out.error());
        hidden = std::move(block_out);
    }

    auto norm_w = get_weight("model.norm.weight");
    if (!norm_w) return std::unexpected(norm_w.error());
    auto normed = rms_norm(*hidden, *norm_w, config_.rms_norm_eps);
    if (!normed) return std::unexpected(normed.error());

    if (weights_.count("lm_head.weight")) {
        return linear(*normed, "lm_head");
    } else if (config_.tie_word_embeddings) {
        if (!dequantized_embed_tokens_.has_value()) {
            auto ew = get_weight("model.embed_tokens.weight");
            if (!ew) return std::unexpected(ew.error());
            auto es = get_weight("model.embed_tokens.scales");
            if (es) {
                const int gs   = config_.quantization ? config_.quantization->group_size : 64;
                const int bits = config_.quantization ? config_.quantization->bits : 4;
                auto eb = get_weight("model.embed_tokens.biases");
                if (!eb) return std::unexpected(eb.error());
                auto deq = backend_->dequantize(*ew, *es, *eb, gs, bits);
                if (!deq) return std::unexpected(deq.error());
                dequantized_embed_tokens_ = std::move(*deq);
            } else {
                dequantized_embed_tokens_ = std::move(*ew);
            }
        }
        auto embed_t = backend_->swapaxes(*dequantized_embed_tokens_, 0, 1);
        if (!embed_t) return std::unexpected(embed_t.error());
        return backend_->matmul(*normed, *embed_t);
    }
    return std::unexpected(Error{ErrorCode::TensorNotFound,
        "lm_head.weight not found and tie_word_embeddings is false"});
}

// ── Metadata ──────────────────────────────────────────────────────────────────

size_t LlamaModel::num_parameters() const {
    size_t total = 0;
    for (const auto& [name, tensor] : weights_)
        total += tensor.size();
    return total;
}

// ── Tool-use ──────────────────────────────────────────────────────────────────

LlamaModel::ToolFamily LlamaModel::detect_tool_family(
    const SimpleBpeTokenizer& tok, const ModelConfig& cfg)
{
    // Qwen2.5 / Qwen3 Instruct: has <tool_call> in vocab (same format for both)
    if ((cfg.model_type == "qwen2" || cfg.model_type == "qwen3") &&
        tok.find_token_id("<tool_call>") != -1)
        return ToolFamily::Qwen25;

    // Llama 3.1+: has <|python_tag|> in vocab
    if (cfg.model_type == "llama" && tok.find_token_id("<|python_tag|>") != -1)
        return ToolFamily::Llama31;

    // Mistral tool variants: has [TOOL_CALLS] token in vocab
    if (cfg.model_type == "mistral" && tok.find_token_id("[TOOL_CALLS]") != -1)
        return ToolFamily::MistralTool;

    return ToolFamily::None;
}

bool LlamaModel::supports_tool_use() const {
    return tool_family_ != ToolFamily::None;
}

std::string LlamaModel::format_tool_system_prompt(const std::string& tools_json) const {
    switch (tool_family_) {
    case ToolFamily::Qwen25:
        return
            "You are a helpful assistant with access to the following tools. "
            "Use them when appropriate.\n\n"
            "# Tools\n\n" + tools_json + "\n\n"
            "When you need to call a tool, respond ONLY with:\n"
            "<tool_call>\n"
            "{\"name\": \"<tool_name>\", \"arguments\": {<args>}}\n"
            "</tool_call>\n"
            "Wait for the tool result before continuing.";

    case ToolFamily::Llama31:
        return
            "You have access to the following tools:\n\n" + tools_json + "\n\n"
            "When you need to call a tool, respond with:\n"
            "<function_calls>\n"
            "[{\"name\": \"<tool_name>\", \"arguments\": {<args>}}]\n"
            "</function_calls>\n"
            "Wait for the result before continuing.";

    case ToolFamily::MistralTool:
        return "[AVAILABLE_TOOLS] " + tools_json + " [/AVAILABLE_TOOLS]";

    default:
        return "";
    }
}

std::optional<LanguageModel::ToolCall> LlamaModel::detect_tool_call(
    const std::string& text) const
{
    switch (tool_family_) {

    case ToolFamily::Qwen25: {
        const auto open  = text.find("<tool_call>");
        const auto close = text.find("</tool_call>");
        if (open == std::string::npos || close == std::string::npos) return std::nullopt;
        const std::string payload = text.substr(open + 11, close - open - 11);
        // Parse {"name": "...", "arguments": {...}}
        const auto name_start = payload.find("\"name\"");
        const auto args_start = payload.find("\"arguments\"");
        if (name_start == std::string::npos || args_start == std::string::npos)
            return std::nullopt;
        const auto nq1 = payload.find('"', name_start + 7);
        const auto nq2 = payload.find('"', nq1 + 1);
        if (nq1 == std::string::npos || nq2 == std::string::npos) return std::nullopt;
        const auto ab = payload.find('{', args_start + 12);
        if (ab == std::string::npos) return std::nullopt;
        // Find the matching closing brace for arguments
        int depth = 0; std::string::size_type ae = ab;
        for (; ae < payload.size(); ++ae) {
            if (payload[ae] == '{') ++depth;
            else if (payload[ae] == '}') { if (--depth == 0) break; }
        }
        if (depth != 0) return std::nullopt;
        ToolCall tc;
        tc.name           = payload.substr(nq1 + 1, nq2 - nq1 - 1);
        tc.arguments_json = payload.substr(ab, ae - ab + 1);
        return tc;
    }

    case ToolFamily::Llama31: {
        const auto open  = text.find("<function_calls>");
        const auto close = text.find("</function_calls>");
        if (open == std::string::npos || close == std::string::npos) return std::nullopt;
        const std::string payload = text.substr(open + 16, close - open - 16);
        // Payload is a JSON array: [{"name":"...","arguments":{...}}]
        const auto name_start = payload.find("\"name\"");
        const auto args_start = payload.find("\"arguments\"");
        if (name_start == std::string::npos || args_start == std::string::npos)
            return std::nullopt;
        const auto nq1 = payload.find('"', name_start + 7);
        const auto nq2 = payload.find('"', nq1 + 1);
        if (nq1 == std::string::npos || nq2 == std::string::npos) return std::nullopt;
        const auto ab = payload.find('{', args_start + 12);
        if (ab == std::string::npos) return std::nullopt;
        int depth = 0; std::string::size_type ae = ab;
        for (; ae < payload.size(); ++ae) {
            if (payload[ae] == '{') ++depth;
            else if (payload[ae] == '}') { if (--depth == 0) break; }
        }
        if (depth != 0) return std::nullopt;
        ToolCall tc;
        tc.name           = payload.substr(nq1 + 1, nq2 - nq1 - 1);
        tc.arguments_json = payload.substr(ab, ae - ab + 1);
        return tc;
    }

    case ToolFamily::MistralTool: {
        // [TOOL_CALLS] [{"name":"...","arguments":{...}}]
        const auto marker = text.find("[TOOL_CALLS]");
        if (marker == std::string::npos) return std::nullopt;
        const auto ab = text.find('[', marker + 12);
        if (ab == std::string::npos) return std::nullopt;
        // Find matching ] for the outer array
        int depth = 0; std::string::size_type ae = ab;
        for (; ae < text.size(); ++ae) {
            if (text[ae] == '[') ++depth;
            else if (text[ae] == ']') { if (--depth == 0) break; }
        }
        if (depth != 0) return std::nullopt;
        const std::string arr = text.substr(ab, ae - ab + 1);
        const auto name_start = arr.find("\"name\"");
        const auto args_start = arr.find("\"arguments\"");
        if (name_start == std::string::npos || args_start == std::string::npos)
            return std::nullopt;
        const auto nq1 = arr.find('"', name_start + 7);
        const auto nq2 = arr.find('"', nq1 + 1);
        if (nq1 == std::string::npos || nq2 == std::string::npos) return std::nullopt;
        const auto obj_start = arr.find('{', args_start + 12);
        if (obj_start == std::string::npos) return std::nullopt;
        int d = 0; std::string::size_type obj_end = obj_start;
        for (; obj_end < arr.size(); ++obj_end) {
            if (arr[obj_end] == '{') ++d;
            else if (arr[obj_end] == '}') { if (--d == 0) break; }
        }
        if (d != 0) return std::nullopt;
        ToolCall tc;
        tc.name           = arr.substr(nq1 + 1, nq2 - nq1 - 1);
        tc.arguments_json = arr.substr(obj_start, obj_end - obj_start + 1);
        return tc;
    }

    default:
        return std::nullopt;
    }
}

std::string LlamaModel::format_tool_result(const std::string& /*tool_name*/,
                                            const std::string& result_json) const {
    switch (tool_family_) {
    case ToolFamily::Qwen25:
        return "\n<tool_response>\n" + result_json + "\n</tool_response>\n";
    case ToolFamily::Llama31:
        return "<|start_header_id|>tool<|end_header_id|>\n\n" + result_json +
               "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n";
    case ToolFamily::MistralTool:
        return "[TOOL_RESULTS] " + result_json + " [/TOOL_RESULTS]";
    default:
        return "";
    }
}

} // namespace compute
