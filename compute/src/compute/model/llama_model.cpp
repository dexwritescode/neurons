#include "llama_model.h"
#include "model_loader.h"
#include "sampler.h"
#include "../core/compute_backend.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <unordered_set>

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
#include <mlx/mlx.h>
#include <mlx/compile.h>
#endif

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
#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
    , mlx_embed_mat_(0.0f)
#endif
{}

// ── Factory ───────────────────────────────────────────────────────────────────

Result<LlamaModel> LlamaModel::from_model_dir(
    const std::filesystem::path& model_dir,
    ComputeBackend*              backend,
    size_t                       context_size)
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

    LlamaModel model(
        std::move(config),
        std::move(*tokenizer_result),
        std::move(weights),
        backend);

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
    // Extract MLX arrays from the already-loaded Tensor weights (single disk read).
    std::unordered_map<std::string, mlx::core::array> mlx_weights;
    mlx_weights.reserve(model.weights_.size());
    for (auto& [key, tensor] : model.weights_)
        mlx_weights.insert_or_assign(key, tensor.to_mlx());

    // Dequantize the embedding table eagerly so mx::compile treats it as a constant.
    // Valid models always have embed_tokens.weight (already confirmed by is_valid() above).
    auto ew_it = mlx_weights.find("model.embed_tokens.weight");
    mlx::core::array embed_mat = [&]() -> mlx::core::array {
        auto sc = mlx_weights.find("model.embed_tokens.scales");
        if (sc != mlx_weights.end()) {
            auto bi = mlx_weights.find("model.embed_tokens.biases");
            if (bi != mlx_weights.end()) {
                int gs = model.config_.quantization
                    ? model.config_.quantization->group_size : 64;
                double ratio = static_cast<double>(ew_it->second.shape().back()) /
                               static_cast<double>(sc->second.shape().back());
                int bits = static_cast<int>(std::round(32.0 * ratio / gs));
                if (bits <= 0) bits = 4;
                return mlx::core::dequantize(ew_it->second, sc->second, bi->second, gs, bits);
            }
        }
        return ew_it->second;
    }();
    mlx::core::eval(embed_mat);

    model.mlx_setup(std::move(mlx_weights), std::move(embed_mat), context_size);
#else
    (void)context_size;
#endif

    return model;
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

    // ── Optional per-head QK normalization (Qwen3) ───────────────────────────
    // Qwen3 adds learned RMSNorm per head-dimension before RoPE. Weights are
    // absent in Qwen2/Llama/Mistral, so probing is a no-op for those families.
    {
        auto it = weights_.find(prefix + "q_norm.weight");
        if (it != weights_.end()) {
            auto flat = backend_->reshape(*qt, {n_heads * seq_len, head_dim});
            if (!flat) return std::unexpected(flat.error());
            auto normed = backend_->rms_norm(*flat, it->second, config_.rms_norm_eps);
            if (!normed) return std::unexpected(normed.error());
            qt = backend_->reshape(*normed, {n_heads, seq_len, head_dim});
            if (!qt) return std::unexpected(qt.error());
        }
    }
    {
        auto it = weights_.find(prefix + "k_norm.weight");
        if (it != weights_.end()) {
            auto flat = backend_->reshape(*kt, {n_kv_heads * seq_len, head_dim});
            if (!flat) return std::unexpected(flat.error());
            auto normed = backend_->rms_norm(*flat, it->second, config_.rms_norm_eps);
            if (!normed) return std::unexpected(normed.error());
            kt = backend_->reshape(*normed, {n_kv_heads, seq_len, head_dim});
            if (!kt) return std::unexpected(kt.error());
        }
    }

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
#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
    mlx_state_.reset();
    mlx_pos_ = 0;
#endif
}

// ── Pipelined generate() override ────────────────────────────────────────────
//
// On Apple Silicon with MLX, greedy decode is accelerated by passing the
// unevaluated argmax from step N as input to step N+1 before waiting for step
// N to complete.  This keeps the GPU busy between decode steps and eliminates
// the CPU↔GPU round-trip idle time present in the base-class loop.
// Non-greedy and non-MLX paths fall back to LanguageModel::generate().

Result<std::vector<int>> LlamaModel::generate(
    const std::vector<int>& input_ids,
    size_t max_new_tokens,
    SamplingParams params,
    std::function<bool(int)> on_token)
{
#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
    if (params.temperature < 1e-6f)
        return mlx_generate_pipelined(input_ids, max_new_tokens, params, on_token);
#endif
    return LanguageModel::generate(input_ids, max_new_tokens, params, on_token);
}

Result<std::vector<float>> LlamaModel::prefill(const std::vector<int>& prompt_ids) {
    if (prompt_ids.empty())
        return std::unexpected(Error{ErrorCode::InvalidInput, "prefill: prompt_ids cannot be empty"});

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
    reset_cache();
    mlx_state_.emplace();        // create state struct; KV filled by prefill_batch
    mlx_build_decode_fn();       // build decode lambda for subsequent decode() calls
    return mlx_prefill_batch(prompt_ids);
#else
    reset_cache();
    kv_cache_.assign(config_.num_hidden_layers, LayerKVCache{});
    auto result = forward_impl(prompt_ids, 0, &kv_cache_);
    if (result) cache_position_ = prompt_ids.size();
    return result;
#endif
}

Result<std::vector<float>> LlamaModel::decode(int token_id) {
#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
    if (mlx_pos_ == 0)
        return std::unexpected(Error{ErrorCode::InvalidInput,
            "decode: must call prefill() before decode()"});
    return mlx_run_step(token_id);
#else
    if (cache_position_ == 0)
        return std::unexpected(Error{ErrorCode::InvalidInput,
            "decode: must call prefill() before decode()"});
    auto result = forward_impl({token_id}, static_cast<int>(cache_position_), &kv_cache_);
    if (result) ++cache_position_;
    return result;
#endif
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

// ═══════════════════════════════════════════════════════════════════════════════
// LlamaModel — Apple Silicon MLX fast decode path
// Growing KV cache (concat): each decode step appends one position to the KV.
// mx::eval after each step materialises arrays so the next concat has depth 1.
// Covers Llama, Mistral, Qwen2, Qwen3.
// ═══════════════════════════════════════════════════════════════════════════════
#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)

namespace mx = mlx::core;

namespace compute {

namespace {

using WM = std::unordered_map<std::string, mx::array>;

static int llama_mlx_bits(const mx::array& w, const mx::array& s, int gs) {
    double ratio = static_cast<double>(w.shape().back()) /
                   static_cast<double>(s.shape().back());
    int bits = static_cast<int>(std::round(32.0 * ratio / static_cast<double>(gs)));
    return (bits > 0) ? bits : 4;
}

static mx::array llama_mlx_lin(const mx::array& x, const std::string& key, const WM& wm, int gs) {
    const mx::array& w = wm.at(key + ".weight");
    auto sit = wm.find(key + ".scales");
    if (sit != wm.end()) {
        int bits = llama_mlx_bits(w, sit->second, gs);
        std::optional<mx::array> bias;
        auto bit = wm.find(key + ".biases");
        if (bit != wm.end()) bias = bit->second;
        return mx::quantized_matmul(x, w, sit->second, bias, true, gs, bits, "affine");
    }
    return mx::matmul(x, mx::swapaxes(w, 0, 1));
}

} // anonymous namespace

void LlamaModel::mlx_setup(
    std::unordered_map<std::string, mx::array> mlx_weights,
    mx::array                                  mlx_embed_mat,
    size_t                                     context_size)
{
    mlx_weights_   = std::move(mlx_weights);
    mlx_embed_mat_ = std::move(mlx_embed_mat);
    context_size_  = context_size;
}

void LlamaModel::mlx_init_state() {
    mlx_state_.emplace();
    auto& st = *mlx_state_;

    const size_t n_kv = config_.num_key_value_heads;
    const size_t hd   = config_.effective_head_dim();

    const int n = (int)config_.num_hidden_layers;
    st.kv_keys.reserve(n);
    st.kv_vals.reserve(n);
    for (int i = 0; i < n; ++i) {
        st.kv_keys.push_back(mx::zeros({(int)n_kv, 0, (int)hd}, mx::bfloat16));
        st.kv_vals.push_back(mx::zeros({(int)n_kv, 0, (int)hd}, mx::bfloat16));
    }
}

void LlamaModel::mlx_build_decode_fn() {
    const int    n_layers   = (int)config_.num_hidden_layers;
    const size_t n_heads    = config_.num_attention_heads;
    const size_t n_kv       = config_.num_key_value_heads;
    const size_t head_dim   = config_.effective_head_dim();
    const size_t hidden     = config_.hidden_size;
    const size_t vocab_size = config_.vocab_size;
    const float  rms_eps    = config_.rms_norm_eps;
    const float  rope_theta = config_.rope_theta;
    const int    rope_dims  = (int)(head_dim * config_.partial_rotary_factor.value_or(1.0f));
    const int    gs         = config_.quantization ? config_.quantization->group_size : 64;
    const bool   has_gate   = config_.attn_output_gate.value_or(false);

    WM       wm = mlx_weights_;
    mx::array em = mlx_embed_mat_;

    auto fn =
        [wm, em,
         n_layers, n_heads, n_kv, head_dim, hidden, vocab_size,
         rms_eps, rope_theta, rope_dims, gs,
         has_gate]
        (const std::vector<mx::array>& inputs) mutable -> std::vector<mx::array>
    {
        const mx::array& token_id = inputs[0];
        const mx::array& pos      = inputs[1];

        std::vector<mx::array> kv_k, kv_v;
        kv_k.reserve(n_layers); kv_v.reserve(n_layers);
        for (int i = 0; i < n_layers; ++i) {
            kv_k.push_back(inputs[2 + 2*i]);
            kv_v.push_back(inputs[2 + 2*i + 1]);
        }

        mx::array h = mx::reshape(mx::take(em, token_id, 0), {1, (int)hidden});

        std::vector<mx::array> out_kv_k, out_kv_v;
        out_kv_k.reserve(n_layers); out_kv_v.reserve(n_layers);

        for (int i = 0; i < n_layers; ++i) {
            const std::string lpfx = "model.layers." + std::to_string(i) + ".";
            const std::string apfx = lpfx + "self_attn.";
            const std::string mpfx = lpfx + "mlp.";

            auto normed = mx::fast::rms_norm(h, wm.at(lpfx + "input_layernorm.weight"), rms_eps);

            auto q_raw = llama_mlx_lin(normed, apfx + "q_proj", wm, gs);
            auto k_raw = llama_mlx_lin(normed, apfx + "k_proj", wm, gs);
            auto v_raw = llama_mlx_lin(normed, apfx + "v_proj", wm, gs);

            // Optional projection biases (Qwen2)
            {
                auto it = wm.find(apfx + "q_proj.bias");
                if (it != wm.end()) q_raw = q_raw + it->second;
            }
            {
                auto it = wm.find(apfx + "k_proj.bias");
                if (it != wm.end()) k_raw = k_raw + it->second;
            }
            {
                auto it = wm.find(apfx + "v_proj.bias");
                if (it != wm.end()) v_raw = v_raw + it->second;
            }

            // Attention output gate lives in the Q projection (Qwen3 dense)
            std::optional<mx::array> gate_flat;
            if (has_gate) {
                auto q4d = mx::reshape(q_raw, {1, (int)n_heads, 2, (int)head_dim});
                gate_flat = mx::reshape(mx::take(q4d, mx::array(1, mx::int32), 2),
                                        {1, (int)(n_heads * head_dim)});
                q_raw     = mx::reshape(mx::take(q4d, mx::array(0, mx::int32), 2),
                                        {1, (int)(n_heads * head_dim)});
            }

            // Reshape to [heads, 1, head_dim] (seq=1 during decode)
            auto qt = mx::reshape(q_raw, {(int)n_heads, 1, (int)head_dim});
            auto kt = mx::reshape(k_raw, {(int)n_kv,   1, (int)head_dim});
            auto vt = mx::reshape(v_raw, {(int)n_kv,   1, (int)head_dim});

            // Optional per-head Q/K norms (Qwen3)
            {
                auto it = wm.find(apfx + "q_norm.weight");
                if (it != wm.end()) {
                    auto f = mx::reshape(qt, {(int)n_heads, (int)head_dim});
                    qt = mx::reshape(mx::fast::rms_norm(f, it->second, rms_eps),
                                     {(int)n_heads, 1, (int)head_dim});
                }
            }
            {
                auto it = wm.find(apfx + "k_norm.weight");
                if (it != wm.end()) {
                    auto f = mx::reshape(kt, {(int)n_kv, (int)head_dim});
                    kt = mx::reshape(mx::fast::rms_norm(f, it->second, rms_eps),
                                     {(int)n_kv, 1, (int)head_dim});
                }
            }

            // RoPE — mx::fast::rope requires 4D [batch, heads, seq, head_dim]
            auto q_rope = mx::reshape(
                mx::fast::rope(mx::reshape(qt, {1, (int)n_heads, 1, (int)head_dim}),
                               rope_dims, false, rope_theta, 1.0f, pos),
                {(int)n_heads, 1, (int)head_dim});
            auto k_rope = mx::reshape(
                mx::fast::rope(mx::reshape(kt, {1, (int)n_kv, 1, (int)head_dim}),
                               rope_dims, false, rope_theta, 1.0f, pos),
                {(int)n_kv, 1, (int)head_dim});

            // Growing KV: concat new token onto existing cache
            auto new_k = mx::concatenate({kv_k[i], k_rope}, 1);
            auto new_v = mx::concatenate({kv_v[i], vt},     1);
            out_kv_k.push_back(new_k);
            out_kv_v.push_back(new_v);

            float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
            auto q4 = mx::reshape(q_rope, {1, (int)n_heads, 1,  (int)head_dim});
            auto k4 = mx::reshape(new_k,  {1, (int)n_kv,   -1, (int)head_dim});
            auto v4 = mx::reshape(new_v,  {1, (int)n_kv,   -1, (int)head_dim});
            auto attn4 = mx::fast::scaled_dot_product_attention(q4, k4, v4, scale, "");

            // [1, n_heads, 1, head_dim] → [1, hidden]
            auto attn_flat = mx::reshape(mx::swapaxes(attn4, 1, 2),
                                         {1, (int)(n_heads * head_dim)});

            if (has_gate && gate_flat)
                attn_flat = attn_flat * mx::sigmoid(*gate_flat);

            h = h + llama_mlx_lin(attn_flat, apfx + "o_proj", wm, gs);

            auto normed2 = mx::fast::rms_norm(
                h, wm.at(lpfx + "post_attention_layernorm.weight"), rms_eps);
            auto gate_m  = llama_mlx_lin(normed2, mpfx + "gate_proj", wm, gs);
            auto up      = llama_mlx_lin(normed2, mpfx + "up_proj",   wm, gs);
            h = h + llama_mlx_lin(
                mx::sigmoid(gate_m) * gate_m * up, mpfx + "down_proj", wm, gs);
        }

        h = mx::fast::rms_norm(h, wm.at("model.norm.weight"), rms_eps);

        auto logits_raw = wm.count("lm_head.weight")
            ? llama_mlx_lin(h, "lm_head", wm, gs)
            : mx::matmul(h, mx::swapaxes(em, 0, 1));
        auto logits = mx::astype(mx::reshape(logits_raw, {(int)vocab_size}), mx::float32);

        std::vector<mx::array> outputs;
        outputs.reserve(1 + 2 * n_layers);
        outputs.push_back(logits);
        for (int i = 0; i < n_layers; ++i) {
            outputs.push_back(out_kv_k[i]);
            outputs.push_back(out_kv_v[i]);
        }
        return outputs;
    };

    mlx_state_->compiled_fn = mx::compile(std::move(fn), /*shapeless=*/true);
    mlx_state_->fn_ready    = true;
}

// One-shot batched prefill: processes all prompt tokens in a single forward pass.
// KV cache is filled for all layers; returns logits for the last token position.
Result<std::vector<float>> LlamaModel::mlx_prefill_batch(const std::vector<int>& prompt_ids) {
    const int    seq_len   = (int)prompt_ids.size();
    const int    n_layers  = (int)config_.num_hidden_layers;
    const size_t n_heads   = config_.num_attention_heads;
    const size_t n_kv      = config_.num_key_value_heads;
    const size_t head_dim  = config_.effective_head_dim();
    const size_t hidden    = config_.hidden_size;
    const size_t vocab_size = config_.vocab_size;
    const float  rms_eps   = config_.rms_norm_eps;
    const float  rope_theta = config_.rope_theta;
    const int    rope_dims  = (int)(head_dim * config_.partial_rotary_factor.value_or(1.0f));
    const int    gs         = config_.quantization ? config_.quantization->group_size : 64;
    const bool   has_gate   = config_.attn_output_gate.value_or(false);
    const float  scale      = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const WM&       wm = mlx_weights_;
    const mx::array em = mlx_embed_mat_;

    // Token IDs as int32 array
    std::vector<int32_t> ids_i32(prompt_ids.begin(), prompt_ids.end());
    auto token_ids = mx::array(ids_i32.data(), {seq_len}, mx::int32);

    // Embed all tokens: {seq_len, hidden}
    mx::array h = mx::take(em, token_ids, 0);

    // Lower-triangular causal mask: mask[i,j] = 0 if j<=i else -1e9
    // Broadcast: rows {seq_len,1} vs cols {1,seq_len} → {seq_len,seq_len}
    // Mask dtype must match the model's compute dtype (float16 or bfloat16) — infer from embeddings.
    auto rows = mx::reshape(mx::arange(0, seq_len, mx::int32), {seq_len, 1});
    auto cols = mx::reshape(mx::arange(0, seq_len, mx::int32), {1, seq_len});
    auto causal_mask = mx::reshape(
        mx::astype(mx::where(cols > rows, mx::array(-1e9f), mx::array(0.0f)), em.dtype()),
        {1, 1, seq_len, seq_len});

    std::vector<mx::array> out_kv_k, out_kv_v;
    out_kv_k.reserve(n_layers);
    out_kv_v.reserve(n_layers);

    for (int i = 0; i < n_layers; ++i) {
        const std::string lpfx = "model.layers." + std::to_string(i) + ".";
        const std::string apfx = lpfx + "self_attn.";
        const std::string mpfx = lpfx + "mlp.";

        // Input layernorm: {seq_len, hidden}
        auto normed = mx::fast::rms_norm(h, wm.at(lpfx + "input_layernorm.weight"), rms_eps);

        // QKV projections: {seq_len, n_heads*head_dim}, {seq_len, n_kv*head_dim}, {seq_len, n_kv*head_dim}
        auto q_raw = llama_mlx_lin(normed, apfx + "q_proj", wm, gs);
        auto k_raw = llama_mlx_lin(normed, apfx + "k_proj", wm, gs);
        auto v_raw = llama_mlx_lin(normed, apfx + "v_proj", wm, gs);

        // Optional projection biases (Qwen2)
        {
            auto it = wm.find(apfx + "q_proj.bias");
            if (it != wm.end()) q_raw = q_raw + it->second;
        }
        {
            auto it = wm.find(apfx + "k_proj.bias");
            if (it != wm.end()) k_raw = k_raw + it->second;
        }
        {
            auto it = wm.find(apfx + "v_proj.bias");
            if (it != wm.end()) v_raw = v_raw + it->second;
        }

        // Attention output gate (Qwen3 dense): split Q into actual Q + gate
        std::optional<mx::array> gate_flat;
        if (has_gate) {
            auto q4d = mx::reshape(q_raw, {seq_len, (int)n_heads, 2, (int)head_dim});
            gate_flat = mx::reshape(mx::take(q4d, mx::array(1, mx::int32), 2),
                                    {seq_len, (int)(n_heads * head_dim)});
            q_raw     = mx::reshape(mx::take(q4d, mx::array(0, mx::int32), 2),
                                    {seq_len, (int)(n_heads * head_dim)});
        }

        // Reshape and transpose to {heads, seq_len, head_dim}
        auto qt = mx::swapaxes(mx::reshape(q_raw, {seq_len, (int)n_heads, (int)head_dim}), 0, 1);
        auto kt = mx::swapaxes(mx::reshape(k_raw, {seq_len, (int)n_kv,   (int)head_dim}), 0, 1);
        auto vt = mx::swapaxes(mx::reshape(v_raw, {seq_len, (int)n_kv,   (int)head_dim}), 0, 1);

        // Optional per-head Q/K norms (Qwen3): apply over {heads*seq, head_dim}
        {
            auto it = wm.find(apfx + "q_norm.weight");
            if (it != wm.end()) {
                auto f = mx::reshape(qt, {(int)(n_heads * seq_len), (int)head_dim});
                qt = mx::reshape(mx::fast::rms_norm(f, it->second, rms_eps),
                                 {(int)n_heads, seq_len, (int)head_dim});
            }
        }
        {
            auto it = wm.find(apfx + "k_norm.weight");
            if (it != wm.end()) {
                auto f = mx::reshape(kt, {(int)(n_kv * seq_len), (int)head_dim});
                kt = mx::reshape(mx::fast::rms_norm(f, it->second, rms_eps),
                                 {(int)n_kv, seq_len, (int)head_dim});
            }
        }

        // RoPE over the full sequence: offset=0 → positions [0..seq_len-1]
        auto q_rope = mx::reshape(
            mx::fast::rope(mx::reshape(qt, {1, (int)n_heads, seq_len, (int)head_dim}),
                           rope_dims, false, rope_theta, 1.0f, 0),
            {(int)n_heads, seq_len, (int)head_dim});
        auto k_rope = mx::reshape(
            mx::fast::rope(mx::reshape(kt, {1, (int)n_kv, seq_len, (int)head_dim}),
                           rope_dims, false, rope_theta, 1.0f, 0),
            {(int)n_kv, seq_len, (int)head_dim});

        // Store KV for this layer (decode will concat onto these)
        out_kv_k.push_back(k_rope);
        out_kv_v.push_back(vt);

        // SDPA with causal mask: {1, heads, seq, head_dim}
        auto q4    = mx::reshape(q_rope, {1, (int)n_heads, seq_len, (int)head_dim});
        auto k4    = mx::reshape(k_rope, {1, (int)n_kv,   seq_len, (int)head_dim});
        auto v4    = mx::reshape(vt,     {1, (int)n_kv,   seq_len, (int)head_dim});
        auto attn4 = mx::fast::scaled_dot_product_attention(q4, k4, v4, scale, "", causal_mask);

        // {1, n_heads, seq_len, head_dim} → {seq_len, n_heads*head_dim}
        auto attn_flat = mx::reshape(mx::swapaxes(attn4, 1, 2),
                                     {seq_len, (int)(n_heads * head_dim)});

        if (has_gate && gate_flat)
            attn_flat = attn_flat * mx::sigmoid(*gate_flat);

        h = h + llama_mlx_lin(attn_flat, apfx + "o_proj", wm, gs);

        auto normed2 = mx::fast::rms_norm(h, wm.at(lpfx + "post_attention_layernorm.weight"), rms_eps);
        auto gate_m  = llama_mlx_lin(normed2, mpfx + "gate_proj", wm, gs);
        auto up      = llama_mlx_lin(normed2, mpfx + "up_proj",   wm, gs);
        h = h + llama_mlx_lin(mx::sigmoid(gate_m) * gate_m * up, mpfx + "down_proj", wm, gs);
    }

    h = mx::fast::rms_norm(h, wm.at("model.norm.weight"), rms_eps);

    // Extract last token hidden state: {1, hidden}
    auto h_last = mx::reshape(
        mx::take(h, mx::array(seq_len - 1, mx::int32), 0), {1, (int)hidden});

    auto logits_raw = wm.count("lm_head.weight")
        ? llama_mlx_lin(h_last, "lm_head", wm, gs)
        : mx::matmul(h_last, mx::swapaxes(em, 0, 1));
    auto logits = mx::astype(mx::reshape(logits_raw, {(int)vocab_size}), mx::float32);

    // Evaluate logits + all KV in one GPU dispatch
    std::vector<mx::array> to_eval;
    to_eval.reserve(1 + 2 * n_layers);
    to_eval.push_back(logits);
    for (int i = 0; i < n_layers; ++i) {
        to_eval.push_back(out_kv_k[i]);
        to_eval.push_back(out_kv_v[i]);
    }
    mx::eval(to_eval);

    // Store filled KV into decode state
    auto& st   = *mlx_state_;
    st.kv_keys = std::move(out_kv_k);
    st.kv_vals = std::move(out_kv_v);
    mlx_pos_   = static_cast<size_t>(seq_len);

    std::vector<float> result(vocab_size);
    std::copy(logits.data<float>(), logits.data<float>() + vocab_size, result.data());
    return result;
}

Result<std::vector<float>> LlamaModel::mlx_run_step(int token_id) {
    auto& st = *mlx_state_;
    const int n = (int)st.kv_keys.size();

    std::vector<mx::array> inputs;
    inputs.reserve(2 + 2 * n);

    int32_t tid = static_cast<int32_t>(token_id);
    inputs.push_back(mx::array(&tid, {1}, mx::int32));
    int32_t pos_val = static_cast<int32_t>(mlx_pos_);
    inputs.push_back(mx::array(&pos_val, {1}, mx::int32));
    for (int i = 0; i < n; ++i) {
        inputs.push_back(st.kv_keys[i]);
        inputs.push_back(st.kv_vals[i]);
    }

    auto outputs = st.compiled_fn(inputs);

    // Evaluate only logits immediately; KV caches are lazily evaluated in the
    // next step, overlapping GPU KV computation with CPU sampling.
    mx::eval(outputs[0]);

    for (int i = 0; i < n; ++i) {
        st.kv_keys[i] = outputs[1 + 2*i];
        st.kv_vals[i] = outputs[1 + 2*i + 1];
    }

    ++mlx_pos_;

    const mx::array& logits = outputs[0];
    std::vector<float> result(config_.vocab_size);
    std::copy(logits.data<float>(), logits.data<float>() + config_.vocab_size, result.data());
    return result;
}

Result<std::vector<int>> LlamaModel::mlx_generate_pipelined(
    const std::vector<int>& input_ids,
    size_t max_new_tokens,
    SamplingParams /*params*/,
    std::function<bool(int)> on_token)
{
    // Build EOS set.
    std::unordered_set<int> eos_set;
    if (config_.eos_token_ids.has_value()) {
        for (int id : *config_.eos_token_ids) eos_set.insert(id);
    } else {
        eos_set.insert(2);
    }

    // Prefill: initialises mlx_state_ and mlx_pos_.
    auto prefill_result = prefill(input_ids);
    if (!prefill_result) return std::unexpected(prefill_result.error());

    std::vector<int> generated;
    generated.reserve(max_new_tokens);

    if (max_new_tokens == 0) return generated;

    // First generated token — CPU argmax of prefill logits (greedy only path).
    const auto& fl = *prefill_result;
    int first_token = (int)(std::max_element(fl.begin(), fl.end()) - fl.begin());
    generated.push_back(first_token);
    if (on_token && !on_token(first_token)) return generated;
    if (eos_set.count(first_token) || generated.size() >= max_new_tokens) return generated;

    auto& st  = *mlx_state_;
    const int n = (int)st.kv_keys.size();

    // Helper: assemble inputs for the compiled decode function.
    // Captures st and n by reference; reads st.kv_keys/kv_vals at call time.
    auto build_inputs = [&](mx::array tok, int32_t pos) {
        std::vector<mx::array> inputs;
        inputs.reserve(2 + 2 * n);
        inputs.push_back(std::move(tok));
        inputs.push_back(mx::array(&pos, {1}, mx::int32));
        for (int i = 0; i < n; ++i) {
            inputs.push_back(st.kv_keys[i]);
            inputs.push_back(st.kv_vals[i]);
        }
        return inputs;
    };

    // ── Step 0: kick off the pipeline with the known first_token ─────────────
    {
        int32_t fv  = (int32_t)first_token;
        auto    out0 = st.compiled_fn(
            build_inputs(mx::array(&fv, {1}, mx::int32), (int32_t)mlx_pos_));

        // Lazy argmax: shape {1}, dtype int32 — not yet computed on CPU.
        auto prev_tok_lazy = mx::reshape(mx::argmax(out0[0]), {1});
        mx::async_eval(prev_tok_lazy);   // enqueue GPU work; return immediately

        for (int i = 0; i < n; ++i) {
            st.kv_keys[i] = out0[1 + 2*i];
            st.kv_vals[i] = out0[1 + 2*i + 1];
        }
        ++mlx_pos_;

        // ── Pipelined loop ────────────────────────────────────────────────────
        // At the top of each iteration:
        //   GPU is already computing prev_tok_lazy (step N-1).
        //   We build step N's graph with prev_tok_lazy as its input token, then
        //   async_eval it — so the GPU queues step N immediately after N-1 with
        //   no CPU round-trip stall between them.
        for (size_t step = 1; step + 1 < max_new_tokens; ++step) {
            auto out_n = st.compiled_fn(build_inputs(prev_tok_lazy, (int32_t)mlx_pos_));
            auto curr_tok_lazy = mx::reshape(mx::argmax(out_n[0]), {1});
            mx::async_eval(curr_tok_lazy);   // GPU starts step N

            // Block until step N-1 is done; GPU runs step N concurrently.
            mx::eval(prev_tok_lazy);
            int prev_token = prev_tok_lazy.item<int32_t>();

            for (int i = 0; i < n; ++i) {
                st.kv_keys[i] = out_n[1 + 2*i];
                st.kv_vals[i] = out_n[1 + 2*i + 1];
            }
            ++mlx_pos_;

            generated.push_back(prev_token);
            if (on_token && !on_token(prev_token)) return generated;
            if (eos_set.count(prev_token)) return generated;

            prev_tok_lazy = std::move(curr_tok_lazy);
        }

        // ── Drain the last pipelined token ────────────────────────────────────
        mx::eval(prev_tok_lazy);
        int last_token = prev_tok_lazy.item<int32_t>();
        generated.push_back(last_token);
        if (on_token) on_token(last_token);
    }

    return generated;
}

} // namespace compute

#endif // MLX_BACKEND_ENABLED
