#include "gemma_model.h"
#include "model_loader.h"
#include "sampler.h"
#include "../core/compute_backend.h"
#include <algorithm>
#include <cmath>
#include <sstream>

namespace compute {

// ── Private constructor ───────────────────────────────────────────────────────

GemmaModel::GemmaModel(
    ModelConfig                             config,
    SimpleBpeTokenizer                      tokenizer,
    std::unordered_map<std::string, Tensor> weights,
    ComputeBackend*                         backend)
    : config_(std::move(config))
    , tokenizer_(std::move(tokenizer))
    , weights_(std::move(weights))
    , backend_(backend)
{}

// ── Factory ───────────────────────────────────────────────────────────────────

Result<GemmaModel> GemmaModel::from_model_dir(
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
    if (!config.is_gemma_architecture()) {
        return std::unexpected(Error{ErrorCode::InvalidModel,
            "Not a Gemma architecture: " + config.model_type});
    }

    // Multimodal models (Gemma3ForConditionalGeneration) prefix all text weights
    // with "language_model." — strip it so GemmaModel sees standard "model." keys.
    {
        const std::string prefix = "language_model.";
        const bool has_prefixed = weights.count(prefix + "model.embed_tokens.weight") > 0;
        const bool has_plain    = weights.count("model.embed_tokens.weight") > 0;
        if (has_prefixed && !has_plain) {
            std::unordered_map<std::string, Tensor> remapped;
            remapped.reserve(weights.size());
            for (auto& [key, val] : weights) {
                if (key.size() > prefix.size() &&
                    key.compare(0, prefix.size(), prefix) == 0) {
                    remapped.emplace(key.substr(prefix.size()), std::move(val));
                } else {
                    remapped.emplace(key, std::move(val));
                }
            }
            weights = std::move(remapped);
        }
    }

    // Gemma always stops on <end_of_turn> (ID 106) regardless of config.
    // Some model variants (e.g. gemma-3-1b-it-4bit) only list eos_token_id=1
    // in config.json and omit 106, but the chat template uses <end_of_turn>
    // as the actual conversation boundary token.
    {
        constexpr int kEndOfTurn = 106;
        if (!config.eos_token_ids.has_value()) {
            config.eos_token_ids = std::vector<int>{config.primary_eos_token_id(), kEndOfTurn};
        } else {
            auto& ids = *config.eos_token_ids;
            if (std::find(ids.begin(), ids.end(), kEndOfTurn) == ids.end()) {
                ids.push_back(kEndOfTurn);
            }
        }
    }

    return GemmaModel(
        std::move(config),
        std::move(*tokenizer_result),
        std::move(weights),
        backend);
}

// ── Weight lookup ─────────────────────────────────────────────────────────────

Result<Tensor> GemmaModel::get_weight(const std::string& name) const {
    auto it = weights_.find(name);
    if (it == weights_.end()) {
        return std::unexpected(Error{ErrorCode::TensorNotFound, "Weight not found: " + name});
    }
    return it->second;
}

// ── Linear projection (quantized or unquantized) ─────────────────────────────

Result<Tensor> GemmaModel::linear(const Tensor& input, const std::string& weight_key) {
    auto w = get_weight(weight_key + ".weight");
    if (!w) return std::unexpected(w.error());

    auto s_it = weights_.find(weight_key + ".scales");
    if (s_it != weights_.end()) {
        const int gs   = config_.quantization ? config_.quantization->group_size : 64;
        const int bits = config_.quantization ? config_.quantization->bits : 4;
        auto b_it = weights_.find(weight_key + ".biases");
        const Tensor* biases = (b_it != weights_.end()) ? &b_it->second : nullptr;
        return backend_->quantized_matmul(input, *w, s_it->second, biases, true, gs, bits);
    }

    // Unquantized: weight stored [out, in] — transpose before matmul
    auto w_t = backend_->swapaxes(*w, 0, 1);
    if (!w_t) return std::unexpected(w_t.error());
    return backend_->matmul(input, *w_t);
}

// ── Embedding ─────────────────────────────────────────────────────────────────

Result<Tensor> GemmaModel::embedding(const std::vector<int>& token_ids) {
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

    auto embedded = backend_->concatenate(rows, 0);
    if (!embedded) return std::unexpected(embedded.error());

    // Gemma-specific: scale embeddings by sqrt(hidden_size)
    const float embed_scale = std::sqrt(static_cast<float>(hidden_size));
    auto scale_tensor = backend_->create_tensor(
        std::span<const float>(&embed_scale, 1), {1});
    auto scaled = backend_->multiply(*embedded, scale_tensor);
    if (!scaled) return std::unexpected(scaled.error());

    return *scaled;
}

// ── RMSNorm helper ────────────────────────────────────────────────────────────
//
// Gemma uses (1 + weight) as the effective scale factor — NOT just weight.
// The stored weights are deviations from 0 (HuggingFace initializes them to
// zeros; mlx-lm applies `mx.fast.rms_norm(x, 1.0 + self.weight, eps)`).
// Our backend's rms_norm computes: x / rms(x) * weight.
// So we pass (1 + weight) to get the correct Gemma semantics.

Result<Tensor> GemmaModel::rms_norm(const Tensor& input, const Tensor& weight) {
    auto effective_weight = backend_->matrix_scalar_add(weight, 1.0f);
    return backend_->rms_norm(input, effective_weight, config_.rms_norm_eps);
}

// ── Attention layer ───────────────────────────────────────────────────────────

Result<Tensor> GemmaModel::attention_layer(
    const Tensor&      input,
    int                layer_idx,
    int                position_offset,
    GemmaLayerKVCache* cache)
{
    if (input.shape().size() != 2) {
        return std::unexpected(Error{ErrorCode::InvalidInput,
            "Attention input must be 2D [seq_len, hidden_size]"});
    }

    const size_t seq_len    = input.shape()[0];
    const size_t n_heads    = config_.num_attention_heads;
    const size_t n_kv_heads = config_.num_key_value_heads;
    const size_t head_dim   = config_.effective_head_dim();
    const float  scale      = config_.effective_attention_scale();

    // RoPE theta: local layers use rope_local_base_freq, global layers use rope_theta
    const bool  is_local = config_.is_local_layer(layer_idx);
    const float rope_theta = (is_local && config_.rope_local_base_freq.has_value())
        ? *config_.rope_local_base_freq
        : config_.rope_theta;

    const std::string prefix = "model.layers." + std::to_string(layer_idx) + ".self_attn.";

    // ── QKV projections ───────────────────────────────────────────────────────
    auto queries_flat = linear(input, prefix + "q_proj");
    if (!queries_flat) return std::unexpected(queries_flat.error());

    auto keys_flat = linear(input, prefix + "k_proj");
    if (!keys_flat) return std::unexpected(keys_flat.error());

    auto values_flat = linear(input, prefix + "v_proj");
    if (!values_flat) return std::unexpected(values_flat.error());

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

    // ── Q/K norm (Gemma3-specific, before RoPE) ───────────────────────────────
    // Weights are [head_dim]; MLX rms_norm broadcasts over leading dims.
    auto q_norm_w = get_weight(prefix + "q_norm.weight");
    if (q_norm_w) {
        auto qn = rms_norm(*qt, *q_norm_w);
        if (!qn) return std::unexpected(qn.error());
        qt = std::move(qn);
    }

    auto k_norm_w = get_weight(prefix + "k_norm.weight");
    if (k_norm_w) {
        auto kn = rms_norm(*kt, *k_norm_w);
        if (!kn) return std::unexpected(kn.error());
        kt = std::move(kn);
    }

    // ── RoPE ──────────────────────────────────────────────────────────────────
    auto q_rope = backend_->rope(*qt, static_cast<int>(head_dim), rope_theta, position_offset);
    if (!q_rope) return std::unexpected(q_rope.error());
    auto k_rope = backend_->rope(*kt, static_cast<int>(head_dim), rope_theta, position_offset);
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
    auto attn_out = backend_->scaled_dot_product_attention(
        *q_rope, full_k, full_v, scale, attn_mask);
    if (!attn_out) return std::unexpected(attn_out.error());

    // ── Reshape output → [seq, n_heads * head_dim] ────────────────────────────
    auto attn_t    = backend_->swapaxes(*attn_out, 0, 1);
    if (!attn_t) return std::unexpected(attn_t.error());
    auto attn_flat = backend_->reshape(*attn_t, {seq_len, n_heads * head_dim});
    if (!attn_flat) return std::unexpected(attn_flat.error());

    // ── Output projection ─────────────────────────────────────────────────────
    return linear(*attn_flat, prefix + "o_proj");
}

// ── MLP layer (GeGLU: gelu(gate) * up, projected by down) ────────────────────

Result<Tensor> GemmaModel::mlp_layer(const Tensor& input, int layer_idx) {
    const std::string prefix = "model.layers." + std::to_string(layer_idx) + ".mlp.";

    auto gate = linear(input, prefix + "gate_proj");
    if (!gate) return std::unexpected(gate.error());

    auto up = linear(input, prefix + "up_proj");
    if (!up) return std::unexpected(up.error());

    // GeGLU: gelu(gate) * up
    auto activated = backend_->gelu(*gate);
    if (!activated) return std::unexpected(activated.error());
    auto hidden = backend_->multiply(*activated, *up);
    if (!hidden) return std::unexpected(hidden.error());

    return linear(*hidden, prefix + "down_proj");
}

// ── Transformer block (Gemma3: 4 norms, post-norm residuals) ─────────────────
//
// Forward pass per layer:
//   residual = x
//   x = input_layernorm(x)
//   x = attention(x)
//   x = post_attention_layernorm(x)   ← applied to attn OUTPUT
//   x = residual + x
//   residual = x
//   x = pre_feedforward_layernorm(x)
//   x = mlp(x)
//   x = post_feedforward_layernorm(x) ← applied to FFN OUTPUT
//   x = residual + x

Result<Tensor> GemmaModel::transformer_block(
    const Tensor&      input,
    int                layer_idx,
    int                position_offset,
    GemmaLayerKVCache* cache)
{
    const std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";

    // ── Attention sub-block ───────────────────────────────────────────────────
    auto input_norm_w = get_weight(prefix + "input_layernorm.weight");
    if (!input_norm_w) return std::unexpected(input_norm_w.error());
    auto normed_in = rms_norm(input, *input_norm_w);
    if (!normed_in) return std::unexpected(normed_in.error());

    auto attn_out = attention_layer(*normed_in, layer_idx, position_offset, cache);
    if (!attn_out) return std::unexpected(attn_out.error());

    auto post_attn_norm_w = get_weight(prefix + "post_attention_layernorm.weight");
    if (!post_attn_norm_w) return std::unexpected(post_attn_norm_w.error());
    auto normed_attn = rms_norm(*attn_out, *post_attn_norm_w);
    if (!normed_attn) return std::unexpected(normed_attn.error());

    auto residual1 = backend_->add(input, *normed_attn);
    if (!residual1) return std::unexpected(residual1.error());

    // ── FFN sub-block ─────────────────────────────────────────────────────────
    auto pre_ffn_norm_w = get_weight(prefix + "pre_feedforward_layernorm.weight");
    if (!pre_ffn_norm_w) return std::unexpected(pre_ffn_norm_w.error());
    auto normed_ffn_in = rms_norm(*residual1, *pre_ffn_norm_w);
    if (!normed_ffn_in) return std::unexpected(normed_ffn_in.error());

    auto ffn_out = mlp_layer(*normed_ffn_in, layer_idx);
    if (!ffn_out) return std::unexpected(ffn_out.error());

    auto post_ffn_norm_w = get_weight(prefix + "post_feedforward_layernorm.weight");
    if (!post_ffn_norm_w) return std::unexpected(post_ffn_norm_w.error());
    auto normed_ffn = rms_norm(*ffn_out, *post_ffn_norm_w);
    if (!normed_ffn) return std::unexpected(normed_ffn.error());

    return backend_->add(*residual1, *normed_ffn);
}

// ── Internal forward ─────────────────────────────────────────────────────────

Result<std::vector<float>> GemmaModel::forward_impl(
    const std::vector<int>&         input_ids,
    int                             position_offset,
    std::vector<GemmaLayerKVCache>* cache_vec)
{
    auto hidden = embedding(input_ids);
    if (!hidden) return std::unexpected(hidden.error());

    for (int i = 0; i < static_cast<int>(config_.num_hidden_layers); ++i) {
        GemmaLayerKVCache* layer_cache = cache_vec ? &(*cache_vec)[i] : nullptr;
        auto block_out = transformer_block(*hidden, i, position_offset, layer_cache);
        if (!block_out) return std::unexpected(block_out.error());
        hidden = std::move(block_out);
    }

    // Final layer norm
    auto norm_w = get_weight("model.norm.weight");
    if (!norm_w) return std::unexpected(norm_w.error());
    auto normed = rms_norm(*hidden, *norm_w);
    if (!normed) return std::unexpected(normed.error());

    // LM head — Gemma3 1B has a separate lm_head; multimodal variants (4B+)
    // tie word embeddings (no lm_head.weight, reuse embed_tokens transposed).
    Result<Tensor> logits_tensor{std::unexpected(Error{ErrorCode::TensorNotFound, "unset"})};
    if (weights_.count("lm_head.weight")) {
        logits_tensor = linear(*normed, "lm_head");
    } else if (config_.tie_word_embeddings) {
        if (!dequantized_embed_tokens_.has_value()) {
            // Trigger embedding table dequantization without a real token
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

    // Extract last-token logits
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

void GemmaModel::reset_cache() {
    kv_cache_.clear();
    cache_position_ = 0;
}

Result<std::vector<float>> GemmaModel::prefill(const std::vector<int>& prompt_ids) {
    if (prompt_ids.empty()) {
        return std::unexpected(Error{ErrorCode::InvalidInput, "prefill: prompt_ids cannot be empty"});
    }
    reset_cache();
    kv_cache_.assign(config_.num_hidden_layers, GemmaLayerKVCache{});
    auto result = forward_impl(prompt_ids, 0, &kv_cache_);
    if (result) cache_position_ = prompt_ids.size();
    return result;
}

Result<std::vector<float>> GemmaModel::decode(int token_id) {
    if (cache_position_ == 0) {
        return std::unexpected(Error{ErrorCode::InvalidInput,
            "decode: must call prefill() before decode()"});
    }
    auto result = forward_impl({token_id}, static_cast<int>(cache_position_), &kv_cache_);
    if (result) ++cache_position_;
    return result;
}

// ── Metadata ──────────────────────────────────────────────────────────────────

size_t GemmaModel::num_parameters() const {
    size_t total = 0;
    for (const auto& [name, tensor] : weights_)
        total += tensor.size();
    return total;
}

} // namespace compute
