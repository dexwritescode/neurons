#include "gemma_model_mlx.h"

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)

#include "model_loader.h"
#include "mlx_ops.h"
#include "sampler.h"
#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace compute {
namespace mx = mlx::core;

namespace {
using namespace compute::mlx_ops;
} // anonymous namespace

// ── Factory ───────────────────────────────────────────────────────────────────

Result<GemmaModelMLX> GemmaModelMLX::from_model_dir(
    const std::filesystem::path& model_dir)
{
    auto result = ModelLoader::load_model_mlx(model_dir);
    if (!result) return std::unexpected(result.error());

    auto& [config, mlx_weights] = *result;

    if (!config.is_gemma_architecture()) {
        return std::unexpected(Error{ErrorCode::InvalidModel,
            "Not a Gemma architecture: " + config.model_type});
    }

    // Multimodal variants (Gemma3ForConditionalGeneration) prefix all text
    // weights with "language_model." — strip it so GemmaModelMLX sees standard keys.
    {
        const std::string pfx = "language_model.";
        const bool has_prefixed = mlx_weights.count(pfx + "model.embed_tokens.weight") > 0;
        const bool has_plain    = mlx_weights.count("model.embed_tokens.weight") > 0;
        if (has_prefixed && !has_plain) {
            WM remapped;
            remapped.reserve(mlx_weights.size());
            for (auto& [key, val] : mlx_weights) {
                if (key.size() > pfx.size() &&
                    key.compare(0, pfx.size(), pfx) == 0) {
                    remapped.emplace(key.substr(pfx.size()), std::move(val));
                } else {
                    remapped.emplace(key, std::move(val));
                }
            }
            mlx_weights = std::move(remapped);
        }
    }

    // Gemma always stops on <end_of_turn> (token 106).
    {
        constexpr int kEndOfTurn = 106;
        if (!config.eos_token_ids.has_value()) {
            config.eos_token_ids = std::vector<int>{config.primary_eos_token_id(), kEndOfTurn};
        } else {
            auto& ids = *config.eos_token_ids;
            if (std::find(ids.begin(), ids.end(), kEndOfTurn) == ids.end())
                ids.push_back(kEndOfTurn);
        }
    }

    auto tok_result = SimpleBpeTokenizer::from_model_dir(model_dir);
    if (!tok_result) return std::unexpected(tok_result.error());

    // Dequantize embedding eagerly so mx::compile treats it as a constant.
    auto ew_it = mlx_weights.find("model.embed_tokens.weight");
    if (ew_it == mlx_weights.end())
        return std::unexpected(Error{ErrorCode::InvalidModel, "embed_tokens.weight missing"});

    mx::array embed_mat = [&]() -> mx::array {
        auto sc_it = mlx_weights.find("model.embed_tokens.scales");
        if (sc_it != mlx_weights.end()) {
            auto bi_it = mlx_weights.find("model.embed_tokens.biases");
            if (bi_it != mlx_weights.end()) {
                int gs = config.quantization ? config.quantization->group_size : 64;
                int b  = bits(ew_it->second, sc_it->second, gs);
                return mx::dequantize(ew_it->second, sc_it->second, bi_it->second, gs, b);
            }
        }
        return ew_it->second;
    }();
    mx::eval(embed_mat);

    return GemmaModelMLX(
        std::move(config),
        std::move(*tok_result),
        std::move(mlx_weights),
        std::move(embed_mat));
}

// ── Constructor ───────────────────────────────────────────────────────────────

GemmaModelMLX::GemmaModelMLX(
    ModelConfig                                        config,
    SimpleBpeTokenizer                                 tokenizer,
    std::unordered_map<std::string, mx::array>         mlx_weights,
    mx::array                                          embed_mat)
    : GemmaModelBase(std::move(config), std::move(tokenizer), {})
    , mlx_weights_(std::move(mlx_weights))
    , embed_mat_(std::move(embed_mat))
{}

size_t GemmaModelMLX::num_parameters() const {
    size_t total = 0;
    for (const auto& [name, arr] : mlx_weights_) total += arr.size();
    return total;
}

// ── Cache management ──────────────────────────────────────────────────────────

void GemmaModelMLX::reset_cache() {
    mlx_state_.reset();
    cache_position_ = 0;
}

void GemmaModelMLX::init_empty_decode_state() {
    mlx_state_.emplace();
    auto& st = *mlx_state_;

    const size_t n_kv = config_.num_key_value_heads;
    const size_t hd   = config_.effective_head_dim();
    const int    n    = (int)config_.num_hidden_layers;

    st.kv_keys.reserve(n);
    st.kv_vals.reserve(n);
    for (int i = 0; i < n; ++i) {
        st.kv_keys.push_back(mx::zeros({(int)n_kv, 0, (int)hd}, mx::bfloat16));
        st.kv_vals.push_back(mx::zeros({(int)n_kv, 0, (int)hd}, mx::bfloat16));
    }
}

// ── Decode function builder ───────────────────────────────────────────────────

void GemmaModelMLX::build_decode_fn() {
    const int    n_layers    = (int)config_.num_hidden_layers;
    const size_t n_heads     = config_.num_attention_heads;
    const size_t n_kv        = config_.num_key_value_heads;
    const size_t head_dim    = config_.effective_head_dim();
    const size_t hidden      = config_.hidden_size;
    const size_t vocab_size  = config_.vocab_size;
    const float  rms_eps     = config_.rms_norm_eps;
    const int    rope_dims   = (int)head_dim;  // Gemma: full RoPE
    const int    gs          = config_.quantization ? config_.quantization->group_size : 64;
    const float  embed_scale = std::sqrt(static_cast<float>(hidden));
    const bool   tied_embed  = mlx_weights_.find("lm_head.weight") == mlx_weights_.end();

    // Per-layer rope_theta: local layers use rope_local_base_freq (Gemma3).
    std::vector<float> rope_thetas(n_layers);
    for (int i = 0; i < n_layers; ++i) {
        bool is_local = config_.is_local_layer(i);
        rope_thetas[i] = (is_local && config_.rope_local_base_freq.has_value())
            ? *config_.rope_local_base_freq
            : config_.rope_theta;
    }

    WM        wm = mlx_weights_;
    mx::array em = embed_mat_;

    auto fn = [wm, em,
               n_layers, n_heads, n_kv, head_dim, hidden, vocab_size,
               rms_eps, rope_dims, gs, embed_scale, tied_embed,
               rope_thetas]
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

        // Embed + Gemma scale
        mx::array h = mx::reshape(mx::take(em, token_id, 0), {1, (int)hidden}) * embed_scale;

        std::vector<mx::array> out_kv_k, out_kv_v;
        out_kv_k.reserve(n_layers); out_kv_v.reserve(n_layers);

        for (int i = 0; i < n_layers; ++i) {
            const std::string lpfx = "model.layers." + std::to_string(i) + ".";
            const std::string apfx = lpfx + "self_attn.";
            const std::string mpfx = lpfx + "mlp.";
            const float rope_theta = rope_thetas[i];

            // Attention sub-block: input_layernorm with (1 + weight)
            auto normed = mx::fast::rms_norm(
                h, 1.0f + wm.at(lpfx + "input_layernorm.weight"), rms_eps);

            auto q_raw = lin(normed, apfx + "q_proj", wm, gs);
            auto k_raw = lin(normed, apfx + "k_proj", wm, gs);
            auto v_raw = lin(normed, apfx + "v_proj", wm, gs);

            auto qt = mx::reshape(q_raw, {(int)n_heads, 1, (int)head_dim});
            auto kt = mx::reshape(k_raw, {(int)n_kv,   1, (int)head_dim});
            auto vt = mx::reshape(v_raw, {(int)n_kv,   1, (int)head_dim});

            // Q/K per-head norms (Gemma3, optional) — also (1 + weight)
            {
                auto it = wm.find(apfx + "q_norm.weight");
                if (it != wm.end()) {
                    auto f = mx::reshape(qt, {(int)n_heads, (int)head_dim});
                    qt = mx::reshape(mx::fast::rms_norm(f, 1.0f + it->second, rms_eps),
                                     {(int)n_heads, 1, (int)head_dim});
                }
            }
            {
                auto it = wm.find(apfx + "k_norm.weight");
                if (it != wm.end()) {
                    auto f = mx::reshape(kt, {(int)n_kv, (int)head_dim});
                    kt = mx::reshape(mx::fast::rms_norm(f, 1.0f + it->second, rms_eps),
                                     {(int)n_kv, 1, (int)head_dim});
                }
            }

            // RoPE
            auto q_rope = mx::reshape(
                mx::fast::rope(mx::reshape(qt, {1, (int)n_heads, 1, (int)head_dim}),
                               rope_dims, false, rope_theta, 1.0f, pos),
                {(int)n_heads, 1, (int)head_dim});
            auto k_rope = mx::reshape(
                mx::fast::rope(mx::reshape(kt, {1, (int)n_kv, 1, (int)head_dim}),
                               rope_dims, false, rope_theta, 1.0f, pos),
                {(int)n_kv, 1, (int)head_dim});

            // Growing KV cache
            auto new_k = mx::concatenate({kv_k[i], k_rope}, 1);
            auto new_v = mx::concatenate({kv_v[i], vt},     1);
            out_kv_k.push_back(new_k);
            out_kv_v.push_back(new_v);

            // SDPA (no mask — seq=1 during decode)
            const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
            auto q4    = mx::reshape(q_rope, {1, (int)n_heads, 1,  (int)head_dim});
            auto k4    = mx::reshape(new_k,  {1, (int)n_kv,   -1, (int)head_dim});
            auto v4    = mx::reshape(new_v,  {1, (int)n_kv,   -1, (int)head_dim});
            auto attn4 = mx::fast::scaled_dot_product_attention(q4, k4, v4, scale, "");
            auto attn_flat = mx::reshape(mx::swapaxes(attn4, 1, 2),
                                         {1, (int)(n_heads * head_dim)});

            auto attn_out = lin(attn_flat, apfx + "o_proj", wm, gs);

            // post_attention_layernorm on attn OUTPUT, then residual
            h = h + mx::fast::rms_norm(
                attn_out, 1.0f + wm.at(lpfx + "post_attention_layernorm.weight"), rms_eps);

            // FFN sub-block: pre_feedforward_layernorm, GeGLU, post_feedforward_layernorm
            auto normed_ffn = mx::fast::rms_norm(
                h, 1.0f + wm.at(lpfx + "pre_feedforward_layernorm.weight"), rms_eps);
            auto gate_m  = lin(normed_ffn, mpfx + "gate_proj", wm, gs);
            auto up      = lin(normed_ffn, mpfx + "up_proj",   wm, gs);
            // GeGLU: gelu(gate) * up  (not SwiGLU). MLX has no gelu(); use exact form.
            auto gelu_gate = 0.5f * gate_m * (1.0f + mx::erf(gate_m * static_cast<float>(1.0 / std::sqrt(2.0))));
            auto ffn_out = lin(gelu_gate * up, mpfx + "down_proj", wm, gs);

            // post_feedforward_layernorm on FFN OUTPUT, then residual
            h = h + mx::fast::rms_norm(
                ffn_out, 1.0f + wm.at(lpfx + "post_feedforward_layernorm.weight"), rms_eps);
        }

        // Final norm
        h = mx::fast::rms_norm(h, 1.0f + wm.at("model.norm.weight"), rms_eps);

        // LM head: separate or tied to embed
        auto logits_raw = tied_embed
            ? mx::matmul(h, mx::swapaxes(em, 0, 1))
            : lin(h, "lm_head", wm, gs);
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

// ── Batched prefill ───────────────────────────────────────────────────────────

Result<std::vector<float>> GemmaModelMLX::run_prefill(const std::vector<int>& prompt_ids) {
    const int    seq_len    = (int)prompt_ids.size();
    const int    n_layers   = (int)config_.num_hidden_layers;
    const size_t n_heads    = config_.num_attention_heads;
    const size_t n_kv       = config_.num_key_value_heads;
    const size_t head_dim   = config_.effective_head_dim();
    const size_t hidden     = config_.hidden_size;
    const size_t vocab_size = config_.vocab_size;
    const float  rms_eps    = config_.rms_norm_eps;
    const int    rope_dims  = (int)head_dim;
    const int    gs         = config_.quantization ? config_.quantization->group_size : 64;
    const float  embed_scale = std::sqrt(static_cast<float>(hidden));
    const float  scale      = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const bool   tied_embed = mlx_weights_.find("lm_head.weight") == mlx_weights_.end();

    std::vector<float> rope_thetas(n_layers);
    for (int i = 0; i < n_layers; ++i) {
        bool is_local = config_.is_local_layer(i);
        rope_thetas[i] = (is_local && config_.rope_local_base_freq.has_value())
            ? *config_.rope_local_base_freq
            : config_.rope_theta;
    }

    const WM&       wm = mlx_weights_;
    const mx::array em = embed_mat_;

    std::vector<int32_t> ids_i32(prompt_ids.begin(), prompt_ids.end());
    auto token_ids = mx::array(ids_i32.data(), {seq_len}, mx::int32);

    // Embed all tokens and apply Gemma scale
    mx::array h = mx::take(em, token_ids, 0) * embed_scale;

    // Causal mask — dtype matches embedding table
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
        const float rope_theta = rope_thetas[i];

        auto normed = mx::fast::rms_norm(
            h, 1.0f + wm.at(lpfx + "input_layernorm.weight"), rms_eps);

        auto q_raw = lin(normed, apfx + "q_proj", wm, gs);
        auto k_raw = lin(normed, apfx + "k_proj", wm, gs);
        auto v_raw = lin(normed, apfx + "v_proj", wm, gs);

        auto qt = mx::swapaxes(mx::reshape(q_raw, {seq_len, (int)n_heads, (int)head_dim}), 0, 1);
        auto kt = mx::swapaxes(mx::reshape(k_raw, {seq_len, (int)n_kv,   (int)head_dim}), 0, 1);
        auto vt = mx::swapaxes(mx::reshape(v_raw, {seq_len, (int)n_kv,   (int)head_dim}), 0, 1);

        // Q/K per-head norms (Gemma3, optional)
        {
            auto it = wm.find(apfx + "q_norm.weight");
            if (it != wm.end()) {
                auto f = mx::reshape(qt, {(int)(n_heads * seq_len), (int)head_dim});
                qt = mx::reshape(mx::fast::rms_norm(f, 1.0f + it->second, rms_eps),
                                 {(int)n_heads, seq_len, (int)head_dim});
            }
        }
        {
            auto it = wm.find(apfx + "k_norm.weight");
            if (it != wm.end()) {
                auto f = mx::reshape(kt, {(int)(n_kv * seq_len), (int)head_dim});
                kt = mx::reshape(mx::fast::rms_norm(f, 1.0f + it->second, rms_eps),
                                 {(int)n_kv, seq_len, (int)head_dim});
            }
        }

        auto q_rope = mx::reshape(
            mx::fast::rope(mx::reshape(qt, {1, (int)n_heads, seq_len, (int)head_dim}),
                           rope_dims, false, rope_theta, 1.0f, 0),
            {(int)n_heads, seq_len, (int)head_dim});
        auto k_rope = mx::reshape(
            mx::fast::rope(mx::reshape(kt, {1, (int)n_kv, seq_len, (int)head_dim}),
                           rope_dims, false, rope_theta, 1.0f, 0),
            {(int)n_kv, seq_len, (int)head_dim});

        out_kv_k.push_back(k_rope);
        out_kv_v.push_back(vt);

        auto q4    = mx::reshape(q_rope, {1, (int)n_heads, seq_len, (int)head_dim});
        auto k4    = mx::reshape(k_rope, {1, (int)n_kv,   seq_len, (int)head_dim});
        auto v4    = mx::reshape(vt,     {1, (int)n_kv,   seq_len, (int)head_dim});
        auto attn4 = mx::fast::scaled_dot_product_attention(q4, k4, v4, scale, "", causal_mask);
        auto attn_flat = mx::reshape(mx::swapaxes(attn4, 1, 2),
                                     {seq_len, (int)(n_heads * head_dim)});

        auto attn_out = lin(attn_flat, apfx + "o_proj", wm, gs);

        h = h + mx::fast::rms_norm(
            attn_out, 1.0f + wm.at(lpfx + "post_attention_layernorm.weight"), rms_eps);

        auto normed_ffn = mx::fast::rms_norm(
            h, 1.0f + wm.at(lpfx + "pre_feedforward_layernorm.weight"), rms_eps);
        auto gate_m  = lin(normed_ffn, mpfx + "gate_proj", wm, gs);
        auto up      = lin(normed_ffn, mpfx + "up_proj",   wm, gs);
        auto gelu_gate = 0.5f * gate_m * (1.0f + mx::erf(gate_m * static_cast<float>(1.0 / std::sqrt(2.0))));
        auto ffn_out = lin(gelu_gate * up, mpfx + "down_proj", wm, gs);

        h = h + mx::fast::rms_norm(
            ffn_out, 1.0f + wm.at(lpfx + "post_feedforward_layernorm.weight"), rms_eps);
    }

    h = mx::fast::rms_norm(h, 1.0f + wm.at("model.norm.weight"), rms_eps);

    // Extract last token: {1, hidden}
    auto h_last = mx::reshape(
        mx::take(h, mx::array(seq_len - 1, mx::int32), 0), {1, (int)hidden});

    auto logits_raw = tied_embed
        ? mx::matmul(h_last, mx::swapaxes(em, 0, 1))
        : lin(h_last, "lm_head", wm, gs);
    auto logits = mx::astype(mx::reshape(logits_raw, {(int)vocab_size}), mx::float32);

    // Evaluate logits + all KV in one dispatch
    std::vector<mx::array> to_eval;
    to_eval.reserve(1 + 2 * n_layers);
    to_eval.push_back(logits);
    for (int i = 0; i < n_layers; ++i) {
        to_eval.push_back(out_kv_k[i]);
        to_eval.push_back(out_kv_v[i]);
    }
    mx::eval(to_eval);

    auto& st   = *mlx_state_;
    st.kv_keys = std::move(out_kv_k);
    st.kv_vals = std::move(out_kv_v);
    cache_position_ = static_cast<size_t>(seq_len);

    std::vector<float> result(vocab_size);
    std::copy(logits.data<float>(), logits.data<float>() + vocab_size, result.data());
    return result;
}

// ── Single decode step ────────────────────────────────────────────────────────

Result<std::vector<float>> GemmaModelMLX::run_decode_step(int token_id) {
    auto& st = *mlx_state_;
    const int n = (int)st.kv_keys.size();

    std::vector<mx::array> inputs;
    inputs.reserve(2 + 2 * n);

    int32_t tid = static_cast<int32_t>(token_id);
    inputs.push_back(mx::array(&tid, {1}, mx::int32));
    int32_t pos_val = static_cast<int32_t>(cache_position_);
    inputs.push_back(mx::array(&pos_val, {1}, mx::int32));
    for (int i = 0; i < n; ++i) {
        inputs.push_back(st.kv_keys[i]);
        inputs.push_back(st.kv_vals[i]);
    }

    auto outputs = st.compiled_fn(inputs);

    // Evaluate only logits; KV caches lazily evaluated in the next step.
    mx::eval(outputs[0]);

    for (int i = 0; i < n; ++i) {
        st.kv_keys[i] = outputs[1 + 2*i];
        st.kv_vals[i] = outputs[1 + 2*i + 1];
    }
    ++cache_position_;

    const mx::array& logits = outputs[0];
    std::vector<float> result(config_.vocab_size);
    std::copy(logits.data<float>(), logits.data<float>() + config_.vocab_size, result.data());
    return result;
}

// ── LanguageModel interface ───────────────────────────────────────────────────

Result<std::vector<float>> GemmaModelMLX::prefill(const std::vector<int>& prompt_ids) {
    if (prompt_ids.empty())
        return std::unexpected(Error{ErrorCode::InvalidInput, "prefill: prompt_ids cannot be empty"});
    reset_cache();
    mlx_state_.emplace();
    build_decode_fn();
    return run_prefill(prompt_ids);
}

Result<std::vector<float>> GemmaModelMLX::decode(int token_id) {
    if (cache_position_ == 0)
        return std::unexpected(Error{ErrorCode::InvalidInput,
            "decode: must call prefill() before decode()"});
    return run_decode_step(token_id);
}

Result<std::vector<int>> GemmaModelMLX::generate(
    const std::vector<int>& input_ids,
    size_t max_new_tokens,
    SamplingParams params,
    std::function<bool(int)> on_token)
{
    if (params.temperature < 1e-6f)
        return gemma_generate_pipelined(input_ids, max_new_tokens, params, on_token);
    return GenerateHelper::run(
        input_ids, max_new_tokens, params, on_token, config_,
        [this](const std::vector<int>& ids) { return prefill(ids); },
        [this](int tok) { return decode(tok); });
}

// ── GPU-pipelined greedy generate ─────────────────────────────────────────────

Result<std::vector<int>> GemmaModelMLX::gemma_generate_pipelined(
    const std::vector<int>& input_ids,
    size_t max_new_tokens,
    SamplingParams /*params*/,
    std::function<bool(int)> on_token)
{
    std::unordered_set<int> eos_set;
    if (config_.eos_token_ids.has_value()) {
        for (int id : *config_.eos_token_ids) eos_set.insert(id);
    } else {
        eos_set.insert(2);
    }

    auto prefill_result = prefill(input_ids);
    if (!prefill_result) return std::unexpected(prefill_result.error());

    std::vector<int> generated;
    generated.reserve(max_new_tokens);

    if (max_new_tokens == 0) return generated;

    const auto& fl = *prefill_result;
    int first_token = (int)(std::max_element(fl.begin(), fl.end()) - fl.begin());
    generated.push_back(first_token);
    if (on_token && !on_token(first_token)) return generated;
    if (eos_set.count(first_token) || generated.size() >= max_new_tokens) return generated;

    auto& st  = *mlx_state_;
    const int n = (int)st.kv_keys.size();

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

    // ── Step 0: kick off the pipeline ────────────────────────────────────────
    {
        int32_t fv   = (int32_t)first_token;
        auto    out0 = st.compiled_fn(
            build_inputs(mx::array(&fv, {1}, mx::int32), (int32_t)cache_position_));

        // Lazy argmax: not yet evaluated on CPU.
        auto prev_tok_lazy = mx::reshape(mx::argmax(out0[0]), {1});
        mx::async_eval(prev_tok_lazy);

        for (int i = 0; i < n; ++i) {
            st.kv_keys[i] = out0[1 + 2*i];
            st.kv_vals[i] = out0[1 + 2*i + 1];
        }
        ++cache_position_;

        // ── Pipelined loop ────────────────────────────────────────────────────
        for (size_t step = 1; step + 1 < max_new_tokens; ++step) {
            // Build step N graph with unevaluated token from step N-1.
            auto out_n = st.compiled_fn(build_inputs(prev_tok_lazy, (int32_t)cache_position_));
            auto curr_tok_lazy = mx::reshape(mx::argmax(out_n[0]), {1});
            mx::async_eval(curr_tok_lazy);  // GPU starts step N

            // Block until step N-1 is done; GPU runs step N concurrently.
            mx::eval(prev_tok_lazy);
            int prev_token = prev_tok_lazy.item<int32_t>();

            for (int i = 0; i < n; ++i) {
                st.kv_keys[i] = out_n[1 + 2*i];
                st.kv_vals[i] = out_n[1 + 2*i + 1];
            }
            ++cache_position_;

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
