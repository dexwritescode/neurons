#include "qwen3_transformer_moe_model_mlx.h"
#include "model_loader.h"
#include "sampler.h"

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)

#include "mlx_ops.h"
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <mlx/compile.h>

namespace mx = mlx::core;

namespace compute {

namespace {

using namespace compute::mlx_ops;

// ── switch_mlp MoE step ───────────────────────────────────────────────────────
// x:      [1, hidden]
// mlp_pfx: e.g. "model.layers.3.mlp."
// Returns: [1, hidden]
//
// Qwen3 transformer MoE uses stacked expert matrices (no per-expert weight keys).
// Router selects top_k experts; gather_qmm runs them as a fused batched matmul.
// No shared expert (unlike qwen3_5_moe).
static mx::array mlx_qmoe_switch_step(const mx::array& x, const std::string& mlp_pfx,
                                       const WM& wm, int gs,
                                       size_t num_experts, size_t top_k)
{
    const std::string sw = mlp_pfx + "switch_mlp.";

    // Router
    auto gate_logits = lin(x, mlp_pfx + "gate", wm, gs);
    auto logits_1d   = mx::reshape(gate_logits, {(int)num_experts});
    auto gates_soft  = mx::softmax(mx::astype(logits_1d, mx::float32), -1);

    // Top-k via argpartition — O(n) instead of O(n log n) sort.
    int  kth      = (int)num_experts - (int)top_k;
    auto part_idx = mx::argpartition(gates_soft, kth, 0);
    auto tail_rng = mx::arange(kth, (int)num_experts, 1, mx::int32);
    auto topk_idx = mx::astype(mx::take(part_idx, tail_rng, 0), mx::int32);

    // Normalise weights of selected experts (norm_topk_prob=true).
    auto topk_gates  = mx::take(gates_soft, topk_idx, 0);
    auto norm_scores = topk_gates / mx::sum(topk_gates);

    const mx::array& gw = wm.at(sw + "gate_proj.weight");
    const mx::array& gs_ = wm.at(sw + "gate_proj.scales");
    const mx::array& uw = wm.at(sw + "up_proj.weight");
    const mx::array& us = wm.at(sw + "up_proj.scales");
    const mx::array& dw = wm.at(sw + "down_proj.weight");
    const mx::array& ds = wm.at(sw + "down_proj.scales");

    auto gbit = wm.find(sw + "gate_proj.biases");
    auto ubit = wm.find(sw + "up_proj.biases");
    auto dbit = wm.find(sw + "down_proj.biases");
    std::optional<mx::array> gb = (gbit != wm.end()) ? std::optional(gbit->second) : std::nullopt;
    std::optional<mx::array> ub = (ubit != wm.end()) ? std::optional(ubit->second) : std::nullopt;
    std::optional<mx::array> db = (dbit != wm.end()) ? std::optional(dbit->second) : std::nullopt;

    int up_bits   = bits(gw, gs_, gs);
    int down_bits = bits(dw, ds,  gs);

    // Fused: gather experts by index + quantized matmul in one dispatch.
    // Output: {top_k, 1, moe_intermediate_size}
    auto gate_k = mx::gather_qmm(x, gw, gs_, gb, std::nullopt, topk_idx, true, gs, up_bits);
    auto up_k   = mx::gather_qmm(x, uw, us,  ub, std::nullopt, topk_idx, true, gs, up_bits);
    auto h_k    = mx::sigmoid(gate_k) * gate_k * up_k;

    // Down: {top_k, 1, moe_inter} → {top_k, 1, hidden}
    auto down_k = mx::gather_qmm(h_k, dw, ds, db, std::nullopt, topk_idx, true, gs, down_bits);

    // Weighted sum over experts → {1, hidden}
    auto scores_w = mx::reshape(mx::astype(norm_scores, mx::bfloat16), {(int)top_k, 1, 1});
    return mx::reshape(mx::sum(down_k * scores_w, 0), {1, (int)x.shape(1)});
}

// ── Multi-token attention step (prefill) ──────────────────────────────────────
// x:   [1, T, hidden]
// pos: scalar int32 — RoPE offset for the first token (MLX rope auto-increments)
// Returns: (attn_out [1,T,hidden], new_k [n_kv,T+prev,hd], new_v [n_kv,T+prev,hd])
static std::tuple<mx::array, mx::array, mx::array>
mlx_qmoe_attn_prefill(const mx::array& x, const std::string& attn_pfx,
                       const mx::array& kv_k_prev, const mx::array& kv_v_prev,
                       const mx::array& pos,
                       const WM& wm, int gs,
                       size_t n_heads, size_t n_kv, size_t head_dim,
                       float rope_theta, int rope_dims, float rms_eps)
{
    int T     = x.shape(1);
    int hd    = (int)head_dim;
    auto x2d  = mx::reshape(x, {T, (int)x.shape(2)});

    auto q_raw = lin(x2d, attn_pfx + "q_proj", wm, gs);
    auto k_raw = lin(x2d, attn_pfx + "k_proj", wm, gs);
    auto v_raw = lin(x2d, attn_pfx + "v_proj", wm, gs);

    // [T, heads, head_dim] → [heads, T, head_dim]
    auto qt = mx::swapaxes(mx::reshape(q_raw, {T, (int)n_heads, hd}), 0, 1);
    auto kt = mx::swapaxes(mx::reshape(k_raw, {T, (int)n_kv,   hd}), 0, 1);
    auto vt = mx::swapaxes(mx::reshape(v_raw, {T, (int)n_kv,   hd}), 0, 1);

    // Optional QK-norm (Qwen3)
    {
        auto it = wm.find(attn_pfx + "q_norm.weight");
        if (it != wm.end()) {
            auto f = mx::reshape(qt, {(int)n_heads * T, hd});
            qt = mx::reshape(mx::fast::rms_norm(f, it->second, rms_eps),
                             {(int)n_heads, T, hd});
        }
    }
    {
        auto it = wm.find(attn_pfx + "k_norm.weight");
        if (it != wm.end()) {
            auto f = mx::reshape(kt, {(int)n_kv * T, hd});
            kt = mx::reshape(mx::fast::rms_norm(f, it->second, rms_eps),
                             {(int)n_kv, T, hd});
        }
    }

    // RoPE — mx::fast::rope expects [B, heads, T, head_dim]
    auto q_rope = mx::reshape(
        mx::fast::rope(mx::reshape(qt, {1, (int)n_heads, T, hd}),
                       rope_dims, false, rope_theta, 1.0f, pos),
        {(int)n_heads, T, hd});
    auto k_rope = mx::reshape(
        mx::fast::rope(mx::reshape(kt, {1, (int)n_kv, T, hd}),
                       rope_dims, false, rope_theta, 1.0f, pos),
        {(int)n_kv, T, hd});

    // Grow KV cache
    auto new_k = mx::concatenate({kv_k_prev, k_rope}, 1);
    auto new_v = mx::concatenate({kv_v_prev, vt},     1);

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    auto q4 = mx::reshape(q_rope, {1, (int)n_heads, T,  hd});
    auto k4 = mx::reshape(new_k,  {1, (int)n_kv,   -1, hd});
    auto v4 = mx::reshape(new_v,  {1, (int)n_kv,   -1, hd});
    auto attn4 = mx::fast::scaled_dot_product_attention(q4, k4, v4, scale);

    // [1, n_heads, T, head_dim] → [1, T, hidden]
    auto attn_out = mx::reshape(mx::swapaxes(attn4, 1, 2), {1, T, (int)(n_heads * head_dim)});
    attn_out = mx::reshape(lin(mx::reshape(attn_out, {T, (int)(n_heads * head_dim)}),
                               attn_pfx + "o_proj", wm, gs),
                           {1, T, (int)x.shape(2)});

    return {attn_out, new_k, new_v};
}

} // anonymous namespace

// ═══════════════════════════════════════════════════════════════════════════════
// Factory
// ═══════════════════════════════════════════════════════════════════════════════

Result<Qwen3TransformerMoeModelMLX> Qwen3TransformerMoeModelMLX::from_model_dir(
    const std::filesystem::path& model_dir,
    size_t context_size)
{
    auto result = ModelLoader::load_model_mlx(model_dir);
    if (!result) return std::unexpected(result.error());

    auto& [config, mlx_weights] = *result;

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
                int b  = mlx_ops::bits(ew_it->second, sc_it->second, gs);
                return mx::dequantize(ew_it->second, sc_it->second, bi_it->second, gs, b);
            }
        }
        return ew_it->second;
    }();
    mx::eval(embed_mat);

    return Qwen3TransformerMoeModelMLX(
        std::move(config),
        std::move(*tok_result),
        std::move(mlx_weights),
        std::move(embed_mat),
        context_size);
}

// ── Constructor / num_parameters ─────────────────────────────────────────────

Qwen3TransformerMoeModelMLX::Qwen3TransformerMoeModelMLX(
    ModelConfig                                        config,
    SimpleBpeTokenizer                                 tokenizer,
    std::unordered_map<std::string, mx::array>         mlx_weights,
    mx::array                                          embed_mat,
    size_t                                             context_size)
    : config_(std::move(config))
    , tokenizer_(std::move(tokenizer))
    , mlx_weights_(std::move(mlx_weights))
    , embed_mat_(std::move(embed_mat))
{
    (void)context_size;
}

size_t Qwen3TransformerMoeModelMLX::num_parameters() const {
    size_t total = 0;
    for (const auto& [name, arr] : mlx_weights_) total += arr.size();
    return total;
}

// ── Cache management ──────────────────────────────────────────────────────────

void Qwen3TransformerMoeModelMLX::reset_cache() {
    mlx_state_.reset();
    cache_position_ = 0;
}

void Qwen3TransformerMoeModelMLX::init_empty_decode_state() {
    mlx_state_.emplace();
    auto& st = *mlx_state_;

    const int    n    = (int)config_.num_hidden_layers;
    const size_t n_kv = config_.num_key_value_heads;
    const size_t hd   = config_.effective_head_dim();

    st.kv_keys.reserve(n);
    st.kv_vals.reserve(n);
    for (int i = 0; i < n; ++i) {
        // {n_kv, 0, head_dim} — zero-length sequence axis, grows via concat each step.
        st.kv_keys.push_back(mx::zeros({(int)n_kv, 0, (int)hd}, mx::bfloat16));
        st.kv_vals.push_back(mx::zeros({(int)n_kv, 0, (int)hd}, mx::bfloat16));
    }
}

// ── Compiled decode function ──────────────────────────────────────────────────
// Follows the same mx::compile(shapeless=true) pattern as LlamaModel but with
// switch_mlp MoE in place of the dense SwiGLU FFN.
//
// Input layout: [token_id, pos, kv_k[0..n-1], kv_v[0..n-1]]
// Output layout: [logits, out_kv_k[0..n-1], out_kv_v[0..n-1]]

void Qwen3TransformerMoeModelMLX::build_decode_fn() {
    const int    n_layers    = (int)config_.num_hidden_layers;
    const size_t n_heads     = config_.num_attention_heads;
    const size_t n_kv        = config_.num_key_value_heads;
    const size_t head_dim    = config_.effective_head_dim();
    const size_t hidden      = config_.hidden_size;
    const size_t vocab_size  = config_.vocab_size;
    const float  rms_eps     = config_.rms_norm_eps;
    const float  rope_theta  = config_.rope_theta;
    const int    rope_dims   = (int)(head_dim * config_.partial_rotary_factor.value_or(1.0f));
    const int    gs          = config_.quantization ? config_.quantization->group_size : 64;
    const size_t num_experts = config_.num_experts.value_or(128);
    const size_t top_k       = config_.num_experts_per_tok.value_or(8);

    WM       wm = mlx_weights_;
    mx::array em = embed_mat_;

    auto fn =
        [wm          = std::move(wm),
         em          = std::move(em),
         n_layers, n_heads, n_kv, head_dim, hidden, vocab_size,
         rms_eps, rope_theta, rope_dims, gs,
         num_experts, top_k]
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

        // Embed + decode: hidden is [1, hidden] (2D, seq=1)
        mx::array h = mx::reshape(mx::take(em, token_id, 0), {1, (int)hidden});

        std::vector<mx::array> out_kv_k, out_kv_v;
        out_kv_k.reserve(n_layers); out_kv_v.reserve(n_layers);

        for (int i = 0; i < n_layers; ++i) {
            const std::string lpfx = "model.layers." + std::to_string(i) + ".";
            const std::string apfx = lpfx + "self_attn.";
            const std::string mpfx = lpfx + "mlp.";

            auto normed = mx::fast::rms_norm(h, wm.at(lpfx + "input_layernorm.weight"), rms_eps);

            // QKV projections
            auto q_raw = lin(normed, apfx + "q_proj", wm, gs);
            auto k_raw = lin(normed, apfx + "k_proj", wm, gs);
            auto v_raw = lin(normed, apfx + "v_proj", wm, gs);

            // [1, heads, head_dim] for rope; seq=1 during decode
            auto qt = mx::reshape(q_raw, {(int)n_heads, 1, (int)head_dim});
            auto kt = mx::reshape(k_raw, {(int)n_kv,   1, (int)head_dim});
            auto vt = mx::reshape(v_raw, {(int)n_kv,   1, (int)head_dim});

            // QK-norm (Qwen3)
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

            float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
            auto q4    = mx::reshape(q_rope, {1, (int)n_heads, 1,  (int)head_dim});
            auto k4    = mx::reshape(new_k,  {1, (int)n_kv,   -1, (int)head_dim});
            auto v4    = mx::reshape(new_v,  {1, (int)n_kv,   -1, (int)head_dim});
            auto attn4 = mx::fast::scaled_dot_product_attention(q4, k4, v4, scale, "");

            // [1, n_heads, 1, head_dim] → [1, hidden]
            auto attn_flat = mx::reshape(mx::swapaxes(attn4, 1, 2),
                                         {1, (int)(n_heads * head_dim)});
            h = h + lin(attn_flat, apfx + "o_proj", wm, gs);

            // Post-attention norm + switch_mlp MoE
            auto normed2 = mx::fast::rms_norm(
                h, wm.at(lpfx + "post_attention_layernorm.weight"), rms_eps);
            h = h + mlx_qmoe_switch_step(normed2, mpfx, wm, gs, num_experts, top_k);
        }

        h = mx::fast::rms_norm(h, wm.at("model.norm.weight"), rms_eps);

        // lm_head — if not present, fall back to tied embeddings
        auto logits_raw = wm.count("lm_head.weight")
            ? lin(h, "lm_head", wm, gs)
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

// ── Single decode step ────────────────────────────────────────────────────────

Result<std::vector<float>> Qwen3TransformerMoeModelMLX::run_decode_step(int token_id) {
    auto& st       = *mlx_state_;
    const int n    = (int)st.kv_keys.size();

    std::vector<mx::array> inputs;
    inputs.reserve(2 + 2*n);
    int32_t tid = (int32_t)token_id;
    inputs.push_back(mx::array(&tid, {1}, mx::int32));
    int32_t pos_val = (int32_t)cache_position_;
    inputs.push_back(mx::array(&pos_val, {1}, mx::int32));
    for (int i = 0; i < n; ++i) {
        inputs.push_back(st.kv_keys[i]);
        inputs.push_back(st.kv_vals[i]);
    }

    auto outputs = st.compiled_fn(inputs);

    for (int i = 0; i < n; ++i) {
        st.kv_keys[i] = outputs[1 + 2*i];
        st.kv_vals[i] = outputs[1 + 2*i + 1];
    }

    // Materialize KV cache each step to break the lazy computation chain.
    // Without this, step N holds a reference chain back to step 0, causing
    // unbounded memory growth on long sequences.
    std::vector<mx::array> to_eval;
    to_eval.reserve(1 + 2*n);
    to_eval.push_back(outputs[0]);
    for (int i = 0; i < n; ++i) {
        to_eval.push_back(st.kv_keys[i]);
        to_eval.push_back(st.kv_vals[i]);
    }
    mx::async_eval(to_eval);
    mx::eval(outputs[0]);  // block until logits ready for sampling

    ++cache_position_;

    const mx::array& logits = outputs[0];
    std::vector<float> result(config_.vocab_size);
    std::copy(logits.data<float>(), logits.data<float>() + config_.vocab_size, result.data());
    return result;
}

// ── Batched prefill ───────────────────────────────────────────────────────────
// Processes all T prompt tokens in one eager pass.
// Attention: one SDPA over all T positions per layer (causal via growing KV).
// MoE: T per-token router calls (data-dependent, cannot be batched).

Result<std::vector<float>> Qwen3TransformerMoeModelMLX::run_prefill(
    const std::vector<int>& prompt_ids)
{
    auto& st = *mlx_state_;

    const int    n_layers    = (int)config_.num_hidden_layers;
    const size_t n_heads     = config_.num_attention_heads;
    const size_t n_kv        = config_.num_key_value_heads;
    const size_t head_dim    = config_.effective_head_dim();
    const size_t hidden      = config_.hidden_size;
    const size_t vocab_size  = config_.vocab_size;
    const float  rms_eps     = config_.rms_norm_eps;
    const float  rope_theta  = config_.rope_theta;
    const int    rope_dims   = (int)(head_dim * config_.partial_rotary_factor.value_or(1.0f));
    const int    gs          = config_.quantization ? config_.quantization->group_size : 64;
    const size_t num_experts = config_.num_experts.value_or(128);
    const size_t top_k       = config_.num_experts_per_tok.value_or(8);

    int T = (int)prompt_ids.size();
    std::vector<int32_t> ids32(prompt_ids.begin(), prompt_ids.end());
    mx::array token_ids(ids32.data(), {T}, mx::int32);

    mx::array hidden_arr = mx::reshape(mx::take(embed_mat_, token_ids, 0), {1, T, (int)hidden});
    int32_t start_pos = 0;
    mx::array pos_seq = mx::array(&start_pos, {1}, mx::int32);

    for (int i = 0; i < n_layers; ++i) {
        const std::string lpfx = "model.layers." + std::to_string(i) + ".";
        const std::string apfx = lpfx + "self_attn.";
        const std::string mpfx = lpfx + "mlp.";

        auto normed = mx::fast::rms_norm(
            hidden_arr, mlx_weights_.at(lpfx + "input_layernorm.weight"), rms_eps);

        auto [attn_out, nk, nv] = mlx_qmoe_attn_prefill(
            normed, apfx, st.kv_keys[i], st.kv_vals[i], pos_seq,
            mlx_weights_, gs, n_heads, n_kv, head_dim,
            rope_theta, rope_dims, rms_eps);

        st.kv_keys[i] = std::move(nk);
        st.kv_vals[i] = std::move(nv);
        hidden_arr = hidden_arr + attn_out;

        auto normed2    = mx::fast::rms_norm(
            hidden_arr, mlx_weights_.at(lpfx + "post_attention_layernorm.weight"), rms_eps);
        auto normed2_2d = mx::reshape(normed2, {T, (int)hidden});

        // Per-token MoE: router is data-dependent so we process each token separately.
        std::vector<mx::array> moe_outs;
        moe_outs.reserve(T);
        for (int t = 0; t < T; ++t) {
            auto xt = mx::reshape(
                mx::slice(normed2_2d, {t, 0}, {t+1, (int)hidden}), {1, (int)hidden});
            moe_outs.push_back(mlx_qmoe_switch_step(xt, mpfx, mlx_weights_, gs, num_experts, top_k));
        }
        hidden_arr = hidden_arr +
                     mx::reshape(mx::concatenate(moe_outs, 0), {1, T, (int)hidden});

        // Flush lazy graph each layer to bound peak memory for long prompts.
        std::vector<mx::array> to_eval;
        to_eval.reserve(2 * n_layers + 1);
        to_eval.insert(to_eval.end(), st.kv_keys.begin(), st.kv_keys.end());
        to_eval.insert(to_eval.end(), st.kv_vals.begin(), st.kv_vals.end());
        to_eval.push_back(hidden_arr);
        mx::eval(to_eval);
    }

    // Final norm + lm_head on last token only
    auto last_2d = mx::reshape(
        mx::slice(mx::reshape(hidden_arr, {T, (int)hidden}), {T-1, 0}, {T, (int)hidden}),
        {1, (int)hidden});
    last_2d = mx::fast::rms_norm(last_2d, mlx_weights_.at("model.norm.weight"), rms_eps);

    auto logits_raw = mlx_weights_.count("lm_head.weight")
        ? lin(last_2d, "lm_head", mlx_weights_, gs)
        : mx::matmul(last_2d, mx::swapaxes(embed_mat_, 0, 1));
    auto logits = mx::astype(mx::reshape(logits_raw, {(int)vocab_size}), mx::float32);
    mx::eval(logits);

    cache_position_ = T;
    build_decode_fn();

    std::vector<float> result(vocab_size);
    std::copy(logits.data<float>(), logits.data<float>() + vocab_size, result.data());
    return result;
}

// ── LanguageModel interface ───────────────────────────────────────────────────

Result<std::vector<float>> Qwen3TransformerMoeModelMLX::prefill(
    const std::vector<int>& prompt_ids)
{
    if (prompt_ids.empty())
        return std::unexpected(Error{ErrorCode::InvalidInput, "prefill: empty prompt"});
    reset_cache();
    init_empty_decode_state();
    return run_prefill(prompt_ids);
}

Result<std::vector<float>> Qwen3TransformerMoeModelMLX::decode(int token_id) {
    if (cache_position_ == 0)
        return std::unexpected(Error{ErrorCode::InvalidInput, "decode: call prefill() first"});
    return run_decode_step(token_id);
}

Result<std::vector<int>> Qwen3TransformerMoeModelMLX::generate(
    const std::vector<int>& input_ids,
    size_t max_new_tokens,
    SamplingParams params,
    std::function<bool(int)> on_token)
{
    if (params.temperature < 1e-6f)
        return generate_pipelined(input_ids, max_new_tokens, params, on_token);
    return GenerateHelper::run(
        input_ids, max_new_tokens, params, on_token, config_,
        [this](const std::vector<int>& ids) { return prefill(ids); },
        [this](int tok) { return decode(tok); });
}

// ── Greedy pipelined generate ─────────────────────────────────────────────────
// Overlaps mx::compile dispatch with CPU argmax via mx::async_eval.

Result<std::vector<int>> Qwen3TransformerMoeModelMLX::generate_pipelined(
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

    // First token: CPU argmax from prefill logits.
    const auto& fl = *prefill_result;
    int first_token = (int)(std::max_element(fl.begin(), fl.end()) - fl.begin());
    generated.push_back(first_token);
    if (on_token && !on_token(first_token)) return generated;
    if (eos_set.count(first_token) || generated.size() >= max_new_tokens) return generated;

    auto& st   = *mlx_state_;
    const int n = (int)st.kv_keys.size();

    auto build_inputs = [&](mx::array tok, int32_t pos) {
        std::vector<mx::array> inputs;
        inputs.reserve(2 + 2*n);
        inputs.push_back(std::move(tok));
        inputs.push_back(mx::array(&pos, {1}, mx::int32));
        for (int i = 0; i < n; ++i) {
            inputs.push_back(st.kv_keys[i]);
            inputs.push_back(st.kv_vals[i]);
        }
        return inputs;
    };

    auto unpack_state = [&](const std::vector<mx::array>& out) {
        for (int i = 0; i < n; ++i) {
            st.kv_keys[i] = out[1 + 2*i];
            st.kv_vals[i] = out[1 + 2*i + 1];
        }
    };

    // Kick off decode for the first generated token, pipeline from there.
    int32_t fv   = (int32_t)first_token;
    auto    out0 = st.compiled_fn(build_inputs(mx::array(&fv, {1}, mx::int32),
                                               (int32_t)cache_position_));
    auto prev_tok_lazy = mx::astype(mx::reshape(mx::argmax(out0[0]), {1}), mx::int32);
    mx::async_eval(prev_tok_lazy);

    unpack_state(out0);
    ++cache_position_;

    for (size_t step = 1; step + 1 < max_new_tokens; ++step) {
        auto out_n = st.compiled_fn(build_inputs(prev_tok_lazy, (int32_t)cache_position_));
        auto curr_tok_lazy = mx::astype(mx::reshape(mx::argmax(out_n[0]), {1}), mx::int32);
        mx::async_eval(curr_tok_lazy);

        mx::eval(prev_tok_lazy);
        int prev_token = prev_tok_lazy.item<int32_t>();

        unpack_state(out_n);
        ++cache_position_;

        generated.push_back(prev_token);
        if (on_token && !on_token(prev_token)) return generated;
        if (eos_set.count(prev_token)) return generated;

        prev_tok_lazy = std::move(curr_tok_lazy);
    }

    mx::eval(prev_tok_lazy);
    int last_token = prev_tok_lazy.item<int32_t>();
    generated.push_back(last_token);
    if (on_token) on_token(last_token);

    return generated;
}

} // namespace compute

#endif // MLX_BACKEND_ENABLED
