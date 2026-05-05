#include "qwen3_moe_model_mlx.h"
#include "model_loader.h"

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)

#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <mlx/compile.h>

namespace mx = mlx::core;

namespace compute {

// ═══════════════════════════════════════════════════════════════════════════════
// Anonymous-namespace MLX helper functions (translate-unit local)
// ═══════════════════════════════════════════════════════════════════════════════
namespace {

using WM = std::unordered_map<std::string, mx::array>;

// ── Fused GatedDeltaNet decode kernel ─────────────────────────────────────────
// Mirrors mlx-lm gated_delta.py: custom Metal kernel used in inference.
// Handles an arbitrary sequence of T tokens in one GPU dispatch (T=1 for decode).
static constexpr const char* kGatedDeltaSource = R"msl(
    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;

    // q, k: [B, T, Hk, Dk]
    auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
    auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
    // v, y: [B, T, Hv, Dv]
    auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
    y      += b_idx * T * Hv * Dv + hv_idx * Dv;

    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;

    // state_in, state_out: [B, Hv, Dv, Dk]
    auto i_state = state_in  + (n * Dv + dv_idx) * Dk;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk;

    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i)
        state[i] = (float)i_state[n_per_t * dk_idx + i];

    // g: [B, T, Hv]
    auto g_    = g    + b_idx * T * Hv;
    auto beta_ = beta + b_idx * T * Hv;

    for (int t = 0; t < T; ++t) {
        float kv_mem = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_[hv_idx];
            kv_mem += state[i] * k_[s_idx];
        }
        kv_mem = simd_sum(kv_mem);

        auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

        float out = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_[s_idx] * delta;
            out += state[i] * q_[s_idx];
        }
        out = simd_sum(out);
        if (thread_index_in_simdgroup == 0)
            y[dv_idx] = (InT)out;

        q_ += Hk * Dk;  k_ += Hk * Dk;
        v_ += Hv * Dv;  y  += Hv * Dv;
        g_    += Hv;    beta_ += Hv;
    }
    for (int i = 0; i < n_per_t; ++i)
        o_state[n_per_t * dk_idx + i] = (StT)state[i];
)msl";

static const mx::fast::CustomKernelFunction& gated_delta_kernel() {
    static const auto fn = mx::fast::metal_kernel(
        "neurons_gated_delta",
        {"q", "k", "v", "g", "beta", "state_in", "T"},
        {"y", "state_out"},
        kGatedDeltaSource);
    return fn;
}

static int mlx_bits(const mx::array& w, const mx::array& s, int gs) {
    double ratio = static_cast<double>(w.shape().back()) /
                   static_cast<double>(s.shape().back());
    int bits = static_cast<int>(std::round(32.0 * ratio / static_cast<double>(gs)));
    return (bits > 0) ? bits : 4;
}

static mx::array mlx_lin(const mx::array& x, const std::string& key, const WM& wm, int gs) {
    const mx::array& w = wm.at(key + ".weight");
    auto sit = wm.find(key + ".scales");
    if (sit != wm.end()) {
        int bits = mlx_bits(w, sit->second, gs);
        std::optional<mx::array> bias;
        auto bit = wm.find(key + ".biases");
        if (bit != wm.end()) bias = bit->second;
        return mx::quantized_matmul(x, w, sit->second, bias, true, gs, bits, "affine");
    }
    return mx::matmul(x, mx::swapaxes(w, 0, 1));
}

// x: [1, T, hidden]; pos: scalar/[1] int32 RoPE offset.
// context_size == 0 (default) → growing-KV path: concatenate kv_k/kv_v with new
//   k/v and run SDPA over all filled positions.  Used for both prefill and decode
//   (decode runs with compile(shapeless=true) so shapes are allowed to grow).
// context_size > 0  → fixed-KV path: kv_k/kv_v pre-allocated {nkv,ctx,hd},
//   new k/v scattered at pos via mx::where, SDPA masked over full ctx.  Kept for
//   reference; the growing-KV path is faster in practice.
static std::tuple<mx::array, mx::array, mx::array>
mlx_full_attn_step(const mx::array& x, int layer_idx,
                    const mx::array& kv_k, const mx::array& kv_v,
                    const mx::array& pos,
                    const WM& wm, int gs,
                    size_t n_heads, size_t n_kv, size_t head_dim,
                    float rope_theta, int rope_dims, float rms_eps,
                    int context_size = 0)
{
    const std::string pfx = "language_model.model.layers." +
                             std::to_string(layer_idx) + ".self_attn.";

    int T = x.shape(1);
    auto x2d = mx::reshape(x, {T, (int)x.shape(2)});

    auto q_raw = mlx_lin(x2d, pfx + "q_proj", wm, gs);
    auto k_raw = mlx_lin(x2d, pfx + "k_proj", wm, gs);
    auto v_raw = mlx_lin(x2d, pfx + "v_proj", wm, gs);

    auto q_gate = mx::reshape(q_raw, {T, (int)n_heads, 2, (int)head_dim});
    auto q_h    = mx::take(q_gate, mx::array(0, mx::int32), 2);
    auto gate_h = mx::take(q_gate, mx::array(1, mx::int32), 2);
    auto gate   = mx::reshape(gate_h, {1, T, (int)(n_heads * head_dim)});

    auto qt = mx::swapaxes(q_h, 0, 1);
    auto kt = mx::swapaxes(mx::reshape(k_raw, {T, (int)n_kv, (int)head_dim}), 0, 1);
    auto vt = mx::swapaxes(mx::reshape(v_raw, {T, (int)n_kv, (int)head_dim}), 0, 1);

    {
        auto it = wm.find(pfx + "q_norm.weight");
        if (it != wm.end()) {
            auto f = mx::reshape(qt, {(int)n_heads * T, (int)head_dim});
            qt = mx::reshape(mx::fast::rms_norm(f, it->second, rms_eps),
                             {(int)n_heads, T, (int)head_dim});
        }
    }
    {
        auto it = wm.find(pfx + "k_norm.weight");
        if (it != wm.end()) {
            auto f = mx::reshape(kt, {(int)n_kv * T, (int)head_dim});
            kt = mx::reshape(mx::fast::rms_norm(f, it->second, rms_eps),
                             {(int)n_kv, T, (int)head_dim});
        }
    }

    auto qt4    = mx::reshape(qt, {1, (int)n_heads, T, (int)head_dim});
    auto kt4    = mx::reshape(kt, {1, (int)n_kv,   T, (int)head_dim});
    auto q_rope = mx::reshape(mx::fast::rope(qt4, rope_dims, false, rope_theta, 1.0f, pos),
                              {(int)n_heads, T, (int)head_dim});
    auto k_rope = mx::reshape(mx::fast::rope(kt4, rope_dims, false, rope_theta, 1.0f, pos),
                              {(int)n_kv, T, (int)head_dim});

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Returns (new_k, new_v, attn4) — structured binding avoids default-constructed arrays.
    auto [new_k, new_v, attn4] =
        [&]() -> std::tuple<mx::array, mx::array, mx::array> {
        if (context_size > 0) {
            // Fixed-KV decode path: scatter new k/v at pos, attend with causal mask.
            auto range  = mx::arange(0, context_size, 1, mx::int32);                // {ctx}
            auto is_pos = mx::reshape(mx::equal(range, pos), {1, context_size, 1}); // {1,ctx,1}
            auto nk = mx::where(is_pos,
                          mx::broadcast_to(k_rope, {(int)n_kv, context_size, (int)head_dim}),
                          kv_k);
            auto nv = mx::where(is_pos,
                          mx::broadcast_to(vt, {(int)n_kv, context_size, (int)head_dim}),
                          kv_v);
            // Additive mask: 0 for positions 0..pos, -1e9 for pos+1..ctx-1
            // Must be bfloat16 to match attention output dtype.
            auto mask = mx::astype(
                            mx::reshape(
                                mx::where(
                                    mx::less_equal(range, pos),
                                    mx::zeros({context_size}, mx::float32),
                                    mx::full({context_size}, -1e9f, mx::float32)),
                                {1, 1, 1, context_size}),
                            mx::bfloat16);
            auto q4 = mx::reshape(q_rope, {1, (int)n_heads, 1,           (int)head_dim});
            auto k4 = mx::reshape(nk,     {1, (int)n_kv,   context_size, (int)head_dim});
            auto v4 = mx::reshape(nv,     {1, (int)n_kv,   context_size, (int)head_dim});
            return {nk, nv, mx::fast::scaled_dot_product_attention(q4, k4, v4, scale, "array", mask)};
        } else {
            // Growing-KV prefill path: concatenate and attend (no mask needed for causal).
            auto nk = mx::concatenate({kv_k, k_rope}, 1);
            auto nv = mx::concatenate({kv_v, vt},     1);
            auto q4 = mx::reshape(q_rope, {1, (int)n_heads, T,  (int)head_dim});
            auto k4 = mx::reshape(nk,     {1, (int)n_kv,   -1, (int)head_dim});
            auto v4 = mx::reshape(nv,     {1, (int)n_kv,   -1, (int)head_dim});
            return {nk, nv, mx::fast::scaled_dot_product_attention(q4, k4, v4, scale)};
        }
    }();

    auto attn_flat = mx::reshape(attn4, {1, T, (int)(n_heads * head_dim)});
    auto gated     = attn_flat * mx::sigmoid(gate);
    auto out       = mlx_lin(mx::reshape(gated, {T, (int)(n_heads * head_dim)}),
                              pfx + "o_proj", wm, gs);
    out = mx::reshape(out, {1, T, (int)x.shape(2)});

    return {out, new_k, new_v};
}

// Shared guts for both decode (T=1) and prefill (T>1).
// x:         [1, T, hidden] bfloat16
// conv_state:[ks-1, conv_dim] float32
// rec_state: [Hv, Dv, Dk]   float32
// Returns:   (out [1,T,hidden], new_conv_state [ks-1,conv_dim], new_rec_state [Hv,Dv,Dk])
static std::tuple<mx::array, mx::array, mx::array>
mlx_ssm_step(const mx::array& x, int layer_idx,
              const mx::array& conv_state, const mx::array& rec_state,
              const WM& wm, int gs,
              size_t Hv, size_t Dv, size_t Hk, size_t Dk,
              size_t kernel_size, size_t conv_dim,
              float rms_eps)
{
    const std::string pfx = "language_model.model.layers." +
                             std::to_string(layer_idx) + ".linear_attn.";

    // x is [1, T, hidden] — flatten to [T, hidden] for linear projections
    int T = x.shape(1);
    auto x2d = mx::reshape(x, {T, (int)x.shape(2)});

    auto qkv = mlx_lin(x2d, pfx + "in_proj_qkv", wm, gs);  // [T, conv_dim]
    auto z   = mlx_lin(x2d, pfx + "in_proj_z",   wm, gs);  // [T, Hv*Dv]
    auto b   = mlx_lin(x2d, pfx + "in_proj_b",   wm, gs);  // [T, Hv]
    auto a   = mlx_lin(x2d, pfx + "in_proj_a",   wm, gs);  // [T, Hv]

    const mx::array& A_log    = wm.at(pfx + "A_log");
    const mx::array& dt_bias  = wm.at(pfx + "dt_bias");
    const mx::array& conv1d_w = wm.at(pfx + "conv1d.weight");
    const mx::array& norm_w   = wm.at(pfx + "norm.weight");

    // Conv1d over [ks-1 history + T new tokens], depthwise.
    // conv_input: [ks-1+T, conv_dim]; output: [T, conv_dim] (valid conv, stride 1)
    auto conv_input = mx::concatenate({conv_state, qkv}, 0);
    // New conv state = last (ks-1) rows of conv_input (take avoids Slice in compiled fn)
    auto tail_idx = mx::arange(T, T + (int)(kernel_size - 1), 1, mx::int32);
    auto new_conv = mx::take(conv_input, tail_idx, 0);

    auto ci4      = mx::reshape(conv_input, {1, (int)(kernel_size - 1 + T), (int)conv_dim});
    auto conv_raw = mx::squeeze(mx::conv1d(ci4, conv1d_w, 1, 0, 1, (int)conv_dim), 0);
    // conv_raw: [T, conv_dim]
    auto conv_out = mx::sigmoid(conv_raw) * conv_raw;

    const size_t key_dim = Hk * Dk;
    // Split conv_out into q/k/v along feature dim (take avoids Slice in compiled fn)
    auto q_idx  = mx::arange(0,              (int)key_dim,      1, mx::int32);
    auto k_idx  = mx::arange((int)key_dim,   (int)(2*key_dim),  1, mx::int32);
    auto v_idx  = mx::arange((int)(2*key_dim),(int)conv_dim,     1, mx::int32);
    auto q_flat = mx::take(conv_out, q_idx, 1);
    auto k_flat = mx::take(conv_out, k_idx, 1);
    auto v_flat = mx::take(conv_out, v_idx, 1);

    // Reshape to [1, T, H, D] — B=1 throughout
    auto q_3d = mx::reshape(q_flat, {1, T, (int)Hk, (int)Dk});
    auto k_3d = mx::reshape(k_flat, {1, T, (int)Hk, (int)Dk});
    auto v_3d = mx::reshape(v_flat, {1, T, (int)Hv, (int)Dv});
    auto z_3d = mx::reshape(z,      {1, T, (int)Hv, (int)Dv});

    // Per-head RMS norm on q and k (across Dk dim)
    mx::array ones_Dk = mx::ones({(int)Dk}, mx::float32);
    {
        auto f = mx::reshape(q_3d, {T * (int)Hk, (int)Dk});
        q_3d = mx::reshape(mx::fast::rms_norm(f, ones_Dk, 1e-6f), {1, T, (int)Hk, (int)Dk});
    }
    {
        auto f = mx::reshape(k_3d, {T * (int)Hk, (int)Dk});
        k_3d = mx::reshape(mx::fast::rms_norm(f, ones_Dk, 1e-6f), {1, T, (int)Hk, (int)Dk});
    }

    float inv_sqrt_Dk = 1.0f / std::sqrt(static_cast<float>(Dk));
    q_3d = q_3d * mx::array(inv_sqrt_Dk * inv_sqrt_Dk);
    k_3d = k_3d * mx::array(inv_sqrt_Dk);

    auto a_biased = a + dt_bias;                                  // [T, Hv]
    auto sp       = mx::log(mx::exp(a_biased) + mx::array(1.0f));
    auto g_seq    = mx::exp(-(mx::exp(A_log) * sp));              // [T, Hv]
    auto beta_seq = mx::astype(mx::sigmoid(b), mx::float32);      // [T, Hv]

    // Custom Metal GatedDeltaNet kernel — one GPU dispatch for all T tokens.
    using TA = mx::fast::TemplateArg;
    auto T_arr = mx::array(&T, {}, mx::int32);
    auto gd = gated_delta_kernel()(
        {q_3d,                                                           // [1, T, Hk, Dk]
         k_3d,                                                           // [1, T, Hk, Dk]
         v_3d,                                                           // [1, T, Hv, Dv]
         mx::reshape(mx::astype(g_seq,    mx::float32), {1, T, (int)Hv}),
         mx::reshape(mx::astype(beta_seq, mx::float32), {1, T, (int)Hv}),
         mx::reshape(rec_state, {1, (int)Hv, (int)Dv, (int)Dk}),
         T_arr},
        {mx::Shape{1, T, (int)Hv, (int)Dv}, mx::Shape{1, (int)Hv, (int)Dv, (int)Dk}},
        {mx::bfloat16, mx::float32},
        {32, (int)Dv, 1 * (int)Hv},   // grid(32, Dv, B*Hv); T is a loop inside the kernel
        {32, 4, 1},
        {{"InT", TA{mx::bfloat16}}, {"StT", TA{mx::float32}},
         {"Dk",  TA{(int)Dk}},      {"Dv",  TA{(int)Dv}},
         {"Hk",  TA{(int)Hk}},      {"Hv",  TA{(int)Hv}}},
        std::nullopt, false, mx::Device::gpu);

    // y: [1, T, Hv, Dv] bfloat16; norm_w has Dv elements — norm per (T,Hv) row
    auto y_T    = mx::reshape(gd[0], {T * (int)Hv, (int)Dv});
    auto h_new  = mx::reshape(gd[1], {(int)Hv, (int)Dv, (int)Dk});   // float32

    auto out_normed = mx::fast::rms_norm(y_T, norm_w, rms_eps);       // [T*Hv, Dv]
    auto gated      = mx::reshape(out_normed, {1, T, (int)Hv, (int)Dv}) *
                      mx::sigmoid(z_3d) * z_3d;
    auto gated_flat = mx::reshape(gated, {1, T, (int)(Hv * Dv)});
    auto out        = mlx_lin(gated_flat, pfx + "out_proj", wm, gs);  // [1, T, hidden]

    return {out, new_conv, h_new};
}

static mx::array mlx_moe_mlp_step(const mx::array& x, int layer_idx,
                                    const WM& wm, int gs,
                                    size_t num_experts, size_t top_k)
{
    const std::string pfx = "language_model.model.layers." +
                             std::to_string(layer_idx) + ".mlp.";
    const std::string sw  = pfx + "switch_mlp.";

    // Full softmax on all experts first (matches mlx-lm norm_topk_prob=true path)
    auto gate_logits = mlx_lin(x, pfx + "gate", wm, gs);
    auto logits_1d   = mx::reshape(gate_logits, {(int)num_experts});
    auto gates_soft  = mx::softmax(mx::astype(logits_1d, mx::float32), -1);

    // O(n) top-k via argpartition — kth = n - k ensures top-k lands in [kth:]
    int   kth       = (int)num_experts - (int)top_k;
    auto  part_idx  = mx::argpartition(gates_soft, kth, 0);           // {num_experts} uint32
    auto  tail_rng  = mx::arange(kth, (int)num_experts, 1, mx::int32);
    auto  topk_idx  = mx::astype(mx::take(part_idx, tail_rng, 0),     // {top_k} int32
                                  mx::int32);

    auto topk_gates  = mx::take(gates_soft, topk_idx, 0);             // {top_k}
    auto norm_scores = topk_gates / mx::sum(topk_gates);              // {top_k}

    const mx::array& gw3 = wm.at(sw + "gate_proj.weight");
    const mx::array& gs3 = wm.at(sw + "gate_proj.scales");
    const mx::array& uw3 = wm.at(sw + "up_proj.weight");
    const mx::array& us3 = wm.at(sw + "up_proj.scales");
    const mx::array& dw3 = wm.at(sw + "down_proj.weight");
    const mx::array& ds3 = wm.at(sw + "down_proj.scales");

    auto gb3_it = wm.find(sw + "gate_proj.biases");
    auto ub3_it = wm.find(sw + "up_proj.biases");
    auto db3_it = wm.find(sw + "down_proj.biases");
    std::optional<mx::array> gb3 = (gb3_it != wm.end()) ? std::optional(gb3_it->second) : std::nullopt;
    std::optional<mx::array> ub3 = (ub3_it != wm.end()) ? std::optional(ub3_it->second) : std::nullopt;
    std::optional<mx::array> db3 = (db3_it != wm.end()) ? std::optional(db3_it->second) : std::nullopt;

    int bits      = mlx_bits(gw3, gs3, gs);
    int down_bits = mlx_bits(dw3, ds3, gs);

    // Fused gather + quantized matmul: 3 dispatches instead of 6×take + 3×qmatmul.
    // rhs_indices selects expert topk_idx[i] for each output row i.
    // Output shape: {top_k} + x.shape[:-1] + {out_features} = {top_k, 1, moe_inter}
    auto gate_k = mx::gather_qmm(x, gw3, gs3, gb3,
                                  std::nullopt, topk_idx, true, gs, bits);
    auto up_k   = mx::gather_qmm(x, uw3, us3, ub3,
                                  std::nullopt, topk_idx, true, gs, bits);
    auto h_k    = mx::sigmoid(gate_k) * gate_k * up_k;  // {top_k, 1, moe_inter}

    // Down: {top_k, 1, moe_inter} → {top_k, 1, hidden}
    auto down_k = mx::gather_qmm(h_k, dw3, ds3, db3,
                                  std::nullopt, topk_idx, true, gs, down_bits);

    // Weighted sum over experts → {1, hidden}
    auto scores_w   = mx::reshape(mx::astype(norm_scores, mx::bfloat16), {(int)top_k, 1, 1});
    auto switch_out = mx::reshape(mx::sum(down_k * scores_w, 0), {1, (int)x.shape(1)});

    // Shared expert
    auto se_g    = mlx_lin(x, pfx + "shared_expert.gate_proj", wm, gs);
    auto se_u    = mlx_lin(x, pfx + "shared_expert.up_proj",   wm, gs);
    auto se_h    = mx::sigmoid(se_g) * se_g * se_u;
    auto se_out  = mlx_lin(se_h, pfx + "shared_expert.down_proj", wm, gs);
    auto seg     = mlx_lin(x, pfx + "shared_expert_gate", wm, gs);
    auto se_gated = se_out * mx::sigmoid(seg);

    return switch_out + se_gated;
}

} // anonymous namespace

// ═══════════════════════════════════════════════════════════════════════════════
// Qwen3MoeModelMLX — factory + lifecycle
// ═══════════════════════════════════════════════════════════════════════════════

Result<Qwen3MoeModelMLX> Qwen3MoeModelMLX::from_model_dir(
    const std::filesystem::path& model_dir,
    size_t context_size)
{
    auto result = ModelLoader::load_model_mlx(model_dir);
    if (!result) return std::unexpected(result.error());

    auto& [config, mlx_weights] = *result;

    auto tok_result = SimpleBpeTokenizer::from_model_dir(model_dir);
    if (!tok_result) return std::unexpected(tok_result.error());

    // Dequantize embedding eagerly so compile treats it as a constant.
    auto ew_it = mlx_weights.find("language_model.model.embed_tokens.weight");
    if (ew_it == mlx_weights.end())
        return std::unexpected(Error{ErrorCode::InvalidModel, "embed_tokens.weight missing"});

    mx::array embed_mat = [&]() -> mx::array {
        auto sc_it = mlx_weights.find("language_model.model.embed_tokens.scales");
        if (sc_it != mlx_weights.end()) {
            auto bi_it = mlx_weights.find("language_model.model.embed_tokens.biases");
            if (bi_it != mlx_weights.end()) {
                int gs   = config.quantization ? config.quantization->group_size : 64;
                int bits = mlx_bits(ew_it->second, sc_it->second, gs);
                return mx::dequantize(ew_it->second, sc_it->second, bi_it->second, gs, bits);
            }
        }
        return ew_it->second;
    }();
    mx::eval(embed_mat);

    return Qwen3MoeModelMLX(
        std::move(config),
        std::move(*tok_result),
        std::move(mlx_weights),
        std::move(embed_mat),
        context_size);
}

Qwen3MoeModelMLX::Qwen3MoeModelMLX(
    ModelConfig                                       config,
    SimpleBpeTokenizer                                tokenizer,
    std::unordered_map<std::string, mx::array>        mlx_weights,
    mx::array                                         embed_mat,
    size_t                                            context_size)
    : Qwen3MoeModelBase(std::move(config), std::move(tokenizer), {})
    , mlx_weights_(std::move(mlx_weights))
    , embed_mat_(std::move(embed_mat))
    , context_size_(context_size)
{}

size_t Qwen3MoeModelMLX::num_parameters() const {
    size_t total = 0;
    for (const auto& [name, arr] : mlx_weights_) total += arr.size();
    return total;
}

// ── Cache management ──────────────────────────────────────────────────────────

void Qwen3MoeModelMLX::reset_cache() {
    mlx_state_.reset();
    cache_position_ = 0;
}

void Qwen3MoeModelMLX::init_empty_decode_state() {
    mlx_state_.emplace();
    auto& st = *mlx_state_;

    const int n = (int)config_.num_hidden_layers;
    const size_t Hv = config_.linear_num_value_heads.value_or(32);
    const size_t Dv = config_.linear_value_head_dim.value_or(128);
    const size_t Hk = config_.linear_num_key_heads.value_or(Hv);
    const size_t Dk = config_.linear_key_head_dim.value_or(128);
    const size_t ks = config_.linear_conv_kernel_dim.value_or(4);
    const size_t cd = Hk*Dk*2 + Hv*Dv;
    const size_t nkv = config_.num_key_value_heads;
    const size_t hd  = config_.effective_head_dim();

    for (int i = 0; i < n; ++i) {
        bool is_ssm = (i + 1) % 4 != 0;
        if (is_ssm) {
            st.ssm_conv.push_back(mx::zeros({(int)(ks-1), (int)cd},      mx::float32));
            st.ssm_rec.push_back( mx::zeros({(int)Hv, (int)Dv, (int)Dk}, mx::float32));
        } else {
            st.kv_keys.push_back(mx::zeros({(int)nkv, 0, (int)hd}, mx::bfloat16));
            st.kv_vals.push_back(mx::zeros({(int)nkv, 0, (int)hd}, mx::bfloat16));
        }
    }
}

// ── Decode function builder ───────────────────────────────────────────────────

void Qwen3MoeModelMLX::build_decode_fn() {
    const int    n_layers    = (int)config_.num_hidden_layers;
    const size_t n_heads     = config_.num_attention_heads;
    const size_t n_kv        = config_.num_key_value_heads;
    const size_t head_dim    = config_.effective_head_dim();
    const size_t hidden      = config_.hidden_size;
    const size_t vocab_size  = config_.vocab_size;
    const float  rms_eps     = config_.rms_norm_eps;
    const float  rope_theta  = config_.rope_theta;
    const int    rope_dims   = (int)(head_dim * config_.partial_rotary_factor.value_or(0.25f));
    const int    gs          = config_.quantization ? config_.quantization->group_size : 64;
    const size_t num_experts = config_.num_experts.value_or(256);
    const size_t top_k       = config_.num_experts_per_tok.value_or(8);
    const size_t Hv = config_.linear_num_value_heads.value_or(32);
    const size_t Dv = config_.linear_value_head_dim.value_or(128);
    const size_t Hk = config_.linear_num_key_heads.value_or(Hv);
    const size_t Dk = config_.linear_key_head_dim.value_or(128);
    const size_t ks = config_.linear_conv_kernel_dim.value_or(4);
    const size_t cd = Hk*Dk*2 + Hv*Dv;
    int n_full = 0, n_ssm = 0;
    for (int i = 0; i < n_layers; ++i)
        ((i+1) % 4 == 0) ? ++n_full : ++n_ssm;

    WM wm = mlx_weights_;
    mx::array em = embed_mat_;

    auto fn =
        [wm          = std::move(wm),
         em          = std::move(em),
         n_layers, n_full, n_ssm,
         n_heads, n_kv, head_dim, hidden, vocab_size,
         rms_eps, rope_theta, rope_dims, gs,
         num_experts, top_k,
         Hv, Dv, Hk, Dk, ks, cd]
        (const std::vector<mx::array>& inputs) mutable -> std::vector<mx::array>
    {
        const mx::array& token_id = inputs[0];

        std::vector<mx::array> kv_k, kv_v, ssm_conv, ssm_rec;
        kv_k.reserve(n_full);   kv_v.reserve(n_full);
        ssm_conv.reserve(n_ssm); ssm_rec.reserve(n_ssm);
        for (int fi = 0; fi < n_full; ++fi) {
            kv_k.push_back(inputs[1 + 2*fi]);
            kv_v.push_back(inputs[1 + 2*fi + 1]);
        }
        for (int li = 0; li < n_ssm; ++li) {
            ssm_conv.push_back(inputs[1 + 2*n_full + 2*li]);
            ssm_rec.push_back( inputs[1 + 2*n_full + 2*li + 1]);
        }
        const mx::array& pos = inputs[1 + 2*n_full + 2*n_ssm];

        // [1, 1, hidden] throughout so mlx_ssm_step / mlx_full_attn_step see 3-D input.
        mx::array hidden_arr = mx::reshape(mx::take(em, token_id, 0), {1, 1, (int)hidden});

        std::vector<mx::array> out_kv_k, out_kv_v, out_ssm_conv, out_ssm_rec;
        out_kv_k.reserve(n_full);    out_kv_v.reserve(n_full);
        out_ssm_conv.reserve(n_ssm); out_ssm_rec.reserve(n_ssm);
        int fi = 0, li = 0;

        for (int i = 0; i < n_layers; ++i) {
            const std::string lpfx = "language_model.model.layers." +
                                      std::to_string(i) + ".";
            bool is_ssm = (i + 1) % 4 != 0;

            auto normed = mx::fast::rms_norm(
                hidden_arr, wm.at(lpfx + "input_layernorm.weight"), rms_eps);

            mx::array attn_out = [&]() -> mx::array {
                if (is_ssm) {
                    auto [y, nc, nr] = mlx_ssm_step(
                        normed, i, ssm_conv[li], ssm_rec[li],
                        wm, gs, Hv, Dv, Hk, Dk, ks, cd, rms_eps);
                    out_ssm_conv.push_back(nc);
                    out_ssm_rec.push_back(nr);
                    ++li;
                    return y;  // [1, 1, hidden]
                } else {
                    auto [y, nk, nv] = mlx_full_attn_step(
                        normed, i, kv_k[fi], kv_v[fi], pos,
                        wm, gs, n_heads, n_kv, head_dim,
                        rope_theta, rope_dims, rms_eps);
                    out_kv_k.push_back(nk);
                    out_kv_v.push_back(nv);
                    ++fi;
                    return y;  // [1, 1, hidden]
                }
            }();

            hidden_arr = hidden_arr + attn_out;

            auto normed2 = mx::fast::rms_norm(
                hidden_arr, wm.at(lpfx + "post_attention_layernorm.weight"), rms_eps);
            // MoE expects [1, hidden]; result is reshaped back to [1, 1, hidden].
            auto moe_in  = mx::reshape(normed2,    {1, (int)hidden});
            auto moe_out = mlx_moe_mlp_step(moe_in, i, wm, gs, num_experts, top_k);
            hidden_arr   = hidden_arr + mx::reshape(moe_out, {1, 1, (int)hidden});
        }

        hidden_arr = mx::reshape(hidden_arr, {1, (int)hidden});
        hidden_arr = mx::fast::rms_norm(
            hidden_arr, wm.at("language_model.model.norm.weight"), rms_eps);
        hidden_arr = mlx_lin(hidden_arr, "language_model.lm_head", wm, gs);

        // Cast logits to float32 here so run_decode_step needs only one mx::eval.
        auto logits = mx::astype(
            mx::reshape(hidden_arr, {(int)vocab_size}), mx::float32);

        std::vector<mx::array> outputs;
        outputs.reserve(1 + 2*n_full + 2*n_ssm);
        outputs.push_back(logits);
        for (int f = 0; f < n_full; ++f) {
            outputs.push_back(out_kv_k[f]);
            outputs.push_back(out_kv_v[f]);
        }
        for (int l = 0; l < n_ssm; ++l) {
            outputs.push_back(out_ssm_conv[l]);
            outputs.push_back(out_ssm_rec[l]);
        }
        return outputs;
    };

    mlx_state_->compiled_fn = mx::compile(std::move(fn), /*shapeless=*/true);
    mlx_state_->fn_ready    = true;
}

// ── Single decode step ────────────────────────────────────────────────────────

Result<std::vector<float>> Qwen3MoeModelMLX::run_decode_step(int token_id) {
    auto& st = *mlx_state_;
    const int n_full = (int)st.kv_keys.size();
    const int n_ssm  = (int)st.ssm_conv.size();

    std::vector<mx::array> inputs;
    inputs.reserve(1 + 2*n_full + 2*n_ssm + 1);

    int32_t tid = static_cast<int32_t>(token_id);
    inputs.push_back(mx::array(&tid, {1}, mx::int32));
    for (int fi = 0; fi < n_full; ++fi) {
        inputs.push_back(st.kv_keys[fi]);
        inputs.push_back(st.kv_vals[fi]);
    }
    for (int li = 0; li < n_ssm; ++li) {
        inputs.push_back(st.ssm_conv[li]);
        inputs.push_back(st.ssm_rec[li]);
    }
    int32_t pos_val = static_cast<int32_t>(cache_position_);
    inputs.push_back(mx::array(&pos_val, {1}, mx::int32));

    auto outputs = st.compiled_fn(inputs);

    // Evaluate only logits immediately; KV and SSM state remain lazy and are
    // consumed by the next step's compiled_fn call (or async_eval in pipelined path).
    mx::eval(outputs[0]);

    // Unpack updated state
    for (int fi = 0; fi < n_full; ++fi) {
        st.kv_keys[fi] = outputs[1 + 2*fi];
        st.kv_vals[fi] = outputs[1 + 2*fi + 1];
    }
    for (int li = 0; li < n_ssm; ++li) {
        st.ssm_conv[li] = outputs[1 + 2*n_full + 2*li];
        st.ssm_rec[li]  = outputs[1 + 2*n_full + 2*li + 1];
    }

    ++cache_position_;

    const mx::array& logits = outputs[0];  // float32, cast inside compiled fn
    size_t vocab = config_.vocab_size;
    std::vector<float> result(vocab);
    std::copy(logits.data<float>(), logits.data<float>() + vocab, result.data());
    return result;
}

// ── Batched prefill ───────────────────────────────────────────────────────────
// Runs all T prompt tokens in one eager pass:
//   SSM:  1 Metal dispatch per layer  (T-loop inside kernel)
//   Attn: 1 SDPA over all T positions per layer
//   MoE:  T per-token calls (router is data-dependent)
Result<std::vector<float>> Qwen3MoeModelMLX::run_prefill(const std::vector<int>& prompt_ids) {
    auto& st = *mlx_state_;

    const int    n_layers    = (int)config_.num_hidden_layers;
    const size_t n_heads     = config_.num_attention_heads;
    const size_t n_kv        = config_.num_key_value_heads;
    const size_t head_dim    = config_.effective_head_dim();
    const size_t hidden      = config_.hidden_size;
    const size_t vocab_size  = config_.vocab_size;
    const float  rms_eps     = config_.rms_norm_eps;
    const float  rope_theta  = config_.rope_theta;
    const int    rope_dims   = (int)(head_dim * config_.partial_rotary_factor.value_or(0.25f));
    const int    gs          = config_.quantization ? config_.quantization->group_size : 64;
    const size_t num_experts = config_.num_experts.value_or(256);
    const size_t top_k       = config_.num_experts_per_tok.value_or(8);
    const size_t Hv = config_.linear_num_value_heads.value_or(32);
    const size_t Dv = config_.linear_value_head_dim.value_or(128);
    const size_t Hk = config_.linear_num_key_heads.value_or(Hv);
    const size_t Dk = config_.linear_key_head_dim.value_or(128);
    const size_t ks = config_.linear_conv_kernel_dim.value_or(4);
    const size_t cd = Hk*Dk*2 + Hv*Dv;

    int T = (int)prompt_ids.size();
    std::vector<int32_t> ids32(prompt_ids.begin(), prompt_ids.end());
    mx::array token_ids(ids32.data(), {T}, mx::int32);

    mx::array hidden_arr = mx::reshape(mx::take(embed_mat_, token_ids, 0), {1, T, (int)hidden});
    // rope offset = starting position; MLX rope auto-increments over the T sequence dim.
    int32_t start_pos = 0;
    mx::array pos_seq = mx::array(&start_pos, {1}, mx::int32);

    int fi = 0, li = 0;
    for (int i = 0; i < n_layers; ++i) {
        const std::string lpfx = "language_model.model.layers." + std::to_string(i) + ".";
        bool is_ssm = (i + 1) % 4 != 0;

        auto normed = mx::fast::rms_norm(
            hidden_arr, mlx_weights_.at(lpfx + "input_layernorm.weight"), rms_eps);

        if (is_ssm) {
            auto [y, nc, nr] = mlx_ssm_step(
                normed, i, st.ssm_conv[li], st.ssm_rec[li],
                mlx_weights_, gs, Hv, Dv, Hk, Dk, ks, cd, rms_eps);
            st.ssm_conv[li] = std::move(nc);
            st.ssm_rec[li]  = std::move(nr);
            ++li;
            hidden_arr = hidden_arr + y;
        } else {
            auto [y, nk, nv] = mlx_full_attn_step(
                normed, i, st.kv_keys[fi], st.kv_vals[fi], pos_seq,
                mlx_weights_, gs, n_heads, n_kv, head_dim,
                rope_theta, rope_dims, rms_eps);
            st.kv_keys[fi] = std::move(nk);
            st.kv_vals[fi] = std::move(nv);
            ++fi;
            hidden_arr = hidden_arr + y;
        }

        auto normed2    = mx::fast::rms_norm(
            hidden_arr, mlx_weights_.at(lpfx + "post_attention_layernorm.weight"), rms_eps);
        auto normed2_2d = mx::reshape(normed2, {T, (int)hidden});

        std::vector<mx::array> moe_outs;
        moe_outs.reserve(T);
        for (int t = 0; t < T; ++t) {
            auto xt = mx::reshape(
                mx::slice(normed2_2d, {t, 0}, {t+1, (int)hidden}), {1, (int)hidden});
            moe_outs.push_back(mlx_moe_mlp_step(xt, i, mlx_weights_, gs, num_experts, top_k));
        }
        hidden_arr = hidden_arr +
                     mx::reshape(mx::concatenate(moe_outs, 0), {1, T, (int)hidden});

        // Flush lazy graph each layer to bound peak memory for long prompts.
        {
            std::vector<mx::array> to_eval;
            to_eval.insert(to_eval.end(), st.ssm_conv.begin(), st.ssm_conv.end());
            to_eval.insert(to_eval.end(), st.ssm_rec.begin(),  st.ssm_rec.end());
            to_eval.insert(to_eval.end(), st.kv_keys.begin(),  st.kv_keys.end());
            to_eval.insert(to_eval.end(), st.kv_vals.begin(),  st.kv_vals.end());
            to_eval.push_back(hidden_arr);
            mx::eval(to_eval);
        }
    }

    // Final norm + lm_head on last token only
    auto last_2d = mx::reshape(
        mx::slice(mx::reshape(hidden_arr, {T, (int)hidden}), {T-1, 0}, {T, (int)hidden}),
        {1, (int)hidden});
    last_2d = mx::fast::rms_norm(
        last_2d, mlx_weights_.at("language_model.model.norm.weight"), rms_eps);
    auto logits = mx::astype(
        mx::reshape(mlx_lin(last_2d, "language_model.lm_head", mlx_weights_, gs),
                    {(int)vocab_size}),
        mx::float32);
    mx::eval(logits);

    cache_position_ = T;
    build_decode_fn();

    std::vector<float> result(vocab_size);
    std::copy(logits.data<float>(), logits.data<float>() + vocab_size, result.data());
    return result;
}

// ── LanguageModel interface ───────────────────────────────────────────────────

Result<std::vector<float>> Qwen3MoeModelMLX::prefill(const std::vector<int>& prompt_ids) {
    if (prompt_ids.empty())
        return std::unexpected(Error{ErrorCode::InvalidInput, "prefill: empty prompt"});

    reset_cache();
    init_empty_decode_state();
    return run_prefill(prompt_ids);
}

Result<std::vector<float>> Qwen3MoeModelMLX::decode(int token_id) {
    if (cache_position_ == 0)
        return std::unexpected(Error{ErrorCode::InvalidInput, "decode: call prefill() first"});
    return run_decode_step(token_id);
}

Result<std::vector<int>> Qwen3MoeModelMLX::generate(
    const std::vector<int>& input_ids,
    size_t max_new_tokens,
    SamplingParams params,
    std::function<bool(int)> on_token)
{
    if (params.temperature < 1e-6f)
        return moe_generate_pipelined(input_ids, max_new_tokens, params, on_token);
    return LanguageModel::generate(input_ids, max_new_tokens, params, on_token);
}

Result<std::vector<int>> Qwen3MoeModelMLX::moe_generate_pipelined(
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

    auto& st      = *mlx_state_;
    const int nf  = (int)st.kv_keys.size();
    const int ns  = (int)st.ssm_conv.size();

    auto build_inputs = [&](mx::array tok, int32_t pos) {
        std::vector<mx::array> inputs;
        inputs.reserve(1 + 2*nf + 2*ns + 1);
        inputs.push_back(std::move(tok));
        for (int i = 0; i < nf; ++i) {
            inputs.push_back(st.kv_keys[i]);
            inputs.push_back(st.kv_vals[i]);
        }
        for (int i = 0; i < ns; ++i) {
            inputs.push_back(st.ssm_conv[i]);
            inputs.push_back(st.ssm_rec[i]);
        }
        inputs.push_back(mx::array(&pos, {1}, mx::int32));
        return inputs;
    };

    auto unpack_state = [&](const std::vector<mx::array>& out) {
        for (int i = 0; i < nf; ++i) {
            st.kv_keys[i] = out[1 + 2*i];
            st.kv_vals[i] = out[1 + 2*i + 1];
        }
        for (int i = 0; i < ns; ++i) {
            st.ssm_conv[i] = out[1 + 2*nf + 2*i];
            st.ssm_rec[i]  = out[1 + 2*nf + 2*i + 1];
        }
    };

    // Step 0: known first_token kicks off the pipeline.
    int32_t fv   = (int32_t)first_token;
    auto    out0 = st.compiled_fn(build_inputs(mx::array(&fv, {1}, mx::int32),
                                               (int32_t)cache_position_));
    // Cast argmax to int32 to reuse the same compile cache entry as the prefill path.
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
