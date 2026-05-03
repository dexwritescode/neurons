#include "qwen3_moe_model_mlx.h"
#include "model_loader.h"

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)

#include <cmath>
#include <algorithm>
#include <chrono>
#include <mlx/compile.h>

namespace mx = mlx::core;

namespace compute {

// ═══════════════════════════════════════════════════════════════════════════════
// Anonymous-namespace MLX helper functions (translate-unit local)
// ═══════════════════════════════════════════════════════════════════════════════
namespace {

using WM = std::unordered_map<std::string, mx::array>;

// ── Fused GatedDeltaNet decode kernel ─────────────────────────────────────────
static constexpr const char* kGatedDeltaDecodeSource = R"msl(
    auto hv_idx = thread_position_in_grid.z;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;

    auto q_    = q    + hk_idx * Dk;
    auto k_    = k    + hk_idx * Dk;
    auto v_    = v    + hv_idx * Dv;
    y         += hv_idx * Dv;

    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;

    auto i_state = state_in  + (hv_idx * Dv + dv_idx) * Dk;
    auto o_state = state_out + (hv_idx * Dv + dv_idx) * Dk;

    float s[n_per_t];
    for (int i = 0; i < n_per_t; ++i)
        s[i] = (float)i_state[n_per_t * dk_idx + i];

    float g_val    = (float)g[hv_idx];
    float beta_val = (float)beta[hv_idx];

    float kv_mem = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
        s[i] *= g_val;
        kv_mem += s[i] * (float)k_[n_per_t * dk_idx + i];
    }
    kv_mem = simd_sum(kv_mem);

    float delta = ((float)v_[dv_idx] - kv_mem) * beta_val;

    float out = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
        s[i] += (float)k_[n_per_t * dk_idx + i] * delta;
        out  += s[i] * (float)q_[n_per_t * dk_idx + i];
    }
    out = simd_sum(out);
    if (thread_index_in_simdgroup == 0)
        y[dv_idx] = (InT)out;

    for (int i = 0; i < n_per_t; ++i)
        o_state[n_per_t * dk_idx + i] = (StT)s[i];
)msl";

static const mx::fast::CustomKernelFunction& gated_delta_kernel() {
    static const auto fn = mx::fast::metal_kernel(
        "neurons_gated_delta_decode",
        {"q", "k", "v", "g", "beta", "state_in"},
        {"y", "state_out"},
        kGatedDeltaDecodeSource);
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

static std::tuple<mx::array, mx::array, mx::array>
mlx_full_attn_step(const mx::array& x, int layer_idx,
                    const mx::array& kv_k, const mx::array& kv_v,
                    const mx::array& pos, int max_ctx,
                    const WM& wm, int gs,
                    size_t n_heads, size_t n_kv, size_t head_dim,
                    float rope_theta, int rope_dims, float rms_eps)
{
    const std::string pfx = "language_model.model.layers." +
                             std::to_string(layer_idx) + ".self_attn.";

    auto q_raw = mlx_lin(x, pfx + "q_proj", wm, gs);
    auto k_raw = mlx_lin(x, pfx + "k_proj", wm, gs);
    auto v_raw = mlx_lin(x, pfx + "v_proj", wm, gs);

    auto q_gate = mx::reshape(q_raw, {1, (int)n_heads, 2, (int)head_dim});
    auto q_h    = mx::take(q_gate, mx::array(0, mx::int32), 2);
    auto gate_h = mx::take(q_gate, mx::array(1, mx::int32), 2);
    auto gate   = mx::reshape(gate_h, {1, (int)(n_heads * head_dim)});

    auto kt = mx::swapaxes(mx::reshape(k_raw, {1, (int)n_kv, (int)head_dim}), 0, 1);
    auto vt = mx::swapaxes(mx::reshape(v_raw, {1, (int)n_kv, (int)head_dim}), 0, 1);
    auto qt = mx::swapaxes(q_h, 0, 1);

    {
        auto it = wm.find(pfx + "q_norm.weight");
        if (it != wm.end()) {
            auto f = mx::reshape(qt, {(int)n_heads, (int)head_dim});
            qt = mx::reshape(mx::fast::rms_norm(f, it->second, rms_eps),
                             {(int)n_heads, 1, (int)head_dim});
        }
    }
    {
        auto it = wm.find(pfx + "k_norm.weight");
        if (it != wm.end()) {
            auto f = mx::reshape(kt, {(int)n_kv, (int)head_dim});
            kt = mx::reshape(mx::fast::rms_norm(f, it->second, rms_eps),
                             {(int)n_kv, 1, (int)head_dim});
        }
    }

    auto qt4 = mx::reshape(qt, {1, (int)n_heads, 1, (int)head_dim});
    auto kt4 = mx::reshape(kt, {1, (int)n_kv, 1, (int)head_dim});
    auto q_rope = mx::reshape(mx::fast::rope(qt4, rope_dims, false, rope_theta, 1.0f, pos),
                              {(int)n_heads, 1, (int)head_dim});
    auto k_rope = mx::reshape(mx::fast::rope(kt4, rope_dims, false, rope_theta, 1.0f, pos),
                              {(int)n_kv, 1, (int)head_dim});

    // Write new k/v into pre-allocated buffers at current position (dynamic index, fixed shape).
    auto new_k = mx::slice_update(kv_k, k_rope, pos, {1});  // {nkv, max_ctx, hd}
    auto new_v = mx::slice_update(kv_v, vt,     pos, {1});  // {nkv, max_ctx, hd}

    // Additive causal mask: future positions get a large negative so softmax ignores them.
    auto positions = mx::arange(0, max_ctx, 1, mx::int32);
    auto mask = mx::reshape(
        mx::astype(mx::where(positions > pos, mx::array(-1e9f), mx::array(0.0f)),
                   mx::bfloat16),
        {1, 1, 1, max_ctx});

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    auto q4 = mx::reshape(q_rope, {1, (int)n_heads, 1,       (int)head_dim});
    auto k4 = mx::reshape(new_k,  {1, (int)n_kv,   max_ctx, (int)head_dim});
    auto v4 = mx::reshape(new_v,  {1, (int)n_kv,   max_ctx, (int)head_dim});
    auto attn4 = mx::fast::scaled_dot_product_attention(q4, k4, v4, scale, "", mask);

    auto attn_flat = mx::reshape(attn4, {1, (int)(n_heads * head_dim)});
    auto gated = attn_flat * mx::sigmoid(gate);
    auto out   = mlx_lin(gated, pfx + "o_proj", wm, gs);

    return {out, new_k, new_v};
}

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

    auto qkv = mlx_lin(x, pfx + "in_proj_qkv", wm, gs);
    auto z   = mlx_lin(x, pfx + "in_proj_z",   wm, gs);
    auto b   = mlx_lin(x, pfx + "in_proj_b",   wm, gs);
    auto a   = mlx_lin(x, pfx + "in_proj_a",   wm, gs);

    const mx::array& A_log    = wm.at(pfx + "A_log");
    const mx::array& dt_bias  = wm.at(pfx + "dt_bias");
    const mx::array& conv1d_w = wm.at(pfx + "conv1d.weight");
    const mx::array& norm_w   = wm.at(pfx + "norm.weight");

    auto conv_input = mx::concatenate({conv_state, qkv}, 0);
    auto tail_idx   = mx::arange(1, (int)(kernel_size - 1), 1, mx::int32);
    auto conv_tail  = mx::take(conv_state, tail_idx, 0);
    auto new_conv   = mx::concatenate({conv_tail, qkv}, 0);

    auto ci4      = mx::reshape(conv_input, {1, (int)kernel_size, (int)conv_dim});
    auto conv_raw = mx::squeeze(mx::conv1d(ci4, conv1d_w, 1, 0, 1, (int)conv_dim), 0);
    auto conv_out = mx::sigmoid(conv_raw) * conv_raw;

    const size_t key_dim = Hk * Dk;
    auto q_idx  = mx::arange(0,            (int)key_dim,     1, mx::int32);
    auto k_idx  = mx::arange((int)key_dim, (int)(2*key_dim), 1, mx::int32);
    auto v_idx  = mx::arange((int)(2*key_dim), (int)conv_dim, 1, mx::int32);
    auto q_flat = mx::take(conv_out, q_idx, 1);
    auto k_flat = mx::take(conv_out, k_idx, 1);
    auto v_flat = mx::take(conv_out, v_idx, 1);

    auto q_3d = mx::reshape(q_flat, {1, (int)Hk, (int)Dk});
    auto k_3d = mx::reshape(k_flat, {1, (int)Hk, (int)Dk});
    auto v_3d = mx::reshape(v_flat, {1, (int)Hv, (int)Dv});
    auto z_3d = mx::reshape(z,      {1, (int)Hv, (int)Dv});

    mx::array ones_Dk = mx::ones({(int)Dk}, mx::float32);
    {
        auto f = mx::reshape(q_3d, {(int)Hk, (int)Dk});
        q_3d = mx::reshape(mx::fast::rms_norm(f, ones_Dk, 1e-6f), {1, (int)Hk, (int)Dk});
    }
    {
        auto f = mx::reshape(k_3d, {(int)Hk, (int)Dk});
        k_3d = mx::reshape(mx::fast::rms_norm(f, ones_Dk, 1e-6f), {1, (int)Hk, (int)Dk});
    }

    float inv_sqrt_Dk = 1.0f / std::sqrt(static_cast<float>(Dk));
    q_3d = q_3d * mx::array(inv_sqrt_Dk * inv_sqrt_Dk);
    k_3d = k_3d * mx::array(inv_sqrt_Dk);

    auto a_biased = a + dt_bias;
    auto sp       = mx::log(mx::exp(a_biased) + mx::array(1.0f));
    auto g_seq    = mx::exp(-(mx::exp(A_log) * sp));
    auto beta_seq = mx::astype(mx::sigmoid(b), mx::float32);

    // Fused GatedDeltaNet recurrence — single Metal dispatch via custom kernel.
    using TA = mx::fast::TemplateArg;
    auto gd = gated_delta_kernel()(
        {mx::reshape(q_3d, {(int)Hk, (int)Dk}),
         mx::reshape(k_3d, {(int)Hk, (int)Dk}),
         mx::reshape(v_3d, {(int)Hv, (int)Dv}),
         mx::reshape(mx::astype(g_seq, mx::float32), {(int)Hv}),
         mx::reshape(beta_seq, {(int)Hv}),
         rec_state},
        {mx::Shape{(int)Hv, (int)Dv}, mx::Shape{(int)Hv, (int)Dv, (int)Dk}},
        {mx::bfloat16, mx::float32},
        {32, (int)Dv, (int)Hv},
        {32, 4, 1},
        {{"InT", TA{mx::bfloat16}}, {"StT", TA{mx::float32}},
         {"Dk",  TA{(int)Dk}},      {"Dv",  TA{(int)Dv}},
         {"Hk",  TA{(int)Hk}},      {"Hv",  TA{(int)Hv}}},
        std::nullopt, false, mx::Device::gpu);
    auto y_t   = gd[0];  // {Hv, Dv} bfloat16
    auto h_new = gd[1];  // {Hv, Dv, Dk} float32

    auto out_normed = mx::fast::rms_norm(y_t, norm_w, rms_eps);
    auto gated      = mx::reshape(out_normed, {1, (int)Hv, (int)Dv}) *
                      mx::sigmoid(z_3d) * z_3d;
    auto gated_flat = mx::reshape(gated, {1, (int)(Hv * Dv)});
    auto out        = mlx_lin(gated_flat, pfx + "out_proj", wm, gs);

    return {out, new_conv, h_new};
}

static mx::array mlx_moe_mlp_step(const mx::array& x, int layer_idx,
                                    const WM& wm, int gs,
                                    size_t num_experts, size_t top_k)
{
    const std::string pfx = "language_model.model.layers." +
                             std::to_string(layer_idx) + ".mlp.";
    const std::string sw  = pfx + "switch_mlp.";

    auto gate_logits = mlx_lin(x, pfx + "gate", wm, gs);
    auto logits_1d   = mx::reshape(gate_logits, {(int)num_experts});

    auto sorted_idx  = mx::argsort(mx::negative(logits_1d), 0);
    auto range_arr   = mx::arange(0, (int)top_k, 1, mx::int32);
    auto topk_idx    = mx::take(sorted_idx, range_arr, 0);

    auto topk_logits = mx::take(logits_1d, topk_idx, 0);
    auto scores      = mx::reshape(mx::softmax(topk_logits, -1), {1, (int)top_k});

    const mx::array& gw3 = wm.at(sw + "gate_proj.weight");
    const mx::array& gs3 = wm.at(sw + "gate_proj.scales");
    const mx::array& uw3 = wm.at(sw + "up_proj.weight");
    const mx::array& us3 = wm.at(sw + "up_proj.scales");
    const mx::array& dw3 = wm.at(sw + "down_proj.weight");
    const mx::array& ds3 = wm.at(sw + "down_proj.scales");

    int bits = mlx_bits(gw3, gs3, gs);
    const size_t k        = top_k;
    const size_t out_e    = gw3.shape()[1];
    const size_t in_p     = gw3.shape()[2];
    const size_t gss      = gs3.shape()[2];
    const size_t down_out = dw3.shape()[1];
    const size_t dss      = ds3.shape()[2];

    auto gw_k = mx::take(gw3, topk_idx, 0);
    auto gs_k = mx::take(gs3, topk_idx, 0);
    auto uw_k = mx::take(uw3, topk_idx, 0);
    auto us_k = mx::take(us3, topk_idx, 0);
    auto dw_k = mx::take(dw3, topk_idx, 0);
    auto ds_k = mx::take(ds3, topk_idx, 0);

    auto gw2 = mx::reshape(gw_k, {(int)(k*out_e), (int)in_p});
    auto gs2 = mx::reshape(gs_k, {(int)(k*out_e), (int)gss});
    auto uw2 = mx::reshape(uw_k, {(int)(k*out_e), (int)in_p});
    auto us2 = mx::reshape(us_k, {(int)(k*out_e), (int)gss});

    std::optional<mx::array> gb2, ub2;
    {
        auto gb_it = wm.find(sw + "gate_proj.biases");
        if (gb_it != wm.end()) {
            auto gb_k = mx::take(gb_it->second, topk_idx, 0);
            gb2 = mx::reshape(gb_k, {(int)(k*out_e), (int)gb_it->second.shape()[2]});
        }
    }
    {
        auto ub_it = wm.find(sw + "up_proj.biases");
        if (ub_it != wm.end()) {
            auto ub_k = mx::take(ub_it->second, topk_idx, 0);
            ub2 = mx::reshape(ub_k, {(int)(k*out_e), (int)ub_it->second.shape()[2]});
        }
    }

    auto gate_k  = mx::quantized_matmul(x, gw2, gs2, gb2, true, gs, bits, "affine");
    auto up_k    = mx::quantized_matmul(x, uw2, us2, ub2, true, gs, bits, "affine");
    auto gate_ke = mx::reshape(gate_k, {(int)k, (int)out_e});
    auto up_ke   = mx::reshape(up_k,   {(int)k, (int)out_e});
    auto h_ke    = mx::sigmoid(gate_ke) * gate_ke * up_ke;

    std::optional<mx::array> db_k;
    {
        auto db_it = wm.find(sw + "down_proj.biases");
        if (db_it != wm.end())
            db_k = mx::take(db_it->second, topk_idx, 0);
    }
    int d_bits   = mlx_bits(dw_k, ds_k, gs);
    auto h_k3    = mx::reshape(h_ke, {(int)k, 1, (int)out_e});
    auto down_k3 = mx::quantized_matmul(h_k3, dw_k, ds_k, db_k, true, gs, d_bits, "affine");
    auto down_2d = mx::reshape(down_k3, {(int)k, (int)down_out});
    auto switch_out = mx::matmul(scores, down_2d);

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

    const int max_ctx = (context_size_ > 0)
        ? (int)context_size_
        : 4096;
    st.max_ctx = max_ctx;

    for (int i = 0; i < n; ++i) {
        bool is_ssm = (i + 1) % 4 != 0;
        if (is_ssm) {
            st.ssm_conv.push_back(mx::zeros({(int)(ks-1), (int)cd},      mx::float32));
            st.ssm_rec.push_back( mx::zeros({(int)Hv, (int)Dv, (int)Dk}, mx::float32));
        } else {
            st.kv_keys.push_back(mx::zeros({(int)nkv, max_ctx, (int)hd}, mx::bfloat16));
            st.kv_vals.push_back(mx::zeros({(int)nkv, max_ctx, (int)hd}, mx::bfloat16));
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

    const int max_ctx = mlx_state_->max_ctx;

    WM wm = mlx_weights_;
    mx::array em = embed_mat_;

    auto fn =
        [wm          = std::move(wm),
         em          = std::move(em),
         n_layers, n_full, n_ssm,
         n_heads, n_kv, head_dim, hidden, vocab_size,
         rms_eps, rope_theta, rope_dims, gs,
         num_experts, top_k,
         Hv, Dv, Hk, Dk, ks, cd,
         max_ctx]
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

        mx::array hidden_arr = mx::reshape(mx::take(em, token_id, 0), {1, (int)hidden});

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
                    return y;
                } else {
                    auto [y, nk, nv] = mlx_full_attn_step(
                        normed, i, kv_k[fi], kv_v[fi], pos,
                        max_ctx,
                        wm, gs, n_heads, n_kv, head_dim,
                        rope_theta, rope_dims, rms_eps);
                    out_kv_k.push_back(nk);
                    out_kv_v.push_back(nv);
                    ++fi;
                    return y;
                }
            }();

            hidden_arr = hidden_arr + attn_out;

            auto normed2 = mx::fast::rms_norm(
                hidden_arr, wm.at(lpfx + "post_attention_layernorm.weight"), rms_eps);
            hidden_arr = hidden_arr + mlx_moe_mlp_step(
                normed2, i, wm, gs, num_experts, top_k);
        }

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

    mlx_state_->compiled_fn = mx::compile(std::move(fn));
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

    auto t0 = std::chrono::steady_clock::now();
    auto outputs = st.compiled_fn(inputs);
    auto t1 = std::chrono::steady_clock::now();
    mx::eval(outputs);
    auto t2 = std::chrono::steady_clock::now();
    if (cache_position_ >= 80 && cache_position_ <= 90) {
        auto compile_ms = std::chrono::duration<double, std::milli>(t1-t0).count();
        auto eval_ms    = std::chrono::duration<double, std::milli>(t2-t1).count();
        fprintf(stderr, "[pos %zu] compiled_fn: %.1fms  eval: %.1fms  total: %.1fms\n",
                cache_position_, compile_ms, eval_ms, compile_ms+eval_ms);
    }

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

// ── LanguageModel interface ───────────────────────────────────────────────────

Result<std::vector<float>> Qwen3MoeModelMLX::prefill(const std::vector<int>& prompt_ids) {
    if (prompt_ids.empty())
        return std::unexpected(Error{ErrorCode::InvalidInput, "prefill: empty prompt"});

    reset_cache();
    init_empty_decode_state();
    build_decode_fn();

    // Run T sequential decode steps — correct but O(T); parallel prefill is a future opt.
    std::vector<float> last_logits;
    for (int token_id : prompt_ids) {
        auto r = run_decode_step(token_id);
        if (!r) return std::unexpected(r.error());
        last_logits = std::move(*r);
    }
    return last_logits;
}

Result<std::vector<float>> Qwen3MoeModelMLX::decode(int token_id) {
    if (cache_position_ == 0)
        return std::unexpected(Error{ErrorCode::InvalidInput, "decode: call prefill() first"});
    return run_decode_step(token_id);
}

} // namespace compute

#endif // MLX_BACKEND_ENABLED
