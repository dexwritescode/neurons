#include "llama_model.h"
#include "model_loader.h"
#include "sampler.h"
#include <sstream>
#include <unordered_set>

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
#include <mlx/mlx.h>
#include <mlx/compile.h>
#include "mlx_ops.h"
#endif

namespace compute {

// ── Private constructor ───────────────────────────────────────────────────────

LlamaModel::LlamaModel(
    ModelConfig        config,
    SimpleBpeTokenizer tokenizer,
    ComputeBackend*    backend)
    : config_(std::move(config))
    , tokenizer_(std::move(tokenizer))
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
#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
    // O.6.2: load weights directly as mx::array — no Tensor intermediary.
    // weights_ is left empty; all inference goes through the MLX path.
    namespace mx = mlx::core;

    auto mlx_result = ModelLoader::load_model_mlx(model_dir);
    if (!mlx_result) return std::unexpected(mlx_result.error());

    auto& [config, mlx_weights] = *mlx_result;

    auto tokenizer_result = SimpleBpeTokenizer::from_model_dir(model_dir);
    if (!tokenizer_result) return std::unexpected(tokenizer_result.error());

    if (!config.is_valid())
        return std::unexpected(Error{ErrorCode::InvalidModel, "Invalid model configuration"});
    if (!config.is_supported_architecture())
        return std::unexpected(Error{ErrorCode::InvalidModel,
            "Unsupported model architecture: " + config.model_type});

    auto ew_it = mlx_weights.find("model.embed_tokens.weight");
    if (ew_it == mlx_weights.end())
        return std::unexpected(Error{ErrorCode::InvalidModel, "embed_tokens.weight missing"});

    mx::array embed_mat = [&]() -> mx::array {
        int gs = config.quantization ? config.quantization->group_size : 64;
        auto sc = mlx_weights.find("model.embed_tokens.scales");
        if (sc != mlx_weights.end()) {
            auto bi = mlx_weights.find("model.embed_tokens.biases");
            if (bi != mlx_weights.end()) {
                int b = mlx_ops::bits(ew_it->second, sc->second, gs);
                return mx::dequantize(ew_it->second, sc->second, bi->second, gs, b);
            }
        }
        return ew_it->second;
    }();
    mx::eval(embed_mat);

    LlamaModel model(std::move(config), std::move(*tokenizer_result), backend);
    model.mlx_setup(std::move(mlx_weights), std::move(embed_mat), context_size);
    return model;
#else
    (void)model_dir; (void)backend; (void)context_size;
    return std::unexpected(Error{ErrorCode::BackendNotAvailable, "LlamaModel requires MLX backend"});
#endif
}

// ── KV-cache public API ───────────────────────────────────────────────────────

void LlamaModel::reset_cache() {
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
// Non-greedy and non-MLX paths use GenerateHelper::run() (the shared loop).

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
    return GenerateHelper::run(
        input_ids, max_new_tokens, params, on_token, config_,
        [this](const std::vector<int>& ids) { return prefill(ids); },
        [this](int tok) { return decode(tok); });
}

Result<std::vector<float>> LlamaModel::prefill(const std::vector<int>& prompt_ids) {
    if (prompt_ids.empty())
        return std::unexpected(Error{ErrorCode::InvalidInput, "prefill: prompt_ids cannot be empty"});
    reset_cache();
    mlx_state_.emplace();
    mlx_build_decode_fn();
    return mlx_prefill_batch(prompt_ids);
}

Result<std::vector<float>> LlamaModel::decode(int token_id) {
    if (mlx_pos_ == 0)
        return std::unexpected(Error{ErrorCode::InvalidInput,
            "decode: must call prefill() before decode()"});
    return mlx_run_step(token_id);
}

// ── Metadata ──────────────────────────────────────────────────────────────────

size_t LlamaModel::num_parameters() const {
    size_t total = 0;
    for (const auto& [name, w] : mlx_weights_)
        total += static_cast<size_t>(w.size());
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
using namespace compute::mlx_ops;
} // anonymous namespace

void LlamaModel::mlx_setup(
    std::unordered_map<std::string, mx::array> mlx_weights,
    mx::array                                  mlx_embed_mat,
    size_t                                     context_size)
{
    mlx_weights_   = std::move(mlx_weights);
    mlx_embed_mat_ = std::move(mlx_embed_mat);
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

            auto q_raw = lin(normed, apfx + "q_proj", wm, gs);
            auto k_raw = lin(normed, apfx + "k_proj", wm, gs);
            auto v_raw = lin(normed, apfx + "v_proj", wm, gs);

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

            h = h + lin(attn_flat, apfx + "o_proj", wm, gs);

            auto normed2 = mx::fast::rms_norm(
                h, wm.at(lpfx + "post_attention_layernorm.weight"), rms_eps);
            auto gate_m  = lin(normed2, mpfx + "gate_proj", wm, gs);
            auto up      = lin(normed2, mpfx + "up_proj",   wm, gs);
            h = h + lin(
                mx::sigmoid(gate_m) * gate_m * up, mpfx + "down_proj", wm, gs);
        }

        h = mx::fast::rms_norm(h, wm.at("model.norm.weight"), rms_eps);

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
        auto q_raw = lin(normed, apfx + "q_proj", wm, gs);
        auto k_raw = lin(normed, apfx + "k_proj", wm, gs);
        auto v_raw = lin(normed, apfx + "v_proj", wm, gs);

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

        h = h + lin(attn_flat, apfx + "o_proj", wm, gs);

        auto normed2 = mx::fast::rms_norm(h, wm.at(lpfx + "post_attention_layernorm.weight"), rms_eps);
        auto gate_m  = lin(normed2, mpfx + "gate_proj", wm, gs);
        auto up      = lin(normed2, mpfx + "up_proj",   wm, gs);
        h = h + lin(mx::sigmoid(gate_m) * gate_m * up, mpfx + "down_proj", wm, gs);
    }

    h = mx::fast::rms_norm(h, wm.at("model.norm.weight"), rms_eps);

    // Extract last token hidden state: {1, hidden}
    auto h_last = mx::reshape(
        mx::take(h, mx::array(seq_len - 1, mx::int32), 0), {1, (int)hidden});

    auto logits_raw = wm.count("lm_head.weight")
        ? lin(h_last, "lm_head", wm, gs)
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
