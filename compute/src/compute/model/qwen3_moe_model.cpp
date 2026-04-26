#include "qwen3_moe_model.h"
#include "model_loader.h"
#include "simple_bpe_tokenizer.h"
#include "../core/compute_backend.h"

namespace compute {

// ── Factory ──────────────────────────────────────────────────────────────────

Result<Qwen3MoeModel> Qwen3MoeModel::from_model_dir(
    const std::filesystem::path& model_dir,
    ComputeBackend*              backend)
{
    auto model_result = ModelLoader::load_model(model_dir, backend);
    if (!model_result) return std::unexpected(model_result.error());

    auto& [config, weights] = *model_result;

    auto tokenizer_result = SimpleBpeTokenizer::from_model_dir(model_dir);
    if (!tokenizer_result) return std::unexpected(tokenizer_result.error());

    return Qwen3MoeModel(
        std::move(config),
        std::move(*tokenizer_result),
        std::move(weights),
        backend);
}

Qwen3MoeModel::Qwen3MoeModel(
    ModelConfig                             config,
    SimpleBpeTokenizer                      tokenizer,
    std::unordered_map<std::string, Tensor> weights,
    ComputeBackend*                         backend)
    : config_(std::move(config))
    , tokenizer_(std::move(tokenizer))
    , weights_(std::move(weights))
    , backend_(backend)
{}

// ── LanguageModel interface ───────────────────────────────────────────────────

Result<std::vector<float>> Qwen3MoeModel::prefill(const std::vector<int>& prompt_ids) {
    kv_cache_.assign(config_.num_hidden_layers, LayerKVCache{});
    cache_position_ = 0;
    return forward_impl(prompt_ids, 0, &kv_cache_);
}

Result<std::vector<float>> Qwen3MoeModel::decode(int token_id) {
    auto result = forward_impl({token_id}, static_cast<int>(cache_position_), &kv_cache_);
    if (result) ++cache_position_;
    return result;
}

void Qwen3MoeModel::reset_cache() {
    kv_cache_.clear();
    cache_position_ = 0;
}

size_t Qwen3MoeModel::num_parameters() const {
    size_t total = 0;
    for (const auto& [name, tensor] : weights_) {
        size_t n = 1;
        for (auto d : tensor.shape()) n *= d;
        total += n;
    }
    return total;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

Result<Tensor> Qwen3MoeModel::get_weight(const std::string& name) const {
    auto it = weights_.find(name);
    if (it == weights_.end())
        return std::unexpected(Error{ErrorCode::InvalidModel, "Weight not found: " + name});
    return it->second;
}

Result<Tensor> Qwen3MoeModel::linear(const Tensor& input, const std::string& weight_key) {
    auto w_result = get_weight(weight_key + ".weight");
    if (!w_result) return std::unexpected(w_result.error());

    auto scales_it = weights_.find(weight_key + ".scales");
    if (scales_it != weights_.end()) {
        // Quantized path
        auto biases_it = weights_.find(weight_key + ".biases");
        const Tensor* biases = (biases_it != weights_.end()) ? &biases_it->second : nullptr;
        int gs   = config_.quantization ? config_.quantization->group_size : 64;
        int bits = config_.quantization ? config_.quantization->bits : 4;
        return backend_->quantized_matmul(input, *w_result, scales_it->second, biases,
                                          /*transpose=*/true, gs, bits, "affine");
    }

    // Unquantized path
    auto w_T = backend_->swapaxes(*w_result, 0, 1);
    if (!w_T) return std::unexpected(w_T.error());
    return backend_->matmul(input, *w_T);
}

// ── Layer stubs (F.15.3–F.15.5 will implement these) ─────────────────────────

Result<Tensor> Qwen3MoeModel::moe_mlp(const Tensor& /*input*/, int /*layer_idx*/) {
    return std::unexpected(Error{ErrorCode::NotImplemented,
        "Qwen3MoeModel: MoE MLP not yet implemented (F.15.3)"});
}

Result<Tensor> Qwen3MoeModel::full_attention_block(
    const Tensor& /*input*/, int /*layer_idx*/, int /*position_offset*/, LayerKVCache* /*cache*/)
{
    return std::unexpected(Error{ErrorCode::NotImplemented,
        "Qwen3MoeModel: full attention not yet implemented (F.15.4)"});
}

Result<Tensor> Qwen3MoeModel::linear_attention_block(const Tensor& /*input*/, int /*layer_idx*/) {
    return std::unexpected(Error{ErrorCode::NotImplemented,
        "Qwen3MoeModel: linear (SSM) attention not yet implemented (F.15.5)"});
}

Result<std::vector<float>> Qwen3MoeModel::forward_impl(
    const std::vector<int>& /*input_ids*/,
    int                     /*position_offset*/,
    std::vector<LayerKVCache>* /*cache_vec*/)
{
    return std::unexpected(Error{ErrorCode::NotImplemented,
        "Qwen3MoeModel: forward pass not yet implemented (F.15.3–F.15.5)"});
}

} // namespace compute
