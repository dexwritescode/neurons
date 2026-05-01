#include "qwen3_moe_model_base.h"
#include <cmath>

namespace compute {

Qwen3MoeModelBase::Qwen3MoeModelBase(
    ModelConfig                             config,
    SimpleBpeTokenizer                      tokenizer,
    std::unordered_map<std::string, Tensor> weights)
    : config_(std::move(config))
    , tokenizer_(std::move(tokenizer))
    , weights_(std::move(weights))
{}

size_t Qwen3MoeModelBase::num_parameters() const {
    size_t total = 0;
    for (const auto& [name, tensor] : weights_) total += tensor.size();
    return total;
}

Result<Tensor> Qwen3MoeModelBase::get_weight(const std::string& name) const {
    auto it = weights_.find(name);
    if (it == weights_.end())
        return std::unexpected(Error{ErrorCode::InvalidModel, "Weight not found: " + name});
    return it->second;
}

int Qwen3MoeModelBase::infer_quant_bits(const Tensor& w, const Tensor& scales) const {
    size_t in_packed = w.shape().back();
    size_t groups    = scales.shape().back();
    size_t gs = config_.quantization ? config_.quantization->group_size : 64;
    double ratio = static_cast<double>(in_packed) / static_cast<double>(groups);
    int bits = static_cast<int>(std::round(32.0 * ratio / static_cast<double>(gs)));
    return (bits > 0) ? bits : 4;
}

} // namespace compute
