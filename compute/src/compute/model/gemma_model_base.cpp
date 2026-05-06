#include "gemma_model_base.h"

namespace compute {

GemmaModelBase::GemmaModelBase(
    ModelConfig                             config,
    SimpleBpeTokenizer                      tokenizer,
    std::unordered_map<std::string, Tensor> weights)
    : config_(std::move(config))
    , tokenizer_(std::move(tokenizer))
    , weights_(std::move(weights))
{}

size_t GemmaModelBase::num_parameters() const {
    size_t total = 0;
    for (const auto& [name, tensor] : weights_) total += tensor.size();
    return total;
}

Result<Tensor> GemmaModelBase::get_weight(const std::string& name) const {
    auto it = weights_.find(name);
    if (it == weights_.end())
        return std::unexpected(Error{ErrorCode::TensorNotFound, "Weight not found: " + name});
    return it->second;
}

} // namespace compute
