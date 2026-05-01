#pragma once

#include "model_config.h"
#include "simple_bpe_tokenizer.h"
#include "../core/tensor.h"
#include "../core/compute_types.h"
#include <string>
#include <unordered_map>

namespace compute {

class Qwen3MoeModelBase {
public:
    size_t num_parameters() const;

protected:
    Qwen3MoeModelBase(
        ModelConfig                             config,
        SimpleBpeTokenizer                      tokenizer,
        std::unordered_map<std::string, Tensor> weights);

    Result<Tensor> get_weight(const std::string& name) const;
    int infer_quant_bits(const Tensor& w, const Tensor& scales) const;

    ModelConfig                             config_;
    SimpleBpeTokenizer                      tokenizer_;
    std::unordered_map<std::string, Tensor> weights_;
};

} // namespace compute
