#pragma once
// TinyLlamaInference has been renamed to LlamaModel (Phase D refactor).
// This header is kept for backwards compatibility so existing test files
// continue to compile without edits.
//
// New code should #include "llama_model.h" and use compute::LlamaModel directly.
// This alias will be removed in Phase D.6.
#include "llama_model.h"

namespace compute {
    using TinyLlamaInference = LlamaModel;
}
