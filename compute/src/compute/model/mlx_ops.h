#pragma once

// Shared MLX helper utilities used by all MLX-native model implementations.
// Include inside an anonymous namespace or a guarded translation unit.
//
// Required include guard: #if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)

#include <mlx/mlx.h>
#include <optional>
#include <string>
#include <unordered_map>

namespace compute {

using WM = std::unordered_map<std::string, mlx::core::array>;

namespace mlx_ops {

// Infer quantisation bit-width from the weight/scales shape ratio.
inline int bits(const mlx::core::array& w, const mlx::core::array& s, int gs) {
    double ratio = static_cast<double>(w.shape().back()) /
                   static_cast<double>(s.shape().back());
    int b = static_cast<int>(std::round(32.0 * ratio / static_cast<double>(gs)));
    return (b > 0) ? b : 4;
}

// Linear projection: quantized (affine dequant) if scales present, else plain matmul.
inline mlx::core::array lin(const mlx::core::array& x, const std::string& key,
                             const WM& wm, int gs)
{
    namespace mx = mlx::core;
    const mx::array& w = wm.at(key + ".weight");
    auto sit = wm.find(key + ".scales");
    if (sit != wm.end()) {
        int b = bits(w, sit->second, gs);
        std::optional<mx::array> bias;
        auto bit = wm.find(key + ".biases");
        if (bit != wm.end()) bias = bit->second;
        return mx::quantized_matmul(x, w, sit->second, bias, true, gs, b, "affine");
    }
    return mx::matmul(x, mx::swapaxes(w, 0, 1));
}

} // namespace mlx_ops
} // namespace compute
