#include "tensor.h"

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
#include "../backends/mlx/mlx_buffer.h"

namespace compute {

mx::array& Tensor::to_mlx() {
    return static_cast<MLXBuffer*>(buffer_.get())->mlx_array();
}

const mx::array& Tensor::to_mlx() const {
    return static_cast<const MLXBuffer*>(buffer_.get())->mlx_array();
}

} // namespace compute

#endif // defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)