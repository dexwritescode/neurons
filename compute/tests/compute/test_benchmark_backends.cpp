#include <benchmark/benchmark.h>
#include <vector>
#include <random>

#include "compute/core/compute_types.h"
#include "compute/core/compute_backend.h"

using namespace compute;

static std::vector<float>& vec_a() { static std::vector<float> v; return v; }
static std::vector<float>& vec_b() { static std::vector<float> v; return v; }
static bool& data_initialized() { static bool f = false; return f; }

// Initialize test data once
void InitializeBenchmarkData(size_t size) {
    if (data_initialized() && vec_a().size() == size) return;
    
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducible results
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    vec_a().resize(size);
    vec_b().resize(size);

    for (size_t i = 0; i < size; ++i) {
        vec_a()[i] = dis(gen);
        vec_b()[i] = dis(gen);
    }
    
    data_initialized() = true;
}

// Reference CPU implementations
float reference_dot_product(const std::vector<float>& a, const std::vector<float>& b) {
    float result = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

void reference_matrix_scalar_add(const std::vector<float>& input, float scalar, std::vector<float>& output) {
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i] + scalar;
    }
}

// === FAIR BENCHMARK FIXTURE ===

// Fixture for fair backend comparisons with GPU warmup
class BackendBenchmark : public benchmark::Fixture {
protected:
    std::unique_ptr<ComputeBackend> neon_backend;
    std::unique_ptr<ComputeBackend> metal_backend;
    std::unique_ptr<ComputeBackend> mlx_backend;
    bool metal_warmed_up = false;
    bool mlx_warmed_up = false;
    
public:
    void SetUp(const benchmark::State& state) override {
        const size_t vector_size = state.range(0);
        InitializeBenchmarkData(vector_size);
        
        // Pre-initialize all available backends (setup cost excluded from timing)
        auto neon_result = BackendFactory::create(BackendType::SimdNeon);
        if (neon_result && (*neon_result)->initialize()) {
            neon_backend = std::move(*neon_result);
        }
        
        auto metal_result = BackendFactory::create(BackendType::Metal);
        if (metal_result && (*metal_result)->initialize()) {
            metal_backend = std::move(*metal_result);
            // GPU warmup for fair measurement
            WarmupMetal();
        }
        
        auto mlx_result = BackendFactory::create(BackendType::MLX);
        if (mlx_result && (*mlx_result)->initialize()) {
            mlx_backend = std::move(*mlx_result);
            // MLX warmup for fair measurement
            WarmupMLX();
        }
    }
    
private:
    void WarmupMetal() {
        if (!metal_backend || metal_warmed_up) return;
        
        // Warmup with a few small operations to initialize GPU kernels
        std::vector<float> warmup_a = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> warmup_b = {5.0f, 6.0f, 7.0f, 8.0f};
        std::vector<float> warmup_out(4);
        
        // Create tensors using new API
        auto tensor_a = metal_backend->create_tensor(std::span<const float>(warmup_a), {4});
        auto tensor_b = metal_backend->create_tensor(std::span<const float>(warmup_b), {4});
        
        // Warmup dot product
        std::vector<float> result_data(1);
        for (int i = 0; i < 3; ++i) {
            auto result_tensor = metal_backend->dot_product(tensor_a, tensor_b);
            metal_backend->extract(result_tensor, std::span<float>(result_data));
        }
        
        // Warmup matrix operations
        auto matrix_in = metal_backend->create_tensor(std::span<const float>(warmup_a), {2, 2});
        for (int i = 0; i < 3; ++i) {
            auto result_tensor = metal_backend->matrix_scalar_add(matrix_in, 1.0f);
            metal_backend->extract(result_tensor, std::span<float>(warmup_out));
        }
        
        metal_warmed_up = true;
    }
    
    void WarmupMLX() {
        if (!mlx_backend || mlx_warmed_up) return;
        
        // Warmup with a few small operations to initialize MLX kernels
        std::vector<float> warmup_a = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> warmup_b = {5.0f, 6.0f, 7.0f, 8.0f};
        std::vector<float> warmup_out(4);
        
        // Create tensors using new API
        auto tensor_a = mlx_backend->create_tensor(std::span<const float>(warmup_a), {4});
        auto tensor_b = mlx_backend->create_tensor(std::span<const float>(warmup_b), {4});
        
        // Warmup dot product
        std::vector<float> result_data(1);
        for (int i = 0; i < 3; ++i) {
            auto result_tensor = mlx_backend->dot_product(tensor_a, tensor_b);
            mlx_backend->extract(result_tensor, std::span<float>(result_data));
        }
        
        // Warmup matrix operations
        auto matrix_in = mlx_backend->create_tensor(std::span<const float>(warmup_a), {2, 2});
        for (int i = 0; i < 3; ++i) {
            auto result_tensor = mlx_backend->matrix_scalar_add(matrix_in, 1.0f);
            mlx_backend->extract(result_tensor, std::span<float>(warmup_out));
        }
        
        mlx_warmed_up = true;
    }
};

// === DOT PRODUCT BENCHMARKS ===

// Raw CPU implementation (no backend overhead)
BENCHMARK_DEFINE_F(BackendBenchmark, CPU_DotProduct)(benchmark::State& state) {
    const size_t vector_size = state.range(0);
    
    for (auto _ : state) {
        float result = reference_dot_product(vec_a(), vec_b());
        benchmark::DoNotOptimize(result);
    }
    
    state.SetLabel("CPU");
}

// NEON backend (fair comparison - same data structures)
BENCHMARK_DEFINE_F(BackendBenchmark, NEON_DotProduct)(benchmark::State& state) {
    const size_t vector_size = state.range(0);
    
    if (!neon_backend) {
        state.SkipWithError("NEON backend not available on this platform");
        return;
    }
    
    // Create tensors using new API
    auto tensor_a = neon_backend->create_tensor(std::span<const float>(vec_a().data(), vector_size), {vector_size});
    auto tensor_b = neon_backend->create_tensor(std::span<const float>(vec_b().data(), vector_size), {vector_size});
    
    for (auto _ : state) {
        auto result_tensor = neon_backend->dot_product(tensor_a, tensor_b);
        
        std::vector<float> result_data(1);
        auto extract_result = neon_backend->extract(result_tensor, std::span<float>(result_data));
        if (!extract_result) {
            state.SkipWithError("NEON dot product operation failed");
            return;
        }
        benchmark::DoNotOptimize(result_data[0]);
    }
    
    state.SetLabel("NEON");
}

// MLX backend (pre-warmed for fair comparison)
BENCHMARK_DEFINE_F(BackendBenchmark, MLX_DotProduct)(benchmark::State& state) {
    const size_t vector_size = state.range(0);
    
    if (!mlx_backend) {
        state.SkipWithError("MLX backend not available on this platform");
        return;
    }
    
    // Create tensors using new API
    auto tensor_a = mlx_backend->create_tensor(std::span<const float>(vec_a().data(), vector_size), {vector_size});
    auto tensor_b = mlx_backend->create_tensor(std::span<const float>(vec_b().data(), vector_size), {vector_size});
    
    for (auto _ : state) {
        auto result_tensor = mlx_backend->dot_product(tensor_a, tensor_b);
        
        std::vector<float> result_data(1);
        auto extract_result = mlx_backend->extract(result_tensor, std::span<float>(result_data));
        if (!extract_result) {
            state.SkipWithError("MLX dot product operation failed");
            return;
        }
        benchmark::DoNotOptimize(result_data[0]);
    }
    
    state.SetLabel("MLX");
}

// === MATRIX SCALAR ADDITION BENCHMARKS ===

BENCHMARK_DEFINE_F(BackendBenchmark, CPU_MatrixScalarAdd)(benchmark::State& state) {
    const size_t matrix_size = state.range(0);
    std::vector<float> output(matrix_size);
    
    for (auto _ : state) {
        reference_matrix_scalar_add(vec_a(), 10.0f, output);
        benchmark::DoNotOptimize(output.data());
    }
    
    state.SetLabel("CPU");
}

BENCHMARK_DEFINE_F(BackendBenchmark, NEON_MatrixScalarAdd)(benchmark::State& state) {
    const size_t matrix_size = state.range(0);
    
    if (!neon_backend) {
        state.SkipWithError("NEON backend not available");
        return;
    }
    
    std::vector<float> output(matrix_size);
    size_t rows = static_cast<size_t>(std::sqrt(matrix_size));
    size_t cols = matrix_size / rows;
    
    // Create tensor using new API
    auto input_tensor = neon_backend->create_tensor(std::span<const float>(vec_a().data(), matrix_size), {rows, cols});
    
    for (auto _ : state) {
        auto result_tensor = neon_backend->matrix_scalar_add(input_tensor, 10.0f);
        
        auto extract_result = neon_backend->extract(result_tensor, std::span<float>(output));
        if (!extract_result) {
            state.SkipWithError("NEON matrix scalar add failed");
            return;
        }
        benchmark::DoNotOptimize(output.data());
    }
    
    state.SetLabel("NEON");
}

BENCHMARK_DEFINE_F(BackendBenchmark, MLX_MatrixScalarAdd)(benchmark::State& state) {
    const size_t matrix_size = state.range(0);
    
    if (!mlx_backend) {
        state.SkipWithError("MLX backend not available");
        return;
    }
    
    std::vector<float> output(matrix_size);
    size_t rows = static_cast<size_t>(std::sqrt(matrix_size));
    size_t cols = matrix_size / rows;
    
    // Create tensor using new API
    auto input_tensor = mlx_backend->create_tensor(std::span<const float>(vec_a().data(), matrix_size), {rows, cols});
    
    for (auto _ : state) {
        auto result_tensor = mlx_backend->matrix_scalar_add(input_tensor, 10.0f);
        
        auto extract_result = mlx_backend->extract(result_tensor, std::span<float>(output));
        if (!extract_result) {
            state.SkipWithError("MLX matrix scalar add failed");
            return;
        }
        benchmark::DoNotOptimize(output.data());
    }
    
    state.SetLabel("MLX");
}

// === BENCHMARK REGISTRATION ===

// === DOT PRODUCT BENCHMARKS ===
BENCHMARK_REGISTER_F(BackendBenchmark, CPU_DotProduct)
    ->Args({10000})->Args({100000})->Args({1000000});

BENCHMARK_REGISTER_F(BackendBenchmark, NEON_DotProduct)
    ->Args({100000})->Args({1000000})->Args({10000000});

BENCHMARK_REGISTER_F(BackendBenchmark, MLX_DotProduct)
    ->Args({500000})->Args({1000000})->Args({10000000});

// === MATRIX SCALAR ADD BENCHMARKS ===
BENCHMARK_REGISTER_F(BackendBenchmark, CPU_MatrixScalarAdd)
    ->Args({16384})->Args({65536});

BENCHMARK_REGISTER_F(BackendBenchmark, NEON_MatrixScalarAdd)
    ->Args({16384})->Args({65536});

BENCHMARK_REGISTER_F(BackendBenchmark, MLX_MatrixScalarAdd)
    ->Args({262144})->Args({1048576});

BENCHMARK_MAIN();
