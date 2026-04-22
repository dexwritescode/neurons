cmake_minimum_required(VERSION 3.16)

project(compute_backend VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Platform-specific backend sources
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    # MLX backend for Apple Silicon
    include(FetchContent)

    # Fetch MLX from Apple's repository - using stable release
    set(MLX_VERSION "0.31.1")
    FetchContent_Declare(
        mlx
        GIT_REPOSITORY https://github.com/ml-explore/mlx.git
        GIT_TAG        v${MLX_VERSION}
    )

    # Configure MLX build options
    set(MLX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(MLX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)  # We only want C++

    FetchContent_MakeAvailable(mlx)

    if(TARGET mlx)
        message(STATUS "MLX v${MLX_VERSION} fetched successfully - enabling MLX backend")
        list(APPEND BACKEND_SOURCES
            src/compute/backends/mlx/mlx_backend.h
            src/compute/backends/mlx/mlx_backend.cpp
        )
        set(MLX_AVAILABLE TRUE)
    else()
        message(WARNING "MLX fetch failed - MLX backend disabled")
        set(MLX_AVAILABLE FALSE)
    endif()
endif()

# Compute backend library
add_library(compute_backend
    src/compute/core/compute_backend.cpp
    src/compute/core/compute_backend.h
    src/compute/core/compute_types.h
    src/compute/core/tensor.h
    src/compute/core/tensor.cpp
    src/compute/core/graph.h
    src/compute/core/graph.cpp
    src/compute/backends/simd/simd_utils.cpp
    src/compute/backends/simd/simd_utils.h
    src/compute/backends/simd/neon_backend.h
    src/compute/backends/simd/neon_backend.cpp
    src/compute/model/model_config.h
    src/compute/model/model_config.cpp
    src/compute/model/model_loader.h
    src/compute/model/model_loader.cpp
    src/compute/model/tokenizer_config.h
    src/compute/model/tokenizer_config.cpp
    src/compute/model/simple_bpe_tokenizer.h
    src/compute/model/simple_bpe_tokenizer.cpp
    src/compute/model/sampler.h
    src/compute/model/sampler.cpp
    src/compute/model/language_model.h
    src/compute/model/language_model.cpp
    src/compute/model/llama_model.h
    src/compute/model/llama_model.cpp
    src/compute/model/gemma_model.h
    src/compute/model/gemma_model.cpp
    src/compute/model/tinyllama_inference.h          # Backwards-compat alias → LlamaModel
    ${BACKEND_SOURCES}                               # Platform-specific sources
)

target_include_directories(compute_backend 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
    PRIVATE src/compute/
)

# Add Metal framework on Apple platforms
if(APPLE)
    # Link MLX library if available
    if(MLX_AVAILABLE AND TARGET mlx)
        target_link_libraries(compute_backend PUBLIC mlx)
        target_compile_definitions(compute_backend PUBLIC MLX_BACKEND_ENABLED)
        message(STATUS "MLX backend linked successfully")
    endif()

    # Link nlohmann/json — PUBLIC because simple_bpe_tokenizer.h is a public header
    if(TARGET nlohmann_json::nlohmann_json)
        target_link_libraries(compute_backend PUBLIC nlohmann_json::nlohmann_json)
    elseif(TARGET nlohmann_json)
        target_link_libraries(compute_backend PUBLIC nlohmann_json)
    endif()
endif()

# Testing
if(BUILD_TESTING)
    # Set test configuration paths
    # TODO What can we do about this? really tiny model maybe??
    set(TINYLLAMA_MODEL_PATH "/Users/dex/.neurons/models/mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit"
        CACHE STRING "Path to TinyLlama model for integration tests")
    set(MISTRAL_MODEL_PATH "/Users/dex/.neurons/models/mlx-community/Mistral-7B-Instruct-v0.3-4bit"
        CACHE STRING "Path to Mistral-7B model for integration tests")
    set(QWEN2_MODEL_PATH "/Users/dex/.neurons/models/mlx-community/Qwen2.5-3B-Instruct-4bit"
        CACHE STRING "Path to Qwen2.5 model for integration tests")

    # Configure test config header
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/compute/test_config.h.in"
        "${CMAKE_CURRENT_BINARY_DIR}/tests/compute/test_config.h"
        @ONLY
    )

    include(FetchContent)
    
    # GoogleTest for unit tests
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.17.0
    )
    
    # Google Benchmark for performance tests
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.9.4
    )
    
    # Configure benchmark to not require Google Test
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    
    # Configure GoogleTest to be more permissive
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(googletest benchmark)
    
    # Platform-specific test sources
    if(APPLE AND MLX_AVAILABLE AND TARGET mlx)
        # Add MLX test if MLX backend is available
        list(APPEND TEST_BACKEND_SOURCES
            tests/compute/test_mlx_backend.cpp        # MLX tests only on Apple Silicon with MLX
            src/compute/backends/mlx/mlx_backend.cpp
        )
    else()
        set(TEST_BACKEND_SOURCES
            src/compute/core/compute_backend.cpp          # Pure C++ version
        )
    endif()
    
    # Unit tests executable (GoogleTest)
    add_executable(compute_tests
        tests/compute/test_simd.cpp
        tests/compute/test_symbolic_api.cpp
        tests/compute/test_neon_execution.cpp
        tests/compute/test_model_config.cpp
        tests/compute/test_model_loader.cpp
        tests/compute/test_tokenizer_config.cpp
        tests/compute/test_simple_bpe_tokenizer.cpp
        tests/compute/test_attention_qkv_trace.cpp
        tests/compute/test_forward_pass.cpp
        tests/compute/test_mistral_integration.cpp
        tests/compute/test_llama3_integration.cpp
        tests/compute/test_qwen2_integration.cpp
        tests/compute/test_fp16bf16_integration.cpp
        tests/compute/test_gemma_integration.cpp
        src/compute/backends/simd/simd_utils.cpp
        src/compute/backends/simd/neon_backend.cpp
        src/compute/core/graph.cpp
        src/compute/core/tensor.cpp
        src/compute/model/model_config.cpp
        src/compute/model/model_loader.cpp
        src/compute/model/tokenizer_config.cpp
        src/compute/model/simple_bpe_tokenizer.cpp
        src/compute/model/sampler.cpp
        src/compute/model/language_model.cpp
        src/compute/model/llama_model.cpp
        src/compute/model/gemma_model.cpp
        ${TEST_BACKEND_SOURCES}                          # Platform-specific sources
    )

    # Tokenizer comparison test (separate executable for easier debugging)
    add_executable(test_tokenizer_comparison
        tests/compute/test_tokenizer_comparison.cpp
        src/compute/model/tokenizer_config.cpp
        src/compute/model/simple_bpe_tokenizer.cpp
    )

    target_link_libraries(test_tokenizer_comparison PRIVATE
        gtest_main
        gmock
    )

    # Link nlohmann/json for tokenizer comparison test
    if(TARGET nlohmann_json::nlohmann_json)
        target_link_libraries(test_tokenizer_comparison PRIVATE nlohmann_json::nlohmann_json)
    elseif(TARGET nlohmann_json)
        target_link_libraries(test_tokenizer_comparison PRIVATE nlohmann_json)
    endif()

    target_include_directories(test_tokenizer_comparison PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    target_link_libraries(compute_tests PRIVATE
        gtest_main
        gmock
        compute_backend
    )

    # Link nlohmann/json for tests
    if(TARGET nlohmann_json::nlohmann_json)
        target_link_libraries(compute_tests PRIVATE nlohmann_json::nlohmann_json)
    elseif(TARGET nlohmann_json)
        target_link_libraries(compute_tests PRIVATE nlohmann_json)
    endif()
    target_include_directories(compute_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_BINARY_DIR}/tests/compute  # For generated test_config.h
    )
    
    # Benchmark executable (Google Benchmark)
    add_executable(compute_benchmarks
        tests/compute/test_benchmark_backends.cpp
        src/compute/backends/simd/simd_utils.cpp
        src/compute/backends/simd/neon_backend.cpp
        src/compute/core/graph.cpp
        src/compute/core/tensor.cpp
    )
    
    # Add MLX backend to benchmarks if available
    if(MLX_AVAILABLE AND TARGET mlx)
        target_sources(compute_benchmarks PRIVATE
            src/compute/backends/mlx/mlx_backend.cpp
        )
    endif()
    
    target_link_libraries(compute_benchmarks PRIVATE benchmark::benchmark compute_backend)
    target_include_directories(compute_benchmarks PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    
    # Add Metal framework for tests and benchmarks on Apple platforms
    if(APPLE)
        # Add MLX library for tests if available
        if(MLX_AVAILABLE AND TARGET mlx)
            target_link_libraries(compute_tests PRIVATE mlx)
            target_compile_definitions(compute_tests PRIVATE MLX_BACKEND_ENABLED)
            target_link_libraries(compute_benchmarks PRIVATE mlx)
            target_compile_definitions(compute_benchmarks PRIVATE MLX_BACKEND_ENABLED)
        endif()
    endif()
    
    include(CTest)
    include(GoogleTest)
    gtest_discover_tests(compute_tests)
endif()
