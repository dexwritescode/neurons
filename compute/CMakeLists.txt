cmake_minimum_required(VERSION 3.16)

project(compute_backend VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── MLX backend (Apple Silicon only) ─────────────────────────────────────────
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    include(FetchContent)

    set(MLX_VERSION "0.31.2")
    FetchContent_Declare(
        mlx
        GIT_REPOSITORY https://github.com/ml-explore/mlx.git
        GIT_TAG        v${MLX_VERSION}
    )

    set(MLX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(MLX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(mlx)

    if(TARGET mlx)
        message(STATUS "MLX v${MLX_VERSION} fetched successfully - enabling MLX backend")
        list(APPEND BACKEND_SOURCES
            src/compute/backends/mlx/mlx_backend.cpp
        )
        set(MLX_AVAILABLE TRUE)

        # Resolve nlohmann_json (brought in transitively by MLX) once for reuse below
        if(TARGET nlohmann_json::nlohmann_json)
            set(NLOHMANN_JSON_TARGET nlohmann_json::nlohmann_json)
        elseif(TARGET nlohmann_json)
            set(NLOHMANN_JSON_TARGET nlohmann_json)
        endif()
    else()
        message(WARNING "MLX fetch failed - MLX backend disabled")
        set(MLX_AVAILABLE FALSE)
    endif()
endif()

# ── Library ───────────────────────────────────────────────────────────────────
add_library(compute_backend
    src/compute/core/compute_backend.cpp
    src/compute/core/tensor.cpp
    src/compute/core/graph.cpp
    src/compute/model/model_config.cpp
    src/compute/model/model_loader.cpp
    src/compute/model/tokenizer_config.cpp
    src/compute/model/simple_bpe_tokenizer.cpp
    src/compute/model/sampler.cpp
    src/compute/model/chat_template.cpp
    src/compute/model/language_model.cpp
    src/compute/model/llama_model.cpp
    src/compute/model/gemma_model.cpp
    src/compute/model/qwen3_moe_model.cpp
    ${BACKEND_SOURCES}
)

target_include_directories(compute_backend
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
    PRIVATE src/compute/
)

if(MLX_AVAILABLE AND TARGET mlx)
    target_link_libraries(compute_backend PUBLIC mlx)
    target_compile_definitions(compute_backend PUBLIC MLX_BACKEND_ENABLED)
    message(STATUS "MLX backend linked successfully")
endif()

if(NLOHMANN_JSON_TARGET)
    target_link_libraries(compute_backend PUBLIC ${NLOHMANN_JSON_TARGET})
endif()

# ── Tests ─────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
    set(TINYLLAMA_MODEL_PATH "$ENV{HOME}/.neurons/models/mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit"
        CACHE STRING "Path to TinyLlama model for integration tests")
    set(MISTRAL_MODEL_PATH "$ENV{HOME}/.neurons/models/mlx-community/Mistral-7B-Instruct-v0.3-4bit"
        CACHE STRING "Path to Mistral-7B model for integration tests")
    set(QWEN2_MODEL_PATH "$ENV{HOME}/.neurons/models/mlx-community/Qwen2.5-3B-Instruct-4bit"
        CACHE STRING "Path to Qwen2.5 model for integration tests")
    set(QWEN3MOE_MODEL_PATH "$ENV{HOME}/.neurons/models/mlx-community/Qwen3-30B-A3B-4bit"
        CACHE STRING "Path to Qwen3 MoE model for integration tests")

    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/compute/test_config.h.in"
        "${CMAKE_CURRENT_BINARY_DIR}/tests/compute/test_config.h"
        @ONLY
    )

    include(FetchContent)

    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.17.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    # Platform-specific test-only sources (backend/model sources come via compute_backend)
    set(TEST_PLATFORM_SOURCES "")
    if(MLX_AVAILABLE AND TARGET mlx)
        list(APPEND TEST_PLATFORM_SOURCES
            tests/compute/test_mlx_backend.cpp
        )
    endif()

    add_executable(compute_tests
        tests/compute/test_symbolic_api.cpp
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
        tests/compute/test_qwen3_moe_integration.cpp
        tests/compute/test_chat_template.cpp
        ${TEST_PLATFORM_SOURCES}
    )

    target_link_libraries(compute_tests PRIVATE
        gtest_main
        gmock
        compute_backend
    )

    if(NLOHMANN_JSON_TARGET)
        target_link_libraries(compute_tests PRIVATE ${NLOHMANN_JSON_TARGET})
    endif()

    target_include_directories(compute_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_BINARY_DIR}/tests/compute
    )

    if(MLX_AVAILABLE AND TARGET mlx)
        target_link_libraries(compute_tests PRIVATE mlx)
        target_compile_definitions(compute_tests PRIVATE MLX_BACKEND_ENABLED)
    endif()

    # Tokenizer comparison test — minimal standalone executable, does not link compute_backend
    add_executable(test_tokenizer_comparison
        tests/compute/test_tokenizer_comparison.cpp
        src/compute/model/tokenizer_config.cpp
        src/compute/model/simple_bpe_tokenizer.cpp
    )

    target_link_libraries(test_tokenizer_comparison PRIVATE
        gtest_main
        gmock
    )

    if(NLOHMANN_JSON_TARGET)
        target_link_libraries(test_tokenizer_comparison PRIVATE ${NLOHMANN_JSON_TARGET})
    endif()

    target_include_directories(test_tokenizer_comparison PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    include(CTest)
    include(GoogleTest)
    gtest_discover_tests(compute_tests)
endif()
