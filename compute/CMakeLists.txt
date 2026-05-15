cmake_minimum_required(VERSION 3.16)

project(compute_backend VERSION 1.0.0 LANGUAGES CXX)

# ── Rust / Corrosion ──────────────────────────────────────────────────────────
# Corrosion drives cargo build from CMake; tokenizers_sys is the Rust staticlib
# that wraps HuggingFace tokenizers 0.32.1.
include(FetchContent)
FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG        v0.5.1
)
FetchContent_MakeAvailable(Corrosion)
corrosion_import_crate(MANIFEST_PATH ${CMAKE_CURRENT_SOURCE_DIR}/tokenizers-sys/Cargo.toml)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── MLX backend (Apple Silicon only) ─────────────────────────────────────────
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    include(FetchContent)

    set(MLX_VERSION "0.31.2")
    # Using fork until https://github.com/ml-explore/mlx/pull/3485 merges.
    # GIT_TAG is the exact commit on fix/custom-kernel-output-shapes that adds
    # CustomKernel::output_shapes(), required for mx::compile(shapeless=true).
    # TODO: revert to upstream tag once PR is merged and a new release cuts.
    FetchContent_Declare(
        mlx
        GIT_REPOSITORY https://github.com/dexwritescode/mlx.git
        GIT_TAG        1109c8ce5b14801f000fef87cb9cb75b858f8ec4
    )

    set(MLX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(MLX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(mlx)

    if(TARGET mlx)
        message(STATUS "MLX v${MLX_VERSION} fetched successfully - enabling MLX backend")
        list(APPEND BACKEND_SOURCES
            src/compute/backends/mlx/mlx_backend.cpp
            src/compute/model/qwen3_moe_model_mlx.cpp
            src/compute/model/qwen3_transformer_moe_model_mlx.cpp
            src/compute/model/gemma_model_mlx.cpp
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

# ── nlohmann_json ─────────────────────────────────────────────────────────────
# On Apple/arm64 provided transitively by MLX; fetch it on other platforms.
if(NOT NLOHMANN_JSON_TARGET)
    if(TARGET nlohmann_json::nlohmann_json)
        set(NLOHMANN_JSON_TARGET nlohmann_json::nlohmann_json)
    elseif(TARGET nlohmann_json)
        set(NLOHMANN_JSON_TARGET nlohmann_json)
    else()
        FetchContent_Declare(
            nlohmann_json
            GIT_REPOSITORY https://github.com/nlohmann/json.git
            GIT_TAG        v3.11.3
        )
        set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(nlohmann_json)
        set(NLOHMANN_JSON_TARGET nlohmann_json::nlohmann_json)
    endif()
endif()

# ── minja (MIT-licensed Jinja2 subset for chat template rendering) ─────────────
# Use FetchContent_Populate (not MakeAvailable) to avoid running minja's own
# CMakeLists, which executes a Python script to download test data.  We only
# need the header files; we create the INTERFACE target ourselves.
FetchContent_Declare(
    minja
    GIT_REPOSITORY https://github.com/google/minja.git
    GIT_TAG        main
)
FetchContent_GetProperties(minja)
if(NOT minja_POPULATED)
    FetchContent_Populate(minja)
    add_library(minja INTERFACE)
    target_include_directories(minja INTERFACE ${minja_SOURCE_DIR}/include)
    if(NLOHMANN_JSON_TARGET)
        target_link_libraries(minja INTERFACE ${NLOHMANN_JSON_TARGET})
    endif()
endif()

# ── Library ───────────────────────────────────────────────────────────────────
add_library(compute_backend
    src/compute/core/compute_backend.cpp
    src/compute/model/model_config.cpp
    src/compute/model/model_loader.cpp
    src/compute/model/tokenizer_config.cpp
    src/compute/model/hf_tokenizer.cpp
    src/compute/model/sampler.cpp
    src/compute/model/chat_template.cpp
    src/compute/model/language_model.cpp
    src/compute/model/tool_runner.cpp
    src/compute/model/llama_model.cpp
    ${BACKEND_SOURCES}
)

target_include_directories(compute_backend
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        src/compute/
        ${CMAKE_CURRENT_SOURCE_DIR}/tokenizers-sys
)

target_link_libraries(compute_backend PUBLIC tokenizers_sys)

if(APPLE)
    target_link_libraries(compute_backend PUBLIC
        "-framework CoreFoundation"
        "-framework Security"
    )
endif()

if(MLX_AVAILABLE AND TARGET mlx)
    target_link_libraries(compute_backend PUBLIC mlx)
    target_compile_definitions(compute_backend PUBLIC MLX_BACKEND_ENABLED)
    message(STATUS "MLX backend linked successfully")
endif()

if(NLOHMANN_JSON_TARGET)
    target_link_libraries(compute_backend PUBLIC ${NLOHMANN_JSON_TARGET})
endif()

if(TARGET minja)
    target_link_libraries(compute_backend PUBLIC minja)
endif()

# ── Tests ─────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
    set(TINYLLAMA_MODEL_PATH "$ENV{HOME}/.neurons/models/mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit"
        CACHE STRING "Path to TinyLlama model for integration tests")
    set(MISTRAL_MODEL_PATH "$ENV{HOME}/.neurons/models/mlx-community/Mistral-7B-Instruct-v0.3-4bit"
        CACHE STRING "Path to Mistral-7B model for integration tests")
    set(QWEN2_MODEL_PATH "$ENV{HOME}/.neurons/models/mlx-community/Qwen2.5-3B-Instruct-4bit"
        CACHE STRING "Path to Qwen2.5 model for integration tests")
    set(QWEN3MOE_MODEL_PATH "$ENV{HOME}/.neurons/models/mlx-community/Qwen3.6-35B-A3B-4bit"
        CACHE STRING "Path to Qwen3.6-35B MoE model for integration tests")
    set(QWEN3_30B_MODEL_PATH "$ENV{HOME}/.neurons/models/mlx-community/Qwen3-30B-A3B-4bit"
        CACHE STRING "Path to Qwen3-30B-A3B model for integration tests")

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

    add_executable(compute_tests
        tests/compute/test_model_config.cpp
        tests/compute/test_model_loader.cpp
        tests/compute/test_tokenizer_config.cpp
        tests/compute/test_forward_pass.cpp
        tests/compute/test_mistral_integration.cpp
        tests/compute/test_llama3_integration.cpp
        tests/compute/test_qwen2_integration.cpp
        tests/compute/test_fp16bf16_integration.cpp
        tests/compute/test_gemma_integration.cpp
        tests/compute/test_qwen3_moe_integration.cpp
        tests/compute/test_qwen3_transformer_moe_integration.cpp
        tests/compute/test_chat_template.cpp
        tests/compute/test_tool_runner.cpp
        tests/compute/test_sampler.cpp
        tests/compute/test_hf_tokenizer.cpp
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

    include(CTest)
    include(GoogleTest)
    gtest_discover_tests(compute_tests)
endif()
