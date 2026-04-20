cmake_minimum_required(VERSION 3.16)

project(model_manager VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable compiler warnings and Clazy
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    
    # Enable Clazy checks if available
    find_program(CLAZY_EXECUTABLE clazy)
    if(CLAZY_EXECUTABLE)
        set(CMAKE_CXX_COMPILER ${CLAZY_EXECUTABLE})
        add_compile_options(-Xclang -plugin-arg-clazy -Xclang level2)
    endif()
endif()

# Find libcurl for HTTP abstraction (Qt-free)
find_package(CURL REQUIRED)

# Find nlohmann/json for JSON parsing
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG        v3.11.3
    )
    FetchContent_MakeAvailable(json)
endif()

add_library(model_manager
    src/models/registry/model_registry.h
    src/models/registry/model_registry.cpp
    src/models/core/model_info.h
    src/models/api/huggingface_client.h
    src/models/api/huggingface_client.cpp
)

target_include_directories(model_manager
    PUBLIC src/
)

target_link_libraries(model_manager
    PUBLIC CURL::libcurl
)

# Link JSON library (handle both system install and FetchContent)
if(nlohmann_json_FOUND)
    target_link_libraries(model_manager PUBLIC nlohmann_json::nlohmann_json)
else()
    target_link_libraries(model_manager PUBLIC nlohmann_json)
endif()

# core library - require C++20 minimum for string methods
#target_compile_features(model_core PUBLIC cxx_std_20)
#if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
#    target_compile_options(model_core PRIVATE -std=c++20)
#endif()

# Testing (only in Debug builds)
if(BUILD_TESTING AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Find or fetch GoogleTest for core tests
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        v1.15.2
        )
        # Prevent installation of GoogleTest
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()

    # Unit tests
    add_executable(model_tests
        tests/models/test_model_registry.cpp
        tests/models/test_huggingface_client.cpp
    )

    # Integration tests
    add_executable(model_integration_tests
        tests/models/test_huggingface_integration.cpp
    )

    # Link with GoogleTest (handle both system and FetchContent cases)
    if(GTest_FOUND)
        target_link_libraries(model_tests PRIVATE GTest::gtest GTest::gtest_main model_manager)
        target_link_libraries(model_integration_tests PRIVATE GTest::gtest GTest::gtest_main model_manager)
    else()
        target_link_libraries(model_tests PRIVATE gtest gtest_main model_manager)
        target_link_libraries(model_integration_tests PRIVATE gtest gtest_main model_manager)
    endif()

    include(CTest)
    add_test(NAME model_tests COMMAND model_tests)
    add_test(NAME model_integration_tests COMMAND model_integration_tests)
endif()
