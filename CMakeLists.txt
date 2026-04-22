cmake_minimum_required(VERSION 3.16)

project(NeuronsWorkspace VERSION 0.1.0 LANGUAGES CXX)

# Global settings
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable testing by default
option(BUILD_TESTING "Build tests" ON)
if(BUILD_TESTING)
    enable_testing()
endif()

# ── gRPC + Protobuf ─────────────────────────────────────────────────────────
# macOS:   brew install grpc
# Linux:   apt install libgrpc++-dev protobuf-compiler-grpc
# Windows: vcpkg install grpc  (pass -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake)
if(APPLE)
    list(APPEND CMAKE_PREFIX_PATH /opt/homebrew)  # Apple Silicon
    list(APPEND CMAKE_PREFIX_PATH /usr/local)     # Intel Mac
endif()
find_package(gRPC CONFIG QUIET)
find_package(Protobuf CONFIG QUIET)
if(gRPC_FOUND AND Protobuf_FOUND)
    message(STATUS "gRPC ${gRPC_VERSION} found — building neurons_service")
else()
    message(STATUS "gRPC not found — skipping neurons_service")
    message(STATUS "  macOS:   brew install grpc")
    message(STATUS "  Linux:   apt install libgrpc++-dev protobuf-compiler-grpc")
    message(STATUS "  Windows: vcpkg install grpc")
endif()

# Add subdirectories in dependency order
add_subdirectory(compute)
add_subdirectory(models)
add_subdirectory(cli)

# gRPC inference service — optional
if(gRPC_FOUND AND Protobuf_FOUND)
    add_subdirectory(service)
endif()

# Convenience targets for selective building
add_custom_target(compute-only DEPENDS compute_backend)
add_custom_target(models-only DEPENDS model_manager)
add_custom_target(cli-only DEPENDS cli)
if(gRPC_FOUND AND Protobuf_FOUND)
    add_custom_target(service-only DEPENDS neurons_service)
endif()
if(gRPC_FOUND AND Protobuf_FOUND)
    add_custom_target(all-neurons DEPENDS compute_backend model_manager cli neurons_service)
else()
    add_custom_target(all-neurons DEPENDS compute_backend model_manager cli)
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_custom_target(all-tests DEPENDS compute_tests model_tests)
else()
    add_custom_target(all-tests DEPENDS compute_tests)
endif()
