cmake_minimum_required(VERSION 3.16)

# ── Proto codegen ────────────────────────────────────────────────────────────
set(PROTO_FILE "${CMAKE_CURRENT_SOURCE_DIR}/proto/neurons.proto")
set(PROTO_OUT  "${CMAKE_CURRENT_BINARY_DIR}/proto")
file(MAKE_DIRECTORY ${PROTO_OUT})

# Locate the protoc and grpc_cpp_plugin binaries
find_program(PROTOC_BIN        protoc             REQUIRED)
find_program(GRPC_CPP_PLUGIN   grpc_cpp_plugin    REQUIRED)

# Generated C++ source files
set(PROTO_SRCS  "${PROTO_OUT}/neurons.pb.cc")
set(PROTO_HDRS  "${PROTO_OUT}/neurons.pb.h")
set(GRPC_SRCS   "${PROTO_OUT}/neurons.grpc.pb.cc")
set(GRPC_HDRS   "${PROTO_OUT}/neurons.grpc.pb.h")

add_custom_command(
    OUTPUT  ${PROTO_SRCS} ${PROTO_HDRS} ${GRPC_SRCS} ${GRPC_HDRS}
    COMMAND ${PROTOC_BIN}
            --proto_path "${CMAKE_CURRENT_SOURCE_DIR}/proto"
            --cpp_out    "${PROTO_OUT}"
            --grpc_out   "${PROTO_OUT}"
            --plugin=protoc-gen-grpc="${GRPC_CPP_PLUGIN}"
            "${PROTO_FILE}"
    DEPENDS "${PROTO_FILE}"
    COMMENT "Generating protobuf/gRPC C++ stubs from neurons.proto"
)

# ── Dart proto codegen (optional — requires protoc-gen-dart from pub) ─────────
# Install with: dart pub global activate protoc_plugin
# The plugin lands in ~/.pub-cache/bin/protoc-gen-dart
find_program(PROTOC_GEN_DART protoc-gen-dart
    PATHS
        "$ENV{HOME}/.pub-cache/bin"
        "$ENV{HOME}/.dart-tool/pub-global/bin"
    NO_DEFAULT_PATH
)

if(PROTOC_GEN_DART)
    set(DART_OUT "${CMAKE_CURRENT_SOURCE_DIR}/../gui/lib/proto")
    file(MAKE_DIRECTORY "${DART_OUT}")

    set(DART_STUBS
        "${DART_OUT}/neurons.pb.dart"
        "${DART_OUT}/neurons.pbenum.dart"
        "${DART_OUT}/neurons.pbgrpc.dart"
        "${DART_OUT}/neurons.pbjson.dart"
    )

    add_custom_command(
        OUTPUT  ${DART_STUBS}
        COMMAND ${PROTOC_BIN}
                --proto_path "${CMAKE_CURRENT_SOURCE_DIR}/proto"
                "--dart_out=grpc:${DART_OUT}"
                "--plugin=protoc-gen-dart=${PROTOC_GEN_DART}"
                "${PROTO_FILE}"
        DEPENDS "${PROTO_FILE}"
        COMMENT "Generating Dart gRPC stubs from neurons.proto"
    )

    # proto_gen drives both C++ and Dart; any target that depends on it
    # will regenerate both sides in one cmake --build pass when neurons.proto changes.
    add_custom_target(proto_gen ALL DEPENDS
        ${PROTO_SRCS} ${PROTO_HDRS} ${GRPC_SRCS} ${GRPC_HDRS}
        ${DART_STUBS}
    )
    message(STATUS "protoc-gen-dart found — Dart stubs will be regenerated automatically")
else()
    add_custom_target(proto_gen ALL DEPENDS
        ${PROTO_SRCS} ${PROTO_HDRS} ${GRPC_SRCS} ${GRPC_HDRS}
    )
    message(STATUS "protoc-gen-dart not found — Dart stubs will NOT be regenerated")
    message(STATUS "  Install: dart pub global activate protoc_plugin")
endif()

# Proto generated library (shared between service and any future test targets)
add_library(neurons_proto STATIC ${PROTO_SRCS} ${GRPC_SRCS})
add_dependencies(neurons_proto proto_gen)
target_include_directories(neurons_proto PUBLIC ${PROTO_OUT})
target_link_libraries(neurons_proto PUBLIC
    gRPC::grpc++
    protobuf::libprotobuf
)

# ── cpp-httplib (header-only HTTP/HTTPS server) ──────────────────────────────
find_package(httplib QUIET)
if(NOT httplib_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        httplib
        GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
        GIT_TAG        v0.18.5
    )
    set(HTTPLIB_COMPILE OFF)
    FetchContent_MakeAvailable(httplib)
endif()

# ── Service implementation (static lib, shared by executable and dylib) ──────
add_library(neurons_service_lib STATIC
    src/neurons_service.cpp
    src/logger.cpp
    src/http_server.cpp
    src/mcp/mcp_client.cpp
    src/mcp/mcp_manager.cpp
)

target_include_directories(neurons_service_lib PUBLIC
    src
    ${PROTO_OUT}
    ${CMAKE_SOURCE_DIR}/compute/src
    ${CMAKE_SOURCE_DIR}/models/src
)

target_include_directories(neurons_service_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/service/src
)

target_link_libraries(neurons_service_lib PUBLIC
    neurons_proto
    compute_backend
    model_manager
    gRPC::grpc++
    protobuf::libprotobuf
    httplib::httplib
    nlohmann_json::nlohmann_json
)

# ── Service executable ───────────────────────────────────────────────────────
add_executable(neurons_service
    src/main.cpp
)

target_link_libraries(neurons_service PRIVATE
    neurons_service_lib
    gRPC::grpc++_reflection   # enables grpcurl service discovery
)

# Install — COMPONENT Runtime lets callers do:
#   cmake --install <build> --component Runtime --prefix <prefix>
# to install only the service binary without library files.
install(TARGETS neurons_service RUNTIME DESTINATION bin COMPONENT Runtime)

# ── neurons_core shared library (FFI entry point for Flutter GUI) ─────────────
# Platform output names: libneurons_core.dylib / neurons_core.dll / libneurons_core.so
add_library(neurons_core SHARED
    ffi/neurons_ffi.cpp
)

set_target_properties(neurons_core PROPERTIES
    OUTPUT_NAME "neurons_core"
    # Ensure symbols declared extern "C" in the header are exported on all platforms.
    C_VISIBILITY_PRESET   default
    CXX_VISIBILITY_PRESET default
)

target_include_directories(neurons_core PRIVATE
    src
    ${PROTO_OUT}
    ${CMAKE_SOURCE_DIR}/compute/src
    ${CMAKE_SOURCE_DIR}/models/src
)

target_link_libraries(neurons_core PRIVATE
    neurons_service_lib
)

# Install the dylib alongside the service binary so build_service_macos.sh
# can copy both into the Flutter bundle.
install(TARGETS neurons_core
    LIBRARY DESTINATION lib  COMPONENT neurons_core   # .so / .dylib
    RUNTIME DESTINATION bin  COMPONENT neurons_core   # .dll (Windows)
)

# ── Service unit tests ────────────────────────────────────────────────────────

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # gtest/gmock are fetched by compute/CMakeLists.txt (part of the same project)
    include(GoogleTest)

    add_executable(service_tests
        tests/test_mcp_manager.cpp
        tests/test_mcp_rpc.cpp
    )

    target_include_directories(service_tests PRIVATE
        src
        ${PROTO_OUT}
        ${CMAKE_SOURCE_DIR}/compute/src
        ${CMAKE_SOURCE_DIR}/models/src
    )

    target_compile_definitions(service_tests PRIVATE
        MCTS_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures"
    )

    target_link_libraries(service_tests PRIVATE
        neurons_service_lib
        gtest_main
        gmock
    )

    gtest_discover_tests(service_tests
        PROPERTIES LABELS "unit"
    )
endif()
