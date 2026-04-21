cmake_minimum_required(VERSION 3.16)

project(cli VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable compiler warnings
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    add_compile_options(-Wall -Wextra -Wpedantic)

    # Enable Clazy checks if available
    find_program(CLAZY_EXECUTABLE clazy)
    if(CLAZY_EXECUTABLE)
        set(CMAKE_CXX_COMPILER ${CLAZY_EXECUTABLE})
        add_compile_options(-Xclang -plugin-arg-clazy -Xclang level2)
    endif()
endif()

# Find required packages
find_package(PkgConfig REQUIRED)

# Find CLI11
find_package(CLI11 CONFIG)
if(NOT CLI11_FOUND)
    # Try to find via pkg-config
    pkg_check_modules(CLI11 QUIET cli11)
    if(NOT CLI11_FOUND)
        # Fall back to FetchContent
        include(FetchContent)
        FetchContent_Declare(
            cli11
            GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
            GIT_TAG        v2.4.1
        )
        FetchContent_MakeAvailable(cli11)
    endif()
endif()

# Find indicators library
find_package(indicators CONFIG)
if(NOT indicators_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        indicators
        GIT_REPOSITORY https://github.com/p-ranav/indicators.git
        GIT_TAG        v2.3
    )
    FetchContent_MakeAvailable(indicators)
endif()

# Find tabulate library
find_package(tabulate CONFIG)
if(NOT tabulate_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        tabulate
        GIT_REPOSITORY https://github.com/p-ranav/tabulate.git
        GIT_TAG        v1.5
    )
    FetchContent_MakeAvailable(tabulate)
endif()

# Find nlohmann/json for config file support
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

# CLI application sources
add_executable(cli
    src/main.cpp
    src/cli/commands/download_command.h
    src/cli/commands/download_command.cpp
    src/cli/commands/list_command.h
    src/cli/commands/list_command.cpp
    src/cli/commands/load_command.h
    src/cli/commands/load_command.cpp
    src/cli/commands/search_command.h
    src/cli/commands/search_command.cpp
    src/cli/commands/config_command.h
    src/cli/commands/config_command.cpp
    src/cli/commands/token_command.h
    src/cli/commands/token_command.cpp
    src/cli/commands/node_command.h
    src/cli/commands/node_command.cpp
    src/cli/commands/chat_command.h
    src/cli/commands/chat_command.cpp
    src/cli/commands/server_command.h
    src/cli/commands/server_command.cpp
    src/cli/commands/mcp_command.h
    src/cli/commands/mcp_command.cpp
    src/cli/commands/base_command.h
    src/cli/commands/base_command.cpp
    # MCP client/manager (shared sources from service/)
    ../service/src/mcp/mcp_client.cpp
    ../service/src/mcp/mcp_manager.cpp
    src/cli/core/cli_app.h
    src/cli/core/cli_app.cpp
    src/cli/config/neurons_config.h
    src/cli/config/neurons_config.cpp
    src/cli/utils/progress_bar.h
    src/cli/utils/progress_bar.cpp
    src/cli/utils/table_formatter.h
    src/cli/utils/table_formatter.cpp
    src/cli/utils/format_utils.h
    src/cli/utils/format_utils.cpp
)

target_include_directories(cli
    PRIVATE src/
    PRIVATE ../models/src/
    PRIVATE ../compute/src/
    PRIVATE ../service/src/   # for mcp/mcp_types.h, mcp/mcp_client.h, mcp/mcp_manager.h
)

# Link libraries
if(CLI11_FOUND AND TARGET CLI11::CLI11)
    target_link_libraries(cli PRIVATE CLI11::CLI11)
elseif(CLI11_FOUND)
    target_link_libraries(cli PRIVATE ${CLI11_LIBRARIES})
    target_include_directories(cli PRIVATE ${CLI11_INCLUDE_DIRS})
else()
    target_link_libraries(cli PRIVATE CLI11::CLI11)
endif()

if(indicators_FOUND AND TARGET indicators::indicators)
    target_link_libraries(cli PRIVATE indicators::indicators)
else()
    target_link_libraries(cli PRIVATE indicators)
endif()

if(tabulate_FOUND AND TARGET tabulate::tabulate)
    target_link_libraries(cli PRIVATE tabulate::tabulate)
else()
    target_link_libraries(cli PRIVATE tabulate)
endif()

target_link_libraries(cli PRIVATE model_manager)
target_link_libraries(cli PRIVATE compute_backend)

if(nlohmann_json_FOUND)
    target_link_libraries(cli PRIVATE nlohmann_json::nlohmann_json)
else()
    target_link_libraries(cli PRIVATE nlohmann_json::nlohmann_json)
endif()

# Set executable name
set_target_properties(cli PROPERTIES
    OUTPUT_NAME "neurons"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
