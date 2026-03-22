cmake_minimum_required(VERSION 3.20)
project(mujoco_plugin LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)

# =============================================================================
# MuJoCo Dependency — Pre-built binary
# =============================================================================
#
# Using pre-built MuJoCo avoids FetchContent conflicts with vcpkg dependencies
# (spdlog, abseil, etc.). MuJoCo's build tree pulls sdflib which brings its
# own spdlog, conflicting with our vcpkg spdlog.
#
# Download from: https://github.com/google-deepmind/mujoco/releases
# Set MUJOCO_DIR to the extracted directory.

# Try find_package first (works if MuJoCo is installed or MUJOCO_DIR is set)
find_package(mujoco QUIET)
find_package(httplib CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)

if(NOT mujoco_FOUND)
    # Look in common locations
    set(MUJOCO_SEARCH_PATHS
        "$ENV{MUJOCO_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/mujoco"
        "C:/mujoco"
        "D:/mujoco"
        "$ENV{USERPROFILE}/mujoco"
        "$ENV{HOME}/mujoco"
        "/usr/local"
    )

    find_path(MUJOCO_INCLUDE_DIR
        NAMES mujoco/mujoco.h
        PATHS ${MUJOCO_SEARCH_PATHS}
        PATH_SUFFIXES include
    )

    find_library(MUJOCO_LIBRARY
        NAMES mujoco
        PATHS ${MUJOCO_SEARCH_PATHS}
        PATH_SUFFIXES lib bin
    )

    if(MUJOCO_INCLUDE_DIR AND MUJOCO_LIBRARY)
        message(STATUS "MuJoCo found: ${MUJOCO_LIBRARY}")
        message(STATUS "MuJoCo include: ${MUJOCO_INCLUDE_DIR}")

        add_library(mujoco::mujoco SHARED IMPORTED)
        set_target_properties(mujoco::mujoco PROPERTIES
            IMPORTED_LOCATION "${MUJOCO_LIBRARY}"
            IMPORTED_IMPLIB "${MUJOCO_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_INCLUDE_DIR}"
        )

        # Copy MuJoCo DLL to output directory
        get_filename_component(MUJOCO_LIB_DIR "${MUJOCO_LIBRARY}" DIRECTORY)
        file(GLOB MUJOCO_DLLS "${MUJOCO_LIB_DIR}/../bin/mujoco*.dll" "${MUJOCO_LIB_DIR}/mujoco*.dll")

        set(mujoco_FOUND TRUE)
    else()
        message(FATAL_ERROR
            "MuJoCo not found!\n"
            "Download from: https://github.com/google-deepmind/mujoco/releases\n"
            "Then set -DMUJOCO_DIR=<path-to-mujoco> or set MUJOCO_DIR environment variable.\n"
            "Expected structure:\n"
            "  <MUJOCO_DIR>/include/mujoco/mujoco.h\n"
            "  <MUJOCO_DIR>/lib/mujoco.lib (Windows)\n"
        )
    endif()
endif()

# =============================================================================
# Plugin DLL
# =============================================================================

add_library(mujoco_plugin SHARED
    src/mujoco_plugin.cpp
    src/mj_env_manager.cpp
    src/mj_env_library.cpp
    src/mj_env_browser_panel.cpp
    src/mj_renderer.cpp
    src/mj_viewport_panel.cpp
    src/mj_mjcf_parser.cpp
    src/mj_simulation_executor.cpp
    src/mj_menagerie_downloader.cpp
    src/rl/reward_shaper.cpp
    src/rl/observation_filter.cpp
    src/rl/ppo_agent.cpp
    src/export/onnx_exporter.cpp
)

target_include_directories(mujoco_plugin PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# cyxwiz-plugin-sdk provides spdlog, nlohmann_json, imgui
# We additionally need glad (OpenGL loader) and glfw (context management)
target_link_libraries(mujoco_plugin PRIVATE
    cyxwiz-plugin-sdk
    mujoco::mujoco
    glad::glad
    glfw
    httplib::httplib
    OpenSSL::SSL
    OpenSSL::Crypto
)

# =============================================================================
# Output to bin/ subdirectory (for plugin.json "library" path)
# =============================================================================

set_target_properties(mujoco_plugin PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/bin"
)

# Copy MuJoCo DLL next to plugin (Windows)
if(WIN32 AND MUJOCO_DLLS)
    foreach(DLL ${MUJOCO_DLLS})
        add_custom_command(TARGET mujoco_plugin POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DLL}" "${CMAKE_CURRENT_SOURCE_DIR}/bin/"
            COMMENT "Copying MuJoCo DLL: ${DLL}"
        )
    endforeach()
endif()
