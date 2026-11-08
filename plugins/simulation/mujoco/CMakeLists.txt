cmake_minimum_required(VERSION 3.20)
project(mujoco_plugin LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)

# MuJoCo: the pinned official release (cmake/MuJoCoSdk.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/MuJoCoSdk.cmake)
find_package(httplib CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)

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
)

target_include_directories(mujoco_plugin PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# cyxwiz-plugin-sdk is headers only; this plugin logs, draws and reads JSON,
# so it links spdlog, ImGui and nlohmann_json itself
# We additionally need glad (OpenGL loader) and glfw (context management)
target_link_libraries(mujoco_plugin PRIVATE
    cyxwiz-plugin-sdk
    spdlog::spdlog
    imgui::imgui
    nlohmann_json::nlohmann_json
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

# MuJoCo's runtime and its licence files go next to the plugin, as the pack ships them.
add_custom_command(TARGET mujoco_plugin POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CYXWIZ_MUJOCO_RUNTIME}" "${CMAKE_CURRENT_SOURCE_DIR}/bin/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CYXWIZ_MUJOCO_SDK_DIR}/LICENSE" "${CMAKE_CURRENT_SOURCE_DIR}/bin/MUJOCO_LICENSE.txt"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CYXWIZ_MUJOCO_SDK_DIR}/THIRD_PARTY_NOTICES.txt" "${CMAKE_CURRENT_SOURCE_DIR}/bin/MUJOCO_THIRD_PARTY_NOTICES.txt"
    COMMENT "Copying MuJoCo ${CYXWIZ_MUJOCO_VERSION} runtime and licence")

# =============================================================================
# Tests
# =============================================================================

add_executable(test_mujoco_graph_clock tests/test_mujoco_graph_clock.cpp)
target_include_directories(test_mujoco_graph_clock PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_test(NAME mujoco_graph_clock COMMAND test_mujoco_graph_clock)
