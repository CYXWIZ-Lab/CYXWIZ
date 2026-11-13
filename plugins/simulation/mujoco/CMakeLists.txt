cmake_minimum_required(VERSION 3.20)
project(mujoco_plugin LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)

# MuJoCo: the pinned official release (cmake/MuJoCoSdk.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/MuJoCoSdk.cmake)
find_package(httplib CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)
# PNG writing and frame scaling for the Viewport's screenshots and recordings
find_package(Stb REQUIRED)

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
    src/mj_viewport_menus.cpp
    src/mj_viewport_stress.cpp
    src/mj_viewport_inspect.cpp
    src/mj_viewport_timeline.cpp
    src/mj_viewport_settings.cpp
    src/mj_recorder.cpp
    src/mj_gif_writer.cpp
    src/mj_mjcf_parser.cpp
    src/mj_simulation_executor.cpp
    src/mj_menagerie_downloader.cpp
    src/rl/reward_shaper.cpp
    src/rl/observation_filter.cpp
    src/env/mj_task.cpp
    src/env/mj_env_batch.cpp
    src/env/mj_scene.cpp
    src/env/mj_stress.cpp
    src/env/mj_robustness.cpp
    src/env/mj_inspect.cpp
    src/env/mj_physics.cpp
    src/env/mj_history.cpp
    src/env/mj_ghost.cpp
    src/env/mj_model_ref.cpp
)

target_include_directories(mujoco_plugin PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${Stb_INCLUDE_DIR}
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

# The environments for training: settings, autoreset, the same runs on any
# number of threads, steps a second. tests/mujoco_gymnasium_parity.py replays
# its --trajectory output in Gymnasium.
add_executable(test_mujoco_environments
    tests/test_mujoco_environments.cpp
    src/env/mj_task.cpp
    src/env/mj_env_batch.cpp
    src/env/mj_scene.cpp
    src/env/mj_stress.cpp
    src/env/mj_robustness.cpp
)
target_include_directories(test_mujoco_environments PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_mujoco_environments PRIVATE cyxwiz-plugin-sdk mujoco::mujoco)
target_compile_definitions(test_mujoco_environments PRIVATE
    CYXWIZ_MUJOCO_ASSETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets")
add_custom_command(TARGET test_mujoco_environments POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CYXWIZ_MUJOCO_RUNTIME}" "$<TARGET_FILE_DIR:test_mujoco_environments>/"
    COMMENT "Copying the MuJoCo runtime next to test_mujoco_environments")
add_test(NAME mujoco_environments COMMAND test_mujoco_environments)

# The Viewport's GIF writer (decoded back), recorder, saved choices and the
# Stress panel's grounds and event log
add_executable(test_mujoco_viewport
    tests/test_mujoco_viewport.cpp
    src/mj_gif_writer.cpp
    src/mj_recorder.cpp
    src/mj_viewport_settings.cpp
    src/env/mj_task.cpp
    src/env/mj_scene.cpp
    src/env/mj_stress.cpp
    src/env/mj_inspect.cpp
    src/env/mj_physics.cpp
    src/env/mj_history.cpp
    src/env/mj_ghost.cpp
    src/env/mj_model_ref.cpp
    src/mj_env_manager.cpp
)
target_include_directories(test_mujoco_viewport PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${Stb_INCLUDE_DIR})
target_link_libraries(test_mujoco_viewport PRIVATE cyxwiz-plugin-sdk mujoco::mujoco nlohmann_json::nlohmann_json
    spdlog::spdlog)
target_compile_definitions(test_mujoco_viewport PRIVATE CYXWIZ_MUJOCO_ASSETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets")
add_custom_command(TARGET test_mujoco_viewport POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CYXWIZ_MUJOCO_RUNTIME}" "$<TARGET_FILE_DIR:test_mujoco_viewport>/"
    COMMENT "Copying the MuJoCo runtime next to test_mujoco_viewport")
add_test(NAME mujoco_viewport COMMAND test_mujoco_viewport)
# The Gymnasium replay needs CYXWIZ_RL_ORACLE_PYTHON (cyxwiz-engine) to have
# mujoco of the plugin's version too (the Engine's python-embed does).
if(CYXWIZ_RL_ORACLE_PYTHON)
    add_test(NAME mujoco_gymnasium_parity
             COMMAND ${CYXWIZ_RL_ORACLE_PYTHON} -I ${CMAKE_CURRENT_SOURCE_DIR}/tests/mujoco_gymnasium_parity.py
                     $<TARGET_FILE:test_mujoco_environments> ${CMAKE_CURRENT_BINARY_DIR}/parity)
endif()
# pycyxwiz.rl over _cyxwiz_envs (the Engine's module) with the same Python
if(CYXWIZ_RL_ORACLE_PYTHON AND TARGET _cyxwiz_envs)
    add_test(NAME mujoco_python_envs
             COMMAND ${CYXWIZ_RL_ORACLE_PYTHON} -I ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mujoco_python_envs.py
                     $<TARGET_FILE_DIR:_cyxwiz_envs> ${CMAKE_CURRENT_SOURCE_DIR})
endif()
