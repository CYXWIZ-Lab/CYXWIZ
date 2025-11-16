cmake_minimum_required(VERSION 3.20)

project(cyxwiz-engine)

# Find packages
find_package(imgui CONFIG REQUIRED)
find_package(implot CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# ImNodes for node editor (may need manual integration if not in vcpkg)
# Download from: https://github.com/Nelarius/imnodes

# Source files
set(ENGINE_SOURCES
    src/main.cpp
    src/application.cpp
    src/gui/main_window.cpp
    src/gui/node_editor.cpp
    src/gui/console.cpp
    src/gui/viewport.cpp
    src/gui/properties.cpp
    src/gui/panels/toolbar.cpp
    src/gui/panels/asset_browser.cpp
    src/gui/panels/training_dashboard.cpp
    src/gui/panels/plot_window.cpp
    src/gui/panels/plot_test_control.cpp
    src/gui/panels/command_window.cpp
    src/scripting/python_engine.cpp
    src/scripting/script_manager.cpp
    src/scripting/scripting_engine.cpp
    src/network/grpc_client.cpp
    src/network/job_manager.cpp
    # Plotting system
    src/plotting/plot_dataset.cpp
    src/plotting/plot_manager.cpp
    src/plotting/test_data_generator.cpp
    src/plotting/plot_3d_generator.cpp
    src/plotting/backends/implot_backend.cpp
    src/plotting/backends/matplotlib_backend.cpp
)

set(ENGINE_HEADERS
    src/application.h
    src/gui/main_window.h
    src/gui/node_editor.h
    src/gui/console.h
    src/gui/viewport.h
    src/gui/properties.h
    src/gui/panel.h
    src/gui/panels/toolbar.h
    src/gui/panels/asset_browser.h
    src/gui/panels/training_dashboard.h
    src/gui/panels/plot_window.h
    src/gui/panels/plot_test_control.h
    src/gui/panels/command_window.h
    src/scripting/python_engine.h
    src/scripting/script_manager.h
    src/scripting/scripting_engine.h
    src/network/grpc_client.h
    src/network/job_manager.h
    # Plotting system
    src/plotting/plot_dataset.h
    src/plotting/plot_manager.h
    src/plotting/test_data_generator.h
    src/plotting/plot_3d.h
    src/plotting/plot_3d_generator.h
    src/plotting/backends/plot_backend.h
    src/plotting/backends/implot_backend.h
    src/plotting/backends/matplotlib_backend.h
)

# Create executable
add_executable(cyxwiz-engine
    ${ENGINE_SOURCES}
    ${ENGINE_HEADERS}
)

# Include directories
target_include_directories(cyxwiz-engine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Find OpenGL
find_package(OpenGL REQUIRED)

# Link libraries
target_link_libraries(cyxwiz-engine PRIVATE
    imgui::imgui
    implot::implot
    glfw
    glad::glad
    OpenGL::GL
    fmt::fmt
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    pybind11::embed
    cyxwiz-backend
    cyxwiz-protocol
)

# Platform-specific settings
if(WIN32)
    # Windows-specific
    target_compile_definitions(cyxwiz-engine PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
    # Add icon resource if available
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/resources/icon.rc")
        target_sources(cyxwiz-engine PRIVATE resources/icon.rc)
    endif()
endif()

if(APPLE)
    # macOS-specific
    target_link_libraries(cyxwiz-engine PRIVATE
        "-framework Cocoa"
        "-framework IOKit"
        "-framework CoreVideo"
    )
endif()

# Copy resources to build directory
add_custom_command(TARGET cyxwiz-engine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/resources
    $<TARGET_FILE_DIR:cyxwiz-engine>/resources
    COMMENT "Copying resources..."
)

# ===== Python Bindings Module =====

# Create Python module for plotting bindings
pybind11_add_module(cyxwiz_plotting
    python/plot_bindings.cpp
    src/plotting/plot_manager.cpp
    src/plotting/plot_dataset.cpp
    src/plotting/plot_3d_generator.cpp
    src/plotting/test_data_generator.cpp
    src/plotting/backends/implot_backend.cpp
    src/plotting/backends/matplotlib_backend.cpp
    src/plotting/stb_image_impl.cpp
    src/gui/panels/plot_window.cpp
    src/gui/panels/plot_test_control.cpp
    src/gui/panels/plot_test_panel.cpp
)

# Link the plotting module against necessary libraries
target_link_libraries(cyxwiz_plotting PRIVATE
    imgui::imgui
    implot::implot
    glad::glad
    glfw
    fmt::fmt
    spdlog::spdlog
)

# Include directories for the Python module
target_include_directories(cyxwiz_plotting PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Set output directory for Python module
set_target_properties(cyxwiz_plotting PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python"
)

# Platform-specific module settings
if(WIN32)
    # Windows: .pyd extension
    set_target_properties(cyxwiz_plotting PROPERTIES
        SUFFIX ".pyd"
    )
elseif(APPLE)
    # macOS: .so extension
    set_target_properties(cyxwiz_plotting PROPERTIES
        SUFFIX ".so"
    )
else()
    # Linux: .so extension (default)
endif()

# Install Python module
install(TARGETS cyxwiz_plotting
    LIBRARY DESTINATION lib/python/cyxwiz
    RUNTIME DESTINATION lib/python/cyxwiz
)

# Install Python examples
install(DIRECTORY python/examples/
    DESTINATION share/cyxwiz/python/examples
    FILES_MATCHING PATTERN "*.py"
)

# Install
install(TARGETS cyxwiz-engine
    RUNTIME DESTINATION bin
)

install(DIRECTORY resources/
    DESTINATION share/cyxwiz/resources
)
