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
if(CYXWIZ_HAS_PYTHON)
    find_package(pybind11 CONFIG QUIET)
endif()
find_package(httplib CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)

# Cross-platform native file dialogs
find_package(nfd CONFIG REQUIRED)

# Optional: HDF5 support for data files
find_package(HighFive CONFIG)
if(HighFive_FOUND)
    message(STATUS "HighFive found - HDF5 support enabled")
    set(CYXWIZ_HAS_HDF5 ON)
else()
    message(WARNING "HighFive not found - HDF5 support disabled")
    set(CYXWIZ_HAS_HDF5 OFF)
endif()


# Optional: OpenCV for image processing and augmentation
find_package(OpenCV CONFIG)
if(OpenCV_FOUND)
    message(STATUS "OpenCV ${OpenCV_VERSION} found - Image processing enabled")
    set(CYXWIZ_HAS_OPENCV ON)
else()
    message(WARNING "OpenCV not found - Image augmentation disabled (using STB fallback)")
    set(CYXWIZ_HAS_OPENCV OFF)
endif()

# ImNodes for visual node editor - integrated manually
# Source: https://github.com/Nelarius/imnodes

# Source files
set(ENGINE_SOURCES
    src/main.cpp
    src/application.cpp
    src/auth/auth_client.cpp
    src/core/project_manager.cpp
    src/core/data_registry.cpp
    src/core/texture_manager.cpp
    src/core/async_task_manager.cpp
    src/core/graph_compiler.cpp
    src/core/dataset_batcher.cpp
    src/core/training_executor.cpp
    src/core/training_manager.cpp
    src/core/test_executor.cpp
    src/core/test_manager.cpp
    src/core/checkpoint_manager.cpp
    src/core/model_exporter.cpp
    src/core/model_importer.cpp
    src/core/model_analyzer.cpp
    src/core/model_converter.cpp
    src/core/file_dialogs.cpp
    src/core/formats/cyxmodel_format.cpp
    src/gui/main_window.cpp
    src/gui/node_editor.cpp
    src/gui/node_editor_clipboard.cpp
    src/gui/node_editor_codegen.cpp
    src/gui/node_editor_connection.cpp
    src/gui/node_editor_context_menu.cpp
    src/gui/node_editor_io.cpp
    src/gui/node_editor_nodes.cpp
    src/gui/node_editor_validation.cpp
    src/gui/node_editor_skip.cpp
    src/gui/node_editor_add_search.cpp
    src/gui/node_documentation.cpp
    src/gui/console.cpp
    src/gui/viewport.cpp
    src/gui/properties.cpp
    src/gui/panels/toolbar.cpp
    src/gui/panels/toolbar_file_menu.cpp
    src/gui/panels/toolbar_edit_menu.cpp
    src/gui/panels/toolbar_view_menu.cpp
    src/gui/panels/toolbar_other_menus.cpp
    src/gui/panels/toolbar_tools_menu.cpp
    src/gui/panels/asset_browser.cpp
    src/gui/panels/training_dashboard.cpp
    src/gui/panels/training_plot_panel.cpp
    src/gui/panels/training_plot_panel_global.cpp
    src/gui/panels/plot_window.cpp
    src/gui/panels/plot_test_control.cpp
    src/gui/panels/command_window.cpp
    src/gui/panels/script_editor.cpp
    src/gui/panels/table_viewer.cpp
    src/gui/panels/connection_dialog.cpp
    src/gui/panels/dataset_panel.cpp
    src/gui/panels/task_progress_panel.cpp
    src/gui/panels/custom_node_editor.cpp
    src/gui/panels/theme_editor.cpp
    src/gui/panels/profiling_panel.cpp
    src/gui/panels/memory_panel.cpp
    src/gui/panels/memory_monitor.cpp
    src/gui/panels/pattern_browser.cpp
    src/gui/panels/variable_explorer.cpp
    src/gui/panels/test_results_panel.cpp
    src/gui/panels/export_dialog.cpp
    src/gui/panels/import_dialog.cpp
    src/gui/panels/deployment_dialog.cpp
    src/network/deployment_client.cpp
    src/inference/local_inference_server.cpp
    src/gui/panels/model_summary_panel.cpp
    src/gui/panels/architecture_diagram.cpp
    src/gui/panels/lr_finder_panel.cpp
    src/gui/panels/data_profiler_panel.cpp
    src/gui/panels/correlation_matrix_panel.cpp
    src/gui/panels/missing_value_panel.cpp
    src/gui/panels/outlier_detection_panel.cpp
    src/gui/panels/descriptive_stats_panel.cpp
    src/gui/panels/hypothesis_test_panel.cpp
    src/gui/panels/distribution_fitter_panel.cpp
    src/gui/panels/regression_panel.cpp
    src/gui/panels/dim_reduction_panel.cpp
    src/gui/panels/gradcam_panel.cpp
    src/gui/panels/feature_importance_panel.cpp
    src/gui/panels/nas_panel.cpp
    src/gui/panels/kmeans_panel.cpp
    src/gui/panels/dbscan_panel.cpp
    src/gui/panels/hierarchical_panel.cpp
    src/gui/panels/gmm_panel.cpp
    src/gui/panels/cluster_eval_panel.cpp
    src/gui/panels/confusion_matrix_panel.cpp
    src/gui/panels/roc_auc_panel.cpp
    src/gui/panels/pr_curve_panel.cpp
    src/gui/panels/cross_validation_panel.cpp
    src/gui/panels/learning_curves_panel.cpp
    src/gui/panels/normalization_panel.cpp
    src/gui/panels/standardization_panel.cpp
    src/gui/panels/log_transform_panel.cpp
    src/gui/panels/boxcox_panel.cpp
    src/gui/panels/feature_scaling_panel.cpp
    src/gui/panels/matrix_calculator_panel.cpp
    src/gui/panels/eigen_decomp_panel.cpp
    src/gui/panels/svd_panel.cpp
    src/gui/panels/qr_panel.cpp
    src/gui/panels/cholesky_panel.cpp
    src/gui/panels/fft_panel.cpp
    src/gui/panels/spectrogram_panel.cpp
    src/gui/panels/filter_designer_panel.cpp
    src/gui/panels/convolution_panel.cpp
    src/gui/panels/wavelet_panel.cpp
    src/gui/panels/gradient_descent_panel.cpp
    src/gui/panels/convexity_panel.cpp
    src/gui/panels/lp_panel.cpp
    src/gui/panels/qp_panel.cpp
    src/gui/panels/differentiation_panel.cpp
    src/gui/panels/integration_panel.cpp
    src/gui/panels/decomposition_panel.cpp
    src/gui/panels/acf_pacf_panel.cpp
    src/gui/panels/stationarity_panel.cpp
    src/gui/panels/seasonality_panel.cpp
    src/gui/panels/forecasting_panel.cpp
    src/gui/panels/tokenization_panel.cpp
    src/gui/panels/word_frequency_panel.cpp
    src/gui/panels/tfidf_panel.cpp
    src/gui/panels/embeddings_panel.cpp
    src/gui/panels/sentiment_panel.cpp
    src/gui/panels/calculator_panel.cpp
    src/gui/panels/unit_converter_panel.cpp
    src/gui/panels/random_generator_panel.cpp
    src/gui/panels/hash_generator_panel.cpp
    src/gui/panels/json_viewer_panel.cpp
    src/gui/panels/regex_tester_panel.cpp
    src/core/nas_evaluator.cpp
    src/core/data_analyzer.cpp
    src/gui/tutorial/tutorial_system.cpp
    src/gui/panels/query_console.cpp
    src/gui/patterns/pattern_library.cpp
    src/gui/query/cyxql_types.cpp
    src/gui/query/cyxql_lexer.cpp
    src/gui/query/cyxql_parser.cpp
    src/gui/query/cyxql_executor.cpp
    src/data/data_table.cpp
    src/scripting/python_engine.cpp
    src/scripting/script_manager.cpp
    src/scripting/scripting_engine.cpp
    src/scripting/python_sandbox.cpp
    src/scripting/debugger.cpp
    src/scripting/startup_script_manager.cpp
    src/scripting/cell_manager.cpp
    src/gui/panels/output_renderer.cpp
    # ImGuiColorTextEdit (text editor component)
    external/ImGuiColorTextEdit/TextEditor.cpp
    # ImNodes (visual node editor)
    external/imnodes/imnodes.cpp
    src/network/grpc_client.cpp
    src/network/job_manager.cpp
    src/network/p2p_client.cpp
    src/network/dataset_provider.cpp
    src/network/reservation_client.cpp
    src/gui/panels/p2p_training_panel.cpp
    src/gui/panels/job_status_panel.cpp
    src/gui/panels/wallet_panel.cpp
    src/gui/theme.cpp
    src/gui/dock_style.cpp
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
    src/auth/auth_client.h
    src/core/project_manager.h
    src/core/data_registry.h
    src/core/texture_manager.h
    src/core/async_task_manager.h
    src/core/graph_compiler.h
    src/core/dataset_batcher.h
    src/core/training_executor.h
    src/core/training_manager.h
    src/core/test_executor.h
    src/core/test_manager.h
    src/core/checkpoint_manager.h
    src/core/model_format.h
    src/core/model_exporter.h
    src/core/model_importer.h
    src/core/model_analyzer.h
    src/core/model_converter.h
    src/core/file_dialogs.h
    src/core/formats/cyxmodel_format.h
    src/gui/main_window.h
    src/gui/node_editor.h
    src/gui/node_documentation.h
    src/gui/console.h
    src/gui/console_sink.h
    src/gui/viewport.h
    src/gui/properties.h
    src/gui/panel.h
    src/gui/panels/toolbar.h
    src/gui/panels/asset_browser.h
    src/gui/panels/training_dashboard.h
    src/gui/panels/training_plot_panel.h
    src/gui/panels/plot_window.h
    src/gui/panels/plot_test_control.h
    src/gui/panels/command_window.h
    src/gui/panels/script_editor.h
    src/gui/panels/table_viewer.h
    src/gui/panels/connection_dialog.h
    src/gui/panels/dataset_panel.h
    src/gui/panels/task_progress_panel.h
    src/gui/panels/custom_node_editor.h
    src/gui/panels/theme_editor.h
    src/gui/panels/profiling_panel.h
    src/gui/panels/memory_panel.h
    src/gui/panels/memory_monitor.h
    src/gui/panels/pattern_browser.h
    src/gui/panels/test_results_panel.h
    src/gui/panels/export_dialog.h
    src/gui/panels/import_dialog.h
    src/gui/panels/deployment_dialog.h
    src/network/deployment_client.h
    src/inference/local_inference_server.h
    src/gui/panels/model_summary_panel.h
    src/gui/panels/architecture_diagram.h
    src/gui/panels/lr_finder_panel.h
    src/gui/panels/data_profiler_panel.h
    src/gui/panels/correlation_matrix_panel.h
    src/gui/panels/missing_value_panel.h
    src/gui/panels/outlier_detection_panel.h
    src/gui/panels/descriptive_stats_panel.h
    src/gui/panels/hypothesis_test_panel.h
    src/gui/panels/distribution_fitter_panel.h
    src/gui/panels/regression_panel.h
    src/gui/panels/dim_reduction_panel.h
    src/gui/panels/gradcam_panel.h
    src/gui/panels/feature_importance_panel.h
    src/gui/panels/nas_panel.h
    src/gui/panels/kmeans_panel.h
    src/gui/panels/dbscan_panel.h
    src/gui/panels/hierarchical_panel.h
    src/gui/panels/gmm_panel.h
    src/gui/panels/cluster_eval_panel.h
    src/gui/panels/confusion_matrix_panel.h
    src/gui/panels/roc_auc_panel.h
    src/gui/panels/pr_curve_panel.h
    src/gui/panels/cross_validation_panel.h
    src/gui/panels/learning_curves_panel.h
    src/gui/panels/normalization_panel.h
    src/gui/panels/standardization_panel.h
    src/gui/panels/log_transform_panel.h
    src/gui/panels/boxcox_panel.h
    src/gui/panels/feature_scaling_panel.h
    src/gui/panels/matrix_calculator_panel.h
    src/gui/panels/eigen_decomp_panel.h
    src/gui/panels/svd_panel.h
    src/gui/panels/qr_panel.h
    src/gui/panels/cholesky_panel.h
    src/gui/panels/fft_panel.h
    src/gui/panels/spectrogram_panel.h
    src/gui/panels/filter_designer_panel.h
    src/gui/panels/convolution_panel.h
    src/gui/panels/wavelet_panel.h
    src/gui/panels/gradient_descent_panel.h
    src/gui/panels/convexity_panel.h
    src/gui/panels/lp_panel.h
    src/gui/panels/qp_panel.h
    src/gui/panels/differentiation_panel.h
    src/gui/panels/integration_panel.h
    src/gui/panels/decomposition_panel.h
    src/gui/panels/acf_pacf_panel.h
    src/gui/panels/stationarity_panel.h
    src/gui/panels/seasonality_panel.h
    src/gui/panels/forecasting_panel.h
    src/gui/panels/tokenization_panel.h
    src/gui/panels/word_frequency_panel.h
    src/gui/panels/tfidf_panel.h
    src/gui/panels/embeddings_panel.h
    src/gui/panels/sentiment_panel.h
    src/gui/panels/calculator_panel.h
    src/gui/panels/unit_converter_panel.h
    src/gui/panels/random_generator_panel.h
    src/gui/panels/hash_generator_panel.h
    src/gui/panels/json_viewer_panel.h
    src/gui/panels/regex_tester_panel.h
    src/gui/tutorial/tutorial_system.h
    src/gui/panels/query_console.h
    src/gui/patterns/pattern.h
    src/gui/patterns/pattern_library.h
    src/gui/query/cyxql.h
    src/gui/query/cyxql_types.h
    src/gui/query/cyxql_lexer.h
    src/gui/query/cyxql_parser.h
    src/gui/query/cyxql_executor.h
    src/data/data_table.h
    src/scripting/python_engine.h
    src/scripting/script_manager.h
    src/scripting/scripting_engine.h
    src/scripting/python_sandbox.h
    src/scripting/startup_script_manager.h
    src/scripting/cell.h
    src/scripting/cell_manager.h
    src/gui/panels/output_renderer.h
    # ImGuiColorTextEdit (text editor component)
    external/ImGuiColorTextEdit/TextEditor.h
    src/network/grpc_client.h
    src/network/job_manager.h
    src/network/p2p_client.h
    src/network/dataset_provider.h
    src/network/reservation_client.h
    src/gui/panels/p2p_training_panel.h
    src/gui/panels/job_status_panel.h
    src/gui/panels/wallet_panel.h
    src/gui/theme.h
    src/gui/dock_style.h
    # Plotting system
    src/plotting/plot_dataset.h
    src/plotting/plot_manager.h
    src/plotting/test_data_generator.h
    src/plotting/plot_3d.h
    src/plotting/plot_3d_generator.h
    src/plotting/backends/plot_backend.h
    src/plotting/backends/implot_backend.h
    src/plotting/backends/matplotlib_backend.h
    # Transform/Augmentation system
    src/transforms/transform.h
    src/transforms/geometric.h
    src/transforms/color.h
    src/transforms/noise.h
    src/transforms/advanced.h
    src/transforms/transforms.h
)

# Create executable
add_executable(cyxwiz-engine
    ${ENGINE_SOURCES}
    ${ENGINE_HEADERS}
)

# Include directories
# Use BEFORE to ensure bundled libraries take priority over system-installed versions
target_include_directories(cyxwiz-engine BEFORE PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/external/ImGuiColorTextEdit
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imnodes
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
    httplib::httplib
    OpenSSL::SSL
    OpenSSL::Crypto
    cyxwiz-backend
    cyxwiz-protocol
    nfd::nfd
)

# Optional: Link HDF5 if found
if(HighFive_FOUND)
    target_link_libraries(cyxwiz-engine PRIVATE HighFive)
    target_compile_definitions(cyxwiz-engine PRIVATE CYXWIZ_HAS_HDF5)
endif()

# Optional: Link pybind11 for Python scripting
if(CYXWIZ_HAS_PYTHON AND pybind11_FOUND)
    target_link_libraries(cyxwiz-engine PRIVATE pybind11::embed)
    target_compile_definitions(cyxwiz-engine PRIVATE CYXWIZ_HAS_PYTHON)
endif()

# Optional: Link OpenCV if found
if(OpenCV_FOUND)
    target_include_directories(cyxwiz-engine PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(cyxwiz-engine PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(cyxwiz-engine PRIVATE CYXWIZ_HAS_OPENCV)
endif()

# Optional: ONNX Export support
if(CYXWIZ_HAS_ONNX_EXPORT)
    find_package(ONNX CONFIG REQUIRED)
    target_link_libraries(cyxwiz-engine PRIVATE ONNX::onnx)
    target_compile_definitions(cyxwiz-engine PRIVATE
        CYXWIZ_HAS_ONNX_EXPORT
        ONNX_ML  # Required for vcpkg ONNX headers
    )
    message(STATUS "Engine: ONNX Export support enabled")
endif()

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
    # Set Visual Studio debugger working directory to executable location
    # This ensures resources (fonts, icons) are found when running from VS IDE
    set_target_properties(cyxwiz-engine PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:cyxwiz-engine>"
    )
endif()

if(APPLE)
    # macOS-specific
    target_link_libraries(cyxwiz-engine PRIVATE
        "-framework Cocoa"
        "-framework IOKit"
        "-framework CoreVideo"
    )
endif()

# Copy resources to build directory (use cyxwiz-engine's resources, not root)
add_custom_command(TARGET cyxwiz-engine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/resources
    $<TARGET_FILE_DIR:cyxwiz-engine>/resources
    COMMENT "Copying resources..."
)

# Copy startup scripts to build directory
add_custom_command(TARGET cyxwiz-engine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/scripts
    $<TARGET_FILE_DIR:cyxwiz-engine>/scripts
    COMMENT "Copying scripts..."
)

# Copy startup_scripts.txt configuration to build directory
add_custom_command(TARGET cyxwiz-engine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_SOURCE_DIR}/startup_scripts.txt
    $<TARGET_FILE_DIR:cyxwiz-engine>/startup_scripts.txt
    COMMENT "Copying startup_scripts.txt..."
)

# Copy test data files to build directory
add_custom_command(TARGET cyxwiz-engine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_SOURCE_DIR}/test_data.csv
    $<TARGET_FILE_DIR:cyxwiz-engine>/test_data.csv
    COMMENT "Copying test data files..."
)

# Copy pycyxwiz Python module to engine directory (built by cyxwiz-backend)
if(CYXWIZ_HAS_PYTHON AND TARGET pycyxwiz)
    add_custom_command(TARGET cyxwiz-engine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:pycyxwiz>
        $<TARGET_FILE_DIR:cyxwiz-engine>/pycyxwiz.pyd
        COMMENT "Copying pycyxwiz Python module..."
    )
endif()

# ===== Python Bindings Module (Optional) =====

if(CYXWIZ_HAS_PYTHON AND pybind11_FOUND)
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
        src/gui/panels/training_plot_panel.cpp
        src/gui/panels/training_plot_panel_global.cpp
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
endif()

# Install
install(TARGETS cyxwiz-engine
    RUNTIME DESTINATION bin
)

install(DIRECTORY resources/
    DESTINATION share/cyxwiz/resources
)
