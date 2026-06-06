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
find_package(httplib CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)

# Cross-platform native file dialogs
find_package(nfd CONFIG REQUIRED)

# Apache Arrow for Data Studio columnar data support
find_package(Arrow CONFIG REQUIRED)
find_package(Parquet CONFIG REQUIRED)
message(STATUS "Apache Arrow ${Arrow_VERSION} found - Columnar data support enabled")

# DuckDB for zero-copy SQL queries on Arrow tables
find_package(duckdb CONFIG REQUIRED)
message(STATUS "DuckDB found - SQL query engine enabled")

# Optional: HDF5 support for data files
find_package(HighFive CONFIG)
if(HighFive_FOUND)
    message(STATUS "HighFive found - HDF5 support enabled")
    set(CYXWIZ_HAS_HDF5 ON)
else()
    message(WARNING "HighFive not found - HDF5 support disabled")
    set(CYXWIZ_HAS_HDF5 OFF)
endif()


# OpenCV for image processing and augmentation (REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
message(STATUS "OpenCV ${OpenCV_VERSION} found - Image processing enabled")
set(CYXWIZ_HAS_OPENCV ON)

# ImNodes for visual node editor - integrated manually
# Source: https://github.com/Nelarius/imnodes

# Plugin SDK static library (shared between engine and plugins)
add_library(cyxwiz-plugin-sdk STATIC
    src/plugin/plugin_context.cpp
    src/plugin/registries/plugin_node_registry.cpp
    src/plugin/registries/plugin_panel_registry.cpp
    src/plugin/registries/plugin_data_loader_registry.cpp
    src/plugin/registries/plugin_training_hook_manager.cpp
    src/plugin/registries/plugin_analytics_registry.cpp
    src/plugin/security/permission_store.cpp
    src/plugin/security/permission_dialog.cpp
    src/plugin/security/plugin_signature.cpp
    src/plugin/security/safe_execute.cpp
    src/plugin/security/tweetnacl.c
)
target_include_directories(cyxwiz-plugin-sdk PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(cyxwiz-plugin-sdk PUBLIC
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    imgui::imgui
)
set_target_properties(cyxwiz-plugin-sdk PROPERTIES
    CXX_STANDARD 20
    POSITION_INDEPENDENT_CODE ON
)

# Source files
set(ENGINE_SOURCES
    src/main.cpp
    src/application.cpp
    src/auth/auth_client.cpp
    src/core/project_manager.cpp
    src/core/engine_config.cpp
    src/core/python_detector.cpp
    src/core/window_manager.cpp
    src/core/data_registry_core.cpp
    src/core/data_registry.cpp
    src/core/data_registry_config.cpp
    src/core/data_registry_utils.cpp
    src/core/dataset_base.cpp
    src/core/datasets/kaggle_dataset.cpp
    src/core/datasets/hdf5_dataset.cpp
    src/core/datasets/image_csv_dataset.cpp
    src/core/datasets/streaming_dataset.cpp
    src/core/datasets/mnist_dataset.cpp
    src/core/datasets/cifar10_dataset.cpp
    src/core/datasets/csv_dataset.cpp
    src/core/datasets/tsv_dataset.cpp
    src/core/datasets/json_dataset.cpp
    src/core/datasets/txt_dataset.cpp
    src/core/datasets/image_folder_dataset.cpp
    src/core/datasets/huggingface_dataset.cpp
    src/core/datasets/custom_dataset.cpp
    src/core/arrow_dataset.cpp
    src/core/parquet_backed_dataset.cpp
    src/core/parquet_arrow_batcher.cpp
    src/core/image_dataset_batcher.cpp
    src/core/audio_dataset_batcher.cpp
    src/core/text_dataset_batcher.cpp
    src/core/text_arrow_adapter.cpp
    src/core/arrow_dataset_batcher.cpp
    src/core/arrow_converters.cpp
    src/core/duckdb_connector.cpp
    src/core/pipeline_executor.cpp  # Unified Canvas Phase 2: Moved from data_studio
    src/core/pipeline_table_materializer.cpp  # Cat-1 graph walk over an Arrow table
    src/core/pipeline_materializer.cpp  # Cat-1 IPipelineOperator integration into training launch
    src/core/annotation_manager.cpp
    src/core/node_metadata_registry.cpp  # Node Browser: metadata registry
    src/core/texture_manager.cpp
    src/core/async_task_manager.cpp
    src/core/dataset_audit.cpp
    src/core/graph_compiler.cpp
    src/core/compiled_graph_plan.cpp
    src/core/graph_executor.cpp
    src/core/rl_training_executor.cpp
    src/core/rl_script_generator.cpp
    src/core/dataset_batcher.cpp
    src/core/image_utils.cpp
    src/core/model_builder.cpp
    src/core/synthetic_batch.cpp
    src/core/debug_executor.cpp
    src/core/debug_session_manager.cpp
    src/core/preflight_validator.cpp
    src/core/crash_run_recorder.cpp
    src/core/text_preprocessing_tracer.cpp
    src/core/smoke_run_executor.cpp
    src/core/debug_recommendation_engine.cpp
    src/core/training_trace_collector.cpp
    src/core/debug_run_store.cpp
    src/core/graph_topology_utils.cpp
    src/core/training_batcher_setup.cpp
    src/core/training_executor.cpp
    src/core/training_manager.cpp
    src/core/executable_model.cpp
    src/core/graph_executable_model.cpp
    src/core/test_executor.cpp
    # Node executors (shared logic for nodes and panels)
    src/core/node_executors/kmeans_executor.cpp
    src/core/node_executors/pipeline_operator_factory.cpp
    src/core/node_executors/time_series_window_operator.cpp  # Phase 4 Band 1
    src/core/node_executors/time_series_split_operator.cpp   # Phase 4 Band 2
    src/core/node_executors/log_transform_operator.cpp       # Phase 4 Band 1
    src/core/node_executors/differencing_operator.cpp        # Phase 4 Band 1
    src/core/node_executors/time_series_features_operator.cpp  # Phase 4 Band 1
    src/core/node_executors/text_tokenizer_operator.cpp      # Fix B Band 1
    src/core/node_executors/tfidf_vectorizer_operator.cpp    # Tool-to-Node Band 1
    src/core/node_executors/count_vectorizer_operator.cpp    # Tool-to-Node Band 1
    src/core/node_executors/pca_operator.cpp                 # Tool-to-Node Band 1
    src/core/node_executors/sentiment_analyzer_operator.cpp  # Tool-to-Node Band 1
    src/core/node_executors/clustering_operators.cpp         # Tool-to-Node Band 1
    src/core/node_executors/signal_processing_operators.cpp  # Tool-to-Node Band 1
    src/core/node_executors/regression_operators.cpp         # Tool-to-Node Band 1
    src/core/node_executors/preprocessing_operators.cpp      # Tool-to-Node Band 1
    src/core/node_executors/time_series_analysis_operators.cpp  # Phase 5 Band 1
    # Plugin system (loader/manager only; SDK sources in cyxwiz-plugin-sdk)
    src/plugin/plugin_loader.cpp
    src/plugin/plugin_manager.cpp
    # Preprocessing system
    src/preprocessing/preprocessing_config.cpp
    src/preprocessing/statistics_calculator.cpp
    src/preprocessing/normalization_transform.cpp
    src/preprocessing/scaling_transform.cpp
    src/preprocessing/image_transform.cpp
    # OpenCV Transforms (Phase 1)
    src/preprocessing/morphology_transform.cpp
    src/preprocessing/blur_transform.cpp
    src/preprocessing/perspective_transform.cpp
    src/preprocessing/pyramid_transform.cpp
    # Image utilities
    src/utils/image_enhancer.cpp
    src/utils/dataset_analyzer.cpp
    src/utils/image_quality_analyzer.cpp
    src/utils/image_deduplicator.cpp
    src/utils/hdf5_browser.cpp
    # Dataset utilities (Phase 2)
    src/utils/segmentation_utils.cpp
    src/utils/shape_utils.cpp
    src/utils/labeling_utils.cpp
    src/core/test_manager.cpp
    src/core/checkpoint_manager.cpp
    src/core/model_exporter.cpp
    src/core/model_importer.cpp
    src/core/model_analyzer.cpp
    src/core/model_converter.cpp
    src/core/file_dialogs.cpp
    src/core/formats/cyxmodel_format.cpp
    src/core/formats/text_dataset.cpp
    src/core/formats/timeseries_dataset.cpp
    src/core/formats/audio_dataset.cpp
    src/core/gym_connector.cpp
    src/gui/graph_training_launcher.cpp
    src/gui/main_window.cpp
    src/gui/node_editor.cpp
    src/gui/node_editor_clipboard.cpp
    src/gui/node_editor_codegen.cpp
    src/gui/node_editor_connection.cpp
    src/gui/node_editor_context_menu.cpp
    src/gui/node_editor_io.cpp
    src/gui/node_editor_nodes.cpp
    src/gui/node_editor_validation.cpp
    src/gui/node_editor_shape_inference.cpp
    src/gui/node_editor_skip.cpp
    src/gui/node_editor_add_search.cpp
    src/gui/node_editor_icons.cpp
    src/gui/node_documentation.cpp
    src/gui/console.cpp
    src/gui/viewport.cpp
    src/gui/properties.cpp
    src/gui/properties_advanced.cpp
    src/gui/properties_executor.cpp
    src/gui/properties_metadata_editor.cpp
    src/gui/properties_node_editors.cpp
    src/gui/properties_parameter_rules.cpp
    src/gui/properties_presets.cpp
    src/gui/properties_shape_info.cpp
    src/gui/node_config_dialog.cpp
    src/gui/data_input_capabilities.cpp
    src/gui/data_input_preview.cpp
    src/gui/data_input_dialog_preview.cpp
    src/gui/data_input_dialog_profile.cpp
    src/gui/data_input_dialog_helpers.cpp
    src/gui/data_input_dialog_filter_tabs.cpp
    src/gui/data_input_dialog_status_tabs.cpp
    src/gui/data_input_dialog_ml_dataset.cpp
    src/gui/data_input_dialog_external_sources.cpp
    src/gui/data_input_dialog_summary.cpp
    src/gui/data_input_dialog_source_options.cpp
    src/gui/data_input_dialog_apply.cpp
    src/gui/data_io_dialogs.cpp
    src/gui/data_input_dialog.cpp
    src/gui/loaders/audio_loader.cpp
    src/gui/loaders/data_loader.cpp
    src/gui/loaders/image_loader.cpp
    src/gui/loaders/tabular_loader.cpp
    src/gui/loaders/text_loader.cpp
    src/gui/visualization/visualization_nodes.cpp
    src/gui/visualization/bar_chart_dialog.cpp
    src/gui/panels/toolbar.cpp
    src/gui/panels/toolbar_command_palette.cpp
    src/gui/panels/toolbar_file_menu.cpp
    src/gui/panels/toolbar_edit_menu.cpp
    src/gui/panels/toolbar_view_menu.cpp
    src/gui/panels/toolbar_profile_menu.cpp
    src/gui/panels/toolbar_other_menus.cpp
    src/gui/panels/toolbar_tools_menu.cpp
    src/gui/panels/asset_browser.cpp
    src/gui/panels/training_dashboard.cpp
    src/gui/panels/training_plot_panel.cpp
    src/gui/panels/training_plot_panel_global.cpp
    src/gui/panels/plot_window.cpp
    src/gui/panels/python_plot_window_registry.cpp
    src/gui/panels/plot_test_control.cpp
    src/gui/panels/plot_output_panel.cpp
    src/gui/panels/command_window.cpp
    src/gui/panels/script_editor.cpp
    src/gui/panels/script_editor_edit.cpp
    src/gui/panels/table_viewer.cpp
    src/gui/panels/data_explorer_panel.cpp
    src/gui/panels/visualization_panel.cpp
    src/gui/panels/connection_dialog.cpp
    src/gui/dialogs/python_setup_wizard.cpp
    src/gui/dialogs/project_selection_dialog.cpp
    src/gui/dialogs/start_page.cpp
    # Phase 5: Dataset Manager removed - functionality moved to node-based workflow
    # src/gui/panels/dataset_panel.cpp
    # src/gui/panels/dataset_panel_interactive.cpp
    # src/gui/panels/dataset_panel_evaluate.cpp
    # src/gui/panels/dataset_panel_export.cpp
    # Interactive annotation tools (Phase 4) - kept for future Annotation Editor panel
    src/gui/annotation/annotation_tools.cpp
    src/gui/annotation/grabcut_editor.cpp
    src/gui/annotation/watershed_editor.cpp
    # Annotation Editor Panel (Phase 2 - UI Consolidation)
    src/gui/panels/annotation_editor_panel.cpp
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
    src/gui/panels/python_settings_panel.cpp
    src/network/deployment_client.cpp
    src/inference/local_inference_server.cpp
    # DNN inference (Phase 3)
    src/inference/dnn_models.cpp
    src/inference/dnn_preprocessor.cpp
    src/gui/panels/model_summary_panel.cpp
    src/gui/panels/architecture_diagram.cpp
    src/gui/panels/lr_finder_panel.cpp
    src/gui/panels/studio_debugger_panel.cpp
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
    src/gui/panels/hyperparam_search_panel.cpp
    src/core/hyperparam_search.cpp
    src/gui/panels/serving_panel.cpp
    src/gui/panels/dnn_inference_panel.cpp
    src/serving/inference_server.cpp
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
    src/gui/panels/optimizer_settings_panel.cpp
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
    src/data/lazy_data_table.cpp
    src/scripting/python_engine.cpp
    src/scripting/script_manager.cpp
    src/scripting/scripting_engine.cpp
    src/scripting/python_sandbox.cpp
    src/scripting/debugger.cpp
    src/scripting/startup_script_manager.cpp
    src/scripting/cell_manager.cpp
    src/gui/panels/output_renderer.cpp
    # Preview system
    src/gui/preview/preview_renderer.cpp
    src/gui/preview/text_preview.cpp
    src/gui/preview/json_tree.cpp
    src/gui/preview/table_preview.cpp
    src/gui/preview/image_preview.cpp
    src/gui/preview/markdown_preview.cpp
    src/gui/preview/hex_preview.cpp
    src/gui/preview/hdf5_preview.cpp
    # ImGuiColorTextEdit (text editor component)
    external/ImGuiColorTextEdit/TextEditor.cpp
    # ImNodes (visual node editor)
    external/imnodes/imnodes.cpp
    src/network/grpc_client.cpp
    src/network/job_manager.cpp
    src/network/p2p_client.cpp
    src/network/dataset_provider.cpp
    src/network/reservation_client.cpp
    src/network/datastream_client.cpp
    src/core/cloud_data_registry.cpp
    src/gui/panels/cloud_browser.cpp
    src/gui/panels/node_browser_panel.cpp  # Node Browser: KNIME-style node palette
    src/gui/panels/node_info_panel.cpp     # Node Info: Context-sensitive documentation
    src/gui/panels/cloud_dataset_manager.cpp
    src/gui/panels/cloud_upload_dialog.cpp
    src/gui/panels/trust_badge.cpp
    src/gui/panels/p2p_training_panel.cpp
    src/gui/panels/job_status_panel.cpp
    src/gui/panels/wallet_panel.cpp
    src/gui/panels/plugin_manager_panel.cpp
    src/gui/theme.cpp
    src/gui/theme_presets.cpp
    src/gui/dock_style.cpp
    # Data Studio (Phase 1 Week 1)
    # src/gui/data_studio/pipeline_canvas.cpp  # Unified Canvas Phase 5: Replaced by Node Editor
    src/gui/data_studio/query_editor.cpp
    src/gui/data_studio/analyzer.cpp
    src/gui/data_studio/visualizer.cpp
    src/gui/data_studio/data_studio_panel.cpp
    # pipeline_executor.cpp moved to src/core/ (Unified Canvas Phase 2)
    src/gui/data_studio/node_registry.cpp
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
    src/core/engine_config.h
    src/core/window_manager.h
    src/core/data_registry.h
    src/core/arrow_dataset.h
    src/core/arrow_converters.h
    src/core/duckdb_connector.h
    src/core/pipeline_executor.h  # Unified Canvas Phase 2: Moved from data_studio
    src/core/node_metadata.h       # Node Browser: node metadata structures
    src/core/node_metadata_registry.h  # Node Browser: metadata registry
    src/core/texture_manager.h
    src/core/async_task_manager.h
    src/core/graph_compiler.h
    src/core/dataset_batcher.h
    src/core/model_builder.h
    src/core/debug_executor.h
    src/core/synthetic_batch.h
    src/core/training_executor.h
    src/core/training_manager.h
    src/core/test_executor.h
    src/core/node_executors/node_executor.h
    src/core/node_executors/kmeans_executor.h
    src/core/node_executors/node_executor_factory.h
    src/core/node_executors/pipeline_operator.h
    src/core/node_executors/pipeline_operator_factory.h
    src/core/node_executors/identity_operator.h
    src/core/node_executors/time_series_window_operator.h
    src/core/node_executors/time_series_split_operator.h
    src/core/node_executors/log_transform_operator.h
    src/core/node_executors/differencing_operator.h
    src/core/node_executors/time_series_features_operator.h
    src/core/node_executors/text_tokenizer_operator.h
    src/core/node_executors/tfidf_vectorizer_operator.h
    src/core/node_executors/count_vectorizer_operator.h
    src/core/node_executors/pca_operator.h
    src/core/node_executors/sentiment_analyzer_operator.h
    src/core/node_executors/clustering_operators.h
    src/core/node_executors/signal_processing_operators.h
    src/core/node_executors/regression_operators.h
    src/core/node_executors/preprocessing_operators.h
    src/core/node_executors/time_series_analysis_operators.h
    src/core/node_executors/feature_matrix_utils.h
    src/core/node_executors/text_column_utils.h
    src/core/node_executors/ts_column_utils.h
    src/core/pipeline_materializer.h
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
    src/gui/panels/python_plot_window_registry.h
    src/gui/panels/plot_test_control.h
    src/gui/panels/plot_output_panel.h
    src/gui/panels/command_window.h
    src/gui/panels/script_editor.h
    src/gui/panels/table_viewer.h
    src/gui/panels/data_explorer_panel.h
    src/gui/panels/connection_dialog.h
    src/gui/dialogs/python_setup_wizard.h
    src/gui/dialogs/project_selection_dialog.h
    src/gui/dialogs/start_page.h
    # src/gui/panels/dataset_panel.h  # Phase 5: Removed
    # Interactive annotation tools (Phase 4)
    src/gui/annotation/annotation_tools.h
    src/gui/annotation/grabcut_editor.h
    src/gui/annotation/watershed_editor.h
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
    src/gui/panels/python_settings_panel.h
    src/network/deployment_client.h
    src/inference/local_inference_server.h
    # DNN inference (Phase 3)
    src/inference/dnn_models.h
    src/inference/dnn_preprocessor.h
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
    src/gui/panels/hyperparam_search_panel.h
    src/core/hyperparam_search.h
    src/gui/panels/serving_panel.h
    src/serving/inference_server.h
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
    src/gui/panels/optimizer_settings_panel.h
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
    src/network/datastream_client.h
    src/core/cloud_data_registry.h
    src/gui/panels/cloud_browser.h
    src/gui/panels/node_browser_panel.h  # Node Browser: KNIME-style node palette
    src/gui/panels/node_info_panel.h     # Node Info: Context-sensitive documentation
    src/gui/panels/cloud_dataset_manager.h
    src/gui/panels/cloud_upload_dialog.h
    src/gui/panels/trust_badge.h
    src/gui/panels/p2p_training_panel.h
    src/gui/panels/job_status_panel.h
    src/gui/panels/wallet_panel.h
    src/gui/theme.h
    src/gui/dock_style.h
    # Data Studio (Phase 1 Week 1)
    # src/gui/data_studio/pipeline_canvas.h  # Unified Canvas Phase 5: Replaced by Node Editor
    src/gui/data_studio/query_editor.h
    src/gui/data_studio/analyzer.h
    src/gui/data_studio/visualizer.h
    src/gui/data_studio/data_studio_panel.h
    # pipeline_executor.h moved to src/core/ (Unified Canvas Phase 2)
    src/gui/data_studio/node_registry.h
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
    # Preprocessing system
    src/preprocessing/preprocessing_config.h
    src/preprocessing/statistics_calculator.h
    src/preprocessing/normalization_transform.h
    src/preprocessing/scaling_transform.h
    src/preprocessing/image_transform.h
    # OpenCV Transforms (Phase 1)
    src/preprocessing/morphology_transform.h
    src/preprocessing/blur_transform.h
    src/preprocessing/perspective_transform.h
    src/preprocessing/pyramid_transform.h
    # Image utilities
    src/utils/image_enhancer.h
    src/utils/dataset_analyzer.h
    # Dataset utilities (Phase 2)
    src/utils/segmentation_utils.h
    src/utils/shape_utils.h
    src/utils/labeling_utils.h
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
    pybind11::embed
    httplib::httplib
    OpenSSL::SSL
    OpenSSL::Crypto
    cyxwiz-backend
    cyxwiz-protocol
    cyxwiz-plugin-sdk
    nfd::nfd
    Arrow::arrow_shared
    Parquet::parquet_shared
    duckdb
)

# Optional: Link HDF5 if found
if(HighFive_FOUND)
    target_link_libraries(cyxwiz-engine PRIVATE HighFive)
    target_compile_definitions(cyxwiz-engine PRIVATE CYXWIZ_HAS_HDF5)
endif()

# Optional: Link OpenCV if found
if(OpenCV_FOUND)
    target_include_directories(cyxwiz-engine PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(cyxwiz-engine PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(cyxwiz-engine PRIVATE CYXWIZ_HAS_OPENCV)
endif()

# Python scripting support (controlled by CYXWIZ_HAS_PYTHON from root CMakeLists.txt)
if(CYXWIZ_HAS_PYTHON)
    target_compile_definitions(cyxwiz-engine PRIVATE CYXWIZ_HAS_PYTHON)
    message(STATUS "Engine: Python scripting support enabled")
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
    # Link Windows-specific libraries (PDH for CPU metrics in System Monitor)
    target_link_libraries(cyxwiz-engine PRIVATE pdh)
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
add_custom_command(TARGET cyxwiz-engine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:pycyxwiz>
    $<TARGET_FILE_DIR:cyxwiz-engine>/pycyxwiz.pyd
    COMMENT "Copying pycyxwiz Python module..."
)

# Note: Bundled Python has been removed. CyxWiz Engine now uses system Python 3.12+.
# Each project creates its own virtual environment in <project>/python/

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
    src/gui/panels/training_plot_panel.cpp
    src/gui/panels/training_plot_panel_global.cpp
    src/gui/panels/python_plot_window_registry.cpp
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

# Copy cyxwiz_plotting Python module to engine directory after build
add_custom_command(TARGET cyxwiz_plotting POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:cyxwiz_plotting>
    $<TARGET_FILE_DIR:cyxwiz-engine>/cyxwiz_plotting$<TARGET_FILE_SUFFIX:cyxwiz_plotting>
    COMMENT "Copying cyxwiz_plotting Python module..."
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

# ===== Test: Sandbox Timeout =====
add_executable(test_sandbox_timeout
    tests/test_sandbox_timeout.cpp
    src/scripting/python_engine.cpp
    src/scripting/python_sandbox.cpp
)

target_include_directories(test_sandbox_timeout PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_sandbox_timeout PRIVATE
    pybind11::embed
    spdlog::spdlog
    fmt::fmt
)

if(WIN32)
    target_compile_definitions(test_sandbox_timeout PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
    target_link_libraries(test_sandbox_timeout PRIVATE psapi)
endif()

# Set output directory
set_target_properties(test_sandbox_timeout PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: OpenCV Integration =====
add_executable(test_opencv_integration
    tests/test_opencv_integration.cpp
    src/preprocessing/morphology_transform.cpp
    src/preprocessing/blur_transform.cpp
    src/preprocessing/perspective_transform.cpp
    src/preprocessing/pyramid_transform.cpp
    src/utils/segmentation_utils.cpp
    src/utils/shape_utils.cpp
    src/utils/labeling_utils.cpp
    src/core/image_utils.cpp
)

target_include_directories(test_opencv_integration PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_opencv_integration PRIVATE
    ${OpenCV_LIBS}
    spdlog::spdlog
    fmt::fmt
)

if(WIN32)
    target_compile_definitions(test_opencv_integration PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_opencv_integration PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# HDF5 Loading test - run via engine's Python console or dedicated test runner
# The HDF5Dataset class is tested as part of cyxwiz-engine

# ===== Test: Arrow Integration (Phase 0 Day 5) =====
add_executable(test_arrow_integration
    tests/test_arrow_integration.cpp
    src/core/arrow_dataset.cpp
    src/core/arrow_converters.cpp
    src/core/duckdb_connector.cpp
    src/core/data_registry_core.cpp
    src/core/data_registry_utils.cpp
    src/core/dataset_base.cpp
)

target_include_directories(test_arrow_integration PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_arrow_integration PRIVATE
    Arrow::arrow_shared
    Parquet::parquet_shared
    duckdb
    spdlog::spdlog
    fmt::fmt
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_arrow_integration PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_arrow_integration PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Debug Executor (Local Debug Mode - Commit 1) =====
# Smoke test for BuildSequentialFromConfig extraction and MakeSyntheticBatch.
# Compiles the extracted files directly so the test does not depend on the
# full engine executable being built.
add_executable(test_debug_executor
    tests/test_debug_executor.cpp
    src/core/arrow_dataset.cpp
    src/core/arrow_dataset_batcher.cpp
    src/core/dataset_base.cpp
    src/core/model_builder.cpp
    src/core/synthetic_batch.cpp
    src/core/debug_executor.cpp
    src/core/node_executors/text_tokenizer_operator.cpp
    src/core/text_arrow_adapter.cpp
    src/core/text_dataset_batcher.cpp
    src/core/formats/text_dataset.cpp
)

target_include_directories(test_debug_executor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_debug_executor PRIVATE
    Arrow::arrow_shared
    Parquet::parquet_shared
    imgui::imgui               # graph_compiler.h transitively pulls node_editor.h -> imgui.h
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    fmt::fmt
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_debug_executor PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_debug_executor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Dataset Audit =====
add_executable(test_dataset_audit
    tests/test_dataset_audit.cpp
    src/core/arrow_dataset.cpp
    src/core/dataset_audit.cpp
    src/core/image_utils.cpp
    src/core/parquet_backed_dataset.cpp
)

target_include_directories(test_dataset_audit PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_dataset_audit PRIVATE
    Arrow::arrow_shared
    Parquet::parquet_shared
    ${OpenCV_LIBS}
    spdlog::spdlog
    fmt::fmt
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_dataset_audit PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_dataset_audit PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Text Tokenizer Operator =====
add_executable(test_text_tokenizer_operator
    tests/test_text_tokenizer_operator.cpp
    src/core/node_executors/text_tokenizer_operator.cpp
)

target_include_directories(test_text_tokenizer_operator PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_text_tokenizer_operator PRIVATE
    Arrow::arrow_shared
    spdlog::spdlog
    fmt::fmt
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_text_tokenizer_operator PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_text_tokenizer_operator PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Time-Series Analysis Operators =====
add_executable(test_time_series_analysis_operators
    tests/test_time_series_analysis_operators.cpp
    src/core/node_executors/time_series_analysis_operators.cpp
)

target_include_directories(test_time_series_analysis_operators PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_time_series_analysis_operators PRIVATE
    Arrow::arrow_shared
    spdlog::spdlog
    fmt::fmt
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_time_series_analysis_operators PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_time_series_analysis_operators PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Pipeline Operator Metadata Drift Guard =====
add_executable(test_pipeline_operator_metadata
    tests/test_pipeline_operator_metadata.cpp
    src/core/data_analyzer.cpp
    src/data/data_table.cpp
    src/core/node_metadata_registry.cpp
    src/core/node_executors/pipeline_operator_factory.cpp
    src/core/node_executors/clustering_operators.cpp
    src/core/node_executors/count_vectorizer_operator.cpp
    src/core/node_executors/differencing_operator.cpp
    src/core/node_executors/log_transform_operator.cpp
    src/core/node_executors/pca_operator.cpp
    src/core/node_executors/preprocessing_operators.cpp
    src/core/node_executors/regression_operators.cpp
    src/core/node_executors/sentiment_analyzer_operator.cpp
    src/core/node_executors/signal_processing_operators.cpp
    src/core/node_executors/text_tokenizer_operator.cpp
    src/core/node_executors/tfidf_vectorizer_operator.cpp
    src/core/node_executors/time_series_analysis_operators.cpp
    src/core/node_executors/time_series_features_operator.cpp
    src/core/node_executors/time_series_split_operator.cpp
    src/core/node_executors/time_series_window_operator.cpp
)

target_include_directories(test_pipeline_operator_metadata PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_pipeline_operator_metadata PRIVATE
    Arrow::arrow_shared
    spdlog::spdlog
    fmt::fmt
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_pipeline_operator_metadata PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_pipeline_operator_metadata PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Pattern Template Guard =====
add_executable(test_pattern_template_guard
    tests/test_pattern_template_guard.cpp
    src/gui/patterns/pattern_library.cpp
    src/core/node_metadata_registry.cpp
)

target_include_directories(test_pattern_template_guard PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_pattern_template_guard PRIVATE
    imgui::imgui
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    fmt::fmt
)

if(WIN32)
    target_compile_definitions(test_pattern_template_guard PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_pattern_template_guard PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Graph Compiler Deferred Node Guard =====
add_executable(test_graph_compiler_deferred_nodes
    tests/test_graph_compiler_deferred_nodes.cpp
    src/core/graph_compiler.cpp
    src/core/compiled_graph_plan.cpp
    src/core/graph_topology_utils.cpp
    src/core/node_metadata_registry.cpp
    src/core/data_registry_core.cpp
    src/core/data_registry_utils.cpp
    src/core/arrow_dataset.cpp
    src/core/parquet_backed_dataset.cpp
)

target_include_directories(test_graph_compiler_deferred_nodes PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_graph_compiler_deferred_nodes PRIVATE
    Arrow::arrow_shared
    Parquet::parquet_shared
    imgui::imgui
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    fmt::fmt
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_graph_compiler_deferred_nodes PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_graph_compiler_deferred_nodes PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Tensor Reshape Runtime Contract =====
add_executable(test_tensor_reshape_runtime_contract
    tests/test_tensor_reshape_runtime_contract.cpp
    src/core/model_builder.cpp
)

target_include_directories(test_tensor_reshape_runtime_contract PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_tensor_reshape_runtime_contract PRIVATE
    imgui::imgui
    spdlog::spdlog
    fmt::fmt
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_tensor_reshape_runtime_contract PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_tensor_reshape_runtime_contract PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Executable Model Wrapper =====
add_executable(test_executable_model_wrapper
    tests/test_executable_model_wrapper.cpp
    src/core/model_builder.cpp
    src/core/executable_model.cpp
    src/core/graph_executable_model.cpp
)

target_include_directories(test_executable_model_wrapper PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_executable_model_wrapper PRIVATE
    imgui::imgui
    spdlog::spdlog
    fmt::fmt
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_executable_model_wrapper PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_executable_model_wrapper PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Graph Executable Model =====
add_executable(test_graph_executable_model
    tests/test_graph_executable_model.cpp
    src/core/model_builder.cpp
    src/core/executable_model.cpp
    src/core/graph_executable_model.cpp
)

target_include_directories(test_graph_executable_model PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_graph_executable_model PRIVATE
    imgui::imgui
    spdlog::spdlog
    fmt::fmt
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_graph_executable_model PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_graph_executable_model PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Text Arrow Materializer =====
add_executable(test_text_arrow_materializer
    tests/test_text_arrow_materializer.cpp
    src/core/pipeline_table_materializer.cpp
    src/core/node_executors/text_tokenizer_operator.cpp
    src/core/arrow_dataset.cpp
    src/core/arrow_dataset_batcher.cpp
    src/core/model_builder.cpp
)

target_include_directories(test_text_arrow_materializer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_text_arrow_materializer PRIVATE
    Arrow::arrow_shared
    Parquet::parquet_shared
    imgui::imgui
    spdlog::spdlog
    fmt::fmt
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_text_arrow_materializer PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_text_arrow_materializer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Text Arrow Runtime Training Launch =====
add_executable(test_text_arrow_training_launch
    tests/test_text_arrow_training_launch.cpp
    src/core/pipeline_table_materializer.cpp
    src/core/node_executors/text_tokenizer_operator.cpp
    src/core/arrow_dataset.cpp
    src/core/arrow_dataset_batcher.cpp
    src/core/model_builder.cpp
)

target_include_directories(test_text_arrow_training_launch PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${Stb_INCLUDE_DIR}
)

target_link_libraries(test_text_arrow_training_launch PRIVATE
    Arrow::arrow_shared
    Parquet::parquet_shared
    spdlog::spdlog
    fmt::fmt
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_text_arrow_training_launch PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_text_arrow_training_launch PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Text GUI Training Launch Helper =====
add_executable(test_text_gui_training_launch
    tests/test_text_gui_training_launch.cpp
    src/gui/graph_training_launcher.cpp
    src/core/pipeline_materializer.cpp
    src/core/pipeline_table_materializer.cpp
    src/core/node_executors/text_tokenizer_operator.cpp
    src/core/arrow_dataset.cpp
    src/core/arrow_dataset_batcher.cpp
    src/core/model_builder.cpp
    src/core/data_registry_core.cpp
    src/core/data_registry_utils.cpp
    src/core/dataset_base.cpp
    src/core/parquet_backed_dataset.cpp
)

target_include_directories(test_text_gui_training_launch PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${Stb_INCLUDE_DIR}
)

target_link_libraries(test_text_gui_training_launch PRIVATE
    Arrow::arrow_shared
    Parquet::parquet_shared
    imgui::imgui
    spdlog::spdlog
    fmt::fmt
    nlohmann_json::nlohmann_json
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_text_gui_training_launch PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_text_gui_training_launch PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Training Batcher Setup Helper =====
add_executable(test_training_batcher_setup
    tests/test_training_batcher_setup.cpp
    src/core/training_batcher_setup.cpp
    src/core/arrow_dataset.cpp
    src/core/arrow_dataset_batcher.cpp
    src/core/parquet_backed_dataset.cpp
    src/core/parquet_arrow_batcher.cpp
    src/core/dataset_base.cpp
)

target_include_directories(test_training_batcher_setup PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_training_batcher_setup PRIVATE
    Arrow::arrow_shared
    Parquet::parquet_shared
    spdlog::spdlog
    fmt::fmt
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_training_batcher_setup PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_training_batcher_setup PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Graph Topology Utilities =====
add_executable(test_graph_topology_utils
    tests/test_graph_topology_utils.cpp
    src/core/graph_topology_utils.cpp
)

target_include_directories(test_graph_topology_utils PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_graph_topology_utils PRIVATE
    imgui::imgui
)

if(WIN32)
    target_compile_definitions(test_graph_topology_utils PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_graph_topology_utils PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Text Arrow Adapter =====
add_executable(test_text_arrow_adapter
    tests/test_text_arrow_adapter.cpp
    src/core/text_arrow_adapter.cpp
    src/core/formats/text_dataset.cpp
    src/core/dataset_base.cpp
    src/core/node_executors/text_tokenizer_operator.cpp
)

target_include_directories(test_text_arrow_adapter PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_text_arrow_adapter PRIVATE
    Arrow::arrow_shared
    spdlog::spdlog
    fmt::fmt
    nlohmann_json::nlohmann_json
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_text_arrow_adapter PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_text_arrow_adapter PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Text Dataset Batcher Arrow Delegation =====
add_executable(test_text_dataset_batcher_arrow
    tests/test_text_dataset_batcher_arrow.cpp
    src/core/text_dataset_batcher.cpp
    src/core/text_arrow_adapter.cpp
    src/core/formats/text_dataset.cpp
    src/core/dataset_base.cpp
    src/core/node_executors/text_tokenizer_operator.cpp
    src/core/arrow_dataset.cpp
    src/core/arrow_dataset_batcher.cpp
)

target_include_directories(test_text_dataset_batcher_arrow PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_text_dataset_batcher_arrow PRIVATE
    Arrow::arrow_shared
    Parquet::parquet_shared
    spdlog::spdlog
    fmt::fmt
    nlohmann_json::nlohmann_json
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_text_dataset_batcher_arrow PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_text_dataset_batcher_arrow PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Text Preflight Node Validation =====
add_executable(test_text_preflight_nodes
    tests/test_text_preflight_nodes.cpp
    src/core/preflight_validator.cpp
)

target_include_directories(test_text_preflight_nodes PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_text_preflight_nodes PRIVATE
    imgui::imgui
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_text_preflight_nodes PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_text_preflight_nodes PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Saved Text Smoke Graph Patterns =====
add_executable(test_text_smoke_graph_patterns
    tests/test_text_smoke_graph_patterns.cpp
)

target_include_directories(test_text_smoke_graph_patterns PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_text_smoke_graph_patterns PRIVATE
    imgui::imgui
    nlohmann_json::nlohmann_json
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_text_smoke_graph_patterns PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_text_smoke_graph_patterns PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Label Column Resolver =====
add_executable(test_label_column_resolver
    tests/test_label_column_resolver.cpp
)

target_include_directories(test_label_column_resolver PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_label_column_resolver PRIVATE
    Arrow::arrow_shared
)

if(WIN32)
    target_compile_definitions(test_label_column_resolver PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_label_column_resolver PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== ASan Memory Test (standalone, no vcpkg deps) =====
if(CYXWIZ_ENABLE_ASAN)
    add_executable(test_asan_memory
        tests/test_asan_memory.cpp
    )

    target_include_directories(test_asan_memory PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    # Apply ASan to this target
    target_enable_asan(test_asan_memory)

    if(WIN32)
        target_compile_definitions(test_asan_memory PRIVATE
            _CRT_SECURE_NO_WARNINGS
            NOMINMAX
        )
    endif()

    set_target_properties(test_asan_memory PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    message(STATUS "ASan memory test enabled - run test_asan_memory to verify")
endif()
