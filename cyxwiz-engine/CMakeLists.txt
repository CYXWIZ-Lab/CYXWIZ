cmake_minimum_required(VERSION 3.20)

project(cyxwiz-engine)

option(CYXWIZ_ENABLE_TRACY "Enable optional Tracy profiler integration when Tracy is installed" OFF)

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

if(CYXWIZ_ENABLE_TRACY)
    find_package(Tracy CONFIG QUIET)
    if(Tracy_FOUND)
        message(STATUS "Tracy found - CyxWiz semantic profiling zones enabled")
    else()
        message(WARNING "CYXWIZ_ENABLE_TRACY=ON but Tracy was not found - profiling zones compile as no-ops")
    endif()
endif()

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
    src/core/data_convert_service.cpp
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
    src/core/pipeline_runtime_capabilities.cpp  # Data Studio runtime support truth
    src/core/data_studio_execution_plan.cpp  # Track 22 typed Data Studio plan
    src/core/pipeline_table_materializer.cpp  # Cat-1 graph walk over an Arrow table
    src/core/pipeline_materializer.cpp  # Cat-1 IPipelineOperator integration into training launch
    src/core/materialization_cache.cpp  # Persistent prepared dataset materialization cache
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
    src/core/debug_graph_trace_executor.cpp
    src/core/debug_operator_trace_adapter.cpp
    src/core/debug_operator_trace_producer.cpp
    src/core/preflight_validator.cpp
    src/core/crash_run_recorder.cpp
    src/core/text_preprocessing_tracer.cpp
    src/core/smoke_run_executor.cpp
    src/core/debug_recommendation_engine.cpp
    src/core/training_trace_collector.cpp
    src/core/debug_run_store.cpp
    src/core/graph_topology_utils.cpp
    src/core/training_batcher_setup.cpp
    src/core/sequence_arrow_batcher.cpp
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
    src/core/node_executors/tree_classification_utils.cpp    # Tool-to-Node Band 1
    src/core/node_executors/tree_model_artifact.cpp          # Tool-to-Node Band 1
    src/core/node_executors/decision_tree_model.cpp          # Tool-to-Node Band 1
    src/core/node_executors/decision_tree_trainer.cpp        # Tool-to-Node Band 1
    src/core/node_executors/decision_tree_operator.cpp       # Tool-to-Node Band 1
    src/core/node_executors/random_forest_model.cpp          # Tool-to-Node Band 1
    src/core/node_executors/random_forest_trainer.cpp        # Tool-to-Node Band 1
    src/core/node_executors/random_forest_operator.cpp       # Tool-to-Node Band 1
    src/core/node_executors/gradient_boosting_model.cpp      # Tool-to-Node Band 1
    src/core/node_executors/gradient_boosting_trainer.cpp    # Tool-to-Node Band 1
    src/core/node_executors/gradient_boosting_operator.cpp   # Tool-to-Node Band 1
    src/core/node_executors/tree_model_predictor_operator.cpp # Tool-to-Node Band 1
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
    src/core/language_model_generation.cpp
    src/core/language_model_training.cpp
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
    src/gui/properties_contract.cpp
    src/gui/properties_executor.cpp
    src/gui/properties_metadata_editor.cpp
    src/gui/properties_node_editors.cpp
    src/gui/properties_data_node_editors.cpp
    src/gui/properties_plugin_node_editors.cpp
    src/gui/properties_parameter_rules.cpp
    src/gui/properties_presets.cpp
    src/gui/properties_shape_info.cpp
    src/gui/properties_truth.cpp
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
    src/gui/panels/toolbar_search.cpp
    src/gui/panels/toolbar_file_dialogs.cpp
    src/gui/panels/toolbar_command_palette.cpp
    src/gui/panels/toolbar_file_menu.cpp
    src/gui/panels/toolbar_edit_menu.cpp
    src/gui/panels/toolbar_view_menu.cpp
    src/gui/panels/toolbar_profile_menu.cpp
    src/gui/panels/toolbar_other_menus.cpp
    src/gui/panels/toolbar_tools_menu.cpp
    src/gui/panels/toolbar_dialogs.cpp
    src/gui/panels/toolbar_account_dialogs.cpp
    src/gui/panels/toolbar_preferences_dialog.cpp
    src/gui/panels/toolbar_search_dialogs.cpp
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
    src/gui/panels/script_editor_search.cpp
    src/gui/panels/script_editor_settings.cpp
    src/gui/panels/script_editor_cells.cpp
    src/gui/panels/script_editor_debug.cpp
    src/gui/panels/script_editor_completion.cpp
    src/gui/panels/script_editor_language.cpp
    src/gui/panels/script_editor_save.cpp
    src/gui/panels/script_editor_execution.cpp
    src/gui/panels/script_editor_files.cpp
    src/gui/panels/script_editor_file_workflow.cpp
    src/gui/panels/table_viewer.cpp
    src/gui/panels/table_viewer_clipboard.cpp
    src/gui/panels/table_viewer_dialogs.cpp
    src/gui/panels/table_viewer_editing.cpp
    src/gui/panels/table_viewer_visualization.cpp
    src/gui/panels/table_viewer_stats.cpp
    src/gui/panels/table_viewer_loading.cpp
    src/gui/panels/table_viewer_tabs.cpp
    src/gui/panels/table_viewer_transforms.cpp
    src/gui/panels/table_viewer_plot.cpp
    src/gui/panels/table_viewer_context_menus.cpp
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
    src/inference/metric_learning_inference_input.cpp
    src/inference/text_inference_input.cpp
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
    src/gui/panels/language_model_generation_panel.cpp
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
    src/gui/theme_nodes.cpp
    src/gui/theme_dock.cpp
    src/gui/theme_presets.cpp
    src/gui/dock_style.cpp
    # Data Studio (Phase 1 Week 1)
    # src/gui/data_studio/pipeline_canvas.cpp  # Unified Canvas Phase 5: Replaced by Node Editor
    src/gui/data_studio/query_editor.cpp
    src/gui/data_studio/query_editor_render.cpp
    src/gui/data_studio/analyzer.cpp
    src/gui/data_studio/analyzer_render.cpp
    src/gui/data_studio/visualizer.cpp
    src/gui/data_studio/visualizer_render.cpp
    src/gui/data_studio/data_studio_panel.cpp
    src/gui/data_studio/data_studio_panel_render.cpp
    # pipeline_executor.cpp moved to src/core/ (Unified Canvas Phase 2)
    src/gui/data_studio/node_registry.cpp
    src/gui/data_studio/node_registry_builtin.cpp
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
    src/core/data_studio_execution_plan.h
    src/core/node_metadata.h       # Node Browser: node metadata structures
    src/core/node_metadata_registry.h  # Node Browser: metadata registry
    src/core/texture_manager.h
    src/core/async_task_manager.h
    src/core/graph_compiler.h
    src/core/dataset_batcher.h
    src/core/metric_learning_batch.h
    src/core/metric_learning_batcher.h
    src/core/metric_learning_dataset_builder.h
    src/core/metric_learning_graph_contract.h
    src/core/metric_learning_inference_outputs.h
    src/core/metric_learning_losses.h
    src/core/metric_learning_metrics.h
    src/core/metric_learning_shared_encoder.h
    src/core/metric_learning_training_step.h
    src/core/ner_sequence_builder.h
    src/core/sequence_arrow_batcher.h
    src/core/sequence_model_input.h
    src/core/sequence_training_step.h
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
    src/core/node_executors/tree_classification_utils.h
    src/core/node_executors/tree_model_artifact.h
    src/core/node_executors/decision_tree_model.h
    src/core/node_executors/decision_tree_trainer.h
    src/core/node_executors/decision_tree_operator.h
    src/core/node_executors/random_forest_model.h
    src/core/node_executors/random_forest_trainer.h
    src/core/node_executors/random_forest_operator.h
    src/core/node_executors/gradient_boosting_model.h
    src/core/node_executors/gradient_boosting_trainer.h
    src/core/node_executors/gradient_boosting_operator.h
    src/core/node_executors/tree_model_predictor_operator.h
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
    src/inference/metric_learning_inference_input.h
    src/inference/text_inference_input.h
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
    src/gui/panels/language_model_generation_panel.h
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

if(CYXWIZ_ENABLE_TRACY AND Tracy_FOUND)
    target_link_libraries(cyxwiz-engine PRIVATE Tracy::TracyClient)
    target_compile_definitions(cyxwiz-engine PRIVATE CYXWIZ_HAS_TRACY)
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

# Copy runtime scripts to build directory. Avoid copying token-metadata/node_modules:
# it is not a runtime asset for the engine and makes MSBuild post-build copies
# pathologically slow.
add_custom_command(TARGET cyxwiz-engine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
    $<TARGET_FILE_DIR:cyxwiz-engine>/scripts
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_SOURCE_DIR}/scripts/build.bat
    ${CMAKE_SOURCE_DIR}/scripts/build.sh
    ${CMAKE_SOURCE_DIR}/scripts/rebuild.cmd
    ${CMAKE_SOURCE_DIR}/scripts/rebuild.sh
    ${CMAKE_SOURCE_DIR}/scripts/setup-onnxruntime-gpu.ps1
    ${CMAKE_SOURCE_DIR}/scripts/setup-onnxruntime-gpu.sh
    ${CMAKE_SOURCE_DIR}/scripts/setup.bat
    ${CMAKE_SOURCE_DIR}/scripts/setup.sh
    $<TARGET_FILE_DIR:cyxwiz-engine>/scripts
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/scripts/startup
    $<TARGET_FILE_DIR:cyxwiz-engine>/scripts/startup
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/scripts/templates
    $<TARGET_FILE_DIR:cyxwiz-engine>/scripts/templates
    COMMAND ${CMAKE_COMMAND} -E make_directory
    $<TARGET_FILE_DIR:cyxwiz-engine>/scripts/token-metadata
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_SOURCE_DIR}/scripts/token-metadata/README.md
    ${CMAKE_SOURCE_DIR}/scripts/token-metadata/add-metadata.mjs
    ${CMAKE_SOURCE_DIR}/scripts/token-metadata/airdrop.mjs
    ${CMAKE_SOURCE_DIR}/scripts/token-metadata/check-metadata.mjs
    ${CMAKE_SOURCE_DIR}/scripts/token-metadata/metadata.json
    ${CMAKE_SOURCE_DIR}/scripts/token-metadata/package-lock.json
    ${CMAKE_SOURCE_DIR}/scripts/token-metadata/package.json
    ${CMAKE_SOURCE_DIR}/scripts/token-metadata/update-metadata-uri.mjs
    ${CMAKE_SOURCE_DIR}/scripts/token-metadata/upload-image.mjs
    ${CMAKE_SOURCE_DIR}/scripts/token-metadata/upload-imgbb.ps1
    $<TARGET_FILE_DIR:cyxwiz-engine>/scripts/token-metadata
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
    cyxwiz-backend
)

target_compile_definitions(cyxwiz_plotting PRIVATE CYXWIZ_PLOTTING_MODULE)

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
    src/core/executable_model.cpp
    src/core/graph_executable_model.cpp
    src/core/model_builder.cpp
    src/core/checkpoint_manager.cpp
    src/core/training_batcher_setup.cpp
    src/core/synthetic_batch.cpp
    src/core/debug_executor.cpp
    src/core/node_executors/text_tokenizer_operator.cpp
    src/core/text_arrow_adapter.cpp
    src/core/text_dataset_batcher.cpp
    src/core/parquet_backed_dataset.cpp
    src/core/parquet_arrow_batcher.cpp
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

# ===== Test: Siamese Backend Smoke =====
add_executable(test_siamese_backend_smoke
    tests/test_siamese_backend_smoke.cpp
)

target_link_libraries(test_siamese_backend_smoke PRIVATE
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_siamese_backend_smoke PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_siamese_backend_smoke PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Metric-Learning Batch Contracts =====
add_executable(test_metric_learning_batch_contracts
    tests/test_metric_learning_batch_contracts.cpp
)

target_include_directories(test_metric_learning_batch_contracts PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_metric_learning_batch_contracts PRIVATE
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_metric_learning_batch_contracts PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_metric_learning_batch_contracts PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Metric-Learning Shared Encoder Contracts =====
add_executable(test_metric_learning_shared_encoder_contracts
    tests/test_metric_learning_shared_encoder_contracts.cpp
    src/core/executable_model.cpp
)

target_include_directories(test_metric_learning_shared_encoder_contracts PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_metric_learning_shared_encoder_contracts PRIVATE
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_metric_learning_shared_encoder_contracts PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_metric_learning_shared_encoder_contracts PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Metric-Learning Metrics =====
add_executable(test_metric_learning_metrics
    tests/test_metric_learning_metrics.cpp
)

target_include_directories(test_metric_learning_metrics PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_metric_learning_metrics PRIVATE
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_metric_learning_metrics PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_metric_learning_metrics PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Metric-Learning Inference Outputs =====
add_executable(test_metric_learning_inference_outputs
    tests/test_metric_learning_inference_outputs.cpp
)

target_include_directories(test_metric_learning_inference_outputs PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_metric_learning_inference_outputs PRIVATE
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_metric_learning_inference_outputs PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_metric_learning_inference_outputs PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Metric-Learning Inference Input Parsing =====
add_executable(test_metric_learning_inference_input
    tests/test_metric_learning_inference_input.cpp
    src/inference/metric_learning_inference_input.cpp
)

target_include_directories(test_metric_learning_inference_input PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_metric_learning_inference_input PRIVATE
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_metric_learning_inference_input PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_metric_learning_inference_input PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Metric-Learning Loss Contracts =====
add_executable(test_metric_learning_losses
    tests/test_metric_learning_losses.cpp
)

target_include_directories(test_metric_learning_losses PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_metric_learning_losses PRIVATE
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_metric_learning_losses PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_metric_learning_losses PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: ML Primitive CPU/GPU Coverage Matrix =====
add_executable(test_ml_primitive_coverage_matrix
    tests/test_ml_primitive_coverage_matrix.cpp
)

target_include_directories(test_ml_primitive_coverage_matrix PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_ml_primitive_coverage_matrix PRIVATE
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_ml_primitive_coverage_matrix PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_ml_primitive_coverage_matrix PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Metric-Learning Training Step Contracts =====
add_executable(test_metric_learning_training_step
    tests/test_metric_learning_training_step.cpp
    src/core/executable_model.cpp
)

target_include_directories(test_metric_learning_training_step PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_metric_learning_training_step PRIVATE
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_metric_learning_training_step PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_metric_learning_training_step PROPERTIES
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

# ===== Test: Computation Truth - TF-IDF + Loss Parity =====
add_executable(test_computation_truth_tfidf_loss
    tests/computation_truth/test_computation_truth_tfidf_loss.cpp
    src/core/node_executors/tfidf_vectorizer_operator.cpp
)

target_include_directories(test_computation_truth_tfidf_loss PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_computation_truth_tfidf_loss PRIVATE
    Arrow::arrow_shared
    spdlog::spdlog
    fmt::fmt
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_computation_truth_tfidf_loss PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_computation_truth_tfidf_loss PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Computation Truth - Transformer Primitive Parity =====
add_executable(test_computation_truth_transformer_primitives
    tests/computation_truth/test_computation_truth_transformer_primitives.cpp
    src/core/language_model_training.cpp
    src/core/transformer_primitive_contracts.cpp
)

target_include_directories(test_computation_truth_transformer_primitives PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_computation_truth_transformer_primitives PRIVATE
    spdlog::spdlog
    fmt::fmt
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_computation_truth_transformer_primitives PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

if(CYXWIZ_HAS_PYTORCH)
    target_link_libraries(test_computation_truth_transformer_primitives PRIVATE
        ${TORCH_LIBRARIES}
    )
    target_include_directories(test_computation_truth_transformer_primitives PRIVATE
        ${TORCH_INCLUDE_DIRS}
    )
    target_compile_definitions(test_computation_truth_transformer_primitives PRIVATE
        CYXWIZ_HAS_PYTORCH
    )
endif()

set_target_properties(test_computation_truth_transformer_primitives PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Language Model Generation Controls =====
add_executable(test_language_model_generation
    tests/test_language_model_generation.cpp
    src/core/language_model_generation.cpp
)

target_include_directories(test_language_model_generation PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_language_model_generation PRIVATE
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_language_model_generation PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_language_model_generation PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Pipeline Operator Configure Reset Semantics =====
add_executable(test_operator_configure_resets
    tests/test_operator_configure_resets.cpp
    src/core/node_executors/clustering_operators.cpp
    src/core/node_executors/count_vectorizer_operator.cpp
    src/core/node_executors/pca_operator.cpp
    src/core/node_executors/time_series_features_operator.cpp
    src/core/node_executors/time_series_window_operator.cpp
)

target_include_directories(test_operator_configure_resets PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_operator_configure_resets PRIVATE
    Arrow::arrow_shared
    spdlog::spdlog
    fmt::fmt
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_operator_configure_resets PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_operator_configure_resets PROPERTIES
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

# ===== Test: Decision Tree Operator =====
add_executable(test_decision_tree_operator
    tests/test_decision_tree_operator.cpp
    src/core/node_executors/tree_classification_utils.cpp
    src/core/node_executors/tree_model_artifact.cpp
    src/core/node_executors/decision_tree_model.cpp
    src/core/node_executors/decision_tree_trainer.cpp
    src/core/node_executors/random_forest_model.cpp
    src/core/node_executors/gradient_boosting_model.cpp
    src/core/node_executors/decision_tree_operator.cpp
)

target_include_directories(test_decision_tree_operator PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_decision_tree_operator PRIVATE
    Arrow::arrow_shared
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    fmt::fmt
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_decision_tree_operator PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_decision_tree_operator PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Random Forest Operator =====
add_executable(test_random_forest_operator
    tests/test_random_forest_operator.cpp
    src/core/node_executors/tree_classification_utils.cpp
    src/core/node_executors/tree_model_artifact.cpp
    src/core/node_executors/decision_tree_model.cpp
    src/core/node_executors/decision_tree_trainer.cpp
    src/core/node_executors/random_forest_model.cpp
    src/core/node_executors/random_forest_trainer.cpp
    src/core/node_executors/gradient_boosting_model.cpp
    src/core/node_executors/random_forest_operator.cpp
)

target_include_directories(test_random_forest_operator PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_random_forest_operator PRIVATE
    Arrow::arrow_shared
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    fmt::fmt
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_random_forest_operator PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_random_forest_operator PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Gradient Boosting Operator =====
add_executable(test_gradient_boosting_operator
    tests/test_gradient_boosting_operator.cpp
    src/core/node_executors/tree_classification_utils.cpp
    src/core/node_executors/tree_model_artifact.cpp
    src/core/node_executors/decision_tree_model.cpp
    src/core/node_executors/random_forest_model.cpp
    src/core/node_executors/gradient_boosting_model.cpp
    src/core/node_executors/gradient_boosting_trainer.cpp
    src/core/node_executors/gradient_boosting_operator.cpp
)

target_include_directories(test_gradient_boosting_operator PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_gradient_boosting_operator PRIVATE
    Arrow::arrow_shared
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    fmt::fmt
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_gradient_boosting_operator PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_gradient_boosting_operator PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Tree Model Artifact Round Trip =====
add_executable(test_tree_model_artifact
    tests/test_tree_model_artifact.cpp
    src/core/node_executors/tree_model_artifact.cpp
    src/core/node_executors/decision_tree_model.cpp
    src/core/node_executors/decision_tree_trainer.cpp
    src/core/node_executors/random_forest_model.cpp
    src/core/node_executors/random_forest_trainer.cpp
    src/core/node_executors/gradient_boosting_model.cpp
    src/core/node_executors/gradient_boosting_trainer.cpp
)

target_include_directories(test_tree_model_artifact PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_tree_model_artifact PRIVATE
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    fmt::fmt
)

if(WIN32)
    target_compile_definitions(test_tree_model_artifact PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_tree_model_artifact PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Tree Model Predictor Operator =====
add_executable(test_tree_model_predictor_operator
    tests/test_tree_model_predictor_operator.cpp
    src/core/node_executors/tree_classification_utils.cpp
    src/core/node_executors/tree_model_artifact.cpp
    src/core/node_executors/decision_tree_model.cpp
    src/core/node_executors/decision_tree_trainer.cpp
    src/core/node_executors/random_forest_model.cpp
    src/core/node_executors/random_forest_trainer.cpp
    src/core/node_executors/gradient_boosting_model.cpp
    src/core/node_executors/gradient_boosting_trainer.cpp
    src/core/node_executors/tree_model_predictor_operator.cpp
)

target_include_directories(test_tree_model_predictor_operator PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_tree_model_predictor_operator PRIVATE
    Arrow::arrow_shared
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    fmt::fmt
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_tree_model_predictor_operator PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_tree_model_predictor_operator PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Pipeline Operator Metadata Drift Guard =====
add_executable(test_pipeline_operator_metadata
    tests/test_pipeline_operator_metadata.cpp
    src/core/data_analyzer.cpp
    src/data/data_table.cpp
    src/core/node_metadata_registry.cpp
    src/core/node_executors/pipeline_operator_factory.cpp
    src/core/pipeline_runtime_capabilities.cpp
    src/gui/properties_contract.cpp
    src/gui/properties_truth.cpp
    src/gui/data_studio/node_registry.cpp
    src/gui/data_studio/node_registry_builtin.cpp
    src/core/node_executors/clustering_operators.cpp
    src/core/node_executors/tree_classification_utils.cpp
    src/core/node_executors/tree_model_artifact.cpp
    src/core/node_executors/decision_tree_model.cpp
    src/core/node_executors/decision_tree_trainer.cpp
    src/core/node_executors/decision_tree_operator.cpp
    src/core/node_executors/random_forest_model.cpp
    src/core/node_executors/random_forest_trainer.cpp
    src/core/node_executors/random_forest_operator.cpp
    src/core/node_executors/gradient_boosting_model.cpp
    src/core/node_executors/gradient_boosting_trainer.cpp
    src/core/node_executors/gradient_boosting_operator.cpp
    src/core/node_executors/tree_model_predictor_operator.cpp
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
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imnodes
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

# ===== Test: Legacy PipelineExecutor operator routing =====
add_executable(test_pipeline_executor_operator_routing
    tests/test_pipeline_executor_operator_routing.cpp
    src/core/arrow_dataset.cpp
    src/core/data_registry.cpp
    src/core/data_registry_core.cpp
    src/core/data_registry_utils.cpp
    src/core/data_convert_service.cpp
    src/core/data_analyzer.cpp
    src/core/dataset_base.cpp
    src/core/duckdb_connector.cpp
    src/core/image_utils.cpp
    src/core/parquet_backed_dataset.cpp
    src/core/pipeline_executor.cpp
    src/core/pipeline_materializer.cpp
    src/core/materialization_cache.cpp
    src/core/pipeline_table_materializer.cpp
    src/core/pipeline_runtime_capabilities.cpp
    src/core/datasets/image_csv_dataset.cpp
    src/core/datasets/kaggle_dataset.cpp
    src/core/datasets/streaming_dataset.cpp
    src/core/node_executors/pipeline_operator_factory.cpp
    src/core/node_executors/clustering_operators.cpp
    src/core/node_executors/tree_classification_utils.cpp
    src/core/node_executors/tree_model_artifact.cpp
    src/core/node_executors/decision_tree_model.cpp
    src/core/node_executors/decision_tree_trainer.cpp
    src/core/node_executors/decision_tree_operator.cpp
    src/core/node_executors/random_forest_model.cpp
    src/core/node_executors/random_forest_trainer.cpp
    src/core/node_executors/random_forest_operator.cpp
    src/core/node_executors/gradient_boosting_model.cpp
    src/core/node_executors/gradient_boosting_trainer.cpp
    src/core/node_executors/gradient_boosting_operator.cpp
    src/core/node_executors/tree_model_predictor_operator.cpp
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
    src/data/data_table.cpp
    src/plotting/stb_image_impl.cpp
    src/plugin/registries/plugin_data_loader_registry.cpp
)

target_include_directories(test_pipeline_executor_operator_routing PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_pipeline_executor_operator_routing PRIVATE
    cyxwiz-backend
    Arrow::arrow_shared
    Parquet::parquet_shared
    duckdb
    nlohmann_json::nlohmann_json
    ${OpenCV_LIBS}
    spdlog::spdlog
    fmt::fmt
)

if(WIN32)
    target_compile_definitions(test_pipeline_executor_operator_routing PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_pipeline_executor_operator_routing PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Pipeline operator executor/materializer parity =====
add_executable(test_pipeline_operator_materializer_parity
    tests/test_pipeline_operator_materializer_parity.cpp
    src/core/arrow_dataset.cpp
    src/core/data_registry.cpp
    src/core/data_registry_core.cpp
    src/core/data_registry_utils.cpp
    src/core/data_convert_service.cpp
    src/core/data_analyzer.cpp
    src/core/dataset_base.cpp
    src/core/duckdb_connector.cpp
    src/core/image_utils.cpp
    src/core/parquet_backed_dataset.cpp
    src/core/pipeline_executor.cpp
    src/core/pipeline_materializer.cpp
    src/core/materialization_cache.cpp
    src/core/pipeline_table_materializer.cpp
    src/core/pipeline_runtime_capabilities.cpp
    src/core/datasets/image_csv_dataset.cpp
    src/core/datasets/kaggle_dataset.cpp
    src/core/datasets/streaming_dataset.cpp
    src/core/node_executors/pipeline_operator_factory.cpp
    src/core/node_executors/clustering_operators.cpp
    src/core/node_executors/tree_classification_utils.cpp
    src/core/node_executors/tree_model_artifact.cpp
    src/core/node_executors/decision_tree_model.cpp
    src/core/node_executors/decision_tree_trainer.cpp
    src/core/node_executors/decision_tree_operator.cpp
    src/core/node_executors/random_forest_model.cpp
    src/core/node_executors/random_forest_trainer.cpp
    src/core/node_executors/random_forest_operator.cpp
    src/core/node_executors/gradient_boosting_model.cpp
    src/core/node_executors/gradient_boosting_trainer.cpp
    src/core/node_executors/gradient_boosting_operator.cpp
    src/core/node_executors/tree_model_predictor_operator.cpp
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
    src/data/data_table.cpp
    src/plotting/stb_image_impl.cpp
    src/plugin/registries/plugin_data_loader_registry.cpp
)

target_include_directories(test_pipeline_operator_materializer_parity PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_pipeline_operator_materializer_parity PRIVATE
    cyxwiz-backend
    Arrow::arrow_shared
    Parquet::parquet_shared
    duckdb
    nlohmann_json::nlohmann_json
    ${OpenCV_LIBS}
    spdlog::spdlog
    fmt::fmt
)

if(WIN32)
    target_compile_definitions(test_pipeline_operator_materializer_parity PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_pipeline_operator_materializer_parity PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: DataConvert upstream dataset input =====
add_executable(test_data_convert_pipeline_input
    tests/test_data_convert_pipeline_input.cpp
    src/core/arrow_dataset.cpp
    src/core/data_registry.cpp
    src/core/data_registry_core.cpp
    src/core/data_registry_utils.cpp
    src/core/data_convert_service.cpp
    src/core/data_analyzer.cpp
    src/core/dataset_base.cpp
    src/core/duckdb_connector.cpp
    src/core/image_utils.cpp
    src/core/parquet_backed_dataset.cpp
    src/core/pipeline_executor.cpp
    src/core/pipeline_materializer.cpp
    src/core/materialization_cache.cpp
    src/core/pipeline_table_materializer.cpp
    src/core/pipeline_runtime_capabilities.cpp
    src/core/datasets/image_csv_dataset.cpp
    src/core/datasets/kaggle_dataset.cpp
    src/core/datasets/streaming_dataset.cpp
    src/core/node_executors/pipeline_operator_factory.cpp
    src/core/node_executors/clustering_operators.cpp
    src/core/node_executors/tree_classification_utils.cpp
    src/core/node_executors/tree_model_artifact.cpp
    src/core/node_executors/decision_tree_model.cpp
    src/core/node_executors/decision_tree_trainer.cpp
    src/core/node_executors/decision_tree_operator.cpp
    src/core/node_executors/random_forest_model.cpp
    src/core/node_executors/random_forest_trainer.cpp
    src/core/node_executors/random_forest_operator.cpp
    src/core/node_executors/gradient_boosting_model.cpp
    src/core/node_executors/gradient_boosting_trainer.cpp
    src/core/node_executors/gradient_boosting_operator.cpp
    src/core/node_executors/tree_model_predictor_operator.cpp
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
    src/data/data_table.cpp
    src/plotting/stb_image_impl.cpp
    src/plugin/registries/plugin_data_loader_registry.cpp
)

target_include_directories(test_data_convert_pipeline_input PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_data_convert_pipeline_input PRIVATE
    cyxwiz-backend
    Arrow::arrow_shared
    Parquet::parquet_shared
    duckdb
    nlohmann_json::nlohmann_json
    ${OpenCV_LIBS}
    spdlog::spdlog
    fmt::fmt
)

if(WIN32)
    target_compile_definitions(test_data_convert_pipeline_input PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_data_convert_pipeline_input PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Data Studio typed execution plan =====
add_executable(test_data_studio_execution_plan
    tests/test_data_studio_execution_plan.cpp
    src/core/data_studio_execution_plan.cpp
    src/core/pipeline_runtime_capabilities.cpp
)

target_include_directories(test_data_studio_execution_plan PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_data_studio_execution_plan PRIVATE
    spdlog::spdlog
    fmt::fmt
)

if(WIN32)
    target_compile_definitions(test_data_studio_execution_plan PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_data_studio_execution_plan PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Pattern Template Guard =====
add_executable(test_pattern_template_guard
    tests/test_pattern_template_guard.cpp
    src/gui/patterns/pattern_library.cpp
    src/core/node_metadata_registry.cpp
    src/core/pipeline_runtime_capabilities.cpp
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
    src/core/pipeline_runtime_capabilities.cpp
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

# ===== Test: Graph Compiler Causal LM Shape Contract =====
add_executable(test_graph_compiler_causal_lm_shape
    tests/test_graph_compiler_causal_lm_shape.cpp
    src/core/graph_compiler.cpp
    src/core/compiled_graph_plan.cpp
    src/core/graph_topology_utils.cpp
    src/core/node_metadata_registry.cpp
    src/core/pipeline_runtime_capabilities.cpp
    src/core/data_registry_core.cpp
    src/core/data_registry_utils.cpp
    src/core/arrow_dataset.cpp
    src/core/parquet_backed_dataset.cpp
)

target_include_directories(test_graph_compiler_causal_lm_shape PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_graph_compiler_causal_lm_shape PRIVATE
    Arrow::arrow_shared
    Parquet::parquet_shared
    imgui::imgui
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    fmt::fmt
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_graph_compiler_causal_lm_shape PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_graph_compiler_causal_lm_shape PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Recurrent Backend Placement Preflight =====
add_executable(test_recurrent_backend_placement
    tests/test_recurrent_backend_placement.cpp
    src/core/graph_compiler.cpp
    src/core/compiled_graph_plan.cpp
    src/core/graph_topology_utils.cpp
    src/core/node_metadata_registry.cpp
    src/core/pipeline_runtime_capabilities.cpp
    src/core/data_registry_core.cpp
    src/core/data_registry_utils.cpp
    src/core/arrow_dataset.cpp
    src/core/parquet_backed_dataset.cpp
)

target_include_directories(test_recurrent_backend_placement PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_recurrent_backend_placement PRIVATE
    Arrow::arrow_shared
    Parquet::parquet_shared
    imgui::imgui
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    fmt::fmt
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_recurrent_backend_placement PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_recurrent_backend_placement PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Recurrent ArrayFire Profile Smoke =====
add_executable(test_recurrent_af_profile_smoke
    tests/test_recurrent_af_profile_smoke.cpp
)

target_link_libraries(test_recurrent_af_profile_smoke PRIVATE
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_recurrent_af_profile_smoke PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_recurrent_af_profile_smoke PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Tabular Loader Apply Context Guard =====
add_executable(test_tabular_loader_apply_context
    tests/test_tabular_loader_apply_context.cpp
)

target_include_directories(test_tabular_loader_apply_context PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_tabular_loader_apply_context PRIVATE
    imgui::imgui
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_tabular_loader_apply_context PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_tabular_loader_apply_context PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Text Loader CSV Preflight =====
add_executable(test_text_loader_csv_preflight
    tests/test_text_loader_csv_preflight.cpp
    src/core/arrow_dataset.cpp
)

target_include_directories(test_text_loader_csv_preflight PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_text_loader_csv_preflight PRIVATE
    Arrow::arrow_shared
    Parquet::parquet_shared
    spdlog::spdlog
    fmt::fmt
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_text_loader_csv_preflight PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_text_loader_csv_preflight PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Tensor Reshape Runtime Contract =====
add_executable(test_tensor_reshape_runtime_contract
    tests/test_tensor_reshape_runtime_contract.cpp
    src/core/model_builder.cpp
    src/core/executable_model.cpp
    src/core/graph_executable_model.cpp
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
    src/core/pipeline_runtime_capabilities.cpp
    src/core/node_executors/text_tokenizer_operator.cpp
    src/core/arrow_dataset.cpp
    src/core/arrow_dataset_batcher.cpp
    src/core/model_builder.cpp
    src/core/executable_model.cpp
    src/core/graph_executable_model.cpp
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
    src/core/pipeline_runtime_capabilities.cpp
    src/core/node_executors/text_tokenizer_operator.cpp
    src/core/arrow_dataset.cpp
    src/core/arrow_dataset_batcher.cpp
    src/core/model_builder.cpp
    src/core/executable_model.cpp
    src/core/graph_executable_model.cpp
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

# ===== Test: Graph Training Sequence Preflight =====
add_executable(test_graph_training_sequence_preflight
    tests/test_graph_training_sequence_preflight.cpp
    src/gui/graph_training_launcher.cpp
    src/gui/panels/training_plot_panel.cpp
    src/core/async_task_manager.cpp
    src/core/training_trace_collector.cpp
    src/core/crash_run_recorder.cpp
    src/core/pipeline_materializer.cpp
    src/core/materialization_cache.cpp
    src/core/pipeline_runtime_capabilities.cpp
    src/core/pipeline_table_materializer.cpp
    src/core/data_analyzer.cpp
    src/data/data_table.cpp
    src/core/node_executors/pipeline_operator_factory.cpp
    src/core/node_executors/clustering_operators.cpp
    src/core/node_executors/tree_classification_utils.cpp
    src/core/node_executors/tree_model_artifact.cpp
    src/core/node_executors/decision_tree_model.cpp
    src/core/node_executors/decision_tree_trainer.cpp
    src/core/node_executors/decision_tree_operator.cpp
    src/core/node_executors/random_forest_model.cpp
    src/core/node_executors/random_forest_trainer.cpp
    src/core/node_executors/random_forest_operator.cpp
    src/core/node_executors/gradient_boosting_model.cpp
    src/core/node_executors/gradient_boosting_trainer.cpp
    src/core/node_executors/gradient_boosting_operator.cpp
    src/core/node_executors/tree_model_predictor_operator.cpp
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
    src/core/arrow_dataset.cpp
    src/core/data_registry_core.cpp
    src/core/data_registry_utils.cpp
    src/core/dataset_base.cpp
    src/core/parquet_backed_dataset.cpp
)

target_include_directories(test_graph_training_sequence_preflight PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${Stb_INCLUDE_DIR}
)

target_link_libraries(test_graph_training_sequence_preflight PRIVATE
    Arrow::arrow_shared
    Parquet::parquet_shared
    imgui::imgui
    implot::implot
    spdlog::spdlog
    fmt::fmt
    nlohmann_json::nlohmann_json
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_graph_training_sequence_preflight PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
        CYXWIZ_PLOTTING_MODULE
    )
endif()

set_target_properties(test_graph_training_sequence_preflight PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Text GUI Training Launch Helper =====
add_executable(test_text_gui_training_launch
    tests/test_text_gui_training_launch.cpp
    src/gui/graph_training_launcher.cpp
    src/gui/panels/training_plot_panel.cpp
    src/core/async_task_manager.cpp
    src/core/training_trace_collector.cpp
    src/core/crash_run_recorder.cpp
    src/core/sequence_arrow_batcher.cpp
    src/core/pipeline_materializer.cpp
    src/core/materialization_cache.cpp
    src/core/pipeline_runtime_capabilities.cpp
    src/core/pipeline_table_materializer.cpp
    src/core/node_executors/text_tokenizer_operator.cpp
    src/core/arrow_dataset.cpp
    src/core/arrow_dataset_batcher.cpp
    src/core/model_builder.cpp
    src/core/executable_model.cpp
    src/core/graph_executable_model.cpp
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
    implot::implot
    spdlog::spdlog
    fmt::fmt
    nlohmann_json::nlohmann_json
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_text_gui_training_launch PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
        CYXWIZ_PLOTTING_MODULE
    )
endif()

set_target_properties(test_text_gui_training_launch PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: CyxModel Sequence Assets =====
add_executable(test_cyxmodel_sequence_assets
    tests/test_cyxmodel_sequence_assets.cpp
    src/core/formats/cyxmodel_format.cpp
)

target_include_directories(test_cyxmodel_sequence_assets PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_cyxmodel_sequence_assets PRIVATE
    spdlog::spdlog
    fmt::fmt
    nlohmann_json::nlohmann_json
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_cyxmodel_sequence_assets PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_cyxmodel_sequence_assets PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: CyxModel Generation Metadata =====
add_executable(test_cyxmodel_generation_metadata
    tests/test_cyxmodel_generation_metadata.cpp
    src/core/formats/cyxmodel_format.cpp
)

target_include_directories(test_cyxmodel_generation_metadata PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_cyxmodel_generation_metadata PRIVATE
    spdlog::spdlog
    fmt::fmt
    nlohmann_json::nlohmann_json
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_cyxmodel_generation_metadata PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_cyxmodel_generation_metadata PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: CyxModel Exporter Generation Metadata =====
add_executable(test_cyxmodel_exporter_generation_metadata
    tests/test_cyxmodel_exporter_generation_metadata.cpp
    src/core/model_exporter.cpp
    src/core/formats/cyxmodel_format.cpp
)

target_include_directories(test_cyxmodel_exporter_generation_metadata PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_cyxmodel_exporter_generation_metadata PRIVATE
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    fmt::fmt
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_cyxmodel_exporter_generation_metadata PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_cyxmodel_exporter_generation_metadata PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: CyxModel LayerNorm Graph Round Trip =====
add_executable(test_cyxmodel_layernorm_roundtrip
    tests/test_cyxmodel_layernorm_roundtrip.cpp
    src/core/model_exporter.cpp
    src/core/model_importer.cpp
    src/core/model_builder.cpp
    src/core/executable_model.cpp
    src/core/graph_executable_model.cpp
    src/core/graph_compiler.cpp
    src/core/compiled_graph_plan.cpp
    src/core/graph_topology_utils.cpp
    src/core/node_metadata_registry.cpp
    src/core/pipeline_runtime_capabilities.cpp
    src/core/data_registry_core.cpp
    src/core/data_registry_utils.cpp
    src/core/arrow_dataset.cpp
    src/core/parquet_backed_dataset.cpp
    src/core/formats/cyxmodel_format.cpp
    src/inference/text_inference_input.cpp
)

target_include_directories(test_cyxmodel_layernorm_roundtrip PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_cyxmodel_layernorm_roundtrip PRIVATE
    Arrow::arrow_shared
    Parquet::parquet_shared
    imgui::imgui
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    fmt::fmt
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_cyxmodel_layernorm_roundtrip PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_cyxmodel_layernorm_roundtrip PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: CyxModel MultiHeadAttention Graph Round Trip =====
add_executable(test_cyxmodel_mha_roundtrip
    tests/test_cyxmodel_mha_roundtrip.cpp
    src/core/model_exporter.cpp
    src/core/model_importer.cpp
    src/core/model_builder.cpp
    src/core/executable_model.cpp
    src/core/graph_executable_model.cpp
    src/core/graph_compiler.cpp
    src/core/compiled_graph_plan.cpp
    src/core/graph_topology_utils.cpp
    src/core/node_metadata_registry.cpp
    src/core/pipeline_runtime_capabilities.cpp
    src/core/data_registry_core.cpp
    src/core/data_registry_utils.cpp
    src/core/arrow_dataset.cpp
    src/core/parquet_backed_dataset.cpp
    src/core/formats/cyxmodel_format.cpp
    src/inference/text_inference_input.cpp
)

target_include_directories(test_cyxmodel_mha_roundtrip PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_cyxmodel_mha_roundtrip PRIVATE
    Arrow::arrow_shared
    Parquet::parquet_shared
    imgui::imgui
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    fmt::fmt
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_cyxmodel_mha_roundtrip PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_cyxmodel_mha_roundtrip PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: CyxModel TransformerEncoder Graph Round Trip =====
add_executable(test_cyxmodel_transformer_encoder_roundtrip
    tests/test_cyxmodel_transformer_encoder_roundtrip.cpp
    src/core/model_exporter.cpp
    src/core/model_importer.cpp
    src/core/model_builder.cpp
    src/core/executable_model.cpp
    src/core/graph_executable_model.cpp
    src/core/graph_compiler.cpp
    src/core/compiled_graph_plan.cpp
    src/core/graph_topology_utils.cpp
    src/core/node_metadata_registry.cpp
    src/core/pipeline_runtime_capabilities.cpp
    src/core/data_registry_core.cpp
    src/core/data_registry_utils.cpp
    src/core/arrow_dataset.cpp
    src/core/parquet_backed_dataset.cpp
    src/core/formats/cyxmodel_format.cpp
)

target_include_directories(test_cyxmodel_transformer_encoder_roundtrip PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_cyxmodel_transformer_encoder_roundtrip PRIVATE
    Arrow::arrow_shared
    Parquet::parquet_shared
    imgui::imgui
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    fmt::fmt
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_cyxmodel_transformer_encoder_roundtrip PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_cyxmodel_transformer_encoder_roundtrip PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: CyxModel TransformerDecoder Graph Round Trip =====
add_executable(test_cyxmodel_transformer_decoder_roundtrip
    tests/test_cyxmodel_transformer_decoder_roundtrip.cpp
    src/core/model_exporter.cpp
    src/core/model_importer.cpp
    src/core/model_builder.cpp
    src/core/executable_model.cpp
    src/core/graph_executable_model.cpp
    src/core/graph_compiler.cpp
    src/core/compiled_graph_plan.cpp
    src/core/graph_topology_utils.cpp
    src/core/node_metadata_registry.cpp
    src/core/pipeline_runtime_capabilities.cpp
    src/core/data_registry_core.cpp
    src/core/data_registry_utils.cpp
    src/core/arrow_dataset.cpp
    src/core/parquet_backed_dataset.cpp
    src/core/formats/cyxmodel_format.cpp
)

target_include_directories(test_cyxmodel_transformer_decoder_roundtrip PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_cyxmodel_transformer_decoder_roundtrip PRIVATE
    Arrow::arrow_shared
    Parquet::parquet_shared
    imgui::imgui
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    fmt::fmt
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_cyxmodel_transformer_decoder_roundtrip PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_cyxmodel_transformer_decoder_roundtrip PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: CyxModel Causal LM Generation Round Trip =====
add_executable(test_cyxmodel_causal_lm_generation_roundtrip
    tests/test_cyxmodel_causal_lm_generation_roundtrip.cpp
    src/core/model_exporter.cpp
    src/core/model_importer.cpp
    src/core/model_builder.cpp
    src/core/executable_model.cpp
    src/core/graph_executable_model.cpp
    src/core/graph_compiler.cpp
    src/core/compiled_graph_plan.cpp
    src/core/graph_topology_utils.cpp
    src/core/node_metadata_registry.cpp
    src/core/pipeline_runtime_capabilities.cpp
    src/core/data_registry_core.cpp
    src/core/data_registry_utils.cpp
    src/core/arrow_dataset.cpp
    src/core/parquet_backed_dataset.cpp
    src/core/formats/cyxmodel_format.cpp
    src/inference/text_inference_input.cpp
)

target_include_directories(test_cyxmodel_causal_lm_generation_roundtrip PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_cyxmodel_causal_lm_generation_roundtrip PRIVATE
    Arrow::arrow_shared
    Parquet::parquet_shared
    imgui::imgui
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    fmt::fmt
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_cyxmodel_causal_lm_generation_roundtrip PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_cyxmodel_causal_lm_generation_roundtrip PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Saved NER Sequence Smoke =====
add_executable(test_saved_ner_sequence_smoke
    tests/test_saved_ner_sequence_smoke.cpp
    src/core/graph_compiler.cpp
    src/core/compiled_graph_plan.cpp
    src/core/graph_topology_utils.cpp
    src/core/node_metadata_registry.cpp
    src/core/pipeline_runtime_capabilities.cpp
    src/core/data_registry_core.cpp
    src/core/data_registry_utils.cpp
    src/core/sequence_arrow_batcher.cpp
    src/core/training_executor.cpp
    src/core/training_batcher_setup.cpp
    src/core/arrow_dataset.cpp
    src/core/arrow_dataset_batcher.cpp
    src/core/model_builder.cpp
    src/core/executable_model.cpp
    src/core/graph_executable_model.cpp
    src/core/parquet_backed_dataset.cpp
    src/core/parquet_arrow_batcher.cpp
    src/core/dataset_base.cpp
    src/core/checkpoint_manager.cpp
    src/core/crash_run_recorder.cpp
    src/core/training_trace_collector.cpp
    src/core/formats/cyxmodel_format.cpp
    src/plugin/registries/plugin_training_hook_manager.cpp
)

target_include_directories(test_saved_ner_sequence_smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_saved_ner_sequence_smoke PRIVATE
    Arrow::arrow_shared
    Parquet::parquet_shared
    imgui::imgui
    spdlog::spdlog
    fmt::fmt
    nlohmann_json::nlohmann_json
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_saved_ner_sequence_smoke PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
        CYXWIZ_TRAINING_EXECUTOR_MODERN_ONLY
    )
else()
    target_compile_definitions(test_saved_ner_sequence_smoke PRIVATE
        CYXWIZ_TRAINING_EXECUTOR_MODERN_ONLY
    )
endif()

set_target_properties(test_saved_ner_sequence_smoke PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Training Batcher Setup Helper =====
add_executable(test_training_batcher_setup
    tests/test_training_batcher_setup.cpp
    src/core/training_batcher_setup.cpp
    src/core/arrow_dataset.cpp
    src/core/arrow_dataset_batcher.cpp
    src/core/model_builder.cpp
    src/core/executable_model.cpp
    src/core/graph_executable_model.cpp
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

# ===== Test: DataConvert CSV -> Parquet =====
add_executable(test_data_convert_service
    tests/test_data_convert_service.cpp
    src/core/data_convert_service.cpp
    src/core/arrow_dataset.cpp
)

target_include_directories(test_data_convert_service PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/cyxwiz-backend/include
)

target_link_libraries(test_data_convert_service PRIVATE
    Arrow::arrow_shared
    Parquet::parquet_shared
    spdlog::spdlog
    fmt::fmt
    nlohmann_json::nlohmann_json
)

if(HighFive_FOUND)
    target_link_libraries(test_data_convert_service PRIVATE HighFive)
    target_compile_definitions(test_data_convert_service PRIVATE CYXWIZ_HAS_HDF5)
endif()

if(WIN32)
    target_compile_definitions(test_data_convert_service PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_data_convert_service PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Stable error-code catalog and formatting =====
add_executable(test_error_codes
    tests/test_error_codes.cpp
)

target_include_directories(test_error_codes PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/cyxwiz-backend/include
)

if(WIN32)
    target_compile_definitions(test_error_codes PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_error_codes PROPERTIES
    CXX_STANDARD 20
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: TrainingExecutor Arrow/Parquet Parity =====
add_executable(test_training_executor_arrow_parquet
    tests/test_training_executor_arrow_parquet.cpp
    src/core/training_executor.cpp
    src/core/training_batcher_setup.cpp
    src/core/arrow_dataset.cpp
    src/core/arrow_dataset_batcher.cpp
    src/core/model_builder.cpp
    src/core/executable_model.cpp
    src/core/graph_executable_model.cpp
    src/core/parquet_backed_dataset.cpp
    src/core/parquet_arrow_batcher.cpp
    src/core/dataset_base.cpp
    src/core/checkpoint_manager.cpp
    src/core/crash_run_recorder.cpp
    src/core/training_trace_collector.cpp
    src/plugin/registries/plugin_training_hook_manager.cpp
)

target_include_directories(test_training_executor_arrow_parquet PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_training_executor_arrow_parquet PRIVATE
    Arrow::arrow_shared
    Parquet::parquet_shared
    spdlog::spdlog
    fmt::fmt
    nlohmann_json::nlohmann_json
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_training_executor_arrow_parquet PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
        CYXWIZ_TRAINING_EXECUTOR_MODERN_ONLY
    )
else()
    target_compile_definitions(test_training_executor_arrow_parquet PRIVATE
        CYXWIZ_TRAINING_EXECUTOR_MODERN_ONLY
    )
endif()

set_target_properties(test_training_executor_arrow_parquet PROPERTIES
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

# ===== Test: Studio Debugger Contracts =====
add_executable(test_debugger_contracts
    tests/test_debugger_contracts.cpp
    src/core/debug_export_correlation_tracer.cpp
    src/core/debug_graph_trace_executor.cpp
    src/core/debug_memory_ownership_tracer.cpp
    src/core/debug_node_inspector.cpp
    src/core/debug_operator_trace_adapter.cpp
    src/core/debug_operator_trace_producer.cpp
    src/core/debug_runtime_backend_classifier.cpp
    src/core/debug_smoke_sample_selector.cpp
    src/core/debug_support_bundle_builder.cpp
    src/core/debug_windows_crash_importer.cpp
    src/core/debug_session_manager.cpp
    src/core/debug_recommendation_engine.cpp
    src/core/debug_run_store.cpp
    src/core/node_executors/text_tokenizer_operator.cpp
    src/core/preflight_validator.cpp
    src/core/text_preprocessing_tracer.cpp
    src/core/formats/text_dataset.cpp
    src/core/dataset_base.cpp
)

target_include_directories(test_debugger_contracts PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_debugger_contracts PRIVATE
    Arrow::arrow_shared
    imgui::imgui
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    fmt::fmt
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_debugger_contracts PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_debugger_contracts PROPERTIES
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

# ===== Test: Properties Truth Resolver =====
add_executable(test_properties_truth
    tests/test_properties_truth.cpp
    src/gui/properties_truth.cpp
)

target_include_directories(test_properties_truth PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_properties_truth PRIVATE
    imgui::imgui
)

if(WIN32)
    target_compile_definitions(test_properties_truth PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_properties_truth PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ===== Test: Pipeline Materializer Cache Integration =====
add_executable(test_pipeline_materializer_cache
    tests/test_pipeline_materializer_cache.cpp
    src/core/pipeline_materializer.cpp
    src/core/materialization_cache.cpp
    src/core/pipeline_runtime_capabilities.cpp
    src/core/pipeline_table_materializer.cpp
    src/core/node_executors/text_tokenizer_operator.cpp
    src/core/arrow_dataset.cpp
    src/core/data_registry_core.cpp
    src/core/data_registry_utils.cpp
    src/core/dataset_base.cpp
    src/core/parquet_backed_dataset.cpp
)

target_include_directories(test_pipeline_materializer_cache PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${Stb_INCLUDE_DIR}
)

target_link_libraries(test_pipeline_materializer_cache PRIVATE
    Arrow::arrow_shared
    Parquet::parquet_shared
    spdlog::spdlog
    fmt::fmt
    nlohmann_json::nlohmann_json
    cyxwiz-backend
)

if(WIN32)
    target_compile_definitions(test_pipeline_materializer_cache PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_pipeline_materializer_cache PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
# ===== Test: Materialization Cache =====
add_executable(test_materialization_cache
    tests/test_materialization_cache.cpp
    src/core/materialization_cache.cpp
)

target_include_directories(test_materialization_cache PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_materialization_cache PRIVATE
    Arrow::arrow_shared
)

if(WIN32)
    target_compile_definitions(test_materialization_cache PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

set_target_properties(test_materialization_cache PROPERTIES
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










