// Windows header order fix - must come first to prevent winsock conflicts
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

#include "main_window.h"
#include "node_editor.h"
#include "console.h"
#include "viewport.h"
#include "properties.h"
#include "dock_style.h"
#include "icons.h"
#include "theme.h"
#include "../core/keyboard_shortcuts.h"
#include "panels/toolbar.h"
#include "panels/asset_browser.h"
#include "panels/training_dashboard.h"
#include "panels/training_plot_panel.h"
#include "panels/plot_test_control.h"
#include "panels/command_window.h"
#include "panels/script_editor.h"
#include "panels/table_viewer.h"
#include "panels/data_explorer_panel.h"
#include "panels/annotation_editor_panel.h"
#include "panels/visualization_panel.h"
#include "panels/connection_dialog.h"
#include "panels/job_status_panel.h"
#include "panels/p2p_training_panel.h"
#include "panels/wallet_panel.h"
#include "panels/task_progress_panel.h"
#include "panels/pattern_browser.h"
#include "panels/query_console.h"
#include "panels/custom_node_editor.h"
#include "panels/theme_editor.h"
#include "panels/profiling_panel.h"
#include "panels/memory_panel.h"
#include "panels/memory_monitor.h"
#include "panels/variable_explorer.h"
#include "panels/plot_output_panel.h"
#include "panels/python_plot_window_registry.h"
#include "panels/test_results_panel.h"
#include "panels/export_dialog.h"
#include "panels/import_dialog.h"
#include "panels/deployment_dialog.h"
#include "panels/model_summary_panel.h"
#include "panels/architecture_diagram.h"
#include "panels/lr_finder_panel.h"
#include "panels/data_profiler_panel.h"
#include "panels/correlation_matrix_panel.h"
#include "panels/missing_value_panel.h"
#include "panels/outlier_detection_panel.h"
#include "panels/descriptive_stats_panel.h"
#include "panels/hypothesis_test_panel.h"
#include "panels/distribution_fitter_panel.h"
#include "panels/regression_panel.h"
#include "panels/dim_reduction_panel.h"
#include "panels/gradcam_panel.h"
#include "panels/feature_importance_panel.h"
#include "panels/nas_panel.h"
#include "panels/hyperparam_search_panel.h"
#include "panels/serving_panel.h"
#include "panels/dnn_inference_panel.h"
#include "panels/kmeans_panel.h"
#include "panels/dbscan_panel.h"
#include "panels/hierarchical_panel.h"
#include "panels/gmm_panel.h"
#include "panels/cluster_eval_panel.h"
#include "panels/confusion_matrix_panel.h"
#include "panels/roc_auc_panel.h"
#include "panels/pr_curve_panel.h"
#include "panels/cross_validation_panel.h"
#include "panels/learning_curves_panel.h"
#include "panels/normalization_panel.h"
#include "panels/standardization_panel.h"
#include "panels/log_transform_panel.h"
#include "panels/boxcox_panel.h"
#include "panels/feature_scaling_panel.h"
#include "panels/optimizer_settings_panel.h"
#include "panels/matrix_calculator_panel.h"
#include "panels/eigen_decomp_panel.h"
#include "panels/svd_panel.h"
#include "panels/qr_panel.h"
#include "panels/cholesky_panel.h"
#include "panels/fft_panel.h"
#include "panels/spectrogram_panel.h"
#include "panels/filter_designer_panel.h"
#include "panels/convolution_panel.h"
#include "panels/wavelet_panel.h"
#include "panels/gradient_descent_panel.h"
#include "panels/convexity_panel.h"
#include "panels/lp_panel.h"
#include "panels/qp_panel.h"
#include "panels/differentiation_panel.h"
#include "panels/integration_panel.h"
#include "panels/decomposition_panel.h"
#include "panels/acf_pacf_panel.h"
#include "panels/stationarity_panel.h"
#include "panels/seasonality_panel.h"
#include "panels/forecasting_panel.h"
#include "panels/tokenization_panel.h"
#include "panels/word_frequency_panel.h"
#include "panels/tfidf_panel.h"
#include "panels/embeddings_panel.h"
#include "panels/sentiment_panel.h"
// Utilities panels (Phase 12)
#include "panels/calculator_panel.h"
#include "panels/unit_converter_panel.h"
#include "panels/random_generator_panel.h"
#include "panels/hash_generator_panel.h"
#include "panels/json_viewer_panel.h"
#include "panels/regex_tester_panel.h"
#include "panels/cloud_browser.h"
#include "panels/node_browser_panel.h"
#include "panels/node_info_panel.h"
#include "../core/node_metadata.h"
#include "panels/cloud_dataset_manager.h"
#include "panels/plugin_manager_panel.h"
#include "data_studio/data_studio_panel.h"
#include "../plugin/plugin_manager.h"
#include "../plugin/registries/plugin_panel_registry.h"
#include "tutorial/tutorial_system.h"
#include "../scripting/scripting_engine.h"
#include "../scripting/startup_script_manager.h"
#include "../network/job_manager.h"
#include "../network/grpc_client.h"
#include "../network/reservation_client.h"
#include "../network/p2p_client.h"
#include "../network/datastream_client.h"
#include "../auth/auth_client.h"
#include "../core/project_manager.h"
#include "../core/engine_config.h"
#include "../core/data_registry.h"
#include "../core/parquet_backed_dataset.h"
#include "../core/graph_compiler.h"
#include "../core/training_executor.h"
#include "../core/training_manager.h"
#include "../core/test_manager.h"
#include "../core/model_converter.h"
#include "../serving/inference_server.h"

#include <imgui.h>
#include <imgui_internal.h>
#include <cyxwiz/cyxwiz.h>
#include <spdlog/spdlog.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>

#include "../core/file_dialogs.h"

namespace gui {

MainWindow::MainWindow()
    : show_about_dialog_(false), show_demo_window_(false), first_time_layout_(true) {

    // Original panels
    node_editor_ = std::make_unique<NodeEditor>();
    console_ = std::make_unique<Console>();
    viewport_ = std::make_unique<Viewport>();
    properties_ = std::make_unique<Properties>();

    // Initialize scripting engine (shared resource)
    scripting_engine_ = std::make_shared<scripting::ScriptingEngine>();

    // New panel system
    toolbar_ = std::make_unique<cyxwiz::ToolbarPanel>();
    asset_browser_ = std::make_unique<cyxwiz::AssetBrowserPanel>();
    training_plot_panel_ = std::make_unique<cyxwiz::TrainingPlotPanel>();  // Now named "Training Dashboard"

    plot_test_control_ = std::make_unique<cyxwiz::PlotTestControlPanel>();
    command_window_ = std::make_unique<cyxwiz::CommandWindowPanel>();
    script_editor_ = std::make_unique<cyxwiz::ScriptEditorPanel>();
    table_viewer_ = std::make_unique<cyxwiz::TableViewerPanel>();
    data_explorer_panel_ = std::make_unique<cyxwiz::DataExplorerPanel>();
    annotation_editor_panel_ = std::make_unique<cyxwiz::AnnotationEditorPanel>();
    visualization_panel_ = std::make_unique<cyxwiz::VisualizationPanel>();
    job_status_panel_ = std::make_unique<cyxwiz::JobStatusPanel>();
    p2p_training_panel_ = std::make_unique<cyxwiz::P2PTrainingPanel>();
    wallet_panel_ = std::make_unique<gui::WalletPanel>();
    task_progress_panel_ = std::make_unique<cyxwiz::TaskProgressPanel>();
    pattern_browser_ = std::make_unique<cyxwiz::PatternBrowserPanel>();
    query_console_ = std::make_unique<cyxwiz::QueryConsolePanel>();
    custom_node_editor_ = std::make_unique<gui::CustomNodeEditorPanel>();
    theme_editor_ = std::make_unique<gui::ThemeEditorPanel>();
    profiling_panel_ = std::make_unique<cyxwiz::ProfilingPanel>();
    memory_panel_ = std::make_unique<cyxwiz::MemoryPanel>();
    memory_monitor_ = std::make_unique<cyxwiz::MemoryMonitor>();
    variable_explorer_ = std::make_unique<cyxwiz::VariableExplorerPanel>();
    variable_explorer_->SetScriptingEngine(scripting_engine_);

    // Plot output panel (like MATLAB's figure window)
    plot_output_panel_ = std::make_unique<cyxwiz::PlotOutputPanel>();
    plot_output_panel_->SetScriptingEngine(scripting_engine_);

    // Test results panel
    test_results_panel_ = std::make_unique<cyxwiz::TestResultsPanel>();

    // Export/Import dialogs
    export_dialog_ = std::make_unique<cyxwiz::ExportDialog>();
    import_dialog_ = std::make_unique<cyxwiz::ImportDialog>();
    deployment_dialog_ = std::make_unique<cyxwiz::DeploymentDialog>();

    // Model Analysis panels (Phase 2)
    model_summary_panel_ = std::make_unique<cyxwiz::ModelSummaryPanel>();
    model_summary_panel_->SetNodeEditor(node_editor_.get());

    architecture_diagram_ = std::make_unique<cyxwiz::ArchitectureDiagram>();
    architecture_diagram_->SetNodeEditor(node_editor_.get());

    lr_finder_panel_ = std::make_unique<cyxwiz::LRFinderPanel>();
    lr_finder_panel_->SetNodeEditor(node_editor_.get());

    // Data Science panels (Phase 3)
    data_profiler_panel_ = std::make_unique<cyxwiz::DataProfilerPanel>();
    correlation_matrix_panel_ = std::make_unique<cyxwiz::CorrelationMatrixPanel>();
    missing_value_panel_ = std::make_unique<cyxwiz::MissingValuePanel>();
    outlier_detection_panel_ = std::make_unique<cyxwiz::OutlierDetectionPanel>();

    // Statistics panels (Phase 4)
    descriptive_stats_panel_ = std::make_unique<cyxwiz::DescriptiveStatsPanel>();
    hypothesis_test_panel_ = std::make_unique<cyxwiz::HypothesisTestPanel>();
    distribution_fitter_panel_ = std::make_unique<cyxwiz::DistributionFitterPanel>();
    regression_panel_ = std::make_unique<cyxwiz::RegressionPanel>();

    // Wire Data Explorer Hub to analysis panels
    if (data_explorer_panel_) {
        data_explorer_panel_->SetPanelReferences(
            descriptive_stats_panel_.get(),
            correlation_matrix_panel_.get(),
            regression_panel_.get(),
            outlier_detection_panel_.get(),
            missing_value_panel_.get(),
            data_profiler_panel_.get()
        );
        data_explorer_panel_->SetVisualizationPanel(visualization_panel_.get());
    }

    // Connect TableViewer to VisualizationPanel
    if (table_viewer_ && visualization_panel_) {
        table_viewer_->SetVisualizationPanel(visualization_panel_.get());
    }

    // Advanced Tools panels (Phase 5)
    dim_reduction_panel_ = std::make_unique<cyxwiz::DimReductionPanel>();
    gradcam_panel_ = std::make_unique<cyxwiz::GradCAMPanel>();
    feature_importance_panel_ = std::make_unique<cyxwiz::FeatureImportancePanel>();
    nas_panel_ = std::make_unique<cyxwiz::NASPanel>();
    hyperparam_search_panel_ = std::make_unique<cyxwiz::HyperparamSearchPanel>();
    serving_panel_ = std::make_unique<cyxwiz::ServingPanel>();
    dnn_inference_panel_ = std::make_unique<cyxwiz::DNNInferencePanel>();

    // Clustering panels (Phase 6A)
    kmeans_panel_ = std::make_unique<cyxwiz::KMeansPanel>();
    dbscan_panel_ = std::make_unique<cyxwiz::DBSCANPanel>();
    hierarchical_panel_ = std::make_unique<cyxwiz::HierarchicalPanel>();
    gmm_panel_ = std::make_unique<cyxwiz::GMMPanel>();
    cluster_eval_panel_ = std::make_unique<cyxwiz::ClusterEvalPanel>();

    // Model Evaluation panels (Phase 6B)
    confusion_matrix_panel_ = std::make_unique<cyxwiz::ConfusionMatrixPanel>();
    roc_auc_panel_ = std::make_unique<cyxwiz::ROCAUCPanel>();
    pr_curve_panel_ = std::make_unique<cyxwiz::PRCurvePanel>();
    cross_validation_panel_ = std::make_unique<cyxwiz::CrossValidationPanel>();
    learning_curves_panel_ = std::make_unique<cyxwiz::LearningCurvesPanel>();

    // Data Transformation panels (Phase 6C)
    normalization_panel_ = std::make_unique<cyxwiz::NormalizationPanel>();
    standardization_panel_ = std::make_unique<cyxwiz::StandardizationPanel>();
    log_transform_panel_ = std::make_unique<cyxwiz::LogTransformPanel>();
    boxcox_panel_ = std::make_unique<cyxwiz::BoxCoxPanel>();
    feature_scaling_panel_ = std::make_unique<cyxwiz::FeatureScalingPanel>();
    optimizer_settings_panel_ = std::make_unique<cyxwiz::OptimizerSettingsPanel>();

    // Linear Algebra panels (Phase 7)
    matrix_calculator_panel_ = std::make_unique<cyxwiz::MatrixCalculatorPanel>();
    eigen_decomp_panel_ = std::make_unique<cyxwiz::EigenDecompPanel>();
    svd_panel_ = std::make_unique<cyxwiz::SVDPanel>();
    qr_panel_ = std::make_unique<cyxwiz::QRPanel>();
    cholesky_panel_ = std::make_unique<cyxwiz::CholeskyPanel>();

    // Signal Processing panels (Phase 8)
    fft_panel_ = std::make_unique<cyxwiz::FFTPanel>();
    spectrogram_panel_ = std::make_unique<cyxwiz::SpectrogramPanel>();
    filter_designer_panel_ = std::make_unique<cyxwiz::FilterDesignerPanel>();
    convolution_panel_ = std::make_unique<cyxwiz::ConvolutionPanel>();
    wavelet_panel_ = std::make_unique<cyxwiz::WaveletPanel>();

    // Optimization & Calculus panels (Phase 9)
    gradient_descent_panel_ = std::make_unique<cyxwiz::GradientDescentPanel>();
    convexity_panel_ = std::make_unique<cyxwiz::ConvexityPanel>();
    lp_panel_ = std::make_unique<cyxwiz::LPPanel>();
    qp_panel_ = std::make_unique<cyxwiz::QPPanel>();
    differentiation_panel_ = std::make_unique<cyxwiz::DifferentiationPanel>();
    integration_panel_ = std::make_unique<cyxwiz::IntegrationPanel>();

    // Time Series Analysis panels (Phase 10)
    decomposition_panel_ = std::make_unique<cyxwiz::DecompositionPanel>();
    acf_pacf_panel_ = std::make_unique<cyxwiz::ACFPACFPanel>();
    stationarity_panel_ = std::make_unique<cyxwiz::StationarityPanel>();
    seasonality_panel_ = std::make_unique<cyxwiz::SeasonalityPanel>();
    forecasting_panel_ = std::make_unique<cyxwiz::ForecastingPanel>();

    // Text Processing panels (Phase 11)
    tokenization_panel_ = std::make_unique<cyxwiz::TokenizationPanel>();
    word_frequency_panel_ = std::make_unique<cyxwiz::WordFrequencyPanel>();
    tfidf_panel_ = std::make_unique<cyxwiz::TFIDFPanel>();
    embeddings_panel_ = std::make_unique<cyxwiz::EmbeddingsPanel>();
    sentiment_panel_ = std::make_unique<cyxwiz::SentimentPanel>();

    // Utilities panels (Phase 12)
    calculator_panel_ = std::make_unique<cyxwiz::CalculatorPanel>();
    unit_converter_panel_ = std::make_unique<cyxwiz::UnitConverterPanel>();
    random_generator_panel_ = std::make_unique<cyxwiz::RandomGeneratorPanel>();
    hash_generator_panel_ = std::make_unique<cyxwiz::HashGeneratorPanel>();
    json_viewer_panel_ = std::make_unique<cyxwiz::JSONViewerPanel>();
    regex_tester_panel_ = std::make_unique<cyxwiz::RegexTesterPanel>();
    
    // Cloud panels (DataStream)
    cloud_browser_panel_ = std::make_unique<gui::CloudBrowserPanel>();
    node_browser_panel_ = std::make_unique<gui::NodeBrowserPanel>();
    node_browser_panel_->SetNodeEditor(node_editor_.get());
    node_info_panel_ = std::make_unique<cyxwiz::NodeInfoPanel>();
    // Connect Node Browser hover to Info Panel
    if (node_browser_panel_ && node_info_panel_) {
        node_browser_panel_->SetNodeHoverCallback([this](cyxwiz::NodeType type) {
            node_info_panel_->SetSelectedNode(type);
        });
    }
    cloud_dataset_manager_panel_ = std::make_unique<gui::CloudDatasetManagerPanel>();

    // Plugin Manager panel
    plugin_manager_panel_ = std::make_unique<cyxwiz::PluginManagerPanel>();

    // Data Studio panel (Phase 1 Week 1)
    data_studio_panel_ = std::make_unique<cyxwiz::DataStudioPanel>();

    // Set NAS panel callbacks for node editor integration
    nas_panel_->SetGetArchitectureCallback([this]() -> std::pair<std::vector<MLNode>, std::vector<NodeLink>> {
        if (node_editor_) {
            return {node_editor_->GetNodes(), node_editor_->GetLinks()};
        }
        return {{}, {}};
    });

    nas_panel_->SetApplyArchitectureCallback([this](const std::vector<MLNode>& nodes, const std::vector<NodeLink>& links) {
        if (node_editor_) {
            node_editor_->InsertPattern(nodes, links);
        }
    });

    // Set pattern browser callback to insert patterns into node editor
    pattern_browser_->SetInsertCallback([this](const std::vector<MLNode>& nodes, const std::vector<NodeLink>& links) {
        if (node_editor_) {
            node_editor_->InsertPattern(nodes, links);
        }
    });

    // Set scripting engine for command window and script editor
    command_window_->SetScriptingEngine(scripting_engine_);
    script_editor_->SetScriptingEngine(scripting_engine_);

    // Expose TrainingPlotPanel to Python scripts through the scripting engine
    // Registration is deferred - actual Python import happens on first script execution
    if (scripting_engine_) {
        scripting_engine_->RegisterTrainingDashboard(training_plot_panel_.get());
    }

    // Connect Viewport to TrainingPlotPanel for real-time metrics display
    viewport_->SetTrainingPanel(training_plot_panel_.get());

    // Connect script editor to command window for output display
    script_editor_->SetCommandWindow(command_window_.get());

    // Connect Node Editor to Script Editor for code generation output
    node_editor_->SetScriptEditor(script_editor_.get());

    // Connect Node Editor to Properties panel for node selection display
    node_editor_->SetPropertiesPanel(properties_.get());

    // Connect Node Editor to ScriptingEngine for Python-based RL training
    node_editor_->SetScriptingEngine(scripting_engine_.get());

    // Connect Properties panel to Node Editor for shape inference
    properties_->SetNodeEditor(node_editor_.get());

    // Connect Pattern Browser to Node Editor for proper node creation
    pattern_browser_->SetNodeEditor(node_editor_.get());

    // Connect Query Console to Node Editor for graph queries
    query_console_->SetNodeEditor(node_editor_.get());

    // Set up training callback for Node Editor
    node_editor_->SetCompileCallback([this]() {
        this->CompileGraphAndReport();
    });

    node_editor_->SetTrainCallback([this](const std::vector<MLNode>& nodes, const std::vector<NodeLink>& links) {
        this->StartTrainingFromGraph(nodes, links);
        // Show Training Dashboard when training starts
        if (training_plot_panel_) {
            training_plot_panel_->SetVisible(true);
        }
    });

    // Set up callbacks in the toolbar
    toolbar_->SetResetLayoutCallback([this]() {
        this->ResetDockLayout();
    });

    toolbar_->SetSaveLayoutCallback([this]() {
        this->SaveLayout();
    });

    toolbar_->SetSaveProjectSettingsCallback([this]() {
        this->SaveProjectSettings();
    });

    // When app theme changes from View menu, save settings immediately
    toolbar_->SetAppThemeChangedCallback([this](int theme_index) {
        auto& pm = cyxwiz::ProjectManager::Instance();
        if (pm.HasActiveProject()) {
            pm.GetConfig().editor_settings.app_theme = theme_index;
            pm.SaveProject();
            spdlog::info("App theme saved to project: {}", theme_index);
        }
    });

    toolbar_->SetTogglePlotTestControlCallback([this]() {
        if (plot_test_control_) {
            plot_test_control_->Toggle();
        }
    });

    toolbar_->SetImportDatasetCallback([this]() {
        if (data_explorer_panel_) {
            data_explorer_panel_->SetVisible(true);
            spdlog::info("Opened Data Explorer panel");
        }
    });

    toolbar_->SetCreateCustomDatasetCallback([this]() {
        if (data_explorer_panel_) {
            data_explorer_panel_->SetVisible(true);
            spdlog::info("Opened Data Explorer panel");
        }
    });

    // Preprocess - opens Feature Scaling panel
    toolbar_->SetPreprocessDatasetCallback([this]() {
        if (feature_scaling_panel_) {
            feature_scaling_panel_->SetVisible(true);
            spdlog::info("Opened Feature Scaling panel for preprocessing");
        }
    });

    // Tokenize - opens existing TokenizationPanel
    toolbar_->SetTokenizeDatasetCallback([this]() {
        if (tokenization_panel_) {
            tokenization_panel_->SetVisible(true);
            spdlog::info("Opened Tokenization panel");
        }
    });

    toolbar_->SetAugmentDatasetCallback([this]() {
        if (node_editor_) {
            node_editor_->Show();
            spdlog::info("Opened Node Editor - use Augmentation nodes for data augmentation");
        }
    });

    toolbar_->SetDatasetStatisticsCallback([this]() {
        if (data_explorer_panel_) {
            data_explorer_panel_->SetVisible(true);
            spdlog::info("Opened Data Explorer panel for statistics");
        }
    });

    // Compile Graph - dry-run the graph compilation and show results in a popup
    toolbar_->SetCompileGraphCallback([this]() {
        CompileGraphAndReport();
    });

    // Start Training - compiles and runs training from node editor graph
    toolbar_->SetStartTrainingCallback([this]() {
        if (node_editor_) {
            auto nodes = node_editor_->GetNodes();
            auto links = node_editor_->GetLinks();
            StartTrainingFromGraph(nodes, links);
            if (training_plot_panel_) {
                training_plot_panel_->SetVisible(true);
            }
            spdlog::info("Started training from Train menu");
        }
    });

    // Pause Training
    toolbar_->SetPauseTrainingCallback([this]() {
        auto& tm = cyxwiz::TrainingManager::Instance();
        if (tm.IsTrainingActive()) {
            tm.PauseTraining();
            spdlog::info("Paused training");
        }
    });

    // Stop Training
    toolbar_->SetStopTrainingCallback([this]() {
        auto& tm = cyxwiz::TrainingManager::Instance();
        if (tm.IsTrainingActive()) {
            tm.StopTraining();
            spdlog::info("Stopped training");
        }
    });

    // Training Settings - opens Training Dashboard
    toolbar_->SetTrainingSettingsCallback([this]() {
        if (training_plot_panel_) {
            training_plot_panel_->SetVisible(true);
            spdlog::info("Opened Training Dashboard");
        }
    });

    // Optimizer Settings - opens dedicated Optimizer Settings panel
    toolbar_->SetOptimizerSettingsCallback([this]() {
        if (optimizer_settings_panel_) {
            optimizer_settings_panel_->SetVisible(true);
            spdlog::info("Opened Optimizer Settings panel");
        }
    });

    // Set up Run Test callback
    toolbar_->SetRunTestCallback([this]() {
        if (node_editor_) {
            auto nodes = node_editor_->GetNodes();
            auto links = node_editor_->GetLinks();
            StartTestingFromGraph(nodes, links);
        }
    });

    // Set up View Test Results callback
    toolbar_->SetViewTestResultsCallback([this]() {
        if (test_results_panel_) {
            test_results_panel_->Show();
            spdlog::info("Opened Test Results panel");
        }
    });

    // Set up Save Model callback (native CyxWiz format)
    toolbar_->SetSaveModelCallback([this]() {
        auto& tm = cyxwiz::TrainingManager::Instance();
        if (!tm.HasTrainedModel()) {
            spdlog::warn("No trained model available to save");
            return;
        }

        // Show save file dialog
        auto result = cyxwiz::FileDialogs::SaveModel();
        if (result) {
            std::string path = *result;
            // Remove extension if present (our save adds .json and .bin)
            if (path.size() > 9 && path.substr(path.size() - 9) == ".cyxmodel") {
                path = path.substr(0, path.size() - 9);
            }

            // Get model name from filename
            std::filesystem::path file_path(path);
            std::string model_name = file_path.stem().string();
            std::string model_desc = "Model trained with CyxWiz Engine";

            if (tm.SaveModel(path, model_name, model_desc)) {
                spdlog::info("Model saved successfully to: {}", path);
            } else {
                spdlog::error("Failed to save model to: {}", path);
            }
        }
    });

    // Set up Export Model callback
    toolbar_->SetExportModelCallback([this](int format_index) {
        if (export_dialog_) {
            // Get trained model from TrainingManager
            auto& tm = cyxwiz::TrainingManager::Instance();
            if (tm.HasTrainedModel()) {
                auto* model = tm.GetLastTrainedModel();
                auto* optimizer = tm.GetLastOptimizer();
                auto& metrics = tm.GetLastMetrics();

                // Get current graph JSON from node editor
                std::string graph_json;
                if (node_editor_) {
                    graph_json = node_editor_->GetGraphJson();
                }

                export_dialog_->SetModelData(model, optimizer, &metrics, graph_json);
                spdlog::info("Loaded trained model into Export dialog");
            } else {
                spdlog::warn("No trained model available for export");
            }

            export_dialog_->Open();
            spdlog::info("Opened Export Model dialog (format index: {})", format_index);
        }
    });

    // Set up Import Model callback
    toolbar_->SetImportModelCallback([this]() {
        if (import_dialog_) {
            import_dialog_->Open();
            spdlog::info("Opened Import Model dialog");
        }
    });

    // Set up Import Complete callback - load graph into node editor
    import_dialog_->SetImportCompleteCallback(
        [this](const cyxwiz::ImportResult& result, const std::string& graph_json) {
            if (result.success) {
                spdlog::info("Model imported successfully: {} ({} layers, {} params)",
                             result.model_name, result.num_layers, result.num_parameters);

                // Load graph into node editor if available
                if (!graph_json.empty() && node_editor_) {
                    if (node_editor_->LoadGraphFromString(graph_json)) {
                        node_editor_->Show();
                        spdlog::info("Loaded imported model graph into Node Editor");
                    } else {
                        spdlog::warn("Failed to load imported graph into Node Editor");
                    }
                }
            } else {
                spdlog::error("Model import failed: {}", result.error_message);
            }
        }
    );

    // Set up Model Conversion callbacks
    toolbar_->SetConvertBinaryToDirCallback([this]() {
        spdlog::info("Binary to Directory conversion requested");

        // Step 1: Select input binary file
        auto input_result = cyxwiz::FileDialogs::OpenModel();
        if (!input_result) {
            spdlog::info("Binary to Directory conversion cancelled");
            return;
        }

        std::string input_path = *input_result;

        // Verify it's a binary file
        if (!cyxwiz::ModelConverter::IsBinaryFormat(input_path)) {
            spdlog::error("Selected file is not a binary .cyxmodel format");
            return;
        }

        // Step 2: Select output folder
        auto output_result = cyxwiz::FileDialogs::SelectOutputFolder();
        if (!output_result) {
            spdlog::info("Binary to Directory conversion cancelled (output selection)");
            return;
        }

        // Generate output path
        std::filesystem::path in_path(input_path);
        std::string output_path = *output_result + "/" + in_path.stem().string() + ".cyxmodel";

        spdlog::info("Converting binary '{}' to directory '{}'", input_path, output_path);

        if (cyxwiz::ModelConverter::BinaryToDirectory(input_path, output_path)) {
            spdlog::info("Successfully converted to directory format: {}", output_path);
        } else {
            spdlog::error("Conversion failed: {}", cyxwiz::ModelConverter::GetLastError());
        }
    });

    toolbar_->SetConvertDirToBinaryCallback([this]() {
        spdlog::info("Directory to Binary conversion requested");

        // Step 1: Select input directory
        auto input_result = cyxwiz::FileDialogs::SelectFolder("Select .cyxmodel Directory");
        if (!input_result) {
            spdlog::info("Directory to Binary conversion cancelled");
            return;
        }

        std::string input_path = *input_result;

        // Verify it's a directory format
        if (!cyxwiz::ModelConverter::IsDirectoryFormat(input_path)) {
            spdlog::error("Selected path is not a valid .cyxmodel directory format");
            return;
        }

        // Step 2: Select output file
        auto output_result = cyxwiz::FileDialogs::SaveModel();
        if (!output_result) {
            spdlog::info("Directory to Binary conversion cancelled (output selection)");
            return;
        }

        std::string output_path = *output_result;

        spdlog::info("Converting directory '{}' to binary '{}'", input_path, output_path);

        if (cyxwiz::ModelConverter::DirectoryToBinary(input_path, output_path)) {
            spdlog::info("Successfully converted to binary format: {}", output_path);
        } else {
            spdlog::error("Conversion failed: {}", cyxwiz::ModelConverter::GetLastError());
        }
    });

    // Set up Deploy to Server callback
    toolbar_->SetDeployToServerCallback([this]() {
        if (deployment_dialog_) {
            deployment_dialog_->Open();
            spdlog::info("Opened Deployment dialog");
        }
    });

    // Set up Custom Node Editor callback
    toolbar_->SetOpenCustomNodeEditorCallback([this]() {
        if (custom_node_editor_) {
            custom_node_editor_->Show();
            spdlog::info("Opened Custom Node Editor panel");
        }
    });

    // Set up Icon Pack selection callbacks
    toolbar_->SetSetIconPackCallback([this](int pack) {
        if (node_editor_) {
            node_editor_->SetIconPack(static_cast<gui::IconPack>(pack));
            const char* pack_names[] = {"FontAwesome", "Tabler", "Remix", "Lucide", "Iconoir", "Phosphor"};
            spdlog::info("Changed icon pack to: {}", pack < 6 ? pack_names[pack] : "Unknown");
        }
    });
    toolbar_->SetGetIconPackCallback([this]() -> int {
        return node_editor_ ? static_cast<int>(node_editor_->GetIconPack()) : 0;
    });

    // Set up Node Editor operation callbacks (Nodes menu)
    toolbar_->SetAddDenseNodeCallback([this]() {
        if (node_editor_) {
            node_editor_->AddNodeFromMenu(gui::NodeType::Dense, "Dense (128)");
        }
    });
    toolbar_->SetAddConvNodeCallback([this]() {
        if (node_editor_) {
            node_editor_->AddNodeFromMenu(gui::NodeType::Conv2D, "Conv2D (32)");
        }
    });
    toolbar_->SetAddPoolingNodeCallback([this]() {
        if (node_editor_) {
            node_editor_->AddNodeFromMenu(gui::NodeType::MaxPool2D, "MaxPool2D");
        }
    });
    toolbar_->SetAddDropoutNodeCallback([this]() {
        if (node_editor_) {
            node_editor_->AddNodeFromMenu(gui::NodeType::Dropout, "Dropout (0.5)");
        }
    });
    toolbar_->SetAddBatchNormNodeCallback([this]() {
        if (node_editor_) {
            node_editor_->AddNodeFromMenu(gui::NodeType::BatchNorm, "BatchNorm");
        }
    });
    toolbar_->SetAddAttentionNodeCallback([this]() {
        if (node_editor_) {
            node_editor_->AddNodeFromMenu(gui::NodeType::MultiHeadAttention, "MultiHeadAttention");
        }
    });
    toolbar_->SetDuplicateNodesCallback([this]() {
        if (node_editor_) {
            node_editor_->DuplicateSelectedNodes();
        }
    });
    toolbar_->SetDeleteNodesCallback([this]() {
        if (node_editor_) {
            node_editor_->DeleteSelectedNodes();
        }
    });
    toolbar_->SetGroupNodesCallback([this]() {
        if (node_editor_) {
            node_editor_->GroupSelectedNodes();
        }
    });
    toolbar_->SetUngroupNodesCallback([this]() {
        if (node_editor_) {
            node_editor_->UngroupSelectedNodes();
        }
    });

    // Set up Theme Editor callback
    toolbar_->SetOpenThemeEditorCallback([this]() {
        if (theme_editor_) {
            theme_editor_->Show();
            spdlog::info("Opened Theme Editor panel");
        }
    });

    // Set up Profiler callback
    toolbar_->SetOpenProfilerCallback([this]() {
        if (profiling_panel_) {
            profiling_panel_->Show();
            spdlog::info("Opened Performance Profiler panel");
        }
    });

    // Set up Memory Monitor callback (System Monitor in Profile menu)
    toolbar_->SetOpenMemoryMonitorCallback([this]() {
        if (memory_monitor_) {
            memory_monitor_->Toggle();
            spdlog::info("Toggled Memory Monitor");
        }
    });

    // Set up Memory Panel callback (Memory Visualization in Profile menu)
    toolbar_->SetOpenMemoryPanelCallback([this]() {
        if (memory_panel_) {
            memory_panel_->Show();
            spdlog::info("Opened Memory Visualization panel");
        }
    });

    // Set up Compute Device selection callback (Preferences -> Device tab)
    toolbar_->SetComputeDeviceChangedCallback([](cyxwiz::DeviceType type, int device_id) {
        // Create device and set it as active
        cyxwiz::Device device(type, device_id);
        device.SetActive();

        // Log the change
        const char* type_str = "Unknown";
        switch (type) {
            case cyxwiz::DeviceType::CPU: type_str = "CPU"; break;
            case cyxwiz::DeviceType::CUDA: type_str = "CUDA"; break;
            case cyxwiz::DeviceType::OPENCL: type_str = "OpenCL"; break;
            case cyxwiz::DeviceType::METAL: type_str = "Metal"; break;
            case cyxwiz::DeviceType::VULKAN: type_str = "Vulkan"; break;
        }
        spdlog::info("Compute device changed to {} device {}", type_str, device_id);
    });

    // Set up Verbose Python Logging pointer (View menu - Developer Tools)
    if (scripting_engine_) {
        toolbar_->SetVerbosePythonLogPtr(scripting_engine_->GetVerboseLoggingPtr());
        toolbar_->SetPythonDiagnosticsCallback([this]() -> std::string {
            return scripting_engine_->GetPythonRuntimeDiagnostics();
        });
    }

    // Set up Model Analysis callbacks (Tools menu - Phase 2)
    toolbar_->SetOpenModelSummaryCallback([this]() {
        if (model_summary_panel_) {
            model_summary_panel_->Toggle();
            if (model_summary_panel_->IsVisible()) {
                model_summary_panel_->RefreshAnalysis();
            }
            spdlog::info("Toggled Model Summary panel");
        }
    });

    toolbar_->SetOpenArchitectureDiagramCallback([this]() {
        if (architecture_diagram_) {
            architecture_diagram_->Toggle();
            if (architecture_diagram_->IsVisible()) {
                architecture_diagram_->RefreshDiagram();
            }
            spdlog::info("Toggled Architecture Diagram panel");
        }
    });

    toolbar_->SetOpenLRFinderCallback([this]() {
        if (lr_finder_panel_) {
            lr_finder_panel_->Toggle();
            spdlog::info("Toggled Learning Rate Finder panel");
        }
    });

    // Set up Data Science callbacks (Tools menu - Phase 3)
    toolbar_->SetOpenDataProfilerCallback([this]() {
        if (data_profiler_panel_) {
            data_profiler_panel_->Toggle();
            spdlog::info("Toggled Data Profiler panel");
        }
    });

    toolbar_->SetOpenCorrelationMatrixCallback([this]() {
        if (correlation_matrix_panel_) {
            correlation_matrix_panel_->Toggle();
            spdlog::info("Toggled Correlation Matrix panel");
        }
    });

    toolbar_->SetOpenMissingValuePanelCallback([this]() {
        if (missing_value_panel_) {
            missing_value_panel_->Toggle();
            spdlog::info("Toggled Missing Value panel");
        }
    });

    toolbar_->SetOpenOutlierDetectionCallback([this]() {
        if (outlier_detection_panel_) {
            outlier_detection_panel_->Toggle();
            spdlog::info("Toggled Outlier Detection panel");
        }
    });

    // Set up Statistics callbacks (Tools menu - Phase 4)
    toolbar_->SetOpenDescriptiveStatsCallback([this]() {
        if (descriptive_stats_panel_) {
            descriptive_stats_panel_->Toggle();
            spdlog::info("Toggled Descriptive Statistics panel");
        }
    });

    toolbar_->SetOpenHypothesisTestCallback([this]() {
        if (hypothesis_test_panel_) {
            hypothesis_test_panel_->Toggle();
            spdlog::info("Toggled Hypothesis Testing panel");
        }
    });

    toolbar_->SetOpenDistributionFitterCallback([this]() {
        if (distribution_fitter_panel_) {
            distribution_fitter_panel_->Toggle();
            spdlog::info("Toggled Distribution Fitter panel");
        }
    });

    toolbar_->SetOpenRegressionCallback([this]() {
        if (regression_panel_) {
            regression_panel_->Toggle();
            spdlog::info("Toggled Regression Analysis panel");
        }
    });

    // Advanced Tools callbacks (Phase 5)
    toolbar_->SetOpenDimReductionCallback([this]() {
        if (dim_reduction_panel_) {
            dim_reduction_panel_->Toggle();
            spdlog::info("Toggled Dimensionality Reduction panel");
        }
    });

    toolbar_->SetOpenGradCAMCallback([this]() {
        if (gradcam_panel_) {
            gradcam_panel_->Toggle();
            spdlog::info("Toggled Grad-CAM panel");
        }
    });

    toolbar_->SetOpenFeatureImportanceCallback([this]() {
        if (feature_importance_panel_) {
            feature_importance_panel_->Toggle();
            spdlog::info("Toggled Feature Importance panel");
        }
    });

    toolbar_->SetOpenNASCallback([this]() {
        if (nas_panel_) {
            nas_panel_->Toggle();
            spdlog::info("Toggled Neural Architecture Search panel");
        }
    });

    toolbar_->SetOpenHyperparamSearchCallback([this]() {
        if (hyperparam_search_panel_) {
            hyperparam_search_panel_->Toggle();
            spdlog::info("Toggled Hyperparameter Search panel");
        }
    });

    toolbar_->SetOpenServingCallback([this]() {
        if (serving_panel_) {
            serving_panel_->Toggle();
            spdlog::info("Toggled Model Serving panel");
        }
    });

    toolbar_->SetOpenDNNInferenceCallback([this]() {
        if (dnn_inference_panel_) {
            dnn_inference_panel_->Toggle();
            spdlog::info("Toggled DNN Inference panel");
        }
    });

    // Clustering callbacks (Phase 6A)
    toolbar_->SetOpenKMeansCallback([this]() {
        if (kmeans_panel_) {
            kmeans_panel_->Toggle();
            spdlog::info("Toggled K-Means Clustering panel");
        }
    });

    toolbar_->SetOpenDBSCANCallback([this]() {
        if (dbscan_panel_) {
            dbscan_panel_->Toggle();
            spdlog::info("Toggled DBSCAN panel");
        }
    });

    toolbar_->SetOpenHierarchicalCallback([this]() {
        if (hierarchical_panel_) {
            hierarchical_panel_->Toggle();
            spdlog::info("Toggled Hierarchical Clustering panel");
        }
    });

    toolbar_->SetOpenGMMCallback([this]() {
        if (gmm_panel_) {
            gmm_panel_->Toggle();
            spdlog::info("Toggled GMM panel");
        }
    });

    toolbar_->SetOpenClusterEvalCallback([this]() {
        if (cluster_eval_panel_) {
            cluster_eval_panel_->Toggle();
            spdlog::info("Toggled Cluster Evaluation panel");
        }
    });

    // Model Evaluation callbacks (Phase 6B)
    toolbar_->SetOpenConfusionMatrixCallback([this]() {
        if (confusion_matrix_panel_) {
            confusion_matrix_panel_->Toggle();
            spdlog::info("Toggled Confusion Matrix panel");
        }
    });

    toolbar_->SetOpenROCAUCCallback([this]() {
        if (roc_auc_panel_) {
            roc_auc_panel_->Toggle();
            spdlog::info("Toggled ROC/AUC panel");
        }
    });

    toolbar_->SetOpenPRCurveCallback([this]() {
        if (pr_curve_panel_) {
            pr_curve_panel_->Toggle();
            spdlog::info("Toggled PR Curve panel");
        }
    });

    toolbar_->SetOpenCrossValidationCallback([this]() {
        if (cross_validation_panel_) {
            cross_validation_panel_->Toggle();
            spdlog::info("Toggled Cross-Validation panel");
        }
    });

    toolbar_->SetOpenLearningCurvesCallback([this]() {
        if (learning_curves_panel_) {
            learning_curves_panel_->Toggle();
            spdlog::info("Toggled Learning Curves panel");
        }
    });

    // Data Transformation callbacks (Phase 6C)
    toolbar_->SetOpenNormalizationCallback([this]() {
        if (normalization_panel_) {
            normalization_panel_->Toggle();
            spdlog::info("Toggled Normalization panel");
        }
    });

    toolbar_->SetOpenStandardizationCallback([this]() {
        if (standardization_panel_) {
            standardization_panel_->Toggle();
            spdlog::info("Toggled Standardization panel");
        }
    });

    toolbar_->SetOpenLogTransformCallback([this]() {
        if (log_transform_panel_) {
            log_transform_panel_->Toggle();
            spdlog::info("Toggled Log Transform panel");
        }
    });

    toolbar_->SetOpenBoxCoxCallback([this]() {
        if (boxcox_panel_) {
            boxcox_panel_->Toggle();
            spdlog::info("Toggled Box-Cox panel");
        }
    });

    toolbar_->SetOpenFeatureScalingCallback([this]() {
        if (feature_scaling_panel_) {
            feature_scaling_panel_->Toggle();
            spdlog::info("Toggled Feature Scaling panel");
        }
    });

    // Linear Algebra callbacks (Phase 7)
    toolbar_->SetOpenMatrixCalculatorCallback([this]() {
        if (matrix_calculator_panel_) {
            matrix_calculator_panel_->Toggle();
            spdlog::info("Toggled Matrix Calculator panel");
        }
    });

    toolbar_->SetOpenEigenDecompCallback([this]() {
        if (eigen_decomp_panel_) {
            eigen_decomp_panel_->Toggle();
            spdlog::info("Toggled Eigen Decomposition panel");
        }
    });

    toolbar_->SetOpenSVDCallback([this]() {
        if (svd_panel_) {
            svd_panel_->Toggle();
            spdlog::info("Toggled SVD panel");
        }
    });

    toolbar_->SetOpenQRCallback([this]() {
        if (qr_panel_) {
            qr_panel_->Toggle();
            spdlog::info("Toggled QR Decomposition panel");
        }
    });

    toolbar_->SetOpenCholeskyCallback([this]() {
        if (cholesky_panel_) {
            cholesky_panel_->Toggle();
            spdlog::info("Toggled Cholesky Decomposition panel");
        }
    });

    // Signal Processing callbacks (Phase 8)
    toolbar_->SetOpenFFTCallback([this]() {
        if (fft_panel_) {
            fft_panel_->Toggle();
            spdlog::info("Toggled FFT panel");
        }
    });
    toolbar_->SetOpenSpectrogramCallback([this]() {
        if (spectrogram_panel_) {
            spectrogram_panel_->Toggle();
            spdlog::info("Toggled Spectrogram panel");
        }
    });
    toolbar_->SetOpenFilterDesignerCallback([this]() {
        if (filter_designer_panel_) {
            filter_designer_panel_->Toggle();
            spdlog::info("Toggled Filter Designer panel");
        }
    });
    toolbar_->SetOpenConvolutionCallback([this]() {
        if (convolution_panel_) {
            convolution_panel_->Toggle();
            spdlog::info("Toggled Convolution panel");
        }
    });
    toolbar_->SetOpenWaveletCallback([this]() {
        if (wavelet_panel_) {
            wavelet_panel_->Toggle();
            spdlog::info("Toggled Wavelet Transform panel");
        }
    });

    // Optimization & Calculus callbacks (Phase 9)
    toolbar_->SetOpenGradientDescentCallback([this]() {
        if (gradient_descent_panel_) {
            gradient_descent_panel_->Toggle();
            spdlog::info("Toggled Gradient Descent panel");
        }
    });
    toolbar_->SetOpenConvexityCallback([this]() {
        if (convexity_panel_) {
            convexity_panel_->Toggle();
            spdlog::info("Toggled Convexity Analyzer panel");
        }
    });
    toolbar_->SetOpenLPCallback([this]() {
        if (lp_panel_) {
            lp_panel_->Toggle();
            spdlog::info("Toggled Linear Programming panel");
        }
    });
    toolbar_->SetOpenQPCallback([this]() {
        if (qp_panel_) {
            qp_panel_->Toggle();
            spdlog::info("Toggled Quadratic Programming panel");
        }
    });
    toolbar_->SetOpenDifferentiationCallback([this]() {
        if (differentiation_panel_) {
            differentiation_panel_->Toggle();
            spdlog::info("Toggled Numerical Differentiation panel");
        }
    });
    toolbar_->SetOpenIntegrationCallback([this]() {
        if (integration_panel_) {
            integration_panel_->Toggle();
            spdlog::info("Toggled Numerical Integration panel");
        }
    });

    // Time Series Analysis callbacks (Phase 10)
    toolbar_->SetOpenDecompositionCallback([this]() {
        if (decomposition_panel_) {
            decomposition_panel_->Toggle();
            spdlog::info("Toggled Time Series Decomposition panel");
        }
    });
    toolbar_->SetOpenACFPACFCallback([this]() {
        if (acf_pacf_panel_) {
            acf_pacf_panel_->Toggle();
            spdlog::info("Toggled ACF/PACF panel");
        }
    });
    toolbar_->SetOpenStationarityCallback([this]() {
        if (stationarity_panel_) {
            stationarity_panel_->Toggle();
            spdlog::info("Toggled Stationarity Testing panel");
        }
    });
    toolbar_->SetOpenSeasonalityCallback([this]() {
        if (seasonality_panel_) {
            seasonality_panel_->Toggle();
            spdlog::info("Toggled Seasonality Detection panel");
        }
    });
    toolbar_->SetOpenForecastingCallback([this]() {
        if (forecasting_panel_) {
            forecasting_panel_->Toggle();
            spdlog::info("Toggled Forecasting panel");
        }
    });

    // Text Processing callbacks (Phase 11)
    toolbar_->SetOpenTokenizationCallback([this]() {
        if (tokenization_panel_) {
            tokenization_panel_->SetVisible(true);
            spdlog::info("Opened Tokenization panel");
        }
    });
    toolbar_->SetOpenWordFrequencyCallback([this]() {
        if (word_frequency_panel_) {
            word_frequency_panel_->SetVisible(true);
            spdlog::info("Opened Word Frequency panel");
        }
    });
    toolbar_->SetOpenTFIDFCallback([this]() {
        if (tfidf_panel_) {
            tfidf_panel_->SetVisible(true);
            spdlog::info("Opened TF-IDF panel");
        }
    });
    toolbar_->SetOpenEmbeddingsCallback([this]() {
        if (embeddings_panel_) {
            embeddings_panel_->SetVisible(true);
            spdlog::info("Opened Embeddings panel");
        }
    });
    toolbar_->SetOpenSentimentCallback([this]() {
        if (sentiment_panel_) {
            sentiment_panel_->SetVisible(true);
            spdlog::info("Opened Sentiment Analysis panel");
        }
    });

    // Utilities panel callbacks (Phase 12)
    toolbar_->SetOpenCalculatorCallback([this]() {
        if (calculator_panel_) {
            calculator_panel_->SetVisible(true);
            spdlog::info("Opened Calculator panel");
        }
    });
    toolbar_->SetOpenUnitConverterCallback([this]() {
        if (unit_converter_panel_) {
            unit_converter_panel_->SetVisible(true);
            spdlog::info("Opened Unit Converter panel");
        }
    });
    toolbar_->SetOpenRandomGeneratorCallback([this]() {
        if (random_generator_panel_) {
            random_generator_panel_->SetVisible(true);
            spdlog::info("Opened Random Generator panel");
        }
    });
    toolbar_->SetOpenHashGeneratorCallback([this]() {
        if (hash_generator_panel_) {
            hash_generator_panel_->SetVisible(true);
            spdlog::info("Opened Hash Generator panel");
        }
    });
    toolbar_->SetOpenJSONViewerCallback([this]() {
        if (json_viewer_panel_) {
            json_viewer_panel_->SetVisible(true);
            spdlog::info("Opened JSON Viewer panel");
        }
    });
    toolbar_->SetOpenRegexTesterCallback([this]() {
        if (regex_tester_panel_) {
            regex_tester_panel_->SetVisible(true);
            spdlog::info("Opened Regex Tester panel");
        }
    });

    // Set minimap visibility pointers for View -> Minimaps menu
    toolbar_->SetNodeEditorMinimapPtr(node_editor_->GetShowMinimapPtr());
    toolbar_->SetScriptEditorMinimapPtr(script_editor_->GetShowMinimapPtr());

    // Register callbacks with ProjectManager for project lifecycle events
    cyxwiz::ProjectManager::Instance().SetOnProjectOpened([this](const std::string& project_root) {
        this->OnProjectOpened(project_root);
    });

    cyxwiz::ProjectManager::Instance().SetOnProjectClosed([this](const std::string& project_root) {
        this->OnProjectClosed(project_root);
    });

    cyxwiz::ProjectManager::Instance().SetOnProjectVenvReady([this](const std::string& project_root) {
        this->OnProjectVenvReady(project_root);
    });

    // Set up New Script callback - creates new untitled script and opens editor
    toolbar_->SetNewScriptCallback([this]() {
        if (script_editor_) {
            if (!script_editor_->HasEmptyNewTab()) {
                script_editor_->NewFile();
            }
            script_editor_->SetVisible(true);
            spdlog::info("Created new script in editor");
        }
    });

    // Set up Open Script in Editor callback (called with file path to open)
    toolbar_->SetOpenScriptInEditorCallback([this](const std::string& file_path) {
        if (script_editor_) {
            script_editor_->OpenFile(file_path);
            script_editor_->SetVisible(true);  // Show the Script Editor panel
            spdlog::info("Opened script in editor: {}", file_path);
        }
    });

    // Set up Open Python Console callback
    toolbar_->SetOpenPythonConsoleCallback([this]() {
        if (command_window_) {
            command_window_->SetVisible(true);
            spdlog::info("Opened Python Console (Command Window)");
        }
    });

    // Set up Save All callback
    toolbar_->SetSaveAllCallback([this]() {
        SaveAllFiles();
        spdlog::info("All files saved via toolbar");
    });

    // Set up Exit callback (called when user confirms exit)
    toolbar_->SetExitCallback([this]() {
        if (exit_request_callback_) {
            exit_request_callback_();
        }
    });

    // Set up Has Unsaved Changes callback (called to check if confirmation dialog is needed)
    toolbar_->SetHasUnsavedChangesCallback([this]() -> bool {
        return HasUnsavedFiles();
    });

    // Set up Edit menu callbacks for Find/Replace
    toolbar_->SetFindCallback([this](const std::string& text, bool case_sensitive, bool whole_word, bool use_regex) {
        if (script_editor_) {
            script_editor_->FindInEditor(text, case_sensitive, whole_word, use_regex);
        }
    });

    toolbar_->SetReplaceCallback([this](const std::string& find_text, const std::string& replace_text,
                                        bool case_sensitive, bool whole_word, bool use_regex) {
        if (script_editor_) {
            script_editor_->Replace(find_text, replace_text, case_sensitive, whole_word, use_regex);
        }
    });

    toolbar_->SetReplaceAllCallback([this](const std::string& find_text, const std::string& replace_text,
                                           bool case_sensitive, bool whole_word, bool use_regex) {
        if (script_editor_) {
            int count = script_editor_->ReplaceAll(find_text, replace_text, case_sensitive, whole_word, use_regex);
            spdlog::info("Replaced {} occurrences", count);
        }
    });

    // Set up edit operation callbacks
    toolbar_->SetUndoCallback([this]() {
        if (script_editor_) {
            script_editor_->Undo();
        }
    });

    toolbar_->SetRedoCallback([this]() {
        if (script_editor_) {
            script_editor_->Redo();
        }
    });

    toolbar_->SetCutCallback([this]() {
        if (script_editor_) {
            script_editor_->Cut();
        }
    });

    toolbar_->SetCopyCallback([this]() {
        if (script_editor_) {
            script_editor_->Copy();
        }
    });

    toolbar_->SetPasteCallback([this]() {
        if (script_editor_) {
            script_editor_->Paste();
        }
    });

    toolbar_->SetDeleteCallback([this]() {
        if (script_editor_) {
            script_editor_->Delete();
        }
    });

    toolbar_->SetSelectAllCallback([this]() {
        if (script_editor_) {
            script_editor_->SelectAll();
        }
    });

    // Set up comment toggle callbacks
    toolbar_->SetToggleLineCommentCallback([this]() {
        if (script_editor_) {
            script_editor_->ToggleLineComment();
        }
    });

    toolbar_->SetToggleBlockCommentCallback([this]() {
        if (script_editor_) {
            script_editor_->ToggleBlockComment();
        }
    });

    // Set up Go to Line callback
    toolbar_->SetGoToLineCallback([this](int line) {
        if (script_editor_) {
            script_editor_->GoToLine(line);
        }
    });

    // Set up line operation callbacks
    toolbar_->SetDuplicateLineCallback([this]() {
        if (script_editor_) {
            script_editor_->DuplicateLine();
        }
    });

    toolbar_->SetMoveLineUpCallback([this]() {
        if (script_editor_) {
            script_editor_->MoveLineUp();
        }
    });

    toolbar_->SetMoveLineDownCallback([this]() {
        if (script_editor_) {
            script_editor_->MoveLineDown();
        }
    });

    toolbar_->SetIndentCallback([this]() {
        if (script_editor_) {
            script_editor_->Indent();
        }
    });

    toolbar_->SetOutdentCallback([this]() {
        if (script_editor_) {
            script_editor_->Outdent();
        }
    });

    // Set up text transformation callbacks
    toolbar_->SetTransformUppercaseCallback([this]() {
        if (script_editor_) {
            script_editor_->TransformToUppercase();
        }
    });

    toolbar_->SetTransformLowercaseCallback([this]() {
        if (script_editor_) {
            script_editor_->TransformToLowercase();
        }
    });

    toolbar_->SetTransformTitleCaseCallback([this]() {
        if (script_editor_) {
            script_editor_->TransformToTitleCase();
        }
    });

    // Set up sort and join lines callbacks
    toolbar_->SetSortLinesAscCallback([this]() {
        if (script_editor_) {
            script_editor_->SortLinesAscending();
        }
    });

    toolbar_->SetSortLinesDescCallback([this]() {
        if (script_editor_) {
            script_editor_->SortLinesDescending();
        }
    });

    toolbar_->SetJoinLinesCallback([this]() {
        if (script_editor_) {
            script_editor_->JoinLines();
        }
    });

    // Set up editor settings callbacks (Preferences -> Script Editor synchronization)
    toolbar_->SetEditorThemeCallback([this](int theme_index) {
        if (script_editor_) {
            script_editor_->SetTheme(theme_index);
            spdlog::info("Editor theme changed to index {}", theme_index);
        }
    });

    toolbar_->SetEditorTabSizeCallback([this](int tab_size) {
        if (script_editor_) {
            script_editor_->SetTabSize(tab_size);
            spdlog::info("Editor tab size changed to {}", tab_size);
        }
    });

    toolbar_->SetEditorFontScaleCallback([this](float scale) {
        if (script_editor_) {
            script_editor_->SetFontScale(scale);
            spdlog::info("Editor font scale changed to {}", scale);
        }
    });

    toolbar_->SetEditorShowWhitespaceCallback([this](bool show) {
        if (script_editor_) {
            script_editor_->SetShowWhitespace(show);
            spdlog::info("Editor show whitespace changed to {}", show);
        }
    });

    toolbar_->SetEditorWordWrapCallback([this](bool wrap) {
        if (script_editor_) {
            script_editor_->SetWordWrap(wrap);
            spdlog::info("Editor word wrap changed to {}", wrap);
        }
    });

    toolbar_->SetEditorAutoIndentCallback([this](bool indent) {
        if (script_editor_) {
            script_editor_->SetAutoIndent(indent);
            spdlog::info("Editor auto indent changed to {}", indent);
        }
    });

    // Initialize toolbar editor settings from script editor's current values
    if (script_editor_) {
        toolbar_->SetEditorTheme(script_editor_->GetThemeIndex());
        toolbar_->SetEditorTabSize(script_editor_->GetTabSize());
        toolbar_->SetEditorFontScale(script_editor_->GetFontScale());
        toolbar_->SetEditorShowWhitespace(script_editor_->GetShowWhitespace());
        toolbar_->SetEditorWordWrap(script_editor_->GetWordWrap());
        toolbar_->SetEditorAutoIndent(script_editor_->GetAutoIndent());

        // Set up callback for when settings change in Script Editor (View menu)
        // This syncs changes back to the Preferences dialog and saves to project
        script_editor_->SetOnSettingsChangedCallback([this]() {
            if (toolbar_ && script_editor_) {
                toolbar_->SetEditorTheme(script_editor_->GetThemeIndex());
                toolbar_->SetEditorTabSize(script_editor_->GetTabSize());
                toolbar_->SetEditorFontScale(script_editor_->GetFontScale());
                toolbar_->SetEditorShowWhitespace(script_editor_->GetShowWhitespace());
                toolbar_->SetEditorWordWrap(script_editor_->GetWordWrap());
                toolbar_->SetEditorAutoIndent(script_editor_->GetAutoIndent());
            }
            // Save settings to project file immediately
            SaveProjectSettings();
        });
    }

    // Set up asset browser double-click callback to open files in script editor
    asset_browser_->SetOnAssetDoubleClick([this](const cyxwiz::AssetBrowserPanel::AssetItem& item) {
        if (!item.is_directory && script_editor_) {
            // Get file extension
            std::string ext = std::filesystem::path(item.absolute_path).extension().string();

            // Open text-based files in script editor
            if (ext == ".py" || ext == ".cyx" || ext == ".txt" ||
                ext == ".json" || ext == ".md" || ext == ".csv" ||
                ext == ".yaml" || ext == ".yml" || ext == ".toml" ||
                ext == ".ini" || ext == ".cfg" || ext == ".conf") {
                script_editor_->OpenFile(item.absolute_path);
                spdlog::info("Opened file in script editor: {}", item.name);
            }
        }
    });

    // Set up asset browser callback for dataset loading
    asset_browser_->SetOnDatasetLoaded([this](const std::string& path, cyxwiz::DatasetHandle handle) {
        if (handle.IsValid()) {
            spdlog::info("Dataset loaded from Asset Browser: {}", path);
            if (data_explorer_panel_) {
                data_explorer_panel_->SetVisible(true);
            }

            // Update node editor with dataset name
            if (node_editor_) {
                // Extract dataset name from path (e.g., "mnist.npz" -> "MNIST")
                std::filesystem::path fs_path(path);
                std::string dataset_name = fs_path.stem().string();
                // Capitalize first letter
                if (!dataset_name.empty()) {
                    dataset_name[0] = std::toupper(dataset_name[0]);
                }
                node_editor_->UpdateDatasetNodeName(dataset_name);
            }
        }
    });

    // Set up asset browser callback for "View in Table" option
    asset_browser_->SetOnViewInTable([this](const std::string& path) {
        if (table_viewer_) {
            std::filesystem::path fs_path(path);
            std::string ext = fs_path.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            bool loaded = false;

            // Load based on file extension
            if (ext == ".csv") {
                loaded = table_viewer_->LoadCSV(path);
            } else if (ext == ".tsv") {
                loaded = table_viewer_->LoadTXT(path, '\t');
            } else if (ext == ".txt") {
                loaded = table_viewer_->LoadTXT(path);
            } else if (ext == ".h5" || ext == ".hdf5") {
                loaded = table_viewer_->LoadHDF5(path);
            } else if (ext == ".xlsx" || ext == ".xls") {
                loaded = table_viewer_->LoadExcel(path);
            } else {
                // Try to load as CSV by default
                loaded = table_viewer_->LoadCSV(path);
            }

            if (loaded) {
                table_viewer_->Show();
                spdlog::info("Opened file in table viewer: {}", fs_path.filename().string());
            } else {
                spdlog::error("Failed to open file in table viewer: {}", path);
            }
        }
    });

    // Set up asset browser callback for "Open in Node Editor" (.cyxgraph files)
    asset_browser_->SetOnOpenInNodeEditor([this](const std::string& path) {
        if (node_editor_) {
            if (node_editor_->LoadGraph(path)) {
                node_editor_->Show();
                spdlog::info("Opened graph in Node Editor: {}", std::filesystem::path(path).filename().string());
            } else {
                spdlog::error("Failed to open graph file: {}", path);
            }
        }
    });

    // Initialize startup script manager
    startup_script_manager_ = std::make_unique<scripting::StartupScriptManager>(scripting_engine_);

    // Load and execute startup scripts
    if (startup_script_manager_->LoadConfig()) {
        spdlog::info("Executing startup scripts...");
        startup_script_manager_->ExecuteAll(command_window_.get());
    } else {
        spdlog::debug("No startup scripts configured or startup_scripts.txt not found");
    }

    // Install custom dock node handler for Unreal-style tabs
    DockStyle::InstallCustomHandler();

    // Initialize Tutorial System
    auto& tutorial = cyxwiz::TutorialSystem::Instance();
    tutorial.SetWindowRectCallback([](const std::string& window_name) -> ImRect {
        ImGuiWindow* window = ImGui::FindWindowByName(window_name.c_str());
        if (window) {
            return ImRect(window->Pos, ImVec2(window->Pos.x + window->Size.x, window->Pos.y + window->Size.y));
        }
        return ImRect();
    });
    tutorial.LoadProgress();

    // Check for first-launch welcome
    if (tutorial.ShouldShowWelcome()) {
        tutorial.MarkWelcomeShown();
        tutorial.SaveProgress();
        // Optionally start the getting started tutorial automatically
        // tutorial.StartTutorial("getting_started");
    }

    // Set default panel visibility (hides tool panels on first launch)
    SetDefaultPanelVisibility();

    // Register panels with sidebar for hide/unhide toggles
    RegisterPanelsWithSidebar();

    spdlog::info("MainWindow initialized with docking layout system");
}

MainWindow::~MainWindow() {
    spdlog::info("MainWindow destructor: starting cleanup");

    // IMPORTANT: Destroy panels that use PlotManager/Python BEFORE scripting_engine_
    // PlotWindow destructor calls PlotManager::DeletePlot() which may use Python
    spdlog::info("~MainWindow: plot_test_control_ (before scripting_engine)");
    plot_test_control_.reset();
    spdlog::info("~MainWindow: training_plot_panel_ (before scripting_engine)");
    training_plot_panel_.reset();

    // IMPORTANT: Destroy panels that hold shared_ptr<ScriptingEngine> BEFORE
    // the MainWindow's scripting_engine_, so all references are released
    // before we begin any Python cleanup
    spdlog::info("~MainWindow: command_window_ (holds scripting_engine)");
    command_window_.reset();
    spdlog::info("~MainWindow: script_editor_ (holds scripting_engine)");
    script_editor_.reset();
    spdlog::info("~MainWindow: variable_explorer_ (holds scripting_engine)");
    variable_explorer_.reset();
    spdlog::info("~MainWindow: plot_output_panel_ (holds scripting_engine)");
    plot_output_panel_.reset();
    spdlog::info("~MainWindow: startup_script_manager_ (holds scripting_engine)");
    startup_script_manager_.reset();

    // NOW reset MainWindow's scripting_engine_ - it should be the last reference
    spdlog::info("~MainWindow: scripting_engine_");
    scripting_engine_.reset();
    spdlog::info("~MainWindow: regex_tester_panel_");
    regex_tester_panel_.reset();
    spdlog::info("~MainWindow: json_viewer_panel_");
    json_viewer_panel_.reset();
    spdlog::info("~MainWindow: hash_generator_panel_");
    hash_generator_panel_.reset();
    spdlog::info("~MainWindow: random_generator_panel_");
    random_generator_panel_.reset();
    spdlog::info("~MainWindow: unit_converter_panel_");
    unit_converter_panel_.reset();
    spdlog::info("~MainWindow: calculator_panel_");
    calculator_panel_.reset();
    spdlog::info("~MainWindow: sentiment_panel_");
    sentiment_panel_.reset();
    spdlog::info("~MainWindow: embeddings_panel_");
    embeddings_panel_.reset();
    spdlog::info("~MainWindow: tfidf_panel_");
    tfidf_panel_.reset();
    spdlog::info("~MainWindow: word_frequency_panel_");
    word_frequency_panel_.reset();
    spdlog::info("~MainWindow: tokenization_panel_");
    tokenization_panel_.reset();
    spdlog::info("~MainWindow: forecasting_panel_");
    forecasting_panel_.reset();
    spdlog::info("~MainWindow: seasonality_panel_");
    seasonality_panel_.reset();
    spdlog::info("~MainWindow: stationarity_panel_");
    stationarity_panel_.reset();
    spdlog::info("~MainWindow: acf_pacf_panel_");
    acf_pacf_panel_.reset();
    spdlog::info("~MainWindow: decomposition_panel_");
    decomposition_panel_.reset();
    spdlog::info("~MainWindow: integration_panel_");
    integration_panel_.reset();
    spdlog::info("~MainWindow: differentiation_panel_");
    differentiation_panel_.reset();
    spdlog::info("~MainWindow: qp_panel_");
    qp_panel_.reset();
    spdlog::info("~MainWindow: lp_panel_");
    lp_panel_.reset();
    spdlog::info("~MainWindow: convexity_panel_");
    convexity_panel_.reset();
    spdlog::info("~MainWindow: gradient_descent_panel_");
    gradient_descent_panel_.reset();
    spdlog::info("~MainWindow: wavelet_panel_");
    wavelet_panel_.reset();
    spdlog::info("~MainWindow: convolution_panel_");
    convolution_panel_.reset();
    spdlog::info("~MainWindow: filter_designer_panel_");
    filter_designer_panel_.reset();
    spdlog::info("~MainWindow: spectrogram_panel_");
    spectrogram_panel_.reset();
    spdlog::info("~MainWindow: fft_panel_");
    fft_panel_.reset();
    spdlog::info("~MainWindow: cholesky_panel_");
    cholesky_panel_.reset();
    spdlog::info("~MainWindow: qr_panel_");
    qr_panel_.reset();
    spdlog::info("~MainWindow: svd_panel_");
    svd_panel_.reset();
    spdlog::info("~MainWindow: eigen_decomp_panel_");
    eigen_decomp_panel_.reset();
    spdlog::info("~MainWindow: matrix_calculator_panel_");
    matrix_calculator_panel_.reset();
    spdlog::info("~MainWindow: feature_scaling_panel_");
    feature_scaling_panel_.reset();
    spdlog::info("~MainWindow: boxcox_panel_");
    boxcox_panel_.reset();
    spdlog::info("~MainWindow: log_transform_panel_");
    log_transform_panel_.reset();
    spdlog::info("~MainWindow: standardization_panel_");
    standardization_panel_.reset();
    spdlog::info("~MainWindow: normalization_panel_");
    normalization_panel_.reset();
    spdlog::info("~MainWindow: learning_curves_panel_");
    learning_curves_panel_.reset();
    spdlog::info("~MainWindow: cross_validation_panel_");
    cross_validation_panel_.reset();
    spdlog::info("~MainWindow: pr_curve_panel_");
    pr_curve_panel_.reset();
    spdlog::info("~MainWindow: roc_auc_panel_");
    roc_auc_panel_.reset();
    spdlog::info("~MainWindow: confusion_matrix_panel_");
    confusion_matrix_panel_.reset();
    spdlog::info("~MainWindow: cluster_eval_panel_");
    cluster_eval_panel_.reset();
    spdlog::info("~MainWindow: gmm_panel_");
    gmm_panel_.reset();
    spdlog::info("~MainWindow: hierarchical_panel_");
    hierarchical_panel_.reset();
    spdlog::info("~MainWindow: dbscan_panel_");
    dbscan_panel_.reset();
    spdlog::info("~MainWindow: kmeans_panel_");
    kmeans_panel_.reset();
    spdlog::info("~MainWindow: serving_panel_");
    serving_panel_.reset();
    spdlog::info("~MainWindow: hyperparam_search_panel_");
    hyperparam_search_panel_.reset();
    spdlog::info("~MainWindow: nas_panel_");
    nas_panel_.reset();
    spdlog::info("~MainWindow: feature_importance_panel_");
    feature_importance_panel_.reset();
    spdlog::info("~MainWindow: gradcam_panel_");
    gradcam_panel_.reset();
    spdlog::info("~MainWindow: dim_reduction_panel_");
    dim_reduction_panel_.reset();
    spdlog::info("~MainWindow: regression_panel_");
    regression_panel_.reset();
    spdlog::info("~MainWindow: distribution_fitter_panel_");
    distribution_fitter_panel_.reset();
    spdlog::info("~MainWindow: hypothesis_test_panel_");
    hypothesis_test_panel_.reset();
    spdlog::info("~MainWindow: descriptive_stats_panel_");
    descriptive_stats_panel_.reset();
    spdlog::info("~MainWindow: outlier_detection_panel_");
    outlier_detection_panel_.reset();
    spdlog::info("~MainWindow: missing_value_panel_");
    missing_value_panel_.reset();
    spdlog::info("~MainWindow: correlation_matrix_panel_");
    correlation_matrix_panel_.reset();
    spdlog::info("~MainWindow: data_profiler_panel_");
    data_profiler_panel_.reset();
    spdlog::info("~MainWindow: lr_finder_panel_");
    lr_finder_panel_.reset();
    spdlog::info("~MainWindow: architecture_diagram_");
    architecture_diagram_.reset();
    spdlog::info("~MainWindow: model_summary_panel_");
    model_summary_panel_.reset();
    spdlog::info("~MainWindow: deployment_dialog_");
    deployment_dialog_.reset();
    spdlog::info("~MainWindow: import_dialog_");
    import_dialog_.reset();
    spdlog::info("~MainWindow: export_dialog_");
    export_dialog_.reset();
    spdlog::info("~MainWindow: test_results_panel_");
    test_results_panel_.reset();
    // variable_explorer_ already reset at the beginning
    spdlog::info("~MainWindow: memory_monitor_");
    memory_monitor_.reset();
    spdlog::info("~MainWindow: memory_panel_");
    memory_panel_.reset();
    spdlog::info("~MainWindow: profiling_panel_");
    profiling_panel_.reset();
    spdlog::info("~MainWindow: theme_editor_");
    theme_editor_.reset();
    spdlog::info("~MainWindow: custom_node_editor_");
    custom_node_editor_.reset();
    spdlog::info("~MainWindow: query_console_");
    query_console_.reset();
    spdlog::info("~MainWindow: pattern_browser_");
    pattern_browser_.reset();
    spdlog::info("~MainWindow: task_progress_panel_");
    task_progress_panel_.reset();
    spdlog::info("~MainWindow: wallet_panel_");
    wallet_panel_.reset();
    spdlog::info("~MainWindow: p2p_training_panel_");
    p2p_training_panel_.reset();
    spdlog::info("~MainWindow: job_status_panel_");
    job_status_panel_.reset();
    spdlog::info("~MainWindow: connection_dialog_");
    connection_dialog_.reset();
    spdlog::info("~MainWindow: table_viewer_");
    table_viewer_.reset();
    spdlog::info("~MainWindow: data_explorer_panel_");
    data_explorer_panel_.reset();
    // script_editor_, command_window_ already reset at the beginning (with scripting_engine panels)
    // plot_test_control_, training_plot_panel_ already reset at the beginning
    spdlog::info("~MainWindow: asset_browser_");
    asset_browser_.reset();
    spdlog::info("~MainWindow: toolbar_");
    toolbar_.reset();
    spdlog::info("~MainWindow: properties_");
    properties_.reset();
    spdlog::info("~MainWindow: viewport_");
    viewport_.reset();
    spdlog::info("~MainWindow: console_");
    console_.reset();
    spdlog::info("~MainWindow: node_editor_");
    node_editor_.reset();

    spdlog::info("~MainWindow: all panels destroyed");
}

void MainWindow::SetIdleLogPtr(bool* ptr) {
    if (toolbar_) {
        toolbar_->SetIdleLogPtr(ptr);
    }
}

void MainWindow::SetNetworkComponents(network::GRPCClient* client, network::JobManager* job_manager) {
    // Store job manager reference
    job_manager_ = job_manager;

    // Create connection dialog with network components
    connection_dialog_ = std::make_unique<cyxwiz::ConnectionDialog>(client, job_manager);

    // Connect NodeEditor so ConnectionDialog can access the model graph
    if (connection_dialog_ && node_editor_) {
        connection_dialog_->SetNodeEditor(node_editor_.get());
    }

    // Create and wire up ReservationClient for node reservation (async connect to avoid blocking startup)
    if (connection_dialog_ && client) {
        auto reservation_client = std::make_shared<network::ReservationClient>();
        // Set auth token from AuthClient if authenticated
        auto& auth = cyxwiz::auth::AuthClient::Instance();
        if (auth.IsAuthenticated()) {
            reservation_client->SetAuthToken(auth.GetJwtToken());
        }
        // Connect asynchronously - won't block the main thread
        reservation_client->ConnectAsync(cyxwiz::core::EngineConfig::Instance().GetCentralServerAddress());
        connection_dialog_->SetReservationClient(reservation_client);
        spdlog::info("ReservationClient wired to ConnectionDialog (connecting in background)");
    }

    // Create a shared P2PClient for reservation-based connections
    if (connection_dialog_) {
        auto p2p_client = std::make_shared<network::P2PClient>();
        connection_dialog_->SetP2PClient(p2p_client);
        spdlog::info("P2PClient created and wired to ConnectionDialog");
    }

    // Wire up P2PTrainingPanel to ConnectionDialog
    if (connection_dialog_ && p2p_training_panel_) {
        connection_dialog_->SetP2PTrainingPanel(p2p_training_panel_.get());
        spdlog::info("P2PTrainingPanel wired to ConnectionDialog");
    }

    // Wire up WalletPanel to ConnectionDialog for wallet address
    if (connection_dialog_ && wallet_panel_) {
        connection_dialog_->SetWalletPanel(wallet_panel_.get());
        spdlog::info("WalletPanel wired to ConnectionDialog (ptr={:p})", (void*)wallet_panel_.get());
    } else {
        spdlog::warn("Failed to wire WalletPanel: connection_dialog_={}, wallet_panel_={}",
            (bool)connection_dialog_, (bool)wallet_panel_);
    }

    // Set JobManager for JobStatusPanel
    if (job_status_panel_) {
        job_status_panel_->SetJobManager(job_manager);
        job_manager->SetJobStatusPanel(job_status_panel_.get());
    }

    // Set up callback in toolbar to show connection dialog
    if (toolbar_) {
        toolbar_->SetConnectToServerCallback([this]() {
            if (connection_dialog_) {
                connection_dialog_->Show();
            }
        });

        // Set up auth callbacks to propagate JWT token to gRPC client
        toolbar_->SetOnLoginSuccessCallback([client](const std::string& jwt_token) {
            if (client) {
                client->SetAuthToken(jwt_token);
                spdlog::info("Auth token set on gRPC client after login");
            }
        });

        toolbar_->SetOnLogoutCallback([client]() {
            if (client) {
                client->ClearAuthToken();
                spdlog::info("Auth token cleared from gRPC client after logout");
            }
        });
    }

    spdlog::info("Network components set in MainWindow");
}

void MainWindow::SetDataStreamClient(network::DataStreamClient* client) {
    datastream_client_ = client;

    // Pass DataStreamClient to cloud panels
    if (cloud_browser_panel_) {
        cloud_browser_panel_->SetDataStreamClient(client);
        spdlog::info("DataStreamClient set on CloudBrowserPanel");
    }
    if (cloud_dataset_manager_panel_) {
        cloud_dataset_manager_panel_->SetDataStreamClient(client);
        spdlog::info("DataStreamClient set on CloudDatasetManagerPanel");
    }

    spdlog::info("DataStreamClient set in MainWindow");
}

void MainWindow::StartJobMonitoring(const std::string& job_id) {
    spdlog::info("Starting P2P monitoring for job: {}", job_id);

    monitoring_job_id_ = job_id;

    // Note: The P2PClient won't be available immediately - it's created when
    // the job is assigned to a node. For now, we'll start monitoring with
    // the job_id, and the P2PTrainingPanel will connect when the client is ready.

    if (p2p_training_panel_) {
        // Start monitoring - this sets up the panel state
        // The P2PClient will be connected later when available
        p2p_training_panel_->StartMonitoring(job_id, "");
        p2p_training_panel_->Show();

        spdlog::info("P2PTrainingPanel now monitoring job: {}", job_id);
    }

    // Try to get the P2PClient if it's already available
    if (job_manager_) {
        auto p2p_client = job_manager_->GetP2PClient(job_id);
        if (p2p_client && p2p_training_panel_) {
            p2p_training_panel_->SetP2PClient(p2p_client);
            spdlog::info("P2PClient connected to monitoring panel");
        }
    }
}

bool MainWindow::IsScriptRunning() const {
    if (scripting_engine_) {
        return scripting_engine_->IsScriptRunning();
    }
    return false;
}

void MainWindow::StopRunningScript() {
    if (scripting_engine_) {
        scripting_engine_->StopScript();
    }
}

bool MainWindow::HasUnsavedFiles() const {
    if (script_editor_) {
        return script_editor_->HasUnsavedFiles();
    }
    return false;
}

std::vector<std::string> MainWindow::GetUnsavedFileNames() const {
    if (script_editor_) {
        return script_editor_->GetUnsavedFileNames();
    }
    return {};
}

void MainWindow::SaveAllFiles() {
    if (script_editor_) {
        script_editor_->SaveAllFiles();
    }
}

void MainWindow::PrepareForShutdown() {
    spdlog::info("MainWindow: Preparing for shutdown...");

    // Stop any running script first (this can take time to interrupt Python)
    if (scripting_engine_ && scripting_engine_->IsScriptRunning()) {
        spdlog::info("MainWindow: Stopping running script...");
        scripting_engine_->StopScript();
    }

    // Stop P2P training monitoring and cancel any model downloads
    if (p2p_training_panel_) {
        if (p2p_training_panel_->IsDownloading()) {
            spdlog::info("MainWindow: Cancelling model download...");
            p2p_training_panel_->CancelDownloadAndWait();
        }
        if (p2p_training_panel_->IsMonitoring()) {
            spdlog::info("MainWindow: Stopping P2P monitoring...");
            p2p_training_panel_->StopMonitoring();
        }
    }

    // Stop inference server if running
    if (serving_panel_ && serving_panel_->GetServer()) {
        spdlog::info("MainWindow: Stopping inference server...");
        serving_panel_->GetServer()->Stop();
    }

    spdlog::info("MainWindow: Shutdown preparation complete");
}

void MainWindow::ResetDockLayout() {
    // Force rebuild of the docking layout
    first_time_layout_ = true;
    spdlog::info("Dock layout reset requested");
}

void MainWindow::Render() {
    // Handle global keyboard shortcuts
    HandleGlobalShortcuts();

    // Check if we need to connect P2PClient to monitoring panel
    if (!monitoring_job_id_.empty() && job_manager_ && p2p_training_panel_) {
        auto p2p_client = job_manager_->GetP2PClient(monitoring_job_id_);
        if (p2p_client) {
            // Check if P2PTrainingPanel doesn't have the client yet
            // (SetP2PClient is idempotent so we can call it multiple times safely)
            p2p_training_panel_->SetP2PClient(p2p_client);
        }
    }

    RenderDockSpace();

    // Render Unreal-style sidebar for panel toggles
    RenderSidebar();

    // Render new panel system - Toolbar (replaces old menu bar)
    if (toolbar_) toolbar_->Render();

    // Render new panels
    if (asset_browser_) asset_browser_->Render();
    if (training_plot_panel_) training_plot_panel_->Render();  // Now "Training Dashboard"
    if (plot_test_control_) plot_test_control_->Render();
    if (command_window_) command_window_->Render();
    if (script_editor_) script_editor_->Render();
    if (table_viewer_) table_viewer_->Render();
    if (data_explorer_panel_) data_explorer_panel_->Render();
    if (annotation_editor_panel_) annotation_editor_panel_->Render();
    if (visualization_panel_) visualization_panel_->Render();
    if (connection_dialog_) connection_dialog_->Render();
    if (job_status_panel_) job_status_panel_->Render();
    if (p2p_training_panel_) p2p_training_panel_->Render();
    if (wallet_panel_) wallet_panel_->Render();
    if (task_progress_panel_) task_progress_panel_->Render();
    if (pattern_browser_) pattern_browser_->Render();
    if (query_console_) query_console_->Render();
    if (custom_node_editor_) custom_node_editor_->Render();
    if (theme_editor_) theme_editor_->Render();
    if (profiling_panel_) profiling_panel_->Render();
    if (memory_panel_) memory_panel_->Render();
    if (memory_monitor_) memory_monitor_->Render();
    if (variable_explorer_) variable_explorer_->Render();
    if (plot_output_panel_) plot_output_panel_->Render();
    if (test_results_panel_) test_results_panel_->Render();
    if (export_dialog_) export_dialog_->Render();
    if (import_dialog_) import_dialog_->Render();
    if (deployment_dialog_) deployment_dialog_->Render();

    // Render Python-created plot windows (cyxwiz_plotting)
    for (const auto& plot_window : cyxwiz::GetPythonPlotWindows()) {
        if (plot_window) {
            plot_window->Render();
        }
    }

    // Render Model Analysis panels (Phase 2)
    if (model_summary_panel_) model_summary_panel_->Render();
    if (architecture_diagram_) architecture_diagram_->Render();
    if (lr_finder_panel_) lr_finder_panel_->Render();

    // Render Data Science panels (Phase 3)
    if (data_profiler_panel_) data_profiler_panel_->Render();
    if (correlation_matrix_panel_) correlation_matrix_panel_->Render();
    if (missing_value_panel_) missing_value_panel_->Render();
    if (outlier_detection_panel_) outlier_detection_panel_->Render();

    // Render Statistics panels (Phase 4)
    if (descriptive_stats_panel_) descriptive_stats_panel_->Render();
    if (hypothesis_test_panel_) hypothesis_test_panel_->Render();
    if (distribution_fitter_panel_) distribution_fitter_panel_->Render();
    if (regression_panel_) regression_panel_->Render();

    // Render Advanced Tools panels (Phase 5)
    if (dim_reduction_panel_) dim_reduction_panel_->Render();
    if (gradcam_panel_) gradcam_panel_->Render();
    if (feature_importance_panel_) feature_importance_panel_->Render();
    if (nas_panel_) nas_panel_->Render();
    if (hyperparam_search_panel_) hyperparam_search_panel_->Render();
    if (serving_panel_) serving_panel_->Render();
    if (dnn_inference_panel_) dnn_inference_panel_->Render();

    // Render Clustering panels (Phase 6A)
    if (kmeans_panel_) kmeans_panel_->Render();
    if (dbscan_panel_) dbscan_panel_->Render();
    if (hierarchical_panel_) hierarchical_panel_->Render();
    if (gmm_panel_) gmm_panel_->Render();
    if (cluster_eval_panel_) cluster_eval_panel_->Render();

    // Render Model Evaluation panels (Phase 6B)
    if (confusion_matrix_panel_) confusion_matrix_panel_->Render();
    if (roc_auc_panel_) roc_auc_panel_->Render();
    if (pr_curve_panel_) pr_curve_panel_->Render();
    if (cross_validation_panel_) cross_validation_panel_->Render();
    if (learning_curves_panel_) learning_curves_panel_->Render();

    // Render Data Transformation panels (Phase 6C)
    if (normalization_panel_) normalization_panel_->Render();
    if (standardization_panel_) standardization_panel_->Render();
    if (log_transform_panel_) log_transform_panel_->Render();
    if (boxcox_panel_) boxcox_panel_->Render();
    if (feature_scaling_panel_) feature_scaling_panel_->Render();
    if (optimizer_settings_panel_) optimizer_settings_panel_->Render();

    // Linear Algebra panels (Phase 7)
    if (matrix_calculator_panel_) matrix_calculator_panel_->Render();
    if (eigen_decomp_panel_) eigen_decomp_panel_->Render();
    if (svd_panel_) svd_panel_->Render();
    if (qr_panel_) qr_panel_->Render();
    if (cholesky_panel_) cholesky_panel_->Render();

    // Signal Processing panels (Phase 8)
    if (fft_panel_) fft_panel_->Render();
    if (spectrogram_panel_) spectrogram_panel_->Render();
    if (filter_designer_panel_) filter_designer_panel_->Render();
    if (convolution_panel_) convolution_panel_->Render();
    if (wavelet_panel_) wavelet_panel_->Render();

    // Optimization & Calculus panels (Phase 9)
    if (gradient_descent_panel_) gradient_descent_panel_->Render();
    if (convexity_panel_) convexity_panel_->Render();
    if (lp_panel_) lp_panel_->Render();
    if (qp_panel_) qp_panel_->Render();
    if (differentiation_panel_) differentiation_panel_->Render();
    if (integration_panel_) integration_panel_->Render();

    // Time Series Analysis panels (Phase 10)
    if (decomposition_panel_) decomposition_panel_->Render();
    if (acf_pacf_panel_) acf_pacf_panel_->Render();
    if (stationarity_panel_) stationarity_panel_->Render();
    if (seasonality_panel_) seasonality_panel_->Render();
    if (forecasting_panel_) forecasting_panel_->Render();

    // Text Processing panels (Phase 11)
    if (tokenization_panel_) tokenization_panel_->Render();
    if (word_frequency_panel_) word_frequency_panel_->Render();
    if (tfidf_panel_) tfidf_panel_->Render();
    if (embeddings_panel_) embeddings_panel_->Render();
    if (sentiment_panel_) sentiment_panel_->Render();

    // Render Utilities panels (Phase 12)
    if (calculator_panel_) calculator_panel_->Render();
    if (unit_converter_panel_) unit_converter_panel_->Render();
    if (random_generator_panel_) random_generator_panel_->Render();
    if (hash_generator_panel_) hash_generator_panel_->Render();
    if (json_viewer_panel_) json_viewer_panel_->Render();
    if (regex_tester_panel_) regex_tester_panel_->Render();
    
    // Cloud panels
    if (cloud_browser_panel_) cloud_browser_panel_->Render();
    if (node_browser_panel_) node_browser_panel_->Render();
    if (node_info_panel_) node_info_panel_->Render();
    if (cloud_dataset_manager_panel_) cloud_dataset_manager_panel_->Render();

    // Plugin Manager panel
    if (plugin_manager_panel_) plugin_manager_panel_->Render();

    // Data Studio panel (Phase 1 Week 1)
    if (data_studio_panel_) data_studio_panel_->Render();

    // Render plugin-provided panels
    cyxwiz::plugin::PluginPanelRegistry::Instance().RenderAllVisible();

    // Render permission approval dialogs
    cyxwiz::plugin::PluginManager::Instance().RenderPermissionDialogs();

    // Render original panels
    if (node_editor_) node_editor_->Render();
    if (console_) console_->Render();
    if (viewport_) viewport_->Render();
    if (properties_) properties_->Render();

    if (show_about_dialog_) {
        ShowAboutDialog();
    }

    if (show_demo_window_) {
        ImGui::ShowDemoWindow(&show_demo_window_);
    }

    // Render the Compile Graph result popup (triggered from Train → Compile Graph menu)
    RenderCompileResultPopup();

    // Render tutorial overlay (on top of all panels)
    cyxwiz::TutorialSystem::Instance().Render();

    // Render status bar at the bottom of the screen
    RenderStatusBar();
}

// Helper function to draw active tab indicator on a dock node
static void DrawDockNodeActiveTabIndicator(ImGuiDockNode* node) {
    if (!node) return;

    // Recursively process child nodes
    if (node->ChildNodes[0]) DrawDockNodeActiveTabIndicator(node->ChildNodes[0]);
    if (node->ChildNodes[1]) DrawDockNodeActiveTabIndicator(node->ChildNodes[1]);

    // Only process leaf nodes with tab bars
    if (!node->TabBar || node->Windows.Size == 0) return;

    ImGuiTabBar* tab_bar = node->TabBar;
    const DockTabStyle& style = GetDockStyle().GetStyle();

    // Only draw if we have an active tab and indicator is enabled
    if (!style.show_active_indicator || tab_bar->VisibleTabId == 0) return;

    ImGuiTabItem* active_tab = ImGui::TabBarFindTabByID(tab_bar, tab_bar->VisibleTabId);
    if (!active_tab) return;

    // Get the draw list for the host window
    ImGuiWindow* host_window = node->HostWindow;
    if (!host_window) return;

    ImDrawList* draw_list = host_window->DrawList;

    // Calculate tab position relative to the tab bar
    ImVec2 tab_bar_min = tab_bar->BarRect.Min;
    ImVec2 tab_min = ImVec2(tab_bar_min.x + active_tab->Offset, tab_bar_min.y);
    ImVec2 tab_max = ImVec2(tab_min.x + active_tab->Width, tab_bar_min.y + style.active_indicator_height);

    // Draw the indicator line at the TOP of the active tab
    ImU32 indicator_color = ImGui::ColorConvertFloat4ToU32(style.active_indicator_color);
    draw_list->AddRectFilled(tab_min, tab_max, indicator_color);
}

void MainWindow::RenderDockSpace() {
    static bool opt_fullscreen = false;  // Set to false to show native title bar with window controls
    static bool opt_padding = false;
    static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None;

    // Status bar height constant - must match RenderStatusBar()
    const float status_bar_height = 24.0f;

    // Always get viewport to fill the available space, minus status bar
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(ImVec2(viewport->WorkSize.x, viewport->WorkSize.y - status_bar_height));
    ImGui::SetNextWindowViewport(viewport->ID);

    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
    window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

    if (opt_fullscreen) {
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    }

    if (!opt_padding)
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("DockSpace", nullptr, window_flags);
    if (!opt_padding)
        ImGui::PopStyleVar();

    if (opt_fullscreen)
        ImGui::PopStyleVar(2);

    // DockSpace
    ImGuiIO& io = ImGui::GetIO();
    if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable) {
        ImGuiID dockspace_id = ImGui::GetID("CyxWizDockSpace");

        // Build the initial layout ONLY if no saved layout exists
        if (first_time_layout_) {
            // Check if dockspace node already exists (loaded from imgui.ini)
            ImGuiDockNode* node = ImGui::DockBuilderGetNode(dockspace_id);
            if (node == nullptr || !node->IsSplitNode()) {
                // No saved layout, build default
                BuildInitialDockLayout();
            }
            first_time_layout_ = false;
        }

        ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);

        // Draw Unreal-style active tab indicators on all dock nodes
        ImGuiDockNode* root_node = ImGui::DockBuilderGetNode(dockspace_id);
        if (root_node) {
            DrawDockNodeActiveTabIndicator(root_node);
        }
    }

    ImGui::End();
}

void MainWindow::BuildInitialDockLayout() {
    ImGuiID dockspace_id = ImGui::GetID("CyxWizDockSpace");

    // Clear any existing layout
    ImGui::DockBuilderRemoveNode(dockspace_id);
    ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dockspace_id, ImGui::GetMainViewport()->Size);

    // Create the docking layout structure
    // We'll build this structure:
    // - Left column (Asset Browser) ~15%
    // - Center-Left (CyxWiz Studio) ~50% of center
    // - Center-Right (Data Studio) ~50% of center - side by side
    // - Right column (Properties) ~20%
    // - Bottom section split into:
    //   - Bottom-left (Console) ~70% of bottom
    //   - Bottom-right (Training Dashboard) ~30% of bottom
    //   - Bottom-bottom (Viewport)

    ImGuiID dock_id_left = 0;
    ImGuiID dock_id_right = 0;
    ImGuiID dock_id_center = dockspace_id;
    ImGuiID dock_id_center_left = 0;
    ImGuiID dock_id_center_right = 0;
    ImGuiID dock_id_bottom = 0;
    ImGuiID dock_id_bottom_left = 0;
    ImGuiID dock_id_bottom_right = 0;
    ImGuiID dock_id_bottom_bottom = 0;

    // Split left side for Asset Browser (15% width)
    dock_id_left = ImGui::DockBuilderSplitNode(dock_id_center, ImGuiDir_Left, 0.15f, nullptr, &dock_id_center);

    // Split right side for Properties (20% width)
    dock_id_right = ImGui::DockBuilderSplitNode(dock_id_center, ImGuiDir_Right, 0.25f, nullptr, &dock_id_center);

    // Split bottom for Console area (25% height of remaining)
    dock_id_bottom = ImGui::DockBuilderSplitNode(dock_id_center, ImGuiDir_Down, 0.30f, nullptr, &dock_id_center);

    // Split center horizontally: CyxWiz Studio (left) and Data Studio (right) side by side
    dock_id_center_right = ImGui::DockBuilderSplitNode(dock_id_center, ImGuiDir_Right, 0.5f, nullptr, &dock_id_center_left);

    // Split bottom section horizontally: left (Console) and right (Inspector)
    dock_id_bottom_right = ImGui::DockBuilderSplitNode(dock_id_bottom, ImGuiDir_Right, 0.25f, nullptr, &dock_id_bottom_left);

    // Split bottom-left further for Profiler/Viewport
    dock_id_bottom_bottom = ImGui::DockBuilderSplitNode(dock_id_bottom_left, ImGuiDir_Down, 0.40f, nullptr, &dock_id_bottom_left);

    // Dock windows to their designated areas
    // Window names must EXACTLY match the names in ImGui::Begin() calls in each panel
    ImGui::DockBuilderDockWindow("Asset Browser", dock_id_left);
    ImGui::DockBuilderDockWindow("CyxWiz Studio", dock_id_center_left);
    ImGui::DockBuilderDockWindow("Script Editor", dock_id_center_left); // Tabbed with CyxWiz Studio
    ImGui::DockBuilderDockWindow("Data Studio", dock_id_center_right); // Beside CyxWiz Studio
    ImGui::DockBuilderDockWindow("Properties", dock_id_right);
    ImGui::DockBuilderDockWindow("Console", dock_id_bottom_left);
    ImGui::DockBuilderDockWindow("Command Window", dock_id_bottom_left); // Tabbed with Console
    ImGui::DockBuilderDockWindow("Training Dashboard", dock_id_bottom_right);
    ImGui::DockBuilderDockWindow("Viewport", dock_id_bottom_bottom);

    // Finish the docking layout
    ImGui::DockBuilderFinish(dockspace_id);

    spdlog::info("Initial dock layout built successfully");
    spdlog::info("  - Left: Asset Browser (15%)");
    spdlog::info("  - Center-Left: CyxWiz Studio (50%)");
    spdlog::info("  - Center-Right: Data Studio (50%)");
    spdlog::info("  - Right: Properties (20%)");
    spdlog::info("  - Bottom-Left: Console");
    spdlog::info("  - Bottom-Right: Training Dashboard");
    spdlog::info("  - Bottom-Bottom: Viewport");
}

void MainWindow::ShowAboutDialog() {
    if (!ImGui::Begin("About CyxWiz Engine", &show_about_dialog_, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::End();
        return;
    }

    ImGui::Text("CyxWiz Engine");
    ImGui::Text("Version: %s", cyxwiz::GetVersionString());
    ImGui::Separator();
    ImGui::Text("Decentralized ML Compute Platform");
    ImGui::Text("Built with ImGui, ArrayFire, and gRPC");
    ImGui::Separator();
    if (ImGui::Button("OK")) {
        show_about_dialog_ = false;
    }

    ImGui::End();
}

void MainWindow::RegisterPanelsWithSidebar() {
    DockStyle& dock_style = GetDockStyle();

    // Clear any existing registrations
    dock_style.ClearPanels();

    // Register main panels with FontAwesome icons

    // Main editing panels
    if (node_editor_) {
        dock_style.RegisterPanel("CyxWiz Studio", ICON_FA_DIAGRAM_PROJECT, node_editor_->GetVisiblePtr());
    }
    if (script_editor_) {
        dock_style.RegisterPanel("Script Editor", ICON_FA_CODE, script_editor_->GetVisiblePtr());
    }

    // Side panels
    if (asset_browser_) {
        dock_style.RegisterPanel("Asset Browser", ICON_FA_IMAGES, asset_browser_->GetVisiblePtr());
    }
    if (properties_) {
        dock_style.RegisterPanel("Properties", ICON_FA_SLIDERS, properties_->GetVisiblePtr());
    }

    // Bottom panels
    if (console_) {
        dock_style.RegisterPanel("Console", ICON_FA_TERMINAL, console_->GetVisiblePtr());
    }
    if (command_window_) {
        dock_style.RegisterPanel("Command", ICON_FA_CHEVRON_RIGHT, command_window_->GetVisiblePtr());
    }
    if (training_plot_panel_) {
        dock_style.RegisterPanel("Training", ICON_FA_CHART_LINE, training_plot_panel_->GetVisiblePtr());
    }
    if (viewport_) {
        dock_style.RegisterPanel("Viewport", ICON_FA_CUBES, viewport_->GetVisiblePtr());
    }

    // Additional panels (less commonly used)
    if (table_viewer_) {
        dock_style.RegisterPanel("Table Viewer", ICON_FA_TABLE, table_viewer_->GetVisiblePtr());
    }
    if (data_explorer_panel_) {
        dock_style.RegisterPanel("Data Explorer", ICON_FA_DATABASE, data_explorer_panel_->GetVisiblePtr());
    }
    if (annotation_editor_panel_) {
        dock_style.RegisterPanel("Annotation Editor", ICON_FA_DRAW_POLYGON, annotation_editor_panel_->GetVisiblePtr());
    }
    if (data_studio_panel_) {
        dock_style.RegisterPanel("Data Studio", ICON_FA_DIAGRAM_PROJECT, data_studio_panel_->GetVisiblePtr());
    }
    if (visualization_panel_) {
        dock_style.RegisterPanel("Visualizer", ICON_FA_CHART_SIMPLE, visualization_panel_->GetVisiblePtr());
    }
    if (job_status_panel_) {
        dock_style.RegisterPanel("Jobs", ICON_FA_LIST_CHECK, job_status_panel_->GetVisiblePtr());
    }
    if (p2p_training_panel_) {
        dock_style.RegisterPanel("P2P Training", ICON_FA_NETWORK_WIRED, p2p_training_panel_->GetVisiblePtr());
    }
    if (wallet_panel_) {
        dock_style.RegisterPanel("Wallet", ICON_FA_WALLET, wallet_panel_->GetVisiblePtr());
    }
    if (task_progress_panel_) {
        dock_style.RegisterPanel("Tasks", ICON_FA_SPINNER, task_progress_panel_->GetVisiblePtr());
    }
    if (pattern_browser_) {
        dock_style.RegisterPanel("Patterns", ICON_FA_CUBES, pattern_browser_->GetVisiblePtr());
    }
    if (query_console_) {
        dock_style.RegisterPanel("Query Console", ICON_FA_TERMINAL, query_console_->GetVisiblePtr());
    }
    if (variable_explorer_) {
        dock_style.RegisterPanel("Variable Explorer", ICON_FA_LIST_UL, variable_explorer_->GetVisiblePtr());
    }
    if (plot_output_panel_) {
        dock_style.RegisterPanel("Plot Output", ICON_FA_CHART_LINE, plot_output_panel_->GetVisiblePtr());
    }
    if (cloud_browser_panel_) {
        dock_style.RegisterPanel("Cloud Browser", ICON_FA_CLOUD, cloud_browser_panel_->GetVisiblePtr());
    }
    if (node_browser_panel_) {
        dock_style.RegisterPanel("Node Browser", ICON_FA_CUBES, node_browser_panel_->GetVisiblePtr());
    }
    if (node_info_panel_) {
        dock_style.RegisterPanel("Node Info", ICON_FA_CIRCLE_INFO, node_info_panel_->GetVisiblePtr());
    }
    if (cloud_dataset_manager_panel_) {
        dock_style.RegisterPanel("Cloud Manager", ICON_FA_DATABASE, cloud_dataset_manager_panel_->GetVisiblePtr());
    }

    // Plugin Manager panel
    if (plugin_manager_panel_) {
        dock_style.RegisterPanel("Plugin Manager", ICON_FA_PLUG, plugin_manager_panel_->GetVisiblePtr());
    }

    // Command Palette - action button (not a panel toggle)
    if (toolbar_) {
        dock_style.RegisterPanel("Command Palette", ICON_FA_MAGNIFYING_GLASS, nullptr, [this]() {
            toolbar_->OpenCommandPalette();
        });
    }

    spdlog::info("Registered {} panels with sidebar", dock_style.GetPanels().size());
}

void MainWindow::SetDefaultPanelVisibility() {
    // Only set defaults on first launch (no imgui.ini exists)
    if (!first_time_layout_) {
        return;
    }

    spdlog::info("Setting default panel visibility (first launch)");

    // === CORE PANELS ===
    // Original panels (node_editor_, console_, properties_, viewport_) don't have SetVisible
    // They use ImGui window visibility directly. We leave them alone.
    // New-style panels that should be visible:
    if (asset_browser_) asset_browser_->SetVisible(true);

    // === HIDE ALL OTHER PANELS ===
    // Users can enable these via View menu when needed

    // Main panels - hide by default
    if (training_plot_panel_) training_plot_panel_->SetVisible(false);
    if (plot_test_control_) plot_test_control_->SetVisible(false);
    if (command_window_) command_window_->SetVisible(false);
    if (script_editor_) script_editor_->SetVisible(false);
    if (table_viewer_) table_viewer_->SetVisible(false);
    if (job_status_panel_) job_status_panel_->SetVisible(false);
    if (p2p_training_panel_) p2p_training_panel_->SetVisible(false);
    if (wallet_panel_) wallet_panel_->SetVisible(false);
    if (task_progress_panel_) task_progress_panel_->SetVisible(false);
    if (pattern_browser_) pattern_browser_->SetVisible(false);
    if (query_console_) query_console_->SetVisible(false);
    if (custom_node_editor_) custom_node_editor_->SetVisible(false);
    if (theme_editor_) theme_editor_->SetVisible(false);
    if (profiling_panel_) profiling_panel_->SetVisible(false);
    if (memory_panel_) memory_panel_->SetVisible(false);
    if (memory_monitor_) memory_monitor_->SetVisible(false);
    if (variable_explorer_) variable_explorer_->SetVisible(false);
    if (plot_output_panel_) plot_output_panel_->SetVisible(false);
    if (test_results_panel_) test_results_panel_->SetVisible(false);

    // Model Analysis panels (Phase 2)
    if (model_summary_panel_) model_summary_panel_->SetVisible(false);
    if (architecture_diagram_) architecture_diagram_->SetVisible(false);
    if (lr_finder_panel_) lr_finder_panel_->SetVisible(false);

    // Data Science panels (Phase 3)
    if (data_profiler_panel_) data_profiler_panel_->SetVisible(false);
    if (correlation_matrix_panel_) correlation_matrix_panel_->SetVisible(false);
    if (missing_value_panel_) missing_value_panel_->SetVisible(false);
    if (outlier_detection_panel_) outlier_detection_panel_->SetVisible(false);

    // Statistics panels (Phase 4)
    if (descriptive_stats_panel_) descriptive_stats_panel_->SetVisible(false);
    if (hypothesis_test_panel_) hypothesis_test_panel_->SetVisible(false);
    if (distribution_fitter_panel_) distribution_fitter_panel_->SetVisible(false);
    if (regression_panel_) regression_panel_->SetVisible(false);

    // Advanced Tools panels (Phase 5)
    if (dim_reduction_panel_) dim_reduction_panel_->SetVisible(false);
    if (gradcam_panel_) gradcam_panel_->SetVisible(false);
    if (feature_importance_panel_) feature_importance_panel_->SetVisible(false);
    if (nas_panel_) nas_panel_->SetVisible(false);

    // Clustering panels (Phase 6A)
    if (kmeans_panel_) kmeans_panel_->SetVisible(false);
    if (dbscan_panel_) dbscan_panel_->SetVisible(false);
    if (hierarchical_panel_) hierarchical_panel_->SetVisible(false);
    if (gmm_panel_) gmm_panel_->SetVisible(false);
    if (cluster_eval_panel_) cluster_eval_panel_->SetVisible(false);

    // Model Evaluation panels (Phase 6B)
    if (confusion_matrix_panel_) confusion_matrix_panel_->SetVisible(false);
    if (roc_auc_panel_) roc_auc_panel_->SetVisible(false);
    if (pr_curve_panel_) pr_curve_panel_->SetVisible(false);
    if (cross_validation_panel_) cross_validation_panel_->SetVisible(false);
    if (learning_curves_panel_) learning_curves_panel_->SetVisible(false);

    // Data Transformation panels (Phase 6C)
    if (normalization_panel_) normalization_panel_->SetVisible(false);
    if (standardization_panel_) standardization_panel_->SetVisible(false);
    if (log_transform_panel_) log_transform_panel_->SetVisible(false);
    if (boxcox_panel_) boxcox_panel_->SetVisible(false);
    if (feature_scaling_panel_) feature_scaling_panel_->SetVisible(false);

    // Linear Algebra panels (Phase 7)
    if (matrix_calculator_panel_) matrix_calculator_panel_->SetVisible(false);
    if (eigen_decomp_panel_) eigen_decomp_panel_->SetVisible(false);
    if (svd_panel_) svd_panel_->SetVisible(false);
    if (qr_panel_) qr_panel_->SetVisible(false);
    if (cholesky_panel_) cholesky_panel_->SetVisible(false);

    // Signal Processing panels (Phase 8)
    if (fft_panel_) fft_panel_->SetVisible(false);
    if (spectrogram_panel_) spectrogram_panel_->SetVisible(false);
    if (filter_designer_panel_) filter_designer_panel_->SetVisible(false);
    if (convolution_panel_) convolution_panel_->SetVisible(false);
    if (wavelet_panel_) wavelet_panel_->SetVisible(false);

    // Optimization & Calculus panels (Phase 9)
    if (gradient_descent_panel_) gradient_descent_panel_->SetVisible(false);
    if (convexity_panel_) convexity_panel_->SetVisible(false);
    if (lp_panel_) lp_panel_->SetVisible(false);
    if (qp_panel_) qp_panel_->SetVisible(false);
    if (differentiation_panel_) differentiation_panel_->SetVisible(false);
    if (integration_panel_) integration_panel_->SetVisible(false);

    // Time Series Analysis panels (Phase 10)
    if (decomposition_panel_) decomposition_panel_->SetVisible(false);
    if (acf_pacf_panel_) acf_pacf_panel_->SetVisible(false);
    if (stationarity_panel_) stationarity_panel_->SetVisible(false);
    if (seasonality_panel_) seasonality_panel_->SetVisible(false);
    if (forecasting_panel_) forecasting_panel_->SetVisible(false);

    // Text Processing panels (Phase 11)
    if (tokenization_panel_) tokenization_panel_->SetVisible(false);
    if (word_frequency_panel_) word_frequency_panel_->SetVisible(false);
    if (tfidf_panel_) tfidf_panel_->SetVisible(false);
    if (embeddings_panel_) embeddings_panel_->SetVisible(false);
    if (sentiment_panel_) sentiment_panel_->SetVisible(false);

    // Utilities panels (Phase 12)
    if (calculator_panel_) calculator_panel_->SetVisible(false);
    if (unit_converter_panel_) unit_converter_panel_->SetVisible(false);
    if (random_generator_panel_) random_generator_panel_->SetVisible(false);
    if (hash_generator_panel_) hash_generator_panel_->SetVisible(false);
    if (json_viewer_panel_) json_viewer_panel_->SetVisible(false);
    if (regex_tester_panel_) regex_tester_panel_->SetVisible(false);
    
    // Cloud panels (hidden by default - access via sidebar/view menu)
    if (cloud_browser_panel_) cloud_browser_panel_->SetVisible(false);
    if (cloud_dataset_manager_panel_) cloud_dataset_manager_panel_->SetVisible(false);

    // Plugin Manager (hidden by default)
    if (plugin_manager_panel_) plugin_manager_panel_->SetVisible(false);

    spdlog::info("Default panel visibility set - showing only core panels");
}

void MainWindow::StartTrainingFromGraph(const std::vector<MLNode>& nodes, const std::vector<NodeLink>& links) {
    try {
        spdlog::info("StartTrainingFromGraph: Compiling {} nodes, {} links", nodes.size(), links.size());

        // Compile the graph
        cyxwiz::GraphCompiler compiler;
        cyxwiz::TrainingConfiguration config = compiler.Compile(nodes, links);

    if (!config.is_valid) {
        spdlog::error("Graph compilation failed: {}", config.error_message);
        return;
    }

    spdlog::info("Graph compiled successfully: {} layers, input={}, output={}",
                 config.layers.size(), config.input_size, config.output_size);

    // Get dataset from DataRegistry (loaded by DataInput node)
    auto& registry = cyxwiz::DataRegistry::Instance();

    // Find the dataset name from the DataInput node parameters
    std::string dataset_name;
    for (const auto& node : nodes) {
        if (node.type == NodeType::DataInput || node.type == NodeType::DatasetInput) {
            auto it = node.parameters.find("dataset_name");
            if (it != node.parameters.end() && !it->second.empty()) {
                dataset_name = it->second;
                break;
            }
        }
    }

    if (dataset_name.empty()) {
        spdlog::error("No dataset loaded. Please configure the Data Input node first.");
        return;
    }

    // Get label column from DataInput node parameters
    std::string label_column;
    for (const auto& node : nodes) {
        if (node.type == NodeType::DataInput) {
            auto it = node.parameters.find("label_column");
            if (it != node.parameters.end() && !it->second.empty()) {
                label_column = it->second;
            }
            break;
        }
    }

    // batch_size / shuffle / drop_last now live on the DataLoader node and are
    // extracted by GraphCompiler into the TrainingConfiguration. If no DataLoader
    // is present, config.batch_size already holds the default (32).
    int batch_size = config.batch_size;

    // epochs still lives on the optimizer node for this pass - it will move to a
    // global Training Settings panel later (see Training Dashboard work).
    int epochs = 10;
    for (const auto& node : nodes) {
        if (node.type == NodeType::Adam || node.type == NodeType::SGD ||
            node.type == NodeType::AdamW || node.type == NodeType::RMSprop) {
            auto it = node.parameters.find("epochs");
            if (it != node.parameters.end() && !it->second.empty()) {
                try { epochs = std::stoi(it->second); } catch (...) {}
            }
            // Legacy: batch_size on optimizer node is deprecated in favor of DataLoader.
            // Honor it only if no DataLoader node is present, and log a warning.
            auto bs_it = node.parameters.find("batch_size");
            if (bs_it != node.parameters.end() && !bs_it->second.empty()) {
                if (config.has_data_loader) {
                    spdlog::warn("batch_size is set on the optimizer node AND a DataLoader node is "
                                 "present - using DataLoader's value ({}). Remove batch_size from "
                                 "the optimizer node to clear this warning.", batch_size);
                } else {
                    spdlog::warn("batch_size on the optimizer node is deprecated - move it to a "
                                 "DataLoader node. Honoring legacy value for now.");
                    try { batch_size = std::stoi(bs_it->second); } catch (...) {}
                }
            }
            break;
        }
    }

    // Start training using TrainingManager
    auto& tm = cyxwiz::TrainingManager::Instance();

    // Set up node editor callback to update training animation
    auto node_editor_callback = [this](bool is_training) {
        if (node_editor_) {
            node_editor_->SetTrainingActive(is_training);
        }
    };

    bool started = false;

    // Dispatch by dataset type. In-memory Arrow is the default fast path;
    // disk-backed Parquet kicks in when LoadTabularCSV decided the CSV was
    // too big to fit in RAM (see DataRegistry::LoadTabularCSV auto-detect).
    // Legacy DatasetHandle is the pre-Arrow path, kept for back-compat
    // until the Arrow migration is complete.
    if (registry.IsArrowDataset(dataset_name)) {
        auto arrow_dataset = registry.GetArrowDataset(dataset_name);
        if (arrow_dataset) {
            spdlog::info("Starting Arrow-based training: dataset={}, epochs={}, batch_size={}, label={}",
                         dataset_name, epochs, batch_size, label_column);

            started = tm.StartTrainingArrow(
                std::move(config),
                arrow_dataset,
                label_column,
                epochs,
                batch_size,
                training_plot_panel_.get(),
                node_editor_callback
            );
        } else {
            spdlog::error("Arrow dataset '{}' is registered but could not be retrieved", dataset_name);
        }
    } else if (registry.IsParquetBackedDataset(dataset_name)) {
        auto parquet_dataset = registry.GetParquetBackedDataset(dataset_name);
        if (parquet_dataset) {
            spdlog::info("Starting Parquet-backed training: dataset={}, epochs={}, batch_size={}, "
                         "label={}, {:.1f} MB cache on disk",
                         dataset_name, epochs, batch_size, label_column,
                         parquet_dataset->GetFileSizeBytes() / (1024.0 * 1024.0));

            started = tm.StartTrainingParquet(
                std::move(config),
                parquet_dataset,
                label_column,
                epochs,
                batch_size,
                training_plot_panel_.get(),
                node_editor_callback
            );
        } else {
            spdlog::error("Parquet-backed dataset '{}' is registered but could not be retrieved", dataset_name);
        }
    } else {
        // Fall back to legacy DatasetHandle path
        auto dataset = registry.GetDataset(dataset_name);
        if (!dataset) {
            spdlog::error("Dataset '{}' not found in registry. Load data first.", dataset_name);
            return;
        }

        spdlog::info("Starting legacy training: dataset={}, epochs={}, batch_size={}",
                     dataset_name, epochs, batch_size);

        started = tm.StartTraining(
            std::move(config),
            dataset,
            epochs,
            batch_size,
            training_plot_panel_.get(),
            node_editor_callback
        );
    }

    if (started) {
        spdlog::info("Training started successfully");
        if (training_plot_panel_) {
            training_plot_panel_->SetVisible(true);
        }
    } else {
        spdlog::error("Failed to start training - another training session may be active");
    }

    } catch (const std::exception& e) {
        spdlog::error("StartTrainingFromGraph exception: {}", e.what());
    } catch (...) {
        spdlog::error("StartTrainingFromGraph unknown exception");
    }
}

// ---------------------------------------------------------------------------
// Compile Graph (dry-run) — validates and compiles the node graph without
// starting training, then shows the result in a modal popup. Lets users catch
// config errors before committing to a full training run.
// ---------------------------------------------------------------------------

void MainWindow::CompileGraphAndReport() {
    if (!node_editor_) {
        compile_result_success_ = false;
        compile_result_message_ = "Node editor is not available.";
        show_compile_result_popup_ = true;
        return;
    }

    auto nodes = node_editor_->GetNodes();
    auto links = node_editor_->GetLinks();

    spdlog::info("CompileGraphAndReport: compiling {} nodes, {} links", nodes.size(), links.size());

    try {
        cyxwiz::GraphCompiler compiler;
        cyxwiz::TrainingConfiguration config = compiler.Compile(nodes, links);

        std::ostringstream out;
        if (!config.is_valid) {
            compile_result_success_ = false;
            out << "Compilation failed:\n\n" << config.error_message;
            spdlog::error("CompileGraphAndReport: {}", config.error_message);
        } else {
            compile_result_success_ = true;
            out << "Compilation successful.\n\n";
            out << "Layers:          " << config.layers.size() << "\n";
            out << "Input size:      " << config.input_size;
            if (!config.input_shape.empty()) {
                out << "  (shape [";
                for (size_t i = 0; i < config.input_shape.size(); ++i) {
                    if (i) out << ", ";
                    out << config.input_shape[i];
                }
                out << "])";
            }
            out << "\n";
            out << "Output size:     " << config.output_size << "\n";
            out << "\n";

            // Read dataset_name directly from the DataInput node parameters.
            // GraphCompiler reads parameters["dataset"] (old key) while the
            // DataInput dialog writes parameters["dataset_name"] (new key),
            // so config.dataset_name is often empty. Mirror the same lookup
            // that StartTrainingFromGraph uses — that path always works.
            std::string dataset_name_from_graph;
            for (const auto& node : nodes) {
                if (node.type == NodeType::DataInput || node.type == NodeType::DatasetInput) {
                    auto it = node.parameters.find("dataset_name");
                    if (it != node.parameters.end() && !it->second.empty()) {
                        dataset_name_from_graph = it->second;
                        break;
                    }
                }
            }

            out << "Dataset:         "
                << (dataset_name_from_graph.empty() ? "<not set in DataInput>" : dataset_name_from_graph)
                << "\n";

            // Dataset backing — tell the user whether this dataset will
            // train from an in-memory Arrow table or a disk-backed Parquet
            // cache, and show the memory footprint so they know what to
            // expect when they click Train.
            if (!dataset_name_from_graph.empty()) {
                auto& reg = cyxwiz::DataRegistry::Instance();
                if (reg.IsArrowDataset(dataset_name_from_graph)) {
                    auto arrow_ds = reg.GetArrowDataset(dataset_name_from_graph);
                    if (arrow_ds) {
                        out << "Backing:         In-memory Arrow ("
                            << (arrow_ds->GetMemoryUsage() / (1024.0 * 1024.0))
                            << " MB resident)\n";
                    }
                } else if (reg.IsParquetBackedDataset(dataset_name_from_graph)) {
                    auto pq_ds = reg.GetParquetBackedDataset(dataset_name_from_graph);
                    if (pq_ds) {
                        out << "Backing:         Disk-backed Parquet ("
                            << (pq_ds->GetFileSizeBytes() / (1024.0 * 1024.0))
                            << " MB on disk, "
                            << pq_ds->GetNumRowGroups() << " row groups)\n";
                    }
                } else {
                    out << "Backing:         <not registered — Apply the DataInput node first>\n";
                }
            }

            out << "Split ratios:    train=" << config.train_ratio
                << ", val=" << config.val_ratio
                << ", test=" << config.test_ratio
                << (config.has_data_split ? " (from DataSplit node)" : " (default, no DataSplit node)")
                << "\n";

            out << "Batching:        batch_size=" << config.batch_size
                << ", shuffle=" << (config.shuffle ? "true" : "false")
                << ", drop_last=" << (config.drop_last ? "true" : "false")
                << (config.has_data_loader ? " (from DataLoader node)" : " (default, no DataLoader node)")
                << "\n";

            if (config.num_workers > 0) {
                out << "                 num_workers=" << config.num_workers
                    << " (requested — not yet implemented)\n";
            }

            out << "\n";
            out << "Loss:            " << config.GetLossName() << "\n";
            out << "Optimizer:       " << config.GetOptimizerName()
                << " (lr=" << config.learning_rate << ")\n";

            if (config.preprocessing.has_normalization) {
                out << "Preprocessing:   Normalize(mean=" << config.preprocessing.norm_mean
                    << ", std=" << config.preprocessing.norm_std << ")\n";
            }
            if (config.preprocessing.has_onehot) {
                out << "                 One-hot encoding ("
                    << config.preprocessing.num_classes << " classes)\n";
            }

            spdlog::info("CompileGraphAndReport: success - {} layers, input={}, output={}, "
                         "batch_size={}, shuffle={}, split={:.2f}/{:.2f}/{:.2f}",
                         config.layers.size(), config.input_size, config.output_size,
                         config.batch_size, config.shuffle,
                         config.train_ratio, config.val_ratio, config.test_ratio);
        }

        compile_result_message_ = out.str();
    } catch (const std::exception& e) {
        compile_result_success_ = false;
        compile_result_message_ = std::string("Compilation threw an exception:\n\n") + e.what();
        spdlog::error("CompileGraphAndReport exception: {}", e.what());
    } catch (...) {
        compile_result_success_ = false;
        compile_result_message_ = "Compilation threw an unknown exception.";
        spdlog::error("CompileGraphAndReport unknown exception");
    }

    show_compile_result_popup_ = true;
}

void MainWindow::RenderCompileResultPopup() {
    if (show_compile_result_popup_) {
        ImGui::OpenPopup("Compile Graph Result");
        show_compile_result_popup_ = false;
    }

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(620, 440), ImGuiCond_Appearing);

    if (ImGui::BeginPopupModal("Compile Graph Result", nullptr, ImGuiWindowFlags_NoResize)) {
        if (compile_result_success_) {
            ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.4f, 1.0f), "%s", "[OK]  Graph is valid and ready to train");
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", "[FAIL]  Graph has errors — see below");
        }
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::BeginChild("CompileResultText",
                          ImVec2(0, -ImGui::GetFrameHeightWithSpacing() - 8),
                          true);
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextUnformatted(compile_result_message_.c_str());
        ImGui::PopTextWrapPos();
        ImGui::EndChild();

        ImGui::Spacing();
        float button_width = 120.0f;
        ImGui::SetCursorPosX((ImGui::GetWindowSize().x - button_width) * 0.5f);
        if (ImGui::Button("OK", ImVec2(button_width, 0)) || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
}

void MainWindow::StartTestingFromGraph(const std::vector<MLNode>& nodes, const std::vector<NodeLink>& links) {
    spdlog::info("StartTestingFromGraph: Compiling {} nodes, {} links", nodes.size(), links.size());

    // Compile the graph (same as training)
    cyxwiz::GraphCompiler compiler;
    cyxwiz::TrainingConfiguration config = compiler.Compile(nodes, links);

    if (!config.is_valid) {
        spdlog::error("Graph compilation failed: {}", config.error_message);
        return;
    }

    spdlog::info("Graph compiled successfully: {} layers, input={}, output={}",
                 config.layers.size(), config.input_size, config.output_size);

    spdlog::warn("Local testing from node graph is temporarily disabled.");
    spdlog::info("To test your model:");
    spdlog::info("  1. Use P2P Training panel for remote testing on compute nodes");
    spdlog::info("  2. Or use 'Generate Code' to export and run locally");
    (void)config;  // Suppress unused variable warning
}

void MainWindow::RenderSidebar() {
    // Render the Unreal-style sidebar (auto-hides, appears on hover)
    GetDockStyle().RenderSidebarToggles();
}


void MainWindow::DetectKeyboardContext() {
    auto& kb = KeyboardShortcutManager::Instance();

    // Check for modal dialogs first (highest priority)
    if (toolbar_) {
        if (toolbar_->IsFindDialogOpen() || toolbar_->IsReplaceDialogOpen() ||
            toolbar_->IsFindInFilesDialogOpen() || toolbar_->IsReplaceInFilesDialogOpen() ||
            toolbar_->IsCommandPaletteOpen()) {
            kb.SetActiveContext(KeyboardContext::ModalDialog);
            return;
        }
    }

    // Check for completion popup in script editor
    if (script_editor_ && script_editor_->IsCompletionPopupOpen()) {
        kb.SetActiveContext(KeyboardContext::CompletionPopup);
        return;
    }

    // Check which panel is focused using panel-level focus tracking
    // This is more reliable than window name matching since panels track child window focus

    // Script Editor - uses IsFocused() which tracks child windows (TextEditor)
    if (script_editor_ && script_editor_->IsFocused()) {
        kb.SetActiveContext(KeyboardContext::ScriptEditor);
        return;
    }

    // For other panels, fall back to window name matching
    if (ImGui::IsWindowFocused(ImGuiFocusedFlags_AnyWindow)) {
        // Get the focused window name
        ImGuiWindow* focused = ImGui::GetCurrentContext()->NavWindow;
        if (focused) {
            std::string window_name = focused->Name ? focused->Name : "";

            // CyxWiz Studio detection
            if (window_name.find("CyxWiz Studio") != std::string::npos ||
                window_name.find("##NodeEditor") != std::string::npos) {
                kb.SetActiveContext(KeyboardContext::NodeEditor);
                return;
            }

            // Console detection
            if (window_name.find("Console") != std::string::npos ||
                window_name.find("Python Console") != std::string::npos) {
                kb.SetActiveContext(KeyboardContext::Console);
                return;
            }

            // Asset Browser detection
            if (window_name.find("Asset") != std::string::npos ||
                window_name.find("Project") != std::string::npos) {
                kb.SetActiveContext(KeyboardContext::AssetBrowser);
                return;
            }

            // Table Viewer detection
            if (window_name.find("Table") != std::string::npos) {
                kb.SetActiveContext(KeyboardContext::TableViewer);
                return;
            }
        }
    }

    // Default to main window context
    kb.SetActiveContext(KeyboardContext::MainWindow);
}

void MainWindow::HandleGlobalShortcuts() {
    // First, detect which context is active
    DetectKeyboardContext();

    auto& kb = KeyboardShortcutManager::Instance();
    KeyboardContext context = kb.GetActiveContext();

    ImGuiIO& io = ImGui::GetIO();
    bool ctrl = io.KeyCtrl;
    bool shift = io.KeyShift;
    bool alt = io.KeyAlt;

    // Modal dialogs block all shortcuts (except Escape which is handled by the dialog)
    if (context == KeyboardContext::ModalDialog) {
        return;
    }

    // Completion popup has its own keyboard handling in script_editor
    if (context == KeyboardContext::CompletionPopup) {
        return;
    }

    // ========================================================================
    // GLOBAL SHORTCUTS - Work in any context
    // ========================================================================

    // Python Console (F12) - Always available
    if (!ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_F12)) {
        if (command_window_) {
            command_window_->SetVisible(true);
            spdlog::info("Opened Python Console via F12");
        }
    }

    // Compile Graph (F7) - dry-run the node graph without starting training
    if (!ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_F7)) {
        CompileGraphAndReport();
        spdlog::info("Compile Graph invoked via F7");
    }

    // Pattern Browser (Ctrl+Shift+P) - Always available
    if (ctrl && shift && !alt && ImGui::IsKeyPressed(ImGuiKey_P)) {
        if (pattern_browser_) {
            pattern_browser_->Toggle();
            spdlog::info("Toggled Pattern Browser via Ctrl+Shift+P");
        }
    }

    // ========================================================================
    // MAIN WINDOW CONTEXT - Only when no specific panel is focused
    // ========================================================================
    if (context == KeyboardContext::MainWindow) {
        // Command Palette (Ctrl+P)
        if (ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_P)) {
            if (toolbar_) {
                toolbar_->OpenCommandPalette();
                spdlog::info("Opened Command Palette via Ctrl+P");
            }
        }

        // Find in Files (Ctrl+Shift+F) - Project-wide search
        if (ctrl && shift && !alt && ImGui::IsKeyPressed(ImGuiKey_F)) {
            if (toolbar_) {
                toolbar_->OpenFindInFilesDialog();
                spdlog::info("Opened Find in Files dialog via Ctrl+Shift+F");
            }
        }

        // Replace in Files (Ctrl+Shift+H)
        if (ctrl && shift && !alt && ImGui::IsKeyPressed(ImGuiKey_H)) {
            if (toolbar_) {
                toolbar_->OpenReplaceInFilesDialog();
                spdlog::info("Opened Replace in Files dialog via Ctrl+Shift+H");
            }
        }
    }

    // ========================================================================
    // SCRIPT EDITOR CONTEXT - Only when script editor is focused
    // ========================================================================
    if (context == KeyboardContext::ScriptEditor) {
        // Find (Ctrl+F)
        if (ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_F)) {
            if (toolbar_) {
                toolbar_->OpenFindDialog();
                spdlog::info("Opened Find dialog via Ctrl+F");
            }
        }

        // Replace (Ctrl+H)
        if (ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_H)) {
            if (toolbar_) {
                toolbar_->OpenReplaceDialog();
                spdlog::info("Opened Replace dialog via Ctrl+H");
            }
        }

        // Toggle Line Comment (Ctrl+/)
        if (ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_Slash)) {
            if (script_editor_) {
                script_editor_->ToggleLineComment();
                spdlog::info("Toggled line comment via Ctrl+/");
            }
        }

        // Toggle Block Comment (Shift+Alt+A)
        if (!ctrl && shift && alt && ImGui::IsKeyPressed(ImGuiKey_A)) {
            if (script_editor_) {
                script_editor_->ToggleBlockComment();
                spdlog::info("Toggled block comment via Shift+Alt+A");
            }
        }

        // Go to Line (Ctrl+G)
        if (ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_G)) {
            // Opens the Go to Line dialog in the toolbar
        }

        // Duplicate Line (Ctrl+D)
        if (ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_D)) {
            if (script_editor_) {
                script_editor_->DuplicateLine();
                spdlog::info("Duplicated line via Ctrl+D");
            }
        }

        // Move Line Up (Alt+Up)
        if (!ctrl && !shift && alt && ImGui::IsKeyPressed(ImGuiKey_UpArrow)) {
            if (script_editor_) {
                script_editor_->MoveLineUp();
                spdlog::info("Moved line up via Alt+Up");
            }
        }

        // Move Line Down (Alt+Down)
        if (!ctrl && !shift && alt && ImGui::IsKeyPressed(ImGuiKey_DownArrow)) {
            if (script_editor_) {
                script_editor_->MoveLineDown();
                spdlog::info("Moved line down via Alt+Down");
            }
        }

        // Join Lines (Ctrl+J)
        if (ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_J)) {
            if (script_editor_) {
                script_editor_->JoinLines();
                spdlog::info("Joined lines via Ctrl+J");
            }
        }
    }

    // ========================================================================
    // NODE EDITOR CONTEXT - Only when node editor is focused
    // ========================================================================
    if (context == KeyboardContext::NodeEditor) {
        // Future: Add node editor specific shortcuts here
        // e.g., Ctrl+N for new node, Del for delete, etc.
    }

    // ========================================================================
    // CONSOLE CONTEXT - Only when console is focused
    // ========================================================================
    if (context == KeyboardContext::Console) {
        // Console handles its own shortcuts internally
    }
}

// ============================================================================
// Project Settings Persistence
// ============================================================================

void MainWindow::SaveLayout() {
    // Save to the default imgui.ini in the executable directory
    // This ensures consistent layout across all projects
    ImGui::SaveIniSettingsToDisk("imgui.ini");
    spdlog::info("Saved layout to imgui.ini");
}

void MainWindow::LoadLayout() {
    // Layout is loaded automatically from imgui.ini by ImGui
    // This function is kept for API compatibility but doesn't need to do anything
    // Per-project layouts are disabled to avoid dock corruption issues
}

void MainWindow::SaveProjectSettings() {
    auto& pm = cyxwiz::ProjectManager::Instance();
    if (!pm.HasActiveProject()) {
        spdlog::warn("Cannot save project settings: no active project");
        return;
    }

    // Get current editor settings from script editor
    cyxwiz::EditorSettings& settings = pm.GetConfig().editor_settings;
    if (script_editor_) {
        settings.theme = script_editor_->GetThemeIndex();
        settings.font_scale = script_editor_->GetFontScale();
        settings.tab_size = script_editor_->GetTabSize();
        settings.show_whitespace = script_editor_->GetShowWhitespace();
        settings.word_wrap = script_editor_->GetWordWrap();
        settings.auto_indent = script_editor_->GetAutoIndent();
        settings.syntax_highlighting = script_editor_->GetSyntaxHighlighting();

        // Persist open scripts for project restore
        pm.GetConfig().open_scripts.clear();
        auto open_paths = script_editor_->GetOpenFilePaths();
        for (const auto& path : open_paths) {
            pm.GetConfig().open_scripts.push_back(path);
        }
        pm.GetConfig().active_script_index = script_editor_->GetActiveTabIndex();
    }

    // Save application-wide settings
    settings.app_theme = static_cast<int>(GetTheme().GetCurrentPreset());
    settings.ui_scale = ImGui::GetIO().FontGlobalScale;

    // Save layout file
    SaveLayout();

    // Save project file (includes editor settings)
    pm.SaveProject();
    spdlog::info("Saved project settings");
}

void MainWindow::LoadProjectSettings() {
    auto& pm = cyxwiz::ProjectManager::Instance();
    if (!pm.HasActiveProject()) {
        return;
    }

    const cyxwiz::EditorSettings& settings = pm.GetConfig().editor_settings;

    // Apply editor settings to script editor
    if (script_editor_) {
        script_editor_->SetTheme(settings.theme);
        script_editor_->SetFontScale(settings.font_scale);
        script_editor_->SetTabSize(settings.tab_size);
        script_editor_->SetShowWhitespace(settings.show_whitespace);
        script_editor_->SetWordWrap(settings.word_wrap);
        script_editor_->SetAutoIndent(settings.auto_indent);
        script_editor_->SetSyntaxHighlighting(settings.syntax_highlighting);
    }

    // Sync settings to toolbar/preferences
    if (toolbar_) {
        toolbar_->SetEditorTheme(settings.theme);
        toolbar_->SetEditorTabSize(settings.tab_size);
        toolbar_->SetEditorFontScale(settings.font_scale);
        toolbar_->SetEditorShowWhitespace(settings.show_whitespace);
        toolbar_->SetEditorWordWrap(settings.word_wrap);
        toolbar_->SetEditorAutoIndent(settings.auto_indent);
    }

    // Apply application theme
    if (settings.app_theme >= 0 && settings.app_theme < static_cast<int>(ThemePreset::COUNT)) {
        GetTheme().ApplyPreset(static_cast<ThemePreset>(settings.app_theme));
        spdlog::info("Loaded app theme from project: {}", settings.app_theme);
    }

    // Apply UI scale
    ImGui::GetIO().FontGlobalScale = settings.ui_scale;

    // Load layout file
    LoadLayout();

    // Restore open scripts
    const auto& open_scripts = pm.GetConfig().open_scripts;
    for (const auto& script_path : open_scripts) {
        std::string resolved_path = script_path;
        std::filesystem::path fs_path(script_path);
        if (fs_path.is_relative()) {
            resolved_path = pm.ResolveAssetPath(script_path);
        }
        if (script_editor_ && std::filesystem::exists(resolved_path)) {
            script_editor_->OpenFile(resolved_path);
        }
    }
    if (script_editor_ && !open_scripts.empty()) {
        script_editor_->SetActiveTabIndex(pm.GetConfig().active_script_index);
    }

    spdlog::info("Loaded project settings (theme={}, font_scale={:.1f}, tab_size={})",
                 settings.theme, settings.font_scale, settings.tab_size);
}

void MainWindow::OnProjectOpened(const std::string& project_root) {
    spdlog::info("Project opened: {}", project_root);

    // Load project settings and layout
    LoadProjectSettings();

    // If Python is already initialized, check for interpreter mismatch
    if (scripting_engine_ && scripting_engine_->IsInitialized()) {
        if (!scripting_engine_->ReloadPythonForProject()) {
            spdlog::warn("Python interpreter mismatch after project open (project: {})", project_root);
        }
    }

    // Set project root and refresh asset browser to show project files
    if (asset_browser_) {
        asset_browser_->SetProjectRoot(project_root);
        asset_browser_->Refresh();
    }
}

void MainWindow::OnProjectClosed(const std::string& project_root) {
    spdlog::info("Project closed: {}", project_root);

    // Note: Settings should be saved before CloseProject() is called
    // (the toolbar handles this in its Close Project menu action)

    // If Python is already initialized, check for interpreter mismatch after closing
    if (scripting_engine_ && scripting_engine_->IsInitialized()) {
        if (!scripting_engine_->ReloadPythonForProject()) {
            spdlog::warn("Python interpreter mismatch after project close");
        }
    }

    // Clear the asset browser
    if (asset_browser_) {
        asset_browser_->Clear();
    }

    // Drop every tabular dataset (Arrow + Parquet-backed) so the next
    // project starts with an empty registry. Without this, datasets from
    // the previous project linger and can collide with same-named entries
    // (auto-generated from file stem) in the new project, leading to
    // stale dispatch in StartTrainingFromGraph.
    cyxwiz::DataRegistry::Instance().ClearAllTabularDatasets();

    // Reset the dock layout to default when project is closed
    // This gives a clean slate for the next project
    first_time_layout_ = true;  // This will trigger BuildInitialDockLayout on next render
}

void MainWindow::OnProjectVenvReady(const std::string& project_root) {
    spdlog::info("Project venv ready: {}", project_root);

    auto& pm = cyxwiz::ProjectManager::Instance();
    if (!pm.HasActiveProject()) {
        spdlog::info("Skipping Python reload: no active project");
        return;
    }
    if (pm.GetProjectRoot() != project_root) {
        spdlog::info("Skipping Python reload: active project root differs");
        return;
    }

    if (scripting_engine_ && scripting_engine_->IsInitialized()) {
        if (!scripting_engine_->ReloadPythonForProject()) {
            spdlog::warn("Python interpreter mismatch after venv creation");
        }
    }
}

void MainWindow::RenderStatusBar() {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float status_bar_height = 24.0f;

    // Position status bar at the bottom of the screen
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x, viewport->WorkPos.y + viewport->WorkSize.y - status_bar_height));
    ImGui::SetNextWindowSize(ImVec2(viewport->WorkSize.x, status_bar_height));

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                             ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings |
                             ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoNav;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 4));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.15f, 0.15f, 0.15f, 1.0f));

    if (ImGui::Begin("##StatusBar", nullptr, flags)) {
        // Left side: Project info
        auto& pm = cyxwiz::ProjectManager::Instance();
        if (pm.HasActiveProject()) {
            ImGui::Text("%s %s", ICON_FA_FOLDER_OPEN, pm.GetProjectName().c_str());
        } else {
            ImGui::TextDisabled("%s No Project", ICON_FA_FOLDER);
        }

        // Center: Spacer
        ImGui::SameLine();
        (void)ImGui::GetContentRegionAvail().x; // Reserve for future use

        // Right side: Task indicator (if tasks are running)
        if (task_progress_panel_ && task_progress_panel_->HasActiveTasks()) {
            // Calculate position for right-aligned content
            std::string status = task_progress_panel_->GetStatusSummary();
            float text_width = ImGui::CalcTextSize(status.c_str()).x + 30; // + icon width

            ImGui::SameLine(ImGui::GetWindowWidth() - text_width - 16);

            // Show task indicator that opens the task panel when clicked
            if (cyxwiz::TaskStatusIndicator::Render()) {
                task_progress_panel_->Show();
            }

            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.3f, 0.7f, 1.0f, 1.0f), "%s", status.c_str());
        } else {
            // Show ready status on the right
            float text_width = ImGui::CalcTextSize("Ready").x + 20;
            ImGui::SameLine(ImGui::GetWindowWidth() - text_width - 16);
            ImGui::TextDisabled("%s Ready", ICON_FA_CIRCLE_CHECK);
        }
    }
    ImGui::End();

    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
}

} // namespace gui
