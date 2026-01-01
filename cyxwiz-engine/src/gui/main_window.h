#pragma once

#include <memory>
#include <string>
#include <vector>
#include <functional>

namespace cyxwiz {
class TaskProgressPanel;
} // namespace cyxwiz

namespace gui {

class NodeEditor;
class Console;
class Viewport;
class Properties;
class DatasetPanel;
class WalletPanel;
class CustomNodeEditorPanel;
class ThemeEditorPanel;
struct MLNode;
struct NodeLink;

} // namespace gui

namespace cyxwiz {
class ToolbarPanel;
class AssetBrowserPanel;
class TrainingDashboardPanel;
class TrainingPlotPanel;
class PlotTestControlPanel;
class CommandWindowPanel;
class ScriptEditorPanel;
class TableViewerPanel;
class DataExplorerPanel;
class VisualizationPanel;
class ConnectionDialog;
class JobStatusPanel;
class P2PTrainingPanel;
class PatternBrowserPanel;
class QueryConsolePanel;
class ProfilingPanel;
class MemoryPanel;
class MemoryMonitor;
class VariableExplorerPanel;
class TestResultsPanel;
class ExportDialog;
class ImportDialog;
class DeploymentDialog;
class ModelSummaryPanel;
class ArchitectureDiagram;
class LRFinderPanel;
class DataProfilerPanel;
class CorrelationMatrixPanel;
class MissingValuePanel;
class OutlierDetectionPanel;
class DescriptiveStatsPanel;
class HypothesisTestPanel;
class DistributionFitterPanel;
class RegressionPanel;
class DimReductionPanel;
class GradCAMPanel;
class FeatureImportancePanel;
class NASPanel;
class HyperparamSearchPanel;
class ServingPanel;
class KMeansPanel;
class DBSCANPanel;
class HierarchicalPanel;
class GMMPanel;
class ClusterEvalPanel;
// Model Evaluation panels (Phase 6B)
class ConfusionMatrixPanel;
class ROCAUCPanel;
class PRCurvePanel;
class CrossValidationPanel;
class LearningCurvesPanel;
// Data Transformation panels (Phase 6C)
class NormalizationPanel;
class StandardizationPanel;
class LogTransformPanel;
class BoxCoxPanel;
class FeatureScalingPanel;
// Linear Algebra panels (Phase 7)
class MatrixCalculatorPanel;
class EigenDecompPanel;
class SVDPanel;
class QRPanel;
class CholeskyPanel;
// Signal Processing panels (Phase 8)
class FFTPanel;
class SpectrogramPanel;
class FilterDesignerPanel;
class ConvolutionPanel;
class WaveletPanel;
// Optimization & Calculus panels (Phase 9)
class GradientDescentPanel;
class ConvexityPanel;
class LPPanel;
class QPPanel;
class DifferentiationPanel;
class IntegrationPanel;
// Time Series Analysis panels (Phase 10)
class DecompositionPanel;
class ACFPACFPanel;
class StationarityPanel;
class SeasonalityPanel;
class ForecastingPanel;
// Text Processing panels (Phase 11)
class TokenizationPanel;
class WordFrequencyPanel;
class TFIDFPanel;
class EmbeddingsPanel;
class SentimentPanel;
// Utilities panels (Phase 12)
class CalculatorPanel;
class UnitConverterPanel;
class RandomGeneratorPanel;
class HashGeneratorPanel;
class JSONViewerPanel;
class RegexTesterPanel;
} // namespace cyxwiz

namespace scripting {
class ScriptingEngine;
class StartupScriptManager;
} // namespace scripting

namespace network {
class GRPCClient;
class JobManager;
} // namespace network

namespace gui {

class MainWindow {
public:
    MainWindow();
    ~MainWindow();

    void Render();
    void ResetDockLayout();
    Console* GetConsole() { return console_.get(); }
    cyxwiz::PlotTestControlPanel* GetPlotTestControl() { return plot_test_control_.get(); }
    cyxwiz::ScriptEditorPanel* GetScriptEditor() { return script_editor_.get(); }

    // Set network components (called by Application after construction)
    void SetNetworkComponents(network::GRPCClient* client, network::JobManager* job_manager);

    // Get P2PTrainingPanel for job monitoring
    cyxwiz::P2PTrainingPanel* GetP2PTrainingPanel() { return p2p_training_panel_.get(); }

    // Start monitoring a job in P2PTrainingPanel
    void StartJobMonitoring(const std::string& job_id);

    // Check if a script is currently running
    bool IsScriptRunning() const;

    // Stop any running script
    void StopRunningScript();

    // Check for unsaved files in Script Editor
    bool HasUnsavedFiles() const;
    std::vector<std::string> GetUnsavedFileNames() const;
    void SaveAllFiles();

    // Get the scripting engine
    std::shared_ptr<scripting::ScriptingEngine> GetScriptingEngine() { return scripting_engine_; }

    // Prepare for shutdown - stops all background threads before destruction
    // Call this before destroying MainWindow for faster, cleaner shutdown
    void PrepareForShutdown();

    // Exit request callback (set by Application to trigger window close)
    using ExitRequestCallback = std::function<void()>;
    void SetExitRequestCallback(ExitRequestCallback callback) { exit_request_callback_ = callback; }

    // Debug logging pointers (set by Application)
    void SetIdleLogPtr(bool* ptr);

    // Project settings persistence
    void SaveProjectSettings();      // Save current editor settings and layout to project
    void LoadProjectSettings();      // Load editor settings and layout from project
    void SaveLayout();               // Save only ImGui layout to project
    void LoadLayout();               // Load only ImGui layout from project

    // Called when project is opened/closed
    void OnProjectOpened(const std::string& project_root);
    void OnProjectClosed(const std::string& project_root);

private:
    void RenderDockSpace();
    void BuildInitialDockLayout();
    void ShowAboutDialog();
    void RegisterPanelsWithSidebar();
    void SetDefaultPanelVisibility();  // Hide tool panels, show only core panels
    void RenderSidebar();
    void RenderStatusBar();
    void DetectKeyboardContext();
    void HandleGlobalShortcuts();

    // Training from node graph
    void StartTrainingFromGraph(const std::vector<MLNode>& nodes, const std::vector<NodeLink>& links);

    // Testing from node graph
    void StartTestingFromGraph(const std::vector<MLNode>& nodes, const std::vector<NodeLink>& links);

    // Original panels
    std::unique_ptr<NodeEditor> node_editor_;
    std::unique_ptr<Console> console_;
    std::unique_ptr<Viewport> viewport_;
    std::unique_ptr<Properties> properties_;
    std::unique_ptr<DatasetPanel> dataset_panel_;

    // New panel system
    std::unique_ptr<cyxwiz::ToolbarPanel> toolbar_;
    std::unique_ptr<cyxwiz::AssetBrowserPanel> asset_browser_;
    std::unique_ptr<cyxwiz::TrainingDashboardPanel> training_dashboard_;
    std::unique_ptr<cyxwiz::TrainingPlotPanel> training_plot_panel_;
    std::unique_ptr<cyxwiz::PlotTestControlPanel> plot_test_control_;
    std::unique_ptr<cyxwiz::CommandWindowPanel> command_window_;
    std::unique_ptr<cyxwiz::ScriptEditorPanel> script_editor_;
    std::unique_ptr<cyxwiz::TableViewerPanel> table_viewer_;
    std::unique_ptr<cyxwiz::DataExplorerPanel> data_explorer_panel_;
    std::unique_ptr<cyxwiz::VisualizationPanel> visualization_panel_;
    std::unique_ptr<cyxwiz::ConnectionDialog> connection_dialog_;
    std::unique_ptr<cyxwiz::JobStatusPanel> job_status_panel_;
    std::unique_ptr<cyxwiz::P2PTrainingPanel> p2p_training_panel_;
    std::unique_ptr<gui::WalletPanel> wallet_panel_;
    std::unique_ptr<cyxwiz::TaskProgressPanel> task_progress_panel_;
    std::unique_ptr<cyxwiz::PatternBrowserPanel> pattern_browser_;
    std::unique_ptr<cyxwiz::QueryConsolePanel> query_console_;
    std::unique_ptr<gui::CustomNodeEditorPanel> custom_node_editor_;
    std::unique_ptr<gui::ThemeEditorPanel> theme_editor_;
    std::unique_ptr<cyxwiz::ProfilingPanel> profiling_panel_;
    std::unique_ptr<cyxwiz::MemoryPanel> memory_panel_;
    std::unique_ptr<cyxwiz::MemoryMonitor> memory_monitor_;
    std::unique_ptr<cyxwiz::VariableExplorerPanel> variable_explorer_;
    std::unique_ptr<cyxwiz::TestResultsPanel> test_results_panel_;
    std::unique_ptr<cyxwiz::ExportDialog> export_dialog_;
    std::unique_ptr<cyxwiz::ImportDialog> import_dialog_;
    std::unique_ptr<cyxwiz::DeploymentDialog> deployment_dialog_;

    // Model Analysis panels (Phase 2)
    std::unique_ptr<cyxwiz::ModelSummaryPanel> model_summary_panel_;
    std::unique_ptr<cyxwiz::ArchitectureDiagram> architecture_diagram_;
    std::unique_ptr<cyxwiz::LRFinderPanel> lr_finder_panel_;

    // Data Science panels (Phase 3)
    std::unique_ptr<cyxwiz::DataProfilerPanel> data_profiler_panel_;
    std::unique_ptr<cyxwiz::CorrelationMatrixPanel> correlation_matrix_panel_;
    std::unique_ptr<cyxwiz::MissingValuePanel> missing_value_panel_;
    std::unique_ptr<cyxwiz::OutlierDetectionPanel> outlier_detection_panel_;

    // Statistics panels (Phase 4)
    std::unique_ptr<cyxwiz::DescriptiveStatsPanel> descriptive_stats_panel_;
    std::unique_ptr<cyxwiz::HypothesisTestPanel> hypothesis_test_panel_;
    std::unique_ptr<cyxwiz::DistributionFitterPanel> distribution_fitter_panel_;
    std::unique_ptr<cyxwiz::RegressionPanel> regression_panel_;

    // Advanced Tools panels (Phase 5)
    std::unique_ptr<cyxwiz::DimReductionPanel> dim_reduction_panel_;
    std::unique_ptr<cyxwiz::GradCAMPanel> gradcam_panel_;
    std::unique_ptr<cyxwiz::FeatureImportancePanel> feature_importance_panel_;
    std::unique_ptr<cyxwiz::NASPanel> nas_panel_;
    std::unique_ptr<cyxwiz::HyperparamSearchPanel> hyperparam_search_panel_;
    std::unique_ptr<cyxwiz::ServingPanel> serving_panel_;

    // Clustering panels (Phase 6A)
    std::unique_ptr<cyxwiz::KMeansPanel> kmeans_panel_;
    std::unique_ptr<cyxwiz::DBSCANPanel> dbscan_panel_;
    std::unique_ptr<cyxwiz::HierarchicalPanel> hierarchical_panel_;
    std::unique_ptr<cyxwiz::GMMPanel> gmm_panel_;
    std::unique_ptr<cyxwiz::ClusterEvalPanel> cluster_eval_panel_;

    // Model Evaluation panels (Phase 6B)
    std::unique_ptr<cyxwiz::ConfusionMatrixPanel> confusion_matrix_panel_;
    std::unique_ptr<cyxwiz::ROCAUCPanel> roc_auc_panel_;
    std::unique_ptr<cyxwiz::PRCurvePanel> pr_curve_panel_;
    std::unique_ptr<cyxwiz::CrossValidationPanel> cross_validation_panel_;
    std::unique_ptr<cyxwiz::LearningCurvesPanel> learning_curves_panel_;

    // Data Transformation panels (Phase 6C)
    std::unique_ptr<cyxwiz::NormalizationPanel> normalization_panel_;
    std::unique_ptr<cyxwiz::StandardizationPanel> standardization_panel_;
    std::unique_ptr<cyxwiz::LogTransformPanel> log_transform_panel_;
    std::unique_ptr<cyxwiz::BoxCoxPanel> boxcox_panel_;
    std::unique_ptr<cyxwiz::FeatureScalingPanel> feature_scaling_panel_;

    // Linear Algebra panels (Phase 7)
    std::unique_ptr<cyxwiz::MatrixCalculatorPanel> matrix_calculator_panel_;
    std::unique_ptr<cyxwiz::EigenDecompPanel> eigen_decomp_panel_;
    std::unique_ptr<cyxwiz::SVDPanel> svd_panel_;
    std::unique_ptr<cyxwiz::QRPanel> qr_panel_;
    std::unique_ptr<cyxwiz::CholeskyPanel> cholesky_panel_;

    // Signal Processing panels (Phase 8)
    std::unique_ptr<cyxwiz::FFTPanel> fft_panel_;
    std::unique_ptr<cyxwiz::SpectrogramPanel> spectrogram_panel_;
    std::unique_ptr<cyxwiz::FilterDesignerPanel> filter_designer_panel_;
    std::unique_ptr<cyxwiz::ConvolutionPanel> convolution_panel_;
    std::unique_ptr<cyxwiz::WaveletPanel> wavelet_panel_;

    // Optimization & Calculus panels (Phase 9)
    std::unique_ptr<cyxwiz::GradientDescentPanel> gradient_descent_panel_;
    std::unique_ptr<cyxwiz::ConvexityPanel> convexity_panel_;
    std::unique_ptr<cyxwiz::LPPanel> lp_panel_;
    std::unique_ptr<cyxwiz::QPPanel> qp_panel_;
    std::unique_ptr<cyxwiz::DifferentiationPanel> differentiation_panel_;
    std::unique_ptr<cyxwiz::IntegrationPanel> integration_panel_;

    // Time Series Analysis panels (Phase 10)
    std::unique_ptr<cyxwiz::DecompositionPanel> decomposition_panel_;
    std::unique_ptr<cyxwiz::ACFPACFPanel> acf_pacf_panel_;
    std::unique_ptr<cyxwiz::StationarityPanel> stationarity_panel_;
    std::unique_ptr<cyxwiz::SeasonalityPanel> seasonality_panel_;
    std::unique_ptr<cyxwiz::ForecastingPanel> forecasting_panel_;

    // Text Processing panels (Phase 11)
    std::unique_ptr<cyxwiz::TokenizationPanel> tokenization_panel_;
    std::unique_ptr<cyxwiz::WordFrequencyPanel> word_frequency_panel_;
    std::unique_ptr<cyxwiz::TFIDFPanel> tfidf_panel_;
    std::unique_ptr<cyxwiz::EmbeddingsPanel> embeddings_panel_;
    std::unique_ptr<cyxwiz::SentimentPanel> sentiment_panel_;

    // Utilities panels (Phase 12)
    std::unique_ptr<cyxwiz::CalculatorPanel> calculator_panel_;
    std::unique_ptr<cyxwiz::UnitConverterPanel> unit_converter_panel_;
    std::unique_ptr<cyxwiz::RandomGeneratorPanel> random_generator_panel_;
    std::unique_ptr<cyxwiz::HashGeneratorPanel> hash_generator_panel_;
    std::unique_ptr<cyxwiz::JSONViewerPanel> json_viewer_panel_;
    std::unique_ptr<cyxwiz::RegexTesterPanel> regex_tester_panel_;

    // Scripting engine (shared between panels)
    std::shared_ptr<scripting::ScriptingEngine> scripting_engine_;

    // Startup script manager
    std::unique_ptr<scripting::StartupScriptManager> startup_script_manager_;

    bool show_about_dialog_;
    bool show_demo_window_;
    bool first_time_layout_;

    // Network components
    network::JobManager* job_manager_ = nullptr;
    std::string monitoring_job_id_;

    // Exit request callback
    ExitRequestCallback exit_request_callback_;
};

} // namespace gui
