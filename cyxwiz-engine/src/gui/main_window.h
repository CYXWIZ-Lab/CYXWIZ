#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <functional>
#include <map>
#include <utility>
#include "../core/studio_toolbar_presentation.h"
#include "../core/graph_compiler.h"  // for cyxwiz::ValidationIssue
#include "../core/compile_report_presentation.h"
#include "../core/cross_validation_runner.h"
#include "graph_training_launcher.h"

namespace cyxwiz {
class ProjectManager;
class TaskProgressPanel;
class AsyncTask;
struct DebugArtifactConsistencyInput;
} // namespace cyxwiz

namespace cyxwiz {
class LiveGraphCompile;
class PythonNodeProvider;
}  // namespace cyxwiz

namespace gui {

class NodeEditor;
class DataInputDialog;
class Console;
class Viewport;
class Properties;
class WalletPanel;
class CustomNodeEditorPanel;
class ThemeEditorPanel;
class CloudBrowserPanel;
class CloudDatasetManagerPanel;
class NodeBrowserPanel;
struct MLNode;
struct NodeLink;

} // namespace gui

namespace cyxwiz {
class ToolbarPanel;
class AssetBrowserPanel;
class TrainingPlotPanel;
class ScriptEditorPanel;
class TableViewerPanel;
class AnnotationEditorPanel;
class ConnectionDialog;
class JobStatusPanel;
class P2PTrainingPanel;
class PatternBrowserPanel;
class QueryConsolePanel;
class MemoryPanel;
class MemoryMonitor;
class VariableExplorerPanel;
class PlotOutputPanel;
class ExportDialog;
class ImportDialog;
class DeploymentDialog;
class ModelSummaryPanel;
class ArchitectureDiagram;
class LRFinderPanel;
class StudioDebuggerPanel;
class DimReductionPanel;
class GradCAMPanel;
class FeatureImportancePanel;
class NASPanel;
class HyperparamSearchPanel;
class ServingPanel;
class DNNInferencePanel;
class KMeansPanel;
class DBSCANPanel;
class HierarchicalPanel;
class GMMPanel;
class ClusterEvalPanel;
// Model Evaluation panels (Phase 6B)
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
class LanguageModelGenerationPanel;
// Utilities panels (Phase 12)
class CalculatorPanel;
class UnitConverterPanel;
class RandomGeneratorPanel;
class HashGeneratorPanel;
class JSONViewerPanel;
class RegexTesterPanel;
// Plugin Manager panel
class PluginManagerPanel;
// Data Studio panel (Phase 1 Week 1)
class DataStudioPanel;
class FoldsView;
// Node Info panel (Phase 2)
class NodeInfoPanel;
struct StudioDebuggerSnapshot;
struct StudioDebuggerRunControl;
struct StudioDebuggerRunInputs;
enum class StudioDebuggerRunMode;
} // namespace cyxwiz

namespace scripting {
class ScriptingEngine;
class StartupScriptManager;
} // namespace scripting

namespace network {
class DataStreamClient;
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
    cyxwiz::ScriptEditorPanel* GetScriptEditor() { return script_editor_.get(); }
    NodeEditor* GetNodeEditor() { return node_editor_.get(); }
    bool OpenGraphInNodeEditor(const std::string& path);

    // Set network components (called by Application after construction)
    void SetNetworkComponents(network::GRPCClient* client, network::JobManager* job_manager);
    
    // Set DataStream client for cloud panels
    void SetDataStreamClient(network::DataStreamClient* client);

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
    // Opens a .py or .cyx file in the Script Editor and shows the editor.
    void OpenScriptFile(const std::string& file_path);

    // Debug logging pointers (set by Application)
    void SetIdleLogPtr(bool* ptr);

    // Project settings persistence
    void SaveProjectSettings();      // Save current editor settings and layout to project
    void LoadProjectSettings();      // Load editor settings and layout from project
    void SaveLayout();               // Save only ImGui layout to project

    // Called when project is opened/closed
    void OnProjectOpened(const std::string& project_root);
    void OnProjectClosed(const std::string& project_root);
    void OnProjectVenvReady(const std::string& project_root);

private:
    // The Export Model dialog for the trained model (a layer stack or a graph
    // with branches); format_index is a ModelFormat.
    void OpenModelExport(int format_index);
    void RenderDockSpace();
    void BuildInitialDockLayout();
    void RegisterPanelsWithSidebar();
    void SetDefaultPanelVisibility(bool force = false);  // Hide tool panels, show only core panels
    std::map<std::string, bool> default_panel_visibility_;  // registered panels before any saved state
    void RenderSidebar();
    void RenderStatusBar();
    // Renders a panel under a frame-metrics timer (Tools > Diagnostics > Frame Time Overlay).
    template <class PanelPtr>
    void TimedRender(const char* name, PanelPtr& panel);
    // Status bar height follows the interface text size (TOFIX129).
    static float StatusBarHeight();
    void RenderPythonEnvSetupStatus(const cyxwiz::ProjectManager& pm);
    void DetectKeyboardContext();
    void HandleGlobalShortcuts();

    // Training from node graph
    void StartTrainingFromGraph(const std::vector<MLNode>& nodes, const std::vector<NodeLink>& links);

    // Testing from node graph
    void StartTestingFromGraph(const std::vector<MLNode>& nodes, const std::vector<NodeLink>& links);
    void StartTestingWithConfig(const std::vector<MLNode>& nodes,
        const std::vector<NodeLink>& links, cyxwiz::TrainingConfiguration config);
    std::shared_ptr<cyxwiz::AsyncTask> test_preparation_task_;
    std::shared_ptr<int> test_callback_lifetime_ = std::make_shared<int>(0);
    // Results > Cross-validation / Learning curve (TOFIX129 D4): Run marks the
    // next Train launch as theirs; it goes through Train's checks, data loads,
    // memory question and preparation, and only the dispatch differs: the
    // prepared table goes to core/cross_validation_runner on a task.
    struct FoldsLaunch {
        std::shared_ptr<cyxwiz::FoldsView> view;
        cyxwiz::CrossValidationSettings settings;
        std::string results_table;  // where the finished run's table is published
    };
    // The graph's Cross-Validation / Learning Curves node (the first one), read.
    struct FoldsNode {
        int id = -1;
        std::string name;
        cyxwiz::CrossValidationSettings settings;
        std::string results_table;
        std::string error;
    };
    std::optional<FoldsNode> FindFoldsNode(bool learning_curve) const;
    // Results' Add to the graph: the node after the graph's Data Input.
    std::string AddFoldsNode(bool learning_curve, const cyxwiz::CrossValidationSettings& settings);
    std::optional<FoldsLaunch> folds_launch_;
    uint64_t folds_task_ = 0;  // the running cross-validation / learning curve task
    void StartFoldsFromGraph(std::shared_ptr<cyxwiz::FoldsView> view, const cyxwiz::CrossValidationSettings& settings);
    // A marked launch that stopped before its task: the view says why.
    void EndFoldsLaunch(const std::string& why);
    std::string FoldsBlocker() const;
    ::gui::GraphTrainingDispatch MakeFoldsDispatch(const FoldsLaunch& launch);


    // Compile the current graph as a dry-run (no training) and show results in a popup
    void CompileGraphAndReport();
    void RenderCompileResultPopup();
    // Opens the compile report (main_window_compile_report.cpp) for what the
    // compile found, or for why Train / Local Debug / a checkpoint load
    // stopped (`stop_reason`, empty when the findings say it).
    void ShowCompileReport(cyxwiz::compile_report::Mode mode, std::string stop_reason = {});
    // User-facing failure for an action that has no result panel of its own
    // (checkpoint load, Run Test refusals): shown on screen, also logged.
    void ShowOperationError(std::string title, std::string message);
    void LoadCheckpointForTesting();
    void RenderOperationErrorPopup();
    void RenderMaterializationMemoryConfirmationPopup();
    bool BuildStudioDebuggerSession(cyxwiz::StudioDebuggerSnapshot& session,
                                    cyxwiz::StudioDebuggerRunMode mode,
                                    int sample_index);
    bool BuildStudioDebuggerSessionFromSnapshot(cyxwiz::StudioDebuggerSnapshot& session,
                                                cyxwiz::StudioDebuggerRunMode mode,
                                                int sample_index,
                                                std::vector<MLNode> nodes,
                                                std::vector<NodeLink> links,
                                                int explain_node_id = -1,
                                                const cyxwiz::StudioDebuggerRunControl* control = nullptr,
                                                const cyxwiz::StudioDebuggerRunInputs* inputs = nullptr);
    void RecordArtifactConsistencyTrace(
        const cyxwiz::DebugArtifactConsistencyInput& input);

    // Run Local Debug (F6) as a background Studio Debugger run: compile
    // gate, preflight, then one synthetic forward/backward pass.
    void LocalDebugGraphAndReport();

    // Shared compile pass used by both the explicit Compile button and
    // the Train button's pre-flight gate. Populates compile_result_*
    // members from a fresh GraphCompiler::Compile call.
    void BuildCompileResult(const std::vector<MLNode>& nodes,
                            const std::vector<NodeLink>& links);

    // The compile report (TOFIX129 B3): the findings, placements and
    // configuration of the last compile, and the report built from them when
    // the popup opens (core/compile_report_presentation).
    bool show_compile_result_popup_ = false;
    bool show_operation_error_popup_ = false;
    std::string operation_error_title_;
    std::string operation_error_message_;
    bool compile_result_success_ = false;
    bool compile_result_has_config_ = false;
    cyxwiz::compile_report::ConfigFacts compile_result_config_;
    std::vector<cyxwiz::ValidationIssue> compile_result_issues_;
    std::vector<cyxwiz::BackendPlacementEntry> compile_result_backend_placements_;
    cyxwiz::compile_report::Report compile_report_;

    // Main-thread confirmation state for a warning/risky operator estimate.
    // The worker is not queued until the user accepts this frozen graph copy.
    bool show_materialization_memory_confirmation_popup_ = false;
    bool materialization_memory_confirmation_accepted_once_ = false;
    int materialization_memory_confirmation_node_id_ = -1;
    uint64_t materialization_memory_confirmation_estimated_bytes_ = 0;
    uint64_t materialization_memory_confirmation_available_bytes_ = 0;
    uint64_t materialization_memory_confirmation_budget_bytes_ = 0;
    std::string materialization_memory_confirmation_detail_;
    std::string materialization_memory_confirmation_risk_;
    std::string materialization_memory_confirmation_node_;
    int accepted_materialization_memory_node_id_ = -1;
    uint64_t accepted_materialization_memory_estimated_bytes_ = 0;
    std::string accepted_materialization_memory_risk_;
    std::vector<MLNode> pending_memory_confirmation_nodes_;
    std::vector<NodeLink> pending_memory_confirmation_links_;

    // Train on a reopened project: the Data Inputs it has not loaded load
    // first, each through an unshown Data Input dialog (the same Apply as its
    // OK), then training starts with the loaded graph.
    struct DataInputLoad {
        int node_id = -1;
        std::string node_name;
        std::unique_ptr<DataInputDialog> dialog;
    };
    std::vector<DataInputLoad> data_input_loads_;
    std::string data_input_loads_project_;
    bool StartDataInputLoads(const std::vector<MLNode>& nodes, const std::vector<NodeLink>& links);
    // Train on a graph whose Training Dataset path has pipeline-run steps
    // (SQL, Text Clean ...): run them first, then train on the result.
    bool StartTrainingPipelineRun(const std::vector<MLNode>& nodes, const std::vector<NodeLink>& links);
    void PollTrainingPipelineRun();
    struct TrainingPipelineRun {
        uint64_t task_id = 0;
        int feeder = -1;
        std::string project;
    };
    std::optional<TrainingPipelineRun> training_pipeline_run_;
    std::string training_pipeline_result_;  // read by the next StartTrainingFromGraph
    void PollDataInputLoads();

    // The last Local Debug run (TOFIX178 UI-1): passed or failed, when, why,
    // and the HashGraphStructure of the graph it ran on. The canvas bar shows
    // it; StartTrainingFromGraph warns when it is not a pass on the graph as
    // it is now, or, under EngineConfig::RequireDebugBeforeTrain(), refuses.
    cyxwiz::studio_toolbar::DebugInputs local_debug_;
    uint64_t local_debug_graph_hash_ = 0;
    double local_debug_checked_at_ = -1.0;
    // Twice a second: is the canvas still the graph the run used?
    void UpdateLocalDebugStatus();

    // Original panels
    std::unique_ptr<NodeEditor> node_editor_;
    std::unique_ptr<Console> console_;
    std::unique_ptr<Viewport> viewport_;
    std::unique_ptr<Properties> properties_;
    // Background compile of the canvas graph (TOFIX123): Properties AS
    // COMPILED card, link colours and tooltips follow it.
    std::unique_ptr<cyxwiz::LiveGraphCompile> live_compile_;
    uint64_t live_compile_serial_seen_ = 0;
    void UpdateLiveCompile();
    // Python node bundles (TOFIX125 P4): registered from <project>/nodes,
    // the user nodes folder and plugin nodes/ folders; Python is started
    // here, on the UI thread, when a Python node needs it.
    std::unique_ptr<cyxwiz::PythonNodeProvider> python_node_provider_;
    double python_nodes_checked_at_ = -1.0;
    std::string python_start_error_logged_;
    void UpdatePythonNodes();

    // New panel system
    std::unique_ptr<cyxwiz::ToolbarPanel> toolbar_;
    std::unique_ptr<cyxwiz::AssetBrowserPanel> asset_browser_;
    std::shared_ptr<cyxwiz::TrainingPlotPanel> training_plot_panel_;
    std::unique_ptr<cyxwiz::ScriptEditorPanel> script_editor_;
    std::unique_ptr<cyxwiz::TableViewerPanel> table_viewer_;
    std::unique_ptr<cyxwiz::AnnotationEditorPanel> annotation_editor_panel_;
    std::unique_ptr<cyxwiz::ConnectionDialog> connection_dialog_;
    std::unique_ptr<cyxwiz::JobStatusPanel> job_status_panel_;
    std::unique_ptr<cyxwiz::P2PTrainingPanel> p2p_training_panel_;
    std::unique_ptr<gui::WalletPanel> wallet_panel_;
    std::unique_ptr<cyxwiz::TaskProgressPanel> task_progress_panel_;
    std::unique_ptr<cyxwiz::PatternBrowserPanel> pattern_browser_;
    std::unique_ptr<cyxwiz::QueryConsolePanel> query_console_;
    std::unique_ptr<gui::CustomNodeEditorPanel> custom_node_editor_;
    std::unique_ptr<gui::ThemeEditorPanel> theme_editor_;
    std::unique_ptr<cyxwiz::MemoryPanel> memory_panel_;
    std::unique_ptr<cyxwiz::MemoryMonitor> memory_monitor_;
    std::unique_ptr<cyxwiz::VariableExplorerPanel> variable_explorer_;
    std::unique_ptr<cyxwiz::PlotOutputPanel> plot_output_panel_;
    std::unique_ptr<cyxwiz::ExportDialog> export_dialog_;
    std::unique_ptr<cyxwiz::ImportDialog> import_dialog_;
    std::unique_ptr<cyxwiz::DeploymentDialog> deployment_dialog_;
    std::unique_ptr<cyxwiz::StudioDebuggerPanel> studio_debugger_panel_;

    // Model Analysis panels (Phase 2)
    std::unique_ptr<cyxwiz::ModelSummaryPanel> model_summary_panel_;
    std::unique_ptr<cyxwiz::ArchitectureDiagram> architecture_diagram_;
    std::unique_ptr<cyxwiz::LRFinderPanel> lr_finder_panel_;

    // Statistics panels (Phase 4)

    // Advanced Tools panels (Phase 5)
    std::unique_ptr<cyxwiz::DimReductionPanel> dim_reduction_panel_;
    std::unique_ptr<cyxwiz::GradCAMPanel> gradcam_panel_;
    std::unique_ptr<cyxwiz::FeatureImportancePanel> feature_importance_panel_;
    std::unique_ptr<cyxwiz::NASPanel> nas_panel_;
    std::unique_ptr<cyxwiz::HyperparamSearchPanel> hyperparam_search_panel_;
    std::unique_ptr<cyxwiz::ServingPanel> serving_panel_;
    std::unique_ptr<cyxwiz::DNNInferencePanel> dnn_inference_panel_;

    // Clustering panels (Phase 6A)
    std::unique_ptr<cyxwiz::KMeansPanel> kmeans_panel_;
    std::unique_ptr<cyxwiz::DBSCANPanel> dbscan_panel_;
    std::unique_ptr<cyxwiz::HierarchicalPanel> hierarchical_panel_;
    std::unique_ptr<cyxwiz::GMMPanel> gmm_panel_;
    std::unique_ptr<cyxwiz::ClusterEvalPanel> cluster_eval_panel_;

    // Model Evaluation panels (Phase 6B)

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
    std::unique_ptr<cyxwiz::LanguageModelGenerationPanel> language_model_generation_panel_;

    // Utilities panels (Phase 12)
    std::unique_ptr<cyxwiz::CalculatorPanel> calculator_panel_;
    std::unique_ptr<cyxwiz::UnitConverterPanel> unit_converter_panel_;
    std::unique_ptr<cyxwiz::RandomGeneratorPanel> random_generator_panel_;
    std::unique_ptr<cyxwiz::HashGeneratorPanel> hash_generator_panel_;
    std::unique_ptr<cyxwiz::JSONViewerPanel> json_viewer_panel_;
    std::unique_ptr<cyxwiz::RegexTesterPanel> regex_tester_panel_;
    
    // Cloud panels (DataStream)
    std::unique_ptr<gui::CloudBrowserPanel> cloud_browser_panel_;
    std::unique_ptr<gui::NodeBrowserPanel> node_browser_panel_;
    std::unique_ptr<cyxwiz::NodeInfoPanel> node_info_panel_;
    std::unique_ptr<gui::CloudDatasetManagerPanel> cloud_dataset_manager_panel_;

    // Plugin Manager
    std::unique_ptr<cyxwiz::PluginManagerPanel> plugin_manager_panel_;

    // Data Studio (Phase 1 Week 1)
    std::unique_ptr<cyxwiz::DataStudioPanel> data_studio_panel_;

    // Scripting engine (shared between panels)
    std::shared_ptr<scripting::ScriptingEngine> scripting_engine_;

    // Startup script manager
    std::unique_ptr<scripting::StartupScriptManager> startup_script_manager_;

    bool first_time_layout_;
    bool reset_layout_requested_ = false;
    int front_tabs_countdown_ = 0;  // frames until the default tabs come to the front

    // Network components
    network::JobManager* job_manager_ = nullptr;
    network::DataStreamClient* datastream_client_ = nullptr;
    std::string monitoring_job_id_;


    // Exit request callback
    ExitRequestCallback exit_request_callback_;
};

} // namespace gui
