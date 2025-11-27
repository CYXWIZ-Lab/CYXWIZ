#pragma once

#include <memory>
#include <string>
#include <vector>

namespace gui {

class NodeEditor;
class Console;
class Viewport;
class Properties;
class DatasetPanel;
class WalletPanel;

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
class ConnectionDialog;
class JobStatusPanel;
class P2PTrainingPanel;
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

private:
    void RenderDockSpace();
    void BuildInitialDockLayout();
    void ShowAboutDialog();

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
    std::unique_ptr<cyxwiz::ConnectionDialog> connection_dialog_;
    std::unique_ptr<cyxwiz::JobStatusPanel> job_status_panel_;
    std::unique_ptr<cyxwiz::P2PTrainingPanel> p2p_training_panel_;
    std::unique_ptr<gui::WalletPanel> wallet_panel_;

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
};

} // namespace gui
