// tui_application.h - FTXUI-based terminal user interface with tab navigation
#pragma once

#include <atomic>
#include <thread>
#include <memory>
#include <vector>
#include <string>
#include <deque>
#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>

namespace cyxwiz::servernode::ipc {
    class DaemonClient;
}

namespace cyxwiz::servernode::tui {

// View tabs matching GUI sidebar
enum class TUIView {
    Dashboard = 0,
    Jobs,
    Models,
    Deploy,
    Settings,
    Logs,
    COUNT
};

// Per-device metrics for multi-GPU display
struct DeviceMetrics {
    int id = 0;
    std::string name;
    float utilization = 0.0f;
    size_t vram_used = 0;
    size_t vram_total = 0;
    float temperature = 0.0f;
    bool in_use = false;
    std::string assigned_job;
};

// Job info for display
struct TUIJobInfo {
    std::string id;
    std::string type;
    std::string status;
    float progress = 0.0f;
    int current_epoch = 0;
    int total_epochs = 0;
    double loss = 0.0;
    double accuracy = 0.0;
    std::string device;
};

// Model info for display
struct TUIModelInfo {
    std::string name;
    std::string path;
    std::string format;
    int64_t size_bytes = 0;
    bool deployed = false;
    int port = 0;
    int request_count = 0;
};

// Log entry
struct LogEntry {
    std::string timestamp;
    std::string level;
    std::string message;
};

// Deployment info for TUI
struct TUIDeploymentInfo {
    std::string id;
    std::string model_name;
    std::string model_path;
    std::string status;
    int port = 0;
    int64_t request_count = 0;
};

class TUIApplication {
public:
    TUIApplication(int argc, char** argv,
                   std::shared_ptr<ipc::DaemonClient> daemon_client = nullptr);
    ~TUIApplication();

    void Run();
    void Stop();

private:
    // Main UI creation
    ftxui::Component CreateUI();
    ftxui::Component CreateTabBar();
    ftxui::Component CreateStatusBar();

    // View creators (now return interactive components)
    ftxui::Component CreateDashboardView();
    ftxui::Component CreateJobsView();
    ftxui::Component CreateModelsView();
    ftxui::Component CreateDeployView();
    ftxui::Component CreateSettingsView();
    ftxui::Component CreateLogsView();

    // Dashboard components
    ftxui::Element RenderResourceGauges();
    ftxui::Element RenderDeviceList();
    ftxui::Element RenderActiveJobs();
    ftxui::Element RenderEarnings();

    // Jobs components
    ftxui::Element RenderJobsList();
    ftxui::Element RenderJobDetails();

    // Models components
    ftxui::Element RenderModelsList();
    ftxui::Element RenderModelDetails();

    // Deploy components
    ftxui::Element RenderDeploymentsList();
    ftxui::Element RenderDeploymentControls();

    // Settings components
    ftxui::Element RenderConnectionSettings();
    ftxui::Element RenderNodeSettings();

    // Actions
    void CancelSelectedJob();
    void DeploySelectedModel();
    void UndeploySelectedModel();
    void StopSelectedDeployment();
    void SaveSettings();
    void TestConnection();
    void ClearLogs();
    void Reconnect();

    // Data refresh
    void RefreshLoop();
    void RefreshMetrics();
    void RefreshJobs();
    void RefreshModels();
    void RefreshDeployments();
    void RefreshLogs();

    // Input handling
    bool HandleGlobalInput(ftxui::Event event);

    // State
    std::atomic<bool> running_{false};
    std::thread refresh_thread_;
    int argc_;
    char** argv_;

    // Daemon client
    std::shared_ptr<ipc::DaemonClient> daemon_client_;

    // Current view
    int current_view_ = 0;
    std::vector<std::string> view_names_ = {
        "Dashboard", "Jobs", "Models", "Deploy", "Settings", "Logs"
    };

    // Cached metrics
    float cpu_usage_ = 0.0f;
    float gpu_usage_ = 0.0f;
    float ram_usage_ = 0.0f;
    float ram_used_gb_ = 0.0f;
    float ram_total_gb_ = 0.0f;
    float vram_usage_ = 0.0f;
    bool connected_ = false;
    std::string node_id_ = "unknown";

    // Multi-GPU data
    std::vector<DeviceMetrics> devices_;

    // Jobs data
    std::vector<TUIJobInfo> jobs_;
    int selected_job_ = 0;

    // Models data
    std::vector<TUIModelInfo> models_;
    int selected_model_ = 0;

    // Logs data
    std::deque<LogEntry> logs_;
    static constexpr size_t MAX_LOGS = 500;
    int log_scroll_ = 0;
    int log_level_filter_ = 0;  // 0=All, 1=Error, 2=Warn, 3=Info, 4=Debug

    // Deployments data
    std::vector<TUIDeploymentInfo> deployments_;
    int selected_deployment_ = 0;

    // Settings (editable)
    std::string daemon_address_ = "localhost:50054";
    std::string daemon_address_input_;  // For editing
    bool use_tls_ = false;
    std::string node_name_ = "";
    std::string node_name_input_;
    int max_concurrent_jobs_ = 4;
    std::string max_jobs_input_;
    std::string central_server_address_ = "localhost:50051";
    std::string central_server_input_;

    // Earnings
    double earnings_today_ = 0.0;
    double earnings_week_ = 0.0;
    double earnings_month_ = 0.0;

    // Deploy dialog state
    bool show_deploy_dialog_ = false;
    int deploy_port_ = 8082;
    std::string deploy_port_input_ = "8082";
    int deploy_gpu_layers_ = 35;
    std::string deploy_gpu_layers_input_ = "35";
    int deploy_context_size_ = 4096;
    std::string deploy_context_size_input_ = "4096";

    // UI state
    bool show_help_ = false;
    bool settings_dirty_ = false;
    std::string status_message_;
    std::string error_message_;

    // Screen reference for refresh
    ftxui::ScreenInteractive* screen_ = nullptr;
};

} // namespace cyxwiz::servernode::tui
