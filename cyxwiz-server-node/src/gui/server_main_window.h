// server_main_window.h - Main window with sidebar navigation
#pragma once

#include "core/state_manager.h"
#include <memory>
#include <string>

namespace cyxwiz::servernode::ipc {
    class DaemonClient;
}

namespace cyxwiz::servernode::gui {

// Forward declarations for panels
class DashboardPanel;
class JobMonitorPanel;
class ModelBrowserPanel;
class DeploymentPanel;
class APIKeysPanel;
class SettingsPanel;
class LogsPanel;
class PoolMiningPanel;
class MarketplacePanel;
class WalletPanel;
class AnalyticsPanel;
class FineTuningPanel;
class LoginPanel;
class AccountSettingsPanel;
class HardwarePanel;
class AllocationPanel;

class ServerMainWindow : public core::StateObserver {
public:
    explicit ServerMainWindow(ipc::DaemonClient* daemon_client = nullptr);
    ~ServerMainWindow() override;

    void Render();
    void ResetDockLayout();

    // StateObserver overrides
    void OnJobsChanged() override;
    void OnDeploymentsChanged() override;
    void OnMetricsUpdated() override;
    void OnConnectionStatusChanged() override;

private:
    void RenderDockSpace();
    void RenderSidebar();
    void RenderStatusBar();
    void RenderCurrentPanel();
    void BuildInitialDockLayout();

    // Sidebar items
    enum class SidebarItem {
        Dashboard = 0,
        Hardware,
        Allocation,
        Analytics,
        Jobs,
        Models,
        FineTuning,
        Deploy,
        APIKeys,
        Marketplace,
        PoolMining,
        Settings,
        Logs,
        Wallet,
        Account,
        COUNT
    };

    struct SidebarEntry {
        SidebarItem item;
        const char* icon;
        const char* label;
        const char* tooltip;
    };

    static const SidebarEntry sidebar_entries_[];

    // Panels
    std::unique_ptr<LoginPanel> login_;
    std::unique_ptr<DashboardPanel> dashboard_;
    std::unique_ptr<HardwarePanel> hardware_;
    std::unique_ptr<AllocationPanel> allocation_;
    std::unique_ptr<AnalyticsPanel> analytics_;
    std::unique_ptr<JobMonitorPanel> job_monitor_;
    std::unique_ptr<ModelBrowserPanel> model_browser_;
    std::unique_ptr<FineTuningPanel> fine_tuning_;
    std::unique_ptr<DeploymentPanel> deployment_;
    std::unique_ptr<APIKeysPanel> api_keys_;
    std::unique_ptr<SettingsPanel> settings_;
    std::unique_ptr<LogsPanel> logs_;
    std::unique_ptr<PoolMiningPanel> pool_mining_;
    std::unique_ptr<MarketplacePanel> marketplace_;
    std::unique_ptr<WalletPanel> wallet_;
    std::unique_ptr<AccountSettingsPanel> account_settings_;

    // State
    SidebarItem selected_item_ = SidebarItem::Dashboard;
    bool first_frame_ = true;
    bool show_about_dialog_ = false;

    // Cached state for status bar
    int active_jobs_count_ = 0;
    int deployed_models_count_ = 0;
    std::string connection_status_text_ = "Disconnected";

    // Daemon client (for dual-process mode)
    ipc::DaemonClient* daemon_client_ = nullptr;
};

} // namespace cyxwiz::servernode::gui
