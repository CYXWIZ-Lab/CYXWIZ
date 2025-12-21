// server_main_window.cpp - Main window implementation
#include "gui/server_main_window.h"
#include "gui/icons.h"
#include "gui/theme.h"
#include "gui/panels/dashboard_panel.h"
#include "gui/panels/job_monitor_panel.h"
#include "gui/panels/model_browser_panel.h"
#include "gui/panels/deployment_panel.h"
#include "gui/panels/api_keys_panel.h"
#include "gui/panels/settings_panel.h"
#include "gui/panels/logs_panel.h"
#include "gui/panels/pool_mining_panel.h"
#include "gui/panels/marketplace_panel.h"
#include "gui/panels/wallet_panel.h"
#include "gui/panels/analytics_panel.h"
#include "gui/panels/fine_tuning_panel.h"
#include "gui/panels/login_panel.h"
#include "gui/panels/account_settings_panel.h"
#include "gui/panels/hardware_panel.h"
#include "gui/panels/allocation_panel.h"
#include "auth/auth_manager.h"
#include "core/backend_manager.h"
#include "ipc/daemon_client.h"

#include <imgui.h>
#include <imgui_internal.h>
#include <spdlog/spdlog.h>

namespace cyxwiz::servernode::gui {

const ServerMainWindow::SidebarEntry ServerMainWindow::sidebar_entries_[] = {
    { SidebarItem::Dashboard,   ICON_FA_GAUGE_HIGH,       "Dashboard",    "System overview" },
    { SidebarItem::Hardware,    ICON_FA_MICROCHIP,        "Hardware",     "Detected hardware & resources" },
    { SidebarItem::Allocation,  ICON_FA_SLIDERS,          "Allocation",   "Resource allocation for sharing" },
    { SidebarItem::Analytics,   ICON_FA_CHART_LINE,       "Analytics",    "Historical metrics & trends" },
    { SidebarItem::Jobs,        ICON_FA_BARS_PROGRESS,    "Jobs",         "Active training jobs" },
    { SidebarItem::Models,      ICON_FA_CUBE,             "Models",       "Browse and manage models" },
    { SidebarItem::FineTuning,  ICON_FA_BRAIN,            "Fine-tuning",  "Model fine-tuning & transfer learning" },
    { SidebarItem::Deploy,      ICON_FA_ROCKET,           "Deploy",       "Deploy models for inference" },
    { SidebarItem::APIKeys,     ICON_FA_KEY,              "API Keys",     "Manage API keys" },
    { SidebarItem::Marketplace, ICON_FA_STORE,            "Marketplace",  "Model marketplace" },
    { SidebarItem::PoolMining,  ICON_FA_COINS,            "Pool Mining",  "Join mining pools" },
    { SidebarItem::Settings,    ICON_FA_GEAR,             "Settings",     "Node configuration" },
    { SidebarItem::Logs,        ICON_FA_SCROLL,           "Logs",         "View logs" },
    { SidebarItem::Wallet,      ICON_FA_WALLET,           "Wallet",       "Wallet & earnings" },
    { SidebarItem::Account,     ICON_FA_CIRCLE_USER,      "Account",      "Account settings" },
};

ServerMainWindow::ServerMainWindow(ipc::DaemonClient* daemon_client)
    : daemon_client_(daemon_client) {
    spdlog::info("Creating ServerMainWindow (daemon_client={})",
                 daemon_client ? "connected" : "disconnected");

    // Create panels
    login_ = std::make_unique<LoginPanel>();
    dashboard_ = std::make_unique<DashboardPanel>();
    hardware_ = std::make_unique<HardwarePanel>();
    allocation_ = std::make_unique<AllocationPanel>();
    analytics_ = std::make_unique<AnalyticsPanel>();
    job_monitor_ = std::make_unique<JobMonitorPanel>();
    model_browser_ = std::make_unique<ModelBrowserPanel>();
    fine_tuning_ = std::make_unique<FineTuningPanel>();
    deployment_ = std::make_unique<DeploymentPanel>();
    api_keys_ = std::make_unique<APIKeysPanel>();
    settings_ = std::make_unique<SettingsPanel>();
    logs_ = std::make_unique<LogsPanel>();
    pool_mining_ = std::make_unique<PoolMiningPanel>();
    marketplace_ = std::make_unique<MarketplacePanel>();
    wallet_ = std::make_unique<WalletPanel>();
    account_settings_ = std::make_unique<AccountSettingsPanel>();

    // Set daemon client on all panels
    if (daemon_client_) {
        dashboard_->SetDaemonClient(daemon_client_);
        hardware_->SetDaemonClient(daemon_client_);
        allocation_->SetDaemonClient(daemon_client_);
        analytics_->SetDaemonClient(daemon_client_);
        job_monitor_->SetDaemonClient(daemon_client_);
        model_browser_->SetDaemonClient(daemon_client_);
        fine_tuning_->SetDaemonClient(daemon_client_);
        deployment_->SetDaemonClient(daemon_client_);
        api_keys_->SetDaemonClient(daemon_client_);
        settings_->SetDaemonClient(daemon_client_);
        logs_->SetDaemonClient(daemon_client_);
        pool_mining_->SetDaemonClient(daemon_client_);
        marketplace_->SetDaemonClient(daemon_client_);
        wallet_->SetDaemonClient(daemon_client_);
        account_settings_->SetDaemonClient(daemon_client_);
        login_->SetDaemonClient(daemon_client_);
    }

    // Register as state observer
    auto* state_manager = core::BackendManager::Instance().GetStateManager();
    if (state_manager) {
        state_manager->AddObserver(this);
    }
}

ServerMainWindow::~ServerMainWindow() {
    auto* state_manager = core::BackendManager::Instance().GetStateManager();
    if (state_manager) {
        state_manager->RemoveObserver(this);
    }
}

void ServerMainWindow::Render() {
    // Update login panel (check async login results)
    if (login_) {
        login_->Update();
    }

    // Check if user is logged in
    bool logged_in = login_ && login_->IsLoggedIn();

    if (logged_in) {
        // Normal app rendering when logged in
        RenderDockSpace();
        RenderSidebar();
        RenderCurrentPanel();
        RenderStatusBar();

        if (first_frame_) {
            BuildInitialDockLayout();
            first_frame_ = false;
        }
    }

    // Login overlay renders on top when not logged in
    if (login_) {
        login_->Render();
    }
}

void ServerMainWindow::RenderDockSpace() {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();

    // Setup dockspace window
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::SetNextWindowViewport(viewport->ID);

    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking |
                                    ImGuiWindowFlags_NoTitleBar |
                                    ImGuiWindowFlags_NoCollapse |
                                    ImGuiWindowFlags_NoResize |
                                    ImGuiWindowFlags_NoMove |
                                    ImGuiWindowFlags_NoBringToFrontOnFocus |
                                    ImGuiWindowFlags_NoNavFocus |
                                    ImGuiWindowFlags_NoBackground;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

    ImGui::Begin("##ServerNodeDockSpace", nullptr, window_flags);
    ImGui::PopStyleVar(3);

    ImGuiID dockspace_id = ImGui::GetID("ServerNodeDockSpace");
    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_PassthruCentralNode);

    ImGui::End();
}

void ServerMainWindow::RenderSidebar() {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float sidebar_width = 200.0f;
    const float status_bar_height = 28.0f;

    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(ImVec2(sidebar_width, viewport->WorkSize.y - status_bar_height));

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar |
                             ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoMove |
                             ImGuiWindowFlags_NoCollapse |
                             ImGuiWindowFlags_NoDocking;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 8.0f));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.10f, 0.10f, 0.12f, 1.0f));

    if (ImGui::Begin("##Sidebar", nullptr, flags)) {
        // Header
        ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[2]);  // Large font
        ImGui::TextColored(ImVec4(0.4f, 0.7f, 1.0f, 1.0f), ICON_FA_SERVER);
        ImGui::SameLine();
        ImGui::Text("CyxWiz Node");
        ImGui::PopFont();

        ImGui::Separator();
        ImGui::Spacing();

        // Sidebar items
        for (const auto& entry : sidebar_entries_) {
            bool selected = (selected_item_ == entry.item);

            // Style for selected item
            if (selected) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.45f, 0.75f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.25f, 0.50f, 0.80f, 1.0f));
            } else {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.15f, 0.15f, 0.20f, 1.0f));
            }

            // Full-width button
            ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0.0f, 0.5f));
            std::string label = std::string(entry.icon) + "  " + entry.label;

            if (ImGui::Button(label.c_str(), ImVec2(sidebar_width - 16.0f, 32.0f))) {
                selected_item_ = entry.item;
            }

            ImGui::PopStyleVar();
            ImGui::PopStyleColor(2);

            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("%s", entry.tooltip);
            }
        }

        // Bottom section
        ImGui::SetCursorPosY(ImGui::GetWindowHeight() - 80.0f);
        ImGui::Separator();

        // Connection status indicator - use daemon_client if available
        ImVec4 status_color;
        const char* status_text;

        if (daemon_client_ && daemon_client_->IsConnected()) {
            status_color = ImVec4(0.2f, 0.8f, 0.2f, 1.0f);
            status_text = "Daemon Connected";
        } else if (daemon_client_) {
            status_color = ImVec4(0.8f, 0.2f, 0.2f, 1.0f);
            status_text = "Daemon Disconnected";
        } else {
            // Fallback to state manager (single-process mode)
            auto* state_manager = core::BackendManager::Instance().GetStateManager();
            if (state_manager) {
                auto status = state_manager->GetConnectionStatus();
                switch (status) {
                    case core::ConnectionStatus::Connected:
                        status_color = ImVec4(0.2f, 0.8f, 0.2f, 1.0f);
                        status_text = "Connected";
                        break;
                    case core::ConnectionStatus::Connecting:
                    case core::ConnectionStatus::Reconnecting:
                        status_color = ImVec4(1.0f, 0.8f, 0.0f, 1.0f);
                        status_text = "Connecting...";
                        break;
                    default:
                        status_color = ImVec4(0.8f, 0.2f, 0.2f, 1.0f);
                        status_text = "Disconnected";
                        break;
                }
            } else {
                status_color = ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
                status_text = "Standalone";
            }
        }

        ImGui::TextColored(status_color, ICON_FA_CIRCLE);
        ImGui::SameLine();
        ImGui::Text("%s", status_text);

        // Wallet info
        auto* sm = core::BackendManager::Instance().GetStateManager();
        if (sm) {
            std::string wallet = sm->GetWalletAddress();
            if (!wallet.empty()) {
                std::string short_addr = wallet.substr(0, 4) + "..." + wallet.substr(wallet.length() - 4);
                ImGui::Text("%s %s", ICON_FA_WALLET, short_addr.c_str());
            }
        }
    }
    ImGui::End();

    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
}

void ServerMainWindow::RenderCurrentPanel() {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float sidebar_width = 200.0f;
    const float status_bar_height = 28.0f;

    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + sidebar_width, viewport->WorkPos.y));
    ImGui::SetNextWindowSize(ImVec2(viewport->WorkSize.x - sidebar_width, viewport->WorkSize.y - status_bar_height));

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar |
                             ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoMove |
                             ImGuiWindowFlags_NoCollapse |
                             ImGuiWindowFlags_NoDocking;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(16.0f, 16.0f));

    if (ImGui::Begin("##MainContent", nullptr, flags)) {
        switch (selected_item_) {
            case SidebarItem::Dashboard:
                if (dashboard_) dashboard_->Render();
                break;
            case SidebarItem::Hardware:
                if (hardware_) hardware_->Render();
                break;
            case SidebarItem::Allocation:
                if (allocation_) allocation_->Render();
                break;
            case SidebarItem::Analytics:
                if (analytics_) analytics_->Render();
                break;
            case SidebarItem::Jobs:
                if (job_monitor_) job_monitor_->Render();
                break;
            case SidebarItem::Models:
                if (model_browser_) model_browser_->Render();
                break;
            case SidebarItem::FineTuning:
                if (fine_tuning_) fine_tuning_->Render();
                break;
            case SidebarItem::Deploy:
                if (deployment_) deployment_->Render();
                break;
            case SidebarItem::APIKeys:
                if (api_keys_) api_keys_->Render();
                break;
            case SidebarItem::Marketplace:
                if (marketplace_) marketplace_->Render();
                break;
            case SidebarItem::PoolMining:
                if (pool_mining_) pool_mining_->Render();
                break;
            case SidebarItem::Settings:
                if (settings_) settings_->Render();
                break;
            case SidebarItem::Logs:
                if (logs_) logs_->Render();
                break;
            case SidebarItem::Wallet:
                if (wallet_) wallet_->Render();
                break;
            case SidebarItem::Account:
                if (account_settings_) account_settings_->Render();
                break;
            default:
                ImGui::Text("Select an item from the sidebar");
                break;
        }
    }
    ImGui::End();

    ImGui::PopStyleVar();
}

void ServerMainWindow::RenderStatusBar() {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float status_bar_height = 28.0f;

    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x, viewport->WorkPos.y + viewport->WorkSize.y - status_bar_height));
    ImGui::SetNextWindowSize(ImVec2(viewport->WorkSize.x, status_bar_height));

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar |
                             ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoMove |
                             ImGuiWindowFlags_NoScrollbar |
                             ImGuiWindowFlags_NoSavedSettings |
                             ImGuiWindowFlags_NoDocking;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 4.0f));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.08f, 0.08f, 0.10f, 1.0f));

    if (ImGui::Begin("##StatusBar", nullptr, flags)) {
        // Left side - Status
        ImGui::Text("%s Ready", ICON_FA_CIRCLE_CHECK);

        ImGui::SameLine(150);
        ImGui::Text("%s Jobs: %d", ICON_FA_BARS_PROGRESS, active_jobs_count_);

        ImGui::SameLine(280);
        ImGui::Text("%s Models: %d", ICON_FA_CUBE, deployed_models_count_);

        // Right side - Version
        ImGui::SameLine(ImGui::GetWindowWidth() - 150);
        ImGui::TextDisabled("CyxWiz Server Node v0.3.0");
    }
    ImGui::End();

    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
}

void ServerMainWindow::BuildInitialDockLayout() {
    // For now, we're using fixed layout instead of docking
    // Docking can be added later if needed
}

void ServerMainWindow::ResetDockLayout() {
    first_frame_ = true;
}

// StateObserver callbacks
void ServerMainWindow::OnJobsChanged() {
    auto* state_manager = core::BackendManager::Instance().GetStateManager();
    if (state_manager) {
        active_jobs_count_ = static_cast<int>(state_manager->GetActiveJobs().size());
    }
}

void ServerMainWindow::OnDeploymentsChanged() {
    auto* state_manager = core::BackendManager::Instance().GetStateManager();
    if (state_manager) {
        deployed_models_count_ = static_cast<int>(state_manager->GetDeployments().size());
    }
}

void ServerMainWindow::OnMetricsUpdated() {
    // Dashboard panel will handle this
}

void ServerMainWindow::OnConnectionStatusChanged() {
    // Connection status is read directly in RenderSidebar
}

} // namespace cyxwiz::servernode::gui
