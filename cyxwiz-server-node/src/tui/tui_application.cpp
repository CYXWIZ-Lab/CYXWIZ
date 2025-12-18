// tui_application.cpp - Enhanced FTXUI-based TUI with tab navigation
#include "tui/tui_application.h"
#include "core/backend_manager.h"
#include "core/state_manager.h"
#include "core/metrics_collector.h"
#include "core/device_pool.h"
#include "ipc/daemon_client.h"

#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/dom/table.hpp>
#include <spdlog/spdlog.h>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <ctime>

namespace cyxwiz::servernode::tui {

using namespace ftxui;

// Helper to format bytes
static std::string FormatBytes(int64_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit = 0;
    double size = static_cast<double>(bytes);
    while (size >= 1024.0 && unit < 4) {
        size /= 1024.0;
        unit++;
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1) << size << " " << units[unit];
    return ss.str();
}

// Helper to format time
static std::string GetCurrentTime() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::ostringstream ss;
    ss << std::put_time(std::localtime(&time), "%H:%M:%S");
    return ss.str();
}

TUIApplication::TUIApplication(int argc, char** argv,
                               std::shared_ptr<ipc::DaemonClient> daemon_client)
    : argc_(argc), argv_(argv), daemon_client_(std::move(daemon_client)) {

    // Initialize input fields
    daemon_address_input_ = daemon_address_;
    central_server_input_ = central_server_address_;
    max_jobs_input_ = std::to_string(max_concurrent_jobs_);

    // Get node ID and config from daemon if connected
    if (daemon_client_ && daemon_client_->IsConnected()) {
        ipc::DaemonStatus status;
        if (daemon_client_->GetStatus(status)) {
            node_id_ = status.node_id;
        }

        ipc::NodeConfig config;
        if (daemon_client_->GetConfig(config)) {
            node_name_ = config.node_name;
            node_name_input_ = node_name_;
            central_server_address_ = config.central_server_address;
            central_server_input_ = central_server_address_;
            max_concurrent_jobs_ = config.max_concurrent_jobs;
            max_jobs_input_ = std::to_string(max_concurrent_jobs_);
        }
    }
}

TUIApplication::~TUIApplication() {
    Stop();
}

void TUIApplication::Run() {
    spdlog::info("Starting TUI application");

    running_.store(true);

    // Start background refresh thread
    refresh_thread_ = std::thread(&TUIApplication::RefreshLoop, this);

    // Create screen and store reference
    auto screen = ScreenInteractive::Fullscreen();
    screen_ = &screen;

    // Create main UI
    auto ui = CreateUI();

    // Run the event loop
    screen.Loop(ui);

    screen_ = nullptr;
    running_.store(false);

    if (refresh_thread_.joinable()) {
        refresh_thread_.join();
    }

    spdlog::info("TUI application stopped");
}

void TUIApplication::Stop() {
    running_.store(false);
}

void TUIApplication::RefreshLoop() {
    while (running_.load()) {
        RefreshMetrics();
        RefreshJobs();
        RefreshModels();
        RefreshDeployments();
        RefreshLogs();

        // Trigger screen refresh if available
        if (screen_) {
            screen_->Post(Event::Custom);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

void TUIApplication::RefreshMetrics() {
    if (daemon_client_ && daemon_client_->IsConnected()) {
        ipc::DaemonStatus status;
        if (daemon_client_->GetStatus(status)) {
            cpu_usage_ = status.metrics.cpu_usage;
            gpu_usage_ = status.metrics.gpu_usage;
            ram_usage_ = status.metrics.ram_usage;
            vram_usage_ = status.metrics.vram_usage;
            connected_ = status.connected_to_central;
            node_id_ = status.node_id;
            ram_used_gb_ = static_cast<float>(status.metrics.ram_used) / (1024.0f * 1024.0f * 1024.0f);
            ram_total_gb_ = static_cast<float>(status.metrics.ram_total) / (1024.0f * 1024.0f * 1024.0f);

            // Single GPU info from status
            if (status.gpu_count > 0) {
                devices_.clear();
                DeviceMetrics dm;
                dm.id = 0;
                dm.name = status.gpu_name;
                dm.utilization = status.metrics.gpu_usage;
                dm.vram_used = status.metrics.vram_used;
                dm.vram_total = status.metrics.vram_total;
                devices_.push_back(dm);
            }
        }

        // Get earnings separately
        ipc::EarningsInfo earnings;
        if (daemon_client_->GetEarnings(earnings)) {
            earnings_today_ = earnings.today;
            earnings_week_ = earnings.this_week;
            earnings_month_ = earnings.this_month;
        }
    } else {
        // Single-process mode fallback
        auto& backend = core::BackendManager::Instance();
        if (auto* metrics = backend.GetMetricsCollector()) {
            auto m = metrics->GetCurrentMetrics();
            cpu_usage_ = m.cpu_usage;
            gpu_usage_ = m.gpu_usage;
            ram_usage_ = m.ram_usage;
            vram_usage_ = m.vram_usage;
        }

        if (auto* state = backend.GetStateManager()) {
            connected_ = state->GetConnectionStatus() == core::ConnectionStatus::Connected;
            earnings_today_ = state->GetEarningsToday().total_earnings;
            earnings_week_ = state->GetEarningsThisWeek().total_earnings;
            earnings_month_ = state->GetEarningsThisMonth().total_earnings;
        }
    }
}

// Helper to convert job status int to string
static std::string JobStatusToString(int status) {
    switch (status) {
        case 1: return "Pending";
        case 2: return "Running";
        case 3: return "Paused";
        case 4: return "Completed";
        case 5: return "Failed";
        case 6: return "Cancelled";
        default: return "Unknown";
    }
}

void TUIApplication::RefreshJobs() {
    jobs_.clear();

    if (daemon_client_ && daemon_client_->IsConnected()) {
        std::vector<ipc::JobInfo> daemon_jobs;
        if (daemon_client_->ListJobs(daemon_jobs)) {
            for (const auto& j : daemon_jobs) {
                TUIJobInfo job;
                job.id = j.id;
                job.type = j.type;
                job.status = JobStatusToString(j.status);
                job.progress = j.progress;
                job.current_epoch = j.current_epoch;
                job.total_epochs = j.total_epochs;
                job.loss = j.loss;
                job.accuracy = j.accuracy;
                jobs_.push_back(job);
            }
        }
    } else {
        auto& backend = core::BackendManager::Instance();
        if (auto* state = backend.GetStateManager()) {
            auto active = state->GetActiveJobs();
            for (const auto& j : active) {
                TUIJobInfo job;
                job.id = j.id;
                job.type = j.type;
                job.status = "Running";
                job.progress = j.progress;
                job.current_epoch = j.current_epoch;
                job.total_epochs = j.total_epochs;
                job.loss = j.loss;
                jobs_.push_back(job);
            }
        }
    }
}

void TUIApplication::RefreshModels() {
    models_.clear();

    if (daemon_client_ && daemon_client_->IsConnected()) {
        std::vector<ipc::ModelInfo> daemon_models;
        if (daemon_client_->ListLocalModels(daemon_models)) {
            // Also get deployments to know which models are deployed
            std::vector<ipc::DeploymentInfo> deployments;
            daemon_client_->ListDeployments(deployments);

            for (const auto& m : daemon_models) {
                TUIModelInfo model;
                model.name = m.name;
                model.path = m.path;
                model.format = m.format;
                model.size_bytes = m.size_bytes;
                model.deployed = m.is_deployed;
                model.port = 0;
                model.request_count = 0;

                // Find deployment info for this model
                for (const auto& dep : deployments) {
                    if (dep.model_path == m.path || dep.model_name == m.name) {
                        model.deployed = true;
                        model.port = dep.port;
                        model.request_count = static_cast<int>(dep.request_count);
                        break;
                    }
                }
                models_.push_back(model);
            }
        }
    }
}

void TUIApplication::RefreshLogs() {
    if (daemon_client_ && daemon_client_->IsConnected()) {
        std::vector<ipc::LogEntry> daemon_logs;
        if (daemon_client_->GetLogs(daemon_logs, 100)) {
            logs_.clear();
            for (const auto& log : daemon_logs) {
                // Convert timestamp to readable format
                auto time = std::chrono::system_clock::from_time_t(log.timestamp);
                auto time_t = std::chrono::system_clock::to_time_t(time);
                std::ostringstream ss;
                ss << std::put_time(std::localtime(&time_t), "%H:%M:%S");

                LogEntry entry;
                entry.timestamp = ss.str();
                entry.level = log.level;
                entry.message = log.message;
                logs_.push_back(entry);

                // Keep only MAX_LOGS entries
                while (logs_.size() > MAX_LOGS) {
                    logs_.pop_front();
                }
            }
        }
    }
}

void TUIApplication::RefreshDeployments() {
    deployments_.clear();

    if (daemon_client_ && daemon_client_->IsConnected()) {
        std::vector<ipc::DeploymentInfo> daemon_deployments;
        if (daemon_client_->ListDeployments(daemon_deployments)) {
            for (const auto& dep : daemon_deployments) {
                TUIDeploymentInfo info;
                info.id = dep.id;
                info.model_name = dep.model_name;
                info.model_path = dep.model_path;
                // Convert int status to string
                switch (dep.status) {
                    case 1: info.status = "Loading"; break;
                    case 2: info.status = "Running"; break;
                    case 3: info.status = "Stopped"; break;
                    case 4: info.status = "Error"; break;
                    default: info.status = "Unknown"; break;
                }
                info.port = dep.port;
                info.request_count = dep.request_count;
                deployments_.push_back(info);
            }
        }
    } else {
        auto& backend = core::BackendManager::Instance();
        if (auto* state = backend.GetStateManager()) {
            auto deps = state->GetDeployments();
            for (const auto& dep : deps) {
                TUIDeploymentInfo info;
                info.id = dep.id;
                info.model_name = dep.model_name;
                info.model_path = dep.model_id;  // Use model_id as fallback
                info.status = dep.status;
                info.port = dep.port;
                info.request_count = static_cast<int64_t>(dep.request_count);
                deployments_.push_back(info);
            }
        }
    }
}

// ============================================================================
// Action Methods
// ============================================================================

void TUIApplication::CancelSelectedJob() {
    if (selected_job_ >= static_cast<int>(jobs_.size())) {
        error_message_ = "No job selected";
        return;
    }

    const auto& job = jobs_[selected_job_];
    if (daemon_client_ && daemon_client_->IsConnected()) {
        std::string error;
        if (daemon_client_->CancelJob(job.id, error)) {
            status_message_ = "Job " + job.id.substr(0, 8) + " cancelled";
            RefreshJobs();
        } else {
            error_message_ = error.empty() ? "Failed to cancel job" : error;
        }
    } else {
        error_message_ = "Daemon not connected";
    }
}

void TUIApplication::DeploySelectedModel() {
    if (selected_model_ >= static_cast<int>(models_.size())) {
        error_message_ = "No model selected";
        return;
    }

    const auto& model = models_[selected_model_];
    if (model.deployed) {
        error_message_ = "Model is already deployed";
        return;
    }

    if (daemon_client_ && daemon_client_->IsConnected()) {
        try {
            int port = std::stoi(deploy_port_input_);
            int gpu_layers = std::stoi(deploy_gpu_layers_input_);
            int context_size = std::stoi(deploy_context_size_input_);

            std::string deployment_id, error;
            if (daemon_client_->DeployModel(model.path, port, gpu_layers, context_size, deployment_id, error)) {
                status_message_ = "Model " + model.name + " deployed on port " + std::to_string(port);
                RefreshModels();
                RefreshDeployments();
            } else {
                error_message_ = error.empty() ? "Failed to deploy model" : error;
            }
        } catch (const std::exception& e) {
            error_message_ = "Invalid input: " + std::string(e.what());
        }
    } else {
        error_message_ = "Daemon not connected";
    }

    show_deploy_dialog_ = false;
}

void TUIApplication::UndeploySelectedModel() {
    if (selected_model_ >= static_cast<int>(models_.size())) {
        error_message_ = "No model selected";
        return;
    }

    const auto& model = models_[selected_model_];
    if (!model.deployed) {
        error_message_ = "Model is not deployed";
        return;
    }

    if (daemon_client_ && daemon_client_->IsConnected()) {
        // Find deployment ID for this model
        std::string deployment_id;
        for (const auto& dep : deployments_) {
            if (dep.model_path == model.path || dep.model_name == model.name) {
                deployment_id = dep.id;
                break;
            }
        }

        if (deployment_id.empty()) {
            error_message_ = "Could not find deployment for model";
            return;
        }

        std::string error;
        if (daemon_client_->UndeployModel(deployment_id, error)) {
            status_message_ = "Model " + model.name + " undeployed";
            RefreshModels();
            RefreshDeployments();
        } else {
            error_message_ = error.empty() ? "Failed to undeploy model" : error;
        }
    } else {
        error_message_ = "Daemon not connected";
    }
}

void TUIApplication::StopSelectedDeployment() {
    if (selected_deployment_ >= static_cast<int>(deployments_.size())) {
        error_message_ = "No deployment selected";
        return;
    }

    const auto& dep = deployments_[selected_deployment_];

    if (daemon_client_ && daemon_client_->IsConnected()) {
        std::string error;
        if (daemon_client_->UndeployModel(dep.id, error)) {
            status_message_ = "Deployment " + dep.model_name + " stopped";
            RefreshDeployments();
            RefreshModels();
        } else {
            error_message_ = error.empty() ? "Failed to stop deployment" : error;
        }
    } else {
        error_message_ = "Daemon not connected";
    }
}

void TUIApplication::SaveSettings() {
    if (daemon_client_ && daemon_client_->IsConnected()) {
        try {
            ipc::NodeConfig config;
            config.node_name = node_name_input_;
            config.central_server_address = central_server_input_;
            config.max_concurrent_jobs = std::stoi(max_jobs_input_);

            bool restart_required = false;
            std::string error;
            if (daemon_client_->SetConfig(config, restart_required, error)) {
                status_message_ = restart_required ? "Settings saved (restart required)" : "Settings saved successfully";
                node_name_ = node_name_input_;
                central_server_address_ = central_server_input_;
                max_concurrent_jobs_ = config.max_concurrent_jobs;
                settings_dirty_ = false;
            } else {
                error_message_ = error.empty() ? "Failed to save settings" : error;
            }
        } catch (const std::exception& e) {
            error_message_ = "Invalid input: " + std::string(e.what());
        }
    } else {
        error_message_ = "Daemon not connected";
    }
}

void TUIApplication::TestConnection() {
    if (daemon_client_ && daemon_client_->IsConnected()) {
        ipc::DaemonStatus status;
        if (daemon_client_->GetStatus(status)) {
            status_message_ = "Connection test successful - Node: " + status.node_id;
            connected_ = status.connected_to_central;
        } else {
            error_message_ = "Connection test failed";
        }
    } else {
        error_message_ = "Daemon not connected";
    }
}

void TUIApplication::ClearLogs() {
    logs_.clear();
    status_message_ = "Logs cleared";
}

void TUIApplication::Reconnect() {
    if (daemon_client_) {
        daemon_client_->Disconnect();
        if (daemon_client_->Connect(daemon_address_)) {
            status_message_ = "Reconnected to daemon";
            RefreshMetrics();
            RefreshJobs();
            RefreshModels();
            RefreshDeployments();
        } else {
            error_message_ = "Failed to reconnect to " + daemon_address_;
        }
    } else {
        error_message_ = "No daemon client configured";
    }
}

Component TUIApplication::CreateUI() {
    // Create tab bar
    auto tab_bar = CreateTabBar();

    // Create views
    auto dashboard = CreateDashboardView();
    auto jobs_view = CreateJobsView();
    auto models_view = CreateModelsView();
    auto deploy_view = CreateDeployView();
    auto settings_view = CreateSettingsView();
    auto logs_view = CreateLogsView();

    auto content = Container::Tab({
        dashboard,
        jobs_view,
        models_view,
        deploy_view,
        settings_view,
        logs_view,
    }, &current_view_);

    auto status_bar = CreateStatusBar();

    auto main_container = Container::Vertical({
        tab_bar,
        content,
        status_bar,
    });

    auto main_renderer = Renderer(main_container, [=] {
        return vbox({
            // Header
            hbox({
                text(" CyxWiz Server Node ") | bold | color(Color::Cyan) | bgcolor(Color::Blue),
                filler(),
                text(" " + node_id_ + " ") | dim,
                text(connected_ ? " Connected " : " Standalone ") |
                    (connected_ ? color(Color::Green) : color(Color::Yellow)),
                text(" " + GetCurrentTime() + " "),
            }) | bgcolor(Color::GrayDark),

            // Tab bar
            tab_bar->Render(),

            separator(),

            // Content
            content->Render() | flex,

            separator(),

            // Status bar
            status_bar->Render(),
        });
    });

    // Handle keyboard input
    return CatchEvent(main_renderer, [this](Event event) {
        return HandleGlobalInput(event);
    });
}

Component TUIApplication::CreateTabBar() {
    return Renderer([this] {
        Elements tabs;
        for (size_t i = 0; i < view_names_.size(); i++) {
            bool selected = (static_cast<int>(i) == current_view_);
            std::string label = " " + std::to_string(i + 1) + ":" + view_names_[i] + " ";

            if (selected) {
                tabs.push_back(text(label) | bold | inverted);
            } else {
                tabs.push_back(text(label) | dim);
            }
            if (i < view_names_.size() - 1) {
                tabs.push_back(text("|") | dim);
            }
        }
        return hbox(tabs) | center;
    });
}

Component TUIApplication::CreateStatusBar() {
    return Renderer([this] {
        return hbox({
            text(" [1-6] Switch tabs") | dim,
            text(" | ") | dim,
            text("[Q]uit") | bold,
            text(" | ") | dim,
            text("[R]efresh") | dim,
            text(" | ") | dim,
            text("[H]elp") | dim,
            filler(),
            text(" CPU: ") | dim,
            text(std::to_string(static_cast<int>(cpu_usage_ * 100)) + "%") |
                (cpu_usage_ > 0.8f ? color(Color::Red) : color(Color::Green)),
            text(" | GPU: ") | dim,
            text(std::to_string(static_cast<int>(gpu_usage_ * 100)) + "%") |
                (gpu_usage_ > 0.8f ? color(Color::Red) : color(Color::Green)),
            text(" "),
        }) | bgcolor(Color::GrayDark);
    });
}

// ============================================================================
// Dashboard View
// ============================================================================

Component TUIApplication::CreateDashboardView() {
    return Renderer([this] {
        return vbox({
            // Resource gauges
            RenderResourceGauges() | border,

            // Device list
            RenderDeviceList() | border | flex,

            // Active jobs summary
            RenderActiveJobs() | border,

            // Earnings
            RenderEarnings() | border,
        });
    });
}

Element TUIApplication::RenderResourceGauges() {
    auto make_gauge = [](const std::string& label, float value, Color c) {
        int pct = static_cast<int>(value * 100);
        return hbox({
            text(label) | size(WIDTH, EQUAL, 6),
            gauge(value) | flex | color(c),
            text(" " + std::to_string(pct) + "%") | size(WIDTH, EQUAL, 5),
        });
    };

    return vbox({
        text(" System Resources ") | bold | center,
        separator(),
        hbox({
            make_gauge("CPU", cpu_usage_, Color::Blue) | flex,
            separator(),
            make_gauge("GPU", gpu_usage_, Color::Green) | flex,
        }),
        hbox({
            make_gauge("RAM", ram_usage_, Color::Yellow) | flex,
            separator(),
            make_gauge("VRAM", vram_usage_, Color::Magenta) | flex,
        }),
    });
}

Element TUIApplication::RenderDeviceList() {
    Elements rows;
    rows.push_back(text(" GPU Devices ") | bold | center);
    rows.push_back(separator());

    if (devices_.empty()) {
        // Show placeholder if no device data
        rows.push_back(hbox({
            text(" 0: ") | bold,
            text("GPU") | flex,
            text(" ") | size(WIDTH, EQUAL, 8),
            gauge(gpu_usage_) | size(WIDTH, EQUAL, 20),
            text(" " + std::to_string(static_cast<int>(gpu_usage_ * 100)) + "%"),
        }));
    } else {
        for (const auto& dev : devices_) {
            Color status_color = dev.in_use ? Color::Yellow : Color::Green;
            std::string status = dev.in_use ? "BUSY" : "IDLE";

            rows.push_back(hbox({
                text(" " + std::to_string(dev.id) + ": ") | bold,
                text(dev.name.substr(0, 25)) | size(WIDTH, EQUAL, 26),
                text(status) | color(status_color) | size(WIDTH, EQUAL, 6),
                gauge(dev.utilization) | size(WIDTH, EQUAL, 15) | color(Color::Green),
                text(" " + FormatBytes(dev.vram_used) + "/" + FormatBytes(dev.vram_total)),
            }));
        }
    }

    return vbox(rows);
}

Element TUIApplication::RenderActiveJobs() {
    Elements rows;
    rows.push_back(hbox({
        text(" Active Jobs ") | bold,
        text("(" + std::to_string(jobs_.size()) + ")") | dim,
    }));
    rows.push_back(separator());

    if (jobs_.empty()) {
        rows.push_back(text("  No active jobs") | dim | center);
    } else {
        // Header
        rows.push_back(hbox({
            text("ID") | bold | size(WIDTH, EQUAL, 12),
            text("Type") | bold | size(WIDTH, EQUAL, 10),
            text("Progress") | bold | flex,
            text("Epoch") | bold | size(WIDTH, EQUAL, 10),
            text("Loss") | bold | size(WIDTH, EQUAL, 10),
        }));

        for (size_t i = 0; i < std::min(jobs_.size(), size_t(5)); i++) {
            const auto& job = jobs_[i];
            rows.push_back(hbox({
                text(job.id.substr(0, 10)) | size(WIDTH, EQUAL, 12),
                text(job.type) | size(WIDTH, EQUAL, 10),
                gauge(job.progress) | flex | color(Color::Cyan),
                text(std::to_string(job.current_epoch) + "/" + std::to_string(job.total_epochs)) | size(WIDTH, EQUAL, 10),
                text(std::to_string(job.loss).substr(0, 8)) | size(WIDTH, EQUAL, 10),
            }));
        }
    }

    return vbox(rows);
}

Element TUIApplication::RenderEarnings() {
    auto format_earnings = [](double val) {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(4) << val;
        return ss.str();
    };

    return hbox({
        text(" Earnings: ") | bold,
        text("Today: ") | dim,
        text(format_earnings(earnings_today_) + " CYXWIZ") | color(Color::Green),
        text(" | Week: ") | dim,
        text(format_earnings(earnings_week_) + " CYXWIZ") | color(Color::Green),
        text(" | Month: ") | dim,
        text(format_earnings(earnings_month_) + " CYXWIZ") | color(Color::Green),
        filler(),
    });
}

// ============================================================================
// Jobs View
// ============================================================================

Component TUIApplication::CreateJobsView() {
    // Action buttons
    auto cancel_button = Button(" Cancel Job ", [this] { CancelSelectedJob(); });
    auto refresh_button = Button(" Refresh ", [this] { RefreshJobs(); });

    auto button_container = Container::Horizontal({
        cancel_button,
        refresh_button,
    });

    return Renderer(button_container, [=, this] {
        return vbox({
            text(" Job Monitor ") | bold | center,
            separator(),
            RenderJobsList() | flex,
            separator(),
            RenderJobDetails(),
            separator(),
            hbox({
                cancel_button->Render() | color(Color::Red),
                text(" "),
                refresh_button->Render() | color(Color::Blue),
                filler(),
                text(" [Up/Down] Select | [C] Cancel | [R] Refresh ") | dim,
            }),
            // Status/Error messages
            (!status_message_.empty() ? text(" " + status_message_ + " ") | color(Color::Green) | center : text("")),
            (!error_message_.empty() ? text(" " + error_message_ + " ") | color(Color::Red) | center : text("")),
        });
    });
}

Element TUIApplication::RenderJobsList() {
    if (jobs_.empty()) {
        return vbox({
            filler(),
            text("No jobs running") | center | dim,
            text("Jobs will appear here when training starts") | center | dim,
            filler(),
        });
    }

    Elements rows;
    // Header
    rows.push_back(hbox({
        text(" ") | size(WIDTH, EQUAL, 2),
        text("Job ID") | bold | size(WIDTH, EQUAL, 14),
        text("Type") | bold | size(WIDTH, EQUAL, 12),
        text("Status") | bold | size(WIDTH, EQUAL, 10),
        text("Progress") | bold | size(WIDTH, EQUAL, 20),
        text("Epoch") | bold | size(WIDTH, EQUAL, 10),
        text("Loss") | bold | size(WIDTH, EQUAL, 12),
        text("Accuracy") | bold | flex,
    }));
    rows.push_back(separator());

    for (size_t i = 0; i < jobs_.size(); i++) {
        const auto& job = jobs_[i];
        bool selected = (static_cast<int>(i) == selected_job_);

        Color status_color = Color::Yellow;
        if (job.status == "Completed") status_color = Color::Green;
        else if (job.status == "Failed") status_color = Color::Red;

        auto row = hbox({
            text(selected ? ">" : " ") | size(WIDTH, EQUAL, 2),
            text(job.id.substr(0, 12)) | size(WIDTH, EQUAL, 14),
            text(job.type) | size(WIDTH, EQUAL, 12),
            text(job.status) | color(status_color) | size(WIDTH, EQUAL, 10),
            gauge(job.progress) | size(WIDTH, EQUAL, 20) | color(Color::Cyan),
            text(std::to_string(job.current_epoch) + "/" + std::to_string(job.total_epochs)) | size(WIDTH, EQUAL, 10),
            text(std::to_string(job.loss).substr(0, 10)) | size(WIDTH, EQUAL, 12),
            text(std::to_string(job.accuracy).substr(0, 8)) | flex,
        });

        if (selected) {
            rows.push_back(row | inverted);
        } else {
            rows.push_back(row);
        }
    }

    return vbox(rows);
}

Element TUIApplication::RenderJobDetails() {
    if (jobs_.empty() || selected_job_ >= static_cast<int>(jobs_.size())) {
        return text(" Select a job to view details ") | dim | center;
    }

    const auto& job = jobs_[selected_job_];
    return vbox({
        text(" Job Details ") | bold,
        hbox({
            vbox({
                hbox({text("ID: ") | bold, text(job.id)}),
                hbox({text("Type: ") | bold, text(job.type)}),
                hbox({text("Device: ") | bold, text(job.device.empty() ? "Auto" : job.device)}),
            }) | flex,
            separator(),
            vbox({
                hbox({text("Progress: ") | bold, text(std::to_string(static_cast<int>(job.progress * 100)) + "%")}),
                hbox({text("Epoch: ") | bold, text(std::to_string(job.current_epoch) + "/" + std::to_string(job.total_epochs))}),
                hbox({text("Loss: ") | bold, text(std::to_string(job.loss))}),
            }) | flex,
        }),
    }) | border;
}

// ============================================================================
// Models View
// ============================================================================

Component TUIApplication::CreateModelsView() {
    // Deploy dialog inputs
    auto port_input = Input(&deploy_port_input_, "8082");
    auto gpu_layers_input = Input(&deploy_gpu_layers_input_, "35");
    auto context_size_input = Input(&deploy_context_size_input_, "4096");

    // Action buttons
    auto deploy_button = Button(" Deploy ", [this] {
        if (selected_model_ < static_cast<int>(models_.size()) && !models_[selected_model_].deployed) {
            show_deploy_dialog_ = true;
        }
    });
    auto undeploy_button = Button(" Undeploy ", [this] { UndeploySelectedModel(); });
    auto refresh_button = Button(" Refresh ", [this] { RefreshModels(); });

    // Deploy dialog buttons
    auto deploy_confirm = Button(" Deploy ", [this] { DeploySelectedModel(); });
    auto deploy_cancel = Button(" Cancel ", [this] { show_deploy_dialog_ = false; });

    auto button_container = Container::Horizontal({
        deploy_button,
        undeploy_button,
        refresh_button,
    });

    auto dialog_container = Container::Vertical({
        port_input,
        gpu_layers_input,
        context_size_input,
        Container::Horizontal({deploy_confirm, deploy_cancel}),
    });

    auto main_container = Container::Vertical({
        button_container,
        dialog_container,
    });

    return Renderer(main_container, [=, this] {
        Element content = vbox({
            text(" Model Browser ") | bold | center,
            separator(),
            RenderModelsList() | flex,
            separator(),
            RenderModelDetails(),
            separator(),
            hbox({
                deploy_button->Render() | color(Color::Green),
                text(" "),
                undeploy_button->Render() | color(Color::Red),
                text(" "),
                refresh_button->Render() | color(Color::Blue),
                filler(),
                text(" [Up/Down] Select | [D] Deploy | [U] Undeploy ") | dim,
            }),
            // Status/Error messages
            (!status_message_.empty() ? text(" " + status_message_ + " ") | color(Color::Green) | center : text("")),
            (!error_message_.empty() ? text(" " + error_message_ + " ") | color(Color::Red) | center : text("")),
        });

        // Deploy dialog overlay
        if (show_deploy_dialog_) {
            auto dialog = vbox({
                text(" Deploy Model ") | bold | center,
                separator(),
                hbox({text("Port: ") | size(WIDTH, EQUAL, 15), port_input->Render() | border}),
                hbox({text("GPU Layers: ") | size(WIDTH, EQUAL, 15), gpu_layers_input->Render() | border}),
                hbox({text("Context Size: ") | size(WIDTH, EQUAL, 15), context_size_input->Render() | border}),
                separator(),
                hbox({
                    deploy_confirm->Render() | color(Color::Green),
                    text(" "),
                    deploy_cancel->Render() | color(Color::Red),
                }) | center,
            }) | border | bgcolor(Color::Black) | size(WIDTH, EQUAL, 50) | center;

            return dbox({
                content,
                dialog | vcenter | hcenter,
            });
        }

        return content;
    });
}

Element TUIApplication::RenderModelsList() {
    if (models_.empty()) {
        return vbox({
            filler(),
            text("No models found") | center | dim,
            text("Models will appear here when loaded") | center | dim,
            filler(),
        });
    }

    Elements rows;
    rows.push_back(hbox({
        text(" ") | size(WIDTH, EQUAL, 2),
        text("Name") | bold | size(WIDTH, EQUAL, 20),
        text("Format") | bold | size(WIDTH, EQUAL, 10),
        text("Size") | bold | size(WIDTH, EQUAL, 12),
        text("Status") | bold | size(WIDTH, EQUAL, 10),
        text("Port") | bold | size(WIDTH, EQUAL, 8),
        text("Requests") | bold | flex,
    }));
    rows.push_back(separator());

    for (size_t i = 0; i < models_.size(); i++) {
        const auto& model = models_[i];
        bool selected = (static_cast<int>(i) == selected_model_);

        std::string status = model.deployed ? "Deployed" : "Stopped";
        Color status_color = model.deployed ? Color::Green : Color::GrayDark;

        auto row = hbox({
            text(selected ? ">" : " ") | size(WIDTH, EQUAL, 2),
            text(model.name.substr(0, 18)) | size(WIDTH, EQUAL, 20),
            text(model.format) | size(WIDTH, EQUAL, 10),
            text(FormatBytes(model.size_bytes)) | size(WIDTH, EQUAL, 12),
            text(status) | color(status_color) | size(WIDTH, EQUAL, 10),
            text(model.deployed ? std::to_string(model.port) : "-") | size(WIDTH, EQUAL, 8),
            text(std::to_string(model.request_count)) | flex,
        });

        if (selected) {
            rows.push_back(row | inverted);
        } else {
            rows.push_back(row);
        }
    }

    return vbox(rows);
}

Element TUIApplication::RenderModelDetails() {
    if (models_.empty() || selected_model_ >= static_cast<int>(models_.size())) {
        return text(" Select a model to view details ") | dim | center;
    }

    const auto& model = models_[selected_model_];
    return vbox({
        text(" Model Details ") | bold,
        hbox({
            text("Name: ") | bold, text(model.name),
            text(" | Format: ") | bold, text(model.format),
            text(" | Size: ") | bold, text(FormatBytes(model.size_bytes)),
        }),
        hbox({
            text("Path: ") | bold, text(model.path) | dim,
        }),
    }) | border;
}

// ============================================================================
// Deploy View
// ============================================================================

Component TUIApplication::CreateDeployView() {
    // Action buttons
    auto stop_button = Button(" Stop Deployment ", [this] { StopSelectedDeployment(); });
    auto refresh_button = Button(" Refresh ", [this] { RefreshDeployments(); });

    auto button_container = Container::Horizontal({
        stop_button,
        refresh_button,
    });

    return Renderer(button_container, [=, this] {
        return vbox({
            text(" Model Deployment ") | bold | center,
            separator(),
            RenderDeploymentsList() | flex,
            separator(),
            hbox({
                stop_button->Render() | color(Color::Red),
                text(" "),
                refresh_button->Render() | color(Color::Blue),
                filler(),
                text(" [Up/Down] Select | [S] Stop | [R] Refresh ") | dim,
            }),
            // Status/Error messages
            (!status_message_.empty() ? text(" " + status_message_ + " ") | color(Color::Green) | center : text("")),
            (!error_message_.empty() ? text(" " + error_message_ + " ") | color(Color::Red) | center : text("")),
        });
    });
}

Element TUIApplication::RenderDeploymentsList() {
    Elements rows;
    rows.push_back(hbox({
        text(" Active Deployments ") | bold,
        text("(" + std::to_string(deployments_.size()) + ")") | dim,
    }));
    rows.push_back(separator());

    if (deployments_.empty()) {
        rows.push_back(filler());
        rows.push_back(text("No active deployments") | center | dim);
        rows.push_back(text("Deploy a model from the Models tab (3)") | center | dim);
        rows.push_back(filler());
    } else {
        rows.push_back(hbox({
            text(" ") | size(WIDTH, EQUAL, 2),
            text("Model") | bold | size(WIDTH, EQUAL, 20),
            text("Status") | bold | size(WIDTH, EQUAL, 10),
            text("Port") | bold | size(WIDTH, EQUAL, 8),
            text("Requests") | bold | size(WIDTH, EQUAL, 12),
            text("Path") | bold | flex,
        }));
        rows.push_back(separator());

        for (size_t i = 0; i < deployments_.size(); i++) {
            const auto& dep = deployments_[i];
            bool selected = (static_cast<int>(i) == selected_deployment_);

            Color status_color = (dep.status == "Running") ? Color::Green : Color::Yellow;

            auto row = hbox({
                text(selected ? ">" : " ") | size(WIDTH, EQUAL, 2),
                text(dep.model_name.substr(0, 18)) | size(WIDTH, EQUAL, 20),
                text(dep.status) | color(status_color) | size(WIDTH, EQUAL, 10),
                text(std::to_string(dep.port)) | size(WIDTH, EQUAL, 8),
                text(std::to_string(dep.request_count)) | size(WIDTH, EQUAL, 12),
                text(dep.model_path.length() > 30 ? "..." + dep.model_path.substr(dep.model_path.length() - 27) : dep.model_path) | dim | flex,
            });

            if (selected) {
                rows.push_back(row | inverted);
            } else {
                rows.push_back(row);
            }
        }
    }

    return vbox(rows);
}

Element TUIApplication::RenderDeploymentControls() {
    return hbox({
        text(" Controls: ") | bold,
        text("[D]eploy model") | dim,
        text(" | ") | dim,
        text("[S]top deployment") | dim,
        text(" | ") | dim,
        text("[R]estart") | dim,
        filler(),
    }) | border;
}

// ============================================================================
// Settings View
// ============================================================================

Component TUIApplication::CreateSettingsView() {
    // Use only Buttons - no Input/Dropdown (they crash on Windows terminals)
    // Settings are displayed as read-only text

    auto test_button = Button(" Test Connection ", [this] { TestConnection(); });
    auto reconnect_button = Button(" Reconnect ", [this] { Reconnect(); });
    auto tls_toggle = Button(use_tls_ ? " [X] TLS " : " [ ] TLS ", [this] {
        use_tls_ = !use_tls_;
    });

    // Container for focus navigation - only buttons
    auto container = Container::Horizontal({
        test_button,
        reconnect_button,
        tls_toggle,
    });

    return Renderer(container, [=, this] {
        std::string conn_status = (daemon_client_ && daemon_client_->IsConnected())
            ? "Connected" : "Disconnected";
        Color status_color = (daemon_client_ && daemon_client_->IsConnected())
            ? Color::Green : Color::Red;

        return vbox({
            text(" Settings ") | bold | center,
            separator(),

            // Connection Settings (read-only)
            vbox({
                text(" Connection Settings ") | bold,
                separator(),
                hbox({
                    text("Daemon Address: ") | size(WIDTH, EQUAL, 20),
                    text(daemon_address_) | color(Color::Cyan),
                }),
                hbox({
                    text("Status: ") | size(WIDTH, EQUAL, 20),
                    text(conn_status) | color(status_color),
                }),
                hbox({
                    text("TLS: ") | size(WIDTH, EQUAL, 20),
                    text(use_tls_ ? "Enabled" : "Disabled") | color(use_tls_ ? Color::Green : Color::Yellow),
                }),
            }) | border,

            // Node Configuration (read-only)
            vbox({
                text(" Node Configuration ") | bold,
                separator(),
                hbox({
                    text("Node Name: ") | size(WIDTH, EQUAL, 20),
                    text(node_name_.empty() ? "(not set)" : node_name_) | color(Color::Cyan),
                }),
                hbox({
                    text("Node ID: ") | size(WIDTH, EQUAL, 20),
                    text(node_id_.empty() ? "(not assigned)" : node_id_) | dim,
                }),
                hbox({
                    text("Max Jobs: ") | size(WIDTH, EQUAL, 20),
                    text(std::to_string(max_concurrent_jobs_)) | color(Color::Cyan),
                }),
            }) | border,

            // Central Server (read-only)
            vbox({
                text(" Central Server ") | bold,
                separator(),
                hbox({
                    text("Server Address: ") | size(WIDTH, EQUAL, 20),
                    text(central_server_address_) | color(Color::Cyan),
                }),
                hbox({
                    text("Status: ") | size(WIDTH, EQUAL, 20),
                    text(connected_ ? "Connected" : "Not connected") |
                        (connected_ ? color(Color::Green) : color(Color::Yellow)),
                }),
            }) | border,

            // Action Buttons
            hbox({
                test_button->Render() | color(Color::Blue),
                text("  "),
                reconnect_button->Render() | color(Color::Yellow),
                text("  "),
                tls_toggle->Render() | color(Color::Magenta),
            }) | center,

            // Status/Error messages
            (!status_message_.empty() ? text(" " + status_message_ + " ") | color(Color::Green) | center : text("")),
            (!error_message_.empty() ? text(" " + error_message_ + " ") | color(Color::Red) | center : text("")),

            filler(),

            // Help
            hbox({
                text(" [Tab] Switch | [Enter] Activate | Edit via config file ") | dim,
            }) | center,
        });
    });
}

Element TUIApplication::RenderConnectionSettings() {
    // No longer used - integrated into CreateSettingsView
    return text("");
}

Element TUIApplication::RenderNodeSettings() {
    // No longer used - integrated into CreateSettingsView
    return text("");
}

// ============================================================================
// Logs View
// ============================================================================

Component TUIApplication::CreateLogsView() {
    // Use only Buttons - no Dropdown (it crashes on Windows terminals)
    auto clear_button = Button(" Clear Logs ", [this] { ClearLogs(); });
    auto refresh_button = Button(" Refresh ", [this] { RefreshLogs(); });

    // Filter cycle button instead of dropdown
    auto filter_button = Button(" Filter: All ", [this] {
        log_level_filter_ = (log_level_filter_ + 1) % 5;
    });

    auto controls = Container::Horizontal({
        filter_button,
        clear_button,
        refresh_button,
    });

    return Renderer(controls, [=, this] {
        Elements log_lines;

        // Get current filter name
        std::string filter_names[] = {"All", "Error", "Warn", "Info", "Debug"};
        std::string current_filter = filter_names[log_level_filter_];

        if (logs_.empty()) {
            log_lines.push_back(filler());
            log_lines.push_back(text("No logs available") | center | dim);
            log_lines.push_back(text("Logs will appear here as events occur") | center | dim);
            log_lines.push_back(filler());
        } else {
            // Filter logs based on selected level
            std::string filter_level;
            switch (log_level_filter_) {
                case 1: filter_level = "ERROR"; break;
                case 2: filter_level = "WARN"; break;
                case 3: filter_level = "INFO"; break;
                case 4: filter_level = "DEBUG"; break;
                default: filter_level = ""; break;
            }

            int shown_count = 0;
            for (const auto& log : logs_) {
                // Apply filter
                if (!filter_level.empty() && log.level != filter_level) {
                    continue;
                }

                Color level_color = Color::White;
                if (log.level == "ERROR") level_color = Color::Red;
                else if (log.level == "WARN") level_color = Color::Yellow;
                else if (log.level == "INFO") level_color = Color::Green;
                else if (log.level == "DEBUG") level_color = Color::GrayLight;

                log_lines.push_back(hbox({
                    text(log.timestamp) | dim | size(WIDTH, EQUAL, 10),
                    text(" [" + log.level + "] ") | color(level_color) | size(WIDTH, EQUAL, 9),
                    text(log.message) | flex,
                }));
                shown_count++;
            }

            if (shown_count == 0) {
                log_lines.push_back(text("No logs matching filter") | center | dim);
            }
        }

        return vbox({
            text(" System Logs ") | bold | center,
            separator(),
            vbox(log_lines) | flex | frame,
            separator(),
            hbox({
                text("Filter: " + current_filter + " ") | dim,
                filter_button->Render() | color(Color::Cyan),
                text(" "),
                clear_button->Render() | color(Color::Red),
                text(" "),
                refresh_button->Render() | color(Color::Blue),
                filler(),
                text("Total: " + std::to_string(logs_.size()) + " entries") | dim,
            }),
            // Status/Error messages
            (!status_message_.empty() ? text(" " + status_message_ + " ") | color(Color::Green) | center : text("")),
        });
    });
}

// ============================================================================
// Input Handling
// ============================================================================

bool TUIApplication::HandleGlobalInput(Event event) {
    // Clear messages on any input
    if (!status_message_.empty() || !error_message_.empty()) {
        status_message_.clear();
        error_message_.clear();
    }

    // Quit
    if (event == Event::Character('q') || event == Event::Character('Q')) {
        running_.store(false);
        return true;
    }

    // Tab switching with number keys
    if (event == Event::Character('1')) { current_view_ = 0; return true; }
    if (event == Event::Character('2')) { current_view_ = 1; return true; }
    if (event == Event::Character('3')) { current_view_ = 2; return true; }
    if (event == Event::Character('4')) { current_view_ = 3; return true; }
    if (event == Event::Character('5')) { current_view_ = 4; return true; }
    if (event == Event::Character('6')) { current_view_ = 5; return true; }

    // Arrow keys for list navigation (only when not in Settings view)
    if (current_view_ != 4) {  // Not Settings view
        if (event == Event::ArrowUp) {
            if (current_view_ == 1 && selected_job_ > 0) selected_job_--;
            else if (current_view_ == 2 && selected_model_ > 0) selected_model_--;
            else if (current_view_ == 3 && selected_deployment_ > 0) selected_deployment_--;
            return true;
        }
        if (event == Event::ArrowDown) {
            if (current_view_ == 1 && selected_job_ < static_cast<int>(jobs_.size()) - 1) selected_job_++;
            else if (current_view_ == 2 && selected_model_ < static_cast<int>(models_.size()) - 1) selected_model_++;
            else if (current_view_ == 3 && selected_deployment_ < static_cast<int>(deployments_.size()) - 1) selected_deployment_++;
            return true;
        }
    }

    // View-specific shortcuts
    switch (current_view_) {
        case 1:  // Jobs view
            if (event == Event::Character('c') || event == Event::Character('C')) {
                CancelSelectedJob();
                return true;
            }
            break;

        case 2:  // Models view
            if (event == Event::Character('d') || event == Event::Character('D')) {
                if (selected_model_ < static_cast<int>(models_.size()) && !models_[selected_model_].deployed) {
                    show_deploy_dialog_ = true;
                }
                return true;
            }
            if (event == Event::Character('u') || event == Event::Character('U')) {
                UndeploySelectedModel();
                return true;
            }
            break;

        case 3:  // Deploy view
            if (event == Event::Character('s') || event == Event::Character('S')) {
                StopSelectedDeployment();
                return true;
            }
            break;

        case 5:  // Logs view
            if (event == Event::Character('c') || event == Event::Character('C')) {
                ClearLogs();
                return true;
            }
            break;
    }

    // Close deploy dialog with Escape
    if (show_deploy_dialog_ && event == Event::Escape) {
        show_deploy_dialog_ = false;
        return true;
    }

    // Tab key cycles through views
    if (event == Event::Tab) {
        current_view_ = (current_view_ + 1) % static_cast<int>(view_names_.size());
        return true;
    }

    // Help toggle
    if (event == Event::Character('h') || event == Event::Character('H') && !show_deploy_dialog_) {
        show_help_ = !show_help_;
        return true;
    }

    // Refresh
    if (event == Event::Character('r') || event == Event::Character('R')) {
        RefreshMetrics();
        RefreshJobs();
        RefreshModels();
        RefreshDeployments();
        RefreshLogs();
        status_message_ = "Data refreshed";
        return true;
    }

    return false;
}

} // namespace cyxwiz::servernode::tui
