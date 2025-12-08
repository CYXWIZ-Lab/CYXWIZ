// job_monitor_panel.cpp - Active jobs monitoring with daemon integration
#include "gui/panels/job_monitor_panel.h"
#include "gui/icons.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <chrono>
#include <ctime>

namespace cyxwiz::servernode::gui {

void JobMonitorPanel::Render() {
    ImGui::PushFont(GetSafeFont(FONT_LARGE));
    ImGui::Text("%s Job Monitor", ICON_FA_BARS_PROGRESS);
    ImGui::PopFont();
    ImGui::Separator();

    // Connection status
    if (IsDaemonConnected()) {
        ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "%s Daemon Connected", ICON_FA_LINK);
    } else {
        ImGui::TextColored(ImVec4(0.8f, 0.3f, 0.3f, 1.0f), "%s Daemon Disconnected", ICON_FA_LINK_SLASH);
        ImGui::TextDisabled("Connect to daemon to monitor jobs.");
        return;
    }

    ImGui::Spacing();

    // Controls
    if (ImGui::Button(ICON_FA_ROTATE " Refresh")) {
        RefreshJobs();
    }

    ImGui::SameLine();
    ImGui::Checkbox("Show Completed", &show_completed_);
    ImGui::SameLine();
    ImGui::Checkbox("Show Failed", &show_failed_);

    ImGui::Spacing();

    // Load jobs if not loaded
    if (!jobs_loaded_) {
        RefreshJobs();
        jobs_loaded_ = true;
    }

    // Count active jobs
    int active_count = 0;
    int pending_count = 0;
    for (const auto& job : jobs_) {
        if (job.status == 2) active_count++;  // running
        if (job.status == 1) pending_count++; // pending
    }

    ImGui::TextDisabled("%d active, %d pending jobs", active_count, pending_count);
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    if (jobs_.empty()) {
        ImGui::TextDisabled("No jobs to display.");
        ImGui::Text("Jobs will appear here when submitted to this node.");
        return;
    }

    // Render job cards
    for (const auto& job : jobs_) {
        // Filter based on status
        if (job.status == 4 && !show_completed_) continue;  // completed
        if ((job.status == 5 || job.status == 6) && !show_failed_) continue;  // failed/cancelled

        RenderJobCard(job);
    }

    // Dialogs
    RenderCancelDialog();
}

void JobMonitorPanel::RenderJobCard(const ipc::JobInfo& job) {
    ImGui::PushID(job.id.c_str());

    // Create a visual card
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
    ImGui::BeginChild(("JobCard_" + job.id).c_str(), ImVec2(-1, 180), true);

    // Header: Job ID and Status
    ImGui::Text("%s %s", ICON_FA_CUBE, job.model_name.empty() ? job.id.c_str() : job.model_name.c_str());
    ImGui::SameLine(ImGui::GetContentRegionAvail().x - 80);
    ImGui::TextColored(GetStatusColor(job.status), "%s", GetStatusText(job.status));

    ImGui::TextDisabled("Type: %s | ID: %s", job.type.c_str(), job.id.c_str());
    ImGui::Spacing();

    // Progress section
    if (job.status == 2 || job.status == 3) {  // running or paused
        ImGui::Text("Epoch %d / %d", job.current_epoch, job.total_epochs);
        ImGui::ProgressBar(job.progress, ImVec2(-1, 18));
    }

    // Metrics
    ImGui::Columns(3, nullptr, false);

    ImGui::Text("Loss");
    ImGui::TextColored(ImVec4(0.8f, 0.6f, 0.3f, 1.0f), "%.6f", job.loss);
    ImGui::NextColumn();

    ImGui::Text("Accuracy");
    ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "%.2f%%", job.accuracy * 100);
    ImGui::NextColumn();

    ImGui::Text("Earnings");
    ImGui::TextColored(ImVec4(0.3f, 0.6f, 0.9f, 1.0f), "$%.4f", job.earnings);
    ImGui::Columns(1);

    // Loss mini-chart (if we have history)
    auto it = loss_history_.find(job.id);
    if (it != loss_history_.end() && !it->second.empty()) {
        ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(0, 0));
        if (ImPlot::BeginPlot("##LossSparkline", ImVec2(-1, 40), ImPlotFlags_NoTitle | ImPlotFlags_NoLegend |
                             ImPlotFlags_NoMouseText | ImPlotFlags_NoInputs | ImPlotFlags_NoMenus)) {
            ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_NoDecorations);
            ImPlot::SetNextLineStyle(ImVec4(0.8f, 0.6f, 0.3f, 1.0f));
            ImPlot::PlotLine("Loss", it->second.data(), (int)it->second.size());
            ImPlot::EndPlot();
        }
        ImPlot::PopStyleVar();
    }

    ImGui::Spacing();

    // Action buttons (only for active jobs)
    if (job.status == 1 || job.status == 2 || job.status == 3) {  // pending, running, paused
        if (ImGui::Button(ICON_FA_STOP " Cancel")) {
            pending_cancel_id_ = job.id;
            pending_cancel_name_ = job.model_name.empty() ? job.id : job.model_name;
            cancel_error_.clear();
            show_cancel_dialog_ = true;
        }
    }

    if (job.started_at > 0) {
        auto now = std::chrono::system_clock::now();
        auto started = std::chrono::system_clock::from_time_t(job.started_at);
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - started).count();
        ImGui::SameLine();
        ImGui::TextDisabled("Running: %s", FormatDuration(duration).c_str());
    }

    ImGui::EndChild();
    ImGui::PopStyleVar();

    ImGui::Spacing();
    ImGui::PopID();
}

void JobMonitorPanel::RenderCancelDialog() {
    if (!show_cancel_dialog_) return;

    ImGui::OpenPopup("Cancel Job?");

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    if (ImGui::BeginPopupModal("Cancel Job?", &show_cancel_dialog_, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("%s Are you sure you want to cancel:", ICON_FA_TRIANGLE_EXCLAMATION);
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "  %s", pending_cancel_name_.c_str());
        ImGui::Spacing();
        ImGui::Text("This will stop the job and cannot be undone.");
        ImGui::Spacing();

        if (!cancel_error_.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "%s %s", ICON_FA_TRIANGLE_EXCLAMATION, cancel_error_.c_str());
            ImGui::Spacing();
        }

        ImGui::Separator();
        ImGui::Spacing();

        if (ImGui::Button(ICON_FA_STOP " Cancel Job", ImVec2(120, 0))) {
            auto* client = GetDaemonClient();
            if (client && client->IsConnected()) {
                std::string error;
                if (client->CancelJob(pending_cancel_id_, error)) {
                    spdlog::info("Cancelled job: {}", pending_cancel_name_);
                    show_cancel_dialog_ = false;
                    RefreshJobs();
                } else {
                    cancel_error_ = error.empty() ? "Cancel failed" : error;
                }
            } else {
                cancel_error_ = "Daemon not connected";
            }
        }

        ImGui::SameLine();
        if (ImGui::Button("Keep Running", ImVec2(120, 0))) {
            show_cancel_dialog_ = false;
            cancel_error_.clear();
        }

        ImGui::EndPopup();
    }
}

void JobMonitorPanel::RefreshJobs() {
    jobs_.clear();
    auto* client = GetDaemonClient();
    if (client && client->IsConnected()) {
        // Get all jobs (including completed if filter is on)
        bool include_completed = show_completed_ || show_failed_;
        client->ListJobs(jobs_, include_completed);
        spdlog::debug("Loaded {} jobs", jobs_.size());

        // Store loss values for history (simulate - in real implementation, daemon would track history)
        for (const auto& job : jobs_) {
            if (job.status == 2) {  // running
                auto& history = loss_history_[job.id];
                history.push_back(job.loss);
                if (history.size() > 60) {
                    history.erase(history.begin());
                }
            }
        }
    }
}

const char* JobMonitorPanel::GetStatusText(int status) {
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

ImVec4 JobMonitorPanel::GetStatusColor(int status) {
    switch (status) {
        case 1: return ImVec4(0.7f, 0.7f, 0.7f, 1.0f);  // Gray - pending
        case 2: return ImVec4(0.3f, 0.8f, 0.3f, 1.0f);  // Green - running
        case 3: return ImVec4(1.0f, 0.8f, 0.3f, 1.0f);  // Yellow - paused
        case 4: return ImVec4(0.3f, 0.6f, 0.9f, 1.0f);  // Blue - completed
        case 5: return ImVec4(1.0f, 0.3f, 0.3f, 1.0f);  // Red - failed
        case 6: return ImVec4(0.5f, 0.5f, 0.5f, 1.0f);  // Dark gray - cancelled
        default: return ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
    }
}

std::string JobMonitorPanel::FormatDuration(int64_t seconds) {
    if (seconds < 60) {
        return std::to_string(seconds) + "s";
    } else if (seconds < 3600) {
        return std::to_string(seconds / 60) + "m " + std::to_string(seconds % 60) + "s";
    } else {
        int64_t hours = seconds / 3600;
        int64_t mins = (seconds % 3600) / 60;
        return std::to_string(hours) + "h " + std::to_string(mins) + "m";
    }
}

} // namespace cyxwiz::servernode::gui
