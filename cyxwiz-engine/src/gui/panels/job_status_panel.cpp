#include "job_status_panel.h"
#include "common.pb.h"
#include "../../network/p2p_client.h"
#include <spdlog/spdlog.h>
#include <imgui.h>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <ctime>

// Undefine Windows macros that conflict with protobuf STATUS_* enums
#ifdef STATUS_PENDING
#undef STATUS_PENDING
#endif
#ifdef STATUS_ERROR
#undef STATUS_ERROR
#endif

namespace cyxwiz {

JobStatusPanel::JobStatusPanel()
    : Panel("Job Status & Orchestration")
    , job_manager_(nullptr)
    , selected_job_id_("")
    , auto_refresh_(true)
    , refresh_interval_(5.0f)  // Refresh every 5 seconds
    , last_refresh_(std::chrono::steady_clock::now())
    , show_all_jobs_(false)  // Only show P2P jobs by default
    , show_completed_jobs_(true)
    , max_displayed_jobs_(20)
    , submit_job_dialog_open_(false)
{
    model_definition_input_[0] = '\0';
    dataset_uri_input_[0] = '\0';
}

JobStatusPanel::~JobStatusPanel() {
}

void JobStatusPanel::SetJobManager(network::JobManager* job_manager) {
    job_manager_ = job_manager;
}

void JobStatusPanel::Refresh() {
    last_refresh_ = std::chrono::steady_clock::now();
}

void JobStatusPanel::Render() {
    if (!visible_) {
        return;
    }

    if (!ImGui::Begin(name_.c_str(), &visible_)) {
        ImGui::End();
        return;
    }

    // Check if auto-refresh is enabled
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_refresh_).count();
    if (auto_refresh_ && elapsed >= refresh_interval_) {
        Refresh();
    }

    // Top toolbar with controls
    RenderJobControls();

    ImGui::Separator();

    // Split view: Job list on left, details on right
    if (ImGui::BeginTable("JobStatusSplit", 2, ImGuiTableFlags_Resizable)) {
        ImGui::TableSetupColumn("Jobs", ImGuiTableColumnFlags_WidthFixed, 300.0f);
        ImGui::TableSetupColumn("Details", ImGuiTableColumnFlags_WidthStretch);

        ImGui::TableNextRow();

        // Left column: Job list
        ImGui::TableSetColumnIndex(0);
        RenderJobList();

        // Right column: Selected job details
        ImGui::TableSetColumnIndex(1);
        RenderSelectedJobDetails();

        ImGui::EndTable();
    }

    ImGui::End();
}

void JobStatusPanel::RenderJobControls() {
    // Auto-refresh toggle
    if (ImGui::Checkbox("Auto-refresh", &auto_refresh_)) {
        if (auto_refresh_) {
            Refresh();
        }
    }

    ImGui::SameLine();

    // Manual refresh button
    if (ImGui::Button("Refresh Now")) {
        Refresh();
    }

    ImGui::SameLine();

    // Filter options
    ImGui::Checkbox("Show All Jobs", &show_all_jobs_);
    ImGui::SameLine();
    ImGui::Checkbox("Show Completed", &show_completed_jobs_);

    ImGui::SameLine();

    // Connection status
    if (job_manager_ && job_manager_->IsConnected()) {
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Connected to Server");
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Disconnected");
    }
}

void JobStatusPanel::RenderJobList() {
    ImGui::Text("Active Jobs");
    ImGui::Separator();

    if (!job_manager_) {
        ImGui::TextDisabled("No JobManager connected");
        return;
    }

    const auto& jobs = job_manager_->GetActiveJobs();

    if (jobs.empty()) {
        ImGui::TextDisabled("No active jobs");
        return;
    }

    // Job list
    if (ImGui::BeginChild("JobListScrollArea", ImVec2(0, 0), true)) {
        int displayed_count = 0;

        for (const auto& job : jobs) {
            // Apply filters
            if (!show_all_jobs_ && !job.is_p2p_job) {
                continue;  // Skip non-P2P jobs
            }

            int status_code = job.status.status();
            bool is_completed = (status_code == cyxwiz::protocol::STATUS_SUCCESS ||
                                status_code == cyxwiz::protocol::STATUS_COMPLETED ||
                                status_code == cyxwiz::protocol::STATUS_FAILED ||
                                status_code == cyxwiz::protocol::STATUS_CANCELLED);

            if (!show_completed_jobs_ && is_completed) {
                continue;  // Skip completed jobs
            }

            if (displayed_count >= max_displayed_jobs_) {
                break;
            }

            // Job item
            bool is_selected = (selected_job_id_ == job.job_id);
            ImVec4 status_color = GetStatusColor(status_code);

            ImGui::PushStyleColor(ImGuiCol_Text, status_color);

            std::string label = job.job_id.substr(0, 8) + "... (" + GetStatusString(status_code) + ")";
            if (job.is_p2p_job) {
                label += " [P2P]";
            }

            if (ImGui::Selectable(label.c_str(), is_selected)) {
                SelectJob(job.job_id);
            }

            ImGui::PopStyleColor();

            // Tooltip with full job ID
            if (ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                ImGui::Text("Full Job ID: %s", job.job_id.c_str());
                ImGui::EndTooltip();
            }

            displayed_count++;
        }
    }
    ImGui::EndChild();
}

void JobStatusPanel::RenderSelectedJobDetails() {
    if (selected_job_id_.empty()) {
        ImGui::TextDisabled("No job selected");
        ImGui::TextDisabled("Select a job from the list to view details");
        return;
    }

    if (!job_manager_) {
        ImGui::TextDisabled("No JobManager connected");
        return;
    }

    // Find selected job
    const auto& jobs = job_manager_->GetActiveJobs();
    const network::ActiveJob* selected_job = nullptr;

    for (const auto& job : jobs) {
        if (job.job_id == selected_job_id_) {
            selected_job = &job;
            break;
        }
    }

    if (!selected_job) {
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Selected job not found");
        selected_job_id_.clear();
        return;
    }

    // Job details header
    ImGui::Text("Job Details");
    ImGui::Separator();

    // Basic info
    ImGui::Text("Job ID: %s", selected_job->job_id.c_str());
    ImGui::Text("Job Type: %s", selected_job->is_p2p_job ? "P2P Training" : "Traditional");

    int status_code = selected_job->status.status();
    ImVec4 status_color = GetStatusColor(status_code);
    ImGui::Text("Status: ");
    ImGui::SameLine();
    ImGui::TextColored(status_color, "%s", GetStatusString(status_code).c_str());

    ImGui::Text("Progress: %.1f%%", selected_job->status.progress() * 100.0f);

    // Show epoch if available
    if (selected_job->status.current_epoch() > 0) {
        ImGui::Text("Epoch: %d", selected_job->status.current_epoch());
    }

    // Show metrics from P2P updates
    if (selected_job->status.metrics_size() > 0) {
        const auto& metrics = selected_job->status.metrics();
        if (metrics.count("loss")) {
            ImGui::Text("Loss: %.4f", metrics.at("loss"));
        }
        if (metrics.count("accuracy")) {
            ImGui::Text("Accuracy: %.2f%%", metrics.at("accuracy") * 100.0);
        }
    }

    // Render training graphs for P2P jobs
    if (selected_job->is_p2p_job) {
        RenderTrainingGraphs(selected_job);
    }

    ImGui::Separator();

    // P2P-specific information
    if (selected_job->is_p2p_job) {
        ImGui::Text("P2P Orchestration Status");
        ImGui::Indent();

        // NodeAssignment visualization
        RenderNodeAssignment();

        // P2P connection status
        RenderP2PConnectionStatus();

        ImGui::Unindent();
    }

    // Error information
    if (selected_job->status.has_error()) {
        ImGui::Separator();
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Error");
        ImGui::TextWrapped("%s", selected_job->status.error().message().c_str());
    }

    // Job action buttons
    ImGui::Separator();
    ImGui::Text("Actions:");
    ImGui::Spacing();

    // Reuse status_code from above
    bool is_active = (status_code == cyxwiz::protocol::STATUS_PENDING ||
                      status_code == cyxwiz::protocol::STATUS_IN_PROGRESS);

    // Cancel button - only for active jobs
    if (is_active) {
        if (ImGui::Button("Cancel", ImVec2(70, 0))) {
            std::string job_to_cancel = selected_job_id_;  // Copy before cancel
            job_manager_->CancelJob(job_to_cancel);
            spdlog::info("Cancel requested for job: {}", job_to_cancel);
        }
        ImGui::SameLine();

        // Pause/Resume buttons for P2P jobs - show always, disabled if not connected
        if (selected_job->is_p2p_job) {
            bool p2p_connected = selected_job->p2p_client && selected_job->p2p_client->IsConnected();

            if (!p2p_connected) {
                ImGui::BeginDisabled();
            }
            if (ImGui::Button("Pause", ImVec2(60, 0))) {
                if (p2p_connected && selected_job->p2p_client->PauseTraining()) {
                    spdlog::info("Pause sent for job: {}", selected_job_id_);
                }
            }
            ImGui::SameLine();

            if (ImGui::Button("Resume", ImVec2(60, 0))) {
                if (p2p_connected && selected_job->p2p_client->ResumeTraining()) {
                    spdlog::info("Resume sent for job: {}", selected_job_id_);
                }
            }
            ImGui::SameLine();
            if (!p2p_connected) {
                ImGui::EndDisabled();
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "(Waiting for P2P)");
            }
        }
    }

    // Delete button - always visible, but disabled for active jobs
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.2f, 0.2f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f, 0.3f, 0.3f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.6f, 0.1f, 0.1f, 1.0f));
    if (is_active) {
        ImGui::BeginDisabled();
    }
    if (ImGui::Button("Delete", ImVec2(70, 0))) {
        std::string job_to_delete = selected_job_id_;
        if (job_manager_->DeleteJob(job_to_delete)) {
            selected_job_id_.clear();  // Clear selection after delete
            spdlog::info("Job deleted: {}", job_to_delete);
        }
    }
    if (is_active) {
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip("Cancel job first before deleting");
        }
    }
    ImGui::PopStyleColor(3);

    ImGui::SameLine();

    // Hide button - removes from local list only
    if (ImGui::Button("Hide", ImVec2(60, 0))) {
        std::string job_to_hide = selected_job_id_;
        job_manager_->RemoveLocalJob(job_to_hide);
        selected_job_id_.clear();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Remove from this list only.\nJob remains on server.");
    }
}

void JobStatusPanel::RenderNodeAssignment() {
    if (!job_manager_) {
        return;
    }

    const auto& jobs = job_manager_->GetActiveJobs();
    const network::ActiveJob* selected_job = nullptr;

    for (const auto& job : jobs) {
        if (job.job_id == selected_job_id_) {
            selected_job = &job;
            break;
        }
    }

    if (!selected_job) {
        return;
    }

    ImGui::Separator();

    if (selected_job->assigned_node_address.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Waiting for node assignment...");
        ImGui::TextDisabled("Central Server is finding a suitable node");
    } else {
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Node Assignment Received!");

        ImGui::Text("Node Endpoint: %s", selected_job->assigned_node_address.c_str());

        // JWT token display (truncated)
        if (!selected_job->p2p_auth_token.empty()) {
            std::string token_preview = selected_job->p2p_auth_token.substr(0, 30) + "...";
            ImGui::Text("Auth Token: %s", token_preview.c_str());

            // Token expiration (we don't have this directly, but could add it to ActiveJob)
            // For now, show that token exists
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "JWT Token Ready");

            if (ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                ImGui::Text("Full JWT Token:");
                ImGui::TextWrapped("%s", selected_job->p2p_auth_token.c_str());
                ImGui::EndTooltip();
            }
        }
    }
}

void JobStatusPanel::RenderP2PConnectionStatus() {
    if (!job_manager_) {
        return;
    }

    const auto& jobs = job_manager_->GetActiveJobs();
    const network::ActiveJob* selected_job = nullptr;

    for (const auto& job : jobs) {
        if (job.job_id == selected_job_id_) {
            selected_job = &job;
            break;
        }
    }

    if (!selected_job) {
        return;
    }

    ImGui::Separator();
    ImGui::Text("P2P Connection");

    if (!selected_job->p2p_client) {
        if (!selected_job->assigned_node_address.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Auto-connecting...");
            ImGui::TextDisabled("Engine is establishing P2P connection");
        } else {
            ImGui::TextDisabled("Not connected (waiting for assignment)");
        }
    } else {
        // P2P client exists, check connection status
        auto p2p_client = selected_job->p2p_client;

        if (p2p_client->IsConnected()) {
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Connected!");
            ImGui::Text("Endpoint: %s", selected_job->assigned_node_address.c_str());
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Training stream active");
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Connection failed");

            std::string error = p2p_client->GetLastError();
            if (!error.empty()) {
                ImGui::TextWrapped("Error: %s", error.c_str());
            }
        }
    }
}

void JobStatusPanel::SelectJob(const std::string& job_id) {
    selected_job_id_ = job_id;
    spdlog::debug("Selected job: {}", job_id);
}

std::string JobStatusPanel::GetStatusString(int status_code) const {
    // Using if-else to avoid Windows macro conflicts with STATUS_*
    if (status_code == cyxwiz::protocol::STATUS_UNKNOWN) return "UNKNOWN";
    if (status_code == cyxwiz::protocol::STATUS_SUCCESS) return "SUCCESS";
    if (status_code == cyxwiz::protocol::STATUS_ERROR) return "ERROR";
    if (status_code == cyxwiz::protocol::STATUS_PENDING) return "PENDING";
    if (status_code == cyxwiz::protocol::STATUS_IN_PROGRESS) return "IN PROGRESS";
    if (status_code == cyxwiz::protocol::STATUS_COMPLETED) return "COMPLETED";
    if (status_code == cyxwiz::protocol::STATUS_FAILED) return "FAILED";
    if (status_code == cyxwiz::protocol::STATUS_CANCELLED) return "CANCELLED";
    return "INVALID";
}

ImVec4 JobStatusPanel::GetStatusColor(int status_code) const {
    // Using if-else to avoid Windows macro conflicts with STATUS_*
    if (status_code == cyxwiz::protocol::STATUS_UNKNOWN)
        return ImVec4(0.5f, 0.5f, 0.5f, 1.0f);  // Gray
    if (status_code == cyxwiz::protocol::STATUS_PENDING)
        return ImVec4(1.0f, 1.0f, 0.0f, 1.0f);  // Yellow
    if (status_code == cyxwiz::protocol::STATUS_IN_PROGRESS)
        return ImVec4(0.0f, 0.5f, 1.0f, 1.0f);  // Blue
    if (status_code == cyxwiz::protocol::STATUS_SUCCESS || status_code == cyxwiz::protocol::STATUS_COMPLETED)
        return ImVec4(0.0f, 1.0f, 0.0f, 1.0f);  // Green
    if (status_code == cyxwiz::protocol::STATUS_FAILED ||
        status_code == cyxwiz::protocol::STATUS_CANCELLED ||
        status_code == cyxwiz::protocol::STATUS_ERROR)
        return ImVec4(1.0f, 0.0f, 0.0f, 1.0f);  // Red
    return ImVec4(1.0f, 1.0f, 1.0f, 1.0f);  // White
}

std::string JobStatusPanel::FormatTimestamp(int64_t unix_timestamp) const {
    std::time_t time = static_cast<std::time_t>(unix_timestamp);
    std::tm* tm_info = std::localtime(&time);

    std::stringstream ss;
    ss << std::put_time(tm_info, "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

std::string JobStatusPanel::GetTimeUntilExpiration(int64_t expiration_timestamp) const {
    auto now = std::chrono::system_clock::now();
    int64_t current_timestamp = std::chrono::duration_cast<std::chrono::seconds>(
        now.time_since_epoch()).count();

    int64_t seconds_remaining = expiration_timestamp - current_timestamp;

    if (seconds_remaining <= 0) {
        return "EXPIRED";
    }

    int64_t minutes = seconds_remaining / 60;
    int64_t hours = minutes / 60;

    if (hours > 0) {
        return std::to_string(hours) + "h " + std::to_string(minutes % 60) + "m";
    } else {
        return std::to_string(minutes) + "m " + std::to_string(seconds_remaining % 60) + "s";
    }
}

void JobStatusPanel::RenderTrainingGraphs(const network::ActiveJob* job) {
    if (!job) return;

    std::lock_guard<std::mutex> lock(metrics_mutex_);

    auto it = job_metrics_.find(job->job_id);
    if (it == job_metrics_.end() || it->second.epochs.empty()) {
        ImGui::TextDisabled("Waiting for training data...");
        return;
    }

    const auto& metrics = it->second;

    if (ImGui::CollapsingHeader("Training Metrics", ImGuiTreeNodeFlags_DefaultOpen)) {
        // Loss plot
        if (ImPlot::BeginPlot("Loss", ImVec2(-1, 150))) {
            ImPlot::SetupAxes("Epoch", "Loss");
            if (!metrics.loss.empty()) {
                ImPlot::PlotLine("Loss", metrics.epochs.data(), metrics.loss.data(),
                                static_cast<int>(metrics.loss.size()));
            }
            ImPlot::EndPlot();
        }

        // Accuracy plot
        if (ImPlot::BeginPlot("Accuracy", ImVec2(-1, 150))) {
            ImPlot::SetupAxes("Epoch", "Accuracy (%)");
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 100, ImGuiCond_Once);
            if (!metrics.accuracy.empty()) {
                ImPlot::PlotLine("Accuracy", metrics.epochs.data(), metrics.accuracy.data(),
                                static_cast<int>(metrics.accuracy.size()));
            }
            ImPlot::EndPlot();
        }
    }
}

void JobStatusPanel::OnP2PProgressUpdate(const std::string& job_id,
                                          const network::TrainingProgress& progress) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);

    auto& metrics = job_metrics_[job_id];

    // Calculate epoch value (epoch + batch fraction)
    float epoch_val = static_cast<float>(progress.current_epoch);
    if (progress.total_batches > 0) {
        epoch_val += static_cast<float>(progress.current_batch) / progress.total_batches;
    }

    // Get loss and accuracy from metrics map
    float loss = 0.0f;
    float acc = 0.0f;

    auto loss_it = progress.metrics.find("loss");
    if (loss_it != progress.metrics.end()) {
        loss = loss_it->second;
    }

    auto acc_it = progress.metrics.find("accuracy");
    if (acc_it != progress.metrics.end()) {
        acc = acc_it->second * 100.0f;  // Convert to percentage
    }

    metrics.AddPoint(epoch_val, loss, acc);
}

} // namespace cyxwiz
