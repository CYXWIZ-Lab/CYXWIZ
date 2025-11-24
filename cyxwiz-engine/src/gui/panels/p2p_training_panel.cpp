#include "p2p_training_panel.h"
#include <spdlog/spdlog.h>
#include <imgui.h>
#include <implot.h>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace cyxwiz {

P2PTrainingPanel::P2PTrainingPanel()
    : Panel("P2P Training Progress")
    , p2p_client_(nullptr)
    , job_id_("")
    , node_address_("")
    , is_monitoring_(false)
    , training_complete_(false)
    , has_error_(false)
    , error_message_("")
    , current_progress_(0.0f)
    , current_epoch_(0)
    , total_epochs_(0)
    , current_batch_(0)
    , total_batches_(0)
    , estimated_time_remaining_(0)
{
}

P2PTrainingPanel::~P2PTrainingPanel() {
    StopMonitoring();
}

void P2PTrainingPanel::SetP2PClient(std::shared_ptr<network::P2PClient> client) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    p2p_client_ = client;

    if (p2p_client_) {
        // Register callbacks
        p2p_client_->SetProgressCallback([this](const network::TrainingProgress& prog) {
            OnProgress(prog);
        });

        p2p_client_->SetCheckpointCallback([this](const network::CheckpointInfo& ckpt) {
            OnCheckpoint(ckpt);
        });

        p2p_client_->SetCompletionCallback([this](const network::TrainingComplete& comp) {
            OnComplete(comp);
        });

        p2p_client_->SetErrorCallback([this](const std::string& err, bool is_fatal) {
            OnError(err, is_fatal);
        });

        p2p_client_->SetLogCallback([this](const std::string& source, const std::string& message) {
            OnLog(source, message);
        });
    }
}

void P2PTrainingPanel::StartMonitoring(const std::string& job_id, const std::string& node_address) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    job_id_ = job_id;
    node_address_ = node_address;
    is_monitoring_ = true;
    training_complete_ = false;
    has_error_ = false;
    error_message_.clear();
    training_start_time_ = std::chrono::steady_clock::now();

    // Clear previous data
    loss_history_.Clear();
    accuracy_history_.Clear();
    gpu_usage_history_.Clear();
    memory_usage_history_.Clear();
    checkpoint_history_.clear();
    log_entries_.clear();

    AddLogEntry("INFO", "Started monitoring job " + job_id);
}

void P2PTrainingPanel::StopMonitoring() {
    std::lock_guard<std::mutex> lock(data_mutex_);

    if (!is_monitoring_) {
        return;
    }

    is_monitoring_ = false;
    AddLogEntry("INFO", "Stopped monitoring");
}

void P2PTrainingPanel::Clear() {
    std::lock_guard<std::mutex> lock(data_mutex_);

    job_id_.clear();
    node_address_.clear();
    is_monitoring_ = false;
    training_complete_ = false;
    has_error_ = false;
    error_message_.clear();
    current_progress_ = 0.0f;
    current_epoch_ = 0;
    total_epochs_ = 0;

    loss_history_.Clear();
    accuracy_history_.Clear();
    gpu_usage_history_.Clear();
    memory_usage_history_.Clear();
    checkpoint_history_.clear();
    log_entries_.clear();
}

void P2PTrainingPanel::Render() {
    if (!visible_) {
        return;
    }

    if (!ImGui::Begin(name_.c_str(), &visible_)) {
        ImGui::End();
        return;
    }

    // Connection status
    RenderConnectionStatus();

    ImGui::Separator();

    // Training controls
    if (is_monitoring_ && !training_complete_) {
        RenderTrainingControls();
        ImGui::Separator();
    }

    // Progress bar
    if (is_monitoring_) {
        RenderProgressBar();
        ImGui::Separator();
    }

    // Main content area - metrics plots
    if (is_monitoring_ && ImGui::BeginTabBar("MetricsTabs")) {
        if (ImGui::BeginTabItem("Metrics")) {
            RenderMetricsPlots();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Logs")) {
            RenderLogs();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Checkpoints")) {
            RenderCheckpoints();
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }

    ImGui::End();
}

void P2PTrainingPanel::RenderConnectionStatus() {
    ImGui::Text("Job ID: %s", job_id_.empty() ? "N/A" : job_id_.c_str());
    ImGui::SameLine();

    if (is_monitoring_) {
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "[ACTIVE]");
    } else {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "[IDLE]");
    }

    ImGui::Text("Node: %s", node_address_.empty() ? "N/A" : node_address_.c_str());

    if (p2p_client_ && p2p_client_->IsConnected()) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Connected");
    }

    if (has_error_) {
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Error: %s", error_message_.c_str());
    }
}

void P2PTrainingPanel::RenderTrainingControls() {
    float button_width = 80.0f;

    // Pause button
    if (!pause_button_enabled_) {
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
    }
    if (ImGui::Button("Pause", ImVec2(button_width, 0))) {
        if (p2p_client_ && pause_button_enabled_) {
            if (p2p_client_->PauseTraining()) {
                AddLogEntry("INFO", "Pause command sent");
                pause_button_enabled_ = false;
                resume_button_enabled_ = true;
            } else {
                AddLogEntry("ERROR", "Failed to pause training");
            }
        }
    }
    if (!pause_button_enabled_) {
        ImGui::PopStyleVar();
    }

    ImGui::SameLine();

    // Resume button
    if (!resume_button_enabled_) {
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
    }
    if (ImGui::Button("Resume", ImVec2(button_width, 0))) {
        if (p2p_client_ && resume_button_enabled_) {
            if (p2p_client_->ResumeTraining()) {
                AddLogEntry("INFO", "Resume command sent");
                pause_button_enabled_ = true;
                resume_button_enabled_ = false;
            } else {
                AddLogEntry("ERROR", "Failed to resume training");
            }
        }
    }
    if (!resume_button_enabled_) {
        ImGui::PopStyleVar();
    }

    ImGui::SameLine();

    // Checkpoint button
    if (!checkpoint_button_enabled_ || !node_capabilities_.supports_checkpointing) {
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
    }
    if (ImGui::Button("Checkpoint", ImVec2(button_width, 0))) {
        if (p2p_client_ && checkpoint_button_enabled_) {
            if (p2p_client_->RequestCheckpoint()) {
                AddLogEntry("INFO", "Checkpoint requested");
            } else {
                AddLogEntry("ERROR", "Failed to request checkpoint");
            }
        }
    }
    if (!checkpoint_button_enabled_ || !node_capabilities_.supports_checkpointing) {
        ImGui::PopStyleVar();
    }

    ImGui::SameLine();

    // Stop button
    if (ImGui::Button("Stop", ImVec2(button_width, 0))) {
        if (p2p_client_ && stop_button_enabled_) {
            if (p2p_client_->StopTraining()) {
                AddLogEntry("WARN", "Stop command sent");
                stop_button_enabled_ = false;
            } else {
                AddLogEntry("ERROR", "Failed to stop training");
            }
        }
    }
}

void P2PTrainingPanel::RenderProgressBar() {
    std::lock_guard<std::mutex> lock(data_mutex_);

    char progress_text[128];
    snprintf(progress_text, sizeof(progress_text),
             "Epoch %u/%u | Batch %u/%u (%.1f%%)",
             current_epoch_, total_epochs_,
             current_batch_, total_batches_,
             current_progress_ * 100.0f);

    ImGui::ProgressBar(current_progress_, ImVec2(-1, 0), progress_text);

    // ETA
    if (estimated_time_remaining_ > 0) {
        ImGui::Text("Estimated Time Remaining: %s", FormatDuration(estimated_time_remaining_).c_str());
    }

    // Current metrics
    if (!latest_progress_.metrics.empty()) {
        ImGui::Text("Loss: %.4f | Accuracy: %.2f%% | GPU: %.1f%% | Memory: %.1f%%",
                    latest_progress_.metrics.count("loss") ? latest_progress_.metrics.at("loss") : 0.0f,
                    latest_progress_.metrics.count("accuracy") ? latest_progress_.metrics.at("accuracy") * 100.0f : 0.0f,
                    latest_progress_.gpu_usage * 100.0f,
                    latest_progress_.memory_usage * 100.0f);
    }
}

void P2PTrainingPanel::RenderMetricsPlots() {
    std::lock_guard<std::mutex> lock(data_mutex_);

    if (ImPlot::BeginPlot("Training Metrics", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("Epoch", "Value");
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, total_epochs_ > 0 ? total_epochs_ : 10, ImGuiCond_Always);

        // Plot loss
        if (!loss_history_.epochs.empty()) {
            ImPlot::PlotLine("Loss", loss_history_.epochs.data(), loss_history_.values.data(),
                           static_cast<int>(loss_history_.epochs.size()));
        }

        // Plot accuracy
        if (!accuracy_history_.epochs.empty()) {
            ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
            ImPlot::SetupAxis(ImAxis_Y2, "Accuracy", ImPlotAxisFlags_AuxDefault);
            ImPlot::PlotLine("Accuracy", accuracy_history_.epochs.data(), accuracy_history_.values.data(),
                           static_cast<int>(accuracy_history_.epochs.size()));
        }

        ImPlot::EndPlot();
    }

    // GPU and Memory usage plots
    if (ImPlot::BeginPlot("Resource Usage", ImVec2(-1, 200))) {
        ImPlot::SetupAxes("Epoch", "Usage (%)");
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 100, ImGuiCond_Always);

        if (!gpu_usage_history_.epochs.empty()) {
            ImPlot::PlotLine("GPU", gpu_usage_history_.epochs.data(), gpu_usage_history_.values.data(),
                           static_cast<int>(gpu_usage_history_.epochs.size()));
        }

        if (!memory_usage_history_.epochs.empty()) {
            ImPlot::PlotLine("Memory", memory_usage_history_.epochs.data(), memory_usage_history_.values.data(),
                           static_cast<int>(memory_usage_history_.epochs.size()));
        }

        ImPlot::EndPlot();
    }
}

void P2PTrainingPanel::RenderLogs() {
    std::lock_guard<std::mutex> lock(data_mutex_);

    ImGui::Checkbox("Auto-scroll", &auto_scroll_logs_);
    ImGui::SameLine();
    if (ImGui::Button("Clear Logs")) {
        log_entries_.clear();
    }

    ImGui::Separator();

    if (ImGui::BeginChild("LogsScrollArea", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {
        for (const auto& entry : log_entries_) {
            ImVec4 color = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);  // White for INFO
            if (entry.level == "WARN") {
                color = ImVec4(1.0f, 1.0f, 0.0f, 1.0f);  // Yellow for warnings
            } else if (entry.level == "ERROR") {
                color = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);  // Red for errors
            }

            ImGui::TextColored(color, "[%s] [%s] [%s] %s",
                             entry.timestamp.c_str(),
                             entry.level.c_str(),
                             entry.source.c_str(),
                             entry.message.c_str());
        }

        if (auto_scroll_logs_ && ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) {
            ImGui::SetScrollHereY(1.0f);
        }
    }
    ImGui::EndChild();
}

void P2PTrainingPanel::RenderCheckpoints() {
    std::lock_guard<std::mutex> lock(data_mutex_);

    if (checkpoint_history_.empty()) {
        ImGui::TextDisabled("No checkpoints yet");
        return;
    }

    if (ImGui::BeginTable("CheckpointsTable", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Epoch");
        ImGui::TableSetupColumn("Timestamp");
        ImGui::TableSetupColumn("Hash");
        ImGui::TableSetupColumn("Size");
        ImGui::TableSetupColumn("Storage URI");
        ImGui::TableHeadersRow();

        for (const auto& ckpt : checkpoint_history_) {
            ImGui::TableNextRow();

            ImGui::TableSetColumnIndex(0);
            ImGui::Text("%u", ckpt.epoch);

            ImGui::TableSetColumnIndex(1);
            auto time_t_val = std::chrono::system_clock::to_time_t(ckpt.timestamp);
            ImGui::Text("%s", std::ctime(&time_t_val));

            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%s", ckpt.checkpoint_hash.substr(0, 8).c_str());

            ImGui::TableSetColumnIndex(3);
            ImGui::Text("%.2f MB", ckpt.size_bytes / (1024.0 * 1024.0));

            ImGui::TableSetColumnIndex(4);
            ImGui::Text("%s", ckpt.storage_uri.c_str());
        }

        ImGui::EndTable();
    }
}

void P2PTrainingPanel::OnProgress(const network::TrainingProgress& progress) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    latest_progress_ = progress;
    current_epoch_ = progress.current_epoch;
    total_epochs_ = progress.total_epochs;
    current_batch_ = progress.current_batch;
    total_batches_ = progress.total_batches;
    current_progress_ = progress.progress_percentage;

    // Update metrics history
    float epoch_value = static_cast<float>(progress.current_epoch) +
                        (static_cast<float>(progress.current_batch) / progress.total_batches);

    if (progress.metrics.count("loss")) {
        loss_history_.AddPoint(epoch_value, progress.metrics.at("loss"));
    }

    if (progress.metrics.count("accuracy")) {
        accuracy_history_.AddPoint(epoch_value, progress.metrics.at("accuracy"));
    }

    gpu_usage_history_.AddPoint(epoch_value, progress.gpu_usage * 100.0f);
    memory_usage_history_.AddPoint(epoch_value, progress.memory_usage * 100.0f);

    // Update ETA
    UpdateETA();

    // Enable pause button if not already paused
    if (!pause_button_enabled_ && !resume_button_enabled_) {
        pause_button_enabled_ = true;
    }
}

void P2PTrainingPanel::OnCheckpoint(const network::CheckpointInfo& checkpoint) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    CheckpointEntry entry;
    entry.epoch = checkpoint.epoch;
    entry.checkpoint_hash = checkpoint.checkpoint_hash;
    entry.storage_uri = checkpoint.storage_uri;
    entry.size_bytes = checkpoint.size_bytes;
    entry.timestamp = std::chrono::system_clock::now();

    checkpoint_history_.push_back(entry);

    AddLogEntry("INFO", "Checkpoint saved at epoch " + std::to_string(checkpoint.epoch));
}

void P2PTrainingPanel::OnComplete(const network::TrainingComplete& complete) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    training_complete_ = true;
    current_progress_ = 1.0f;
    estimated_time_remaining_ = 0;

    if (complete.success) {
        AddLogEntry("INFO", "Training completed successfully!");
        AddLogEntry("INFO", "Total training time: " + FormatDuration(complete.total_training_time));

        if (complete.final_metrics.count("loss")) {
            AddLogEntry("INFO", "Final loss: " + std::to_string(complete.final_metrics.at("loss")));
        }
        if (complete.final_metrics.count("accuracy")) {
            AddLogEntry("INFO", "Final accuracy: " + std::to_string(complete.final_metrics.at("accuracy") * 100.0) + "%");
        }
    } else {
        AddLogEntry("ERROR", "Training failed");
    }

    // Disable control buttons
    pause_button_enabled_ = false;
    resume_button_enabled_ = false;
    stop_button_enabled_ = false;
    checkpoint_button_enabled_ = false;
}

void P2PTrainingPanel::OnError(const std::string& error_message, bool is_fatal) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    has_error_ = is_fatal;
    error_message_ = error_message;

    AddLogEntry("ERROR", error_message);

    if (is_fatal) {
        is_monitoring_ = false;
        // Disable control buttons
        pause_button_enabled_ = false;
        resume_button_enabled_ = false;
        stop_button_enabled_ = false;
        checkpoint_button_enabled_ = false;
    }
}

void P2PTrainingPanel::OnLog(const std::string& source, const std::string& message) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    AddLogEntry("INFO", "[" + source + "] " + message);
}

void P2PTrainingPanel::UpdateETA() {
    if (current_progress_ <= 0.0f) {
        estimated_time_remaining_ = 0;
        return;
    }

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - training_start_time_).count();

    // Estimate total time based on current progress
    uint64_t total_estimated = static_cast<uint64_t>(elapsed / current_progress_);
    estimated_time_remaining_ = total_estimated - elapsed;
}

void P2PTrainingPanel::AddLogEntry(const std::string& level, const std::string& message) {
    // Get current timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t_val = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t_val), "%H:%M:%S");

    LogEntry entry;
    entry.timestamp = ss.str();
    entry.level = level;
    entry.source = "P2P";
    entry.message = message;

    log_entries_.push_back(entry);

    // Trim old logs
    while (log_entries_.size() > max_log_entries_) {
        log_entries_.pop_front();
    }

    spdlog::info("[P2PTrainingPanel] [{}] {}", level, message);
}

std::string P2PTrainingPanel::FormatDuration(uint64_t seconds) {
    uint64_t hours = seconds / 3600;
    uint64_t minutes = (seconds % 3600) / 60;
    uint64_t secs = seconds % 60;

    std::stringstream ss;
    if (hours > 0) {
        ss << hours << "h " << minutes << "m " << secs << "s";
    } else if (minutes > 0) {
        ss << minutes << "m " << secs << "s";
    } else {
        ss << secs << "s";
    }

    return ss.str();
}

} // namespace cyxwiz
