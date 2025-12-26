#include "p2p_training_panel.h"
#include "../../core/model_format.h"
#include "../../core/model_exporter.h"
#include "../../core/formats/cyxmodel_format.h"
#include "../../core/project_manager.h"
#include <spdlog/spdlog.h>
#include <imgui.h>
#include <implot.h>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <thread>
#include <fstream>
#include <cstring>

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
    CancelDownloadAndWait();
    StopMonitoring();
}

void P2PTrainingPanel::CancelDownloadAndWait() {
    // Signal download thread to stop
    cancel_download_ = true;

    // Wait for download thread to complete
    if (download_thread_.joinable()) {
        spdlog::debug("P2PTrainingPanel: Waiting for download thread to complete...");
        download_thread_.join();
        spdlog::debug("P2PTrainingPanel: Download thread completed");
    }

    // Reset state
    cancel_download_ = false;
    downloading_model_ = false;
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

    // Re-enable control buttons for new training
    pause_button_enabled_ = true;
    resume_button_enabled_ = false;
    stop_button_enabled_ = true;
    checkpoint_button_enabled_ = true;

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

void P2PTrainingPanel::SetTrainingConfig(const std::string& graph_json, int epochs, int batch_size, float learning_rate) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    graph_json_ = graph_json;
    training_epochs_ = epochs;
    training_batch_size_ = batch_size;
    training_learning_rate_ = learning_rate;
    spdlog::info("P2PTrainingPanel: Stored training config (graph_json size: {}, epochs: {}, batch: {}, lr: {})",
                 graph_json.size(), epochs, batch_size, learning_rate);
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

    // Reset download state
    model_weights_location_.clear();
    model_available_ = false;
    downloading_model_ = false;
    download_progress_ = 0.0f;
    download_error_.clear();
    std::memset(download_path_, 0, sizeof(download_path_));
    export_format_ = 0;

    // Reset training config
    graph_json_.clear();
    training_epochs_ = 0;
    training_batch_size_ = 32;
    training_learning_rate_ = 0.001f;
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

    // Training controls (only during active training)
    if (is_monitoring_ && !training_complete_) {
        RenderTrainingControls();
        ImGui::Separator();
    }

    // Progress bar
    if (is_monitoring_) {
        RenderProgressBar();
        ImGui::Separator();
    }

    // Model export section (shown when training is complete or stopped)
    if (training_complete_) {
        RenderModelExport();
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

    // Render stop confirmation popup (must be rendered even when not monitoring)
    RenderStopConfirmPopup();

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

    // Stop button - shows confirmation popup
    if (ImGui::Button("Stop", ImVec2(button_width, 0))) {
        if (stop_button_enabled_) {
            show_stop_confirm_popup_ = true;
        }
    }
}

void P2PTrainingPanel::RenderModelExport() {
    ImGui::Text("Model Export");

    // Check if project is open
    auto& project = ProjectManager::Instance();
    if (!project.HasActiveProject()) {
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f),
            "No project open.");
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
            "Please create or open a project to export models.");
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
            "File > New Project or File > Open Project");
        return;
    }

    // Check if model is available
    if (!model_available_) {
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f),
            "No model weights available for export.");
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
            "Training may have been stopped before any weights were saved.");
        return;
    }

    // Initialize default path if empty (project was opened after training completed)
    if (strlen(download_path_) == 0) {
        std::string models_folder = project.GetModelsPath();
        std::string ext = (export_format_ == 1) ? ".cyxmodel" : ".pt";
        std::string default_path = models_folder + "/" + job_id_ + "_model" + ext;
        strncpy(download_path_, default_path.c_str(), sizeof(download_path_) - 1);
        download_path_[sizeof(download_path_) - 1] = '\0';
    }

    // Export format selection
    ImGui::Text("Format:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(180);
    const char* formats[] = { "Raw Weights (.pt)", "CyxModel (.cyxmodel)" };
    ImGui::Combo("##export_format", &export_format_, formats, IM_ARRAYSIZE(formats));

    // Adjust default path based on format
    if (export_format_ == 1 && strlen(download_path_) > 0) {
        std::string path(download_path_);
        size_t dot_pos = path.rfind('.');
        if (dot_pos != std::string::npos && path.substr(dot_pos) != ".cyxmodel") {
            path = path.substr(0, dot_pos) + ".cyxmodel";
            strncpy(download_path_, path.c_str(), sizeof(download_path_) - 1);
        }
    } else if (export_format_ == 0 && strlen(download_path_) > 0) {
        std::string path(download_path_);
        size_t dot_pos = path.rfind('.');
        if (dot_pos != std::string::npos && path.substr(dot_pos) == ".cyxmodel") {
            path = path.substr(0, dot_pos) + ".pt";
            strncpy(download_path_, path.c_str(), sizeof(download_path_) - 1);
        }
    }

    ImGui::Text("Save to:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(350);
    ImGui::InputText("##download_path", download_path_, sizeof(download_path_));
    ImGui::SameLine();

    if (downloading_model_) {
        ImGui::ProgressBar(download_progress_, ImVec2(100, 0));
    } else {
        if (ImGui::Button("Export", ImVec2(80, 0))) {
            DownloadModel(download_path_);
        }
    }

    // Show info about .cyxmodel format
    if (export_format_ == 1) {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
            "CyxModel includes: graph, weights, training config, and history");
        if (graph_json_.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f),
                "Warning: Graph JSON not available. Export will include weights only.");
        }
    }

    if (!download_error_.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Error: %s", download_error_.c_str());
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
        // All SetupAxis calls must happen BEFORE any PlotLine calls
        ImPlot::SetupAxes("Epoch", "Loss");
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, total_epochs_ > 0 ? total_epochs_ : 10, ImGuiCond_Always);
        ImPlot::SetupAxis(ImAxis_Y2, "Accuracy", ImPlotAxisFlags_AuxDefault);

        // Plot loss on Y1
        if (!loss_history_.epochs.empty()) {
            ImPlot::PlotLine("Loss", loss_history_.epochs.data(), loss_history_.values.data(),
                           static_cast<int>(loss_history_.epochs.size()));
        }

        // Plot accuracy on Y2
        if (!accuracy_history_.epochs.empty()) {
            ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
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

void P2PTrainingPanel::RenderStopConfirmPopup() {
    // Open popup if requested
    if (show_stop_confirm_popup_) {
        ImGui::OpenPopup("Stop Training?");
        show_stop_confirm_popup_ = false;
    }

    // Center the popup
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    if (ImGui::BeginPopupModal("Stop Training?", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Are you sure you want to stop the current training?");
        ImGui::Text("This action cannot be undone.");
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Center the buttons
        float button_width = 120.0f;
        float total_width = button_width * 2 + ImGui::GetStyle().ItemSpacing.x;
        float start_x = (ImGui::GetContentRegionAvail().x - total_width) * 0.5f;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + start_x);

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.2f, 0.2f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.9f, 0.3f, 0.3f, 1.0f));
        if (ImGui::Button("Yes, Stop", ImVec2(button_width, 0))) {
            if (p2p_client_ && p2p_client_->StopTraining()) {
                AddLogEntry("WARN", "Training stopped by user");
                stop_button_enabled_ = false;
                pause_button_enabled_ = false;
                resume_button_enabled_ = false;
            } else {
                AddLogEntry("ERROR", "Failed to stop training");
            }
            ImGui::CloseCurrentPopup();
        }
        ImGui::PopStyleColor(2);

        ImGui::SameLine();

        if (ImGui::Button("Cancel", ImVec2(button_width, 0))) {
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
}

void P2PTrainingPanel::OnProgress(const network::TrainingProgress& progress) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    spdlog::debug("P2PTrainingPanel::OnProgress - epoch {}/{}, batch {}/{}, gpu={:.2f}%, mem={:.2f}%",
                  progress.current_epoch, progress.total_epochs,
                  progress.current_batch, progress.total_batches,
                  progress.gpu_usage * 100.0f, progress.memory_usage * 100.0f);

    latest_progress_ = progress;
    current_epoch_ = progress.current_epoch;
    total_epochs_ = progress.total_epochs;
    current_batch_ = progress.current_batch;
    total_batches_ = progress.total_batches;
    current_progress_ = progress.progress_percentage;

    // Update metrics history - protect against division by zero
    float epoch_value = static_cast<float>(progress.current_epoch);
    if (progress.total_batches > 0) {
        epoch_value += static_cast<float>(progress.current_batch) / progress.total_batches;
    }

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
        // Check if stopped by user (not an error)
        if (complete.final_metrics.count("stopped_by_user")) {
            AddLogEntry("WARN", "Training stopped by user");
        } else {
            AddLogEntry("ERROR", "Training failed");
        }
    }

    // Check if model weights are available for download
    if (!complete.model_uri.empty()) {
        model_weights_location_ = complete.model_uri;
        model_available_ = true;
        AddLogEntry("INFO", "Model weights available for download");

        // Set default download path to project's models folder
        if (strlen(download_path_) == 0) {
            auto& project = ProjectManager::Instance();
            if (project.HasActiveProject()) {
                std::string models_folder = project.GetModelsPath();
                std::string default_path = models_folder + "/" + job_id_ + "_model.pt";
                strncpy(download_path_, default_path.c_str(), sizeof(download_path_) - 1);
                download_path_[sizeof(download_path_) - 1] = '\0';
                AddLogEntry("INFO", "Default export path: " + default_path);
            }
            // If no project, leave empty - user will be prompted to create one in export UI
        }
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

void P2PTrainingPanel::DownloadModel(const std::string& output_path) {
    if (!p2p_client_ || !p2p_client_->IsConnected()) {
        download_error_ = "Not connected to server";
        return;
    }

    if (output_path.empty()) {
        download_error_ = "Please specify a download path";
        return;
    }

    // Wait for any previous download to complete
    if (download_thread_.joinable()) {
        spdlog::debug("Waiting for previous download to complete...");
        download_thread_.join();
    }

    download_error_.clear();
    downloading_model_ = true;
    download_progress_ = 0.0f;
    cancel_download_ = false;  // Reset cancellation flag

    // Capture export format (0 = raw weights, 1 = .cyxmodel)
    int format = export_format_;

    // Capture training config for .cyxmodel export
    std::string graph_json = graph_json_;
    int epochs = training_epochs_;
    int batch_size = training_batch_size_;
    float learning_rate = training_learning_rate_;

    // Capture training history for .cyxmodel export
    std::vector<float> loss_epochs, loss_values;
    std::vector<float> acc_epochs, acc_values;
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        loss_epochs = loss_history_.epochs;
        loss_values = loss_history_.values;
        acc_epochs = accuracy_history_.epochs;
        acc_values = accuracy_history_.values;
    }

    std::string job_id = job_id_;

    // Start download in background thread (assigned to member for tracking)
    download_thread_ = std::thread([this, output_path, format, graph_json, epochs, batch_size, learning_rate,
                 loss_epochs, loss_values, acc_epochs, acc_values, job_id]() {
        try {
            // Check for cancellation before starting
            if (cancel_download_) {
                std::lock_guard<std::mutex> lock(data_mutex_);
                downloading_model_ = false;
                download_error_ = "Download cancelled";
                return;
            }

            // Create directory if needed
            std::filesystem::path path(output_path);
            if (path.has_parent_path()) {
                std::filesystem::create_directories(path.parent_path());
            }

            if (format == 0) {
                // Raw weights download
                spdlog::info("Downloading raw weights to: {}", output_path);

                // Check for cancellation and connection before download
                if (cancel_download_ || !p2p_client_ || !p2p_client_->IsConnected()) {
                    std::lock_guard<std::mutex> lock(data_mutex_);
                    downloading_model_ = false;
                    download_error_ = cancel_download_ ? "Download cancelled" : "Not connected";
                    return;
                }

                bool success = p2p_client_->DownloadWeights(job_id, output_path);

                {
                    std::lock_guard<std::mutex> lock(data_mutex_);
                    downloading_model_ = false;
                    if (cancel_download_) {
                        download_error_ = "Download cancelled";
                    } else if (success) {
                        download_progress_ = 1.0f;
                        AddLogEntry("INFO", "Model weights downloaded to: " + output_path);
                    } else {
                        download_error_ = p2p_client_ ? p2p_client_->GetLastError() : "Client disconnected";
                        AddLogEntry("ERROR", "Download failed: " + download_error_);
                    }
                }
            } else {
                // .cyxmodel export - download weights first to temp, then package
                spdlog::info("Exporting as CyxModel to: {}", output_path);

                // Check for cancellation and connection before download
                if (cancel_download_ || !p2p_client_ || !p2p_client_->IsConnected()) {
                    std::lock_guard<std::mutex> lock(data_mutex_);
                    downloading_model_ = false;
                    download_error_ = cancel_download_ ? "Download cancelled" : "Not connected";
                    return;
                }

                // Download weights to temp file
                std::filesystem::path temp_weights = std::filesystem::temp_directory_path() /
                    ("cyxwiz_temp_" + job_id + ".weights");

                bool success = p2p_client_->DownloadWeights(job_id, temp_weights.string());

                // Check for cancellation after download
                if (cancel_download_) {
                    std::filesystem::remove(temp_weights);  // Cleanup temp file
                    std::lock_guard<std::mutex> lock(data_mutex_);
                    downloading_model_ = false;
                    download_error_ = "Download cancelled";
                    return;
                }

                if (!success) {
                    std::lock_guard<std::mutex> lock(data_mutex_);
                    downloading_model_ = false;
                    download_error_ = p2p_client_ ? p2p_client_->GetLastError() : "Client disconnected";
                    AddLogEntry("ERROR", "Failed to download weights: " + download_error_);
                    return;
                }

                {
                    std::lock_guard<std::mutex> lock(data_mutex_);
                    download_progress_ = 0.5f;
                }

                // Check for cancellation before packaging
                if (cancel_download_) {
                    std::filesystem::remove(temp_weights);  // Cleanup temp file
                    std::lock_guard<std::mutex> lock(data_mutex_);
                    downloading_model_ = false;
                    download_error_ = "Download cancelled";
                    return;
                }

                // Package as .cyxmodel
                spdlog::info("Packaging weights into .cyxmodel...");

                // Read weights from temp file
                std::ifstream weights_file(temp_weights, std::ios::binary);
                if (!weights_file) {
                    std::lock_guard<std::mutex> lock(data_mutex_);
                    downloading_model_ = false;
                    download_error_ = "Failed to read downloaded weights";
                    AddLogEntry("ERROR", download_error_);
                    return;
                }
                std::vector<uint8_t> weights_data((std::istreambuf_iterator<char>(weights_file)),
                                                   std::istreambuf_iterator<char>());
                weights_file.close();

                // Create model manifest
                ModelManifest manifest;
                manifest.version = "1.0";
                manifest.format = "cyxmodel";
                manifest.model_name = job_id;
                manifest.description = "P2P trained model";
                manifest.cyxwiz_version = "1.0.0";
                manifest.has_graph = !graph_json.empty();
                manifest.has_training_history = true;
                manifest.has_optimizer_state = false;
                manifest.num_parameters = static_cast<int>(weights_data.size() / sizeof(float));  // Rough estimate

                // Create training config
                TrainingConfig config;
                config.epochs = epochs;
                config.batch_size = batch_size;
                config.learning_rate = learning_rate;
                config.optimizer_type = "AdamW";

                // Create training history from captured metrics
                TrainingHistory history;
                for (size_t i = 0; i < loss_values.size(); ++i) {
                    history.loss_history.push_back(loss_values[i]);
                }
                for (size_t i = 0; i < acc_values.size(); ++i) {
                    history.accuracy_history.push_back(acc_values[i]);
                }

                // Create weights map (single blob for now - can be refined)
                std::map<std::string, std::vector<uint8_t>> weights_map;
                std::map<std::string, std::vector<int64_t>> weight_shapes;
                weights_map["model_weights"] = weights_data;
                weight_shapes["model_weights"] = {static_cast<int64_t>(weights_data.size())};

                // Export options
                ExportOptions options;
                options.format = ModelFormat::CyxModel;
                options.include_graph = !graph_json.empty();
                options.include_optimizer_state = false;
                options.compress = true;

                // Create .cyxmodel file
                formats::CyxModelFormat exporter;
                bool export_success = exporter.Create(
                    output_path,
                    manifest,
                    graph_json,
                    config,
                    &history,
                    weights_map,
                    weight_shapes,
                    nullptr,  // No optimizer state
                    options
                );

                // Cleanup temp file
                std::filesystem::remove(temp_weights);

                {
                    std::lock_guard<std::mutex> lock(data_mutex_);
                    downloading_model_ = false;
                    if (export_success) {
                        download_progress_ = 1.0f;
                        AddLogEntry("INFO", "Model exported to: " + output_path);
                    } else {
                        download_error_ = exporter.GetLastError();
                        AddLogEntry("ERROR", "Export failed: " + download_error_);
                    }
                }
            }
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(data_mutex_);
            downloading_model_ = false;
            download_error_ = e.what();
            AddLogEntry("ERROR", "Export exception: " + std::string(e.what()));
        }
    });  // Thread is tracked via download_thread_ member, not detached
}

} // namespace cyxwiz
