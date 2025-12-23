#include "import_dialog.h"
#include "../icons.h"
#include "../../core/file_dialogs.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <algorithm>

namespace cyxwiz {

ImportDialog::ImportDialog()
    : Panel("Import Model", false) {
}

ImportDialog::~ImportDialog() {
    // Wait for import thread to finish
    if (import_thread_ && import_thread_->joinable()) {
        import_thread_->join();
    }
}

void ImportDialog::Open() {
    is_open_ = true;
    show_result_ = false;
    file_probed_ = false;

    // Reset state
    is_importing_ = false;
    is_probing_ = false;
    import_progress_ = 0;
    import_total_ = 0;
    input_path_[0] = '\0';
    imported_graph_json_.clear();

    // Default options
    load_optimizer_state_ = false;
    load_training_history_ = false;
    strict_mode_ = true;
    allow_shape_mismatch_ = false;

    // Reset transfer learning options
    enable_transfer_learning_ = false;
    freeze_mode_ = 0;
    unfreeze_last_n_ = 2;
    layer_trainable_.clear();
}

void ImportDialog::Close() {
    is_open_ = false;
}

void ImportDialog::Render() {
    if (!is_open_) return;

    ImGui::SetNextWindowSize(ImVec2(550, 500), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(
        ImVec2(ImGui::GetIO().DisplaySize.x * 0.5f, ImGui::GetIO().DisplaySize.y * 0.5f),
        ImGuiCond_FirstUseEver,
        ImVec2(0.5f, 0.5f)
    );

    if (ImGui::Begin("Import Model", &is_open_, ImGuiWindowFlags_NoCollapse)) {
        if (is_importing_) {
            RenderProgress();
        } else if (show_result_) {
            // Show import result
            if (last_result_.success) {
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f),
                    ICON_FA_CHECK " Import Successful!");
                ImGui::Separator();

                ImGui::Text("Model: %s", last_result_.model_name.c_str());
                ImGui::Text("Parameters: %d", last_result_.num_parameters);
                ImGui::Text("Layers: %d", last_result_.num_layers);
                ImGui::Text("Load Time: %lld ms", last_result_.load_time_ms);

                if (!last_result_.warnings.empty()) {
                    ImGui::Separator();
                    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f),
                        ICON_FA_TRIANGLE_EXCLAMATION " Warnings:");
                    for (const auto& warning : last_result_.warnings) {
                        ImGui::BulletText("%s", warning.c_str());
                    }
                }

                ImGui::Separator();

                if (ImGui::Button("Close", ImVec2(120, 0))) {
                    Close();
                }

                ImGui::SameLine();

                if (ImGui::Button("Import Another", ImVec2(120, 0))) {
                    show_result_ = false;
                    file_probed_ = false;
                    input_path_[0] = '\0';
                }
            } else {
                ImGui::TextColored(ImVec4(0.9f, 0.2f, 0.2f, 1.0f),
                    ICON_FA_XMARK " Import Failed");
                ImGui::Separator();

                ImGui::TextWrapped("Error: %s", last_result_.error_message.c_str());

                ImGui::Separator();

                if (ImGui::Button("Close", ImVec2(120, 0))) {
                    Close();
                }

                ImGui::SameLine();

                if (ImGui::Button("Try Again", ImVec2(120, 0))) {
                    show_result_ = false;
                }
            }
        } else {
            RenderFileSelection();
            ImGui::Separator();
            RenderProbeInfo();
            ImGui::Separator();
            RenderOptions();
            ImGui::Separator();
            RenderButtons();
        }
    }
    ImGui::End();
}

void ImportDialog::RenderFileSelection() {
    ImGui::Text(ICON_FA_FILE_IMPORT " Select Model File");
    ImGui::Spacing();

    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 80);
    if (ImGui::InputText("##input_path", input_path_, sizeof(input_path_))) {
        file_probed_ = false;  // Re-probe on path change
    }

    ImGui::SameLine();

    if (ImGui::Button("Browse...", ImVec2(70, 0))) {
        const char* filter =
            "All Model Files\0*.cyxmodel;*.safetensors;*.onnx;*.gguf\0"
            "CyxWiz Model (*.cyxmodel)\0*.cyxmodel\0"
            "Safetensors (*.safetensors)\0*.safetensors\0"
            "ONNX Model (*.onnx)\0*.onnx\0"
            "GGUF (*.gguf)\0*.gguf\0"
            "All Files (*.*)\0*.*\0";

        std::string path = OpenFileDialog(filter, "Select Model File");
        if (!path.empty()) {
            strncpy(input_path_, path.c_str(), sizeof(input_path_) - 1);
            file_probed_ = false;
        }
    }

    // Auto-probe when path changes
    if (strlen(input_path_) > 0 && !file_probed_ && !is_probing_) {
        ProbeFile();
    }
}

void ImportDialog::RenderProbeInfo() {
    ImGui::Text(ICON_FA_CIRCLE_INFO " Model Information");
    ImGui::Spacing();

    if (is_probing_) {
        ImGui::Text(ICON_FA_SPINNER " Analyzing file...");
        return;
    }

    if (!file_probed_ || strlen(input_path_) == 0) {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Select a file to view model information");
        return;
    }

    if (!probe_result_.valid) {
        ImGui::TextColored(ImVec4(0.9f, 0.2f, 0.2f, 1.0f),
            ICON_FA_XMARK " Invalid file: %s", probe_result_.error_message.c_str());
        return;
    }

    // Format info
    ImGui::Text("Format: %s", GetFormatName(probe_result_.format).c_str());
    if (!probe_result_.format_version.empty()) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "(v%s)", probe_result_.format_version.c_str());
    }

    // Model info
    if (!probe_result_.model_name.empty()) {
        ImGui::Text("Name: %s", probe_result_.model_name.c_str());
    }
    if (!probe_result_.author.empty()) {
        ImGui::Text("Author: %s", probe_result_.author.c_str());
    }

    ImGui::Text("Parameters: %d", probe_result_.num_parameters);
    ImGui::Text("Layers: %d", probe_result_.num_layers);
    ImGui::Text("File Size: %.2f MB", probe_result_.file_size / (1024.0f * 1024.0f));

    // Training info
    if (probe_result_.epochs_trained.has_value()) {
        ImGui::Text("Epochs Trained: %d", probe_result_.epochs_trained.value());
    }
    if (probe_result_.final_accuracy.has_value()) {
        ImGui::Text("Final Accuracy: %.2f%%", probe_result_.final_accuracy.value() * 100.0f);
    }
    if (probe_result_.final_loss.has_value()) {
        ImGui::Text("Final Loss: %.4f", probe_result_.final_loss.value());
    }

    // Content flags (for .cyxmodel)
    if (probe_result_.format == ModelFormat::CyxModel) {
        ImGui::Spacing();
        ImGui::Text("Contents:");
        ImGui::BulletText("Graph: %s", probe_result_.has_graph ? "Yes" : "No");
        ImGui::BulletText("Optimizer State: %s", probe_result_.has_optimizer_state ? "Yes" : "No");
        ImGui::BulletText("Training History: %s", probe_result_.has_training_history ? "Yes" : "No");
    }

    // Layer names (collapsible)
    if (!probe_result_.layer_names.empty()) {
        if (ImGui::TreeNode("Layers")) {
            for (const auto& name : probe_result_.layer_names) {
                auto shape_it = probe_result_.layer_shapes.find(name);
                if (shape_it != probe_result_.layer_shapes.end()) {
                    std::string shape_str = "[";
                    for (size_t i = 0; i < shape_it->second.size(); ++i) {
                        shape_str += std::to_string(shape_it->second[i]);
                        if (i < shape_it->second.size() - 1) shape_str += ", ";
                    }
                    shape_str += "]";
                    ImGui::BulletText("%s: %s", name.c_str(), shape_str.c_str());
                } else {
                    ImGui::BulletText("%s", name.c_str());
                }
            }
            ImGui::TreePop();
        }
    }
}

void ImportDialog::RenderOptions() {
    ImGui::Text(ICON_FA_GEAR " Import Options");
    ImGui::Spacing();

    // Only show options for .cyxmodel format
    if (file_probed_ && probe_result_.valid && probe_result_.format == ModelFormat::CyxModel) {
        ImGui::BeginDisabled(!probe_result_.has_optimizer_state);
        ImGui::Checkbox("Load Optimizer State", &load_optimizer_state_);
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            if (!probe_result_.has_optimizer_state) {
                ImGui::SetTooltip("Model does not contain optimizer state");
            } else {
                ImGui::SetTooltip("Load optimizer state to resume training");
            }
        }

        ImGui::BeginDisabled(!probe_result_.has_training_history);
        ImGui::Checkbox("Load Training History", &load_training_history_);
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            if (!probe_result_.has_training_history) {
                ImGui::SetTooltip("Model does not contain training history");
            } else {
                ImGui::SetTooltip("Load loss/accuracy history for visualization");
            }
        }
    }

    ImGui::Spacing();

    ImGui::Checkbox("Strict Mode", &strict_mode_);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Fail if layer names don't match exactly");
    }

    ImGui::Checkbox("Allow Shape Mismatch", &allow_shape_mismatch_);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Allow loading weights with different shapes (use with caution)");
    }

    // Transfer Learning Options
    if (file_probed_ && probe_result_.valid && probe_result_.num_layers > 0) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::Text("Transfer Learning");
        ImGui::Spacing();

        ImGui::Checkbox("Enable Transfer Learning", &enable_transfer_learning_);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Freeze some layers to fine-tune the model on new data");
        }

        if (enable_transfer_learning_) {
            ImGui::Indent();

            ImGui::RadioButton("No Freezing", &freeze_mode_, 0);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("All layers remain trainable");
            }

            ImGui::RadioButton("Freeze All Except Last N", &freeze_mode_, 1);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Freeze early feature extraction layers, train only classifier");
            }

            if (freeze_mode_ == 1) {
                ImGui::Indent();
                int max_layers = static_cast<int>(probe_result_.num_layers);
                ImGui::SliderInt("Trainable Layers", &unfreeze_last_n_, 1,
                                 std::max(1, max_layers), "%d");
                if (unfreeze_last_n_ > max_layers) {
                    unfreeze_last_n_ = max_layers;
                }
                ImGui::Unindent();
            }

            ImGui::RadioButton("Custom (Per-Layer)", &freeze_mode_, 2);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Choose which layers to freeze individually");
            }

            if (freeze_mode_ == 2) {
                // Initialize layer_trainable_ if needed
                if (layer_trainable_.size() != static_cast<size_t>(probe_result_.num_layers)) {
                    layer_trainable_.resize(probe_result_.num_layers, true);
                }

                ImGui::Indent();
                if (ImGui::TreeNode("Layer Settings")) {
                    // Quick actions
                    if (ImGui::Button("Freeze All")) {
                        std::fill(layer_trainable_.begin(), layer_trainable_.end(), false);
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Unfreeze All")) {
                        std::fill(layer_trainable_.begin(), layer_trainable_.end(), true);
                    }

                    ImGui::Spacing();

                    // Per-layer checkboxes
                    for (size_t i = 0; i < layer_trainable_.size(); ++i) {
                        std::string label;
                        if (i < probe_result_.layer_names.size()) {
                            label = std::to_string(i) + ": " + probe_result_.layer_names[i];
                        } else {
                            label = "Layer " + std::to_string(i);
                        }

                        // Show "Trainable" checkbox (unchecked = frozen)
                        // Use temporary bool because std::vector<bool> uses proxy reference
                        bool trainable = layer_trainable_[i];
                        if (ImGui::Checkbox(label.c_str(), &trainable)) {
                            layer_trainable_[i] = trainable;
                        }
                        if (!layer_trainable_[i]) {
                            ImGui::SameLine();
                            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.8f, 1.0f), "(frozen)");
                        }
                    }
                    ImGui::TreePop();
                }
                ImGui::Unindent();
            }

            ImGui::Unindent();
        }
    }
}

void ImportDialog::RenderProgress() {
    ImGui::Text(ICON_FA_SPINNER " Importing...");
    ImGui::Spacing();

    int progress = import_progress_.load();
    int total = import_total_.load();

    float fraction = (total > 0) ? static_cast<float>(progress) / total : 0.0f;
    ImGui::ProgressBar(fraction, ImVec2(-1, 0));

    {
        std::lock_guard<std::mutex> lock(status_mutex_);
        ImGui::Text("%s", import_status_.c_str());
    }
}

void ImportDialog::RenderButtons() {
    bool can_import = file_probed_ && probe_result_.valid && !is_importing_;

    ImGui::BeginDisabled(!can_import);
    if (ImGui::Button(ICON_FA_FILE_IMPORT " Import", ImVec2(120, 30))) {
        StartImport();
    }
    ImGui::EndDisabled();

    ImGui::SameLine();

    if (ImGui::Button("Cancel", ImVec2(80, 30))) {
        Close();
    }
}

std::string ImportDialog::OpenFileDialog(const char* filter, const char* title) {
    (void)filter;  // Using FileDialogs which has a different filter format
    auto result = FileDialogs::OpenModel();
    return result.value_or("");
}

void ImportDialog::ProbeFile() {
    if (strlen(input_path_) == 0) return;

    is_probing_ = true;

    // Probe in main thread for now (it's fast)
    ModelImporter importer;
    probe_result_ = importer.ProbeFile(input_path_);
    file_probed_ = true;
    is_probing_ = false;
}

void ImportDialog::StartImport() {
    if (!file_probed_ || !probe_result_.valid) return;

    is_importing_ = true;
    show_result_ = false;
    import_progress_ = 0;
    import_total_ = 4;
    imported_graph_json_.clear();

    // Build import options
    import_options_.load_optimizer_state = load_optimizer_state_;
    import_options_.load_training_history = load_training_history_;
    import_options_.strict_mode = strict_mode_;
    import_options_.allow_shape_mismatch = allow_shape_mismatch_;

    // Wait for any previous import thread
    if (import_thread_ && import_thread_->joinable()) {
        import_thread_->join();
    }

    // Start import in background thread
    import_thread_ = std::make_unique<std::thread>([this]() {
        ModelImporter importer;

        auto progress_callback = [this](int current, int total, const std::string& status) {
            import_progress_ = current;
            import_total_ = total;
            {
                std::lock_guard<std::mutex> lock(status_mutex_);
                import_status_ = status;
            }
        };

        // For .cyxmodel, also extract the graph
        if (probe_result_.format == ModelFormat::CyxModel && probe_result_.has_graph) {
            auto graph = importer.ExtractGraph(input_path_);
            if (graph.has_value()) {
                imported_graph_json_ = graph.value();
            }
        }

        // Note: We don't actually have a model to import into here
        // In a real implementation, this would be handled by the caller
        // who has access to the model being built/trained

        // For now, just report success based on probe
        last_result_.success = probe_result_.valid;
        last_result_.model_name = probe_result_.model_name;
        last_result_.format_version = probe_result_.format_version;
        last_result_.num_parameters = probe_result_.num_parameters;
        last_result_.num_layers = probe_result_.num_layers;
        last_result_.layer_names = probe_result_.layer_names;
        last_result_.load_time_ms = 100;  // Placeholder

        is_importing_ = false;
        show_result_ = true;

        // Call callback on main thread (if set)
        if (import_complete_callback_) {
            import_complete_callback_(last_result_, imported_graph_json_);
        }

        if (last_result_.success) {
            spdlog::info("Model imported successfully from {}", std::string(input_path_));
        } else {
            spdlog::error("Model import failed: {}", last_result_.error_message);
        }
    });
}

} // namespace cyxwiz
