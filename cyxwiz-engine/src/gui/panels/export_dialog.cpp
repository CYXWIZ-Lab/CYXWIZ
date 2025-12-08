#include "export_dialog.h"
#include "../icons.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <cyxwiz/sequential.h>
#include <cyxwiz/optimizer.h>
#include "../../core/training_executor.h"

#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#include <shlobj.h>
#endif

namespace cyxwiz {

ExportDialog::ExportDialog()
    : Panel("Export Model", false) {
}

ExportDialog::~ExportDialog() {
    // Wait for export thread to finish
    if (export_thread_ && export_thread_->joinable()) {
        export_thread_->join();
    }
}

void ExportDialog::Open() {
    is_open_ = true;
    show_result_ = false;

    // Reset state
    is_exporting_ = false;
    export_progress_ = 0;
    export_total_ = 0;

    // Default values
    selected_format_ = ModelFormat::CyxModel;
    include_optimizer_state_ = true;
    include_training_history_ = true;
    include_graph_ = true;
    quantization_index_ = 0;
    compress_ = true;
}

void ExportDialog::Close() {
    is_open_ = false;
}

void ExportDialog::SetModelData(
    SequentialModel* model,
    const Optimizer* optimizer,
    const TrainingMetrics* metrics,
    const std::string& graph_json
) {
    model_ = model;
    optimizer_ = optimizer;
    metrics_ = metrics;
    graph_json_ = graph_json;
}

void ExportDialog::Render() {
    if (!is_open_) return;

    ImGui::SetNextWindowSize(ImVec2(500, 550), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(
        ImVec2(ImGui::GetIO().DisplaySize.x * 0.5f, ImGui::GetIO().DisplaySize.y * 0.5f),
        ImGuiCond_FirstUseEver,
        ImVec2(0.5f, 0.5f)
    );

    if (ImGui::Begin("Export Model", &is_open_, ImGuiWindowFlags_NoCollapse)) {
        if (is_exporting_) {
            RenderProgress();
        } else if (show_result_) {
            // Show export result
            if (last_result_.success) {
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f),
                    ICON_FA_CHECK " Export Successful!");
                ImGui::Separator();

                ImGui::Text("Output: %s", last_result_.output_path.c_str());
                ImGui::Text("File Size: %.2f MB", last_result_.file_size_bytes / (1024.0f * 1024.0f));
                ImGui::Text("Parameters: %d", last_result_.num_parameters);
                ImGui::Text("Layers: %d", last_result_.num_layers);
                ImGui::Text("Export Time: %lld ms", last_result_.export_time_ms);

                ImGui::Separator();

                if (ImGui::Button("Close", ImVec2(120, 0))) {
                    Close();
                }

                ImGui::SameLine();

                if (ImGui::Button("Export Another", ImVec2(120, 0))) {
                    show_result_ = false;
                }
            } else {
                ImGui::TextColored(ImVec4(0.9f, 0.2f, 0.2f, 1.0f),
                    ICON_FA_XMARK " Export Failed");
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
            // Check if model is available
            if (!model_) {
                ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f),
                    ICON_FA_TRIANGLE_EXCLAMATION " No trained model available");
                ImGui::TextWrapped(
                    "Please train a model first before exporting. "
                    "Use the node editor to build a model and start training.");

                ImGui::Separator();

                if (ImGui::Button("Close", ImVec2(120, 0))) {
                    Close();
                }
            } else {
                RenderFormatSelection();
                ImGui::Separator();
                RenderOptions();
                ImGui::Separator();
                RenderMetadata();
                ImGui::Separator();
                RenderButtons();
            }
        }
    }
    ImGui::End();
}

void ExportDialog::RenderFormatSelection() {
    ImGui::Text(ICON_FA_DOWNLOAD " Export Format");
    ImGui::Spacing();

    // Format radio buttons
    const char* format_names[] = {
        "CyxWiz Model (.cyxmodel)",
        "Safetensors (.safetensors)",
        "ONNX (.onnx)",
        "GGUF (.gguf)"
    };

    const char* format_descriptions[] = {
        "Native format with graph, weights, config, and training history",
        "Safe tensor serialization (HuggingFace compatible)",
        "Industry standard for cross-platform inference",
        "GGML format for LLM inference (quantization support)"
    };

    bool format_supported[] = {
        true,
        true,
#ifdef CYXWIZ_HAS_ONNX
        true,
#else
        false,
#endif
        false  // GGUF not yet implemented
    };

    for (int i = 0; i < 4; i++) {
        ModelFormat format = static_cast<ModelFormat>(i);
        bool selected = (selected_format_ == format);

        ImGui::BeginDisabled(!format_supported[i]);

        if (ImGui::RadioButton(format_names[i], selected)) {
            selected_format_ = format;

            // Update output path extension
            std::string path(output_path_);
            if (!path.empty()) {
                size_t dot = path.rfind('.');
                if (dot != std::string::npos) {
                    path = path.substr(0, dot);
                }
                path += GetFormatExtension(format);
                strncpy(output_path_, path.c_str(), sizeof(output_path_) - 1);
            }
        }

        ImGui::EndDisabled();

        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::BeginTooltip();
            ImGui::TextUnformatted(format_descriptions[i]);
            if (!format_supported[i]) {
                ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Not available");
            }
            ImGui::EndTooltip();
        }
    }
}

void ExportDialog::RenderOptions() {
    ImGui::Text(ICON_FA_GEAR " Options");
    ImGui::Spacing();

    // Output path
    ImGui::Text("Output Path:");
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 80);
    ImGui::InputText("##output_path", output_path_, sizeof(output_path_));
    ImGui::SameLine();
    if (ImGui::Button("Browse...", ImVec2(70, 0))) {
        std::string ext;
        std::string filter;
        switch (selected_format_) {
            case ModelFormat::CyxModel:
                filter = "CyxWiz Model (*.cyxmodel)\0*.cyxmodel\0";
                ext = "cyxmodel";
                break;
            case ModelFormat::Safetensors:
                filter = "Safetensors (*.safetensors)\0*.safetensors\0";
                ext = "safetensors";
                break;
            case ModelFormat::ONNX:
                filter = "ONNX Model (*.onnx)\0*.onnx\0";
                ext = "onnx";
                break;
            case ModelFormat::GGUF:
                filter = "GGUF (*.gguf)\0*.gguf\0";
                ext = "gguf";
                break;
            default:
                filter = "All Files (*.*)\0*.*\0";
                ext = "";
                break;
        }

        std::string path = SaveFileDialog(filter.c_str(), ext.c_str());
        if (!path.empty()) {
            strncpy(output_path_, path.c_str(), sizeof(output_path_) - 1);
        }
    }

    ImGui::Spacing();

    // Format-specific options
    if (selected_format_ == ModelFormat::CyxModel) {
        ImGui::Text("Include:");
        ImGui::Checkbox("Optimizer State", &include_optimizer_state_);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Include optimizer state for resuming training");
        }

        ImGui::SameLine(200);
        ImGui::Checkbox("Training History", &include_training_history_);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Include loss/accuracy history for visualization");
        }

        ImGui::Checkbox("Graph Definition", &include_graph_);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Include node graph for editing in CyxWiz");
        }

        ImGui::SameLine(200);
        ImGui::Checkbox("Compress", &compress_);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Compress weights (reduces file size)");
        }
    }
    else if (selected_format_ == ModelFormat::ONNX) {
        ImGui::SetNextItemWidth(100);
        ImGui::InputInt("Opset Version", &opset_version_);
        if (opset_version_ < 1) opset_version_ = 1;
        if (opset_version_ > 20) opset_version_ = 20;
    }
    else if (selected_format_ == ModelFormat::GGUF) {
        const char* quant_names[] = {
            "None (FP32)",
            "FP16",
            "Q8_0",
            "Q4_0",
            "Q4_1"
        };

        ImGui::SetNextItemWidth(150);
        ImGui::Combo("Quantization", &quantization_index_, quant_names, IM_ARRAYSIZE(quant_names));
    }
}

void ExportDialog::RenderMetadata() {
    ImGui::Text(ICON_FA_CIRCLE_INFO " Metadata");
    ImGui::Spacing();

    ImGui::SetNextItemWidth(300);
    ImGui::InputText("Model Name", model_name_, sizeof(model_name_));

    ImGui::SetNextItemWidth(300);
    ImGui::InputText("Author", author_, sizeof(author_));

    ImGui::SetNextItemWidth(-1);
    ImGui::InputTextMultiline("Description", description_, sizeof(description_),
                              ImVec2(0, 60));
}

void ExportDialog::RenderProgress() {
    ImGui::Text(ICON_FA_SPINNER " Exporting...");
    ImGui::Spacing();

    int progress = export_progress_.load();
    int total = export_total_.load();

    float fraction = (total > 0) ? static_cast<float>(progress) / total : 0.0f;
    ImGui::ProgressBar(fraction, ImVec2(-1, 0));

    {
        std::lock_guard<std::mutex> lock(status_mutex_);
        ImGui::Text("%s", export_status_.c_str());
    }
}

void ExportDialog::RenderButtons() {
    // Model info
    if (model_) {
        // Note: GetParameters() is not const, so we display basic info
        ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f),
            ICON_FA_CIRCLE_INFO " Model: %zu layers",
            model_->Size());
    }

    ImGui::Spacing();

    // Export button
    bool can_export = (strlen(output_path_) > 0) && model_ && !is_exporting_;

    ImGui::BeginDisabled(!can_export);
    if (ImGui::Button(ICON_FA_DOWNLOAD " Export", ImVec2(120, 30))) {
        StartExport();
    }
    ImGui::EndDisabled();

    ImGui::SameLine();

    if (ImGui::Button("Cancel", ImVec2(80, 30))) {
        Close();
    }
}

std::string ExportDialog::SaveFileDialog(const char* filter, const char* default_ext) {
#ifdef _WIN32
    char filename[MAX_PATH] = "";

    OPENFILENAMEA ofn = {};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = nullptr;
    ofn.lpstrFilter = filter;
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrDefExt = default_ext;
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST;
    ofn.lpstrTitle = "Export Model";

    if (GetSaveFileNameA(&ofn)) {
        return std::string(filename);
    }
#endif
    return "";
}

void ExportDialog::StartExport() {
    if (!model_) return;

    is_exporting_ = true;
    show_result_ = false;
    export_progress_ = 0;
    export_total_ = 6;

    // Build export options
    export_options_.format = selected_format_;
    export_options_.include_optimizer_state = include_optimizer_state_;
    export_options_.include_training_history = include_training_history_;
    export_options_.include_graph = include_graph_;
    export_options_.compress = compress_;
    export_options_.opset_version = opset_version_;
    export_options_.model_name = model_name_;
    export_options_.author = author_;
    export_options_.description = description_;

    // Set quantization
    switch (quantization_index_) {
        case 0: export_options_.quantization = Quantization::None; break;
        case 1: export_options_.quantization = Quantization::FP16; break;
        case 2: export_options_.quantization = Quantization::Q8_0; break;
        case 3: export_options_.quantization = Quantization::Q4_0; break;
        case 4: export_options_.quantization = Quantization::Q4_1; break;
        default: export_options_.quantization = Quantization::None; break;
    }

    // Wait for any previous export thread
    if (export_thread_ && export_thread_->joinable()) {
        export_thread_->join();
    }

    // Start export in background thread
    export_thread_ = std::make_unique<std::thread>([this]() {
        ModelExporter exporter;

        auto progress_callback = [this](int current, int total, const std::string& status) {
            export_progress_ = current;
            export_total_ = total;
            {
                std::lock_guard<std::mutex> lock(status_mutex_);
                export_status_ = status;
            }
        };

        last_result_ = exporter.Export(
            *model_,
            optimizer_,
            metrics_,
            graph_json_,
            output_path_,
            export_options_,
            progress_callback
        );

        is_exporting_ = false;
        show_result_ = true;

        // Call callback on main thread (if set)
        if (export_complete_callback_) {
            export_complete_callback_(last_result_);
        }

        if (last_result_.success) {
            spdlog::info("Model exported successfully to {}", last_result_.output_path);
        } else {
            spdlog::error("Model export failed: {}", last_result_.error_message);
        }
    });
}

} // namespace cyxwiz
