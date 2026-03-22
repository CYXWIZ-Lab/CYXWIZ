/**
 * Dataset Panel - Export Tab
 * Model export/import UI embedding existing ExportDialog and ImportDialog inline.
 */

#include "dataset_panel.h"
#include "../icons.h"
#include "../../core/model_exporter.h"
#include "../../core/model_importer.h"
#include "../../core/training_manager.h"
#include "../../core/file_dialogs.h"
#include "../../data/data_table.h"
#include <imgui.h>
#include <spdlog/spdlog.h>

namespace gui {

void TrainingEvaluationPanel::RenderExportContent() {
    ImGui::Spacing();

    // Export section
    if (ImGui::CollapsingHeader(ICON_FA_FILE_EXPORT " Export Model", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent(8);

        if (!has_trained_model_) {
            ImGui::TextDisabled("No trained model available. Train a model first.");
        } else {
            ImGui::Text("Model: %s", trained_model_name_.c_str());
            ImGui::Spacing();

            // Format selection
            ImGui::Text("Export Format:");
            ImGui::RadioButton("CyxModel (.cyx)", &export_format_idx_, 0);
            ImGui::SameLine();
            ImGui::RadioButton("ONNX (.onnx)", &export_format_idx_, 1);
            ImGui::SameLine();
            ImGui::RadioButton("Safetensors (.safetensors)", &export_format_idx_, 2);
            ImGui::SameLine();
            ImGui::RadioButton("GGUF (.gguf)", &export_format_idx_, 3);

            ImGui::Spacing();

            // Format info
            const char* format_descriptions[] = {
                "CyxWiz native format. Preserves full model architecture, optimizer state, and training metadata.",
                "Open Neural Network Exchange. Compatible with ONNX Runtime, TensorRT, and many frameworks.",
                "Safe, efficient tensor serialization. Used by Hugging Face ecosystem.",
                "GPT-Generated Unified Format. Used for llama.cpp and local LLM inference."
            };
            ImGui::TextWrapped("%s", format_descriptions[export_format_idx_]);

            ImGui::Spacing();

            // Metadata
            if (ImGui::TreeNode("Metadata")) {
                ImGui::InputText("Model Name", export_model_name_, sizeof(export_model_name_));
                ImGui::InputText("Author", export_author_, sizeof(export_author_));
                ImGui::InputTextMultiline("Description", export_description_, sizeof(export_description_),
                                          ImVec2(-1, 60));
                ImGui::TreePop();
            }

            ImGui::Spacing();

            if (ImGui::Button(ICON_FA_FILE_EXPORT " Export...", ImVec2(150, 0))) {
                auto& tmgr = cyxwiz::TrainingManager::Instance();
                auto* model = tmgr.GetLastTrainedModel();
                if (model) {
                    cyxwiz::ModelFormat fmt;
                    std::string ext;
                    switch (export_format_idx_) {
                        case 0: fmt = cyxwiz::ModelFormat::CyxModel; ext = ".cyxmodel"; break;
                        case 1: fmt = cyxwiz::ModelFormat::ONNX; ext = ".onnx"; break;
                        case 2: fmt = cyxwiz::ModelFormat::Safetensors; ext = ".safetensors"; break;
                        default: fmt = cyxwiz::ModelFormat::GGUF; ext = ".gguf"; break;
                    }
                    auto result = cyxwiz::FileDialogs::SaveFile("Export Model",
                        {{"Model File", "*" + ext}});
                    if (result) {
                        cyxwiz::ModelExporter exporter;
                        cyxwiz::ExportOptions opts;
                        opts.format = fmt;
                        opts.model_name = std::string(export_model_name_);
                        opts.author = std::string(export_author_);
                        opts.description = std::string(export_description_);
                        auto export_result = exporter.Export(*model, nullptr,
                            &tmgr.GetLastMetrics(), "", *result, opts);
                        if (export_result.success) {
                            spdlog::info("Export: Model saved to {}", *result);
                        } else {
                            spdlog::error("Export: Failed - {}", export_result.error_message);
                        }
                    }
                }
            }
        }

        ImGui::Unindent(8);
    }

    ImGui::Spacing();

    // Import section
    if (ImGui::CollapsingHeader(ICON_FA_FILE_IMPORT " Import Model")) {
        ImGui::Indent(8);

        ImGui::Text("Supported formats: CyxModel, ONNX, Safetensors, GGUF, PyTorch (.pt)");
        ImGui::Spacing();

        if (ImGui::Button(ICON_FA_FOLDER_OPEN " Browse...", ImVec2(150, 0))) {
            auto result = cyxwiz::FileDialogs::OpenFile("Import Model",
                {{"Model Files", "*.cyxmodel;*.onnx;*.safetensors;*.gguf;*.pt"},
                 {"All Files", "*"}});
            if (result) {
                spdlog::info("Export: Selected model file for import: {}", *result);
                // Probe the file to show info
                cyxwiz::ModelImporter importer;
                auto probe = importer.ProbeFile(*result);
                if (probe.valid) {
                    spdlog::info("Export: Probed model - format: {}, layers: {}",
                                 (int)probe.format, probe.num_layers);
                } else {
                    spdlog::warn("Export: Failed to probe model file");
                }
            }
        }

        ImGui::Unindent(8);
    }

    ImGui::Spacing();

    // Dataset export
    if (ImGui::CollapsingHeader(ICON_FA_DATABASE " Export Dataset")) {
        ImGui::Indent(8);

        if (!IsDatasetLoaded()) {
            ImGui::TextDisabled("No dataset loaded.");
        } else {
            ImGui::Text("Export preprocessed dataset for use in other tools.");
            ImGui::Spacing();

            static int ds_format = 0;
            ImGui::Combo("Format##DatasetExport", &ds_format,
                         "CSV\0NumPy (.npy)\0HDF5 (.h5)\0JSON\0");

            ImGui::Spacing();

            if (ImGui::Button(ICON_FA_FILE_EXPORT " Export Dataset...", ImVec2(180, 0))) {
                const char* exts[] = {".csv", ".npy", ".h5", ".json"};
                const char* descs[] = {"CSV Files", "NumPy Files", "HDF5 Files", "JSON Files"};
                auto result = cyxwiz::FileDialogs::SaveFile("Export Dataset",
                    {{descs[ds_format], std::string("*") + exts[ds_format]}});
                if (result) {
                    // For CSV, use DataTable's built-in SaveToCSV
                    if (ds_format == 0) {
                        auto& registry = cyxwiz::DataTableRegistry::Instance();
                        auto table = registry.GetTable(cached_info_.name);
                        if (table) {
                            table->SaveToCSV(*result);
                            spdlog::info("Export: Dataset saved to {}", *result);
                        }
                    } else {
                        spdlog::info("Export: {} format export not yet implemented", exts[ds_format]);
                    }
                }
            }
        }

        ImGui::Unindent(8);
    }
}

} // namespace gui
