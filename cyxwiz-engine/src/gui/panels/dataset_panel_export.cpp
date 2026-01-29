/**
 * Dataset Panel - Export Tab
 * Model export/import UI embedding existing ExportDialog and ImportDialog inline.
 */

#include "dataset_panel.h"
#include "../icons.h"
#include <imgui.h>

namespace gui {

void DatasetPanel::RenderExportContent() {
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
                // TODO: Wire to ExportDialog / ModelExporter
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
            // TODO: Wire to ImportDialog / ModelImporter
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
                // TODO: Wire dataset export
            }
        }

        ImGui::Unindent(8);
    }
}

} // namespace gui
