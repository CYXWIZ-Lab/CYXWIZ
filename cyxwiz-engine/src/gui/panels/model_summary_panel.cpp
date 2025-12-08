#include "model_summary_panel.h"
#include "../node_editor.h"
#include "../icons.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <fstream>

#ifdef _WIN32
#include <windows.h>
#endif

namespace cyxwiz {

ModelSummaryPanel::ModelSummaryPanel() {
    export_path_.resize(256);
}

void ModelSummaryPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(700, 500), ImGuiCond_FirstUseEver);
    if (ImGui::Begin(ICON_FA_LIST " Model Summary", &visible_, ImGuiWindowFlags_MenuBar)) {
        RenderToolbar();

        if (!analysis_.is_valid) {
            if (analysis_.error_message.empty()) {
                ImGui::TextDisabled("No model to analyze. Create a node graph first.");
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Error: %s", analysis_.error_message.c_str());
            }

            if (ImGui::Button(ICON_FA_ROTATE " Refresh Analysis")) {
                RefreshAnalysis();
            }
        } else {
            // Tab bar for different views
            if (ImGui::BeginTabBar("ModelSummaryTabs")) {
                if (ImGui::BeginTabItem(ICON_FA_TABLE " Layer Table")) {
                    RenderLayerTable();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem(ICON_FA_CHART_PIE " Summary")) {
                    RenderSummaryStats();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem(ICON_FA_ARROWS_LEFT_RIGHT " Shape Flow")) {
                    RenderShapeFlow();
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
        }

        RenderDetailPopup();
    }
    ImGui::End();
}

void ModelSummaryPanel::RenderToolbar() {
    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("Options")) {
            ImGui::Checkbox("Auto-refresh", &auto_refresh_);
            ImGui::Separator();
            ImGui::Checkbox("Show FLOPs", &show_flops_);
            ImGui::Checkbox("Show Memory", &show_memory_);
            ImGui::Checkbox("Show Non-trainable", &show_non_trainable_);
            ImGui::Separator();
            ImGui::SetNextItemWidth(100);
            if (ImGui::InputInt("Batch Size", &batch_size_, 1, 10)) {
                batch_size_ = std::max(1, batch_size_);
                RefreshAnalysis();
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Export")) {
            if (ImGui::MenuItem(ICON_FA_FILE_LINES " Export as Text...")) {
                ExportAsText();
            }
            if (ImGui::MenuItem(ICON_FA_FILE_CODE " Export as JSON...")) {
                ExportAsJson();
            }
            ImGui::Separator();
            if (ImGui::MenuItem(ICON_FA_CLIPBOARD " Copy to Clipboard")) {
                CopyToClipboard();
            }
            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }

    // Refresh button
    if (ImGui::Button(ICON_FA_ROTATE " Refresh")) {
        RefreshAnalysis();
    }
    ImGui::SameLine();
    if (analysis_.is_valid) {
        ImGui::TextDisabled("Last analysis: %zu layers", analysis_.layers.size());
    }
}

void ModelSummaryPanel::RenderLayerTable() {
    // Column count depends on options
    int column_count = 4;  // Layer, Type, Output Shape, Parameters
    if (show_flops_) column_count++;
    if (show_memory_) column_count++;
    if (show_non_trainable_) column_count++;

    ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                            ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY |
                            ImGuiTableFlags_Sortable;

    if (ImGui::BeginTable("##ModelSummaryLayerTable", column_count, flags)) {
        // Headers
        ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 30.0f);
        ImGui::TableSetupColumn("Layer", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Output Shape", ImGuiTableColumnFlags_WidthFixed, 120.0f);
        ImGui::TableSetupColumn("Parameters", ImGuiTableColumnFlags_WidthFixed, 100.0f);
        if (show_flops_) ImGui::TableSetupColumn("FLOPs", ImGuiTableColumnFlags_WidthFixed, 100.0f);
        if (show_memory_) ImGui::TableSetupColumn("Memory", ImGuiTableColumnFlags_WidthFixed, 80.0f);
        if (show_non_trainable_) ImGui::TableSetupColumn("Non-train", ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableHeadersRow();

        // Rows
        for (size_t i = 0; i < analysis_.layers.size(); ++i) {
            const auto& layer = analysis_.layers[i];
            ImGui::TableNextRow();

            // Index
            ImGui::TableNextColumn();
            ImGui::Text("%zu", i + 1);

            // Layer name/type
            ImGui::TableNextColumn();
            std::string display_name = layer.name.empty() ?
                GetNodeTypeName(layer.type) :
                layer.name + " (" + GetNodeTypeName(layer.type) + ")";

            // Selectable row
            bool selected = (selected_layer_ == static_cast<int>(i));
            if (ImGui::Selectable(display_name.c_str(), selected, ImGuiSelectableFlags_SpanAllColumns)) {
                selected_layer_ = static_cast<int>(i);
            }
            if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
                detail_layer_index_ = static_cast<int>(i);
                show_detail_popup_ = true;
            }

            // Output shape
            ImGui::TableNextColumn();
            ImGui::Text("%s", FormatShape(layer.output_shape).c_str());

            // Parameters
            ImGui::TableNextColumn();
            if (layer.parameters > 0) {
                ImGui::Text("%s", FormatParameterCount(layer.parameters).c_str());
            } else {
                ImGui::TextDisabled("-");
            }

            // FLOPs
            if (show_flops_) {
                ImGui::TableNextColumn();
                if (layer.flops > 0) {
                    ImGui::Text("%s", FormatFLOPs(layer.flops).c_str());
                } else {
                    ImGui::TextDisabled("-");
                }
            }

            // Memory
            if (show_memory_) {
                ImGui::TableNextColumn();
                ImGui::Text("%s", FormatMemory(layer.memory_bytes).c_str());
            }

            // Non-trainable
            if (show_non_trainable_) {
                ImGui::TableNextColumn();
                if (layer.non_trainable_params > 0) {
                    ImGui::Text("%s", FormatParameterCount(layer.non_trainable_params).c_str());
                } else {
                    ImGui::TextDisabled("-");
                }
            }
        }

        ImGui::EndTable();
    }
}

void ModelSummaryPanel::RenderSummaryStats() {
    // Summary cards
    float card_width = 200.0f;
    float card_height = 80.0f;
    ImVec4 card_bg = ImVec4(0.15f, 0.15f, 0.2f, 1.0f);

    static int card_id = 0;
    card_id = 0;  // Reset each frame
    auto DrawCard = [&](const char* title, const char* value, const char* icon, ImVec4 accent_color) {
        ImGui::PushID(card_id++);
        ImGui::BeginChild("##card", ImVec2(card_width, card_height), true);
        ImGui::PushStyleColor(ImGuiCol_Text, accent_color);
        ImGui::Text("%s", icon);
        ImGui::PopStyleColor();
        ImGui::SameLine();
        ImGui::Text("%s", title);
        ImGui::Separator();
        ImGui::SetWindowFontScale(1.3f);
        ImGui::Text("%s", value);
        ImGui::SetWindowFontScale(1.0f);
        ImGui::EndChild();
        ImGui::PopID();
    };

    // Row 1: Parameters
    DrawCard("Total Parameters",
             FormatParameterCount(analysis_.total_parameters).c_str(),
             ICON_FA_WEIGHT_SCALE,
             ImVec4(0.4f, 0.7f, 1.0f, 1.0f));
    ImGui::SameLine();
    DrawCard("Trainable",
             FormatParameterCount(analysis_.trainable_parameters).c_str(),
             ICON_FA_GRADUATION_CAP,
             ImVec4(0.4f, 0.9f, 0.4f, 1.0f));
    ImGui::SameLine();
    DrawCard("Non-trainable",
             FormatParameterCount(analysis_.non_trainable_parameters).c_str(),
             ICON_FA_LOCK,
             ImVec4(0.9f, 0.6f, 0.3f, 1.0f));

    ImGui::Spacing();

    // Row 2: Compute
    DrawCard("Total FLOPs",
             FormatFLOPs(analysis_.total_flops).c_str(),
             ICON_FA_MICROCHIP,
             ImVec4(0.9f, 0.4f, 0.9f, 1.0f));
    ImGui::SameLine();
    DrawCard("Memory (Activations)",
             FormatMemory(analysis_.total_memory_bytes).c_str(),
             ICON_FA_MEMORY,
             ImVec4(1.0f, 0.7f, 0.3f, 1.0f));
    ImGui::SameLine();
    DrawCard("Layers",
             std::to_string(analysis_.layers.size()).c_str(),
             ICON_FA_LAYER_GROUP,
             ImVec4(0.5f, 0.8f, 0.9f, 1.0f));

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Shape info
    ImGui::Text(ICON_FA_RIGHT_TO_BRACKET " Input Shape: %s", FormatShape(analysis_.input_shape).c_str());
    ImGui::Text(ICON_FA_RIGHT_FROM_BRACKET " Output Shape: %s", FormatShape(analysis_.output_shape).c_str());

    ImGui::Spacing();

    // Model size estimation
    int64_t param_bytes = analysis_.total_parameters * sizeof(float);
    ImGui::Text(ICON_FA_HARD_DRIVE " Estimated Model Size (FP32): %s", FormatMemory(param_bytes).c_str());
    ImGui::Text(ICON_FA_HARD_DRIVE " Estimated Model Size (FP16): %s", FormatMemory(param_bytes / 2).c_str());
}

void ModelSummaryPanel::RenderShapeFlow() {
    if (analysis_.layers.empty()) {
        ImGui::TextDisabled("No layers to visualize");
        return;
    }

    ImGui::Text(ICON_FA_DIAGRAM_PROJECT " Shape Flow Visualization");
    ImGui::Separator();

    // Draw shape flow as a horizontal diagram
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
    float box_width = 100.0f;
    float box_height = 50.0f;
    float spacing = 30.0f;
    float arrow_size = 10.0f;

    // Draw input shape
    ImVec2 pos = canvas_pos;
    ImU32 input_color = IM_COL32(80, 180, 80, 255);
    draw_list->AddRectFilled(pos, ImVec2(pos.x + box_width, pos.y + box_height), input_color, 4.0f);
    draw_list->AddRect(pos, ImVec2(pos.x + box_width, pos.y + box_height), IM_COL32(255, 255, 255, 200), 4.0f);

    // Input text
    std::string input_text = "Input";
    std::string input_shape_text = FormatShape(analysis_.input_shape);
    ImVec2 text_size = ImGui::CalcTextSize(input_text.c_str());
    draw_list->AddText(ImVec2(pos.x + (box_width - text_size.x) / 2, pos.y + 5), IM_COL32(255, 255, 255, 255), input_text.c_str());
    text_size = ImGui::CalcTextSize(input_shape_text.c_str());
    draw_list->AddText(ImVec2(pos.x + (box_width - text_size.x) / 2, pos.y + 25), IM_COL32(200, 200, 200, 255), input_shape_text.c_str());

    pos.x += box_width + spacing;

    // Draw layers (first few only to fit)
    int max_visible = 5;
    int layer_count = std::min(static_cast<int>(analysis_.layers.size()), max_visible);

    for (int i = 0; i < layer_count; ++i) {
        const auto& layer = analysis_.layers[i];

        // Arrow
        draw_list->AddLine(ImVec2(pos.x - spacing + 5, canvas_pos.y + box_height / 2),
                          ImVec2(pos.x - 5, canvas_pos.y + box_height / 2),
                          IM_COL32(150, 150, 150, 255), 2.0f);

        // Layer box
        ImU32 layer_color = IM_COL32(80, 120, 180, 255);
        draw_list->AddRectFilled(pos, ImVec2(pos.x + box_width, pos.y + box_height), layer_color, 4.0f);
        draw_list->AddRect(pos, ImVec2(pos.x + box_width, pos.y + box_height), IM_COL32(255, 255, 255, 200), 4.0f);

        // Layer text
        std::string layer_name = GetNodeTypeName(layer.type);
        if (layer_name.length() > 12) layer_name = layer_name.substr(0, 10) + "..";
        std::string shape_text = FormatShape(layer.output_shape);
        if (shape_text.length() > 14) shape_text = shape_text.substr(0, 12) + "..";

        text_size = ImGui::CalcTextSize(layer_name.c_str());
        draw_list->AddText(ImVec2(pos.x + (box_width - text_size.x) / 2, pos.y + 5), IM_COL32(255, 255, 255, 255), layer_name.c_str());
        text_size = ImGui::CalcTextSize(shape_text.c_str());
        draw_list->AddText(ImVec2(pos.x + (box_width - text_size.x) / 2, pos.y + 25), IM_COL32(200, 200, 200, 255), shape_text.c_str());

        pos.x += box_width + spacing;
    }

    // Show ellipsis if more layers
    if (analysis_.layers.size() > static_cast<size_t>(max_visible)) {
        draw_list->AddText(ImVec2(pos.x - spacing / 2 - 10, canvas_pos.y + box_height / 2 - 5),
                          IM_COL32(200, 200, 200, 255), "...");
        pos.x += 30;
    }

    // Output box
    ImU32 output_color = IM_COL32(180, 80, 80, 255);
    draw_list->AddRectFilled(pos, ImVec2(pos.x + box_width, pos.y + box_height), output_color, 4.0f);
    draw_list->AddRect(pos, ImVec2(pos.x + box_width, pos.y + box_height), IM_COL32(255, 255, 255, 200), 4.0f);

    std::string output_text = "Output";
    std::string output_shape_text = FormatShape(analysis_.output_shape);
    text_size = ImGui::CalcTextSize(output_text.c_str());
    draw_list->AddText(ImVec2(pos.x + (box_width - text_size.x) / 2, pos.y + 5), IM_COL32(255, 255, 255, 255), output_text.c_str());
    text_size = ImGui::CalcTextSize(output_shape_text.c_str());
    draw_list->AddText(ImVec2(pos.x + (box_width - text_size.x) / 2, pos.y + 25), IM_COL32(200, 200, 200, 255), output_shape_text.c_str());

    // Reserve space
    ImGui::Dummy(ImVec2(pos.x - canvas_pos.x + box_width + 20, box_height + 20));

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Full shape list
    ImGui::Text("Complete Shape Flow:");
    if (ImGui::BeginChild("ShapeFlowList", ImVec2(0, 0), true)) {
        ImGui::Text("Input: %s", FormatShape(analysis_.input_shape).c_str());
        for (size_t i = 0; i < analysis_.layers.size(); ++i) {
            const auto& layer = analysis_.layers[i];
            ImGui::Text("  -> [%zu] %s: %s",
                       i + 1,
                       GetNodeTypeName(layer.type).c_str(),
                       FormatShape(layer.output_shape).c_str());
        }
    }
    ImGui::EndChild();
}

void ModelSummaryPanel::RenderDetailPopup() {
    if (show_detail_popup_ && detail_layer_index_ >= 0 &&
        detail_layer_index_ < static_cast<int>(analysis_.layers.size())) {

        ImGui::OpenPopup("Layer Details");
        show_detail_popup_ = false;
    }

    if (ImGui::BeginPopupModal("Layer Details", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        if (detail_layer_index_ >= 0 && detail_layer_index_ < static_cast<int>(analysis_.layers.size())) {
            const auto& layer = analysis_.layers[detail_layer_index_];

            ImGui::Text("Layer #%d: %s", detail_layer_index_ + 1,
                       layer.name.empty() ? GetNodeTypeName(layer.type).c_str() : layer.name.c_str());
            ImGui::Separator();

            ImGui::Text("Type: %s", GetNodeTypeName(layer.type).c_str());
            ImGui::Text("Input Shape: %s", FormatShape(layer.input_shape).c_str());
            ImGui::Text("Output Shape: %s", FormatShape(layer.output_shape).c_str());
            ImGui::Separator();
            ImGui::Text("Parameters: %s (%lld)", FormatParameterCount(layer.parameters).c_str(), layer.parameters);
            ImGui::Text("Non-trainable: %s (%lld)", FormatParameterCount(layer.non_trainable_params).c_str(), layer.non_trainable_params);
            ImGui::Text("FLOPs: %s (%lld)", FormatFLOPs(layer.flops).c_str(), layer.flops);
            ImGui::Text("Memory: %s (%lld bytes)", FormatMemory(layer.memory_bytes).c_str(), layer.memory_bytes);
        }

        ImGui::Separator();
        if (ImGui::Button("Close", ImVec2(120, 0))) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

void ModelSummaryPanel::RefreshAnalysis() {
    if (!node_editor_) {
        analysis_.is_valid = false;
        analysis_.error_message = "No node editor connected";
        return;
    }

    const auto& nodes = node_editor_->GetNodes();
    const auto& links = node_editor_->GetLinks();

    if (nodes.empty()) {
        analysis_.is_valid = false;
        analysis_.error_message = "";  // Not an error, just empty
        return;
    }

    analysis_ = analyzer_.AnalyzeGraph(nodes, links, batch_size_);
    selected_layer_ = -1;

    if (analysis_.is_valid) {
        spdlog::info("Model analysis complete: {} layers, {} params, {} FLOPs",
                    analysis_.layers.size(),
                    FormatParameterCount(analysis_.total_parameters),
                    FormatFLOPs(analysis_.total_flops));
    } else {
        spdlog::warn("Model analysis failed: {}", analysis_.error_message);
    }
}

void ModelSummaryPanel::ExportAsText() {
    if (!analysis_.is_valid) return;

    std::string summary = analyzer_.GenerateSummary(analysis_);

    // Simple file dialog replacement - save to current directory
    std::string filename = "model_summary.txt";
    std::ofstream file(filename);
    if (file.is_open()) {
        file << summary;
        file.close();
        spdlog::info("Model summary exported to: {}", filename);
    } else {
        spdlog::error("Failed to export model summary");
    }
}

void ModelSummaryPanel::ExportAsJson() {
    if (!analysis_.is_valid) return;

    std::string json = analyzer_.ExportToJson(analysis_);

    std::string filename = "model_summary.json";
    std::ofstream file(filename);
    if (file.is_open()) {
        file << json;
        file.close();
        spdlog::info("Model analysis exported to: {}", filename);
    } else {
        spdlog::error("Failed to export model analysis");
    }
}

void ModelSummaryPanel::CopyToClipboard() {
    if (!analysis_.is_valid) return;

    std::string summary = analyzer_.GenerateSummary(analysis_);

#ifdef _WIN32
    if (OpenClipboard(nullptr)) {
        EmptyClipboard();
        HGLOBAL hGlob = GlobalAlloc(GMEM_MOVEABLE, summary.size() + 1);
        if (hGlob) {
            char* pGlob = static_cast<char*>(GlobalLock(hGlob));
            if (pGlob) {
                memcpy(pGlob, summary.c_str(), summary.size() + 1);
                GlobalUnlock(hGlob);
                SetClipboardData(CF_TEXT, hGlob);
            }
        }
        CloseClipboard();
        spdlog::info("Model summary copied to clipboard");
    }
#else
    // Linux/macOS - just log for now
    spdlog::info("Clipboard not implemented on this platform. Summary:\n{}", summary);
#endif
}

} // namespace cyxwiz
