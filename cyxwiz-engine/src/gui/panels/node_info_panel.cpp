#include "node_info_panel.h"
#include "../icons.h"
#include "../../core/node_metadata_registry.h"
#include <imgui.h>

namespace cyxwiz {

NodeInfoPanel::NodeInfoPanel() : Panel("Info") {
    visible_ = true;
}

void NodeInfoPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(320, 450), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(GetName(), &visible_)) {
        if (HasSelection() && metadata_) {
            RenderHeader();
            ImGui::Separator();
            RenderDescription();
            ImGui::Separator();
            RenderPorts();
            ImGui::Separator();
            RenderParameters();

            if (!metadata_->example_usage.empty()) {
                ImGui::Separator();
                RenderExamples();
            }
        } else {
            RenderPlaceholder();
        }
    }
    ImGui::End();
}

const char* NodeInfoPanel::GetIcon() const {
    return ICON_FA_CIRCLE_INFO;
}

void NodeInfoPanel::SetSelectedNode(NodeType type) {
    selected_type_ = type;
    if (type != NodeType::Unknown) {
        metadata_ = NodeMetadataRegistry::Instance().GetMetadata(type);
    } else {
        metadata_ = nullptr;
    }
}

void NodeInfoPanel::ClearSelection() {
    selected_type_ = NodeType::Unknown;
    metadata_ = nullptr;
}

void NodeInfoPanel::RenderHeader() {
    if (!metadata_) return;

    // Icon and name
    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]); // Use default font (with icons)

    // Large icon
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.7f, 1.0f, 1.0f));
    ImGui::SetWindowFontScale(1.5f);
    ImGui::TextUnformatted(metadata_->icon.c_str());
    ImGui::SetWindowFontScale(1.0f);
    ImGui::PopStyleColor();

    ImGui::SameLine();

    // Name
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 4);
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
    ImGui::Text("%s", metadata_->name.c_str());
    ImGui::PopStyleColor();

    ImGui::PopFont();

    // Category badge
    ImGui::TextDisabled("%s", GetCategoryDisplayName(metadata_->category).c_str());

    // Status badge (if template or deprecated)
    if (metadata_->IsTemplate()) {
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.8f, 0.2f, 1.0f));
        ImGui::Text("%s Coming Soon", ICON_FA_CLOCK);
        ImGui::PopStyleColor();
    } else if (metadata_->IsDeprecated()) {
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
        ImGui::Text("%s Deprecated", ICON_FA_TRIANGLE_EXCLAMATION);
        ImGui::PopStyleColor();
    }
}

void NodeInfoPanel::RenderDescription() {
    if (!metadata_) return;

    ImGui::Spacing();

    // Brief description
    if (!metadata_->brief_description.empty()) {
        ImGui::TextWrapped("%s", metadata_->brief_description.c_str());
        ImGui::Spacing();
    }

    // Detailed help text
    if (!metadata_->help_text.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.7f, 0.7f, 0.7f, 1.0f));
        ImGui::TextWrapped("%s", metadata_->help_text.c_str());
        ImGui::PopStyleColor();
    }

    ImGui::Spacing();
}

void NodeInfoPanel::RenderPorts() {
    if (!metadata_) return;

    // Inputs
    if (!metadata_->inputs.empty()) {
        ImGui::Text("%s Inputs", ICON_FA_RIGHT_TO_BRACKET);
        ImGui::Indent(16.0f);

        for (const auto& port : metadata_->inputs) {
            ImU32 color = GetPinTypeColor(port.type);
            ImGui::PushStyleColor(ImGuiCol_Text, ImGui::ColorConvertU32ToFloat4(color));
            ImGui::Text("%s", ICON_FA_CIRCLE);
            ImGui::PopStyleColor();

            ImGui::SameLine();
            ImGui::Text("%s", port.name.c_str());

            ImGui::SameLine();
            ImGui::TextDisabled("(%s)", GetPinTypeName(port.type));

            if (!port.required) {
                ImGui::SameLine();
                ImGui::TextDisabled("[optional]");
            }

            if (!port.description.empty()) {
                ImGui::Indent(16.0f);
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 0.6f, 0.6f, 1.0f));
                ImGui::TextWrapped("%s", port.description.c_str());
                ImGui::PopStyleColor();
                ImGui::Unindent(16.0f);
            }
        }

        ImGui::Unindent(16.0f);
        ImGui::Spacing();
    }

    // Outputs
    if (!metadata_->outputs.empty()) {
        ImGui::Text("%s Outputs", ICON_FA_RIGHT_FROM_BRACKET);
        ImGui::Indent(16.0f);

        for (const auto& port : metadata_->outputs) {
            ImU32 color = GetPinTypeColor(port.type);
            ImGui::PushStyleColor(ImGuiCol_Text, ImGui::ColorConvertU32ToFloat4(color));
            ImGui::Text("%s", ICON_FA_CIRCLE);
            ImGui::PopStyleColor();

            ImGui::SameLine();
            ImGui::Text("%s", port.name.c_str());

            ImGui::SameLine();
            ImGui::TextDisabled("(%s)", GetPinTypeName(port.type));

            if (!port.description.empty()) {
                ImGui::Indent(16.0f);
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 0.6f, 0.6f, 1.0f));
                ImGui::TextWrapped("%s", port.description.c_str());
                ImGui::PopStyleColor();
                ImGui::Unindent(16.0f);
            }
        }

        ImGui::Unindent(16.0f);
        ImGui::Spacing();
    }
}

void NodeInfoPanel::RenderParameters() {
    if (!metadata_ || metadata_->parameters.empty()) return;

    ImGui::Text("%s Parameters", ICON_FA_SLIDERS);
    ImGui::Indent(16.0f);

    for (const auto& param : metadata_->parameters) {
        // Parameter name and type
        ImGui::Text("%s", param.name.c_str());
        ImGui::SameLine();
        ImGui::TextDisabled("(%s)", param.type.c_str());

        // Default value
        if (!param.default_value.empty()) {
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.8f, 0.5f, 1.0f));
            ImGui::Text("= %s", param.default_value.c_str());
            ImGui::PopStyleColor();
        }

        // Description
        if (!param.description.empty()) {
            ImGui::Indent(16.0f);
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 0.6f, 0.6f, 1.0f));
            ImGui::TextWrapped("%s", param.description.c_str());
            ImGui::PopStyleColor();
            ImGui::Unindent(16.0f);
        }

        // Enum values
        if (!param.enum_values.empty()) {
            ImGui::Indent(16.0f);
            ImGui::TextDisabled("Options:");
            for (const auto& val : param.enum_values) {
                ImGui::SameLine();
                ImGui::TextDisabled("%s", val.c_str());
            }
            ImGui::Unindent(16.0f);
        }
    }

    ImGui::Unindent(16.0f);
    ImGui::Spacing();
}

void NodeInfoPanel::RenderExamples() {
    if (!metadata_ || metadata_->example_usage.empty()) return;

    ImGui::Text("%s Example", ICON_FA_CODE);
    ImGui::Indent(16.0f);

    // Code block styling
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.1f, 0.1f, 0.1f, 1.0f));
    ImGui::BeginChild("ExampleCode", ImVec2(0, 80), true);

    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.8f, 0.9f, 0.8f, 1.0f));
    ImGui::TextWrapped("%s", metadata_->example_usage.c_str());
    ImGui::PopStyleColor();

    ImGui::EndChild();
    ImGui::PopStyleColor();

    ImGui::Unindent(16.0f);
    ImGui::Spacing();
}

void NodeInfoPanel::RenderPlaceholder() {
    ImVec2 window_size = ImGui::GetContentRegionAvail();

    // Center the placeholder content
    float icon_size = 48.0f;
    ImGui::SetCursorPosY(window_size.y * 0.3f);

    // Center icon
    ImGui::SetCursorPosX((window_size.x - icon_size) * 0.5f);
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 0.3f, 0.3f, 1.0f));
    ImGui::SetWindowFontScale(2.0f);
    ImGui::TextUnformatted(ICON_FA_CIRCLE_INFO);
    ImGui::SetWindowFontScale(1.0f);
    ImGui::PopStyleColor();

    ImGui::Spacing();
    ImGui::Spacing();

    // Instruction text
    const char* text = "Select a node to view details";
    ImVec2 text_size = ImGui::CalcTextSize(text);
    ImGui::SetCursorPosX((window_size.x - text_size.x) * 0.5f);
    ImGui::TextDisabled("%s", text);

    ImGui::Spacing();

    const char* hint = "Hover in Node Browser or\nselect in Node Editor";
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.4f, 0.4f, 1.0f));
    // Center multi-line hint
    ImVec2 hint_size = ImGui::CalcTextSize(hint);
    ImGui::SetCursorPosX((window_size.x - hint_size.x) * 0.5f);
    ImGui::TextUnformatted(hint);
    ImGui::PopStyleColor();
}

ImU32 NodeInfoPanel::GetPinTypeColor(PinType type) const {
    switch (type) {
        case PinType::Tensor:      return IM_COL32(100, 200, 255, 255);  // Blue
        case PinType::Labels:      return IM_COL32(255, 200, 100, 255);  // Orange
        case PinType::Parameters:  return IM_COL32(200, 100, 255, 255);  // Purple
        case PinType::Loss:        return IM_COL32(255, 100, 100, 255);  // Red
        case PinType::Optimizer:   return IM_COL32(100, 255, 200, 255);  // Cyan
        case PinType::Dataset:     return IM_COL32(100, 255, 100, 255);  // Green
        default:                   return IM_COL32(180, 180, 180, 255);  // Gray
    }
}

const char* NodeInfoPanel::GetPinTypeName(PinType type) const {
    switch (type) {
        case PinType::Tensor:      return "Tensor";
        case PinType::Labels:      return "Labels";
        case PinType::Parameters:  return "Parameters";
        case PinType::Loss:        return "Loss";
        case PinType::Optimizer:   return "Optimizer";
        case PinType::Dataset:     return "Dataset";
        default:                   return "Unknown";
    }
}

} // namespace cyxwiz
