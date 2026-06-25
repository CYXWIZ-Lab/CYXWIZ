#include "node_info_panel.h"
#include "../icons.h"
#include "../../core/node_metadata_registry.h"
#include <imgui.h>
#include <algorithm>

namespace cyxwiz {

namespace {

const SupportAxisDefinition* FindSupportState(const NodeMetadata* metadata) {
    if (!metadata) return nullptr;
    auto it = std::find_if(
        metadata->support_axes.begin(),
        metadata->support_axes.end(),
        [](const SupportAxisDefinition& axis) {
            return axis.name == "Support State";
        });
    return it != metadata->support_axes.end() ? &*it : nullptr;
}

const SupportAxisDefinition* FindSupportAxis(const NodeMetadata* metadata,
                                             const char* axis_name) {
    if (!metadata) return nullptr;
    auto it = std::find_if(
        metadata->support_axes.begin(),
        metadata->support_axes.end(),
        [axis_name](const SupportAxisDefinition& axis) {
            return axis.name == axis_name;
        });
    return it != metadata->support_axes.end() ? &*it : nullptr;
}

ImVec4 SupportStateColor(const std::string& state) {
    if (state == "real") return ImVec4(0.45f, 0.85f, 0.55f, 1.0f);
    if (state == "partial") return ImVec4(1.0f, 0.72f, 0.32f, 1.0f);
    if (state == "blocked") return ImVec4(1.0f, 0.45f, 0.35f, 1.0f);
    return ImVec4(0.65f, 0.68f, 0.75f, 1.0f);
}

std::string SupportStateLabel(const std::string& state) {
    if (state == "real") return "Real";
    if (state == "partial") return "Partial";
    if (state == "blocked") return "Blocked";
    return state.empty() ? "Unknown" : state;
}

std::string SupportAxisValueLabel(const std::string& value) {
    if (value == "real") return "Real";
    if (value == "partial") return "Partial";
    if (value == "blocked") return "Blocked";
    if (value == "none") return "None";
    if (value == "training_backend") return "Training backend";
    if (value == "pipeline_executor") return "Pipeline executor";
    if (value == "pipeline_operator_factory") return "Pipeline operator factory";
    if (value == "ui_only") return "UI-only";
    if (value == "classic_ml") return "Classic ML";
    if (value == "deep_learning") return "Deep learning";
    if (value == "model_layer") return "Model layer";
    if (value == "activation") return "Activation";
    if (value == "loss") return "Loss";
    if (value == "optimizer") return "Optimizer";
    if (value == "regression") return "Regression";
    if (value == "multiclass_classification") return "Multiclass classification";
    if (value == "binary_classification") return "Binary classification";
    if (value == "supported") return "Supported";
    if (value == "unsupported") return "Unsupported";
    if (value == "fail_closed") return "Fail closed";
    if (value == "hard_fail") return "Hard fail";

    std::string label = value.empty() ? std::string("Unknown") : value;
    std::replace(label.begin(), label.end(), '_', ' ');
    return label;
}

bool ShouldShowSupportReason(const SupportAxisDefinition& axis) {
    if (axis.reason.empty()) return false;
    if (!axis.supported) return true;
    return axis.name == "Runtime" ||
           axis.name == "Compile" ||
           axis.name == "Training" ||
           axis.name == "Materializer";
}

} // namespace

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
            if (!metadata_->support_axes.empty()) {
                ImGui::Separator();
                RenderSupport();
            }
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

    if (const auto* workflow_lane = FindSupportAxis(metadata_, "Workflow Lane")) {
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.75f, 0.62f, 1.0f, 1.0f));
        const std::string label = SupportAxisValueLabel(workflow_lane->value);
        ImGui::Text("%s %s", ICON_FA_ROUTE, label.c_str());
        ImGui::PopStyleColor();
        if (ImGui::IsItemHovered() && !workflow_lane->reason.empty()) {
            ImGui::SetTooltip("%s", workflow_lane->reason.c_str());
        }
    }

    if (const auto* training_role = FindSupportAxis(metadata_, "Training Role")) {
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.48f, 0.82f, 0.72f, 1.0f));
        const std::string label = SupportAxisValueLabel(training_role->value);
        ImGui::Text("%s %s", ICON_FA_MICROCHIP, label.c_str());
        ImGui::PopStyleColor();
        if (ImGui::IsItemHovered() && !training_role->reason.empty()) {
            ImGui::SetTooltip("%s", training_role->reason.c_str());
        }
    }

    if (const auto* task_type = FindSupportAxis(metadata_, "Task Type")) {
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.55f, 0.75f, 1.0f, 1.0f));
        const std::string label = SupportAxisValueLabel(task_type->value);
        ImGui::Text("%s %s", ICON_FA_BULLSEYE, label.c_str());
        ImGui::PopStyleColor();
        if (ImGui::IsItemHovered() && !task_type->reason.empty()) {
            ImGui::SetTooltip("%s", task_type->reason.c_str());
        }
    }

    if (const auto* owner = FindSupportAxis(metadata_, "Implementation Owner")) {
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.65f, 0.68f, 0.75f, 1.0f));
        const std::string label = SupportAxisValueLabel(owner->value);
        ImGui::Text("%s %s", ICON_FA_CODE_BRANCH, label.c_str());
        ImGui::PopStyleColor();
        if (ImGui::IsItemHovered() && !owner->reason.empty()) {
            ImGui::SetTooltip("%s", owner->reason.c_str());
        }
    }

    if (const auto* support_state = FindSupportState(metadata_)) {
        ImGui::SameLine();
        ImGui::PushStyleColor(
            ImGuiCol_Text,
            SupportStateColor(support_state->value));
        ImGui::Text("%s %s",
                    support_state->supported
                        ? ICON_FA_CIRCLE_CHECK
                        : ICON_FA_TRIANGLE_EXCLAMATION,
                    SupportStateLabel(support_state->value).c_str());
        ImGui::PopStyleColor();
        if (ImGui::IsItemHovered() && !support_state->reason.empty()) {
            ImGui::SetTooltip("%s", support_state->reason.c_str());
        }
    }

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

void NodeInfoPanel::RenderSupport() {
    if (!metadata_ || metadata_->support_axes.empty()) return;

    ImGui::Text("%s Support", ICON_FA_CIRCLE_INFO);
    ImGui::Indent(16.0f);

    for (const auto& axis : metadata_->support_axes) {
        const ImVec4 color = axis.supported
            ? ImVec4(0.45f, 0.85f, 0.55f, 1.0f)
            : ImVec4(1.0f, 0.55f, 0.35f, 1.0f);
        const char* icon = axis.supported
            ? ICON_FA_CIRCLE_CHECK
            : ICON_FA_TRIANGLE_EXCLAMATION;

        ImGui::PushStyleColor(ImGuiCol_Text, color);
        ImGui::TextUnformatted(icon);
        ImGui::PopStyleColor();

        ImGui::SameLine();
        ImGui::Text("%s", axis.name.c_str());
        ImGui::SameLine();
        const std::string axis_value = SupportAxisValueLabel(axis.value);
        ImGui::TextDisabled("%s", axis_value.c_str());

        if (ShouldShowSupportReason(axis)) {
            ImGui::Indent(18.0f);
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.68f, 0.68f, 0.68f, 1.0f));
            ImGui::TextWrapped("%s", axis.reason.c_str());
            ImGui::PopStyleColor();
            ImGui::Unindent(18.0f);
        }
    }

    ImGui::Unindent(16.0f);
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
