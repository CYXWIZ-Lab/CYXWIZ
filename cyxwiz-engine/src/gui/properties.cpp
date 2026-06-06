// Include Windows headers first, then undef conflicting macros
#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
// Undefine Windows macros that conflict with our method names
#ifdef CreateDialog
#undef CreateDialog
#endif
#ifdef CreateDialogA
#undef CreateDialogA
#endif
#ifdef CreateDialogW
#undef CreateDialogW
#endif
#endif

#include "properties.h"
#include "properties_advanced.h"
#include "properties_executor.h"
#include "properties_node_editors.h"
#include "properties_parameter_rules.h"
#include "properties_presets.h"
#include "../core/node_metadata_registry.h"
#include "node_editor.h"
#include "node_config_dialog.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cstdio>
#include <cstring>

namespace gui {

Properties::Properties() : show_window_(true) {
}

Properties::~Properties() = default;

void Properties::SetSelectedNode(MLNode* node) {
    selected_node_ = node;
}

void Properties::ClearSelection() {
    selected_node_ = nullptr;
}

std::string Properties::FormatShape(const std::vector<size_t>& shape) {
    return properties_shape::FormatShape(shape);
}

size_t Properties::GetBatchSize() {
    return properties_shape::GetBatchSize(node_editor_);
}

std::string Properties::FormatShapeMatrix(const std::vector<size_t>& shape, size_t batch_size) {
    return properties_shape::FormatShapeMatrix(shape, batch_size);
}

std::vector<size_t> Properties::GetInputShapeFromDataset() {
    return properties_shape::GetInputShapeFromDataset(node_editor_);
}

std::vector<size_t> Properties::InferOutputShape(
    NodeType type,
    const std::vector<size_t>& input_shape,
    const std::map<std::string, std::string>& params)
{
    return properties_shape::InferOutputShape(node_editor_, type, input_shape, params);
}

LayerParameters Properties::ComputeLayerParameters(
    NodeType type,
    const std::vector<size_t>& input_shape,
    const std::map<std::string, std::string>& params)
{
    return properties_shape::ComputeLayerParameters(type, input_shape, params);
}

NodeShapeInfo Properties::ComputeNodeShape(int node_id) {
    return properties_shape::ComputeNodeShape(node_editor_, node_id);
}

void Properties::RenderShapeInfo(const NodeShapeInfo& shape_info) {
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    size_t batch_size = GetBatchSize();

    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Tensor Shape (batch = %zu)", batch_size);
    ImGui::Spacing();

    if (!shape_info.is_valid) {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "Cannot compute shape");
        if (!shape_info.error.empty()) {
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Error: %s", shape_info.error.c_str());
        }
        return;
    }

    // Calculate memory sizes (assuming Float32 = 4 bytes)
    size_t input_memory = batch_size * shape_info.input_size * sizeof(float);
    size_t output_memory = batch_size * shape_info.output_size * sizeof(float);

    // Input shape section
    ImGui::Text("Input:");
    ImGui::Indent();
    ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "%s", FormatShapeMatrix(shape_info.input_shape, batch_size).c_str());
    if (shape_info.input_shape.size() > 1) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Per-sample: %s", FormatShape(shape_info.input_shape).c_str());
    }
    // Show memory size
    if (input_memory >= 1024 * 1024) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Memory: %.2f MB (%zu elements)",
                           input_memory / (1024.0f * 1024.0f), batch_size * shape_info.input_size);
    } else if (input_memory >= 1024) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Memory: %.2f KB (%zu elements)",
                           input_memory / 1024.0f, batch_size * shape_info.input_size);
    } else {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Memory: %zu bytes (%zu elements)",
                           input_memory, batch_size * shape_info.input_size);
    }
    ImGui::Unindent();

    ImGui::Spacing();

    // Output shape section
    ImGui::Text("Output:");
    ImGui::Indent();
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "%s", FormatShapeMatrix(shape_info.output_shape, batch_size).c_str());
    if (shape_info.output_shape.size() > 1) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Per-sample: %s", FormatShape(shape_info.output_shape).c_str());
    }
    // Show memory size
    if (output_memory >= 1024 * 1024) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Memory: %.2f MB (%zu elements)",
                           output_memory / (1024.0f * 1024.0f), batch_size * shape_info.output_size);
    } else if (output_memory >= 1024) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Memory: %.2f KB (%zu elements)",
                           output_memory / 1024.0f, batch_size * shape_info.output_size);
    } else {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Memory: %zu bytes (%zu elements)",
                           output_memory, batch_size * shape_info.output_size);
    }
    ImGui::Unindent();

    // Shape transformation summary
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1.0f), "Transform: %zu x %zu -> %zu x %zu",
                       batch_size, shape_info.input_size, batch_size, shape_info.output_size);

    // Display learnable parameters if this layer has any
    if (shape_info.params.has_parameters) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), "Learnable Parameters");
        ImGui::Spacing();

        // Weight info
        ImGui::Text("Weight:");
        ImGui::Indent();
        ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "%s", FormatShape(shape_info.params.weight_shape).c_str());
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "%zu parameters", shape_info.params.weight_count);
        size_t weight_memory = shape_info.params.weight_count * sizeof(float);
        if (weight_memory >= 1024 * 1024) {
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Memory: %.2f MB", weight_memory / (1024.0f * 1024.0f));
        } else if (weight_memory >= 1024) {
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Memory: %.2f KB", weight_memory / 1024.0f);
        } else {
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Memory: %zu bytes", weight_memory);
        }
        ImGui::Unindent();

        ImGui::Spacing();

        // Bias info
        ImGui::Text("Bias:");
        ImGui::Indent();
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "%s", FormatShape(shape_info.params.bias_shape).c_str());
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "%zu parameters", shape_info.params.bias_count);
        ImGui::Unindent();

        ImGui::Spacing();
        ImGui::Separator();

        // Total parameters summary
        size_t total_memory = shape_info.params.total_params * sizeof(float);
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.3f, 1.0f), "Total: %zu params", shape_info.params.total_params);
        if (total_memory >= 1024 * 1024) {
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Parameter Memory: %.2f MB", total_memory / (1024.0f * 1024.0f));
        } else if (total_memory >= 1024) {
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Parameter Memory: %.2f KB", total_memory / 1024.0f);
        } else {
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Parameter Memory: %zu bytes", total_memory);
        }
    }
}

void Properties::Render() {
    if (!show_window_) return;

    if (ImGui::Begin("Properties", &show_window_)) {
        if (!selected_node_) {
            // Placeholder when no node selected
            ImVec2 avail = ImGui::GetContentRegionAvail();
            ImGui::SetCursorPosY(avail.y * 0.3f);

            // Center icon
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 0.3f, 0.3f, 1.0f));
            ImGui::SetWindowFontScale(2.0f);
            float icon_width = ImGui::CalcTextSize("\xef\x80\x85").x;  // ICON_FA_SLIDERS
            ImGui::SetCursorPosX((avail.x - icon_width) * 0.5f);
            ImGui::Text("\xef\x80\x85");
            ImGui::SetWindowFontScale(1.0f);
            ImGui::PopStyleColor();

            ImGui::Spacing();

            const char* text = "Select a node to edit properties";
            ImVec2 text_size = ImGui::CalcTextSize(text);
            ImGui::SetCursorPosX((avail.x - text_size.x) * 0.5f);
            ImGui::TextDisabled("%s", text);
        } else {
            // Get metadata for this node type
            const cyxwiz::NodeMetadata* metadata =
                cyxwiz::NodeMetadataRegistry::Instance().GetMetadata(selected_node_->type);

            // Check if this is a dialog-only node (DataInput/DataOutput)
            bool is_dialog_only = (selected_node_->type == NodeType::DataInput ||
                                   selected_node_->type == NodeType::DataOutput);

            // Phase 3: Section-based rendering
            RenderGeneralSection(*selected_node_);

            // Skip other sections for dialog-only nodes
            if (!is_dialog_only) {
                ImGui::Spacing();

                // Parameters section - use metadata-driven rendering if available
                RenderParametersSection(*selected_node_, metadata);

                ImGui::Spacing();

                // Shape information section
                if (ImGui::CollapsingHeader("Shape Info", ImGuiTreeNodeFlags_DefaultOpen)) {
                    NodeShapeInfo shape_info = ComputeNodeShape(selected_node_->id);
                    RenderShapeInfo(shape_info);
                }

                ImGui::Spacing();

                // Advanced section
                section_advanced_open_ = properties_advanced::RenderAdvancedSection(
                    node_editor_, *selected_node_, section_advanced_open_);

                ImGui::Spacing();

                // Presets section
                RenderPresetsSection(*selected_node_);

                ImGui::Spacing();

                // Node Executor section (for analytics nodes like KMeans, PCA, etc.)
                RenderExecutorSection(*selected_node_);
            }
        }
    }
    ImGui::End();

    // Render active configuration dialog (if open)
    if (active_dialog_ && active_dialog_->IsOpen()) {
        if (!active_dialog_->Render()) {
            // Dialog was closed
            active_dialog_.reset();
        }
    }
}

void Properties::RenderNodeProperties(MLNode& node) {
    properties_node_editors::RenderNodeProperties(
        node,
        properties_node_editors::RenderNodePropertiesContext{
            node_editor_,
            scope_buffers_,
            scope_demo_time_,
            [this]() { InvalidateShapes(); }
        });
}

// ========== Phase 3: Enhanced Property Sections ==========

void Properties::RenderGeneralSection(MLNode& node) {
    ImGui::SetNextItemOpen(section_general_open_, ImGuiCond_Once);
    if (ImGui::CollapsingHeader("General", ImGuiTreeNodeFlags_DefaultOpen)) {
        section_general_open_ = true;

        // Node name (editable)
        char name_buf[128];
        strncpy(name_buf, node.name.c_str(), sizeof(name_buf) - 1);
        name_buf[sizeof(name_buf) - 1] = '\0';

        ImGui::Text("Name:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(180.0f);
        if (ImGui::InputText("##node_name", name_buf, sizeof(name_buf))) {
            node.name = name_buf;
        }

        // Node ID (read-only)
        ImGui::Text("ID: %d", node.id);

        // Node type
        auto* metadata = cyxwiz::NodeMetadataRegistry::Instance().GetMetadata(node.type);
        if (metadata) {
            ImGui::Text("Type: %s", metadata->name.c_str());

            // Category badge
            ImGui::SameLine();
            ImGui::TextDisabled("(%s)", cyxwiz::GetCategoryDisplayName(metadata->category).c_str());

            // Icon display
            if (!metadata->icon.empty()) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.7f, 1.0f, 1.0f));
                ImGui::Text("%s", metadata->icon.c_str());
                ImGui::PopStyleColor();
            }
        }

        // KNIME-style "Open Dialog" button for complex nodes
        RenderOpenDialogButton(node);
    } else {
        section_general_open_ = false;
    }
}

void Properties::RenderOpenDialogButton(MLNode& node) {
    // Check if this node type should have an "Open Dialog" button
    bool should_show = ShouldShowOpenDialogButton(node.type);
    if (!should_show) {
        return;
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Center the button
    float button_width = 150.0f;
    float avail_width = ImGui::GetContentRegionAvail().x;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (avail_width - button_width) * 0.5f);

    // Styled "Open Dialog" button (similar to KNIME)
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.4f, 0.6f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.5f, 0.7f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.15f, 0.35f, 0.55f, 1.0f));

    if (ImGui::Button("Open Dialog...", ImVec2(button_width, 0))) {
        // Create and open the dialog for this node
        active_dialog_ = NodeConfigDialogFactory::Instance().CreateDialog(&node);
        if (active_dialog_) {
            // Pass the graph context so visualization / inspection
            // dialogs can walk upstream pins for auto-populating
            // dataset hints. Most dialogs ignore this.
            active_dialog_->SetNodeEditor(node_editor_);
            active_dialog_->Open();
            spdlog::info("Opened configuration dialog for node '{}'", node.name);
        }
    }

    ImGui::PopStyleColor(3);

    // Tooltip
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::Text("Open detailed configuration dialog");
        ImGui::TextDisabled("(Configure all settings with preview)");
        ImGui::EndTooltip();
    }
}

void Properties::RenderParametersSection(MLNode& node, const cyxwiz::NodeMetadata* metadata) {
    ImGui::SetNextItemOpen(section_parameters_open_, ImGuiCond_Once);
    if (ImGui::CollapsingHeader("Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
        section_parameters_open_ = true;

        if (metadata && !metadata->parameters.empty()) {
            bool rendered_any = false;
            for (const auto& param : metadata->parameters) {
                if (properties_rules::ShouldHideGenericParameter(node.type, param)) {
                    validation_errors_.erase(param.name);
                    continue;
                }
                RenderParameter(node, param);
                rendered_any = true;
            }
            if (!rendered_any) {
                ImGui::TextDisabled("Configure this node from its dialog.");
            }
        } else {
            // Fallback to existing node-specific rendering
            RenderNodeProperties(node);
        }
    } else {
        section_parameters_open_ = false;
    }
}

void Properties::RenderPresetsSection(MLNode& node) {
    ImGui::SetNextItemOpen(section_presets_open_, ImGuiCond_Once);
    if (ImGui::CollapsingHeader("Presets")) {
        section_presets_open_ = true;

        // List available presets
        auto presets = properties_presets::GetPresetsForNodeType(node.type);
        if (!presets.empty()) {
            ImGui::Text("Available Presets:");
            for (const auto& preset : presets) {
                if (ImGui::Button(preset.c_str())) {
                    properties_presets::LoadPreset(node, preset);
                    InvalidateShapes();
                }
                ImGui::SameLine();
            }
            ImGui::NewLine();
            ImGui::Separator();
        }

        // Save new preset
        ImGui::Text("Save Current as Preset:");
        ImGui::SetNextItemWidth(150.0f);
        ImGui::InputText("##preset_name", preset_name_buffer_, sizeof(preset_name_buffer_));
        ImGui::SameLine();
        if (ImGui::Button("Save") && preset_name_buffer_[0] != '\0') {
            properties_presets::SavePreset(node, preset_name_buffer_);
            preset_name_buffer_[0] = '\0';
        }
    } else {
        section_presets_open_ = false;
    }
}

void Properties::RenderParameter(MLNode& node, const cyxwiz::ParameterDefinition& param) {
    ImGui::PushID(param.name.c_str());

    std::string& value = node.parameters[param.name];

    // Initialize with default if empty
    if (value.empty() && !param.default_value.empty()) {
        value = param.default_value;
    }

    std::string initial_error;
    if (!properties_rules::ValidateParameter(value, param, initial_error)) {
        validation_errors_[param.name] = initial_error;
    } else {
        validation_errors_.erase(param.name);
    }

    // Check for validation errors
    bool has_error = validation_errors_.count(param.name) > 0;
    if (has_error) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
    }

    // Label
    ImGui::Text("%s:", param.name.c_str());
    if (has_error) {
        ImGui::PopStyleColor();
    }

    // Tooltip with description
    if (!param.description.empty() && ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::TextUnformatted(param.description.c_str());
        if (!param.default_value.empty()) {
            ImGui::TextDisabled("Default: %s", param.default_value.c_str());
        }
        ImGui::EndTooltip();
    }

    ImGui::SameLine();

    bool changed = false;

    // Render based on parameter type
    if (param.type == "int") {
        int int_val = 0;
        properties_rules::TryParseIntStrict(value, int_val);
        ImGui::SetNextItemWidth(120.0f);
        if (ImGui::InputInt("##value", &int_val)) {
            properties_rules::NumericRange range = properties_rules::ParseNumericRange(param.validation);
            if (range.has_range) {
                int_val = std::clamp(
                    int_val,
                    static_cast<int>(range.min_value),
                    static_cast<int>(range.max_value));
            }
            value = std::to_string(int_val);
            changed = true;
        }
    }
    else if (param.type == "float") {
        double parsed_float = 0.0;
        properties_rules::TryParseDoubleStrict(value, parsed_float);
        float float_val = static_cast<float>(parsed_float);
        ImGui::SetNextItemWidth(120.0f);

        properties_rules::NumericRange range = properties_rules::ParseNumericRange(param.validation);
        if (range.has_range) {
            float min_v = static_cast<float>(range.min_value);
            float max_v = static_cast<float>(range.max_value);
            float_val = std::clamp(float_val, min_v, max_v);
            if (ImGui::SliderFloat("##value", &float_val, min_v, max_v, "%.4f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.4f", float_val);
                value = buf;
                changed = true;
            }
        } else {
            if (ImGui::InputFloat("##value", &float_val, 0.01f, 0.1f, "%.4f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.4f", float_val);
                value = buf;
                changed = true;
            }
        }
    }
    else if (param.type == "bool") {
        bool bool_val = (value == "true" || value == "1");
        if (ImGui::Checkbox("##value", &bool_val)) {
            value = bool_val ? "true" : "false";
            changed = true;
        }
    }
    else if ((param.type == "enum" || param.type == "dropdown") && !param.enum_values.empty()) {
        // Find current index
        int current_idx = 0;
        for (size_t i = 0; i < param.enum_values.size(); i++) {
            if (param.enum_values[i] == value) {
                current_idx = static_cast<int>(i);
                break;
            }
        }

        // Build combo items
        std::vector<const char*> items;
        for (const auto& ev : param.enum_values) {
            items.push_back(ev.c_str());
        }

        ImGui::SetNextItemWidth(150.0f);
        if (ImGui::Combo("##value", &current_idx, items.data(), static_cast<int>(items.size()))) {
            value = param.enum_values[current_idx];
            changed = true;
        }
    }
    else if (param.type == "file") {
        char file_buf[512];
        strncpy(file_buf, value.c_str(), sizeof(file_buf) - 1);
        file_buf[sizeof(file_buf) - 1] = '\0';

        ImGui::SetNextItemWidth(180.0f);
        if (ImGui::InputText("##value", file_buf, sizeof(file_buf))) {
            value = file_buf;
            changed = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Browse")) {
#ifdef _WIN32
            OPENFILENAMEA ofn = {};
            char file[512] = {};
            strncpy(file, value.c_str(), sizeof(file) - 1);
            ofn.lStructSize = sizeof(ofn);
            ofn.lpstrFilter = "All Files\0*.*\0";
            ofn.lpstrFile = file;
            ofn.nMaxFile = sizeof(file);
            ofn.Flags = OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
            if (GetOpenFileNameA(&ofn)) {
                value = file;
                changed = true;
            }
#endif
        }
    }
    else {
        // Default: string input
        char str_buf[256];
        strncpy(str_buf, value.c_str(), sizeof(str_buf) - 1);
        str_buf[sizeof(str_buf) - 1] = '\0';

        ImGui::SetNextItemWidth(180.0f);
        if (ImGui::InputText("##value", str_buf, sizeof(str_buf))) {
            value = str_buf;
            changed = true;
        }
    }

    // Validation on change
    if (changed) {
        std::string error;
        if (!properties_rules::ValidateParameter(value, param, error)) {
            validation_errors_[param.name] = error;
        } else {
            validation_errors_.erase(param.name);
        }
        InvalidateShapes();
        has_error = validation_errors_.count(param.name) > 0;
    }

    // Show validation error
    if (has_error) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "!");
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", validation_errors_[param.name].c_str());
            ImGui::EndTooltip();
        }
    }

    ImGui::PopID();
}

// ==================== Node Executor Integration ====================

void Properties::RenderExecutorSection(MLNode& node) {
    properties_executor::RenderExecutorSection(node_editor_, node);
}

} // namespace gui
