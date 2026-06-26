#include "properties.h"
#include "properties_advanced.h"
#include "properties_executor.h"
#include "properties_metadata_editor.h"
#include "properties_node_editors.h"
#include "properties_presets.h"
#include "../core/node_metadata_registry.h"
#include "node_editor.h"
#include "node_config_dialog.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <cstring>

namespace gui {

namespace {

bool IsDialogOnlyPropertiesNode(NodeType type) {
    switch (type) {
        case NodeType::DataInput:
        case NodeType::DataOutput:
        case NodeType::DataConvert:
        case NodeType::TextTokenizer:
        case NodeType::TextVocabulary:
        case NodeType::TextPadding:
        case NodeType::Embedding:
            return true;
        default:
            return false;
    }
}

bool IsCustomSequencePropertiesNode(NodeType type) {
    switch (type) {
        case NodeType::NERSequenceBuilder:
        case NodeType::TokenVocabulary:
        case NodeType::POSVocabulary:
        case NodeType::NERTagVocabulary:
        case NodeType::SequenceTagOutput:
            return true;
        default:
            return false;
    }
}

} // namespace

Properties::Properties() : show_window_(true) {
}

Properties::~Properties() = default;

void Properties::SetSelectedNode(MLNode* node) {
    selected_node_ = node;
}

void Properties::ClearSelection() {
    selected_node_ = nullptr;
}

NodeShapeInfo Properties::ComputeNodeShape(int node_id) {
    return properties_shape::ComputeNodeShape(node_editor_, node_id);
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

            // Dialog-backed nodes keep detailed settings in their dedicated
            // dialogs. The side panel stays compact and avoids duplicated,
            // partial parameter editors.
            bool is_dialog_only = IsDialogOnlyPropertiesNode(selected_node_->type);

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
                    properties_shape::RenderShapeInfo(node_editor_, shape_info);
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
        if (IsCustomSequencePropertiesNode(node.type)) {
            RenderNodeProperties(node);
            return;
        }
        properties_metadata::RenderParametersContent(
            node,
            metadata,
            validation_errors_,
            [this](MLNode& fallback_node) { RenderNodeProperties(fallback_node); },
            [this]() { InvalidateShapes(); });
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

// ==================== Node Executor Integration ====================

void Properties::RenderExecutorSection(MLNode& node) {
    properties_executor::RenderExecutorSection(node_editor_, node);
}

} // namespace gui
