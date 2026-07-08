#include "properties.h"
#include "properties_advanced.h"
#include "properties_contract.h"
#include "properties_executor.h"
#include "properties_metadata_editor.h"
#include "properties_node_editors.h"
#include "properties_presets.h"
#include "properties_truth.h"
#include "../core/arrow_dataset.h"
#include "../core/data_registry.h"
#include "../core/node_metadata_registry.h"
#include "../core/parquet_backed_dataset.h"
#include "node_editor.h"
#include "node_config_dialog.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <array>
#include <algorithm>
#include <cstring>
#include <vector>

namespace gui {

namespace {

ImVec4 StatusColor(properties_truth::TruthStatus status) {
    switch (status) {
        case properties_truth::TruthStatus::OK:
            return ImVec4(0.35f, 0.85f, 0.45f, 1.0f);
        case properties_truth::TruthStatus::Missing:
        case properties_truth::TruthStatus::Conflicting:
        case properties_truth::TruthStatus::Unsupported:
            return ImVec4(1.0f, 0.35f, 0.35f, 1.0f);
        case properties_truth::TruthStatus::Defaulted:
        case properties_truth::TruthStatus::AliasUsed:
        case properties_truth::TruthStatus::RequiresDialog:
            return ImVec4(0.95f, 0.72f, 0.25f, 1.0f);
        default:
            return ImVec4(0.55f, 0.75f, 1.0f, 1.0f);
    }
}

void RenderStatusBadges(const std::vector<properties_truth::TruthStatus>& statuses) {
    for (const auto status : statuses) {
        ImGui::SameLine();
        ImGui::TextColored(StatusColor(status), "[%s]",
                           properties_truth::TruthStatusName(status));
    }
}

const char* ImplementationStatusName(cyxwiz::NodeImplementationStatus status) {
    switch (status) {
        case cyxwiz::NodeImplementationStatus::Implemented:
            return "Implemented";
        case cyxwiz::NodeImplementationStatus::Template:
            return "Planned";
        case cyxwiz::NodeImplementationStatus::Deprecated:
            return "Deprecated";
        case cyxwiz::NodeImplementationStatus::External:
            return "External";
    }
    return "Unknown";
}

ImVec4 ImplementationStatusColor(cyxwiz::NodeImplementationStatus status) {
    switch (status) {
        case cyxwiz::NodeImplementationStatus::Implemented:
            return ImVec4(0.35f, 0.85f, 0.45f, 1.0f);
        case cyxwiz::NodeImplementationStatus::Template:
        case cyxwiz::NodeImplementationStatus::External:
            return ImVec4(0.95f, 0.72f, 0.25f, 1.0f);
        case cyxwiz::NodeImplementationStatus::Deprecated:
            return ImVec4(1.0f, 0.35f, 0.35f, 1.0f);
    }
    return ImVec4(0.55f, 0.75f, 1.0f, 1.0f);
}

ImVec4 SupportAxisColor(const cyxwiz::SupportAxisDefinition& axis) {
    return axis.supported
        ? ImVec4(0.35f, 0.85f, 0.45f, 1.0f)
        : ImVec4(1.0f, 0.35f, 0.35f, 1.0f);
}

void RenderMetadataSupportTruth(const cyxwiz::NodeMetadata& metadata) {
    ImGui::Text("Implementation:");
    ImGui::SameLine();
    ImGui::TextColored(
        ImplementationStatusColor(metadata.status),
        "%s",
        ImplementationStatusName(metadata.status));

    if (!metadata.badge.empty()) {
        ImGui::SameLine();
        ImGui::TextDisabled("[%s]", metadata.badge.c_str());
    }

    if (metadata.support_axes.empty()) {
        return;
    }

    if (!ImGui::TreeNodeEx("Support Truth", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }

    for (const auto& axis : metadata.support_axes) {
        ImGui::Text("%s:", axis.name.c_str());
        ImGui::SameLine();
        ImGui::TextColored(SupportAxisColor(axis), "%s", axis.value.c_str());
        if (!axis.reason.empty()) {
            ImGui::TextDisabled("  %s", axis.reason.c_str());
        }
    }
    ImGui::TreePop();
}

std::string ParamOrEmpty(const MLNode& node, const char* key) {
    const auto it = node.parameters.find(key);
    return it == node.parameters.end() ? std::string() : it->second;
}

properties_truth::DatasetTruthFact BuildDatasetTruthFact(const MLNode& node) {
    properties_truth::DatasetTruthFact fact;
    fact.dataset_name = ParamOrEmpty(node, "dataset_name");
    if (fact.dataset_name.empty()) {
        fact.dataset_name = ParamOrEmpty(node, "dataset");
    }
    if (fact.dataset_name.empty()) {
        return fact;
    }

    auto& registry = cyxwiz::DataRegistry::Instance();
    if (const auto* text = registry.GetTextDatasetEntry(fact.dataset_name)) {
        fact.found = true;
        fact.backing_store = "TextDataset";
        fact.has_labels = text->has_labels;
        fact.has_label_column_metadata = !text->label_column.empty();
        fact.label_column = text->label_column;
        fact.has_class_count = true;
        fact.class_count = text->num_classes;
        return fact;
    }

    if (auto arrow_dataset = registry.GetArrowDataset(fact.dataset_name)) {
        fact.found = true;
        fact.backing_store = "Arrow";
        fact.columns = arrow_dataset->GetColumnNames();
        fact.has_labels = !ParamOrEmpty(node, "label_column").empty() ||
                          !ParamOrEmpty(node, "text_label_column").empty();
        return fact;
    }

    if (auto parquet_dataset = registry.GetParquetBackedDataset(fact.dataset_name)) {
        fact.found = true;
        fact.backing_store = "Parquet";
        fact.columns = parquet_dataset->GetColumnNames();
        fact.has_labels = !ParamOrEmpty(node, "label_column").empty() ||
                          !ParamOrEmpty(node, "text_label_column").empty();
        return fact;
    }

    if (const auto* image = registry.GetImageDatasetEntry(fact.dataset_name)) {
        fact.found = true;
        fact.backing_store = "ImageDataset";
        fact.has_labels = true;
        fact.has_class_count = true;
        fact.class_count = image->num_classes;
        return fact;
    }

    if (const auto* audio = registry.GetAudioDatasetEntry(fact.dataset_name)) {
        fact.found = true;
        fact.backing_store = "AudioDataset";
        fact.has_labels = audio->labeled_subdirs || !audio->label_col.empty();
        fact.has_label_column_metadata = !audio->label_col.empty();
        fact.label_column = audio->label_col;
        fact.has_class_count = true;
        fact.class_count = audio->num_classes;
        return fact;
    }

    if (registry.HasDataset(fact.dataset_name)) {
        auto handle = registry.GetDataset(fact.dataset_name);
        if (handle.IsValid()) {
            const auto info = handle.GetInfo();
            fact.found = true;
            fact.backing_store = "Dataset";
            fact.has_labels = info.num_classes > 0;
            fact.has_class_count = true;
            fact.class_count = info.num_classes;
            return fact;
        }
    }

    fact.message = "Dataset '" + fact.dataset_name + "' is not loaded.";
    return fact;
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

void Properties::SetBackendPlacementFacts(
    std::vector<properties_truth::BackendPlacementTruthFact> facts) {
    backend_placement_facts_ = std::move(facts);
}

void Properties::ClearBackendPlacementFacts() {
    backend_placement_facts_.clear();
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
            bool is_dialog_only =
                properties_contract::IsDialogOnlyPropertiesNode(selected_node_->type);

            // Phase 3: Section-based rendering
            RenderGeneralSection(*selected_node_);

            ImGui::Spacing();
            RenderTruthSummarySection(*selected_node_);

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

            RenderMetadataSupportTruth(*metadata);
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

void Properties::RenderTruthSummarySection(MLNode& node) {
    properties_truth::NodeTruthContext truth_context;
    std::vector<properties_truth::DatasetTruthFact> dataset_facts;
    if (node_editor_) {
        truth_context.nodes = &node_editor_->GetNodes();
        truth_context.links = &node_editor_->GetLinks();
        for (const auto& graph_node : node_editor_->GetNodes()) {
            if (graph_node.type != NodeType::DataInput) {
                continue;
            }
            auto fact = BuildDatasetTruthFact(graph_node);
            if (!fact.dataset_name.empty()) {
                dataset_facts.push_back(std::move(fact));
            }
        }
    }
    if (!backend_placement_facts_.empty()) {
        truth_context.backend_placements = &backend_placement_facts_;
    }
    if (!node_editor_ && node.type == NodeType::DataInput) {
        dataset_facts.push_back(BuildDatasetTruthFact(node));
    }
    if (!dataset_facts.empty()) {
        truth_context.dataset_facts = &dataset_facts;
    }

    auto report = properties_truth::ResolveNodeTruth(node, truth_context);
    if (report.properties.empty() && report.raw_parameters.empty()) {
        return;
    }

    if (!ImGui::CollapsingHeader("Truth Summary", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }

    if (report.properties.empty()) {
        ImGui::TextDisabled("No specialized effective-property rules for this node yet.");
    }

    for (const auto& property : report.properties) {
        ImGui::PushID(property.canonical_key.c_str());
        ImGui::Text("%s:", property.label.c_str());
        ImGui::SameLine();

        if (property.quick_editable) {
            std::array<char, 256> buffer{};
            std::strncpy(buffer.data(), property.effective_value.c_str(), buffer.size() - 1);
            ImGui::SetNextItemWidth(150.0f);
            if (ImGui::InputText("##effective", buffer.data(), buffer.size())) {
                if (!property.canonical_key.empty()) {
                    properties_truth::WriteCanonicalAndAliases(
                        node, property.canonical_key, buffer.data());
                }
                InvalidateShapes();
            }
        } else {
            ImGui::Text("%s", property.effective_value.c_str());
        }

        RenderStatusBadges(property.statuses);
        ImGui::TextDisabled("  Source: %s | Owner: %s",
                            property.source_key.empty() ? "(none)" : property.source_key.c_str(),
                            properties_truth::TruthOwnerName(property.owner));

        for (const auto& alias : property.aliases_present) {
            ImGui::TextDisabled("  Alias: %s=%s",
                                alias.key.c_str(),
                                alias.value.c_str());
        }

        if (!property.message.empty()) {
            ImGui::TextWrapped("  %s", property.message.c_str());
        }
        ImGui::PopID();
    }

    if (!report.raw_parameters.empty() && ImGui::TreeNode("Raw parameter mapping")) {
        std::vector<std::string> raw_keys_to_remove;
        for (const auto& raw : report.raw_parameters) {
            ImGui::PushID(raw.key.c_str());
            ImGui::Text("%s: %s", raw.key.c_str(), raw.value.c_str());
            RenderStatusBadges(raw.statuses);
            if (raw.cleanup_allowed) {
                ImGui::SameLine();
                if (ImGui::SmallButton("Remove")) {
                    raw_keys_to_remove.push_back(raw.key);
                }
                if (ImGui::IsItemHovered() && !raw.cleanup_reason.empty()) {
                    ImGui::BeginTooltip();
                    ImGui::TextWrapped("%s", raw.cleanup_reason.c_str());
                    ImGui::EndTooltip();
                }
            }
            if (!raw.maps_to.empty()) {
                ImGui::TextDisabled("  Maps to: %s", raw.maps_to.c_str());
            } else {
                ImGui::TextDisabled("  Not mapped by the current truth slice.");
            }
            if (raw.cleanup_allowed && !raw.cleanup_reason.empty()) {
                ImGui::TextDisabled("  Cleanup: %s", raw.cleanup_reason.c_str());
            }
            ImGui::PopID();
        }
        for (const auto& key : raw_keys_to_remove) {
            node.parameters.erase(key);
            InvalidateShapes();
        }
        ImGui::TreePop();
    }
}

void Properties::RenderParametersSection(MLNode& node, const cyxwiz::NodeMetadata* metadata) {
    ImGui::SetNextItemOpen(section_parameters_open_, ImGuiCond_Once);
    if (ImGui::CollapsingHeader("Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
        section_parameters_open_ = true;
        if (properties_contract::IsCustomSequencePropertiesNode(node.type)) {
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
