/**
 * Node Editor - Context Menu
 * Handles the right-click context menu for adding nodes
 */

#include "node_editor.h"
#include "icons.h"
#include "properties.h"
#include "../core/node_metadata.h"
#include "../core/node_metadata_registry.h"
#include "../core/project_manager.h"
#include "../plugin/registries/plugin_node_registry.h"
#include <imgui.h>
#include <algorithm>
#include <cstring>

namespace gui {

void NodeEditor::ConfigureNode(int node_id) {
    MLNode* node = FindNodeById(node_id);
    if (!node) {
        return;
    }

    selected_node_id_ = node_id;
    selected_node_ids_.clear();
    selected_node_ids_.push_back(node_id);
    pending_focus_node_id_ = node_id;

    if (properties_panel_) {
        properties_panel_->ConfigureNode(node);
    }
}

void NodeEditor::ShowSingleNodeContextMenu() {
    MLNode* node = FindNodeById(right_clicked_node_id_);
    if (!node) {
        ImGui::Text("Node not found");
        return;
    }

    ImGui::TextDisabled("%s", node->name.c_str());
    ImGui::Separator();

    // Set Description (KNIME-style)
    if (ImGui::MenuItem(ICON_FA_COMMENT " Set Description...")) {
        editing_node_description_ = true;
        editing_node_id_ = right_clicked_node_id_;
        strncpy(node_description_buffer_, node->description.c_str(), sizeof(node_description_buffer_) - 1);
        node_description_buffer_[sizeof(node_description_buffer_) - 1] = '\0';
        ImGui::CloseCurrentPopup();
    }

    // Rename
    if (ImGui::MenuItem(ICON_FA_PEN " Rename...")) {
        // TODO: Implement rename dialog
        ImGui::CloseCurrentPopup();
    }

    ImGui::Separator();

    // Duplicate
    if (ImGui::MenuItem(ICON_FA_COPY " Duplicate", "Ctrl+D")) {
        // Select only this node and duplicate
        selected_node_ids_.clear();
        selected_node_ids_.push_back(right_clicked_node_id_);
        DuplicateSelection();
        ImGui::CloseCurrentPopup();
    }

    // Delete
    if (ImGui::MenuItem(ICON_FA_TRASH " Delete", "Delete")) {
        DeleteNode(right_clicked_node_id_);
        ImGui::CloseCurrentPopup();
    }

    ImGui::Separator();

    // Configure (open properties)
    if (ImGui::MenuItem(ICON_FA_GEAR " Configure...")) {
        ConfigureNode(right_clicked_node_id_);
        ImGui::CloseCurrentPopup();
    }

    if (explain_node_callback_ &&
        ImGui::MenuItem(ICON_FA_BUG " Explain in Studio Debugger")) {
        explain_node_callback_(right_clicked_node_id_);
        ImGui::CloseCurrentPopup();
    }

    // Group operations
    NodeGroup* existing_group = FindGroupContainingNode(right_clicked_node_id_);
    if (existing_group) {
        if (ImGui::MenuItem(ICON_FA_OBJECT_UNGROUP " Remove from Group")) {
            existing_group->node_ids.erase(
                std::remove(existing_group->node_ids.begin(), existing_group->node_ids.end(), right_clicked_node_id_),
                existing_group->node_ids.end());
            ImGui::CloseCurrentPopup();
        }
    }
}

void NodeEditor::ShowNodeDescriptionEditPopup() {
    if (!editing_node_description_) return;

    ImGui::OpenPopup("Edit Node Description");

    // Center the modal
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(450, 250), ImGuiCond_Appearing);

    if (ImGui::BeginPopupModal("Edit Node Description", &editing_node_description_, ImGuiWindowFlags_AlwaysAutoResize)) {
        MLNode* node = FindNodeById(editing_node_id_);
        if (node) {
            ImGui::Text("Node: %s", node->name.c_str());
            ImGui::Separator();

            ImGui::Text("Description:");
            ImGui::PushItemWidth(-1);
            ImGui::InputTextMultiline("##description", node_description_buffer_, sizeof(node_description_buffer_),
                ImVec2(-1, 120), ImGuiInputTextFlags_AllowTabInput);
            ImGui::PopItemWidth();

            ImGui::TextDisabled("Tip: This text will appear below the node on the canvas.");

            ImGui::Separator();

            if (ImGui::Button("Save", ImVec2(120, 0))) {
                node->description = node_description_buffer_;
                editing_node_description_ = false;
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel", ImVec2(120, 0))) {
                editing_node_description_ = false;
                ImGui::CloseCurrentPopup();
            }
        } else {
            ImGui::Text("Node not found");
            if (ImGui::Button("Close")) {
                editing_node_description_ = false;
                ImGui::CloseCurrentPopup();
            }
        }
        ImGui::EndPopup();
    }
}

void NodeEditor::ShowContextMenu() {
    InitializeSearchableNodes();

    ImGui::TextDisabled("Add Node");
    ImGui::SetNextItemWidth(-1);
    ImGui::InputTextWithHint("##ctx_search",
                             ICON_FA_MAGNIFYING_GLASS " Search nodes...",
                             context_menu_search_, sizeof(context_menu_search_));
    ImGui::Separator();

    std::string search_lower = context_menu_search_;
    std::transform(search_lower.begin(), search_lower.end(), search_lower.begin(), ::tolower);
    const bool is_searching = !search_lower.empty();

    // Group nodes by category label (from searchable list)
    std::map<std::string, std::vector<SearchableNode*>> grouped;
    for (auto& node : all_searchable_nodes_) {
        if (node.status == NodeImplementationStatus::Deprecated) {
            continue;
        }
        if (is_searching) {
            std::string name_lower = node.name;
            std::string cat_lower = node.category;
            std::string key_lower = node.keywords;
            std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
            std::transform(cat_lower.begin(), cat_lower.end(), cat_lower.begin(), ::tolower);
            std::transform(key_lower.begin(), key_lower.end(), key_lower.begin(), ::tolower);
            if (name_lower.find(search_lower) == std::string::npos &&
                cat_lower.find(search_lower) == std::string::npos &&
                key_lower.find(search_lower) == std::string::npos) {
                continue;
            }
        }
        grouped[node.category].push_back(&node);
    }

    if (grouped.empty()) {
        ImGui::TextDisabled("No matches");
    } else {
        // When searching, show flat list; otherwise use submenus
        if (is_searching) {
            for (auto& [category, nodes] : grouped) {
                ImGui::TextDisabled("%s", category.c_str());
                for (auto* node : nodes) {
                    const bool is_template = (node->status == NodeImplementationStatus::Template);
                    if (is_template) ImGui::BeginDisabled();
                    if (ImGui::MenuItem(node->name.c_str())) {
                        if (!is_template && node->type != NodeType::Unknown) {
                            AddNode(node->type, node->name);
                            context_menu_search_[0] = '\0';
                            ImGui::CloseCurrentPopup();
                        }
                    }
                    if (is_template) {
                        if (ImGui::IsItemHovered() && !node->tooltip.empty()) {
                            ImGui::SetTooltip("%s", node->tooltip.c_str());
                        }
                        ImGui::EndDisabled();
                    }
                }
            }
        } else {
            // Use submenus with > indicator for categories
            for (auto& [category, nodes] : grouped) {
                if (ImGui::BeginMenu(category.c_str())) {
                    for (auto* node : nodes) {
                        const bool is_template = (node->status == NodeImplementationStatus::Template);
                        if (is_template) ImGui::BeginDisabled();
                        if (ImGui::MenuItem(node->name.c_str())) {
                            if (!is_template && node->type != NodeType::Unknown) {
                                AddNode(node->type, node->name);
                                context_menu_search_[0] = '\0';
                                ImGui::CloseCurrentPopup();
                            }
                        }
                        if (is_template) {
                            if (ImGui::IsItemHovered() && !node->tooltip.empty()) {
                                ImGui::SetTooltip("%s", node->tooltip.c_str());
                            }
                            ImGui::EndDisabled();
                        }
                    }
                    ImGui::EndMenu();
                }
            }
        }
    }

    ImGui::Separator();
    if (ImGui::BeginMenu("Actions")) {
        const bool has_selection = !selected_node_ids_.empty();

        if (ImGui::MenuItem(ICON_FA_PASTE " Paste", "Ctrl+V", false, clipboard_.valid)) {
            PasteClipboard();
            ImGui::CloseCurrentPopup();
        }

        if (ImGui::BeginMenu("Align / Distribute", has_selection)) {
            ImGui::TextDisabled("Align");
            if (ImGui::MenuItem("Align Left", nullptr, false, selected_node_ids_.size() >= 2)) {
                AlignSelectedNodes(AlignmentType::Left);
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("Align Center", nullptr, false, selected_node_ids_.size() >= 2)) {
                AlignSelectedNodes(AlignmentType::Center);
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("Align Right", nullptr, false, selected_node_ids_.size() >= 2)) {
                AlignSelectedNodes(AlignmentType::Right);
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("Align Top", nullptr, false, selected_node_ids_.size() >= 2)) {
                AlignSelectedNodes(AlignmentType::Top);
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("Align Middle", nullptr, false, selected_node_ids_.size() >= 2)) {
                AlignSelectedNodes(AlignmentType::Middle);
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("Align Bottom", nullptr, false, selected_node_ids_.size() >= 2)) {
                AlignSelectedNodes(AlignmentType::Bottom);
                ImGui::CloseCurrentPopup();
            }

            ImGui::Separator();
            ImGui::TextDisabled("Distribute");
            if (ImGui::MenuItem("Distribute Horizontally", nullptr, false, selected_node_ids_.size() >= 3)) {
                DistributeSelectedNodes(DistributeType::Horizontal);
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("Distribute Vertically", nullptr, false, selected_node_ids_.size() >= 3)) {
                DistributeSelectedNodes(DistributeType::Vertical);
                ImGui::CloseCurrentPopup();
            }

            ImGui::Separator();
            if (ImGui::MenuItem("Auto Layout (Grid)", nullptr, false, has_selection)) {
                AutoLayoutSelection();
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndMenu();
        }

        ImGui::Separator();
        if (ImGui::MenuItem(ICON_FA_OBJECT_GROUP " Create Group", "Ctrl+G", false, has_selection)) {
            CreateGroupFromSelection("");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem(ICON_FA_OBJECT_UNGROUP " Ungroup", "Ctrl+Shift+G", false, has_selection)) {
            UngroupSelection();
            ImGui::CloseCurrentPopup();
        }

        if (ImGui::MenuItem(ICON_FA_COMPRESS " Create Subgraph", "Ctrl+Shift+S", false, selected_node_ids_.size() >= 2)) {
            CreateSubgraphFromSelection("");
            ImGui::CloseCurrentPopup();
        }

        if (selected_node_ids_.size() == 1 && IsSubgraphNode(selected_node_ids_[0])) {
            SubgraphData* data = GetSubgraphData(selected_node_ids_[0]);
            if (data) {
                if (data->expanded) {
                    if (ImGui::MenuItem(ICON_FA_COMPRESS " Collapse Subgraph")) {
                        CollapseSubgraph(selected_node_ids_[0]);
                        ImGui::CloseCurrentPopup();
                    }
                } else {
                    if (ImGui::MenuItem(ICON_FA_EXPAND " Expand Subgraph")) {
                        ExpandSubgraph(selected_node_ids_[0]);
                        ImGui::CloseCurrentPopup();
                    }
                }
            }
        }

        ImGui::EndMenu();
    }
}

// ============================================================================
// Unified Canvas Phase 3: Categorized Node Palette
// ============================================================================

const char* NodeEditor::GetCategoryIcon(NodeCategory category) {
    if (category == NodeCategory::DataSources) return ICON_FA_DATABASE;
    if (category == NodeCategory::DataTransform) return ICON_FA_FILTER;
    if (category == NodeCategory::Analytics) return ICON_FA_CHART_LINE;
    if (category == NodeCategory::Preprocessing) return ICON_FA_WAND_MAGIC_SPARKLES;
    if (category == NodeCategory::Layers) return ICON_FA_LAYER_GROUP;
    if (category == NodeCategory::Activation) return ICON_FA_BOLT;
    if (category == NodeCategory::Pooling) return ICON_FA_COMPRESS;
    if (category == NodeCategory::Normalization) return ICON_FA_SCALE_BALANCED;
    if (category == NodeCategory::Attention) return ICON_FA_EYE;
    if (category == NodeCategory::Recurrent) return ICON_FA_ROTATE;
    if (category == NodeCategory::ShapeOps) return ICON_FA_OBJECT_GROUP;  // was ICON_FA_SHAPES
    if (category == NodeCategory::MergeOps) return ICON_FA_RIGHT_LEFT;    // was ICON_FA_CODE_MERGE
    if (category == NodeCategory::Training) return ICON_FA_GRADUATION_CAP;
    if (category == NodeCategory::Regularization) return ICON_FA_SCALE_BALANCED;  // was ICON_FA_SHIELD_HALVED
    if (category == NodeCategory::Utility) return ICON_FA_COG;            // was ICON_FA_SCREWDRIVER_WRENCH
    if (category == NodeCategory::Signal) return ICON_FA_ARROWS_ROTATE; // was ICON_FA_WAVE_SQUARE
    if (category == NodeCategory::DataPipeline) return ICON_FA_DIAGRAM_PROJECT;
    if (category == NodeCategory::DNN) return ICON_FA_LIGHTBULB;      // was ICON_FA_BRAIN
    if (category == NodeCategory::TextProcessing) return ICON_FA_FILE_LINES;
    if (category == NodeCategory::Upsampling) return ICON_FA_EXPAND;         // was ICON_FA_UP_RIGHT_AND_DOWN_LEFT_FROM_CENTER
    if (category == NodeCategory::TimeSeries) return ICON_FA_ARROW_TREND_UP; // was ICON_FA_CHART_AREA
    if (category == NodeCategory::Audio) return ICON_FA_STETHOSCOPE;    // was ICON_FA_VOLUME_HIGH
    if (category == NodeCategory::RL) return ICON_FA_GRADUATION_CAP; // was ICON_FA_ROBOT
    if (category == NodeCategory::Plugin) return ICON_FA_DIAGRAM_PROJECT; // was ICON_FA_PUZZLE_PIECE
    return ICON_FA_CUBE;
}

const char* NodeEditor::GetCategoryName(NodeCategory category) {
    if (category == NodeCategory::DataSources) return "Data Sources";
    if (category == NodeCategory::DataTransform) return "Data Transform";
    if (category == NodeCategory::Analytics) return "Analytics";
    if (category == NodeCategory::Preprocessing) return "Preprocessing";
    if (category == NodeCategory::Layers) return "Layers";
    if (category == NodeCategory::Activation) return "Activation";
    if (category == NodeCategory::Pooling) return "Pooling";
    if (category == NodeCategory::Normalization) return "Normalization";
    if (category == NodeCategory::Attention) return "Attention";
    if (category == NodeCategory::Recurrent) return "Recurrent";
    if (category == NodeCategory::ShapeOps) return "Shape Operations";
    if (category == NodeCategory::MergeOps) return "Merge Operations";
    if (category == NodeCategory::Training) return "Training";
    if (category == NodeCategory::Regularization) return "Regularization";
    if (category == NodeCategory::Utility) return "Utility";
    if (category == NodeCategory::Signal) return "Signal / Control";
    if (category == NodeCategory::DataPipeline) return "Data Pipeline";
    if (category == NodeCategory::DNN) return "DNN / Pre-trained";
    if (category == NodeCategory::TextProcessing) return "Text Processing";
    if (category == NodeCategory::Upsampling) return "Upsampling";
    if (category == NodeCategory::TimeSeries) return "Time Series";
    if (category == NodeCategory::Audio) return "Audio";
    if (category == NodeCategory::RL) return "Reinforcement Learning";
    if (category == NodeCategory::Plugin) return "Plugin Nodes";
    return "Unknown";
}

void NodeEditor::ShowCategorizedNodeMenu() {
    // Initialize nodes_by_category_ on first call. As of 2026-04-17 this
    // is driven by NodeMetadataRegistry — the same source the Node
    // Browser and the search palette use — so adding a new node type
    // requires only one RegisterNode() call. Before this change, the
    // context menu had its own ~100-line hand-maintained list that
    // silently omitted most of the enum (Pooling, Normalization,
    // Recurrent, Training, Attention, etc.) so users couldn't add those
    // node types from the right-click menu at all.
    //
    // See docs/plans/node_registration_unification.md.
    if (!nodes_by_category_initialized_) {
        auto& reg = cyxwiz::NodeMetadataRegistry::Instance();
        if (!reg.IsInitialized()) reg.Initialize();

        for (const auto& cat : reg.GetCategories()) {
            auto entries = reg.GetByCategory(cat, /*include_templates=*/false);
            if (entries.empty()) continue;
            auto& bucket = nodes_by_category_[cat];
            bucket.reserve(entries.size());
            for (const auto* meta : entries) {
                if (!meta) continue;
                bucket.push_back({meta->type, meta->name});
            }
        }

        nodes_by_category_initialized_ = true;
    }

    // Search filter
    ImGui::SetNextItemWidth(-1);
    ImGui::InputTextWithHint("##search", ICON_FA_MAGNIFYING_GLASS " Search nodes...", context_menu_search_, sizeof(context_menu_search_));
    ImGui::Separator();

    // Filter nodes based on search
    std::string search_lower = context_menu_search_;
    std::transform(search_lower.begin(), search_lower.end(), search_lower.begin(), ::tolower);
    bool is_searching = strlen(context_menu_search_) > 0;

    // Render categories
    for (auto& [category, nodes] : nodes_by_category_) {
        // Skip empty categories
        if (nodes.empty()) continue;

        // If searching, filter nodes
        std::vector<std::pair<NodeType, std::string>> filtered_nodes;
        if (is_searching) {
            for (auto& [type, name] : nodes) {
                std::string name_lower = name;
                std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
                if (name_lower.find(search_lower) != std::string::npos) {
                    filtered_nodes.push_back({type, name});
                }
            }
            // Skip category if no matches
            if (filtered_nodes.empty()) continue;
        } else {
            filtered_nodes = nodes;
        }

        RenderNodeCategory(category, GetCategoryName(category), GetCategoryIcon(category));
    }
}

void NodeEditor::RenderNodeCategory(NodeCategory category, const char* category_name, const char* icon) {
    auto it = nodes_by_category_.find(category);
    if (it == nodes_by_category_.end()) return;

    // Build menu label with icon
    std::string menu_label = std::string(icon) + " " + category_name;

    // Use submenu with > indicator
    if (ImGui::BeginMenu(menu_label.c_str())) {
        for (auto& [type, name] : it->second) {
            // Filter by search
            if (strlen(context_menu_search_) > 0) {
                std::string name_lower = name;
                std::string search_lower = context_menu_search_;
                std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
                std::transform(search_lower.begin(), search_lower.end(), search_lower.begin(), ::tolower);
                if (name_lower.find(search_lower) == std::string::npos) {
                    continue;
                }
            }

            if (ImGui::MenuItem(name.c_str())) {
                AddNode(type, name);
                context_menu_search_[0] = '\0';  // Clear search on selection
                ImGui::CloseCurrentPopup();
            }
        }
        ImGui::EndMenu();
    }
}

} // namespace gui
