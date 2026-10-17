/**
 * Node Editor - Context Menu
 * Handles the right-click context menu for adding nodes
 */

#include "node_editor.h"
#include "../core/node_palette.h"
#include "icons.h"
#include "properties.h"
#include "../core/node_metadata.h"
#include "../core/node_metadata_registry.h"
#include "../core/project_manager.h"
#include "../core/extension_node_registry.h"
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
    // The canvas comes forward and shows the node; the dialog opened below
    // asks for the front after it, so it ends up in front of the canvas.
    pending_focus_node_id_ = node_id;
    Raise();

    // A Plot node opens its Plot window (TOFIX134 P2).
    if (node->type == NodeType::Plot) {
        OpenPlotNode(node_id);
        return;
    }
    if (node->type == NodeType::Dashboard) {
        OpenDashboardNode(node_id);
        return;
    }

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

    if (node->type == NodeType::SQLQuery) {
        if (ImGui::MenuItem(ICON_FA_PLUS " Add a guided step after")) {
            RequestGuidedStepAfter(right_clicked_node_id_);
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("A new step built from operations (no SQL) that reads this step's result; whatever "
                              "read this step now reads the new one.");
        }
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

    // Group nodes by category label (from searchable list). Palette curation
    // (TOFIX157, core/node_palette): table steps go under Advanced operations
    // by their group, nodes that cannot run yet under Can't run yet.
    std::map<std::string, std::vector<SearchableNode*>> grouped;
    std::map<std::string, std::vector<SearchableNode*>> advanced;
    std::vector<SearchableNode*> cant_run;
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
        if (node.cant_run) cant_run.push_back(&node);
        else if (!node.advanced_group.empty()) advanced[node.advanced_group].push_back(&node);
        else grouped[node.category].push_back(&node);
    }

    // One node as a menu item: added on click, or disabled with its reason.
    const auto item = [&](SearchableNode* node, const std::string& label) {
        const bool is_unavailable =
            node->status == NodeImplementationStatus::Template || node->support_blocked || node->cant_run;
        if (is_unavailable) ImGui::BeginDisabled();
        if (ImGui::MenuItem(label.c_str())) {
            if (!is_unavailable && node->type != NodeType::Unknown) {
                AddNode(node->type, node->type == NodeType::PluginCustom ? node->extension_type_id : node->name);
                context_menu_search_[0] = '\0';
                ImGui::CloseCurrentPopup();
            }
        }
        if (is_unavailable) {
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled) && !node->tooltip.empty()) {
                ImGui::SetTooltip("%s", node->tooltip.c_str());
            }
            ImGui::EndDisabled();
        }
    };

    if (grouped.empty() && advanced.empty() && cant_run.empty()) {
        ImGui::TextDisabled("No matches");
    } else if (is_searching) {
        // When searching, show a flat list.
        for (auto& [category, nodes] : grouped) {
            ImGui::TextDisabled("%s", category.c_str());
            for (auto* node : nodes) item(node, node->name);
        }
        for (const auto& group : cyxwiz::palette::AdvancedGroups()) {
            const auto found = advanced.find(group);
            if (found == advanced.end()) continue;
            ImGui::TextDisabled("Advanced operation \xC2\xB7 %s", group.c_str());
            for (auto* node : found->second) item(node, node->name);
        }
        if (!cant_run.empty()) {
            ImGui::TextDisabled("%s", cyxwiz::palette::kCantRunYet);
            for (auto* node : cant_run) item(node, node->name + "  (" + node->category + ")");
        }
    } else {
        // Use submenus with > indicator for categories
        for (auto& [category, nodes] : grouped) {
            if (ImGui::BeginMenu(category.c_str())) {
                for (auto* node : nodes) item(node, node->name);
                ImGui::EndMenu();
            }
        }
        if (!advanced.empty() || !cant_run.empty()) ImGui::Separator();
        if (!advanced.empty() && ImGui::BeginMenu(cyxwiz::palette::kAdvancedOperations)) {
            for (const auto& group : cyxwiz::palette::AdvancedGroups()) {
                const auto found = advanced.find(group);
                if (found == advanced.end()) continue;
                if (ImGui::BeginMenu(group.c_str())) {
                    for (auto* node : found->second) item(node, node->name);
                    ImGui::EndMenu();
                }
            }
            ImGui::EndMenu();
        }
        if (!cant_run.empty() && ImGui::BeginMenu(cyxwiz::palette::kCantRunYet)) {
            std::map<std::string, std::vector<SearchableNode*>> by_category;
            for (auto* node : cant_run) by_category[node->category].push_back(node);
            for (auto& [category, nodes] : by_category) {
                if (ImGui::BeginMenu(category.c_str())) {
                    for (auto* node : nodes) item(node, node->name);
                    ImGui::EndMenu();
                }
            }
            ImGui::EndMenu();
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

        if (ImGui::MenuItem(ICON_FA_COMPRESS " Create Subgraph", "Ctrl+Shift+U", false, selected_node_ids_.size() >= 2)) {
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
                if (IsPreparationRecipeNode(selected_node_ids_[0])) {
                    if (ImGui::MenuItem(ICON_FA_OBJECT_GROUP " Make Visual Group")) {
                        SetSubgraphPreparationRecipe(selected_node_ids_[0], false);
                        ImGui::CloseCurrentPopup();
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::BeginTooltip();
                        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 28.0f);
                        ImGui::TextUnformatted(
                            "Turn this Preparation Recipe back into a plain visual group.\n\n"
                            "A visual group only tidies the canvas: it does not run. To run "
                            "its steps again, expand it or make it a Preparation Recipe again.");
                        ImGui::PopTextWrapPos();
                        ImGui::EndTooltip();
                    }
                } else {
                    if (ImGui::MenuItem(ICON_FA_GEARS " Make Preparation Recipe")) {
                        SetSubgraphPreparationRecipe(selected_node_ids_[0], true);
                        ImGui::CloseCurrentPopup();
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::BeginTooltip();
                        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 28.0f);
                        ImGui::TextUnformatted(
                            "Turn this subgraph into one runnable preparation stage.\n\n"
                            "When the pipeline runs, the recipe's steps run in order: one "
                            "dataset goes in, one prepared dataset comes out, and the result "
                            "is the same as running those steps as separate nodes.\n\n"
                            "Allowed steps: row and column operations, text cleaning, "
                            "group-by, joins, SQL steps and checks. Sources, exports, splits, "
                            "tokenizers and models stay outside the recipe.\n\n"
                            "The subgraph is checked first; if it does not qualify, you are "
                            "told why and nothing changes.");
                        ImGui::PopTextWrapPos();
                        ImGui::EndTooltip();
                    }
                }
            }
        }

        ImGui::EndMenu();
    }
}

} // namespace gui
