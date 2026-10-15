#include "node_editor.h"
#include "subgraph_presentation.h"
#include "subgraph_document.h"
#include "../core/graph_node_factory.h"
#include "../core/guided_sql.h"
#include "../core/node_metadata_registry.h"
#include "../core/preparation_recipe.h"
#include "../core/sql_step_contract.h"
#include "properties.h"
#include <imnodes.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <map>
#include <set>

namespace gui {
// ===== Subgraph Encapsulation =====

void NodeEditor::CreateSubgraphFromSelection(const std::string& name) {
    if (selected_node_ids_.size() < 2) {
        spdlog::warn("Need at least 2 selected nodes to create subgraph");
        return;
    }

    if (!CanReplaceGraph("create a subgraph")) return;
    for (int id : selected_node_ids_) {
        if (IsSubgraphNode(id) || IsSubgraphMember(id)) {
            spdlog::warn("Nested or overlapping subgraphs are not supported yet.");
            return;
        }
    }
    SaveUndoState();
    if (properties_panel_) properties_panel_->ClearNodeReferences();

    // Collect selected nodes and their internal links
    std::vector<MLNode> internal_nodes;
    std::vector<NodeLink> internal_links;
    std::set<int> selected_set(selected_node_ids_.begin(), selected_node_ids_.end());

    // Copy selected nodes to internal storage
    for (const auto& node : nodes_) {
        if (selected_set.count(node.id)) {
            internal_nodes.push_back(node);
        }
    }

    // Find internal links (both endpoints in selection)
    for (const auto& link : links_) {
        if (selected_set.count(link.from_node) && selected_set.count(link.to_node)) {
            internal_links.push_back(link);
        }
    }

    // Find boundary pins - inputs are pins with external sources, outputs have external destinations
    std::vector<std::pair<int, int>> input_pins;   // (node_id, pin_id) pairs
    std::vector<std::pair<int, int>> output_pins;  // (node_id, pin_id) pairs

    for (const auto& link : links_) {
        // Link from outside to inside -> input boundary
        if (!selected_set.count(link.from_node) && selected_set.count(link.to_node)) {
            input_pins.push_back({link.to_node, link.to_pin});
        }
        // Link from inside to outside -> output boundary
        if (selected_set.count(link.from_node) && !selected_set.count(link.to_node)) {
            output_pins.push_back({link.from_node, link.from_pin});
        }
    }

    // Calculate center position of selected nodes
    float center_x = 0, center_y = 0;
    int count = 0;
    for (int node_id : selected_node_ids_) {
        auto pos_it = cached_node_positions_.find(node_id);
        if (pos_it != cached_node_positions_.end()) {
            center_x += pos_it->second.x;
            center_y += pos_it->second.y;
            count++;
        }
    }
    if (count > 0) {
        center_x /= count;
        center_y /= count;
    }

    // Create the subgraph node
    MLNode subgraph_node{};
    subgraph_node.category = GetNodeCategoryForType(NodeType::Subgraph);
    subgraph_node.id = next_node_id_++;
    subgraph_node.type = NodeType::Subgraph;
    subgraph_node.name = name.empty() ? "Subgraph" : name;
    subgraph_node.parameters["node_count"] = std::to_string(internal_nodes.size());

    // Create input pins for boundary inputs
    for (size_t i = 0; i < input_pins.size(); ++i) {
        NodePin pin{};
        pin.id = next_pin_id_++;
        const auto* owner = FindNodeById(input_pins[i].first);
        for (const auto& original : owner->inputs) {
            if (original.id == input_pins[i].second) {
                const int id = pin.id;
                pin = original;
                pin.id = id;
                break;
            }
        }
        pin.is_input = true;
        subgraph_node.inputs.push_back(pin);
    }

    // Create output pins for boundary outputs
    for (size_t i = 0; i < output_pins.size(); ++i) {
        NodePin pin{};
        pin.id = next_pin_id_++;
        const auto* owner = FindNodeById(output_pins[i].first);
        for (const auto& original : owner->outputs) {
            if (original.id == output_pins[i].second) {
                const int id = pin.id;
                pin = original;
                pin.id = id;
                break;
            }
        }
        pin.is_input = false;
        subgraph_node.outputs.push_back(pin);
    }

    // Store subgraph data
    SubgraphData data;
    data.subgraph_node_id = subgraph_node.id;
    for (auto& internal : internal_nodes) {
        const auto pos = cached_node_positions_.find(internal.id);
        if (pos != cached_node_positions_.end()) {
            internal.initial_pos_x = pos->second.x;
            internal.initial_pos_y = pos->second.y;
            internal.has_initial_position = true;
        }
    }
    data.internal_nodes = std::move(internal_nodes);
    data.internal_links = std::move(internal_links);
    data.expanded = false;

    // Store pin mappings
    for (const auto& [node_id, pin_id] : input_pins) {
        data.input_pin_mappings.push_back(pin_id);
    }
    for (const auto& [node_id, pin_id] : output_pins) {
        data.output_pin_mappings.push_back(pin_id);
    }

    subgraphs_.push_back(std::move(data));

    // Rewire external connections to the subgraph node
    std::vector<NodeLink> links_to_add;
    std::vector<int> links_to_remove;

    for (size_t i = 0; i < links_.size(); ++i) {
        const auto& link = links_[i];

        // Link from outside to inside -> connect to subgraph input
        if (!selected_set.count(link.from_node) && selected_set.count(link.to_node)) {
            // Find which input pin this maps to
            for (size_t j = 0; j < input_pins.size(); ++j) {
                if (input_pins[j].second == link.to_pin) {
                    NodeLink new_link;
                    new_link.id = next_link_id_++;
                    new_link.from_node = link.from_node;
                    new_link.from_pin = link.from_pin;
                    new_link.to_node = subgraph_node.id;
                    new_link.to_pin = subgraph_node.inputs[j].id;
                    new_link.type = link.type;
                    links_to_add.push_back(new_link);
                    break;
                }
            }
            links_to_remove.push_back(static_cast<int>(i));
        }

        // Link from inside to outside -> connect from subgraph output
        if (selected_set.count(link.from_node) && !selected_set.count(link.to_node)) {
            // Find which output pin this maps to
            for (size_t j = 0; j < output_pins.size(); ++j) {
                if (output_pins[j].second == link.from_pin) {
                    NodeLink new_link;
                    new_link.id = next_link_id_++;
                    new_link.from_node = subgraph_node.id;
                    new_link.from_pin = subgraph_node.outputs[j].id;
                    new_link.to_node = link.to_node;
                    new_link.to_pin = link.to_pin;
                    new_link.type = link.type;
                    links_to_add.push_back(new_link);
                    break;
                }
            }
            links_to_remove.push_back(static_cast<int>(i));
        }

        // Internal links are removed from main graph
        if (selected_set.count(link.from_node) && selected_set.count(link.to_node)) {
            links_to_remove.push_back(static_cast<int>(i));
        }
    }

    // Remove old links (in reverse order to maintain indices)
    std::sort(links_to_remove.begin(), links_to_remove.end(), std::greater<int>());
    for (int idx : links_to_remove) {
        links_.erase(links_.begin() + idx);
    }

    // Add new links
    for (auto& link : links_to_add) {
        links_.push_back(link);
    }

    // Remove selected nodes from main graph
    nodes_.erase(
        std::remove_if(nodes_.begin(), nodes_.end(),
            [&selected_set](const MLNode& n) { return selected_set.count(n.id); }),
        nodes_.end()
    );

    // Add subgraph node
    nodes_.push_back(subgraph_node);

    // Position the subgraph node at the center of removed nodes
    pending_positions_[subgraph_node.id] = ImVec2(center_x, center_y);
    pending_positions_frames_ = 3;

    // Clear selection and select the new subgraph
    selected_node_ids_.clear();
    selected_node_ids_.push_back(subgraph_node.id);

    ImNodes::ClearNodeSelection();
    // SelectNode requires an existing ImNodes object, even before the first draw.
    ImNodes::SetNodeGridSpacePos(subgraph_node.id, ImVec2(center_x, center_y));
    ImNodes::SelectNode(subgraph_node.id);
    RebuildPinLookup();
    ClearValidationState();

    spdlog::info("Created subgraph '{}' with {} internal nodes",
                 subgraph_node.name, subgraphs_.back().internal_nodes.size());
}

void NodeEditor::ExpandSubgraph(int node_id) {
    if (!CanReplaceGraph("expand a subgraph")) return;
    SubgraphData* data = GetSubgraphData(node_id);
    if (!data) {
        spdlog::warn("Node {} is not a subgraph", node_id);
        return;
    }

    if (data->expanded) return;

    SaveUndoState();
    if (properties_panel_) properties_panel_->ClearNodeReferences();
    data->expanded = true;

    // Get position of subgraph node
    ImVec2 base_pos = ImVec2(0, 0);
    auto pos_it = cached_node_positions_.find(node_id);
    if (pos_it != cached_node_positions_.end()) {
        base_pos = pos_it->second;
    }

    if (auto* wrapper = FindNodeById(node_id)) {
        wrapper->initial_pos_x = base_pos.x;
        wrapper->initial_pos_y = base_pos.y;
        wrapper->has_initial_position = true;
    }
    ClearSelection();

    // Add internal nodes back to the main graph
    float offset_x = 0, offset_y = 50;
    for (auto& internal_node : data->internal_nodes) {
        // Offset position relative to subgraph node
        pending_positions_[internal_node.id] = internal_node.has_initial_position
            ? ImVec2(internal_node.initial_pos_x, internal_node.initial_pos_y)
            : ImVec2(base_pos.x + offset_x, base_pos.y + offset_y);
        nodes_.push_back(internal_node);
        offset_x += 180;
        if (offset_x > 500) {
            offset_x = 0;
            offset_y += 120;
        }
    }

    // Add internal links back
    for (const auto& link : data->internal_links) {
        links_.push_back(link);
    }

    pending_positions_frames_ = 3;
    RebuildPinLookup();
    ClearValidationState();
    spdlog::info("Expanded subgraph {} ({} internal nodes, {} internal links; shown in an expanded frame)", node_id, data->internal_nodes.size(), data->internal_links.size());
}

void NodeEditor::CollapseSubgraph(int node_id) {
    if (!CanReplaceGraph("collapse a subgraph")) return;
    SubgraphData* data = GetSubgraphData(node_id);
    if (!data) return;

    if (!data->expanded) return;

    try {
        auto captured = *data;
        detail::CaptureExpandedSubgraph(captured, nodes_, links_);
        for (auto& internal : captured.internal_nodes) {
            const auto pos = cached_node_positions_.find(internal.id);
            if (pos != cached_node_positions_.end()) {
                internal.initial_pos_x = pos->second.x;
                internal.initial_pos_y = pos->second.y;
                internal.has_initial_position = true;
            }
        }
        SaveUndoState();
        *data = std::move(captured);
    } catch (const std::exception& error) {
        spdlog::error("Cannot collapse subgraph: {}", error.what());
        return;
    }
    if (properties_panel_) properties_panel_->ClearNodeReferences();
    data->expanded = false;

    // Remove internal nodes from main graph
    std::set<int> internal_ids;
    for (const auto& node : data->internal_nodes) {
        internal_ids.insert(node.id);
    }

    nodes_.erase(
        std::remove_if(nodes_.begin(), nodes_.end(),
            [&internal_ids](const MLNode& n) { return internal_ids.count(n.id); }),
        nodes_.end()
    );

    // Remove internal links
    links_.erase(
        std::remove_if(links_.begin(), links_.end(),
            [&internal_ids](const NodeLink& l) {
                return internal_ids.count(l.from_node) && internal_ids.count(l.to_node);
            }),
        links_.end()
    );

    selected_node_ids_.clear();
    selected_node_id_ = -1;
    ImNodes::ClearNodeSelection();
    ImNodes::ClearLinkSelection();
    RebuildPinLookup();
    ClearValidationState();
    if (const auto* wrapper = FindNodeById(node_id)) {
        const ImVec2 position(wrapper->initial_pos_x, wrapper->initial_pos_y);
        pending_positions_[node_id] = position;
        pending_positions_frames_ = 3;
        ImNodes::SetNodeGridSpacePos(node_id, position);
    }
    spdlog::info("Collapsed subgraph {}", node_id);
}

void NodeEditor::ToggleSubgraphExpansion(int node_id) {
    SubgraphData* data = GetSubgraphData(node_id);
    if (!data) return;

    if (data->expanded) {
        CollapseSubgraph(node_id);
    } else {
        ExpandSubgraph(node_id);
    }
}

bool NodeEditor::IsPreparationRecipeNode(int node_id) const {
    const MLNode* node = FindNodeById(node_id);
    if (!node || node->type != NodeType::Subgraph) return false;
    const auto it = node->parameters.find(cyxwiz::kRecipeRoleParameter);
    return it != node->parameters.end() && it->second == cyxwiz::kPreparationRecipeRole;
}

bool NodeEditor::SetSubgraphPreparationRecipe(int node_id, bool recipe) {
    MLNode* wrapper = nullptr;
    for (auto& node : nodes_) if (node.id == node_id) wrapper = &node;
    SubgraphData* data = GetSubgraphData(node_id);
    if (!wrapper || !data || wrapper->type != NodeType::Subgraph) return false;
    if (!recipe) {
        SaveUndoState();
        wrapper->parameters.erase(cyxwiz::kRecipeRoleParameter);
        wrapper->parameters.erase(cyxwiz::kRecipeContractParameter);
        spdlog::info("Subgraph {} is now a visual group", node_id);
        return true;
    }
    // Validate against the saved form, exactly what will be lowered at run time.
    std::string why;
    try {
        nlohmann::json document;
        detail::WriteEditorGraphContent(document, nodes_, links_, subgraphs_, {});
        for (const auto& record : document.at("subgraphs")) {
            if (record.at("node_id").get<int>() != node_id) continue;
            for (const auto& node : document.at("nodes")) {
                if (node.at("id").get<int>() == node_id) {
                    why = cyxwiz::PreparationRecipeRejection(node, record);
                }
            }
        }
    } catch (const std::exception& e) {
        why = e.what();
    }
    if (!why.empty()) {
        spdlog::warn("Not a Preparation Recipe: {}", why);
        ShowPipelineNotice("This subgraph cannot be a Preparation Recipe.\n\n" + why);
        return false;
    }
    SaveUndoState();
    wrapper->parameters[cyxwiz::kRecipeRoleParameter] = cyxwiz::kPreparationRecipeRole;
    wrapper->parameters[cyxwiz::kRecipeContractParameter] =
        cyxwiz::kPreparationRecipeContractVersion;
    spdlog::info("Subgraph {} is now a Preparation Recipe", node_id);
    return true;
}

const MLNode* NodeEditor::FindStep(int node_id) const {
    if (const MLNode* node = FindNodeById(node_id)) return node;
    for (const auto& data : subgraphs_)
        for (const auto& internal : data.internal_nodes)
            if (internal.id == node_id) return &internal;
    return nullptr;
}

bool NodeEditor::SetStepParameters(int node_id, const std::map<std::string, std::string>& values,
                                   const std::vector<std::string>& erase) {
    MLNode* node = FindNodeById(node_id);
    for (auto it = subgraphs_.begin(); !node && it != subgraphs_.end(); ++it)
        for (auto& internal : it->internal_nodes)
            if (internal.id == node_id) node = &internal;
    if (!node) return false;
    SaveUndoState();
    for (const auto& [key, value] : values) node->parameters[key] = value;
    for (const auto& key : erase) node->parameters.erase(key);
    spdlog::debug("Set {} parameter(s) of node {} ({})", values.size() + erase.size(), node_id, node->name);
    return true;
}

NodeEditor::SubgraphSummary NodeEditor::GetSubgraphSummary(int node_id) const {
    SubgraphSummary summary;
    const MLNode* wrapper = FindNodeById(node_id);
    const SubgraphData* data = nullptr;
    for (const auto& candidate : subgraphs_) {
        if (candidate.subgraph_node_id == node_id) data = &candidate;
    }
    if (!wrapper || !data || wrapper->type != NodeType::Subgraph) return summary;
    summary.found = true;
    summary.expanded = data->expanded;
    summary.recipe = IsPreparationRecipeNode(node_id);
    if (const auto it = wrapper->parameters.find(cyxwiz::kRecipeContractParameter);
        it != wrapper->parameters.end()) {
        summary.recipe_contract = it->second;
    }
    summary.inputs = wrapper->inputs.size();
    summary.outputs = wrapper->outputs.size();

    // While expanded the steps live on the canvas and the stored copies are
    // stale until it collapses.
    std::vector<const MLNode*> members;
    std::map<int, int> indegree;
    for (const auto& internal : data->internal_nodes) {
        const MLNode* live = data->expanded ? FindNodeById(internal.id) : &internal;
        if (!live) continue;
        members.push_back(live);
        indegree[live->id] = 0;
    }
    std::map<int, std::vector<int>> next;
    for (const auto& link : data->expanded ? links_ : data->internal_links) {
        if (link.from_node == link.to_node || !indegree.count(link.from_node) || !indegree.count(link.to_node)) {
            continue;
        }
        next[link.from_node].push_back(link.to_node);
        ++indegree[link.to_node];
    }
    // Run order: the internal links sorted topologically, ties in stored order;
    // a cycle (rejected at run time) keeps the rest in stored order.
    std::vector<bool> placed(members.size(), false);
    while (summary.steps.size() < members.size()) {
        size_t pick = members.size();
        for (size_t i = 0; i < members.size(); ++i) {
            if (!placed[i] && indegree[members[i]->id] == 0) {
                pick = i;
                break;
            }
        }
        if (pick == members.size()) {
            pick = static_cast<size_t>(std::find(placed.begin(), placed.end(), false) - placed.begin());
            indegree[members[pick]->id] = 0;
        }
        placed[pick] = true;
        summary.steps.push_back({members[pick]->id, members[pick]->type, members[pick]->name});
        for (const int to : next[members[pick]->id]) --indegree[to];
    }
    return summary;
}

void NodeEditor::RequestSubgraphAction(int subgraph_id, SubgraphRequest request, int step_id) {
    pending_subgraph_request_ = {subgraph_id, request, step_id};
}

void NodeEditor::ApplySubgraphRequest() {
    if (pending_subgraph_request_.subgraph_id < 0) return;
    if (pending_subgraph_request_.frames_to_wait > 0) {
        --pending_subgraph_request_.frames_to_wait;
        return;
    }
    const auto pending = pending_subgraph_request_;
    pending_subgraph_request_ = {};
    const int id = pending.subgraph_id;
    // Expanding adds the steps to the canvas; select one only once ImNodes
    // has drawn it, or the selection is dropped.
    const auto expand_then_show = [&](int step_id, SubgraphRequest then) {
        ExpandSubgraph(id);
        if (step_id >= 0) pending_subgraph_request_ = {id, then, step_id, 2};
    };
    switch (pending.request) {
        case SubgraphRequest::Expand: {
            const auto summary = GetSubgraphSummary(id);
            expand_then_show(summary.steps.empty() ? -1 : summary.steps.front().id, SubgraphRequest::ShowStep);
            break;
        }
        case SubgraphRequest::Collapse:
            CollapseSubgraph(id);
            FocusNode(id);
            break;
        case SubgraphRequest::MakeRecipe:
            SetSubgraphPreparationRecipe(id, true);
            break;
        case SubgraphRequest::MakeGroup:
            SetSubgraphPreparationRecipe(id, false);
            break;
        case SubgraphRequest::ShowStep:
        case SubgraphRequest::OpenStep:
            if (const SubgraphData* data = GetSubgraphData(id); data && !data->expanded) {
                expand_then_show(pending.step_id, pending.request);
            } else {
                FocusNode(pending.step_id);
                if (pending.request == SubgraphRequest::OpenStep) ConfigureNode(pending.step_id);
            }
            break;
    }
}

void NodeEditor::ApplyStepAfterInsert() {
    if (deferred_focus_.node_id >= 0) {
        if (deferred_focus_.frames > 0) {
            --deferred_focus_.frames;
        } else {
            FocusNode(deferred_focus_.node_id);
            deferred_focus_ = {};
        }
    }
    if (pending_steps_after_.empty()) return;
    // A recipe being expanded first (Data Studio): its steps join the canvas then.
    if (pending_subgraph_request_.subgraph_id >= 0) return;
    const auto requests = std::move(pending_steps_after_);
    pending_steps_after_.clear();
    for (const auto& request : requests) InsertStepAfter(request);
}

void NodeEditor::InsertStepAfter(const PendingStepAfter& request) {
    const int after_id = request.after;
    const MLNode* after = FindNodeById(after_id);
    if (!after || after->outputs.empty()) return;
    const bool sql = request.type == NodeType::SQLQuery;
    std::string kind = GetNodeTypeName(request.type);
    if (const auto* meta = cyxwiz::NodeMetadataRegistry::Instance().GetMetadata(request.type)) kind = meta->name;
    const std::string operation = sql ? (request.guided ? "add a guided step" : "add an SQL step") : "add a " + kind + " step";
    if (!CanReplaceGraph(operation.c_str())) return;

    SaveUndoState();
    if (properties_panel_) properties_panel_->ClearNodeReferences();
    const int after_out = after->outputs.front().id;
    // To the step's right, as far as the node after it already is (the
    // graph's own spacing); the nodes after it move right by as much, so the
    // links stay straight.
    const auto place = [&](int id, const MLNode& node) {
        // A place set this frame (an earlier insert moved it) comes first.
        if (const auto pending = pending_positions_.find(id); pending != pending_positions_.end()) return pending->second;
        const auto cached = cached_node_positions_.find(id);
        return cached != cached_node_positions_.end() ? cached->second : ImVec2(node.initial_pos_x, node.initial_pos_y);
    };
    const ImVec2 at = place(after_id, *after);
    float shift = 0.0f;
    for (const auto& link : links_) {
        const MLNode* next = link.from_node == after_id ? FindNodeById(link.to_node) : nullptr;
        if (!next) continue;
        const float gap = place(next->id, *next).x - at.x;
        if (gap > 0.0f && (shift == 0.0f || gap < shift)) shift = gap;
    }
    if (shift <= 0.0f) {
        // Nothing to its right: the step's width and a gap.
        const float width = cached_node_positions_.count(after_id) ? ImNodes::GetNodeDimensions(after_id).x / zoom_ : 120.0f;
        shift = width + 100.0f;
    }
    shift = std::clamp(shift, 120.0f, 600.0f);
    const ImVec2 position(at.x + shift, at.y);
    std::set<int> downstream;
    std::vector<int> frontier{after_id};
    while (!frontier.empty()) {
        const int from = frontier.back();
        frontier.pop_back();
        for (const auto& link : links_)
            if (link.from_node == from && link.to_node != after_id && downstream.insert(link.to_node).second) frontier.push_back(link.to_node);
    }
    for (const int id : downstream) {
        const MLNode* node = FindNodeById(id);
        if (!node) continue;
        const ImVec2 from = place(id, *node);
        pending_positions_[id] = ImVec2(from.x + shift, from.y);
    }

    // An empty guided step passes the rows through until operations are
    // added; an SQL step runs the query it was given; other nodes start with
    // their defaults.
    MLNode step = CreateNode(request.type, !request.name.empty() ? request.name
                                           : !sql                ? kind
                                           : request.guided      ? after->name + " (guided)"
                                                                 : after->name + " (query)");
    if (sql) {
        if (request.guided) {
            const cyxwiz::guided::GuidedStep empty;
            step.parameters[cyxwiz::guided::kGuidedOpsParameter] = cyxwiz::guided::ToJson(empty);
            step.parameters["query"] = cyxwiz::guided::GenerateSql(empty, {}).sql;
        } else {
            step.parameters["query"] = request.query;
        }
        step.parameters["input_alias"] = "input";
        step.parameters[cyxwiz::kSqlContractParameter] = cyxwiz::kSqlContractVersion;
    }
    for (const auto& [key, value] : request.parameters) step.parameters[key] = value;
    step.initial_pos_x = position.x;
    step.initial_pos_y = position.y;
    step.has_initial_position = true;
    if (step.inputs.empty() || step.outputs.empty()) {
        spdlog::error("A new {} node has no input or output pin; nothing added", kind);
        return;
    }
    const int step_in = step.inputs.front().id;
    const int step_out = step.outputs.front().id;

    // The new step takes over what read the old step's result.
    for (auto& link : links_) {
        if (link.from_node == after_id && link.from_pin == after_out) {
            link.from_node = step.id;
            link.from_pin = step_out;
        }
    }
    links_.push_back({next_link_id_++, after_id, after_out, step.id, step_in, LinkType::TensorFlow});
    // Inside a recipe it becomes a step of that recipe, and its output when
    // the old step was.
    for (auto& data : subgraphs_) {
        const bool member = std::any_of(data.internal_nodes.begin(), data.internal_nodes.end(),
                                        [&](const MLNode& internal) { return internal.id == after_id; });
        if (!member) continue;
        data.internal_nodes.push_back(step);
        for (int& pin : data.output_pin_mappings) {
            if (pin == after_out) pin = step_out;
        }
    }
    const int step_id = step.id;
    nodes_.push_back(std::move(step));
    pending_positions_[step_id] = position;
    pending_positions_frames_ = 3;
    RebuildPinLookup();
    ClearValidationState();
    deferred_focus_ = {step_id, 2};
    spdlog::info("Added {} node {} after node {} ({} moved right)", kind, step_id, after_id, downstream.size());
}

bool NodeEditor::IsSubgraphNode(int node_id) const {
    const MLNode* node = FindNodeById(node_id);
    return node && node->type == NodeType::Subgraph;
}

SubgraphData* NodeEditor::GetSubgraphData(int node_id) {
    for (auto& data : subgraphs_) {
        if (data.subgraph_node_id == node_id) {
            return &data;
        }
    }
    return nullptr;
}

bool NodeEditor::IsSubgraphMember(int node_id) const {
    for (const auto& data : subgraphs_)
        for (const auto& node : data.internal_nodes)
            if (node.id == node_id) return true;
    return false;
}

}
