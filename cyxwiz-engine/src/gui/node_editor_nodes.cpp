#include "node_editor.h"
#include "ui_tokens.h"
#include "../core/graph_node_factory.h"
#include "properties.h"
#include "../core/node_metadata_registry.h"
#include "../core/extension_node_registry.h"
#include "../core/data_registry.h"
#include "../core/pipeline_executor.h"
#include "plot/plot_node_lane.h"
#include "loaders/data_loader.h"
#include <imgui.h>
#include <imnodes.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <optional>

namespace gui {

#ifndef CYXWIZ_NODE_FACTORY_ONLY
void NodeEditor::AddNode(NodeType type, const std::string& name) {
    if (!CanAddNodeToGraph(type, name)) {
        spdlog::warn("Blocked graph add for unsupported node '{}' (type={})",
                     name, static_cast<int>(type));
        return;
    }

    // Queue the node for deferred addition (after ImNodes::EndNodeEditor())
    pending_nodes_.push_back({type, name, context_menu_pos_});
    ClearValidationState();  // Graph changed — stale compile results
    spdlog::info("Queued node for addition: type={}, name={} at position x={} y={}",
                 static_cast<int>(type), name, context_menu_pos_.x, context_menu_pos_.y);
}

bool NodeEditor::CanAddNodeToGraph(NodeType type, const std::string& name) const {
    // Extension nodes share one NodeType; the catalog's single PluginCustom
    // entry is a preview and says nothing about a particular extension.
    if (type == NodeType::PluginCustom) {
        return cyxwiz::ExtensionNodeRegistry::Instance().Has(name);
    }

    auto& registry = cyxwiz::NodeMetadataRegistry::Instance();
    registry.Initialize();
    const auto* metadata = registry.GetMetadata(type);
    return metadata == nullptr || cyxwiz::CanAddNodeToGraph(*metadata);
}
#endif

MLNode NodeEditor::CreateNodeWithIds(NodeType type,
                                     const std::string& name,
                                     int& next_node_id,
                                     int& next_pin_id) {
    return CreateGraphNode(type, name, next_node_id, next_pin_id);
}

MLNode NodeEditor::CreateNode(NodeType type, const std::string& name) {
    return CreateNodeWithIds(
        type, name, next_node_id_, next_pin_id_);
}

#ifndef CYXWIZ_NODE_FACTORY_ONLY
// Helper: if this node owns a dataset (a Data Input with a non-empty
// dataset_name parameter: its dialog loaded it), drop it from every registry
// so the next graph rebuild doesn't see a stale entry under the same name.
// A Dataset Input only names a dataset something else loaded (Data Studio),
// so removing the node leaves it. Used by DeleteNode and ClearGraph.
static void UnregisterNodeDatasetIfOwned(const MLNode& node) {
    if (node.type != NodeType::DataInput) {
        return;
    }
    auto it = node.parameters.find("dataset_name");
    if (it == node.parameters.end() || it->second.empty()) {
        return;
    }
    auto& reg = cyxwiz::DataRegistry::Instance();
    // Try every registry the node could have populated. Each is a no-op
    // for names that don't exist in that map, so it's safe to call them
    // all without knowing which type the node actually loaded.
    reg.UnregisterTabularDataset(it->second);
    reg.UnregisterImageDataset(it->second);
    reg.UnregisterAudioDataset(it->second);
    reg.UnregisterTextDataset(it->second);
}

void NodeEditor::SyncDataAfterSwap(const std::vector<MLNode>& before) {
    // Every node now on the canvas or inside a collapsed subgraph.
    std::vector<const MLNode*> after;
    for (const auto& node : nodes_) after.push_back(&node);
    for (const auto& data : subgraphs_)
        if (!data.expanded) for (const auto& node : data.internal_nodes) after.push_back(&node);
    const auto dataset_of = [](const MLNode& node) -> std::string {
        if (node.type != NodeType::DataInput) return "";
        const auto it = node.parameters.find("dataset_name");
        return it == node.parameters.end() ? "" : it->second;
    };
    std::map<int, std::string> now;  // id -> its dataset name
    std::set<std::string> named;
    for (const MLNode* node : after) {
        now[node->id] = dataset_of(*node);
        if (!now[node->id].empty()) named.insert(now[node->id]);
    }
    // Gone: a node no longer in the graph, or a Data Input whose undo went
    // back to another file. Its data and its run results go.
    std::set<int> released;
    for (const auto& node : before) {
        const auto it = now.find(node.id);
        const std::string was = dataset_of(node);
        if (it == now.end() || it->second != was) released.insert(node.id);
        if (!was.empty() && !named.count(was)) UnregisterNodeDatasetIfOwned(node);
    }
    ReleaseRunData(released);
    // Back: a Data Input whose data is not held reads Not loaded (its Apply
    // reloads it), not the "loaded" its snapshot remembered.
    for (auto& node : nodes_) {
        const std::string name = dataset_of(node);
        if (name.empty() || cyxwiz::loaders::GetByRegisteredDataset(name)) continue;
        if (node.parameters["data_loaded"] != "false") {
            node.parameters["data_loaded"] = "false";
            spdlog::info("{}: its data was released; Apply loads it again", node.name);
        }
    }
}

void NodeEditor::ReleaseRunData(const std::set<int>& node_ids) {
    if (plot_lane_) plot_lane_->ReleaseNodes(node_ids);  // what plots' runs read for them
    pending_run_release_.insert(node_ids.begin(), node_ids.end());
    ApplyPendingRunRelease();
}

void NodeEditor::ReleaseAllRunData() {
    if (plot_lane_) plot_lane_->ReleaseAll();
    pending_run_release_all_ = true;
    ApplyPendingRunRelease();
}

void NodeEditor::ApplyPendingRunRelease() {
    // A replaced executor whose run has ended: its names go, not the ones
    // the current executor registered too. Not while the current run goes:
    // a node registers its output before it completes (and is owned).
    const bool current_running = pipeline_executor_ && pipeline_executor_->IsExecuting();
    for (auto it = retired_executors_.begin(); !current_running && it != retired_executors_.end();) {
        if ((*it)->IsExecuting()) {
            ++it;
            continue;
        }
        // Names a plot's run registered too stay with it.
        std::set<std::string> keep = pipeline_executor_ ? pipeline_executor_->OwnedNames() : std::set<std::string>{};
        if (plot_lane_) keep.merge(plot_lane_->OwnedNames());
        (*it)->ReleaseAll(keep);
        it = retired_executors_.erase(it);
    }
    if (pending_run_release_.empty() && !pending_run_release_all_) return;
    if (!pipeline_executor_) {
        pending_run_release_.clear();
        pending_run_release_all_ = false;
        return;
    }
    if (pipeline_executor_->IsExecuting()) return;  // when the run ends
    if (pending_run_release_all_) pipeline_executor_->ReleaseAll();
    else pipeline_executor_->ReleaseNodes(pending_run_release_);
    pending_run_release_.clear();
    pending_run_release_all_ = false;
}

void NodeEditor::DeleteNode(int node_id) {
    if (IsSubgraphMember(node_id)) {
        spdlog::warn("Cannot delete a subgraph member individually yet; delete its Subgraph container instead.");
        return;
    }
    if (auto* data = GetSubgraphData(node_id)) {
        const auto members = data->internal_nodes;
        const bool expanded = data->expanded;
        std::erase_if(subgraphs_, [node_id](const SubgraphData& entry) { return entry.subgraph_node_id == node_id; });
        std::set<int> collapsed;
        for (const auto& member : members) {
            if (expanded) {
                DeleteNode(member.id);
            } else {
                UnregisterNodeDatasetIfOwned(member);
                collapsed.insert(member.id);
            }
        }
        ReleaseRunData(collapsed);
    }
    ClearValidationState();  // Graph changed — stale compile results

    // Delete node
    auto node_it = std::find_if(nodes_.begin(), nodes_.end(),
        [node_id](const MLNode& node) {
            return node.id == node_id;
        });

    if (node_it != nodes_.end()) {
        spdlog::info("Deleting node: {} (ID: {})", node_it->name, node_id);

        // Properties and configuration dialogs hold raw pointers into nodes_.
        // Erasing a vector element can invalidate pointers to this node and to
        // any elements shifted after it, so release every such reference first.
        if (properties_panel_) {
            properties_panel_->ClearNodeReferences();
        }

        // If this is a data input node, drop its registered dataset before
        // we erase the node — otherwise the registry entry leaks until app
        // exit (the next graph won't have a node referencing it).
        UnregisterNodeDatasetIfOwned(*node_it);

        // Delete all links connected to this node
        links_.erase(
            std::remove_if(links_.begin(), links_.end(),
                [node_id](const NodeLink& link) {
                    return link.from_node == node_id || link.to_node == node_id;
                }),
            links_.end());

        nodes_.erase(node_it);
        RebuildPinLookup();  // Rebuild pin lookup after deleting node
        // What the runs registered for it goes too (its copy of the data, its
        // results), not only the dataset its dialog loaded.
        ReleaseRunData({node_id});
    }
}

void NodeEditor::ClearGraph() {
    if (!CanReplaceGraph("clear the graph")) return;
    SaveUndoState();
    ++graph_generation_;
    simulation_session_.ClearHistory();  // node ids of the next graph are not this one's Scopes
    ClearValidationState();  // Graph changed — stale compile results

    // IMPORTANT: Clear every properties/dialog node reference BEFORE clearing
    // nodes. Both the panel and active configuration dialog hold raw pointers.
    if (properties_panel_) {
        properties_panel_->ClearNodeReferences();
    }

    // Drop every data input node's registered dataset before we lose the
    // references in nodes_. Without this, "Clear All" leaves the datasets
    // orphaned in the registry.
    for (const auto& node : nodes_) {
        UnregisterNodeDatasetIfOwned(node);
    }

    for (const auto& data : subgraphs_) {
        if (!data.expanded) for (const auto& node : data.internal_nodes) UnregisterNodeDatasetIfOwned(node);
    }
    // And everything the runs registered: the next graph's node ids start at
    // 1 again, so an old run's copy would read as the new node's.
    ReleaseAllRunData();
    subgraphs_.clear();
    nodes_.clear();
    links_.clear();
    next_node_id_ = 1;
    next_pin_id_ = 1;
    next_link_id_ = 1;

    // Reset selection state
    selected_node_id_ = -1;
    selected_node_ids_.clear();

    // Request a full ImNodes context reset - this fully clears ImNodes' internal state
    // which prevents crashes from stale node references
    pending_context_reset_ = true;
    pending_context_keep_view_ = false;

    // Clear any pending positions
    pending_positions_.clear();

    // CyxWiz Studio: Clear groups and annotations
    groups_.clear();
    next_group_id_ = 1;
    annotations_.clear();
    next_annotation_id_ = 1;
    // Note state names notes of the graph that is gone (ids restart at 1).
    selected_annotation_id_ = -1;
    dragging_annotation_id_ = -1;
    editing_annotation_ = false;
    note_press_snapshot_.reset();
    note_menu_id_ = -1;

    // Clear pin lookup
    pin_lookup_.clear();

    spdlog::info("Cleared node graph");
}

void NodeEditor::InsertPattern(const std::vector<MLNode>& nodes, const std::vector<NodeLink>& links) {
    if (!CanReplaceGraph("insert a pattern")) return;
    if (nodes.empty()) {
        spdlog::warn("InsertPattern called with empty nodes list");
        return;
    }

    SaveUndoState();

    std::map<int, ImVec2> positions;
    for (const auto& node : nodes)
        if (node.has_initial_position) positions[node.id] = ImVec2(node.initial_pos_x, node.initial_pos_y);
    // Nodes with no place of their own (a NAS suggestion) go in a row, in
    // their order, below what is on the canvas.
    float left = 0.0f, bottom = 0.0f;
    bool placed = false;
    for (const auto& node : nodes_) {
        const auto at = cached_node_positions_.find(node.id);
        if (at == cached_node_positions_.end()) continue;
        left = placed ? std::min(left, at->second.x) : at->second.x;
        bottom = placed ? std::max(bottom, at->second.y) : at->second.y;
        placed = true;
    }
    float x = left;
    for (const auto& node : nodes) {
        if (node.has_initial_position) continue;
        positions[node.id] = ImVec2(x, placed ? bottom + 220.0f : 0.0f);
        x += 220.0f;
    }
    AddWithFreshIds(nodes, links, positions);

    spdlog::info("Inserted pattern with {} nodes and {} links", nodes.size(), links.size());
}

std::map<int, int> NodeEditor::AddWithFreshIds(const std::vector<MLNode>& nodes, const std::vector<NodeLink>& links,
                                               const std::map<int, ImVec2>& positions) {
    std::map<int, int> node_ids;
    std::map<int, int> pin_ids;
    for (const auto& source : nodes) {
        MLNode node = source;
        node.id = next_node_id_++;
        // Keyed by the source pin's id: the links below name those.
        for (auto* pins : {&node.inputs, &node.outputs}) {
            for (auto& pin : *pins) {
                const int fresh = next_pin_id_++;
                pin_ids[pin.id] = fresh;
                pin.id = fresh;
            }
        }
        node_ids[source.id] = node.id;
        const auto at = positions.find(source.id);
        if (at != positions.end()) pending_positions_[node.id] = at->second;
        nodes_.push_back(std::move(node));
    }
    for (const auto& source : links) {
        const auto from = node_ids.find(source.from_node);
        const auto to = node_ids.find(source.to_node);
        const auto from_pin = pin_ids.find(source.from_pin);
        const auto to_pin = pin_ids.find(source.to_pin);
        if (from == node_ids.end() || to == node_ids.end() || from_pin == pin_ids.end() || to_pin == pin_ids.end())
            continue;  // a link to a node that was not added
        NodeLink link = source;
        link.id = next_link_id_++;
        link.from_node = from->second;
        link.to_node = to->second;
        link.from_pin = from_pin->second;
        link.to_pin = to_pin->second;
        links_.push_back(link);
    }
    // ImNodes places a node only once it exists, so the positions are applied
    // over the next few frames.
    if (!pending_positions_.empty()) pending_positions_frames_ = 3;
    RebuildPinLookup();
    return node_ids;
}

void NodeEditor::ReplaceWithEditedGraph(const std::vector<MLNode>& nodes, const std::vector<NodeLink>& links) {
    if (!CanReplaceGraph("apply the architecture")) return;
    if (nodes.empty()) return;
    SaveUndoState();
    ClearValidationState();
    if (properties_panel_) properties_panel_->ClearNodeReferences();  // it points into nodes_

    std::map<int, const MLNode*> current;
    for (const auto& node : nodes_) current[node.id] = &node;

    // A node still on the canvas as the same type keeps its id (so its place
    // and group); a new or retyped one is built by CreateNode, which owns the
    // pins for its type, with the edited parameters on top.
    std::vector<MLNode> next;
    std::map<int, int> node_ids;
    std::map<int, int> pin_ids;
    std::vector<int> built;  // ids given by CreateNode
    for (const auto& node : nodes) {
        const auto it = current.find(node.id);
        if (it != current.end() && it->second->type == node.type) {
            next.push_back(node);
            node_ids[node.id] = node.id;
            for (const auto& pin : node.inputs) pin_ids[pin.id] = pin.id;
            for (const auto& pin : node.outputs) pin_ids[pin.id] = pin.id;
            continue;
        }
        MLNode fresh = CreateNode(node.type, node.name);
        for (const auto& [key, value] : node.parameters) fresh.parameters[key] = value;
        for (size_t i = 0; i < std::min(node.inputs.size(), fresh.inputs.size()); ++i)
            pin_ids[node.inputs[i].id] = fresh.inputs[i].id;
        for (size_t i = 0; i < std::min(node.outputs.size(), fresh.outputs.size()); ++i)
            pin_ids[node.outputs[i].id] = fresh.outputs[i].id;
        node_ids[node.id] = fresh.id;
        // A retyped node stays where it was.
        if (it != current.end()) {
            const auto was = cached_node_positions_.find(node.id);
            if (was != cached_node_positions_.end()) pending_positions_[fresh.id] = was->second;
        }
        built.push_back(fresh.id);
        next.push_back(std::move(fresh));
    }

    std::vector<NodeLink> next_links;
    for (const auto& source : links) {
        const auto from = node_ids.find(source.from_node);
        const auto to = node_ids.find(source.to_node);
        const auto from_pin = pin_ids.find(source.from_pin);
        const auto to_pin = pin_ids.find(source.to_pin);
        if (from == node_ids.end() || to == node_ids.end() || from_pin == pin_ids.end() || to_pin == pin_ids.end())
            continue;
        NodeLink link = source;
        link.id = next_link_id_++;
        link.from_node = from->second;
        link.to_node = to->second;
        link.from_pin = from_pin->second;
        link.to_pin = to_pin->second;
        next_links.push_back(link);
    }

    // A new node goes midway between the nodes it links to, placed ones
    // first: two new nodes in a row take their places one after the other.
    // A place too near another node moves down until it is free.
    const auto place_of = [this](int id) -> std::optional<ImVec2> {
        if (const auto at = pending_positions_.find(id); at != pending_positions_.end()) return at->second;
        if (const auto at = cached_node_positions_.find(id); at != cached_node_positions_.end()) return at->second;
        return std::nullopt;
    };
    const auto taken = [&](const ImVec2& at) {
        const auto close_to = [&at](const ImVec2& other) {
            return std::abs(other.x - at.x) < 120.0f && std::abs(other.y - at.y) < 80.0f;
        };
        for (const auto& node : next) {
            const auto other = place_of(node.id);
            if (other && close_to(*other)) return true;
        }
        return false;
    };
    std::vector<int> waiting;
    for (const int id : built)
        if (!pending_positions_.count(id)) waiting.push_back(id);
    for (size_t pass = 0; pass < built.size() && !waiting.empty(); ++pass) {
        std::vector<int> still;
        for (const int id : waiting) {
            ImVec2 sum(0.0f, 0.0f);
            int count = 0;
            for (const auto& link : next_links) {
                const int other = link.to_node == id ? link.from_node : link.from_node == id ? link.to_node : 0;
                const auto at = other ? place_of(other) : std::nullopt;
                if (!at) continue;
                sum.x += at->x;
                sum.y += at->y;
                ++count;
            }
            if (!count) {
                still.push_back(id);
                continue;
            }
            ImVec2 at(sum.x / count, sum.y / count + 40.0f);
            for (int step = 0; step < 20 && taken(at); ++step) at.y += 90.0f;
            pending_positions_[id] = at;
        }
        if (still.size() == waiting.size()) break;  // linked to nothing placed
        waiting = std::move(still);
    }
    if (!pending_positions_.empty()) pending_positions_frames_ = 3;

    std::set<int> kept;
    for (const auto& node : next) kept.insert(node.id);
    for (auto& group : groups_) std::erase_if(group.node_ids, [&](int id) { return !kept.count(id); });

    const std::vector<MLNode> before = nodes_;
    nodes_ = std::move(next);
    links_ = std::move(next_links);
    RebuildPinLookup();
    SyncDataAfterSwap(before);
    spdlog::info("Applied an edited architecture: {} nodes, {} links ({} new)", nodes_.size(), links_.size(),
                 built.size());
}

// ===== Undo/Redo System =====

const MLNode* NodeEditor::FindNodeById(int node_id) const {
    for (const auto& node : nodes_) {
        if (node.id == node_id) {
            return &node;
        }
    }
    return nullptr;
}

MLNode* NodeEditor::FindNodeById(int node_id) {
    for (auto& node : nodes_) {
        if (node.id == node_id) {
            return &node;
        }
    }
    return nullptr;
}

// ========== Colour coding ==========
// Node colours come from the theme tokens: each category takes one series
// colour (SeriesColour gives 18 distinct ones) in the fixed order below, so
// the canvas, the Node Browser and the node documentation agree.
ImVec4 NodeCategoryColour(NodeCategory category) {
    using cyxwiz::ui::NodeHue;
    NodeHue hue = NodeHue::Neutral;
    switch (category) {
        case NodeCategory::DataSources:
        case NodeCategory::Database:
        case NodeCategory::CloudStorage:
        case NodeCategory::BigData:        hue = NodeHue::DataSources; break;
        case NodeCategory::DataTransform:
        case NodeCategory::JsonXml:        hue = NodeHue::DataTransform; break;
        case NodeCategory::Analytics:
        case NodeCategory::Visualization:
        case NodeCategory::Reporting:      hue = NodeHue::Analytics; break;
        case NodeCategory::Layers:         hue = NodeHue::Layers; break;
        case NodeCategory::Activation:     hue = NodeHue::Activation; break;
        case NodeCategory::Pooling:        hue = NodeHue::Pooling; break;
        case NodeCategory::Normalization:  hue = NodeHue::Normalization; break;
        case NodeCategory::Attention:      hue = NodeHue::Attention; break;
        case NodeCategory::Recurrent:      hue = NodeHue::Recurrent; break;
        case NodeCategory::ShapeOps:       hue = NodeHue::ShapeOps; break;
        case NodeCategory::MergeOps:       hue = NodeHue::MergeOps; break;
        case NodeCategory::Upsampling:     hue = NodeHue::Upsampling; break;
        case NodeCategory::Training:
        case NodeCategory::ModelIO:
        case NodeCategory::MLServices:
        case NodeCategory::Explainability: hue = NodeHue::Training; break;
        case NodeCategory::Regularization: hue = NodeHue::Regularization; break;
        case NodeCategory::Preprocessing:  hue = NodeHue::Preprocessing; break;
        case NodeCategory::DataPipeline:   hue = NodeHue::DataPipeline; break;
        case NodeCategory::TextProcessing: hue = NodeHue::TextProcessing; break;
        case NodeCategory::TimeSeries:     hue = NodeHue::TimeSeries; break;
        case NodeCategory::Audio:          hue = NodeHue::Audio; break;
        case NodeCategory::DNN:            hue = NodeHue::DNN; break;
        case NodeCategory::RL:             hue = NodeHue::RL; break;
        case NodeCategory::Signal:         hue = NodeHue::Signal; break;
        case NodeCategory::Plugin:         hue = NodeHue::Plugin; break;
        default:                           hue = NodeHue::Neutral; break;  // Utility, Workflow, Widgets, Unknown
    }
    return cyxwiz::ui::NodeColour(cyxwiz::ui::CurrentTokens(), hue);
}

unsigned int NodeEditor::GetNodeColor(NodeType type) {
    return cyxwiz::ui::ToU32(NodeCategoryColour(GetNodeCategoryForType(type)));
}
#endif

} // namespace gui
