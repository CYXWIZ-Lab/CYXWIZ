#include "node_editor.h"
#include "ui_tokens.h"
#include "../core/graph_node_factory.h"
#include "properties.h"
#include "../core/node_metadata_registry.h"
#include "../core/extension_node_registry.h"
#include "../core/data_registry.h"
#include "../core/pipeline_executor.h"
#include "loaders/data_loader.h"
#include <imgui.h>
#include <imnodes.h>
#include <spdlog/spdlog.h>
#include <algorithm>

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
    pending_run_release_.insert(node_ids.begin(), node_ids.end());
    ApplyPendingRunRelease();
}

void NodeEditor::ReleaseAllRunData() {
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
        (*it)->ReleaseAll(pipeline_executor_ ? pipeline_executor_->OwnedNames() : std::set<std::string>{});
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

    // Clear any pending positions
    pending_positions_.clear();

    // CyxWiz Studio: Clear groups and annotations
    groups_.clear();
    next_group_id_ = 1;
    annotations_.clear();
    next_annotation_id_ = 1;

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

    // Add all nodes from the pattern
    for (const auto& node : nodes) {
        nodes_.push_back(node);

        // Queue position for deferred setting (will be applied during render)
        if (node.has_initial_position) {
            pending_positions_[node.id] = ImVec2(node.initial_pos_x, node.initial_pos_y);
        }

        // Update next IDs to avoid collisions
        if (node.id >= next_node_id_) {
            next_node_id_ = node.id + 1;
        }
        for (const auto& pin : node.inputs) {
            if (pin.id >= next_pin_id_) {
                next_pin_id_ = pin.id + 1;
            }
        }
        for (const auto& pin : node.outputs) {
            if (pin.id >= next_pin_id_) {
                next_pin_id_ = pin.id + 1;
            }
        }
    }

    // Add all links from the pattern
    for (const auto& link : links) {
        links_.push_back(link);

        // Update next link ID
        if (link.id >= next_link_id_) {
            next_link_id_ = link.id + 1;
        }
    }

    // Set frame counter to apply positions during next few render frames
    // This is required because ImNodes needs nodes to exist before SetNodeGridSpacePos works
    if (!pending_positions_.empty()) {
        pending_positions_frames_ = 3;  // Apply for 3 frames to ensure positions stick
    }

    // Rebuild pin lookup after inserting pattern
    RebuildPinLookup();

    spdlog::info("Inserted pattern with {} nodes and {} links (positions queued: {})",
                 nodes.size(), links.size(), pending_positions_.size());
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
