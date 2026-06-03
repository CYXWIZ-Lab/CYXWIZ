#include "graph_topology_utils.h"

#include <queue>
#include <unordered_map>

namespace cyxwiz {

std::unordered_set<int> CollectReachableNodeIds(
    int source_node_id,
    const std::vector<gui::NodeLink>& links) {

    std::unordered_map<int, std::vector<int>> outgoing;
    outgoing.reserve(links.size());
    for (const auto& link : links) {
        outgoing[link.from_node].push_back(link.to_node);
    }

    std::unordered_set<int> visited;
    std::queue<int> pending;
    visited.insert(source_node_id);
    pending.push(source_node_id);

    while (!pending.empty()) {
        const int current = pending.front();
        pending.pop();

        auto it = outgoing.find(current);
        if (it == outgoing.end()) {
            continue;
        }

        for (int next : it->second) {
            if (visited.insert(next).second) {
                pending.push(next);
            }
        }
    }

    return visited;
}

std::unordered_set<int> CollectAncestorNodeIds(
    int sink_node_id,
    const std::vector<gui::NodeLink>& links) {

    std::unordered_map<int, std::vector<int>> incoming;
    incoming.reserve(links.size());
    for (const auto& link : links) {
        incoming[link.to_node].push_back(link.from_node);
    }

    std::unordered_set<int> visited;
    std::queue<int> pending;
    visited.insert(sink_node_id);
    pending.push(sink_node_id);

    while (!pending.empty()) {
        const int current = pending.front();
        pending.pop();

        auto it = incoming.find(current);
        if (it == incoming.end()) {
            continue;
        }

        for (int previous : it->second) {
            if (visited.insert(previous).second) {
                pending.push(previous);
            }
        }
    }

    return visited;
}

bool HasOutgoingLink(
    int node_id,
    const std::vector<gui::NodeLink>& links) {

    for (const auto& link : links) {
        if (link.from_node == node_id) {
            return true;
        }
    }
    return false;
}

const gui::MLNode* FindFirstReachableNodeOfType(
    const std::vector<gui::MLNode>& nodes,
    const std::unordered_set<int>& reachable_node_ids,
    gui::NodeType type) {

    for (const auto& node : nodes) {
        if (node.type == type &&
            reachable_node_ids.count(node.id) > 0) {
            return &node;
        }
    }
    return nullptr;
}

} // namespace cyxwiz
