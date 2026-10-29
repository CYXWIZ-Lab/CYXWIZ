#pragma once

// Graph walks shared by training_input_preparation.cpp and
// prepared_step_run.cpp (TOFIX178 S0-f2).

#include "graph_model.h"

#include <set>
#include <string>
#include <vector>

namespace cyxwiz::training_input {

inline const gui::MLNode* FindNode(const std::vector<gui::MLNode>& nodes, int id) {
    for (const auto& node : nodes)
        if (node.id == id) return &node;
    return nullptr;
}

inline int PinIndex(const std::vector<gui::NodePin>& pins, int pin_id) {
    for (size_t i = 0; i < pins.size(); ++i)
        if (pins[i].id == pin_id) return static_cast<int>(i);
    return -1;
}

inline bool IsSource(const gui::MLNode& node) {
    return node.type == gui::NodeType::DataInput || node.type == gui::NodeType::DatasetInput;
}

inline std::string DatasetName(const gui::MLNode& node) {
    const auto it = node.parameters.find("dataset_name");
    return it == node.parameters.end() ? std::string() : it->second;
}

// `node` and every node before it.
inline std::set<int> Upstream(const std::vector<gui::NodeLink>& links, int node) {
    std::set<int> slice;
    std::vector<int> pending{node};
    while (!pending.empty()) {
        const int id = pending.back();
        pending.pop_back();
        if (!slice.insert(id).second) continue;
        for (const auto& link : links)
            if (link.to_node == id) pending.push_back(link.from_node);
    }
    return slice;
}

}  // namespace cyxwiz::training_input
