#include "compiled_graph_plan.h"

#include <unordered_map>

namespace cyxwiz {

namespace {

const gui::MLNode* FindNode(
    const std::unordered_map<int, const gui::MLNode*>& by_id,
    int node_id) {
    auto it = by_id.find(node_id);
    return it != by_id.end() ? it->second : nullptr;
}

bool NameIs(const std::string& name, const char* expected) {
    return name == expected;
}

int FindOutputPin(const gui::MLNode& node,
                  gui::PinType type,
                  const char* preferred_name = nullptr) {
    for (const auto& pin : node.outputs) {
        if (!pin.is_input && pin.type == type &&
            preferred_name && NameIs(pin.name, preferred_name)) {
            return pin.id;
        }
    }
    for (const auto& pin : node.outputs) {
        if (!pin.is_input && pin.type == type) {
            return pin.id;
        }
    }
    return -1;
}

int FindInputPin(const gui::MLNode& node,
                 gui::PinType type,
                 const char* preferred_name = nullptr) {
    for (const auto& pin : node.inputs) {
        if (pin.is_input && pin.type == type &&
            preferred_name && NameIs(pin.name, preferred_name)) {
            return pin.id;
        }
    }
    for (const auto& pin : node.inputs) {
        if (pin.is_input && pin.type == type) {
            return pin.id;
        }
    }
    return -1;
}

CompiledGraphNode CopyNode(const gui::MLNode& node) {
    CompiledGraphNode out;
    out.type = node.type;
    out.node_id = node.id;
    out.name = node.name;
    out.parameters = node.parameters;

    for (const auto& pin : node.inputs) {
        if (pin.is_input) {
            out.input_pin_ids.push_back(pin.id);
        }
    }
    for (const auto& pin : node.outputs) {
        if (!pin.is_input) {
            out.output_pin_ids.push_back(pin.id);
        }
    }
    return out;
}

} // namespace

CompiledGraphPlan BuildCompiledGraphPlan(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    const std::vector<int>& sorted_node_ids,
    const std::unordered_set<int>& training_path_ids,
    int data_node_id,
    int loss_node_id,
    int optimizer_node_id) {

    CompiledGraphPlan plan;
    if (training_path_ids.empty() || data_node_id < 0 || loss_node_id < 0) {
        return plan;
    }

    std::unordered_map<int, const gui::MLNode*> by_id;
    by_id.reserve(nodes.size());
    for (const auto& node : nodes) {
        by_id[node.id] = &node;
    }

    std::unordered_set<int> selected = training_path_ids;
    if (optimizer_node_id >= 0) {
        selected.insert(optimizer_node_id);
    }

    plan.available = true;
    plan.data_node_id = data_node_id;
    plan.loss_node_id = loss_node_id;
    plan.optimizer_node_id = optimizer_node_id;

    if (const gui::MLNode* data = FindNode(by_id, data_node_id)) {
        plan.data_pin_id = FindOutputPin(*data, gui::PinType::Tensor, "Data");
        plan.label_pin_id = FindOutputPin(*data, gui::PinType::Labels, "Labels");
    }
    if (const gui::MLNode* loss = FindNode(by_id, loss_node_id)) {
        plan.prediction_pin_id = FindInputPin(*loss, gui::PinType::Tensor, "Predictions");
        plan.label_target_pin_id = FindInputPin(*loss, gui::PinType::Labels, "Targets");
        plan.loss_output_pin_id = FindOutputPin(*loss, gui::PinType::Loss, "Loss");
    }

    for (int node_id : sorted_node_ids) {
        if (selected.count(node_id) == 0) {
            continue;
        }
        if (const gui::MLNode* node = FindNode(by_id, node_id)) {
            plan.nodes.push_back(CopyNode(*node));
        }
    }

    for (const auto& link : links) {
        if (selected.count(link.from_node) == 0 ||
            selected.count(link.to_node) == 0) {
            continue;
        }

        CompiledGraphEdge edge;
        edge.from_node_id = link.from_node;
        edge.from_pin_id = link.from_pin;
        edge.to_node_id = link.to_node;
        edge.to_pin_id = link.to_pin;
        plan.edges.push_back(edge);
    }

    return plan;
}

} // namespace cyxwiz
