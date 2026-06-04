#include "graph_executable_model.h"

#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace cyxwiz {

namespace {

bool IsLabelEdge(const CompiledGraphPlan& plan, const CompiledGraphEdge& edge) {
    return edge.from_node_id == plan.data_node_id &&
           edge.from_pin_id == plan.label_pin_id &&
           edge.to_node_id == plan.loss_node_id &&
           edge.to_pin_id == plan.label_target_pin_id;
}

bool IsLossToOptimizerEdge(const CompiledGraphPlan& plan,
                           const CompiledGraphEdge& edge) {
    return plan.optimizer_node_id >= 0 &&
           edge.from_node_id == plan.loss_node_id &&
           edge.from_pin_id == plan.loss_output_pin_id &&
           edge.to_node_id == plan.optimizer_node_id;
}

void SetReason(std::string* reason, const std::string& value) {
    if (reason) {
        *reason = value;
    }
}

int OutputPinForNode(const CompiledGraphPlan& plan, int node_id) {
    for (const auto& edge : plan.edges) {
        if (edge.from_node_id == node_id &&
            !IsLabelEdge(plan, edge) &&
            !IsLossToOptimizerEdge(plan, edge)) {
            return edge.from_pin_id;
        }
    }
    return -1;
}

} // namespace

GraphExecutableModel::GraphExecutableModel(std::unique_ptr<SequentialModel> model,
                                           CompiledGraphPlan plan,
                                           std::vector<int> layer_node_ids)
    : model_(std::move(model)),
      plan_(std::move(plan)),
      layer_node_ids_(std::move(layer_node_ids)) {
    if (!model_) {
        throw std::invalid_argument("GraphExecutableModel requires a model");
    }
    if (model_->Size() != layer_node_ids_.size()) {
        throw std::invalid_argument(
            "GraphExecutableModel module count does not match layer node ids");
    }

    std::string reason;
    if (!CanRunLinearPlan(plan_, layer_node_ids_, &reason)) {
        throw std::invalid_argument(
            "GraphExecutableModel requires a linear graph plan: " + reason);
    }
}

bool GraphExecutableModel::CanRunLinearPlan(const CompiledGraphPlan& plan,
                                            const std::vector<int>& layer_node_ids,
                                            std::string* reason) {
    if (!plan.available) {
        SetReason(reason, "plan is not available");
        return false;
    }
    if (plan.data_node_id < 0 || plan.loss_node_id < 0) {
        SetReason(reason, "data or loss node is missing");
        return false;
    }
    if (layer_node_ids.empty()) {
        SetReason(reason, "no executable model layers");
        return false;
    }

    std::unordered_set<int> plan_node_ids;
    plan_node_ids.reserve(plan.nodes.size());
    for (const auto& node : plan.nodes) {
        plan_node_ids.insert(node.node_id);
    }
    if (plan_node_ids.count(plan.data_node_id) == 0 ||
        plan_node_ids.count(plan.loss_node_id) == 0) {
        SetReason(reason, "data or loss node is not present in plan nodes");
        return false;
    }
    for (const int node_id : layer_node_ids) {
        if (plan_node_ids.count(node_id) == 0) {
            SetReason(reason, "layer node is not present in plan nodes");
            return false;
        }
    }

    std::unordered_map<int, int> expected_next;
    expected_next.reserve(layer_node_ids.size() + 1);
    expected_next[plan.data_node_id] = layer_node_ids.front();
    for (size_t i = 0; i + 1 < layer_node_ids.size(); ++i) {
        expected_next[layer_node_ids[i]] = layer_node_ids[i + 1];
    }
    expected_next[layer_node_ids.back()] = plan.loss_node_id;

    std::unordered_map<int, int> incoming_count;
    std::unordered_map<int, int> outgoing_count;
    incoming_count.reserve(layer_node_ids.size() + 1);
    outgoing_count.reserve(layer_node_ids.size() + 1);

    for (const auto& edge : plan.edges) {
        if (IsLabelEdge(plan, edge) || IsLossToOptimizerEdge(plan, edge)) {
            continue;
        }

        auto expected = expected_next.find(edge.from_node_id);
        if (expected == expected_next.end() || expected->second != edge.to_node_id) {
            SetReason(reason, "edge does not match the selected linear chain");
            return false;
        }

        ++outgoing_count[edge.from_node_id];
        ++incoming_count[edge.to_node_id];
    }

    for (const int node_id : layer_node_ids) {
        if (incoming_count[node_id] != 1) {
            SetReason(reason, "layer node has zero or multiple tensor inputs");
            return false;
        }
        if (outgoing_count[node_id] != 1) {
            SetReason(reason, "layer node has zero or multiple tensor outputs");
            return false;
        }
    }
    if (outgoing_count[plan.data_node_id] != 1) {
        SetReason(reason, "data node does not feed exactly one model input");
        return false;
    }
    if (incoming_count[plan.loss_node_id] != 1) {
        SetReason(reason, "loss node does not receive exactly one prediction input");
        return false;
    }

    return true;
}

Tensor GraphExecutableModel::Forward(const Tensor& input) {
    tensor_cache_.clear();
    CacheTensor(plan_.data_node_id, plan_.data_pin_id, input);

    Tensor current = input.Clone();
    for (size_t i = 0; i < layer_node_ids_.size(); ++i) {
        Module* module = model_->GetModule(i);
        if (!module) {
            throw std::runtime_error("GraphExecutableModel missing module for node " +
                                     std::to_string(layer_node_ids_[i]));
        }

        current = module->Forward(current);

        const int output_pin_id = OutputPinForNode(plan_, layer_node_ids_[i]);
        if (output_pin_id >= 0) {
            CacheTensor(layer_node_ids_[i], output_pin_id, current);
        }
    }

    CacheTensor(plan_.loss_node_id, plan_.prediction_pin_id, current);
    return current;
}

Tensor GraphExecutableModel::Backward(const Tensor& grad_output) {
    Tensor grad = grad_output.Clone();
    for (int i = static_cast<int>(layer_node_ids_.size()) - 1; i >= 0; --i) {
        Module* module = model_->GetModule(static_cast<size_t>(i));
        if (!module) {
            throw std::runtime_error("GraphExecutableModel missing module for node " +
                                     std::to_string(layer_node_ids_[i]));
        }
        grad = module->Backward(grad);
    }
    return grad;
}

void GraphExecutableModel::SetTraining(bool training) {
    model_->SetTraining(training);
}

std::map<std::string, Tensor> GraphExecutableModel::GetParameters() {
    return model_->GetParameters();
}

void GraphExecutableModel::SetParameters(const std::map<std::string, Tensor>& params) {
    model_->SetParameters(params);
}

std::map<std::string, Tensor> GraphExecutableModel::GetGradients() {
    return model_->GetGradients();
}

void GraphExecutableModel::UpdateParameters(Optimizer* optimizer) {
    model_->UpdateParameters(optimizer);
}

const Tensor* GraphExecutableModel::FindCachedTensor(int node_id, int pin_id) const {
    const auto it = tensor_cache_.find({node_id, pin_id});
    return it != tensor_cache_.end() ? &it->second : nullptr;
}

void GraphExecutableModel::CacheTensor(int node_id, int pin_id, const Tensor& tensor) {
    if (node_id < 0 || pin_id < 0) {
        return;
    }
    tensor_cache_.insert_or_assign({node_id, pin_id}, tensor.Clone());
}

} // namespace cyxwiz
