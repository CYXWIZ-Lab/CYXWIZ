#include "graph_executable_model.h"

#include <algorithm>
#include <limits>
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

std::vector<CompiledGraphEdge> TensorIncomingEdges(const CompiledGraphPlan& plan,
                                                   int node_id) {
    std::vector<CompiledGraphEdge> edges;
    for (const auto& edge : plan.edges) {
        if (edge.to_node_id == node_id &&
            !IsLabelEdge(plan, edge) &&
            !IsLossToOptimizerEdge(plan, edge)) {
            edges.push_back(edge);
        }
    }
    return edges;
}

const CompiledGraphNode* FindPlanNode(const CompiledGraphPlan& plan, int node_id) {
    for (const auto& node : plan.nodes) {
        if (node.node_id == node_id) {
            return &node;
        }
    }
    return nullptr;
}

const CompiledGraphEdge* PredictionEdge(const CompiledGraphPlan& plan) {
    for (const auto& edge : plan.edges) {
        if (edge.to_node_id == plan.loss_node_id &&
            edge.to_pin_id == plan.prediction_pin_id) {
            return &edge;
        }
    }
    return nullptr;
}

void RequireSameShape(const Tensor& left,
                      const Tensor& right,
                      const char* context) {
    if (left.Shape() != right.Shape()) {
        throw std::runtime_error(std::string(context) +
                                 ": tensor shapes must match exactly");
    }
}

int ParseIntParam(const std::map<std::string, std::string>& params,
                  const std::string& key,
                  int fallback) {
    auto it = params.find(key);
    if (it == params.end() || it->second.empty()) {
        return fallback;
    }
    try {
        size_t parsed = 0;
        const int value = std::stoi(it->second, &parsed);
        if (parsed != it->second.size()) {
            throw std::runtime_error("GraphExecutableModel integer parameter has trailing text");
        }
        return value;
    } catch (...) {
        throw std::runtime_error("GraphExecutableModel invalid integer parameter '" + key + "'");
    }
}

int GraphOpConcatDim(const CompiledGraphNode& node) {
    return ParseIntParam(node.parameters, "dim", 1);
}

std::string GraphOpParam(const CompiledGraphNode& node,
                         const std::string& key,
                         const std::string& fallback) {
    auto it = node.parameters.find(key);
    return it != node.parameters.end() && !it->second.empty()
        ? it->second
        : fallback;
}

Tensor RunMergeForward(gui::NodeType type, const std::vector<const Tensor*>& inputs) {
    if (inputs.size() < 2) {
        throw std::runtime_error("GraphExecutableModel merge op needs at least two inputs");
    }
    for (size_t i = 1; i < inputs.size(); ++i) {
        RequireSameShape(*inputs.front(), *inputs[i], "GraphExecutableModel merge forward");
    }

    if (type == gui::NodeType::Add || type == gui::NodeType::Average) {
        Tensor output = inputs.front()->Clone();
        for (size_t i = 1; i < inputs.size(); ++i) {
            output = output + *inputs[i];
        }
        if (type == gui::NodeType::Average) {
            output = output * (1.0f / static_cast<float>(inputs.size()));
        }
        return output;
    }
    if (type == gui::NodeType::Multiply) {
        Tensor output = inputs.front()->Clone();
        for (size_t i = 1; i < inputs.size(); ++i) {
            output = output * *inputs[i];
        }
        return output;
    }

    throw std::runtime_error("GraphExecutableModel unsupported graph op");
}

Tensor RunMaskForward(const CompiledGraphNode& node,
                      const std::vector<const Tensor*>& inputs) {
    if (node.type == gui::NodeType::TensorCompare) {
        if (inputs.size() != 2) {
            throw std::runtime_error("GraphExecutableModel TensorCompare needs exactly two inputs");
        }
        const std::string op = GraphOpParam(node, "op", ">");
        if (op == ">") return *inputs[0] > *inputs[1];
        if (op == ">=") return *inputs[0] >= *inputs[1];
        if (op == "<") return *inputs[0] < *inputs[1];
        if (op == "<=") return *inputs[0] <= *inputs[1];
        if (op == "==") return *inputs[0] == *inputs[1];
        if (op == "!=") return *inputs[0] != *inputs[1];
        throw std::runtime_error("GraphExecutableModel TensorCompare unsupported op");
    }

    if (node.type == gui::NodeType::TensorLogicalMask) {
        if (inputs.size() != 2) {
            throw std::runtime_error("GraphExecutableModel TensorLogicalMask needs exactly two inputs");
        }
        const std::string op = GraphOpParam(node, "op", "and");
        if (op == "and") return *inputs[0] && *inputs[1];
        if (op == "or") return *inputs[0] || *inputs[1];
        throw std::runtime_error("GraphExecutableModel TensorLogicalMask supports only op=and/or for two inputs");
    }

    throw std::runtime_error("GraphExecutableModel unsupported mask graph op");
}

Tensor RunLinalgForward(const CompiledGraphNode& node,
                        const std::vector<const Tensor*>& inputs) {
    if (node.type == gui::NodeType::TensorDot) {
        if (inputs.size() != 2) {
            throw std::runtime_error("GraphExecutableModel TensorDot needs exactly two inputs");
        }
        if (inputs[0]->GetDataType() != inputs[1]->GetDataType()) {
            throw std::runtime_error("GraphExecutableModel TensorDot input data types must match");
        }
        return inputs[0]->Dot(*inputs[1]);
    }

    throw std::runtime_error("GraphExecutableModel unsupported linalg graph op");
}

Tensor RunConcatForward(const CompiledGraphNode& node,
                        const std::vector<const Tensor*>& inputs) {
    if (inputs.size() < 2) {
        throw std::runtime_error("GraphExecutableModel Concatenate needs at least two inputs");
    }

    std::vector<Tensor> values;
    values.reserve(inputs.size());
    for (const Tensor* input : inputs) {
        values.push_back(input->Clone());
    }
    return Tensor::Cat(values, GraphOpConcatDim(node));
}

std::vector<Tensor> RunMergeBackward(gui::NodeType type,
                                     const std::vector<const Tensor*>& inputs,
                                     const Tensor& grad_output) {
    std::vector<Tensor> grads;
    grads.reserve(inputs.size());

    if (type == gui::NodeType::Add) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            grads.push_back(grad_output.Clone());
        }
        return grads;
    }
    if (type == gui::NodeType::Average) {
        const float scale = 1.0f / static_cast<float>(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            grads.push_back(grad_output * scale);
        }
        return grads;
    }
    if (type == gui::NodeType::Multiply) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            Tensor grad = grad_output.Clone();
            for (size_t j = 0; j < inputs.size(); ++j) {
                if (i != j) {
                    grad = grad * *inputs[j];
                }
            }
            grads.push_back(std::move(grad));
        }
        return grads;
    }

    throw std::runtime_error("GraphExecutableModel unsupported graph op backward");
}

std::vector<Tensor> RunMaskBackward(const std::vector<const Tensor*>& inputs,
                                    const Tensor& grad_output) {
    std::vector<Tensor> grads;
    grads.reserve(inputs.size());
    for (const Tensor* input : inputs) {
        grads.push_back(Tensor::Zeros(input->Shape(), grad_output.GetDataType()));
    }
    return grads;
}

std::vector<Tensor> RunLinalgBackward(const CompiledGraphNode& node,
                                      const std::vector<const Tensor*>& inputs,
                                      const Tensor& grad_output) {
    if (node.type == gui::NodeType::TensorDot) {
        if (inputs.size() != 2) {
            throw std::runtime_error("GraphExecutableModel TensorDot backward needs exactly two inputs");
        }
        if (inputs[0]->Shape() != inputs[1]->Shape()) {
            throw std::runtime_error("GraphExecutableModel TensorDot backward input shapes must match");
        }
        if (inputs[0]->Shape().size() == 2) {
            const auto& shape = inputs[0]->Shape();
            const size_t batch = shape[0];
            const size_t features = shape[1];
            if (grad_output.NumElements() != batch) {
                throw std::runtime_error("GraphExecutableModel TensorDot backward 2D requires one gradient per batch row");
            }
            Tensor left_grad(shape, inputs[0]->GetDataType());
            Tensor right_grad(shape, inputs[1]->GetDataType());
            switch (inputs[0]->GetDataType()) {
                case DataType::Float32: {
                    const float* left = inputs[0]->Data<float>();
                    const float* right = inputs[1]->Data<float>();
                    float* left_out = left_grad.Data<float>();
                    float* right_out = right_grad.Data<float>();
                    for (size_t row = 0; row < batch; ++row) {
                        const float scale = grad_output.At(row);
                        for (size_t col = 0; col < features; ++col) {
                            const size_t idx = row * features + col;
                            left_out[idx] = right[idx] * scale;
                            right_out[idx] = left[idx] * scale;
                        }
                    }
                    return {std::move(left_grad), std::move(right_grad)};
                }
                case DataType::Float64: {
                    const double* left = inputs[0]->Data<double>();
                    const double* right = inputs[1]->Data<double>();
                    double* left_out = left_grad.Data<double>();
                    double* right_out = right_grad.Data<double>();
                    for (size_t row = 0; row < batch; ++row) {
                        const double scale = static_cast<double>(grad_output.At(row));
                        for (size_t col = 0; col < features; ++col) {
                            const size_t idx = row * features + col;
                            left_out[idx] = right[idx] * scale;
                            right_out[idx] = left[idx] * scale;
                        }
                    }
                    return {std::move(left_grad), std::move(right_grad)};
                }
                default:
                    throw std::runtime_error("GraphExecutableModel TensorDot 2D backward supports only floating tensors");
            }
        }
        if (grad_output.NumElements() != 1) {
            throw std::runtime_error("GraphExecutableModel TensorDot backward requires scalar gradient");
        }
        const float scale = grad_output.At(0);
        return {*inputs[1] * scale, *inputs[0] * scale};
    }

    throw std::runtime_error("GraphExecutableModel unsupported linalg graph op backward");
}

std::vector<Tensor> RunConcatBackward(const CompiledGraphNode& node,
                                      const std::vector<const Tensor*>& inputs,
                                      const Tensor& grad_output) {
    std::vector<int> sizes;
    sizes.reserve(inputs.size());
    const int dim = GraphOpConcatDim(node);
    for (const Tensor* input : inputs) {
        const auto& shape = input->Shape();
        const int rank = static_cast<int>(shape.size());
        const int axis = dim < 0 ? dim + rank : dim;
        if (axis < 0 || axis >= rank) {
            throw std::runtime_error("GraphExecutableModel Concatenate dim is out of range");
        }
        const size_t size = shape[static_cast<size_t>(axis)];
        if (size > static_cast<size_t>((std::numeric_limits<int>::max)())) {
            throw std::runtime_error("GraphExecutableModel Concatenate split size is too large");
        }
        sizes.push_back(static_cast<int>(size));
    }
    return grad_output.Split(sizes, dim);
}

void AccumulateNodeGrad(std::map<int, Tensor>& grads, int node_id, const Tensor& grad) {
    auto it = grads.find(node_id);
    if (it == grads.end()) {
        grads.emplace(node_id, grad.Clone());
    } else {
        it->second = it->second + grad;
    }
}

} // namespace

GraphExecutableModel::GraphExecutableModel(std::unique_ptr<SequentialModel> model,
                                           CompiledGraphPlan plan,
                                           std::vector<int> layer_node_ids)
    : GraphExecutableModel(std::move(model),
                           std::move(plan),
                           std::move(layer_node_ids),
                           {}) {
}

GraphExecutableModel::GraphExecutableModel(std::unique_ptr<SequentialModel> model,
                                           CompiledGraphPlan plan,
                                           std::vector<int> layer_node_ids,
                                           std::vector<int> graph_op_node_ids)
    : model_(std::move(model)),
      plan_(std::move(plan)),
      layer_node_ids_(std::move(layer_node_ids)),
      graph_op_node_ids_(std::move(graph_op_node_ids)) {
    if (!model_) {
        throw std::invalid_argument("GraphExecutableModel requires a model");
    }
    if (model_->Size() != layer_node_ids_.size()) {
        throw std::invalid_argument(
            "GraphExecutableModel module count does not match layer node ids");
    }
    if (!plan_.available || plan_.data_node_id < 0 || plan_.loss_node_id < 0) {
        throw std::invalid_argument(
            "GraphExecutableModel requires an available data-to-loss graph plan");
    }

    if (graph_op_node_ids_.empty()) {
        std::string reason;
        if (!CanRunLinearPlan(plan_, layer_node_ids_, &reason)) {
            throw std::invalid_argument(
                "GraphExecutableModel requires a linear graph plan: " + reason);
        }
    }
    for (const int node_id : graph_op_node_ids_) {
        const CompiledGraphNode* node = FindPlanNode(plan_, node_id);
        if (!node) {
            throw std::invalid_argument("GraphExecutableModel graph op is not in plan");
        }
        if (node->type != gui::NodeType::Add &&
            node->type != gui::NodeType::Multiply &&
            node->type != gui::NodeType::Average &&
            node->type != gui::NodeType::Concatenate &&
            node->type != gui::NodeType::TensorCompare &&
            node->type != gui::NodeType::TensorLogicalMask &&
            node->type != gui::NodeType::TensorDot) {
            throw std::invalid_argument("GraphExecutableModel unsupported graph op node");
        }
        if (TensorIncomingEdges(plan_, node_id).size() < 2) {
            throw std::invalid_argument(
                "GraphExecutableModel graph op requires at least two inputs");
        }
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

    for (const auto& node : plan_.nodes) {
        if (node.node_id == plan_.data_node_id ||
            node.node_id == plan_.loss_node_id ||
            node.node_id == plan_.optimizer_node_id) {
            continue;
        }

        size_t module_index = 0;
        Tensor output;
        bool executed = false;

        if (IsLayerNode(node.node_id, &module_index)) {
            Module* module = model_->GetModule(module_index);
            if (!module) {
                throw std::runtime_error("GraphExecutableModel missing module for node " +
                                         std::to_string(node.node_id));
            }

            const auto incoming = TensorIncomingEdges(plan_, node.node_id);
            if (incoming.size() != 1) {
                throw std::runtime_error(
                    "GraphExecutableModel layer node requires exactly one input");
            }
            const Tensor* input_tensor =
                FindCachedTensor(incoming.front().from_node_id,
                                 incoming.front().from_pin_id);
            if (!input_tensor) {
                throw std::runtime_error("GraphExecutableModel missing cached input tensor");
            }
            output = module->Forward(*input_tensor);
            executed = true;
        } else if (IsGraphOpNode(node.node_id)) {
            const auto incoming = TensorIncomingEdges(plan_, node.node_id);
            std::vector<const Tensor*> inputs;
            inputs.reserve(incoming.size());
            for (const auto& edge : incoming) {
                const Tensor* input_tensor =
                    FindCachedTensor(edge.from_node_id, edge.from_pin_id);
                if (!input_tensor) {
                    throw std::runtime_error("GraphExecutableModel missing merge input tensor");
                }
                inputs.push_back(input_tensor);
            }
            if (node.type == gui::NodeType::Concatenate) {
                output = RunConcatForward(node, inputs);
            } else if (node.type == gui::NodeType::TensorCompare ||
                       node.type == gui::NodeType::TensorLogicalMask) {
                output = RunMaskForward(node, inputs);
            } else if (node.type == gui::NodeType::TensorDot) {
                output = RunLinalgForward(node, inputs);
            } else {
                output = RunMergeForward(node.type, inputs);
            }
            executed = true;
        }

        if (!executed) {
            continue;
        }

        const int output_pin_id = OutputPinForNode(plan_, node.node_id);
        if (output_pin_id >= 0) {
            CacheTensor(node.node_id, output_pin_id, output);
        }
    }

    const CompiledGraphEdge* prediction_edge = PredictionEdge(plan_);
    if (!prediction_edge) {
        throw std::runtime_error("GraphExecutableModel missing prediction edge");
    }
    const Tensor* prediction =
        FindCachedTensor(prediction_edge->from_node_id, prediction_edge->from_pin_id);
    if (!prediction) {
        throw std::runtime_error("GraphExecutableModel missing prediction tensor");
    }

    CacheTensor(plan_.loss_node_id, plan_.prediction_pin_id, *prediction);
    return prediction->Clone();
}

Tensor GraphExecutableModel::Backward(const Tensor& grad_output) {
    const CompiledGraphEdge* prediction_edge = PredictionEdge(plan_);
    if (!prediction_edge) {
        throw std::runtime_error("GraphExecutableModel missing prediction edge");
    }

    std::map<int, Tensor> node_grads;
    AccumulateNodeGrad(node_grads, prediction_edge->from_node_id, grad_output);

    for (auto it = plan_.nodes.rbegin(); it != plan_.nodes.rend(); ++it) {
        const auto& node = *it;
        auto grad_it = node_grads.find(node.node_id);
        if (grad_it == node_grads.end()) {
            continue;
        }
        Tensor grad = grad_it->second.Clone();

        size_t module_index = 0;
        if (IsLayerNode(node.node_id, &module_index)) {
            Module* module = model_->GetModule(module_index);
            if (!module) {
                throw std::runtime_error("GraphExecutableModel missing module for node " +
                                         std::to_string(node.node_id));
            }
            const auto incoming = TensorIncomingEdges(plan_, node.node_id);
            if (incoming.size() != 1) {
                throw std::runtime_error(
                    "GraphExecutableModel layer node requires exactly one input");
            }
            Tensor input_grad = module->Backward(grad);
            AccumulateNodeGrad(node_grads, incoming.front().from_node_id, input_grad);
        } else if (IsGraphOpNode(node.node_id)) {
            const auto incoming = TensorIncomingEdges(plan_, node.node_id);
            std::vector<const Tensor*> inputs;
            inputs.reserve(incoming.size());
            for (const auto& edge : incoming) {
                const Tensor* input_tensor =
                    FindCachedTensor(edge.from_node_id, edge.from_pin_id);
                if (!input_tensor) {
                    throw std::runtime_error(
                        "GraphExecutableModel missing merge input tensor for backward");
                }
                inputs.push_back(input_tensor);
            }
            std::vector<Tensor> input_grads;
            if (node.type == gui::NodeType::Concatenate) {
                input_grads = RunConcatBackward(node, inputs, grad);
            } else if (node.type == gui::NodeType::TensorCompare ||
                       node.type == gui::NodeType::TensorLogicalMask) {
                input_grads = RunMaskBackward(inputs, grad);
            } else if (node.type == gui::NodeType::TensorDot) {
                input_grads = RunLinalgBackward(node, inputs, grad);
            } else {
                input_grads = RunMergeBackward(node.type, inputs, grad);
            }
            for (size_t i = 0; i < incoming.size(); ++i) {
                AccumulateNodeGrad(node_grads,
                                   incoming[i].from_node_id,
                                   input_grads[i]);
            }
        }
    }

    auto data_grad = node_grads.find(plan_.data_node_id);
    return data_grad != node_grads.end() ? data_grad->second.Clone()
                                         : Tensor::Zeros(grad_output.Shape(),
                                                         grad_output.GetDataType());
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

bool GraphExecutableModel::IsLayerNode(int node_id, size_t* module_index) const {
    for (size_t i = 0; i < layer_node_ids_.size(); ++i) {
        if (layer_node_ids_[i] == node_id) {
            if (module_index) {
                *module_index = i;
            }
            return true;
        }
    }
    return false;
}

bool GraphExecutableModel::IsGraphOpNode(int node_id) const {
    return std::find(graph_op_node_ids_.begin(),
                     graph_op_node_ids_.end(),
                     node_id) != graph_op_node_ids_.end();
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
