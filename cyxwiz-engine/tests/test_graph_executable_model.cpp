#include "../src/core/graph_executable_model.h"
#include "../src/core/model_builder.h"

#include <cyxwiz/tensor.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

void CheckNear(float actual, float expected, float tolerance, const std::string& message) {
    if (std::abs(actual - expected) > tolerance) {
        std::cerr << "FAIL: " << message
                  << " actual=" << actual
                  << " expected=" << expected
                  << "\n";
        std::exit(1);
    }
}

cyxwiz::Tensor MakeTensor(const std::vector<float>& values) {
    return cyxwiz::Tensor({1, values.size()},
                          values.data(),
                          cyxwiz::DataType::Float32);
}

cyxwiz::CompiledLayer TensorAbsLayer() {
    cyxwiz::CompiledLayer layer;
    layer.type = gui::NodeType::TensorAbs;
    layer.node_id = 2;
    layer.name = "Tensor Abs";
    layer.input_shape = {3};
    layer.output_shape = {3};
    return layer;
}

cyxwiz::CompiledLayer TensorPowLayer() {
    cyxwiz::CompiledLayer layer;
    layer.type = gui::NodeType::TensorPow;
    layer.node_id = 3;
    layer.name = "Tensor Pow";
    layer.parameters["exponent"] = "2";
    layer.input_shape = {3};
    layer.output_shape = {3};
    return layer;
}

cyxwiz::CompiledGraphNode PlanNode(gui::NodeType type, int id, const std::string& name) {
    cyxwiz::CompiledGraphNode node;
    node.type = type;
    node.node_id = id;
    node.name = name;
    return node;
}

cyxwiz::CompiledGraphEdge PlanEdge(int from_node,
                                   int from_pin,
                                   int to_node,
                                   int to_pin) {
    cyxwiz::CompiledGraphEdge edge;
    edge.from_node_id = from_node;
    edge.from_pin_id = from_pin;
    edge.to_node_id = to_node;
    edge.to_pin_id = to_pin;
    return edge;
}

cyxwiz::CompiledGraphPlan LinearPlan() {
    cyxwiz::CompiledGraphPlan plan;
    plan.available = true;
    plan.data_node_id = 1;
    plan.data_pin_id = 101;
    plan.label_pin_id = 102;
    plan.prediction_pin_id = 401;
    plan.label_target_pin_id = 402;
    plan.loss_node_id = 4;
    plan.loss_output_pin_id = 403;
    plan.optimizer_node_id = 5;
    plan.nodes = {
        PlanNode(gui::NodeType::DataInput, 1, "Data"),
        PlanNode(gui::NodeType::TensorAbs, 2, "Tensor Abs"),
        PlanNode(gui::NodeType::TensorPow, 3, "Tensor Pow"),
        PlanNode(gui::NodeType::MSELoss, 4, "Loss"),
        PlanNode(gui::NodeType::SGD, 5, "SGD"),
    };
    plan.edges = {
        PlanEdge(1, 101, 2, 201),
        PlanEdge(2, 202, 3, 301),
        PlanEdge(3, 302, 4, 401),
        PlanEdge(1, 102, 4, 402),
        PlanEdge(4, 403, 5, 501),
    };
    return plan;
}

cyxwiz::CompiledGraphPlan MergePlan(gui::NodeType merge_type) {
    cyxwiz::CompiledGraphPlan plan;
    plan.available = true;
    plan.data_node_id = 1;
    plan.data_pin_id = 101;
    plan.label_pin_id = 102;
    plan.prediction_pin_id = 401;
    plan.label_target_pin_id = 402;
    plan.loss_node_id = 4;
    plan.loss_output_pin_id = 403;
    plan.optimizer_node_id = 5;
    plan.nodes = {
        PlanNode(gui::NodeType::DataInput, 1, "Data"),
        PlanNode(merge_type, 2, "Merge"),
        PlanNode(gui::NodeType::MSELoss, 4, "Loss"),
        PlanNode(gui::NodeType::SGD, 5, "SGD"),
    };
    plan.edges = {
        PlanEdge(1, 101, 2, 201),
        PlanEdge(1, 101, 2, 202),
        PlanEdge(2, 203, 4, 401),
        PlanEdge(1, 102, 4, 402),
        PlanEdge(4, 403, 5, 501),
    };
    return plan;
}

cyxwiz::TrainingConfiguration LinearConfig() {
    cyxwiz::TrainingConfiguration config;
    config.input_size = 3;
    config.output_size = 3;
    config.input_shape = {3};
    config.layers.push_back(TensorAbsLayer());
    config.layers.push_back(TensorPowLayer());
    config.loss_type = gui::NodeType::MSELoss;
    config.optimizer_type = gui::NodeType::SGD;
    config.learning_rate = 0.01f;
    config.graph_plan = LinearPlan();
    return config;
}

cyxwiz::TrainingConfiguration MergeConfig(gui::NodeType merge_type) {
    cyxwiz::TrainingConfiguration config;
    config.input_size = 3;
    config.output_size = 3;
    config.input_shape = {3};
    config.loss_type = gui::NodeType::MSELoss;
    config.optimizer_type = gui::NodeType::SGD;
    config.learning_rate = 0.01f;
    config.graph_plan = MergePlan(merge_type);
    config.graph_op_node_ids = {2};
    return config;
}

void CheckTensorNear(const cyxwiz::Tensor& actual,
                     const cyxwiz::Tensor& expected,
                     const std::string& message);

void CheckMergeOp(gui::NodeType merge_type,
                  const std::vector<float>& expected_output,
                  const std::vector<float>& expected_data_grad,
                  const std::string& name) {
    cyxwiz::BuiltExecutableModel built =
        cyxwiz::BuildGraphExecutableFromConfig(MergeConfig(merge_type));
    Check(built.ok(), name + " graph executable should build through config");
    auto* graph =
        dynamic_cast<cyxwiz::GraphExecutableModel*>(built.model.get());
    Check(graph != nullptr, name + " builder should return GraphExecutableModel");
    Check(graph->GraphOpNodeIds() == std::vector<int>({2}),
          name + " should preserve graph op node ids");

    cyxwiz::Tensor output = graph->Forward(MakeTensor({2.0f, 3.0f, 4.0f}));
    for (size_t col = 0; col < expected_output.size(); ++col) {
        CheckNear(output.At(0, col), expected_output[col], 1e-4f,
                  name + " forward");
    }

    const cyxwiz::Tensor* cached = graph->FindCachedTensor(2, 203);
    Check(cached != nullptr, name + " should cache merge output");
    CheckTensorNear(*cached, output, name + " cached output should match");

    cyxwiz::Tensor backward = graph->Backward(cyxwiz::Tensor::Ones({1, 3}));
    for (size_t col = 0; col < expected_data_grad.size(); ++col) {
        CheckNear(backward.At(0, col), expected_data_grad[col], 1e-4f,
                  name + " backward");
    }
}

void CheckTensorNear(const cyxwiz::Tensor& actual,
                     const cyxwiz::Tensor& expected,
                     const std::string& message) {
    Check(actual.Shape() == expected.Shape(), message + " shape mismatch");
    const auto shape = actual.Shape();
    Check(shape.size() == 2, message + " expected 2D tensor");
    for (size_t col = 0; col < shape[1]; ++col) {
        CheckNear(actual.At(0, col), expected.At(0, col), 1e-4f, message);
    }
}

} // namespace

int main() {
    cyxwiz::TrainingConfiguration config = LinearConfig();

    cyxwiz::BuiltModel sequential = cyxwiz::BuildSequentialFromConfig(config);
    cyxwiz::BuiltExecutableModel graph = cyxwiz::BuildGraphExecutableFromConfig(config);

    Check(sequential.ok(), "sequential model should build");
    Check(graph.ok(), "linear graph executable should build");
    Check(graph.model->AsSequentialModel() == nullptr,
          "graph executable should not expose itself as a sequential model");

    const auto* graph_model =
        dynamic_cast<cyxwiz::GraphExecutableModel*>(graph.model.get());
    Check(graph_model != nullptr, "builder should return GraphExecutableModel");
    Check(graph_model->LayerNodeIds() == std::vector<int>({2, 3}),
          "graph executable should preserve layer node ids");

    cyxwiz::Tensor input = MakeTensor({-2.0f, 3.0f, -4.0f});
    cyxwiz::Tensor sequential_output = sequential.model->Forward(input);
    cyxwiz::Tensor graph_output = graph.model->Forward(input);
    CheckTensorNear(graph_output, sequential_output,
                    "graph forward should match sequential forward");
    CheckNear(graph_output.At(0, 0), 4.0f, 1e-4f,
              "graph forward should execute TensorAbs -> TensorPow");
    CheckNear(graph_output.At(0, 2), 16.0f, 1e-4f,
              "graph forward should preserve final value");

    const cyxwiz::Tensor* data_cached = graph_model->FindCachedTensor(1, 101);
    Check(data_cached != nullptr, "graph executable should cache data output pin");
    CheckNear(data_cached->At(0, 0), -2.0f, 1e-4f,
              "data cache should preserve original input");

    const cyxwiz::Tensor* abs_cached = graph_model->FindCachedTensor(2, 202);
    Check(abs_cached != nullptr, "graph executable should cache first layer output pin");
    CheckNear(abs_cached->At(0, 0), 2.0f, 1e-4f,
              "first layer cache should hold TensorAbs output");

    const cyxwiz::Tensor* pow_cached = graph_model->FindCachedTensor(3, 302);
    Check(pow_cached != nullptr, "graph executable should cache second layer output pin");
    CheckNear(pow_cached->At(0, 2), 16.0f, 1e-4f,
              "second layer cache should hold TensorPow output");

    const cyxwiz::Tensor* prediction_cached = graph_model->FindCachedTensor(4, 401);
    Check(prediction_cached != nullptr,
          "graph executable should cache loss prediction input pin");
    CheckTensorNear(*prediction_cached, graph_output,
                    "prediction input cache should match model output");

    cyxwiz::Tensor grad = cyxwiz::Tensor::Ones({1, 3});
    cyxwiz::Tensor sequential_backward = sequential.model->Backward(grad);
    cyxwiz::Tensor graph_backward = graph.model->Backward(grad);
    CheckTensorNear(graph_backward, sequential_backward,
                    "graph backward should match sequential backward");

    config.graph_plan.edges.push_back(PlanEdge(1, 101, 3, 301));
    cyxwiz::BuiltExecutableModel rejected =
        cyxwiz::BuildGraphExecutableFromConfig(config);
    Check(!rejected.ok(), "fan-in graph plan should remain rejected");

    config = LinearConfig();
    config.graph_plan.nodes.erase(config.graph_plan.nodes.begin() + 2);
    rejected = cyxwiz::BuildGraphExecutableFromConfig(config);
    Check(!rejected.ok(), "graph plan missing a layer node should be rejected");

    CheckMergeOp(gui::NodeType::Add,
                 {4.0f, 6.0f, 8.0f},
                 {2.0f, 2.0f, 2.0f},
                 "Add");
    CheckMergeOp(gui::NodeType::Multiply,
                 {4.0f, 9.0f, 16.0f},
                 {4.0f, 6.0f, 8.0f},
                 "Multiply");
    CheckMergeOp(gui::NodeType::Average,
                 {2.0f, 3.0f, 4.0f},
                 {1.0f, 1.0f, 1.0f},
                 "Average");

    std::cout << "Graph executable model parity passed\n";
    return 0;
}
