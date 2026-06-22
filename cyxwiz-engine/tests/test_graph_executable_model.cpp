#include "../src/core/graph_executable_model.h"
#include "../src/core/model_builder.h"

#include <cyxwiz/memory_manager.h>
#include <cyxwiz/tensor.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

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

cyxwiz::Tensor MakeVector(const std::vector<float>& values) {
    return cyxwiz::Tensor({values.size()},
                          values.data(),
                          cyxwiz::DataType::Float32);
}

cyxwiz::Tensor MakeMatrix(size_t rows, size_t cols, const std::vector<float>& values) {
    Check(values.size() == rows * cols, "MakeMatrix value count must match shape");
    return cyxwiz::Tensor({rows, cols},
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

cyxwiz::CompiledGraphNode PlanNode(gui::NodeType type,
                                   int id,
                                   const std::string& name,
                                   std::map<std::string, std::string> parameters) {
    cyxwiz::CompiledGraphNode node = PlanNode(type, id, name);
    node.parameters = std::move(parameters);
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
        merge_type == gui::NodeType::Concatenate
            ? PlanNode(merge_type, 2, "Merge", {{"dim", "1"}})
            : merge_type == gui::NodeType::TensorCompare
                ? PlanNode(merge_type, 2, "Merge", {{"op", "=="}})
            : merge_type == gui::NodeType::TensorLogicalMask
                ? PlanNode(merge_type, 2, "Merge", {{"op", "and"}})
            : PlanNode(merge_type, 2, "Merge"),
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

cyxwiz::CompiledGraphPlan DotPlan() {
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
        PlanNode(gui::NodeType::TensorDot, 2, "Dot"),
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

cyxwiz::TrainingConfiguration DotConfig() {
    cyxwiz::TrainingConfiguration config;
    config.input_size = 3;
    config.output_size = 1;
    config.input_shape = {3};
    config.loss_type = gui::NodeType::MSELoss;
    config.optimizer_type = gui::NodeType::SGD;
    config.learning_rate = 0.01f;
    config.graph_plan = DotPlan();
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
        cyxwiz::BuildExecutableFromConfig(MergeConfig(merge_type));
    Check(built.ok(), name + " graph executable should build through executable config");
    auto* graph =
        dynamic_cast<cyxwiz::GraphExecutableModel*>(built.model.get());
    Check(graph != nullptr, name + " executable builder should return GraphExecutableModel");
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

void CheckConcatOp() {
    cyxwiz::BuiltExecutableModel built =
        cyxwiz::BuildExecutableFromConfig(MergeConfig(gui::NodeType::Concatenate));
    Check(built.ok(), "Concatenate graph executable should build through executable config");
    auto* graph =
        dynamic_cast<cyxwiz::GraphExecutableModel*>(built.model.get());
    Check(graph != nullptr, "Concatenate builder should return GraphExecutableModel");
    Check(graph->GraphOpNodeIds() == std::vector<int>({2}),
          "Concatenate should preserve graph op node ids");

    cyxwiz::Tensor output = graph->Forward(MakeTensor({2.0f, 3.0f, 4.0f}));
    Check(output.Shape() == std::vector<size_t>({1, 6}),
          "Concatenate forward should concatenate along feature dimension");
    const std::vector<float> expected_output = {2.0f, 3.0f, 4.0f,
                                                2.0f, 3.0f, 4.0f};
    for (size_t col = 0; col < expected_output.size(); ++col) {
        CheckNear(output.At(0, col), expected_output[col], 1e-4f,
                  "Concatenate forward");
    }

    const cyxwiz::Tensor* cached = graph->FindCachedTensor(2, 203);
    Check(cached != nullptr, "Concatenate should cache output");
    CheckTensorNear(*cached, output, "Concatenate cached output should match");

    cyxwiz::Tensor backward = graph->Backward(cyxwiz::Tensor::Ones({1, 6}));
    Check(backward.Shape() == std::vector<size_t>({1, 3}),
          "Concatenate backward should return original input shape");
    for (size_t col = 0; col < 3; ++col) {
        CheckNear(backward.At(0, col), 2.0f, 1e-4f,
                  "Concatenate backward should accumulate split gradients");
    }
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
void CheckMergeOpArrayFireResidency(gui::NodeType merge_type,
                                    const std::vector<float>& expected_output,
                                    const std::string& name) {
    cyxwiz::BuiltExecutableModel built =
        cyxwiz::BuildExecutableFromConfig(MergeConfig(merge_type));
    Check(built.ok(), name + " ArrayFire graph executable should build");
    auto* graph =
        dynamic_cast<cyxwiz::GraphExecutableModel*>(built.model.get());
    Check(graph != nullptr, name + " ArrayFire builder should return GraphExecutableModel");

    cyxwiz::Tensor host_input = MakeTensor({2.0f, 3.0f, 4.0f});
    cyxwiz::Tensor device_input =
        cyxwiz::Tensor::FromArrayRowMajor2D(host_input.GetArrayRowMajor2D());

    const size_t before_host_bytes = cyxwiz::MemoryManager::GetAllocatedBytes();
    cyxwiz::Tensor output = graph->Forward(device_input);

    Check(output.Shape() == std::vector<size_t>({1, expected_output.size()}),
          name + " ArrayFire graph output shape should match");
    Check(cyxwiz::MemoryManager::GetAllocatedBytes() == before_host_bytes,
          name + " ArrayFire graph forward should not materialize host output");

    af::array output_device = output.GetArrayRowMajor2D();
    Check(output_device.dims(0) == 1,
          name + " ArrayFire graph output should preserve device rows");
    Check(output_device.dims(1) == static_cast<dim_t>(expected_output.size()),
          name + " ArrayFire graph output should preserve device columns");
    Check(cyxwiz::MemoryManager::GetAllocatedBytes() == before_host_bytes,
          name + " ArrayFire graph output device access should not materialize host data");

    const cyxwiz::Tensor* cached = graph->FindCachedTensor(2, 203);
    Check(cached != nullptr, name + " ArrayFire graph should cache output");
    af::array cached_device = cached->GetArrayRowMajor2D();
    Check(cached_device.dims(0) == 1,
          name + " ArrayFire cached output should preserve device rows");
    Check(cached_device.dims(1) == static_cast<dim_t>(expected_output.size()),
          name + " ArrayFire cached output should preserve device columns");
    Check(cyxwiz::MemoryManager::GetAllocatedBytes() == before_host_bytes,
          name + " ArrayFire cached device access should not materialize host data");

    for (size_t col = 0; col < expected_output.size(); ++col) {
        CheckNear(output.At(0, col), expected_output[col], 1e-4f,
                  name + " ArrayFire graph forward");
    }
}

void CheckGraphFanInArrayFireResidency() {
    CheckMergeOpArrayFireResidency(gui::NodeType::Add,
                                   {4.0f, 6.0f, 8.0f},
                                   "Add");
    CheckMergeOpArrayFireResidency(gui::NodeType::Multiply,
                                   {4.0f, 9.0f, 16.0f},
                                   "Multiply");
    CheckMergeOpArrayFireResidency(gui::NodeType::Average,
                                   {2.0f, 3.0f, 4.0f},
                                   "Average");
    CheckMergeOpArrayFireResidency(gui::NodeType::Concatenate,
                                   {2.0f, 3.0f, 4.0f,
                                    2.0f, 3.0f, 4.0f},
                                   "Concatenate");
}

void CheckMergeBackwardArrayFireResidency(gui::NodeType merge_type,
                                          const std::vector<float>& grad_values,
                                          const std::vector<float>& expected_grad,
                                          const std::string& name) {
    cyxwiz::BuiltExecutableModel built =
        cyxwiz::BuildExecutableFromConfig(MergeConfig(merge_type));
    Check(built.ok(), name + " ArrayFire backward graph executable should build");
    auto* graph =
        dynamic_cast<cyxwiz::GraphExecutableModel*>(built.model.get());
    Check(graph != nullptr,
          name + " ArrayFire backward builder should return GraphExecutableModel");

    cyxwiz::Tensor host_input = MakeTensor({2.0f, 3.0f, 4.0f});
    cyxwiz::Tensor device_input =
        cyxwiz::Tensor::FromArrayRowMajor2D(host_input.GetArrayRowMajor2D());
    cyxwiz::Tensor host_grad = MakeTensor(grad_values);
    cyxwiz::Tensor device_grad =
        cyxwiz::Tensor::FromArrayRowMajor2D(host_grad.GetArrayRowMajor2D());

    (void)graph->Forward(device_input);

    const size_t before_host_bytes = cyxwiz::MemoryManager::GetAllocatedBytes();
    cyxwiz::Tensor backward = graph->Backward(device_grad);

    Check(backward.Shape() == std::vector<size_t>({1, expected_grad.size()}),
          name + " ArrayFire backward output shape should match input shape");
    Check(cyxwiz::MemoryManager::GetAllocatedBytes() == before_host_bytes,
          name + " ArrayFire backward should not materialize host output");

    af::array backward_device = backward.GetArrayRowMajor2D();
    Check(backward_device.dims(0) == 1,
          name + " ArrayFire backward output should preserve device rows");
    Check(backward_device.dims(1) == static_cast<dim_t>(expected_grad.size()),
          name + " ArrayFire backward output should preserve device columns");
    Check(cyxwiz::MemoryManager::GetAllocatedBytes() == before_host_bytes,
          name + " ArrayFire backward device access should not materialize host data");

    for (size_t col = 0; col < expected_grad.size(); ++col) {
        CheckNear(backward.At(0, col), expected_grad[col], 1e-4f,
                  name + " ArrayFire backward");
    }
}

void CheckGraphFanInBackwardArrayFireResidency() {
    CheckMergeBackwardArrayFireResidency(gui::NodeType::Add,
                                         {1.0f, 1.0f, 1.0f},
                                         {2.0f, 2.0f, 2.0f},
                                         "Add");
    CheckMergeBackwardArrayFireResidency(gui::NodeType::Multiply,
                                         {1.0f, 1.0f, 1.0f},
                                         {4.0f, 6.0f, 8.0f},
                                         "Multiply");
    CheckMergeBackwardArrayFireResidency(gui::NodeType::Average,
                                         {1.0f, 1.0f, 1.0f},
                                         {1.0f, 1.0f, 1.0f},
                                         "Average");
    CheckMergeBackwardArrayFireResidency(gui::NodeType::Concatenate,
                                         {1.0f, 1.0f, 1.0f,
                                          1.0f, 1.0f, 1.0f},
                                         {2.0f, 2.0f, 2.0f},
                                         "Concatenate");
}

void CheckMaskOpArrayFireResidency(gui::NodeType mask_type,
                                   const std::vector<float>& input_values,
                                   const std::vector<float>& expected_output,
                                   const std::string& name) {
    cyxwiz::BuiltExecutableModel built =
        cyxwiz::BuildExecutableFromConfig(MergeConfig(mask_type));
    Check(built.ok(), name + " ArrayFire graph executable should build");
    auto* graph =
        dynamic_cast<cyxwiz::GraphExecutableModel*>(built.model.get());
    Check(graph != nullptr, name + " ArrayFire builder should return GraphExecutableModel");

    cyxwiz::Tensor host_input = MakeTensor(input_values);
    cyxwiz::Tensor device_input =
        cyxwiz::Tensor::FromArrayRowMajor2D(host_input.GetArrayRowMajor2D());

    const size_t before_host_bytes = cyxwiz::MemoryManager::GetAllocatedBytes();
    cyxwiz::Tensor output = graph->Forward(device_input);

    Check(output.Shape() == std::vector<size_t>({1, expected_output.size()}),
          name + " ArrayFire graph output shape should match");
    Check(cyxwiz::MemoryManager::GetAllocatedBytes() == before_host_bytes,
          name + " ArrayFire graph forward should not materialize host output");

    af::array output_device = output.GetArrayRowMajor2D();
    Check(output_device.dims(0) == 1,
          name + " ArrayFire graph output should preserve device rows");
    Check(output_device.dims(1) == static_cast<dim_t>(expected_output.size()),
          name + " ArrayFire graph output should preserve device columns");
    Check(cyxwiz::MemoryManager::GetAllocatedBytes() == before_host_bytes,
          name + " ArrayFire graph output device access should not materialize host data");

    const cyxwiz::Tensor* cached = graph->FindCachedTensor(2, 203);
    Check(cached != nullptr, name + " ArrayFire graph should cache output");
    af::array cached_device = cached->GetArrayRowMajor2D();
    Check(cached_device.dims(0) == 1,
          name + " ArrayFire cached output should preserve device rows");
    Check(cached_device.dims(1) == static_cast<dim_t>(expected_output.size()),
          name + " ArrayFire cached output should preserve device columns");
    Check(cyxwiz::MemoryManager::GetAllocatedBytes() == before_host_bytes,
          name + " ArrayFire cached device access should not materialize host data");

    for (size_t col = 0; col < expected_output.size(); ++col) {
        CheckNear(output.At(0, col), expected_output[col], 1e-4f,
                  name + " ArrayFire graph forward");
    }
}

void CheckGraphMaskArrayFireResidency() {
    CheckMaskOpArrayFireResidency(gui::NodeType::TensorCompare,
                                  {0.0f, 2.0f, -1.0f},
                                  {1.0f, 1.0f, 1.0f},
                                  "TensorCompare");
    CheckMaskOpArrayFireResidency(gui::NodeType::TensorLogicalMask,
                                  {0.0f, 2.0f, -1.0f},
                                  {0.0f, 1.0f, 1.0f},
                                  "TensorLogicalMask");
}
#endif

void CheckBinaryMaskOp(gui::NodeType mask_type,
                       const std::vector<float>& input_values,
                       const std::vector<float>& expected_output,
                       const std::string& name) {
    cyxwiz::BuiltExecutableModel built =
        cyxwiz::BuildExecutableFromConfig(MergeConfig(mask_type));
    Check(built.ok(), name + " graph executable should build through executable config");
    auto* graph =
        dynamic_cast<cyxwiz::GraphExecutableModel*>(built.model.get());
    Check(graph != nullptr, name + " builder should return GraphExecutableModel");

    cyxwiz::Tensor output = graph->Forward(MakeTensor(input_values));
    Check(output.Shape() == std::vector<size_t>({1, expected_output.size()}),
          name + " forward should preserve broadcast output shape");
    for (size_t col = 0; col < expected_output.size(); ++col) {
        CheckNear(output.At(0, col), expected_output[col], 1e-4f,
                  name + " forward");
    }

    cyxwiz::Tensor backward =
        graph->Backward(cyxwiz::Tensor::Ones({1, expected_output.size()}));
    Check(backward.Shape() == std::vector<size_t>({1, input_values.size()}),
          name + " backward should return input shape");
    for (size_t col = 0; col < input_values.size(); ++col) {
        CheckNear(backward.At(0, col), 0.0f, 1e-4f,
                  name + " backward should be zero for masks");
    }
}

void CheckDotOp() {
    cyxwiz::BuiltExecutableModel built =
        cyxwiz::BuildExecutableFromConfig(DotConfig());
    Check(built.ok(), "TensorDot graph executable should build through executable config");
    auto* graph =
        dynamic_cast<cyxwiz::GraphExecutableModel*>(built.model.get());
    Check(graph != nullptr, "TensorDot builder should return GraphExecutableModel");
    Check(graph->GraphOpNodeIds() == std::vector<int>({2}),
          "TensorDot should preserve graph op node ids");

    cyxwiz::Tensor output = graph->Forward(MakeVector({2.0f, 3.0f, 4.0f}));
    Check(output.Shape() == std::vector<size_t>({1}),
          "TensorDot forward should produce scalar tensor");
    CheckNear(output.At(0), 29.0f, 1e-4f,
              "TensorDot forward should compute inner product");

    const cyxwiz::Tensor* cached = graph->FindCachedTensor(2, 203);
    Check(cached != nullptr, "TensorDot should cache output");
    Check(cached->Shape() == std::vector<size_t>({1}),
          "TensorDot cached output should be scalar");
    CheckNear(cached->At(0), 29.0f, 1e-4f,
              "TensorDot cached output should match");

    cyxwiz::Tensor backward = graph->Backward(cyxwiz::Tensor::Ones({1}));
    Check(backward.Shape() == std::vector<size_t>({3}),
          "TensorDot backward should return input vector shape");
    const std::vector<float> expected_grad = {4.0f, 6.0f, 8.0f};
    for (size_t i = 0; i < expected_grad.size(); ++i) {
        CheckNear(backward.At(i), expected_grad[i], 1e-4f,
                  "TensorDot backward should accumulate both shared inputs");
    }

    cyxwiz::Tensor batch_output =
        graph->Forward(MakeMatrix(2, 3, {1.0f, 2.0f, 3.0f,
                                         4.0f, 5.0f, 6.0f}));
    Check(batch_output.Shape() == std::vector<size_t>({2, 1}),
          "TensorDot 2D forward should produce one dot per batch row");
    CheckNear(batch_output.At(0, 0), 14.0f, 1e-4f,
              "TensorDot 2D forward first row");
    CheckNear(batch_output.At(1, 0), 77.0f, 1e-4f,
              "TensorDot 2D forward second row");

    cyxwiz::Tensor batch_backward = graph->Backward(cyxwiz::Tensor::Ones({2, 1}));
    Check(batch_backward.Shape() == std::vector<size_t>({2, 3}),
          "TensorDot 2D backward should return input batch shape");
    const std::vector<float> expected_batch_grad = {
        2.0f, 4.0f, 6.0f,
        8.0f, 10.0f, 12.0f,
    };
    for (size_t row = 0; row < 2; ++row) {
        for (size_t col = 0; col < 3; ++col) {
            CheckNear(batch_backward.At(row, col),
                      expected_batch_grad[row * 3 + col],
                      1e-4f,
                      "TensorDot 2D backward should accumulate shared inputs");
        }
    }
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
void CheckDotOpArrayFireResidency() {
    cyxwiz::BuiltExecutableModel built =
        cyxwiz::BuildExecutableFromConfig(DotConfig());
    Check(built.ok(), "TensorDot ArrayFire graph executable should build");
    auto* graph =
        dynamic_cast<cyxwiz::GraphExecutableModel*>(built.model.get());
    Check(graph != nullptr, "TensorDot ArrayFire builder should return GraphExecutableModel");

    cyxwiz::Tensor host_input =
        MakeMatrix(2, 3, {1.0f, 2.0f, 3.0f,
                          4.0f, 5.0f, 6.0f});
    cyxwiz::Tensor device_input =
        cyxwiz::Tensor::FromArrayRowMajor2D(host_input.GetArrayRowMajor2D());

    const size_t before_host_bytes = cyxwiz::MemoryManager::GetAllocatedBytes();
    cyxwiz::Tensor output = graph->Forward(device_input);

    Check(output.Shape() == std::vector<size_t>({2, 1}),
          "TensorDot ArrayFire graph forward should produce row-wise output");
    Check(cyxwiz::MemoryManager::GetAllocatedBytes() == before_host_bytes,
          "TensorDot ArrayFire graph forward should not materialize host output");

    af::array output_device = output.GetArrayRowMajor2D();
    Check(output_device.dims(0) == 2,
          "TensorDot ArrayFire graph output should preserve device rows");
    Check(output_device.dims(1) == 1,
          "TensorDot ArrayFire graph output should preserve device columns");
    Check(cyxwiz::MemoryManager::GetAllocatedBytes() == before_host_bytes,
          "TensorDot ArrayFire graph output device access should not materialize host data");

    const cyxwiz::Tensor* cached = graph->FindCachedTensor(2, 203);
    Check(cached != nullptr, "TensorDot ArrayFire graph should cache output");
    af::array cached_device = cached->GetArrayRowMajor2D();
    Check(cached_device.dims(0) == 2,
          "TensorDot ArrayFire cached output should preserve device rows");
    Check(cached_device.dims(1) == 1,
          "TensorDot ArrayFire cached output should preserve device columns");
    Check(cyxwiz::MemoryManager::GetAllocatedBytes() == before_host_bytes,
          "TensorDot ArrayFire cached device access should not materialize host data");

    const float* out = output.Data<float>();
    CheckNear(out[0], 14.0f, 1e-4f,
              "TensorDot ArrayFire graph forward first row");
    CheckNear(out[1], 77.0f, 1e-4f,
              "TensorDot ArrayFire graph forward second row");
}
#endif

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
    CheckConcatOp();
    CheckBinaryMaskOp(gui::NodeType::TensorCompare,
                      {0.0f, 2.0f, -1.0f},
                      {1.0f, 1.0f, 1.0f},
                      "TensorCompare");
    CheckBinaryMaskOp(gui::NodeType::TensorLogicalMask,
                      {0.0f, 2.0f, -1.0f},
                      {0.0f, 1.0f, 1.0f},
                      "TensorLogicalMask");
#ifdef CYXWIZ_HAS_ARRAYFIRE
    CheckGraphFanInArrayFireResidency();
    CheckGraphFanInBackwardArrayFireResidency();
    CheckGraphMaskArrayFireResidency();
#endif
    CheckDotOp();
#ifdef CYXWIZ_HAS_ARRAYFIRE
    CheckDotOpArrayFireResidency();
#endif

    std::cout << "Graph executable model parity passed\n";
    return 0;
}
