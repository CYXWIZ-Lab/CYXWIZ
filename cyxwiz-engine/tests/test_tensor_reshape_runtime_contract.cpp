#include "../src/core/model_builder.h"

#include <cyxwiz/sequential.h>
#include <cyxwiz/tensor.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

void CheckShape(const cyxwiz::Tensor& tensor,
                const std::vector<size_t>& expected,
                const std::string& message) {
    Check(tensor.Shape() == expected, message);
}

void CheckNear(float actual, float expected, float tolerance, const std::string& message) {
    if (std::fabs(actual - expected) > tolerance) {
        std::cerr << "FAIL: " << message
                  << " actual=" << actual
                  << " expected=" << expected
                  << "\n";
        std::exit(1);
    }
}

cyxwiz::Tensor MakeTensor(const std::vector<size_t>& shape,
                          const std::vector<float>& values) {
    size_t elements = 1;
    for (size_t dim : shape) {
        elements *= dim;
    }
    Check(elements == values.size(), "MakeTensor value count must match shape");
    return cyxwiz::Tensor(shape, values.data(), cyxwiz::DataType::Float32);
}

cyxwiz::Tensor MakeRangeTensor(const std::vector<size_t>& shape) {
    size_t elements = 1;
    for (size_t dim : shape) {
        elements *= dim;
    }

    std::vector<float> values(elements);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i);
    }
    return cyxwiz::Tensor(shape, values.data(), cyxwiz::DataType::Float32);
}

void CheckUnaryModule(cyxwiz::TensorUnaryOp op,
                      const std::vector<float>& input_values,
                      const std::vector<float>& expected_output,
                      const std::vector<float>& expected_backward,
                      const std::string& name,
                      float scalar = 0.0f,
                      float scalar2 = 0.0f) {
    cyxwiz::Tensor input = MakeTensor({1, input_values.size()}, input_values);
    cyxwiz::TensorUnaryModule module(op, scalar, scalar2);

    cyxwiz::Tensor output = module.Forward(input);
    CheckShape(output, {1, input_values.size()}, name + " forward should preserve shape");
    for (size_t i = 0; i < expected_output.size(); ++i) {
        CheckNear(output.At(0, i), expected_output[i], 1e-4f,
                  name + " forward value mismatch");
    }

    cyxwiz::Tensor grad = cyxwiz::Tensor::Ones({1, input_values.size()});
    cyxwiz::Tensor backward = module.Backward(grad);
    CheckShape(backward, {1, input_values.size()}, name + " backward should preserve shape");
    for (size_t i = 0; i < expected_backward.size(); ++i) {
        CheckNear(backward.At(0, i), expected_backward[i], 1e-4f,
                  name + " backward value mismatch");
    }
}

void CheckReductionModule(cyxwiz::TensorReductionOp op,
                          int dim,
                          bool keepdim,
                          const std::vector<size_t>& expected_shape,
                          const std::vector<float>& expected_output,
                          const std::vector<float>& expected_backward,
                          const std::string& name) {
    cyxwiz::Tensor input = MakeRangeTensor({2, 2, 3});
    cyxwiz::TensorReductionModule module(op, dim, keepdim);

    cyxwiz::Tensor output = module.Forward(input);
    CheckShape(output, expected_shape, name + " forward shape mismatch");
    for (size_t i = 0; i < expected_output.size(); ++i) {
        CheckNear(output.At(i), expected_output[i], 1e-4f,
                  name + " forward value mismatch");
    }

    cyxwiz::Tensor grad = cyxwiz::Tensor::Ones(expected_shape);
    cyxwiz::Tensor backward = module.Backward(grad);
    CheckShape(backward, {2, 2, 3}, name + " backward shape mismatch");
    for (size_t i = 0; i < expected_backward.size(); ++i) {
        CheckNear(backward.At(i), expected_backward[i], 1e-4f,
                  name + " backward value mismatch");
    }
}

} // namespace

int main() {
    {
        cyxwiz::Tensor input = MakeRangeTensor({2, 6});
        cyxwiz::ReshapeModule reshape({2, 3});

        cyxwiz::Tensor output = reshape.Forward(input);
        CheckShape(output, {2, 2, 3}, "ReshapeModule should preserve batch dimension");
        Check(output.At(1, 1, 2) == 11.0f, "ReshapeModule should preserve row-major values");

        cyxwiz::Tensor grad = MakeRangeTensor({2, 2, 3});
        cyxwiz::Tensor backward = reshape.Backward(grad);
        CheckShape(backward, {2, 6}, "ReshapeModule backward should restore original shape");
        Check(backward.At(11) == 11.0f, "ReshapeModule backward should preserve values");
    }

    {
        cyxwiz::Tensor input = MakeRangeTensor({2, 2, 3});
        cyxwiz::PermuteModule permute({1, 0});

        cyxwiz::Tensor output = permute.Forward(input);
        CheckShape(output, {2, 3, 2}, "PermuteModule should preserve batch dimension");
        Check(output.At(0, 2, 1) == input.At(0, 1, 2),
              "PermuteModule should reorder sample dimensions");

        cyxwiz::Tensor backward = permute.Backward(output);
        CheckShape(backward, {2, 2, 3}, "PermuteModule backward should invert shape");
        Check(backward.At(1, 1, 2) == input.At(1, 1, 2),
              "PermuteModule backward should invert data order");
    }

    {
        CheckUnaryModule(cyxwiz::TensorUnaryOp::Abs,
                         {-2.0f, 0.0f, 3.0f},
                         {2.0f, 0.0f, 3.0f},
                         {-1.0f, 0.0f, 1.0f},
                         "TensorAbs");

        CheckUnaryModule(cyxwiz::TensorUnaryOp::Exp,
                         {0.0f, 1.0f},
                         {1.0f, std::exp(1.0f)},
                         {1.0f, std::exp(1.0f)},
                         "TensorExp");

        CheckUnaryModule(cyxwiz::TensorUnaryOp::Log,
                         {1.0f, std::exp(1.0f)},
                         {0.0f, 1.0f},
                         {1.0f, 1.0f / std::exp(1.0f)},
                         "TensorLog");

        CheckUnaryModule(cyxwiz::TensorUnaryOp::Sqrt,
                         {1.0f, 4.0f},
                         {1.0f, 2.0f},
                         {0.5f, 0.25f},
                         "TensorSqrt");

        CheckUnaryModule(cyxwiz::TensorUnaryOp::Sign,
                         {-2.0f, 0.0f, 3.0f},
                         {-1.0f, 0.0f, 1.0f},
                         {0.0f, 0.0f, 0.0f},
                         "TensorSign");

        CheckUnaryModule(cyxwiz::TensorUnaryOp::Pow,
                         {2.0f, 3.0f},
                         {8.0f, 27.0f},
                         {12.0f, 27.0f},
                         "TensorPow",
                         3.0f);

        CheckUnaryModule(cyxwiz::TensorUnaryOp::Clip,
                         {-1.0f, 0.5f, 2.0f},
                         {0.0f, 0.5f, 1.0f},
                         {0.0f, 1.0f, 0.0f},
                         "TensorClip",
                         0.0f,
                         1.0f);
    }

    {
        CheckReductionModule(cyxwiz::TensorReductionOp::Sum,
                             -1,
                             false,
                             {2, 1},
                             {15.0f, 51.0f},
                             {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                              1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
                             "TensorSumAll");

        CheckReductionModule(cyxwiz::TensorReductionOp::Mean,
                             -1,
                             false,
                             {2, 1},
                             {2.5f, 8.5f},
                             {1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f,
                              1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f,
                              1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f,
                              1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f},
                             "TensorMeanAll");

        CheckReductionModule(cyxwiz::TensorReductionOp::Sum,
                             1,
                             false,
                             {2, 2},
                             {3.0f, 12.0f, 21.0f, 30.0f},
                             {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                              1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
                             "TensorSumDim");
    }

    {
        cyxwiz::TrainingConfiguration config;
        config.input_size = 6;
        config.output_size = 6;
        config.loss_type = gui::NodeType::MSELoss;
        config.optimizer_type = gui::NodeType::Adam;

        cyxwiz::CompiledLayer reshape;
        reshape.type = gui::NodeType::Reshape;
        reshape.node_id = 1;
        reshape.name = "Reshape";
        reshape.input_shape = {6};
        reshape.output_shape = {1, 6};

        cyxwiz::CompiledLayer squeeze;
        squeeze.type = gui::NodeType::Squeeze;
        squeeze.node_id = 2;
        squeeze.name = "Squeeze";
        squeeze.input_shape = {1, 6};
        squeeze.output_shape = {6};

        cyxwiz::CompiledLayer unsqueeze;
        unsqueeze.type = gui::NodeType::Unsqueeze;
        unsqueeze.node_id = 3;
        unsqueeze.name = "Unsqueeze";
        unsqueeze.input_shape = {6};
        unsqueeze.output_shape = {6, 1};

        cyxwiz::CompiledLayer permute;
        permute.type = gui::NodeType::Permute;
        permute.node_id = 4;
        permute.name = "Permute";
        permute.input_shape = {6, 1};
        permute.output_shape = {1, 6};
        permute.dims = {1, 0};

        cyxwiz::CompiledLayer view;
        view.type = gui::NodeType::View;
        view.node_id = 5;
        view.name = "View";
        view.input_shape = {1, 6};
        view.output_shape = {3, 2};

        cyxwiz::CompiledLayer abs;
        abs.type = gui::NodeType::TensorAbs;
        abs.node_id = 6;
        abs.name = "TensorAbs";
        abs.input_shape = {3, 2};
        abs.output_shape = {3, 2};

        cyxwiz::CompiledLayer exp;
        exp.type = gui::NodeType::TensorExp;
        exp.node_id = 7;
        exp.name = "TensorExp";
        exp.input_shape = {3, 2};
        exp.output_shape = {3, 2};

        cyxwiz::CompiledLayer log;
        log.type = gui::NodeType::TensorLog;
        log.node_id = 8;
        log.name = "TensorLog";
        log.input_shape = {3, 2};
        log.output_shape = {3, 2};

        cyxwiz::CompiledLayer sqrt;
        sqrt.type = gui::NodeType::TensorSqrt;
        sqrt.node_id = 9;
        sqrt.name = "TensorSqrt";
        sqrt.input_shape = {3, 2};
        sqrt.output_shape = {3, 2};

        cyxwiz::CompiledLayer pow;
        pow.type = gui::NodeType::TensorPow;
        pow.node_id = 10;
        pow.name = "TensorPow";
        pow.input_shape = {3, 2};
        pow.output_shape = {3, 2};
        pow.parameters = {{"exponent", "2.0"}};

        cyxwiz::CompiledLayer clip;
        clip.type = gui::NodeType::TensorClip;
        clip.node_id = 11;
        clip.name = "TensorClip";
        clip.input_shape = {3, 2};
        clip.output_shape = {3, 2};
        clip.parameters = {{"min", "0.0"}, {"max", "10.0"}};

        cyxwiz::CompiledLayer sign;
        sign.type = gui::NodeType::TensorSign;
        sign.node_id = 12;
        sign.name = "TensorSign";
        sign.input_shape = {3, 2};
        sign.output_shape = {3, 2};

        cyxwiz::CompiledLayer mean;
        mean.type = gui::NodeType::TensorMean;
        mean.node_id = 13;
        mean.name = "TensorMean";
        mean.input_shape = {3, 2};
        mean.output_shape = {1};
        mean.parameters = {{"dim", "-1"}, {"keepdim", "false"}};

        cyxwiz::CompiledLayer sum;
        sum.type = gui::NodeType::TensorSum;
        sum.node_id = 14;
        sum.name = "TensorSum";
        sum.input_shape = {1};
        sum.output_shape = {1};
        sum.parameters = {{"dim", "-1"}, {"keepdim", "false"}};

        config.layers = {
            reshape, squeeze, unsqueeze, permute, view,
            abs, exp, log, sqrt, pow, clip, sign, mean, sum
        };

        cyxwiz::BuiltModel built = cyxwiz::BuildSequentialFromConfig(config);
        Check(built.ok(), "BuildSequentialFromConfig should build bounded tensor op modules");

        cyxwiz::Tensor input = MakeTensor({2, 6}, {
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
            7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f
        });
        cyxwiz::Tensor output = built.model->Forward(input);
        CheckShape(output, {2, 1}, "Sequential tensor op forward shape mismatch");
        CheckNear(output.At(1, 0), 1.0f, 1e-4f,
                  "Sequential tensor ops should transform and reduce values");

        cyxwiz::Tensor grad = MakeRangeTensor({2, 1});
        cyxwiz::Tensor backward = built.model->Backward(grad);
        CheckShape(backward, {2, 6}, "Sequential tensor op backward shape mismatch");
        Check(std::isfinite(backward.At(11)),
              "Sequential tensor backward should produce finite gradients");
    }

    std::cout << "Tensor runtime contract passed\n";
    return 0;
}
