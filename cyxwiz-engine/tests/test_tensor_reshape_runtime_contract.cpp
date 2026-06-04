#include "../src/core/model_builder.h"

#include <cyxwiz/sequential.h>
#include <cyxwiz/tensor.h>

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

        config.layers = {reshape, squeeze, unsqueeze, permute, view};

        cyxwiz::BuiltModel built = cyxwiz::BuildSequentialFromConfig(config);
        Check(built.ok(), "BuildSequentialFromConfig should build bounded shape op modules");

        cyxwiz::Tensor input = MakeRangeTensor({2, 6});
        cyxwiz::Tensor output = built.model->Forward(input);
        CheckShape(output, {2, 3, 2}, "Sequential shape op forward shape mismatch");
        Check(output.At(1, 2, 1) == 11.0f, "Sequential shape ops should preserve values");

        cyxwiz::Tensor grad = MakeRangeTensor({2, 3, 2});
        cyxwiz::Tensor backward = built.model->Backward(grad);
        CheckShape(backward, {2, 6}, "Sequential shape op backward shape mismatch");
        Check(backward.At(11) == 11.0f, "Sequential shape op backward should preserve values");
    }

    std::cout << "Tensor shape runtime contract passed\n";
    return 0;
}
