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
        reshape.output_shape = {2, 3};

        cyxwiz::CompiledLayer view;
        view.type = gui::NodeType::View;
        view.node_id = 2;
        view.name = "View";
        view.input_shape = {2, 3};
        view.output_shape = {3, 2};

        config.layers = {reshape, view};

        cyxwiz::BuiltModel built = cyxwiz::BuildSequentialFromConfig(config);
        Check(built.ok(), "BuildSequentialFromConfig should build Reshape/View modules");

        cyxwiz::Tensor input = MakeRangeTensor({2, 6});
        cyxwiz::Tensor output = built.model->Forward(input);
        CheckShape(output, {2, 3, 2}, "Sequential Reshape/View forward shape mismatch");
        Check(output.At(1, 2, 1) == 11.0f, "Sequential Reshape/View should preserve values");

        cyxwiz::Tensor grad = MakeRangeTensor({2, 3, 2});
        cyxwiz::Tensor backward = built.model->Backward(grad);
        CheckShape(backward, {2, 6}, "Sequential Reshape/View backward shape mismatch");
        Check(backward.At(11) == 11.0f, "Sequential Reshape/View backward should preserve values");
    }

    std::cout << "Tensor reshape runtime contract passed\n";
    return 0;
}
