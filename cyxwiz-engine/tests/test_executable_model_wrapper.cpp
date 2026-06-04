#include "../src/core/executable_model.h"
#include "../src/core/model_builder.h"

#include <cyxwiz/sequential.h>
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

} // namespace

int main() {
    {
        auto sequential = std::make_unique<cyxwiz::SequentialModel>();
        sequential->Add<cyxwiz::TensorUnaryModule>(cyxwiz::TensorUnaryOp::Abs);

        cyxwiz::SequentialExecutableModel wrapper(std::move(sequential));
        Check(wrapper.AsSequentialModel() != nullptr,
              "SequentialExecutableModel should expose wrapped sequential model");

        cyxwiz::Tensor output = wrapper.Forward(MakeTensor({-2.0f, 0.0f, 3.0f}));
        Check(output.Shape() == std::vector<size_t>({1, 3}),
              "wrapper forward should preserve shape");
        CheckNear(output.At(0, 0), 2.0f, 1e-4f,
                  "wrapper forward should delegate to sequential model");
        CheckNear(output.At(0, 2), 3.0f, 1e-4f,
                  "wrapper forward should preserve positive values");

        cyxwiz::Tensor grad = cyxwiz::Tensor::Ones({1, 3});
        cyxwiz::Tensor backward = wrapper.Backward(grad);
        Check(backward.Shape() == std::vector<size_t>({1, 3}),
              "wrapper backward should preserve shape");
        CheckNear(backward.At(0, 0), -1.0f, 1e-4f,
                  "wrapper backward should delegate negative sign gradient");
        CheckNear(backward.At(0, 1), 0.0f, 1e-4f,
                  "wrapper backward should delegate zero sign gradient");
        CheckNear(backward.At(0, 2), 1.0f, 1e-4f,
                  "wrapper backward should delegate positive sign gradient");
    }

    {
        cyxwiz::TrainingConfiguration config;
        config.input_size = 3;
        config.output_size = 3;
        config.input_shape = {3};
        config.layers.push_back(TensorAbsLayer());
        config.loss_type = gui::NodeType::MSELoss;
        config.optimizer_type = gui::NodeType::SGD;
        config.learning_rate = 0.01f;

        cyxwiz::BuiltExecutableModel built =
            cyxwiz::BuildExecutableFromConfig(config);
        Check(built.ok(), "BuildExecutableFromConfig should wrap sequential model");
        Check(built.loss != nullptr, "BuildExecutableFromConfig should build loss");
        Check(built.optimizer != nullptr, "BuildExecutableFromConfig should build optimizer");
        Check(built.model->AsSequentialModel() != nullptr,
              "current executable builder should return sequential wrapper");

        cyxwiz::Tensor output = built.model->Forward(MakeTensor({-4.0f, 5.0f, -6.0f}));
        Check(output.Shape() == std::vector<size_t>({1, 3}),
              "built executable forward should preserve shape");
        CheckNear(output.At(0, 0), 4.0f, 1e-4f,
                  "built executable should run TensorAbs");
        CheckNear(output.At(0, 1), 5.0f, 1e-4f,
                  "built executable should preserve positive TensorAbs values");
        CheckNear(output.At(0, 2), 6.0f, 1e-4f,
                  "built executable should run TensorAbs for final value");
    }

    std::cout << "Executable model wrapper passed\n";
    return 0;
}
