#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cyxwiz/sequential.h>

#include "model_test_path.h"

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

TEST_CASE("Conv1DModule exposes clean SequentialModel parameter ownership",
          "[conv][module]") {
    auto module =
        std::make_unique<cyxwiz::Conv1DModule>(1, 1, 2, 1, 0, 1, true);

    float weight_values[] = {1.0f, 2.0f};
    float bias_values[] = {0.5f};
    module->SetParameters({
        {"weights", cyxwiz::Tensor(
                        {1, 1, 2}, weight_values,
                        cyxwiz::DataType::Float32)},
        {"bias", cyxwiz::Tensor(
                     {1}, bias_values, cyxwiz::DataType::Float32)},
    });

    cyxwiz::SequentialModel model;
    model.AddModule(std::move(module));

    const auto parameters = model.GetParameters();
    REQUIRE(parameters.count("layer0.weights") == 1);
    REQUIRE(parameters.count("layer0.bias") == 1);
    REQUIRE(parameters.count("layer0.grad_weights") == 0);
    REQUIRE(parameters.count("layer0.grad_bias") == 0);

    float input_values[] = {1.0f, 2.0f, 3.0f};
    const cyxwiz::Tensor input(
        {3, 1, 1}, input_values, cyxwiz::DataType::Float32);
    const cyxwiz::Tensor output = model.Forward(input);
    REQUIRE(output.Shape() == std::vector<size_t>{2, 1, 1});
    REQUIRE(output.Data<float>()[0] == Catch::Approx(5.5f));
    REQUIRE(output.Data<float>()[1] == Catch::Approx(8.5f));

    float grad_values[] = {1.0f, 2.0f};
    const cyxwiz::Tensor grad_output(
        {2, 1, 1}, grad_values, cyxwiz::DataType::Float32);
    const cyxwiz::Tensor grad_input = model.Backward(grad_output);
    REQUIRE(grad_input.Shape() == input.Shape());
    REQUIRE(grad_input.Data<float>()[0] == Catch::Approx(1.0f));
    REQUIRE(grad_input.Data<float>()[1] == Catch::Approx(4.0f));
    REQUIRE(grad_input.Data<float>()[2] == Catch::Approx(4.0f));

    const auto gradients = model.GetGradients();
    REQUIRE(gradients.count("layer0.weights") == 1);
    REQUIRE(gradients.count("layer0.bias") == 1);
    REQUIRE(gradients.count("layer0.grad_weights") == 0);
    REQUIRE(gradients.at("layer0.weights").Data<float>()[0] ==
            Catch::Approx(5.0f));
    REQUIRE(gradients.at("layer0.weights").Data<float>()[1] ==
            Catch::Approx(8.0f));
    REQUIRE(gradients.at("layer0.bias").Data<float>()[0] ==
            Catch::Approx(3.0f));

    cyxwiz::SGDOptimizer optimizer(0.01);
    model.UpdateParameters(&optimizer);
    const float weight_after =
        model.GetParameters().at("layer0.weights").Data<float>()[0];
    REQUIRE(weight_after == Catch::Approx(0.95f));

    const auto model_path =
        cyxwiz::test::UniqueModelPath("cyxwiz_conv1d_module_");
    REQUIRE(model.Save(model_path.string()));

    cyxwiz::SequentialModel restored;
    restored.Add<cyxwiz::Conv1DModule>(1, 1, 2, 1, 0, 1, true);
    REQUIRE(restored.Load(model_path.string()));
    REQUIRE(restored.GetParameters().at("layer0.weights").Data<float>()[0] ==
            Catch::Approx(weight_after));

    std::error_code remove_error;
    std::filesystem::remove(model_path, remove_error);
    REQUIRE_FALSE(remove_error);
}

TEST_CASE("Conv2DModule exposes clean SequentialModel parameter ownership",
          "[conv][module]") {
    auto module = std::make_unique<cyxwiz::Conv2DModule>(1, 1, 2, 1, 0, true);

    float weight_values[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
    };
    float bias_values[] = {0.5f};
    module->SetParameters({
        {"weights", cyxwiz::Tensor(
                        {2, 2, 1, 1}, weight_values,
                        cyxwiz::DataType::Float32)},
        {"bias", cyxwiz::Tensor(
                     {1}, bias_values, cyxwiz::DataType::Float32)},
    });

    cyxwiz::SequentialModel model;
    model.AddModule(std::move(module));

    const auto parameters = model.GetParameters();
    REQUIRE(parameters.count("layer0.weights") == 1);
    REQUIRE(parameters.count("layer0.bias") == 1);
    REQUIRE(parameters.count("layer0.grad_weights") == 0);
    REQUIRE(parameters.count("layer0.grad_bias") == 0);

    float input_values[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
    };
    const cyxwiz::Tensor input(
        {3, 3, 1, 1}, input_values, cyxwiz::DataType::Float32);
    const cyxwiz::Tensor output = model.Forward(input);
    REQUIRE(output.Shape() == std::vector<size_t>{2, 2, 1, 1});

    float grad_values[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const cyxwiz::Tensor grad_output(
        {2, 2, 1, 1}, grad_values, cyxwiz::DataType::Float32);
    const cyxwiz::Tensor grad_input = model.Backward(grad_output);
    REQUIRE(grad_input.Shape() == std::vector<size_t>{3, 3, 1, 1});

    const auto gradients = model.GetGradients();
    REQUIRE(gradients.count("layer0.weights") == 1);
    REQUIRE(gradients.count("layer0.bias") == 1);
    REQUIRE(gradients.count("layer0.grad_weights") == 0);
    REQUIRE(gradients.at("layer0.weights").Shape() ==
            std::vector<size_t>{2, 2, 1, 1});
    REQUIRE(gradients.at("layer0.bias").Shape() ==
            std::vector<size_t>{1});

    const float weight_before =
        model.GetParameters().at("layer0.weights").Data<float>()[0];
    cyxwiz::SGDOptimizer optimizer(0.01);
    model.UpdateParameters(&optimizer);
    const float weight_after =
        model.GetParameters().at("layer0.weights").Data<float>()[0];
    REQUIRE(weight_before == Catch::Approx(1.0f));
    REQUIRE(weight_after == Catch::Approx(0.63f));

    const auto model_path =
        cyxwiz::test::UniqueModelPath("cyxwiz_conv2d_module_");
    REQUIRE(model.Save(model_path.string()));

    cyxwiz::SequentialModel restored;
    restored.Add<cyxwiz::Conv2DModule>(1, 1, 2, 1, 0, true);
    REQUIRE(restored.Load(model_path.string()));
    REQUIRE(restored.GetParameters().at("layer0.weights").Data<float>()[0] ==
            Catch::Approx(weight_after));

    std::error_code remove_error;
    std::filesystem::remove(model_path, remove_error);
    REQUIRE_FALSE(remove_error);
}

TEST_CASE(
    "ConvTranspose2DModule exposes clean SequentialModel parameter ownership",
    "[conv_transpose][module]") {
    auto module = std::make_unique<cyxwiz::ConvTranspose2DModule>(
        1, 1, 2, 2, 0, 0, true);

    float weight_values[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
    };
    float bias_values[] = {0.5f};
    module->SetParameters({
        {"weights", cyxwiz::Tensor(
                        {2, 2, 1, 1}, weight_values,
                        cyxwiz::DataType::Float32)},
        {"bias", cyxwiz::Tensor(
                     {1}, bias_values, cyxwiz::DataType::Float32)},
    });

    cyxwiz::SequentialModel model;
    model.AddModule(std::move(module));

    const auto parameters = model.GetParameters();
    REQUIRE(parameters.count("layer0.weights") == 1);
    REQUIRE(parameters.count("layer0.bias") == 1);
    REQUIRE(parameters.count("layer0.grad_weights") == 0);
    REQUIRE(parameters.count("layer0.grad_bias") == 0);
    REQUIRE(model.GetModule(0)->GetName().find("output_padding=0") !=
            std::string::npos);

    float input_values[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
    };
    const cyxwiz::Tensor input(
        {2, 2, 1, 1}, input_values, cyxwiz::DataType::Float32);
    const cyxwiz::Tensor output = model.Forward(input);
    REQUIRE(output.Shape() == std::vector<size_t>{4, 4, 1, 1});
    REQUIRE(output.Data<float>()[0] == Catch::Approx(1.5f));
    REQUIRE(output.Data<float>()[15] == Catch::Approx(16.5f));

    float grad_values[16];
    for (float& value : grad_values) {
        value = 1.0f;
    }
    const cyxwiz::Tensor grad_output(
        {4, 4, 1, 1}, grad_values, cyxwiz::DataType::Float32);
    const cyxwiz::Tensor grad_input = model.Backward(grad_output);
    REQUIRE(grad_input.Shape() == input.Shape());
    for (size_t i = 0; i < grad_input.NumElements(); ++i) {
        REQUIRE(grad_input.Data<float>()[i] == Catch::Approx(10.0f));
    }

    const auto gradients = model.GetGradients();
    REQUIRE(gradients.count("layer0.weights") == 1);
    REQUIRE(gradients.count("layer0.bias") == 1);
    REQUIRE(gradients.count("layer0.grad_weights") == 0);
    REQUIRE(gradients.at("layer0.weights").Data<float>()[0] ==
            Catch::Approx(10.0f));
    REQUIRE(gradients.at("layer0.bias").Data<float>()[0] ==
            Catch::Approx(16.0f));

    cyxwiz::SGDOptimizer optimizer(0.01);
    model.UpdateParameters(&optimizer);
    const float weight_after =
        model.GetParameters().at("layer0.weights").Data<float>()[0];
    REQUIRE(weight_after == Catch::Approx(0.9f));

    const auto model_path =
        cyxwiz::test::UniqueModelPath("cyxwiz_conv_transpose2d_module_");
    REQUIRE(model.Save(model_path.string()));

    cyxwiz::SequentialModel restored;
    restored.Add<cyxwiz::ConvTranspose2DModule>(1, 1, 2, 2, 0, 0, true);
    REQUIRE(restored.Load(model_path.string()));
    REQUIRE(restored.GetParameters().at("layer0.weights").Data<float>()[0] ==
            Catch::Approx(weight_after));

    std::error_code remove_error;
    std::filesystem::remove(model_path, remove_error);
    REQUIRE_FALSE(remove_error);
}

TEST_CASE("Upsample2DModule provides parameter-free SequentialModel ownership",
          "[upsample][module]") {
    float input_values[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
    };
    const cyxwiz::Tensor input(
        {2, 2, 1, 1}, input_values, cyxwiz::DataType::Float32);

    for (const auto mode : {cyxwiz::UpsampleMode::Nearest,
                            cyxwiz::UpsampleMode::Bilinear}) {
        cyxwiz::SequentialModel model;
        model.Add<cyxwiz::Upsample2DModule>(2, mode);

        const cyxwiz::Tensor output = model.Forward(input);
        REQUIRE(output.Shape() == std::vector<size_t>{4, 4, 1, 1});
        REQUIRE(output.Data<float>()[0] == Catch::Approx(1.0f));
        REQUIRE(output.Data<float>()[15] == Catch::Approx(4.0f));

        float grad_values[16];
        for (float& value : grad_values) {
            value = 1.0f;
        }
        const cyxwiz::Tensor grad_output(
            {4, 4, 1, 1}, grad_values, cyxwiz::DataType::Float32);
        const cyxwiz::Tensor grad_input = model.Backward(grad_output);
        REQUIRE(grad_input.Shape() == input.Shape());
        for (size_t i = 0; i < grad_input.NumElements(); ++i) {
            REQUIRE(grad_input.Data<float>()[i] == Catch::Approx(4.0f));
        }
        REQUIRE(model.GetParameters().empty());
        REQUIRE(model.GetGradients().empty());
    }
}

TEST_CASE("PixelShuffleModule provides parameter-free SequentialModel ownership",
          "[pixelshuffle][module]") {
    float input_values[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const cyxwiz::Tensor input(
        {1, 1, 4, 1}, input_values, cyxwiz::DataType::Float32);

    cyxwiz::SequentialModel model;
    model.Add<cyxwiz::PixelShuffleModule>(2);
    const cyxwiz::Tensor output = model.Forward(input);
    REQUIRE(output.Shape() == std::vector<size_t>{2, 2, 1, 1});
    for (size_t i = 0; i < output.NumElements(); ++i) {
        REQUIRE(output.Data<float>()[i] == Catch::Approx(input_values[i]));
    }

    float grad_values[] = {10.0f, 20.0f, 30.0f, 40.0f};
    const cyxwiz::Tensor grad_output(
        {2, 2, 1, 1}, grad_values, cyxwiz::DataType::Float32);
    const cyxwiz::Tensor grad_input = model.Backward(grad_output);
    REQUIRE(grad_input.Shape() == input.Shape());
    for (size_t i = 0; i < grad_input.NumElements(); ++i) {
        REQUIRE(grad_input.Data<float>()[i] == Catch::Approx(grad_values[i]));
    }
    REQUIRE(model.GetParameters().empty());
    REQUIRE(model.GetGradients().empty());
}

TEST_CASE("Pooling modules provide parameter-free SequentialModel ownership",
          "[pool][module]") {
    float input_values[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
    };
    const cyxwiz::Tensor input(
        {2, 2, 1, 1}, input_values, cyxwiz::DataType::Float32);

    SECTION("MaxPool2D") {
        cyxwiz::SequentialModel model;
        model.Add<cyxwiz::MaxPool2DModule>(2, 2, 0);
        const cyxwiz::Tensor output = model.Forward(input);
        REQUIRE(output.Shape() == std::vector<size_t>{1, 1, 1, 1});
        REQUIRE(output.Data<float>()[0] == Catch::Approx(4.0f));

        float grad_values[] = {2.0f};
        const cyxwiz::Tensor grad_output(
            {1, 1, 1, 1}, grad_values, cyxwiz::DataType::Float32);
        const cyxwiz::Tensor grad_input = model.Backward(grad_output);
        REQUIRE(grad_input.Shape() == input.Shape());
        REQUIRE(grad_input.Data<float>()[3] == Catch::Approx(2.0f));
        REQUIRE(model.GetParameters().empty());
        REQUIRE(model.GetGradients().empty());
    }

    SECTION("AvgPool2D") {
        cyxwiz::SequentialModel model;
        model.Add<cyxwiz::AvgPool2DModule>(2, 2, 0);
        const cyxwiz::Tensor output = model.Forward(input);
        REQUIRE(output.Shape() == std::vector<size_t>{1, 1, 1, 1});
        REQUIRE(output.Data<float>()[0] == Catch::Approx(2.5f));

        float grad_values[] = {1.0f};
        const cyxwiz::Tensor grad_output(
            {1, 1, 1, 1}, grad_values, cyxwiz::DataType::Float32);
        const cyxwiz::Tensor grad_input = model.Backward(grad_output);
        REQUIRE(grad_input.Shape() == input.Shape());
        for (size_t i = 0; i < grad_input.NumElements(); ++i) {
            REQUIRE(grad_input.Data<float>()[i] == Catch::Approx(0.25f));
        }
        REQUIRE(model.GetParameters().empty());
        REQUIRE(model.GetGradients().empty());
    }

    SECTION("GlobalAvgPool2D") {
        cyxwiz::SequentialModel model;
        model.Add<cyxwiz::GlobalAvgPool2DModule>();
        const cyxwiz::Tensor output = model.Forward(input);
        REQUIRE(output.Shape() == std::vector<size_t>{1, 1});
        REQUIRE(output.Data<float>()[0] == Catch::Approx(2.5f));

        float grad_values[] = {1.0f};
        const cyxwiz::Tensor grad_output(
            {1, 1}, grad_values, cyxwiz::DataType::Float32);
        const cyxwiz::Tensor grad_input = model.Backward(grad_output);
        REQUIRE(grad_input.Shape() == input.Shape());
        for (size_t i = 0; i < grad_input.NumElements(); ++i) {
            REQUIRE(grad_input.Data<float>()[i] == Catch::Approx(0.25f));
        }
        REQUIRE(model.GetParameters().empty());
        REQUIRE(model.GetGradients().empty());
    }
}

TEST_CASE("Pooling layers reject invalid construction", "[pool][validation]") {
    REQUIRE_THROWS_AS(cyxwiz::MaxPool2DLayer(0), std::invalid_argument);
    REQUIRE_THROWS_AS(cyxwiz::MaxPool2DLayer(2, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(cyxwiz::MaxPool2DLayer(2, 2, -1), std::invalid_argument);
    REQUIRE_THROWS_AS(cyxwiz::AvgPool2DLayer(0), std::invalid_argument);
    REQUIRE_THROWS_AS(cyxwiz::AvgPool2DLayer(2, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(cyxwiz::AvgPool2DLayer(2, 2, -1), std::invalid_argument);
}
