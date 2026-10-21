#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cyxwiz/sequential.h>

#include "model_test_path.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

TEST_CASE("GroupNormModule exposes canonical affine ownership",
          "[norm][module]") {
    auto module = std::make_unique<cyxwiz::GroupNormModule>(
        1, 2, 1e-5f, true);

    float gamma_values[] = {1.0f, 2.0f};
    float beta_values[] = {0.5f, -1.0f};
    module->SetParameters({
        {"gamma", cyxwiz::Tensor(
                      {2}, gamma_values, cyxwiz::DataType::Float32)},
        {"beta", cyxwiz::Tensor(
                     {2}, beta_values, cyxwiz::DataType::Float32)},
    });

    cyxwiz::SequentialModel model;
    model.AddModule(std::move(module));
    const auto parameters = model.GetParameters();
    REQUIRE(parameters.count("layer0.gamma") == 1);
    REQUIRE(parameters.count("layer0.beta") == 1);
    REQUIRE(parameters.count("layer0.grad_gamma") == 0);
    REQUIRE(parameters.count("layer0.grad_beta") == 0);

    float input_values[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const cyxwiz::Tensor input(
        {1, 2, 2, 1}, input_values, cyxwiz::DataType::Float32);
    const cyxwiz::Tensor output = model.Forward(input);
    REQUIRE(output.Shape() == input.Shape());
    REQUIRE(output.Data<float>()[0] ==
            Catch::Approx(-0.8416354f).margin(1e-5f));

    float grad_values[] = {1.0f, 0.0f, 0.0f, 1.0f};
    const cyxwiz::Tensor grad_output(
        {1, 2, 2, 1}, grad_values, cyxwiz::DataType::Float32);
    REQUIRE(model.Backward(grad_output).Shape() == input.Shape());

    const auto gradients = model.GetGradients();
    REQUIRE(gradients.count("layer0.gamma") == 1);
    REQUIRE(gradients.count("layer0.beta") == 1);
    REQUIRE(gradients.count("layer0.grad_gamma") == 0);
    REQUIRE(gradients.at("layer0.gamma").Data<float>()[0] ==
            Catch::Approx(-1.3416354f).margin(1e-5f));
    REQUIRE(gradients.at("layer0.beta").Data<float>()[0] ==
            Catch::Approx(1.0f));

    cyxwiz::SGDOptimizer optimizer(0.01);
    model.UpdateParameters(&optimizer);
    const float gamma_after =
        model.GetParameters().at("layer0.gamma").Data<float>()[0];
    REQUIRE(gamma_after == Catch::Approx(1.0134164f).margin(1e-6f));

    const auto model_path =
        cyxwiz::test::UniqueModelPath("cyxwiz_group_norm_module_");
    REQUIRE(model.Save(model_path.string()));
    cyxwiz::SequentialModel restored;
    restored.Add<cyxwiz::GroupNormModule>(1, 2, 1e-5f, true);
    REQUIRE(restored.Load(model_path.string()));
    REQUIRE(restored.GetParameters().at("layer0.gamma").Data<float>()[0] ==
            Catch::Approx(gamma_after));

    std::error_code remove_error;
    std::filesystem::remove(model_path, remove_error);
    REQUIRE_FALSE(remove_error);
}

TEST_CASE("InstanceNorm2DModule exposes canonical affine ownership",
          "[norm][module]") {
    auto module = std::make_unique<cyxwiz::InstanceNorm2DModule>(
        1, 1e-5f, true);

    float gamma_values[] = {2.0f};
    float beta_values[] = {-1.0f};
    module->SetParameters({
        {"gamma", cyxwiz::Tensor(
                      {1}, gamma_values, cyxwiz::DataType::Float32)},
        {"beta", cyxwiz::Tensor(
                     {1}, beta_values, cyxwiz::DataType::Float32)},
    });

    cyxwiz::SequentialModel model;
    model.AddModule(std::move(module));
    const auto parameters = model.GetParameters();
    REQUIRE(parameters.count("layer0.gamma") == 1);
    REQUIRE(parameters.count("layer0.beta") == 1);
    REQUIRE(parameters.count("layer0.grad_gamma") == 0);
    REQUIRE(parameters.count("layer0.grad_beta") == 0);

    float input_values[] = {
        1.0f, 2.0f,
        2.0f, 4.0f,
        3.0f, 6.0f,
        4.0f, 8.0f,
    };
    const cyxwiz::Tensor input(
        {2, 2, 1, 2}, input_values, cyxwiz::DataType::Float32);
    const cyxwiz::Tensor output = model.Forward(input);
    REQUIRE(output.Shape() == input.Shape());
    REQUIRE(output.Data<float>()[0] ==
            Catch::Approx(-3.6832708f).margin(1e-5f));

    float grad_values[] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
        0.0f, 0.0f,
    };
    const cyxwiz::Tensor grad_output(
        {2, 2, 1, 2}, grad_values, cyxwiz::DataType::Float32);
    REQUIRE(model.Backward(grad_output).Shape() == input.Shape());

    const auto gradients = model.GetGradients();
    REQUIRE(gradients.count("layer0.gamma") == 1);
    REQUIRE(gradients.count("layer0.beta") == 1);
    REQUIRE(gradients.count("layer0.grad_gamma") == 0);
    REQUIRE(gradients.at("layer0.gamma").Data<float>()[0] ==
            Catch::Approx(-1.7888486f).margin(1e-5f));
    REQUIRE(gradients.at("layer0.beta").Data<float>()[0] ==
            Catch::Approx(2.0f));

    cyxwiz::SGDOptimizer optimizer(0.01);
    model.UpdateParameters(&optimizer);
    const float gamma_after =
        model.GetParameters().at("layer0.gamma").Data<float>()[0];
    REQUIRE(gamma_after == Catch::Approx(2.0178885f).margin(1e-6f));

    const auto model_path =
        cyxwiz::test::UniqueModelPath("cyxwiz_instance_norm_module_");
    REQUIRE(model.Save(model_path.string()));
    cyxwiz::SequentialModel restored;
    restored.Add<cyxwiz::InstanceNorm2DModule>(1, 1e-5f, true);
    REQUIRE(restored.Load(model_path.string()));
    REQUIRE(restored.GetParameters().at("layer0.gamma").Data<float>()[0] ==
            Catch::Approx(gamma_after));

    std::error_code remove_error;
    std::filesystem::remove(model_path, remove_error);
    REQUIRE_FALSE(remove_error);
}

TEST_CASE("Spatial normalization modules omit state when affine is disabled",
          "[norm][module]") {
    cyxwiz::GroupNormModule group_norm(1, 2, 1e-5f, false);
    REQUIRE_FALSE(group_norm.HasParameters());
    REQUIRE(group_norm.GetParameters().empty());
    REQUIRE(group_norm.GetGradients().empty());

    cyxwiz::InstanceNorm2DModule instance_norm(2, 1e-5f, false);
    REQUIRE_FALSE(instance_norm.HasParameters());
    REQUIRE(instance_norm.GetParameters().empty());
    REQUIRE(instance_norm.GetGradients().empty());
}
