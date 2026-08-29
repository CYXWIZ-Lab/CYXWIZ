#include "../../cyxwiz-engine/src/core/language_model_training.h"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cyxwiz/layer.h>
#include <cyxwiz/tensor.h>
#include "algorithms/arrayfire_backend_utils.h"
#include <map>
#include <stdexcept>
#include <vector>

TEST_CASE("DenseLayer supports unbiased rank-1 affine gradients",
          "[dense][layer][rank1]") {
    cyxwiz::DenseLayer layer(3, 2, false);
    const float weight_values[] = {
        0.5f, -1.0f, 2.0f,
        -0.25f, 0.75f, 1.5f,
    };
    layer.SetParameters({
        {"weights", cyxwiz::Tensor(
                        {2, 3}, weight_values, cyxwiz::DataType::Float32)},
    });

    const float input_values[] = {1.5f, -2.0f, 0.25f};
    const auto output = layer.Forward(cyxwiz::Tensor(
        {3}, input_values, cyxwiz::DataType::Float32));
    REQUIRE(output.Shape() == std::vector<size_t>{2});
    REQUIRE(output.Data<float>()[0] == Catch::Approx(3.25f));
    REQUIRE(output.Data<float>()[1] == Catch::Approx(-1.5f));

    const float grad_values[] = {-0.75f, 1.25f};
    const auto grad_input = layer.Backward(cyxwiz::Tensor(
        {2}, grad_values, cyxwiz::DataType::Float32));
    REQUIRE(grad_input.Shape() == std::vector<size_t>{3});
    REQUIRE(grad_input.Data<float>()[0] == Catch::Approx(-0.6875f));
    REQUIRE(grad_input.Data<float>()[1] == Catch::Approx(1.6875f));
    REQUIRE(grad_input.Data<float>()[2] == Catch::Approx(0.375f));

    const auto gradients = layer.GetGradients();
    REQUIRE(gradients.count("bias") == 0);
    const float expected_weights[] = {
        -1.125f, 1.5f, -0.1875f,
        1.875f, -2.5f, 0.3125f,
    };
    for (size_t index = 0; index < 6; ++index) {
        REQUIRE(gradients.at("weights").Data<float>()[index] ==
                Catch::Approx(expected_weights[index]));
    }
}

TEST_CASE("DenseLayer rejects invalid state before compute",
          "[dense][layer][validation]") {
    REQUIRE_THROWS_AS(cyxwiz::DenseLayer(0, 2), std::invalid_argument);
    REQUIRE_THROWS_AS(cyxwiz::DenseLayer(3, -1), std::invalid_argument);

    cyxwiz::DenseLayer layer(3, 2, true);
    REQUIRE_THROWS_AS(
        layer.Backward(cyxwiz::Tensor::Ones({2, 2})), std::logic_error);
    REQUIRE_THROWS_AS(
        layer.Forward(cyxwiz::Tensor::Ones(
            {2, 3}, cyxwiz::DataType::Float64)),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        layer.Forward(cyxwiz::Tensor::Ones({2, 4})),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        layer.SetParameters({
            {"weights", cyxwiz::Tensor::Ones({3, 2})},
        }),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        layer.SetParameters({
            {"bias", cyxwiz::Tensor::Ones(
                         {2}, cyxwiz::DataType::Float64)},
        }),
        std::invalid_argument);
    const float weight_before_rejected_update =
        layer.GetParameters().at("weights").Data<float>()[0];
    REQUIRE_THROWS_AS(
        layer.SetParameters({
            {"weights", cyxwiz::Tensor::Ones({2, 3})},
            {"bias", cyxwiz::Tensor::Ones(
                         {2}, cyxwiz::DataType::Float64)},
        }),
        std::invalid_argument);
    REQUIRE(layer.GetParameters().at("weights").Data<float>()[0] ==
            Catch::Approx(weight_before_rejected_update));

    cyxwiz::DenseLayer no_bias(3, 2, false);
    REQUIRE_THROWS_AS(
        no_bias.SetParameters({
            {"bias", cyxwiz::Tensor::Zeros({2})},
        }),
        std::invalid_argument);
}

TEST_CASE("DenseLayer computes deterministic forward and backward values", "[dense][layer]") {
    cyxwiz::DenseLayer layer(3, 2, true);

    float weight_values[] = {
        1.0f, 2.0f, 3.0f,
        -1.0f, 0.5f, 1.0f,
    };
    float bias_values[] = {0.5f, -0.5f};
    layer.SetParameters({
        {"weights", cyxwiz::Tensor({2, 3}, weight_values, cyxwiz::DataType::Float32)},
        {"bias", cyxwiz::Tensor({2}, bias_values, cyxwiz::DataType::Float32)},
    });

    float input_values[] = {
        1.0f, 2.0f, 3.0f,
        -1.0f, 0.0f, 2.0f,
    };
    cyxwiz::Tensor input({2, 3}, input_values, cyxwiz::DataType::Float32);

    cyxwiz::Tensor output = layer.Forward(input);
    REQUIRE(output.Shape() == std::vector<size_t>{2, 2});
    const float* output_data = output.Data<float>();
    REQUIRE(output_data[0] == Catch::Approx(14.5f));
    REQUIRE(output_data[1] == Catch::Approx(2.5f));
    REQUIRE(output_data[2] == Catch::Approx(5.5f));
    REQUIRE(output_data[3] == Catch::Approx(2.5f));

    float grad_values[] = {
        1.0f, 2.0f,
        -1.0f, 0.5f,
    };
    cyxwiz::Tensor grad_output({2, 2}, grad_values, cyxwiz::DataType::Float32);

    cyxwiz::Tensor grad_input = layer.Backward(grad_output);
    REQUIRE(grad_input.Shape() == std::vector<size_t>{2, 3});
    const float* grad_input_data = grad_input.Data<float>();
    REQUIRE(grad_input_data[0] == Catch::Approx(-1.0f));
    REQUIRE(grad_input_data[1] == Catch::Approx(3.0f));
    REQUIRE(grad_input_data[2] == Catch::Approx(5.0f));
    REQUIRE(grad_input_data[3] == Catch::Approx(-1.5f));
    REQUIRE(grad_input_data[4] == Catch::Approx(-1.75f));
    REQUIRE(grad_input_data[5] == Catch::Approx(-2.5f));

    const std::map<std::string, cyxwiz::Tensor> params = layer.GetParameters();
    const float* grad_weight_data = params.at("grad_weights").Data<float>();
    REQUIRE(grad_weight_data[0] == Catch::Approx(2.0f));
    REQUIRE(grad_weight_data[1] == Catch::Approx(2.0f));
    REQUIRE(grad_weight_data[2] == Catch::Approx(1.0f));
    REQUIRE(grad_weight_data[3] == Catch::Approx(1.5f));
    REQUIRE(grad_weight_data[4] == Catch::Approx(4.0f));
    REQUIRE(grad_weight_data[5] == Catch::Approx(7.0f));

    const float* grad_bias_data = params.at("grad_bias").Data<float>();
    REQUIRE(grad_bias_data[0] == Catch::Approx(0.0f));
    REQUIRE(grad_bias_data[1] == Catch::Approx(2.5f));
}

TEST_CASE("Conv2DLayer computes deterministic forward and backward values", "[conv][layer]") {
    cyxwiz::Conv2DLayer layer(1, 1, 2, 1, 0, true);

    float weight_values[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
    };
    float bias_values[] = {0.5f};
    layer.SetParameters({
        {"weights", cyxwiz::Tensor({2, 2, 1, 1}, weight_values, cyxwiz::DataType::Float32)},
        {"bias", cyxwiz::Tensor({1}, bias_values, cyxwiz::DataType::Float32)},
    });

    float input_values[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
    };
    cyxwiz::Tensor input({3, 3, 1, 1}, input_values, cyxwiz::DataType::Float32);

    cyxwiz::Tensor output = layer.Forward(input);
    REQUIRE(output.Shape() == std::vector<size_t>{2, 2, 1, 1});
    const float* output_data = output.Data<float>();
    const float expected_output[] = {37.5f, 47.5f, 67.5f, 77.5f};
    for (size_t i = 0; i < 4; ++i) {
        REQUIRE(output_data[i] == Catch::Approx(expected_output[i]));
    }

    float grad_values[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
    };
    cyxwiz::Tensor grad_output({2, 2, 1, 1}, grad_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor grad_input = layer.Backward(grad_output);
    REQUIRE(grad_input.Shape() == std::vector<size_t>{3, 3, 1, 1});
    const float* grad_input_data = grad_input.Data<float>();
    const float expected_grad_input[] = {
        1.0f, 4.0f, 4.0f,
        6.0f, 20.0f, 16.0f,
        9.0f, 24.0f, 16.0f,
    };
    for (size_t i = 0; i < 9; ++i) {
        REQUIRE(grad_input_data[i] == Catch::Approx(expected_grad_input[i]));
    }

    const std::map<std::string, cyxwiz::Tensor> params = layer.GetParameters();
    const float* grad_weight_data = params.at("grad_weights").Data<float>();
    const float expected_grad_weights[] = {37.0f, 47.0f, 67.0f, 77.0f};
    for (size_t i = 0; i < 4; ++i) {
        REQUIRE(grad_weight_data[i] == Catch::Approx(expected_grad_weights[i]));
    }
    REQUIRE(params.at("grad_bias").Data<float>()[0] == Catch::Approx(10.0f));
}

TEST_CASE("Conv1DLayer computes deterministic forward and backward values", "[conv][layer]") {
    cyxwiz::Conv1DLayer layer(1, 1, 2, 1, 0, 1, true);

    float weight_values[] = {1.0f, 2.0f};
    float bias_values[] = {0.5f};
    layer.SetParameters({
        {"weights", cyxwiz::Tensor({1, 1, 2}, weight_values, cyxwiz::DataType::Float32)},
        {"bias", cyxwiz::Tensor({1}, bias_values, cyxwiz::DataType::Float32)},
    });

    float input_values[] = {
        1.0f, 10.0f,
        2.0f, 20.0f,
        3.0f, 30.0f,
        4.0f, 40.0f,
    };
    cyxwiz::Tensor input({4, 1, 2}, input_values, cyxwiz::DataType::Float32);

    cyxwiz::Tensor output = layer.Forward(input);
    REQUIRE(output.Shape() == std::vector<size_t>{3, 1, 2});
    const float* output_data = output.Data<float>();
    const float expected_output[] = {
        5.5f, 50.5f,
        8.5f, 80.5f,
        11.5f, 110.5f,
    };
    for (size_t i = 0; i < 6; ++i) {
        REQUIRE(output_data[i] == Catch::Approx(expected_output[i]));
    }

    float grad_values[] = {
        1.0f, 10.0f,
        2.0f, 20.0f,
        3.0f, 30.0f,
    };
    cyxwiz::Tensor grad_output({3, 1, 2}, grad_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor grad_input = layer.Backward(grad_output);
    REQUIRE(grad_input.Shape() == std::vector<size_t>{4, 1, 2});
    const float* grad_input_data = grad_input.Data<float>();
    const float expected_grad_input[] = {
        1.0f, 10.0f,
        4.0f, 40.0f,
        7.0f, 70.0f,
        6.0f, 60.0f,
    };
    for (size_t i = 0; i < 8; ++i) {
        REQUIRE(grad_input_data[i] == Catch::Approx(expected_grad_input[i]));
    }

    const std::map<std::string, cyxwiz::Tensor> params = layer.GetParameters();
    const float* grad_weight_data = params.at("grad_weights").Data<float>();
    REQUIRE(grad_weight_data[0] == Catch::Approx(1414.0f));
    REQUIRE(grad_weight_data[1] == Catch::Approx(2020.0f));
    REQUIRE(params.at("grad_bias").Data<float>()[0] == Catch::Approx(66.0f));
}

TEST_CASE("ConvTranspose2DLayer computes deterministic forward and backward values", "[conv][layer]") {
    cyxwiz::ConvTranspose2DLayer layer(1, 1, 2, 2, 0, 0, true);

    float weight_values[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
    };
    float bias_values[] = {0.5f};
    layer.SetParameters({
        {"weights", cyxwiz::Tensor({2, 2, 1, 1}, weight_values, cyxwiz::DataType::Float32)},
        {"bias", cyxwiz::Tensor({1}, bias_values, cyxwiz::DataType::Float32)},
    });

    float input_values[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
    };
    cyxwiz::Tensor input({2, 2, 1, 1}, input_values, cyxwiz::DataType::Float32);

    cyxwiz::Tensor output = layer.Forward(input);
    REQUIRE(output.Shape() == std::vector<size_t>{4, 4, 1, 1});
    const float* output_data = output.Data<float>();
    const float expected_output[] = {
        1.5f, 2.5f, 2.5f, 4.5f,
        3.5f, 4.5f, 6.5f, 8.5f,
        3.5f, 6.5f, 4.5f, 8.5f,
        9.5f, 12.5f, 12.5f, 16.5f,
    };
    for (size_t i = 0; i < 16; ++i) {
        REQUIRE(output_data[i] == Catch::Approx(expected_output[i]));
    }

    float grad_values[16];
    for (float& value : grad_values) {
        value = 1.0f;
    }
    cyxwiz::Tensor grad_output({4, 4, 1, 1}, grad_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor grad_input = layer.Backward(grad_output);
    REQUIRE(grad_input.Shape() == std::vector<size_t>{2, 2, 1, 1});
    const float* grad_input_data = grad_input.Data<float>();
    for (size_t i = 0; i < 4; ++i) {
        REQUIRE(grad_input_data[i] == Catch::Approx(10.0f));
    }

    const std::map<std::string, cyxwiz::Tensor> params = layer.GetParameters();
    const float* grad_weight_data = params.at("grad_weights").Data<float>();
    for (size_t i = 0; i < 4; ++i) {
        REQUIRE(grad_weight_data[i] == Catch::Approx(10.0f));
    }
    REQUIRE(params.at("grad_bias").Data<float>()[0] == Catch::Approx(16.0f));
}

TEST_CASE("BatchNorm2DLayer computes training forward and backward values", "[norm][layer]") {
    cyxwiz::BatchNorm2DLayer norm(1, 1e-5f, 0.5f);

    float input_values[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
    };
    cyxwiz::Tensor input({2, 2, 1, 1}, input_values, cyxwiz::DataType::Float32);

    cyxwiz::Tensor output = norm.Forward(input);
    REQUIRE(output.Shape() == std::vector<size_t>{2, 2, 1, 1});
    const float* output_data = output.Data<float>();
    const float expected_output[] = {
        -1.3416354f, -0.4472118f,
        0.4472118f, 1.3416354f,
    };
    for (size_t i = 0; i < 4; ++i) {
        REQUIRE(output_data[i] == Catch::Approx(expected_output[i]).margin(1e-5f));
    }

    const std::map<std::string, cyxwiz::Tensor> forward_params = norm.GetParameters();
    REQUIRE(forward_params.at("running_mean").Data<float>()[0] == Catch::Approx(1.25f));
    REQUIRE(forward_params.at("running_var").Data<float>()[0] == Catch::Approx(1.125f));

    float grad_values[] = {
        1.0f, 0.0f,
        0.0f, 0.0f,
    };
    cyxwiz::Tensor grad_output({2, 2, 1, 1}, grad_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor grad_input = norm.Backward(grad_output);
    REQUIRE(grad_input.Shape() == std::vector<size_t>{2, 2, 1, 1});
    const float* grad_input_data = grad_input.Data<float>();
    const float expected_grad_input[] = {
        0.2683303f, -0.3577684f,
        -0.0894434f, 0.1788815f,
    };
    for (size_t i = 0; i < 4; ++i) {
        REQUIRE(grad_input_data[i] == Catch::Approx(expected_grad_input[i]).margin(1e-5f));
    }

    const std::map<std::string, cyxwiz::Tensor> params = norm.GetParameters();
    REQUIRE(params.at("grad_gamma").Data<float>()[0] == Catch::Approx(-1.3416354f).margin(1e-5f));
    REQUIRE(params.at("grad_beta").Data<float>()[0] == Catch::Approx(1.0f));
}

TEST_CASE("BatchNorm2DLayer uses running statistics in eval mode", "[norm][layer]") {
    cyxwiz::BatchNorm2DLayer norm(1, 1e-5f, 0.1f);

    float gamma_values[] = {2.0f};
    float beta_values[] = {0.5f};
    float running_mean_values[] = {2.5f};
    float running_var_values[] = {1.25f};
    norm.SetParameters({
        {"gamma", cyxwiz::Tensor({1}, gamma_values, cyxwiz::DataType::Float32)},
        {"beta", cyxwiz::Tensor({1}, beta_values, cyxwiz::DataType::Float32)},
        {"running_mean", cyxwiz::Tensor({1}, running_mean_values, cyxwiz::DataType::Float32)},
        {"running_var", cyxwiz::Tensor({1}, running_var_values, cyxwiz::DataType::Float32)},
    });
    norm.SetTraining(false);

    float input_values[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
    };
    cyxwiz::Tensor input({2, 2, 1, 1}, input_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor output = norm.Forward(input);

    const float* output_data = output.Data<float>();
    const float expected_output[] = {
        -2.1832709f, -0.3944236f,
        1.3944236f, 3.1832709f,
    };
    for (size_t i = 0; i < 4; ++i) {
        REQUIRE(output_data[i] == Catch::Approx(expected_output[i]).margin(1e-5f));
    }
}

TEST_CASE("LayerNormLayer computes forward and backward values", "[norm][layer]") {
    cyxwiz::LayerNormLayer norm({3}, 1e-5f, true);

    float gamma_values[] = {1.0f, 2.0f, 3.0f};
    float beta_values[] = {0.5f, -0.5f, 1.0f};
    norm.SetParameters({
        {"gamma", cyxwiz::Tensor({3}, gamma_values, cyxwiz::DataType::Float32)},
        {"beta", cyxwiz::Tensor({3}, beta_values, cyxwiz::DataType::Float32)},
    });

    float input_values[] = {
        1.0f, 2.0f, 3.0f,
        2.0f, 4.0f, 6.0f,
    };
    cyxwiz::Tensor input({2, 3}, input_values, cyxwiz::DataType::Float32);

    cyxwiz::Tensor output = norm.Forward(input);
    REQUIRE(output.Shape() == std::vector<size_t>{2, 3});
    const float* output_data = output.Data<float>();
    const float expected_output[] = {
        -0.7247357f, -0.5f, 4.6742071f,
        -0.7247426f, -0.5f, 4.6742277f,
    };
    for (size_t i = 0; i < 6; ++i) {
        REQUIRE(output_data[i] == Catch::Approx(expected_output[i]).margin(1e-5f));
    }

    float grad_values[] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
    };
    cyxwiz::Tensor grad_output({2, 3}, grad_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor grad_input = norm.Backward(grad_output);
    REQUIRE(grad_input.Shape() == std::vector<size_t>{2, 3});
    const float* grad_input_data = grad_input.Data<float>();
    const float expected_grad_input[] = {
        0.2041318f, -0.4082452f, 0.2041134f,
        -0.4082475f, 0.8164951f, -0.4082475f,
    };
    for (size_t i = 0; i < 6; ++i) {
        REQUIRE(grad_input_data[i] == Catch::Approx(expected_grad_input[i]).margin(1e-5f));
    }

    const std::map<std::string, cyxwiz::Tensor> params = norm.GetParameters();
    const float* grad_gamma_data = params.at("grad_gamma").Data<float>();
    REQUIRE(grad_gamma_data[0] == Catch::Approx(-1.2247357f).margin(1e-5f));
    REQUIRE(grad_gamma_data[1] == Catch::Approx(0.0f).margin(1e-5f));
    REQUIRE(grad_gamma_data[2] == Catch::Approx(0.0f).margin(1e-5f));
    const float* grad_beta_data = params.at("grad_beta").Data<float>();
    REQUIRE(grad_beta_data[0] == Catch::Approx(1.0f));
    REQUIRE(grad_beta_data[1] == Catch::Approx(1.0f));
    REQUIRE(grad_beta_data[2] == Catch::Approx(0.0f));
}

TEST_CASE("InstanceNorm2DLayer normalizes each instance independently", "[norm][layer]") {
    cyxwiz::InstanceNorm2DLayer norm(1, 1e-5f, true);

    float gamma_values[] = {2.0f};
    float beta_values[] = {-1.0f};
    norm.SetParameters({
        {"gamma", cyxwiz::Tensor({1}, gamma_values, cyxwiz::DataType::Float32)},
        {"beta", cyxwiz::Tensor({1}, beta_values, cyxwiz::DataType::Float32)},
    });

    float input_values[] = {
        1.0f, 2.0f,
        2.0f, 4.0f,
        3.0f, 6.0f,
        4.0f, 8.0f,
    };
    cyxwiz::Tensor input({2, 2, 1, 2}, input_values, cyxwiz::DataType::Float32);

    cyxwiz::Tensor output = norm.Forward(input);
    REQUIRE(output.Shape() == std::vector<size_t>{2, 2, 1, 2});
    const float* output_data = output.Data<float>();
    const float expected_output[] = {
        -3.6832708f, -3.6832789f,
        -1.8944236f, -1.8944263f,
        -0.1055764f, -0.1055737f,
        1.6832708f, 1.6832789f,
    };
    for (size_t i = 0; i < 8; ++i) {
        REQUIRE(output_data[i] == Catch::Approx(expected_output[i]).margin(1e-5f));
    }

    float grad_values[] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
        0.0f, 0.0f,
    };
    cyxwiz::Tensor grad_output({2, 2, 1, 2}, grad_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor grad_input = norm.Backward(grad_output);
    REQUIRE(grad_input.Shape() == std::vector<size_t>{2, 2, 1, 2});
    const float* grad_input_data = grad_input.Data<float>();
    const float expected_grad_input[] = {
        0.5366606f, -0.3577703f,
        -0.7155367f, 0.6260985f,
        -0.1788869f, -0.1788853f,
        0.3577630f, -0.0894429f,
    };
    for (size_t i = 0; i < 8; ++i) {
        REQUIRE(grad_input_data[i] == Catch::Approx(expected_grad_input[i]).margin(1e-5f));
    }

    const std::map<std::string, cyxwiz::Tensor> params = norm.GetParameters();
    const float* grad_gamma_data = params.at("grad_gamma").Data<float>();
    REQUIRE(grad_gamma_data[0] == Catch::Approx(-1.7888486f).margin(1e-5f));
    const float* grad_beta_data = params.at("grad_beta").Data<float>();
    REQUIRE(grad_beta_data[0] == Catch::Approx(2.0f));
}

TEST_CASE("GroupNormLayer normalizes grouped channels", "[norm][layer]") {
    cyxwiz::GroupNormLayer norm(1, 2, 1e-5f, true);

    float gamma_values[] = {1.0f, 2.0f};
    float beta_values[] = {0.5f, -1.0f};
    norm.SetParameters({
        {"gamma", cyxwiz::Tensor({2}, gamma_values, cyxwiz::DataType::Float32)},
        {"beta", cyxwiz::Tensor({2}, beta_values, cyxwiz::DataType::Float32)},
    });

    float input_values[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
    };
    cyxwiz::Tensor input({1, 2, 2, 1}, input_values, cyxwiz::DataType::Float32);

    cyxwiz::Tensor output = norm.Forward(input);
    REQUIRE(output.Shape() == std::vector<size_t>{1, 2, 2, 1});
    const float* output_data = output.Data<float>();
    const float expected_output[] = {
        -0.8416354f, -1.8944236f,
        0.9472118f, 1.6832708f,
    };
    for (size_t i = 0; i < 4; ++i) {
        REQUIRE(output_data[i] == Catch::Approx(expected_output[i]).margin(1e-5f));
    }

    float grad_values[] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
    };
    cyxwiz::Tensor grad_output({1, 2, 2, 1}, grad_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor grad_input = norm.Backward(grad_output);
    REQUIRE(grad_input.Shape() == std::vector<size_t>{1, 2, 2, 1});
    const float* grad_input_data = grad_input.Data<float>();
    const float expected_grad_input[] = {
        0.6260933f, -0.5366552f,
        -0.8049802f, 0.7155421f,
    };
    for (size_t i = 0; i < 4; ++i) {
        REQUIRE(grad_input_data[i] == Catch::Approx(expected_grad_input[i]).margin(1e-5f));
    }

    const std::map<std::string, cyxwiz::Tensor> params = norm.GetParameters();
    const float* grad_gamma_data = params.at("grad_gamma").Data<float>();
    REQUIRE(grad_gamma_data[0] == Catch::Approx(-1.3416354f).margin(1e-5f));
    REQUIRE(grad_gamma_data[1] == Catch::Approx(1.3416354f).margin(1e-5f));
    const float* grad_beta_data = params.at("grad_beta").Data<float>();
    REQUIRE(grad_beta_data[0] == Catch::Approx(1.0f));
    REQUIRE(grad_beta_data[1] == Catch::Approx(1.0f));
}

TEST_CASE("MultiHeadAttentionLayer computes deterministic self-attention", "[attention][layer]") {
    cyxwiz::MultiHeadAttentionLayer attention(2, 1, 0.0f, false);

    float identity_values[] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
    };
    attention.SetParameters({
        {"W_q", cyxwiz::Tensor({2, 2}, identity_values, cyxwiz::DataType::Float32)},
        {"W_k", cyxwiz::Tensor({2, 2}, identity_values, cyxwiz::DataType::Float32)},
        {"W_v", cyxwiz::Tensor({2, 2}, identity_values, cyxwiz::DataType::Float32)},
        {"W_o", cyxwiz::Tensor({2, 2}, identity_values, cyxwiz::DataType::Float32)},
    });

    float input_values[] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
    };
    cyxwiz::Tensor input({1, 2, 2}, input_values, cyxwiz::DataType::Float32);

    cyxwiz::Tensor output = attention.Forward(input);
    REQUIRE(output.Shape() == std::vector<size_t>{1, 2, 2});
    const float* output_data = output.Data<float>();
    REQUIRE(output_data[0] == Catch::Approx(0.6697615f).margin(1e-5f));
    REQUIRE(output_data[1] == Catch::Approx(0.3302385f).margin(1e-5f));
    REQUIRE(output_data[2] == Catch::Approx(0.3302385f).margin(1e-5f));
    REQUIRE(output_data[3] == Catch::Approx(0.6697615f).margin(1e-5f));

    cyxwiz::Tensor weights = attention.GetAttentionWeights();
    REQUIRE(weights.Shape() == std::vector<size_t>{2, 2, 1, 1});
    const float* weight_data = weights.Data<float>();
    REQUIRE(weight_data[0] == Catch::Approx(0.6697615f).margin(1e-5f));
    REQUIRE(weight_data[1] == Catch::Approx(0.3302385f).margin(1e-5f));
    REQUIRE(weight_data[2] == Catch::Approx(0.3302385f).margin(1e-5f));
    REQUIRE(weight_data[3] == Catch::Approx(0.6697615f).margin(1e-5f));

    float grad_values[] = {
        1.0f, 0.0f,
        0.0f, 2.0f,
    };
    cyxwiz::Tensor grad_output({1, 2, 2}, grad_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor grad_input = attention.Backward(grad_output);
    REQUIRE(grad_input.Shape() == std::vector<size_t>{1, 2, 2});
    const float* grad_input_data = grad_input.Data<float>();
    REQUIRE(grad_input_data[0] == Catch::Approx(0.9825587f).margin(1e-5f));
    REQUIRE(grad_input_data[1] == Catch::Approx(0.1912811f).margin(1e-5f));
    REQUIRE(grad_input_data[2] == Catch::Approx(-0.1389573f).margin(1e-5f));
    REQUIRE(grad_input_data[3] == Catch::Approx(1.9651175f).margin(1e-5f));

    const std::map<std::string, cyxwiz::Tensor> params = attention.GetParameters();
    const float* grad_wq = params.at("grad_W_q").Data<float>();
    REQUIRE(grad_wq[0] == Catch::Approx(0.1563986f).margin(1e-5f));
    REQUIRE(grad_wq[1] == Catch::Approx(-0.3127972f).margin(1e-5f));
    REQUIRE(grad_wq[2] == Catch::Approx(-0.1563986f).margin(1e-5f));
    REQUIRE(grad_wq[3] == Catch::Approx(0.3127972f).margin(1e-5f));
}

TEST_CASE("MultiHeadAttentionLayer applies causal mask to self-attention", "[attention][layer][causal]") {
    cyxwiz::MultiHeadAttentionLayer attention(1, 1, 0.0f, false);

    float identity_values[] = {1.0f};
    attention.SetParameters({
        {"W_q", cyxwiz::Tensor({1, 1}, identity_values, cyxwiz::DataType::Float32)},
        {"W_k", cyxwiz::Tensor({1, 1}, identity_values, cyxwiz::DataType::Float32)},
        {"W_v", cyxwiz::Tensor({1, 1}, identity_values, cyxwiz::DataType::Float32)},
        {"W_o", cyxwiz::Tensor({1, 1}, identity_values, cyxwiz::DataType::Float32)},
    });

    float input_values[] = {1.0f, 2.0f, 3.0f};
    cyxwiz::Tensor input({1, 3, 1}, input_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor mask = cyxwiz::TransformerDecoderLayer::GenerateCausalMask(3);

    cyxwiz::Tensor output = attention.Forward(input, input, input, &mask);
    REQUIRE(output.Shape() == std::vector<size_t>{1, 3, 1});
    const float* output_data = output.Data<float>();
    REQUIRE(output_data[0] == Catch::Approx(1.0f).margin(1e-5f));
    REQUIRE(output_data[1] == Catch::Approx(1.8807971f).margin(1e-5f));
    REQUIRE(output_data[2] == Catch::Approx(2.9479747f).margin(1e-5f));

    cyxwiz::Tensor weights = attention.GetAttentionWeights();
    REQUIRE(weights.Shape() == std::vector<size_t>{3, 3, 1, 1});
    const float* weight_data = weights.Data<float>();
    REQUIRE(weight_data[1] == Catch::Approx(0.0f).margin(1e-5f));
    REQUIRE(weight_data[2] == Catch::Approx(0.0f).margin(1e-5f));
    REQUIRE(weight_data[5] == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("TransformerDecoderLayer causal mask validates size", "[transformer][decoder][causal]") {
    REQUIRE_THROWS_AS(cyxwiz::TransformerDecoderLayer::GenerateCausalMask(0), std::invalid_argument);
    REQUIRE_THROWS_AS(cyxwiz::TransformerDecoderLayer::GenerateCausalMask(-1), std::invalid_argument);
}

TEST_CASE("MultiHeadAttentionLayer reuses attention dropout mask during backward", "[attention][layer]") {
    cyxwiz::MultiHeadAttentionLayer attention(1, 1, 0.5f, false);

    float identity_values[] = {1.0f};
    attention.SetParameters({
        {"W_q", cyxwiz::Tensor({1, 1}, identity_values, cyxwiz::DataType::Float32)},
        {"W_k", cyxwiz::Tensor({1, 1}, identity_values, cyxwiz::DataType::Float32)},
        {"W_v", cyxwiz::Tensor({1, 1}, identity_values, cyxwiz::DataType::Float32)},
        {"W_o", cyxwiz::Tensor({1, 1}, identity_values, cyxwiz::DataType::Float32)},
    });

    float input_values[] = {1.0f};
    cyxwiz::Tensor input({1, 1, 1}, input_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor output = attention.Forward(input);

    REQUIRE(output.Shape() == std::vector<size_t>{1, 1, 1});
    const float* output_data = output.Data<float>();
    const bool output_matches_dropout_mask =
        output_data[0] == Catch::Approx(0.0f) ||
        output_data[0] == Catch::Approx(2.0f);
    REQUIRE(output_matches_dropout_mask);

    cyxwiz::Tensor weights = attention.GetAttentionWeights();
    REQUIRE(weights.Shape() == std::vector<size_t>{1, 1, 1, 1});
    REQUIRE(weights.Data<float>()[0] == Catch::Approx(1.0f));

    float grad_values[] = {1.0f};
    cyxwiz::Tensor grad_output({1, 1, 1}, grad_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor grad_input = attention.Backward(grad_output);
    REQUIRE(grad_input.Shape() == std::vector<size_t>{1, 1, 1});
    REQUIRE(grad_input.Data<float>()[0] == Catch::Approx(output_data[0]));
}

TEST_CASE("MultiHeadAttentionLayer exposes cross-attention key and value gradients", "[attention][layer]") {
    cyxwiz::MultiHeadAttentionLayer attention(1, 1, 0.0f, false);

    float identity_values[] = {1.0f};
    attention.SetParameters({
        {"W_q", cyxwiz::Tensor({1, 1}, identity_values, cyxwiz::DataType::Float32)},
        {"W_k", cyxwiz::Tensor({1, 1}, identity_values, cyxwiz::DataType::Float32)},
        {"W_v", cyxwiz::Tensor({1, 1}, identity_values, cyxwiz::DataType::Float32)},
        {"W_o", cyxwiz::Tensor({1, 1}, identity_values, cyxwiz::DataType::Float32)},
    });

    float query_values[] = {1.0f};
    float key_values[] = {1.0f, 0.0f};
    float value_values[] = {2.0f, 4.0f};
    cyxwiz::Tensor query({1, 1, 1}, query_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor key({1, 2, 1}, key_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor value({1, 2, 1}, value_values, cyxwiz::DataType::Float32);

    cyxwiz::Tensor output = attention.Forward(query, key, value);
    REQUIRE(output.Shape() == std::vector<size_t>{1, 1, 1});
    REQUIRE(output.Data<float>()[0] == Catch::Approx(2.5378828f).margin(1e-5f));

    float grad_values[] = {1.0f};
    cyxwiz::Tensor grad_output({1, 1, 1}, grad_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor grad_query = attention.Backward(grad_output);
    REQUIRE(grad_query.Shape() == std::vector<size_t>{1, 1, 1});
    REQUIRE(grad_query.Data<float>()[0] == Catch::Approx(-0.3932239f).margin(1e-5f));

    cyxwiz::Tensor grad_key = attention.GetLastKeyGradient();
    REQUIRE(grad_key.Shape() == std::vector<size_t>{1, 2, 1});
    REQUIRE(grad_key.Data<float>()[0] == Catch::Approx(-0.3932239f).margin(1e-5f));
    REQUIRE(grad_key.Data<float>()[1] == Catch::Approx(0.3932239f).margin(1e-5f));

    cyxwiz::Tensor grad_value = attention.GetLastValueGradient();
    REQUIRE(grad_value.Shape() == std::vector<size_t>{1, 2, 1});
    REQUIRE(grad_value.Data<float>()[0] == Catch::Approx(0.7310586f).margin(1e-5f));
    REQUIRE(grad_value.Data<float>()[1] == Catch::Approx(0.2689414f).margin(1e-5f));
}

TEST_CASE("DropoutLayer passes values through in eval mode", "[dropout][layer]") {
    float input_values[] = {1.0f, -2.0f, 3.5f, 4.0f};
    cyxwiz::Tensor input({2, 2}, input_values, cyxwiz::DataType::Float32);
    cyxwiz::DropoutLayer dropout(0.5f);
    dropout.SetTraining(false);

    cyxwiz::Tensor output = dropout.Forward(input);
    cyxwiz::Tensor grad_input = dropout.Backward(input);

    REQUIRE(output.Shape() == std::vector<size_t>{2, 2});
    REQUIRE(grad_input.Shape() == std::vector<size_t>{2, 2});
    const float* output_data = output.Data<float>();
    const float* grad_input_data = grad_input.Data<float>();
    for (size_t i = 0; i < input.NumElements(); ++i) {
        REQUIRE(output_data[i] == Catch::Approx(input_values[i]));
        REQUIRE(grad_input_data[i] == Catch::Approx(input_values[i]));
    }
}

TEST_CASE("DropoutLayer reuses forward mask during backward", "[dropout][layer]") {
    float input_values[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float grad_values[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    cyxwiz::Tensor input({2, 3}, input_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor grad_output({2, 3}, grad_values, cyxwiz::DataType::Float32);
    cyxwiz::DropoutLayer dropout(0.5f);

    cyxwiz::Tensor output = dropout.Forward(input);
    cyxwiz::Tensor grad_input = dropout.Backward(grad_output);

    REQUIRE(output.Shape() == std::vector<size_t>{2, 3});
    REQUIRE(grad_input.Shape() == std::vector<size_t>{2, 3});
    const float* output_data = output.Data<float>();
    const float* grad_input_data = grad_input.Data<float>();
    for (size_t i = 0; i < input.NumElements(); ++i) {
        const bool output_is_scaled_mask = output_data[i] == Catch::Approx(0.0f) ||
                                           output_data[i] == Catch::Approx(2.0f);
        REQUIRE(output_is_scaled_mask);
        REQUIRE(grad_input_data[i] == Catch::Approx(output_data[i]));
    }
}

TEST_CASE("MaxPool2DLayer computes forward and backward values", "[pool][layer]") {
    float input_values[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
    };
    cyxwiz::Tensor input({2, 2, 1, 1}, input_values, cyxwiz::DataType::Float32);
    cyxwiz::MaxPool2DLayer pool(2, 2, 0);

    cyxwiz::Tensor output = pool.Forward(input);
    REQUIRE(output.NumElements() == 1);
    REQUIRE(output.Data<float>()[0] == Catch::Approx(4.0f));

    float grad_values[] = {2.0f};
    cyxwiz::Tensor grad_output({1, 1, 1, 1}, grad_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor grad_input = pool.Backward(grad_output);
    REQUIRE(grad_input.NumElements() == 4);
    const float* grad_input_data = grad_input.Data<float>();
    REQUIRE(grad_input_data[0] == Catch::Approx(0.0f));
    REQUIRE(grad_input_data[1] == Catch::Approx(0.0f));
    REQUIRE(grad_input_data[2] == Catch::Approx(0.0f));
    REQUIRE(grad_input_data[3] == Catch::Approx(2.0f));
}

TEST_CASE("AvgPool2DLayer computes forward and backward values", "[pool][layer]") {
    float input_values[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
    };
    cyxwiz::Tensor input({2, 2, 1, 1}, input_values, cyxwiz::DataType::Float32);
    cyxwiz::AvgPool2DLayer pool(2, 2, 0);

    cyxwiz::Tensor output = pool.Forward(input);
    REQUIRE(output.NumElements() == 1);
    REQUIRE(output.Data<float>()[0] == Catch::Approx(2.5f));

    float grad_values[] = {1.0f};
    cyxwiz::Tensor grad_output({1, 1, 1, 1}, grad_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor grad_input = pool.Backward(grad_output);
    REQUIRE(grad_input.NumElements() == 4);
    const float* grad_input_data = grad_input.Data<float>();
    for (size_t i = 0; i < input.NumElements(); ++i) {
        REQUIRE(grad_input_data[i] == Catch::Approx(0.25f));
    }
}

TEST_CASE("GlobalAvgPool2DLayer computes forward and backward values", "[pool][layer]") {
    float input_values[] = {
        1.0f, 10.0f,
        2.0f, 20.0f,
        3.0f, 30.0f,
        4.0f, 40.0f,
    };
    cyxwiz::Tensor input({2, 2, 2, 1}, input_values, cyxwiz::DataType::Float32);
    cyxwiz::GlobalAvgPool2DLayer pool;

    cyxwiz::Tensor output = pool.Forward(input);
    REQUIRE(output.NumElements() == 2);
    const float* output_data = output.Data<float>();
    REQUIRE(output_data[0] == Catch::Approx(2.5f));
    REQUIRE(output_data[1] == Catch::Approx(25.0f));

    float grad_values[] = {1.0f, 2.0f};
    cyxwiz::Tensor grad_output({2, 1}, grad_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor grad_input = pool.Backward(grad_output);
    REQUIRE(grad_input.NumElements() == 8);
    const float* grad_input_data = grad_input.Data<float>();
    REQUIRE(grad_input_data[0] == Catch::Approx(0.25f));
    REQUIRE(grad_input_data[1] == Catch::Approx(0.5f));
    REQUIRE(grad_input_data[2] == Catch::Approx(0.25f));
    REQUIRE(grad_input_data[3] == Catch::Approx(0.5f));
    REQUIRE(grad_input_data[4] == Catch::Approx(0.25f));
    REQUIRE(grad_input_data[5] == Catch::Approx(0.5f));
    REQUIRE(grad_input_data[6] == Catch::Approx(0.25f));
    REQUIRE(grad_input_data[7] == Catch::Approx(0.5f));
}

TEST_CASE("CPU-only Tensor layers reject strict native fallback before compute",
          "[arrayfire][fallback][policy][attention][pool]") {
    cyxwiz::MultiHeadAttentionLayer attention(2, 1, 0.0f, false);
    const float attention_values[] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
    };
    const cyxwiz::Tensor attention_input(
        {1, 2, 2}, attention_values, cyxwiz::DataType::Float32);

    const float pool_values[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const cyxwiz::Tensor pool_input(
        {2, 2, 1, 1}, pool_values, cyxwiz::DataType::Float32);
    cyxwiz::GlobalAvgPool2DLayer pool;

    const cyxwiz::ScopedArrayFireFallbackPolicy strict(
        cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
    REQUIRE_THROWS_AS(attention.Forward(attention_input), std::runtime_error);
    REQUIRE_THROWS_AS(pool.Forward(pool_input), std::runtime_error);
}

TEST_CASE("Upsample2DLayer nearest computes forward and backward values", "[upsample][layer]") {
    float input_values[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
    };
    cyxwiz::Tensor input({2, 2, 1, 1}, input_values, cyxwiz::DataType::Float32);
    cyxwiz::Upsample2DLayer upsample(2, cyxwiz::UpsampleMode::Nearest);

    cyxwiz::Tensor output = upsample.Forward(input);
    REQUIRE(output.NumElements() == 16);
    const float* output_data = output.Data<float>();
    const float expected_output[] = {
        1.0f, 1.0f, 2.0f, 2.0f,
        1.0f, 1.0f, 2.0f, 2.0f,
        3.0f, 3.0f, 4.0f, 4.0f,
        3.0f, 3.0f, 4.0f, 4.0f,
    };
    for (size_t i = 0; i < 16; ++i) {
        REQUIRE(output_data[i] == Catch::Approx(expected_output[i]));
    }

    float grad_values[16];
    for (float& value : grad_values) {
        value = 1.0f;
    }
    cyxwiz::Tensor grad_output({4, 4, 1, 1}, grad_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor grad_input = upsample.Backward(grad_output);
    REQUIRE(grad_input.NumElements() == 4);
    const float* grad_input_data = grad_input.Data<float>();
    for (size_t i = 0; i < 4; ++i) {
        REQUIRE(grad_input_data[i] == Catch::Approx(4.0f));
    }
}

TEST_CASE("Upsample2DLayer bilinear computes forward and backward values", "[upsample][layer]") {
    float input_values[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
    };
    cyxwiz::Tensor input({2, 2, 1, 1}, input_values, cyxwiz::DataType::Float32);
    cyxwiz::Upsample2DLayer upsample(2, cyxwiz::UpsampleMode::Bilinear);

    cyxwiz::Tensor output = upsample.Forward(input);
    REQUIRE(output.NumElements() == 16);
    const float* output_data = output.Data<float>();
    const float expected_output[] = {
        1.0f, 1.25f, 1.75f, 2.0f,
        1.5f, 1.75f, 2.25f, 2.5f,
        2.5f, 2.75f, 3.25f, 3.5f,
        3.0f, 3.25f, 3.75f, 4.0f,
    };
    for (size_t i = 0; i < 16; ++i) {
        REQUIRE(output_data[i] == Catch::Approx(expected_output[i]));
    }

    float grad_values[16];
    for (float& value : grad_values) {
        value = 1.0f;
    }
    cyxwiz::Tensor grad_output({4, 4, 1, 1}, grad_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor grad_input = upsample.Backward(grad_output);
    REQUIRE(grad_input.NumElements() == 4);
    const float* grad_input_data = grad_input.Data<float>();
    for (size_t i = 0; i < 4; ++i) {
        REQUIRE(grad_input_data[i] == Catch::Approx(4.0f));
    }
}

TEST_CASE("PixelShuffleLayer computes forward and backward values", "[pixelshuffle][layer]") {
    float input_values[] = {1.0f, 2.0f, 3.0f, 4.0f};
    cyxwiz::Tensor input({1, 1, 4, 1}, input_values, cyxwiz::DataType::Float32);
    cyxwiz::PixelShuffleLayer shuffle(2);

    cyxwiz::Tensor output = shuffle.Forward(input);
    REQUIRE(output.NumElements() == 4);
    const float* output_data = output.Data<float>();
    REQUIRE(output_data[0] == Catch::Approx(1.0f));
    REQUIRE(output_data[1] == Catch::Approx(2.0f));
    REQUIRE(output_data[2] == Catch::Approx(3.0f));
    REQUIRE(output_data[3] == Catch::Approx(4.0f));

    float grad_values[] = {10.0f, 20.0f, 30.0f, 40.0f};
    cyxwiz::Tensor grad_output({2, 2, 1, 1}, grad_values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor grad_input = shuffle.Backward(grad_output);
    REQUIRE(grad_input.NumElements() == 4);
    const float* grad_input_data = grad_input.Data<float>();
    REQUIRE(grad_input_data[0] == Catch::Approx(10.0f));
    REQUIRE(grad_input_data[1] == Catch::Approx(20.0f));
    REQUIRE(grad_input_data[2] == Catch::Approx(30.0f));
    REQUIRE(grad_input_data[3] == Catch::Approx(40.0f));
}
TEST_CASE("MultiHeadAttentionLayer causal output matches CPU scaled-dot-product reference",
          "[attention][layer][causal][language_model]") {
    cyxwiz::MultiHeadAttentionLayer attention(1, 1, 0.0f, false);

    float identity_values[] = {1.0f};
    attention.SetParameters({
        {"W_q", cyxwiz::Tensor({1, 1}, identity_values, cyxwiz::DataType::Float32)},
        {"W_k", cyxwiz::Tensor({1, 1}, identity_values, cyxwiz::DataType::Float32)},
        {"W_v", cyxwiz::Tensor({1, 1}, identity_values, cyxwiz::DataType::Float32)},
        {"W_o", cyxwiz::Tensor({1, 1}, identity_values, cyxwiz::DataType::Float32)},
    });

    const std::vector<float> input_values = {1.0f, 2.0f, 3.0f};
    cyxwiz::Tensor input({1, 3, 1}, input_values.data(), cyxwiz::DataType::Float32);
    cyxwiz::Tensor mask = cyxwiz::TransformerDecoderLayer::GenerateCausalMask(3);

    cyxwiz::Tensor output = attention.Forward(input, input, input, &mask);
    const std::vector<float> reference = cyxwiz::ScaledDotProductAttentionCpu(
        input_values,
        input_values,
        input_values,
        cyxwiz::BuildCausalAttentionMask(3),
        3,
        1);

    REQUIRE(output.Shape() == std::vector<size_t>{1, 3, 1});
    const float* actual = output.Data<float>();
    REQUIRE(reference.size() == 3);
    for (size_t i = 0; i < reference.size(); ++i) {
        REQUIRE(actual[i] == Catch::Approx(reference[i]).margin(1.0e-5f));
    }
}
