#include "../../cyxwiz-engine/src/core/language_model_training.h"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cyxwiz/layer.h>
#include <cyxwiz/sequential.h>
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
    // PyTorch: the running estimate takes the unbiased variance (5/3 here).
    REQUIRE(forward_params.at("running_var").Data<float>()[0] == Catch::Approx(1.3333333f));

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

// Two channels over [H=2, W=3, C=2, N=2]: per-channel statistics across H, W and
// the batch (the ArrayFire channel-column view). Values from PyTorch
// functional.batch_norm(training=True), float64, eps 1e-5.
TEST_CASE("BatchNorm2DLayer matches PyTorch across channels", "[norm][layer]") {
    cyxwiz::BatchNorm2DLayer norm(2, 1e-5f, 0.1f);
    float gamma_values[] = {1.5f, 0.5f};
    float beta_values[] = {0.1f, -0.2f};
    norm.SetParameters({
        {"gamma", cyxwiz::Tensor({2}, gamma_values, cyxwiz::DataType::Float32)},
        {"beta", cyxwiz::Tensor({2}, beta_values, cyxwiz::DataType::Float32)},
    });

    float input_values[] = {
        -2.0f, 1.75f, 0.0f, 3.75f, 1.0f, -0.75f, 3.0f, 1.25f, -1.5f, 2.25f, 0.5f, -1.25f,
        1.5f, -0.25f, 3.5f, 1.75f, -1.0f, 2.75f, 1.0f, -0.75f, 2.0f, 0.25f, -1.5f, 2.25f,
    };
    float grad_values[] = {
        -0.9f, 0.6f, 0.0f, -0.6f, 0.9f, 0.3f, -0.3f, -0.9f, 0.6f, 0.0f, -0.6f, 0.9f,
        0.3f, -0.3f, -0.9f, 0.6f, 0.0f, -0.6f, 0.9f, 0.3f, -0.3f, -0.9f, 0.6f, 0.0f,
    };
    const float expected_output[] = {
        -2.3604442f, 1.3302221f, -0.5273263f, 0.5637613f, 0.5920888f, -1.1302221f,
        0.3455438f, -0.1636304f, -1.8683553f, 1.8223109f, -0.3818479f, -0.8910221f,
        1.0841777f, -0.6381333f, 0.4910221f, -0.0181521f, -1.3762665f, 2.3143998f,
        -0.2363696f, -0.7455438f, 1.5762665f, -0.1460444f, -0.9637613f, 0.1273263f,
    };
    const float expected_grad_input[] = {
        -0.7883355f, 0.5787010f, -0.0741244f, -0.0016170f, 0.8958003f, 0.3562677f,
        0.0362537f, -0.2536250f, 0.6733671f, -0.0263696f, -0.2157543f, 0.1053761f,
        0.2907297f, -0.2488029f, -0.1053761f, 0.2157543f, 0.0682964f, -0.6314402f,
        0.2536250f, -0.0362537f, -0.3143409f, -0.8538735f, 0.0016170f, 0.0741244f,
    };

    const cyxwiz::Tensor output =
        norm.Forward(cyxwiz::Tensor({2, 3, 2, 2}, input_values, cyxwiz::DataType::Float32));
    const cyxwiz::Tensor grad_input =
        norm.Backward(cyxwiz::Tensor({2, 3, 2, 2}, grad_values, cyxwiz::DataType::Float32));
    REQUIRE(output.Shape() == std::vector<size_t>{2, 3, 2, 2});
    REQUIRE(grad_input.Shape() == std::vector<size_t>{2, 3, 2, 2});
    const float* output_data = output.ReadData<float>();
    const float* grad_input_data = grad_input.ReadData<float>();
    for (size_t i = 0; i < 24; ++i) {
        REQUIRE(output_data[i] == Catch::Approx(expected_output[i]).margin(1e-5f));
        REQUIRE(grad_input_data[i] == Catch::Approx(expected_grad_input[i]).margin(1e-5f));
    }
    const auto params = norm.GetParameters();
    const float* grad_gamma = params.at("grad_gamma").ReadData<float>();
    const float* grad_beta = params.at("grad_beta").ReadData<float>();
    REQUIRE(grad_gamma[0] == Catch::Approx(0.5412977f).margin(1e-5f));
    REQUIRE(grad_gamma[1] == Catch::Approx(-4.6698549f).margin(1e-5f));
    REQUIRE(grad_beta[0] == Catch::Approx(-0.3f).margin(1e-5f));
    REQUIRE(grad_beta[1] == Catch::Approx(0.0f).margin(1e-5f));
    // Momentum 0.1 from zeros / ones, unbiased variance (PyTorch BatchNorm2d).
    const float* running_mean = params.at("running_mean").ReadData<float>();
    const float* running_var = params.at("running_var").ReadData<float>();
    REQUIRE(running_mean[0] == Catch::Approx(0.05f).margin(1e-6f));
    REQUIRE(running_mean[1] == Catch::Approx(0.1125f).margin(1e-6f));
    REQUIRE(running_var[0] == Catch::Approx(1.1534091f).margin(1e-6f));
    REQUIRE(running_var[1] == Catch::Approx(1.2221591f).margin(1e-6f));
}

// The Engine's BatchNorm node in training: per-feature statistics over the
// batch on [batch=4, features=3]. Values from PyTorch functional.batch_norm.
TEST_CASE("BatchNormModule training forward and backward match PyTorch", "[norm][layer]") {
    cyxwiz::BatchNormModule norm(3, 1e-5f, 0.1f);
    float gamma_values[] = {1.5f, 0.5f, -1.0f};
    float beta_values[] = {0.1f, -0.2f, 0.3f};
    norm.SetParameters({
        {"gamma", cyxwiz::Tensor({3}, gamma_values, cyxwiz::DataType::Float32)},
        {"beta", cyxwiz::Tensor({3}, beta_values, cyxwiz::DataType::Float32)},
    });
    norm.SetTraining(true);
    float input_values[] = {1.0f, -2.0f, 0.5f, 3.0f, 0.0f, 1.5f, -1.0f, 4.0f, 2.5f, 2.0f, 1.0f, -0.5f};
    float grad_values[] = {0.3f, -0.6f, 0.9f, -0.9f, 0.0f, 0.3f, 0.6f, 0.3f, -0.6f, 0.0f, 0.9f, -0.3f};
    const cyxwiz::Tensor output = norm.Forward(cyxwiz::Tensor({4, 3}, input_values, cyxwiz::DataType::Float32));
    const cyxwiz::Tensor grad_input = norm.Backward(cyxwiz::Tensor({4, 3}, grad_values, cyxwiz::DataType::Float32));

    const float expected_output[] = {-0.1535457f, -0.8350846f, 0.7472118f, 1.8748199f, -0.3732049f, -0.1472118f,
                                     -2.1819113f, 0.5505545f, -1.0416354f, 0.8606371f, -0.1422650f, 1.6416354f};
    const float expected_grad_input[] = {0.2173253f, -0.0766722f, -0.6708182f, -0.3042576f, -0.0083139f, -0.2683265f,
                                         -0.1738563f, -0.0794431f, 0.4024922f, 0.2607887f, 0.1644292f, 0.5366526f};
    const float* output_data = output.ReadData<float>();
    const float* grad_input_data = grad_input.ReadData<float>();
    for (size_t i = 0; i < 12; ++i) {
        REQUIRE(output_data[i] == Catch::Approx(expected_output[i]).margin(1e-5f));
        REQUIRE(grad_input_data[i] == Catch::Approx(expected_grad_input[i]).margin(1e-5f));
    }
    const auto gradients = norm.GetGradients();
    const float expected_grad_gamma[] = {-2.0283656f, 1.3163572f, -0.6708177f};
    const float expected_grad_beta[] = {0.0f, 0.6f, 0.3f};
    const float* grad_gamma = gradients.at("gamma").ReadData<float>();
    const float* grad_beta = gradients.at("beta").ReadData<float>();
    for (size_t f = 0; f < 3; ++f) {
        REQUIRE(grad_gamma[f] == Catch::Approx(expected_grad_gamma[f]).margin(1e-5f));
        REQUIRE(grad_beta[f] == Catch::Approx(expected_grad_beta[f]).margin(1e-5f));
    }
}

// The Engine's BatchNorm node: two training steps, then eval on the first
// batch. Values from PyTorch BatchNorm1d(3, momentum=0.1), float64.
TEST_CASE("BatchNormModule running statistics match PyTorch", "[norm][layer]") {
    cyxwiz::BatchNormModule norm(3, 1e-5f, 0.1f);
    float first[] = {1.0f, -2.0f, 0.5f, 3.0f, 0.0f, 1.5f, -1.0f, 4.0f, 2.5f, 2.0f, 1.0f, -0.5f};
    float second[] = {0.5f, 1.0f, -1.0f, 1.5f, -3.0f, 2.0f, 2.5f, 2.0f, 0.0f, -0.5f, 0.0f, 3.0f};
    const cyxwiz::Tensor a({4, 3}, first, cyxwiz::DataType::Float32);
    norm.SetTraining(true);
    norm.Forward(a);
    norm.Forward(cyxwiz::Tensor({4, 3}, second, cyxwiz::DataType::Float32));

    const auto params = norm.GetParameters();
    const float expected_mean[] = {0.2125f, 0.0675f, 0.19f};
    const float expected_var[] = {1.2391667f, 1.8391667f, 1.2933333f};
    const float* running_mean = params.at("running_mean").ReadData<float>();
    const float* running_var = params.at("running_var").ReadData<float>();
    for (size_t f = 0; f < 3; ++f) {
        REQUIRE(running_mean[f] == Catch::Approx(expected_mean[f]).margin(1e-6f));
        REQUIRE(running_var[f] == Catch::Approx(expected_var[f]).margin(1e-6f));
    }

    norm.SetTraining(false);
    const cyxwiz::Tensor output = norm.Forward(a);
    const float expected_eval[] = {0.7074308f, -1.5245223f, 0.2725868f, 2.5040804f, -0.0497728f, 1.1518989f,
                                   -1.0892188f, 2.8997263f, 2.0312111f, 1.6057556f, 0.6876020f, -0.6067254f};
    const float* output_data = output.ReadData<float>();
    for (size_t i = 0; i < 12; ++i) {
        REQUIRE(output_data[i] == Catch::Approx(expected_eval[i]).margin(1e-5f));
    }
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

TEST_CASE("Attention training dropout stays on ArrayFire under strict native fallback policy",
          "[arrayfire][fallback][policy][attention]") {
    cyxwiz::MultiHeadAttentionLayer attention(2, 1, 0.25f, false);
    attention.SetTraining(true);
    const float attention_values[] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
    };
    const cyxwiz::Tensor attention_input(
        {1, 2, 2}, attention_values, cyxwiz::DataType::Float32);

    const cyxwiz::ScopedArrayFireFallbackPolicy strict(
        cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
    // The ArrayFire attention path applies training dropout itself and caches
    // the mask for backward, so the strict policy must not see a CPU fallback
    // in either mode.
    REQUIRE_NOTHROW(attention.Forward(attention_input));
    REQUIRE_NOTHROW(attention.Backward(attention_input));
    attention.SetTraining(false);
    REQUIRE_NOTHROW(attention.Forward(attention_input));
    REQUIRE_NOTHROW(attention.Backward(attention_input));
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
