#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cyxwiz/layer.h>
#include <cyxwiz/tensor.h>
#include <map>
#include <vector>

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
