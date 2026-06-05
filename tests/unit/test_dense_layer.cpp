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
