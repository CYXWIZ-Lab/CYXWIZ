#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cyxwiz/layers/linear.h>
#include <cyxwiz/memory_manager.h>
#include <cyxwiz/tensor.h>
#include <vector>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

TEST_CASE("LinearLayer matches PyTorch forward and backward fixtures",
          "[linear][pytorch][route]") {
    SECTION("rank-2 biased") {
        cyxwiz::LinearLayer layer(3, 2, true);
        const float weights[] = {
            0.5f, -1.0f, 2.0f,
            -0.25f, 0.75f, 1.5f,
        };
        const float bias[] = {0.1f, -0.2f};
        layer.SetParameters({
            {"weight", cyxwiz::Tensor(
                           {2, 3}, weights, cyxwiz::DataType::Float32)},
            {"bias", cyxwiz::Tensor(
                         {2}, bias, cyxwiz::DataType::Float32)},
        });

        const float input_values[] = {
            1.0f, 2.0f, -1.0f,
            0.0f, -3.0f, 4.0f,
        };
        const auto output = layer.Forward(cyxwiz::Tensor(
            {2, 3}, input_values, cyxwiz::DataType::Float32));
        const float expected_output[] = {-3.4f, -0.45f, 11.1f, 3.55f};
        REQUIRE(output.Shape() == std::vector<size_t>{2, 2});
        for (size_t i = 0; i < 4; ++i) {
            REQUIRE(output.ReadData<float>()[i] ==
                    Catch::Approx(expected_output[i]).margin(1e-5f));
        }

        const float grad_output_values[] = {1.0f, -0.5f, 0.25f, 2.0f};
        const auto grad_input = layer.Backward(cyxwiz::Tensor(
            {2, 2}, grad_output_values, cyxwiz::DataType::Float32));
        const float expected_grad_input[] = {
            0.625f, -1.375f, 1.25f,
            -0.375f, 1.25f, 3.5f,
        };
        REQUIRE(grad_input.Shape() == std::vector<size_t>{2, 3});
        for (size_t i = 0; i < 6; ++i) {
            REQUIRE(grad_input.ReadData<float>()[i] ==
                    Catch::Approx(expected_grad_input[i]).margin(1e-5f));
        }

        const auto gradients = layer.GetGradients();
        const float expected_grad_weight[] = {
            1.0f, 1.25f, 0.0f,
            -0.5f, -7.0f, 8.5f,
        };
        const float expected_grad_bias[] = {1.25f, 1.5f};
        for (size_t i = 0; i < 6; ++i) {
            REQUIRE(gradients.at("weight").ReadData<float>()[i] ==
                    Catch::Approx(expected_grad_weight[i]).margin(1e-5f));
        }
        for (size_t i = 0; i < 2; ++i) {
            REQUIRE(gradients.at("bias").ReadData<float>()[i] ==
                    Catch::Approx(expected_grad_bias[i]).margin(1e-5f));
        }
    }

    SECTION("rank-1 unbiased") {
        cyxwiz::LinearLayer layer(3, 2, false);
        const float weights[] = {
            0.5f, -1.0f, 2.0f,
            -0.25f, 0.75f, 1.5f,
        };
        layer.SetParameters({
            {"weight", cyxwiz::Tensor(
                           {2, 3}, weights, cyxwiz::DataType::Float32)},
        });

        const float input_values[] = {1.5f, -2.0f, 0.25f};
        const auto output = layer.Forward(cyxwiz::Tensor(
            {3}, input_values, cyxwiz::DataType::Float32));
        const float expected_output[] = {3.25f, -1.5f};
        REQUIRE(output.Shape() == std::vector<size_t>{2});
        for (size_t i = 0; i < 2; ++i) {
            REQUIRE(output.ReadData<float>()[i] ==
                    Catch::Approx(expected_output[i]).margin(1e-5f));
        }

        const float grad_output_values[] = {-0.75f, 1.25f};
        const auto grad_input = layer.Backward(cyxwiz::Tensor(
            {2}, grad_output_values, cyxwiz::DataType::Float32));
        const float expected_grad_input[] = {-0.6875f, 1.6875f, 0.375f};
        REQUIRE(grad_input.Shape() == std::vector<size_t>{3});
        for (size_t i = 0; i < 3; ++i) {
            REQUIRE(grad_input.ReadData<float>()[i] ==
                    Catch::Approx(expected_grad_input[i]).margin(1e-5f));
        }

        const auto gradients = layer.GetGradients();
        const float expected_grad_weight[] = {
            -1.125f, 1.5f, -0.1875f,
            1.875f, -2.5f, 0.3125f,
        };
        REQUIRE(gradients.count("bias") == 0);
        for (size_t i = 0; i < 6; ++i) {
            REQUIRE(gradients.at("weight").ReadData<float>()[i] ==
                    Catch::Approx(expected_grad_weight[i]).margin(1e-5f));
        }
    }
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
static bool HasArrayFireDeviceBackend() {
    try {
        af::Backend backend = af::getActiveBackend();
        return backend == AF_BACKEND_CUDA || backend == AF_BACKEND_OPENCL;
    } catch (...) {
        return false;
    }
}

TEST_CASE("LinearLayer GPU forward keeps output device resident until host read", "[linear][arrayfire]") {
    if (!HasArrayFireDeviceBackend()) {
        return;
    }

    cyxwiz::LinearLayer layer(3, 2, true);
    cyxwiz::Tensor input = cyxwiz::Tensor::FromArrayRowMajor2D(af::constant(1.0f, 4, 3, f32));

    const size_t before = cyxwiz::MemoryManager::GetAllocatedBytes();
    cyxwiz::Tensor output = layer.Forward(input);

    REQUIRE(output.Shape() == std::vector<size_t>{4, 2});
    REQUIRE(cyxwiz::MemoryManager::GetAllocatedBytes() == before);

    const float* host = output.Data<float>();
    REQUIRE(host != nullptr);
    REQUIRE(cyxwiz::MemoryManager::GetAllocatedBytes() >= before + output.NumBytes());
}

TEST_CASE("LinearLayer GPU backward keeps gradients device resident until host read", "[linear][arrayfire]") {
    if (!HasArrayFireDeviceBackend()) {
        return;
    }

    cyxwiz::LinearLayer layer(3, 2, true);
    cyxwiz::Tensor input = cyxwiz::Tensor::FromArrayRowMajor2D(af::constant(1.0f, 4, 3, f32));
    cyxwiz::Tensor output = layer.Forward(input);
    cyxwiz::Tensor grad_output = cyxwiz::Tensor::FromArrayRowMajor2D(af::constant(1.0f, 4, 2, f32));

    const size_t before = cyxwiz::MemoryManager::GetAllocatedBytes();
    cyxwiz::Tensor grad_input = layer.Backward(grad_output);
    auto gradients = layer.GetGradients();

    REQUIRE(grad_input.Shape() == std::vector<size_t>{4, 3});
    REQUIRE(gradients.at("weight").Shape() == std::vector<size_t>{2, 3});
    REQUIRE(gradients.at("bias").Shape() == std::vector<size_t>{2});
    const size_t after_backward = cyxwiz::MemoryManager::GetAllocatedBytes();
    REQUIRE(after_backward <= before);

    const float* grad_input_host = grad_input.Data<float>();
    const float* weight_grad_host = gradients.at("weight").Data<float>();
    const float* bias_grad_host = gradients.at("bias").Data<float>();
    REQUIRE(grad_input_host != nullptr);
    REQUIRE(weight_grad_host != nullptr);
    REQUIRE(bias_grad_host != nullptr);
    REQUIRE(cyxwiz::MemoryManager::GetAllocatedBytes() >=
            after_backward + grad_input.NumBytes() +
            gradients.at("weight").NumBytes() +
            gradients.at("bias").NumBytes());

    (void)output;
}
#endif
