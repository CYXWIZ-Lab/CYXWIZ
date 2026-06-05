#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cyxwiz/layers/linear.h>
#include <cyxwiz/memory_manager.h>
#include <cyxwiz/tensor.h>
#include <vector>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

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
