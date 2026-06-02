#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cyxwiz/activations/relu.h>
#include <cyxwiz/activations/sigmoid.h>
#include <cyxwiz/activations/tanh.h>
#include <cyxwiz/memory_manager.h>
#include <cyxwiz/tensor.h>
#include <cmath>
#include <vector>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

TEST_CASE("Standalone activations compute forward values", "[activation]") {
    float values[] = {-1.0f, 0.0f, 2.0f};
    cyxwiz::Tensor input({3}, values, cyxwiz::DataType::Float32);

    cyxwiz::ReLU relu;
    cyxwiz::Sigmoid sigmoid;
    cyxwiz::Tanh tanh;

    cyxwiz::Tensor relu_out = relu.Forward(input);
    cyxwiz::Tensor sigmoid_out = sigmoid.Forward(input);
    cyxwiz::Tensor tanh_out = tanh.Forward(input);

    REQUIRE(relu_out.Data<float>()[0] == 0.0f);
    REQUIRE(relu_out.Data<float>()[1] == 0.0f);
    REQUIRE(relu_out.Data<float>()[2] == 2.0f);

    REQUIRE(sigmoid_out.Data<float>()[0] == Catch::Approx(1.0f / (1.0f + std::exp(1.0f))));
    REQUIRE(sigmoid_out.Data<float>()[1] == Catch::Approx(0.5f));
    REQUIRE(sigmoid_out.Data<float>()[2] == Catch::Approx(1.0f / (1.0f + std::exp(-2.0f))));

    REQUIRE(tanh_out.Data<float>()[0] == Catch::Approx(std::tanh(-1.0f)));
    REQUIRE(tanh_out.Data<float>()[1] == Catch::Approx(0.0f));
    REQUIRE(tanh_out.Data<float>()[2] == Catch::Approx(std::tanh(2.0f)));
}

TEST_CASE("Standalone activations compute backward values", "[activation]") {
    float values[] = {-1.0f, 0.0f, 2.0f};
    float grad_values[] = {1.0f, 1.0f, 1.0f};
    cyxwiz::Tensor input({3}, values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor grad({3}, grad_values, cyxwiz::DataType::Float32);

    cyxwiz::ReLU relu;
    cyxwiz::Sigmoid sigmoid;
    cyxwiz::Tanh tanh;

    cyxwiz::Tensor relu_grad = relu.Backward(grad, input);
    cyxwiz::Tensor sigmoid_grad = sigmoid.Backward(grad, input);
    cyxwiz::Tensor tanh_grad = tanh.Backward(grad, input);

    REQUIRE(relu_grad.Data<float>()[0] == 0.0f);
    REQUIRE(relu_grad.Data<float>()[1] == 0.0f);
    REQUIRE(relu_grad.Data<float>()[2] == 1.0f);

    const float sigmoid_neg = 1.0f / (1.0f + std::exp(1.0f));
    const float sigmoid_zero = 0.5f;
    const float sigmoid_pos = 1.0f / (1.0f + std::exp(-2.0f));
    REQUIRE(sigmoid_grad.Data<float>()[0] == Catch::Approx(sigmoid_neg * (1.0f - sigmoid_neg)));
    REQUIRE(sigmoid_grad.Data<float>()[1] == Catch::Approx(sigmoid_zero * (1.0f - sigmoid_zero)));
    REQUIRE(sigmoid_grad.Data<float>()[2] == Catch::Approx(sigmoid_pos * (1.0f - sigmoid_pos)));

    REQUIRE(tanh_grad.Data<float>()[0] == Catch::Approx(1.0f - std::tanh(-1.0f) * std::tanh(-1.0f)));
    REQUIRE(tanh_grad.Data<float>()[1] == Catch::Approx(1.0f));
    REQUIRE(tanh_grad.Data<float>()[2] == Catch::Approx(1.0f - std::tanh(2.0f) * std::tanh(2.0f)));
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

TEST_CASE("Standalone activation GPU outputs materialize host data lazily", "[activation][arrayfire]") {
    if (!HasArrayFireDeviceBackend()) {
        return;
    }

    cyxwiz::ReLU relu;
    cyxwiz::Sigmoid sigmoid;
    cyxwiz::Tanh tanh;

    cyxwiz::Tensor input(af::constant(2.0f, 3, f32));
    const size_t before = cyxwiz::MemoryManager::GetAllocatedBytes();

    cyxwiz::Tensor relu_out = relu.Forward(input);
    cyxwiz::Tensor sigmoid_out = sigmoid.Forward(input);
    cyxwiz::Tensor tanh_out = tanh.Forward(input);

    REQUIRE(cyxwiz::MemoryManager::GetAllocatedBytes() == before);

    REQUIRE(relu_out.Data<float>()[0] == 2.0f);
    REQUIRE(sigmoid_out.Data<float>()[0] == Catch::Approx(1.0f / (1.0f + std::exp(-2.0f))));
    REQUIRE(tanh_out.Data<float>()[0] == Catch::Approx(std::tanh(2.0f)));
    REQUIRE(cyxwiz::MemoryManager::GetAllocatedBytes() >=
            before + relu_out.NumBytes() + sigmoid_out.NumBytes() + tanh_out.NumBytes());
}

TEST_CASE("Standalone activation GPU gradients materialize host data lazily", "[activation][arrayfire]") {
    if (!HasArrayFireDeviceBackend()) {
        return;
    }

    cyxwiz::ReLU relu;
    cyxwiz::Sigmoid sigmoid;
    cyxwiz::Tanh tanh;

    cyxwiz::Tensor input(af::constant(2.0f, 3, f32));
    cyxwiz::Tensor grad(af::constant(1.0f, 3, f32));
    const size_t before = cyxwiz::MemoryManager::GetAllocatedBytes();

    cyxwiz::Tensor relu_grad = relu.Backward(grad, input);
    cyxwiz::Tensor sigmoid_grad = sigmoid.Backward(grad, input);
    cyxwiz::Tensor tanh_grad = tanh.Backward(grad, input);

    REQUIRE(cyxwiz::MemoryManager::GetAllocatedBytes() == before);

    const float sigmoid_value = 1.0f / (1.0f + std::exp(-2.0f));
    const float tanh_value = std::tanh(2.0f);
    REQUIRE(relu_grad.Data<float>()[0] == 1.0f);
    REQUIRE(sigmoid_grad.Data<float>()[0] == Catch::Approx(sigmoid_value * (1.0f - sigmoid_value)));
    REQUIRE(tanh_grad.Data<float>()[0] == Catch::Approx(1.0f - tanh_value * tanh_value));
    REQUIRE(cyxwiz::MemoryManager::GetAllocatedBytes() >=
            before + relu_grad.NumBytes() + sigmoid_grad.NumBytes() + tanh_grad.NumBytes());
}
#endif
