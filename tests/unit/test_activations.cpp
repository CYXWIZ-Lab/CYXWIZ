#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cyxwiz/activation.h>
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

TEST_CASE("Factory activations compute core forward values", "[activation]") {
    float values[] = {-1.0f, 0.0f, 2.0f};
    cyxwiz::Tensor input({3}, values, cyxwiz::DataType::Float32);

    auto relu = cyxwiz::CreateActivation(cyxwiz::ActivationType::ReLU);
    auto sigmoid = cyxwiz::CreateActivation(cyxwiz::ActivationType::Sigmoid);
    auto tanh = cyxwiz::CreateActivation(cyxwiz::ActivationType::Tanh);

    cyxwiz::Tensor relu_out = relu->Forward(input);
    cyxwiz::Tensor sigmoid_out = sigmoid->Forward(input);
    cyxwiz::Tensor tanh_out = tanh->Forward(input);

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

TEST_CASE("Factory activations compute core backward values", "[activation]") {
    float values[] = {-1.0f, 0.0f, 2.0f};
    float grad_values[] = {1.0f, 1.0f, 1.0f};
    cyxwiz::Tensor input({3}, values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor grad({3}, grad_values, cyxwiz::DataType::Float32);

    auto relu = cyxwiz::CreateActivation(cyxwiz::ActivationType::ReLU);
    auto sigmoid = cyxwiz::CreateActivation(cyxwiz::ActivationType::Sigmoid);
    auto tanh = cyxwiz::CreateActivation(cyxwiz::ActivationType::Tanh);

    cyxwiz::Tensor relu_grad = relu->Backward(grad, input);
    cyxwiz::Tensor sigmoid_grad = sigmoid->Backward(grad, input);
    cyxwiz::Tensor tanh_grad = tanh->Backward(grad, input);

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

TEST_CASE("Factory activations compute elementwise forward values", "[activation]") {
    float values[] = {-4.0f, -1.0f, 0.0f, 2.0f, 4.0f};
    cyxwiz::Tensor input({5}, values, cyxwiz::DataType::Float32);

    auto leaky_relu = cyxwiz::CreateActivation(cyxwiz::ActivationType::LeakyReLU, 0.2f);
    auto elu = cyxwiz::CreateActivation(cyxwiz::ActivationType::ELU, 1.5f);
    auto swish = cyxwiz::CreateActivation(cyxwiz::ActivationType::Swish);
    auto mish = cyxwiz::CreateActivation(cyxwiz::ActivationType::Mish);
    auto hardswish = cyxwiz::CreateActivation(cyxwiz::ActivationType::Hardswish);
    auto selu = cyxwiz::CreateActivation(cyxwiz::ActivationType::SELU);
    auto prelu = cyxwiz::CreateActivation(cyxwiz::ActivationType::PReLU, 0.25f);

    REQUIRE(leaky_relu->Forward(input).Data<float>()[0] == Catch::Approx(-0.8f));
    REQUIRE(elu->Forward(input).Data<float>()[1] == Catch::Approx(1.5f * (std::exp(-1.0f) - 1.0f)));

    const float sigmoid_2 = 1.0f / (1.0f + std::exp(-2.0f));
    REQUIRE(swish->Forward(input).Data<float>()[3] == Catch::Approx(2.0f * sigmoid_2));

    const float softplus_2 = std::log1p(std::exp(2.0f));
    REQUIRE(mish->Forward(input).Data<float>()[3] == Catch::Approx(2.0f * std::tanh(softplus_2)));

    REQUIRE(hardswish->Forward(input).Data<float>()[0] == Catch::Approx(0.0f));
    REQUIRE(hardswish->Forward(input).Data<float>()[3] == Catch::Approx(2.0f * 5.0f / 6.0f));

    REQUIRE(selu->Forward(input).Data<float>()[3] ==
            Catch::Approx(cyxwiz::SELUActivation::SCALE * 2.0f));
    REQUIRE(prelu->Forward(input).Data<float>()[1] == Catch::Approx(-0.25f));
}

TEST_CASE("Factory activations compute elementwise backward values", "[activation]") {
    float values[] = {-4.0f, -1.0f, 0.0f, 2.0f, 4.0f};
    float grad_values[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    cyxwiz::Tensor input({5}, values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor grad({5}, grad_values, cyxwiz::DataType::Float32);

    auto leaky_relu = cyxwiz::CreateActivation(cyxwiz::ActivationType::LeakyReLU, 0.2f);
    auto elu = cyxwiz::CreateActivation(cyxwiz::ActivationType::ELU, 1.5f);
    auto swish = cyxwiz::CreateActivation(cyxwiz::ActivationType::Swish);
    auto hardswish = cyxwiz::CreateActivation(cyxwiz::ActivationType::Hardswish);
    auto selu = cyxwiz::CreateActivation(cyxwiz::ActivationType::SELU);
    auto prelu = cyxwiz::CreateActivation(cyxwiz::ActivationType::PReLU, 0.25f);

    REQUIRE(leaky_relu->Backward(grad, input).Data<float>()[0] == Catch::Approx(0.2f));
    REQUIRE(elu->Backward(grad, input).Data<float>()[1] == Catch::Approx(1.5f * std::exp(-1.0f)));

    const float sigmoid_2 = 1.0f / (1.0f + std::exp(-2.0f));
    REQUIRE(swish->Backward(grad, input).Data<float>()[3] ==
            Catch::Approx(sigmoid_2 * (1.0f + 2.0f * (1.0f - sigmoid_2))));

    REQUIRE(hardswish->Backward(grad, input).Data<float>()[0] == Catch::Approx(0.0f));
    REQUIRE(hardswish->Backward(grad, input).Data<float>()[3] == Catch::Approx(7.0f / 6.0f));

    REQUIRE(selu->Backward(grad, input).Data<float>()[3] ==
            Catch::Approx(cyxwiz::SELUActivation::SCALE));
    REQUIRE(prelu->Backward(grad, input).Data<float>()[1] == Catch::Approx(0.25f));
    REQUIRE(static_cast<cyxwiz::PReLUActivation*>(prelu.get())->GetAlphaGradient().Data<float>()[0] ==
            Catch::Approx(-5.0f));
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
