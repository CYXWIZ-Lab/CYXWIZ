#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cyxwiz/activation.h>
#include <cyxwiz/activations/relu.h>
#include <cyxwiz/activations/sigmoid.h>
#include <cyxwiz/activations/tanh.h>
#include <cyxwiz/memory_manager.h>
#include <cyxwiz/sequential.h>
#include <cyxwiz/tensor.h>
#include <nlohmann/json.hpp>
#include <cmath>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace {

using json = nlohmann::json;

json LoadElementwiseActivationFixture() {
    std::ifstream stream(CYXWIZ_TRAINING_CORE_FIXTURE_PATH);
    REQUIRE(stream.is_open());

    json fixture;
    stream >> fixture;
    REQUIRE(fixture.value("schema_version", 0) == 1);
    REQUIRE(fixture.at("oracle").value("name", "") == "PyTorch");
    REQUIRE(fixture.at("oracle").value("device", "") == "cpu");
    REQUIRE(!fixture.at("oracle").value("version", "").empty());
    return fixture.at("cases").at(
        "elementwise_activation_forward_backward_f32");
}

cyxwiz::Tensor ActivationTensorFromFixture(const json& value) {
    const auto shape = value.at("shape").get<std::vector<size_t>>();
    const auto values = value.at("values").get<std::vector<float>>();
    size_t element_count = 1;
    for (size_t dimension : shape) {
        element_count *= dimension;
    }
    REQUIRE(element_count == values.size());
    return cyxwiz::Tensor(
        shape, values.data(), cyxwiz::DataType::Float32);
}

std::unique_ptr<cyxwiz::Activation> CreateFixtureActivation(
    const json& test_case) {
    const std::string name = test_case.at("name").get<std::string>();
    if (name == "relu") {
        return cyxwiz::CreateActivation(cyxwiz::ActivationType::ReLU);
    }
    if (name == "leaky_relu") {
        return cyxwiz::CreateActivation(
            cyxwiz::ActivationType::LeakyReLU,
            test_case.at("parameters").at("alpha").get<float>());
    }
    if (name == "elu") {
        return cyxwiz::CreateActivation(
            cyxwiz::ActivationType::ELU,
            test_case.at("parameters").at("alpha").get<float>());
    }
    if (name == "gelu_tanh") {
        return cyxwiz::CreateActivation(cyxwiz::ActivationType::GELU);
    }
    if (name == "silu") {
        return cyxwiz::CreateActivation(cyxwiz::ActivationType::SiLU);
    }
    if (name == "sigmoid") {
        return cyxwiz::CreateActivation(cyxwiz::ActivationType::Sigmoid);
    }
    if (name == "tanh") {
        return cyxwiz::CreateActivation(cyxwiz::ActivationType::Tanh);
    }
    if (name == "mish") {
        return cyxwiz::CreateActivation(cyxwiz::ActivationType::Mish);
    }
    if (name == "hardswish") {
        return cyxwiz::CreateActivation(cyxwiz::ActivationType::Hardswish);
    }
    if (name == "selu") {
        return cyxwiz::CreateActivation(cyxwiz::ActivationType::SELU);
    }
    throw std::invalid_argument("unsupported activation fixture: " + name);
}

void CheckActivationTensor(const cyxwiz::Tensor& actual,
                           const json& expected,
                           const json& tolerance) {
    const auto expected_shape =
        expected.at("shape").get<std::vector<size_t>>();
    const auto expected_values =
        expected.at("values").get<std::vector<float>>();
    REQUIRE(actual.Shape() == expected_shape);
    REQUIRE(actual.GetDataType() == cyxwiz::DataType::Float32);
    REQUIRE(actual.NumElements() == expected_values.size());

    const double absolute = tolerance.at("atol").get<double>();
    const double relative = tolerance.at("rtol").get<double>();
    const float* actual_values = actual.ReadData<float>();
    for (size_t i = 0; i < expected_values.size(); ++i) {
        CHECK(actual_values[i] ==
              Catch::Approx(expected_values[i])
                  .margin(absolute)
                  .epsilon(relative));
    }
}

} // namespace

TEST_CASE("Elementwise activations match PyTorch forward and autograd",
          "[activation][pytorch]") {
    const auto cases = LoadElementwiseActivationFixture();
    REQUIRE(cases.is_array());
    REQUIRE(cases.size() == 10);

    for (const auto& test_case : cases) {
        const std::string name = test_case.at("name").get<std::string>();
        INFO("case=" << name);
        REQUIRE(test_case.value("dtype", "") == "float32");
        REQUIRE(test_case.value("operation", "").rfind("torch.", 0) == 0);
        REQUIRE(test_case.at("coverage").size() == 4);

        auto activation = CreateFixtureActivation(test_case);
        const auto input =
            ActivationTensorFromFixture(test_case.at("input"));
        const auto grad_output =
            ActivationTensorFromFixture(test_case.at("grad_output"));
        const auto output = activation->Forward(input);
        const auto grad_input = activation->Backward(grad_output, input);

        CheckActivationTensor(
            output,
            test_case.at("expected").at("output"),
            test_case.at("tolerance"));
        CheckActivationTensor(
            grad_input,
            test_case.at("expected").at("grad_input"),
            test_case.at("tolerance"));
    }
}

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

TEST_CASE("Factory softmax computes row-major forward values", "[activation]") {
    float values[] = {1.0f, 2.0f, 3.0f, 1.0f, 3.0f, 5.0f};
    cyxwiz::Tensor input({2, 3}, values, cyxwiz::DataType::Float32);
    cyxwiz::SoftmaxActivation softmax(-1);

    cyxwiz::Tensor output = softmax.Forward(input);
    const float* out = output.Data<float>();

    const float denom0 = std::exp(-2.0f) + std::exp(-1.0f) + 1.0f;
    REQUIRE(out[0] == Catch::Approx(std::exp(-2.0f) / denom0));
    REQUIRE(out[1] == Catch::Approx(std::exp(-1.0f) / denom0));
    REQUIRE(out[2] == Catch::Approx(1.0f / denom0));
    REQUIRE(out[0] + out[1] + out[2] == Catch::Approx(1.0f));

    const float denom1 = std::exp(-4.0f) + std::exp(-2.0f) + 1.0f;
    REQUIRE(out[3] == Catch::Approx(std::exp(-4.0f) / denom1));
    REQUIRE(out[4] == Catch::Approx(std::exp(-2.0f) / denom1));
    REQUIRE(out[5] == Catch::Approx(1.0f / denom1));
    REQUIRE(out[3] + out[4] + out[5] == Catch::Approx(1.0f));
}

TEST_CASE("Factory softmax computes row-major backward values", "[activation]") {
    float values[] = {1.0f, 2.0f, 3.0f, 1.0f, 3.0f, 5.0f};
    float grad_values[] = {0.1f, 0.2f, -0.3f, -0.2f, 0.4f, 0.1f};
    cyxwiz::Tensor input({2, 3}, values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor grad({2, 3}, grad_values, cyxwiz::DataType::Float32);
    cyxwiz::SoftmaxActivation softmax(-1);

    cyxwiz::Tensor output = softmax.Forward(input);
    cyxwiz::Tensor grad_input = softmax.Backward(grad, input);

    const float* out = output.Data<float>();
    const float* grad_in = grad_input.Data<float>();

    const float dot0 = out[0] * grad_values[0] + out[1] * grad_values[1] + out[2] * grad_values[2];
    REQUIRE(grad_in[0] == Catch::Approx(out[0] * (grad_values[0] - dot0)));
    REQUIRE(grad_in[1] == Catch::Approx(out[1] * (grad_values[1] - dot0)));
    REQUIRE(grad_in[2] == Catch::Approx(out[2] * (grad_values[2] - dot0)));

    const float dot1 = out[3] * grad_values[3] + out[4] * grad_values[4] + out[5] * grad_values[5];
    REQUIRE(grad_in[3] == Catch::Approx(out[3] * (grad_values[3] - dot1)));
    REQUIRE(grad_in[4] == Catch::Approx(out[4] * (grad_values[4] - dot1)));
    REQUIRE(grad_in[5] == Catch::Approx(out[5] * (grad_values[5] - dot1)));
}

TEST_CASE("Softmax validates axis dtype and module backward state", "[activation]") {
    float values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    cyxwiz::Tensor input({2, 3}, values, cyxwiz::DataType::Float32);
    cyxwiz::Tensor grad({2, 3}, values, cyxwiz::DataType::Float32);

    cyxwiz::SoftmaxActivation invalid_positive_axis(2);
    cyxwiz::SoftmaxActivation invalid_negative_axis(-3);
    REQUIRE_THROWS(invalid_positive_axis.Forward(input));
    REQUIRE_THROWS(invalid_negative_axis.Forward(input));

    int32_t integer_values[] = {1, 2, 3, 4, 5, 6};
    cyxwiz::Tensor integer_input(
        {2, 3}, integer_values, cyxwiz::DataType::Int32);
    cyxwiz::SoftmaxActivation activation(-1);
    REQUIRE_THROWS(activation.Forward(integer_input));

    cyxwiz::SoftmaxModule module(-1);
    REQUIRE_THROWS(module.Backward(grad));
    REQUIRE_THROWS(module.Forward(integer_input));

    module.Forward(input);
    cyxwiz::Tensor wrong_shape_grad({3, 2}, values,
                                    cyxwiz::DataType::Float32);
    REQUIRE_THROWS(module.Backward(wrong_shape_grad));
}

TEST_CASE("PReLU validates parameter and channel contracts", "[activation]") {
    REQUIRE_THROWS(cyxwiz::PReLUActivation(0));

    cyxwiz::PReLUActivation prelu(3);
    float short_alpha_values[] = {0.1f, 0.2f};
    cyxwiz::Tensor short_alpha(
        {2}, short_alpha_values, cyxwiz::DataType::Float32);
    REQUIRE_THROWS(prelu.SetAlpha(short_alpha));

    int32_t integer_alpha_values[] = {1, 2, 3};
    cyxwiz::Tensor integer_alpha(
        {3}, integer_alpha_values, cyxwiz::DataType::Int32);
    REQUIRE_THROWS(prelu.SetAlpha(integer_alpha));

    float input_values[] = {1.0f, -2.0f, 3.0f, -4.0f};
    cyxwiz::Tensor wrong_channels(
        {2, 2}, input_values, cyxwiz::DataType::Float32);
    REQUIRE_THROWS(prelu.Forward(wrong_channels));
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
