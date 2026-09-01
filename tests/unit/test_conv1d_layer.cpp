#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "algorithms/arrayfire_backend_utils.h"

#include <cyxwiz/device.h>
#include <cyxwiz/layers/convolution.h>
#include <cyxwiz/optimizers/sgd.h>
#include <cyxwiz/sequential.h>
#include <cyxwiz/tensor.h>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void CheckValues(const cyxwiz::Tensor& actual,
                 const std::vector<size_t>& expected_shape,
                 const std::vector<float>& expected_values) {
    REQUIRE(actual.Shape() == expected_shape);
    REQUIRE(actual.GetDataType() == cyxwiz::DataType::Float32);
    REQUIRE(actual.NumElements() == expected_values.size());
    const float* data = actual.ReadData<float>();
    for (size_t index = 0; index < expected_values.size(); ++index) {
        CHECK(data[index] == Catch::Approx(expected_values[index]));
    }
}

void ConfigureReferenceConv(cyxwiz::Conv1DLayer& layer) {
    const std::vector<float> weights{1.0f, 2.0f};
    const std::vector<float> bias{0.5f};
    layer.SetParameters({
        {"weights", cyxwiz::Tensor(
                        {1, 1, 2}, weights.data(),
                        cyxwiz::DataType::Float32)},
        {"bias", cyxwiz::Tensor(
                     {1}, bias.data(), cyxwiz::DataType::Float32)},
    });
}

#ifdef CYXWIZ_HAS_ARRAYFIRE

size_t conv_host_sync_count = 0;
size_t conv_fallback_count = 0;
bool saw_conv_cpu_path = false;
cyxwiz::ArrayFireNativeCpuFallbackEvent last_conv_fallback;

void CountConvHostSync(const cyxwiz::ArrayFireHostSyncEvent& event) {
    ++conv_host_sync_count;
    saw_conv_cpu_path |=
        event.attribution_category == "layer_cpu_path";
}

void CountConvFallback(
    const cyxwiz::ArrayFireNativeCpuFallbackEvent& event) {
    ++conv_fallback_count;
    last_conv_fallback = event;
}

void ResetConvObservations() {
    conv_host_sync_count = 0;
    conv_fallback_count = 0;
    saw_conv_cpu_path = false;
    last_conv_fallback = {};
}

af::dim4 SemanticDims(const std::vector<size_t>& shape) {
    REQUIRE_FALSE(shape.empty());
    REQUIRE(shape.size() <= 4);
    af::dim4 dims(1, 1, 1, 1);
    for (size_t axis = 0; axis < shape.size(); ++axis) {
        dims[static_cast<unsigned>(axis)] =
            static_cast<dim_t>(shape[axis]);
    }
    return dims;
}

cyxwiz::Tensor DeviceOnlyTensor(const std::vector<size_t>& shape,
                                const std::vector<float>& values) {
    const cyxwiz::Tensor host(
        shape, values.data(), cyxwiz::DataType::Float32);
    af::array semantic = host.GetSemanticArray();
    semantic.eval();
    return cyxwiz::Tensor::FromSemanticArray(semantic, shape);
}

cyxwiz::Tensor DeviceOnlyOnes(const std::vector<size_t>& shape) {
    af::array values = af::constant(
        1.0f, SemanticDims(shape), af::dtype::f32);
    values.eval();
    return cyxwiz::Tensor::FromSemanticArray(values, shape);
}

const char* ExpectedBackendName(cyxwiz::DeviceType type) {
    switch (type) {
        case cyxwiz::DeviceType::CPU: return "cpu";
        case cyxwiz::DeviceType::CUDA: return "cuda";
        case cyxwiz::DeviceType::OPENCL: return "opencl";
        case cyxwiz::DeviceType::ONEAPI: return "oneapi";
        default: return "unsupported";
    }
}

void ConfigureReferenceConvDeviceOnly(cyxwiz::Conv1DLayer& layer) {
    layer.SetParameters({
        {"weights", DeviceOnlyTensor({1, 1, 2}, {1.0f, 2.0f})},
        {"bias", DeviceOnlyTensor({1}, {0.5f})},
    });
}

void SetEnvVar(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

void ClearEnvVar(const char* name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

class ScopedEnvVar {
public:
    ScopedEnvVar(const char* name, const char* value) : name_(name) {
        const char* previous = std::getenv(name);
        if (previous != nullptr) {
            had_previous_ = true;
            previous_ = previous;
        }
        SetEnvVar(name_, value);
    }

    ~ScopedEnvVar() {
        if (had_previous_) {
            SetEnvVar(name_, previous_.c_str());
        } else {
            ClearEnvVar(name_);
        }
    }

private:
    const char* name_;
    bool had_previous_ = false;
    std::string previous_;
};

#endif

} // namespace

TEST_CASE("Conv1D computes deterministic forward and backward values",
          "[conv][conv1d][correctness]") {
    cyxwiz::Conv1DLayer layer(1, 1, 2, 1, 0, 1, true);
    ConfigureReferenceConv(layer);

    const std::vector<float> input_values{
        1.0f, 10.0f,
        2.0f, 20.0f,
        3.0f, 30.0f,
        4.0f, 40.0f,
    };
    const cyxwiz::Tensor input(
        {4, 1, 2}, input_values.data(), cyxwiz::DataType::Float32);
    CheckValues(
        layer.Forward(input),
        {3, 1, 2},
        {5.5f, 50.5f, 8.5f, 80.5f, 11.5f, 110.5f});

    const std::vector<float> grad_values{
        1.0f, 10.0f,
        2.0f, 20.0f,
        3.0f, 30.0f,
    };
    const cyxwiz::Tensor grad_output(
        {3, 1, 2}, grad_values.data(), cyxwiz::DataType::Float32);
    CheckValues(
        layer.Backward(grad_output),
        {4, 1, 2},
        {1.0f, 10.0f, 4.0f, 40.0f,
         7.0f, 70.0f, 6.0f, 60.0f});

    const std::map<std::string, cyxwiz::Tensor> params =
        layer.GetParameters();
    CheckValues(
        params.at("grad_weights"), {1, 1, 2}, {1414.0f, 2020.0f});
    CheckValues(params.at("grad_bias"), {1}, {66.0f});
}

TEST_CASE("Conv1D preserves channel output and batch layout",
          "[conv][conv1d][correctness][layout]") {
    cyxwiz::Conv1DLayer layer(2, 2, 1, 1, 0, 1, true);
    const std::vector<float> weights{1.0f, 2.0f, -1.0f, 0.5f};
    const std::vector<float> bias{0.5f, -0.5f};
    layer.SetParameters({
        {"weights", cyxwiz::Tensor(
                        {2, 2, 1}, weights.data(),
                        cyxwiz::DataType::Float32)},
        {"bias", cyxwiz::Tensor(
                     {2}, bias.data(), cyxwiz::DataType::Float32)},
    });

    const std::vector<float> input_values{
        1.0f, 10.0f, 2.0f, 20.0f,
        3.0f, 30.0f, 4.0f, 40.0f,
    };
    const cyxwiz::Tensor input(
        {2, 2, 2}, input_values.data(), cyxwiz::DataType::Float32);
    CheckValues(
        layer.Forward(input),
        {2, 2, 2},
        {5.5f, 50.5f, -0.5f, -0.5f,
         11.5f, 110.5f, -1.5f, -10.5f});

    const std::vector<float> grad_values{
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
    };
    const cyxwiz::Tensor grad_output(
        {2, 2, 2}, grad_values.data(), cyxwiz::DataType::Float32);
    CheckValues(
        layer.Backward(grad_output),
        {2, 2, 2},
        {-2.0f, -2.0f, 3.5f, 6.0f,
         -2.0f, -2.0f, 13.5f, 16.0f});

    const std::map<std::string, cyxwiz::Tensor> params =
        layer.GetParameters();
    CheckValues(
        params.at("grad_weights"),
        {2, 2, 1},
        {216.0f, 302.0f, 304.0f, 434.0f});
    CheckValues(params.at("grad_bias"), {2}, {14.0f, 22.0f});
}

TEST_CASE("Conv1D handles padding and stride in forward and backward",
          "[conv][conv1d][correctness][padding]") {
    cyxwiz::Conv1DLayer layer(1, 1, 2, 2, 1, 1, false);
    const std::vector<float> weights{1.0f, 1.0f};
    layer.SetParameters({
        {"weights", cyxwiz::Tensor(
                        {1, 1, 2}, weights.data(),
                        cyxwiz::DataType::Float32)},
    });

    const std::vector<float> input_values{1.0f, 2.0f, 3.0f};
    const cyxwiz::Tensor input(
        {3, 1, 1}, input_values.data(), cyxwiz::DataType::Float32);
    CheckValues(layer.Forward(input), {2, 1, 1}, {1.0f, 5.0f});

    const std::vector<float> grad_values{1.0f, 2.0f};
    const cyxwiz::Tensor grad_output(
        {2, 1, 1}, grad_values.data(), cyxwiz::DataType::Float32);
    CheckValues(layer.Backward(grad_output), {3, 1, 1}, {1.0f, 2.0f, 2.0f});
    CheckValues(
        layer.GetParameters().at("grad_weights"),
        {1, 1, 2},
        {4.0f, 7.0f});
}

TEST_CASE("Conv1D updates parameters through a multi-batch SequentialModel",
          "[conv][conv1d][module][optimizer][multi_batch]") {
    auto module = std::make_unique<cyxwiz::Conv1DModule>(
        1, 1, 1, 1, 0, 1, false);
    const std::vector<float> weights{2.0f};
    module->SetParameters({
        {"weights", cyxwiz::Tensor(
                        {1, 1, 1}, weights.data(),
                        cyxwiz::DataType::Float32)},
    });

    cyxwiz::SequentialModel model;
    model.AddModule(std::move(module));
    const std::vector<float> input_values{3.0f, 4.0f};
    const cyxwiz::Tensor input(
        {1, 1, 2}, input_values.data(), cyxwiz::DataType::Float32);
    CheckValues(model.Forward(input), {1, 1, 2}, {6.0f, 8.0f});

    const std::vector<float> grad_values{1.0f, 2.0f};
    const cyxwiz::Tensor grad_output(
        {1, 1, 2}, grad_values.data(), cyxwiz::DataType::Float32);
    CheckValues(model.Backward(grad_output), {1, 1, 2}, {2.0f, 4.0f});
    CheckValues(
        model.GetGradients().at("layer0.weights"),
        {1, 1, 1},
        {11.0f});

    cyxwiz::SGDOptimizer optimizer(0.1);
    model.UpdateParameters(&optimizer);
    CheckValues(
        model.GetParameters().at("layer0.weights"),
        {1, 1, 1},
        {0.9f});
    CheckValues(model.Forward(input), {1, 1, 2}, {2.7f, 3.6f});
}

TEST_CASE("Conv1D validates lifecycle shapes dtypes and parameters",
          "[conv][conv1d][validation]") {
    cyxwiz::Conv1DLayer layer(1, 1, 2, 1, 0, 1, true);
    ConfigureReferenceConv(layer);
    const std::vector<float> input_values(4, 1.0f);
    const cyxwiz::Tensor input(
        {4, 1, 1}, input_values.data(), cyxwiz::DataType::Float32);
    const std::vector<float> grad_values(3, 1.0f);
    const cyxwiz::Tensor grad(
        {3, 1, 1}, grad_values.data(), cyxwiz::DataType::Float32);

    CHECK_THROWS_AS(layer.Backward(grad), std::logic_error);
    (void)layer.Forward(input);
    const std::vector<float> wrong_grad_values(2, 1.0f);
    const cyxwiz::Tensor wrong_grad(
        {2, 1, 1}, wrong_grad_values.data(),
        cyxwiz::DataType::Float32);
    CHECK_THROWS_AS(layer.Backward(wrong_grad), std::runtime_error);

    const std::vector<float> wrong_channels(8, 1.0f);
    const cyxwiz::Tensor wrong_channel_input(
        {4, 2, 1}, wrong_channels.data(), cyxwiz::DataType::Float32);
    CHECK_THROWS_AS(
        layer.Forward(wrong_channel_input), std::runtime_error);
    CHECK_THROWS_AS(layer.Backward(grad), std::logic_error);

    ConfigureReferenceConv(layer);
    (void)layer.Forward(input);
    ConfigureReferenceConv(layer);
    CHECK_THROWS_AS(layer.Backward(grad), std::logic_error);

    const std::vector<double> double_values(4, 1.0);
    const cyxwiz::Tensor double_input(
        {4, 1, 1}, double_values.data(), cyxwiz::DataType::Float64);
    CHECK_THROWS_AS(layer.Forward(double_input), std::runtime_error);

    const std::vector<float> wrong_weights(3, 1.0f);
    layer.SetParameters({
        {"weights", cyxwiz::Tensor(
                        {1, 1, 3}, wrong_weights.data(),
                        cyxwiz::DataType::Float32)},
    });
    CHECK_THROWS_AS(layer.Forward(input), std::runtime_error);

    CHECK_THROWS_AS(
        cyxwiz::Conv1DLayer(0, 1, 1), std::invalid_argument);
    CHECK_THROWS_AS(
        cyxwiz::Conv1DLayer(1, 1, 0), std::invalid_argument);
    CHECK_THROWS_AS(
        cyxwiz::Conv1DLayer(1, 1, 1, 0), std::invalid_argument);
    CHECK_THROWS_AS(
        cyxwiz::Conv1DLayer(1, 1, 1, 1, -1), std::invalid_argument);
    CHECK_THROWS_AS(
        cyxwiz::Conv1DLayer(1, 1, 1, 1, 0, 0),
        std::invalid_argument);
}

#ifdef CYXWIZ_HAS_ARRAYFIRE

TEST_CASE("Conv1D remains ArrayFire resident under strict fallback policy",
          "[conv][conv1d][arrayfire][residency]") {
    const cyxwiz::Device* current = cyxwiz::Device::GetCurrentDevice();
    REQUIRE(current != nullptr);
    const cyxwiz::DeviceType original_type = current->GetType();
    const int original_id = current->GetDeviceId();
    struct RestoreDevice {
        cyxwiz::DeviceType type;
        int id;
        ~RestoreDevice() {
            cyxwiz::Device(type, id).SetActive();
        }
    } restore{original_type, original_id};

    const char* requested_value =
        std::getenv("CYXWIZ_TEST_ARRAYFIRE_BACKEND");
    const std::string requested_backend =
        requested_value != nullptr && requested_value[0] != '\0'
            ? requested_value
            : cyxwiz::CurrentArrayFireBackendName();
    size_t exercised_backends = 0;
    for (const cyxwiz::DeviceInfo& info :
         cyxwiz::Device::GetAvailableDevices()) {
        if ((info.type != cyxwiz::DeviceType::CPU &&
             info.type != cyxwiz::DeviceType::CUDA &&
             info.type != cyxwiz::DeviceType::OPENCL &&
             info.type != cyxwiz::DeviceType::ONEAPI) ||
            !info.device_selectable) {
            continue;
        }
        if (info.type == cyxwiz::DeviceType::ONEAPI &&
            !cyxwiz::IsUncertifiedOneAPITrainingEnabled()) {
            continue;
        }
        if (requested_backend != ExpectedBackendName(info.type)) {
            continue;
        }

        const cyxwiz::Device device(info.type, info.device_id);
        const cyxwiz::DeviceActivationResult activation =
            device.ActivateExact(true);
        if (!activation.success) {
            CHECK(info.type != cyxwiz::DeviceType::CPU);
            continue;
        }
        REQUIRE(activation.effective_type == info.type);
        REQUIRE(activation.effective_device_id == info.device_id);
        CHECK(cyxwiz::CurrentArrayFireBackendName() ==
              ExpectedBackendName(info.type));
        ++exercised_backends;

        cyxwiz::Conv1DLayer layer(1, 1, 2, 1, 1, 1, true);
        ConfigureReferenceConvDeviceOnly(layer);
        const cyxwiz::Tensor input = DeviceOnlyTensor(
            {3, 1, 2},
            {1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 30.0f});

        ResetConvObservations();
        cyxwiz::Tensor output;
        cyxwiz::Tensor grad_input;
        {
            const cyxwiz::ScopedArrayFireFallbackPolicy strict(
                cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
            const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver
                fallback_observer(&CountConvFallback);
            const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
                &CountConvHostSync);
            output = layer.Forward(input);
            grad_input = layer.Backward(DeviceOnlyOnes(output.Shape()));
            const std::map<std::string, cyxwiz::Tensor> params =
                layer.GetParameters();
            output.GetSemanticArray().eval();
            grad_input.GetSemanticArray().eval();
            params.at("grad_weights").GetSemanticArray().eval();
            params.at("grad_bias").GetSemanticArray().eval();
            af::sync();
        }

        CHECK(conv_fallback_count == 0);
        CHECK(conv_host_sync_count == 0);
        CHECK(output.Shape() == std::vector<size_t>{4, 1, 2});
        CHECK(grad_input.Shape() == input.Shape());
    }
    if (exercised_backends == 0) {
        if (requested_backend == "oneapi" &&
            !cyxwiz::IsUncertifiedOneAPITrainingEnabled()) {
            SKIP("oneAPI Conv1D is skipped by the existing uncertified-training policy");
        }
        SKIP("Requested ArrayFire backend is unavailable or failed execution validation");
    }
}

TEST_CASE("Conv1D declares unsupported dilation before fallback",
          "[conv][conv1d][arrayfire][fallback][policy]") {
    cyxwiz::Conv1DLayer compatible_layer(1, 1, 2, 1, 0, 2, false);
    compatible_layer.SetParameters({
        {"weights", DeviceOnlyTensor({1, 1, 2}, {1.0f, 2.0f})},
    });
    const cyxwiz::Tensor input =
        DeviceOnlyTensor({3, 1, 1}, {1.0f, 2.0f, 3.0f});

    ResetConvObservations();
    cyxwiz::Tensor output;
    {
        const cyxwiz::ScopedArrayFireFallbackPolicy compatible(
            cyxwiz::ArrayFireFallbackPolicy::AllowNativeCpuFallback);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CountConvFallback);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
            &CountConvHostSync);
        output = compatible_layer.Forward(input);
    }
    CHECK(conv_fallback_count == 1);
    CHECK(conv_host_sync_count >= 1);
    CHECK(saw_conv_cpu_path);
    CHECK(last_conv_fallback.operation_name == "Conv1DLayer::Forward");
    CHECK(last_conv_fallback.reason_code == "unsupported_shape");
    CHECK_FALSE(last_conv_fallback.fallback_forbidden);
    CheckValues(output, {1, 1, 1}, {7.0f});

    cyxwiz::Conv1DLayer strict_layer(1, 1, 2, 1, 0, 2, false);
    strict_layer.SetParameters({
        {"weights", DeviceOnlyTensor({1, 1, 2}, {1.0f, 2.0f})},
    });
    ResetConvObservations();
    {
        const cyxwiz::ScopedArrayFireFallbackPolicy strict(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CountConvFallback);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
            &CountConvHostSync);
        CHECK_THROWS_AS(strict_layer.Forward(input), std::runtime_error);
    }
    CHECK(conv_fallback_count == 1);
    CHECK(conv_host_sync_count == 0);
    CHECK(last_conv_fallback.fallback_forbidden);
}

#ifndef NDEBUG

TEST_CASE("Conv1D forced native fallback is compatible and attributed",
          "[conv][conv1d][arrayfire][fallback]") {
    constexpr const char* force_name =
        "CYXWIZ_TEST_FORCE_ARRAYFIRE_FALLBACK";
    cyxwiz::Conv1DLayer layer(1, 1, 2, 1, 0, 1, true);
    ConfigureReferenceConvDeviceOnly(layer);
    const cyxwiz::Tensor input = DeviceOnlyTensor(
        {4, 1, 1}, {1.0f, 2.0f, 3.0f, 4.0f});

    ResetConvObservations();
    cyxwiz::Tensor output;
    {
        const ScopedEnvVar force(force_name, "Conv1DLayer::Forward");
        const cyxwiz::ScopedArrayFireFallbackPolicy compatible(
            cyxwiz::ArrayFireFallbackPolicy::AllowNativeCpuFallback);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CountConvFallback);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
            &CountConvHostSync);
        output = layer.Forward(input);
    }
    CHECK(conv_fallback_count == 1);
    CHECK(conv_host_sync_count >= 1);
    CHECK(saw_conv_cpu_path);
    CHECK_FALSE(last_conv_fallback.fallback_forbidden);
    CheckValues(output, {3, 1, 1}, {5.5f, 8.5f, 11.5f});

    ResetConvObservations();
    cyxwiz::Tensor grad_input;
    {
        const ScopedEnvVar force(force_name, "Conv1DLayer::Backward");
        const cyxwiz::ScopedArrayFireFallbackPolicy compatible(
            cyxwiz::ArrayFireFallbackPolicy::AllowNativeCpuFallback);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CountConvFallback);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
            &CountConvHostSync);
        grad_input = layer.Backward(DeviceOnlyOnes(output.Shape()));
    }
    CHECK(conv_fallback_count == 1);
    CHECK(conv_host_sync_count >= 1);
    CHECK(saw_conv_cpu_path);
    CHECK_FALSE(last_conv_fallback.fallback_forbidden);
    CheckValues(grad_input, {4, 1, 1}, {1.0f, 3.0f, 3.0f, 2.0f});
}

TEST_CASE("Conv1D strict policy rejects forced fallback before host sync",
          "[conv][conv1d][arrayfire][fallback][policy]") {
    constexpr const char* force_name =
        "CYXWIZ_TEST_FORCE_ARRAYFIRE_FALLBACK";
    const cyxwiz::Tensor input = DeviceOnlyTensor(
        {4, 1, 1}, std::vector<float>(4, 1.0f));

    cyxwiz::Conv1DLayer forward_layer(1, 1, 2, 1, 0, 1, true);
    ConfigureReferenceConvDeviceOnly(forward_layer);
    ResetConvObservations();
    {
        const ScopedEnvVar force(force_name, "Conv1DLayer::Forward");
        const cyxwiz::ScopedArrayFireFallbackPolicy strict(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CountConvFallback);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
            &CountConvHostSync);
        CHECK_THROWS_AS(forward_layer.Forward(input), std::runtime_error);
    }
    CHECK(conv_fallback_count == 1);
    CHECK(conv_host_sync_count == 0);
    CHECK(last_conv_fallback.operation_name == "Conv1DLayer::Forward");
    CHECK(last_conv_fallback.fallback_forbidden);

    cyxwiz::Conv1DLayer backward_layer(1, 1, 2, 1, 0, 1, true);
    ConfigureReferenceConvDeviceOnly(backward_layer);
    const cyxwiz::Tensor output = backward_layer.Forward(input);
    ResetConvObservations();
    {
        const ScopedEnvVar force(force_name, "Conv1DLayer::Backward");
        const cyxwiz::ScopedArrayFireFallbackPolicy strict(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CountConvFallback);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
            &CountConvHostSync);
        CHECK_THROWS_AS(
            backward_layer.Backward(DeviceOnlyOnes(output.Shape())),
            std::runtime_error);
    }
    CHECK(conv_fallback_count == 1);
    CHECK(conv_host_sync_count == 0);
    CHECK(last_conv_fallback.operation_name == "Conv1DLayer::Backward");
    CHECK(last_conv_fallback.fallback_forbidden);
}

#endif
#endif
