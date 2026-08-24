#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "algorithms/arrayfire_backend_utils.h"

#include <cyxwiz/layers/dropout.h>
#include <cyxwiz/memory_manager.h>
#include <cyxwiz/sequential.h>
#include <cyxwiz/tensor.h>

#include <nlohmann/json.hpp>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;

json LoadDropoutFixture() {
    std::ifstream stream(CYXWIZ_TRAINING_CORE_FIXTURE_PATH);
    REQUIRE(stream.is_open());
    json fixture;
    stream >> fixture;
    REQUIRE(fixture.value("schema_version", 0) == 1);
    REQUIRE(fixture.at("oracle").value("name", "") == "PyTorch");
    REQUIRE(fixture.at("oracle").value("version", "") == "2.10.0+cpu");
    return fixture.at("cases").at("dropout_semantics_f32");
}

std::vector<size_t> ShapeFromFixture(const json& value) {
    return value.at("shape").get<std::vector<size_t>>();
}

cyxwiz::Tensor TensorFromFixture(const json& value) {
    const auto shape = ShapeFromFixture(value);
    const auto values = value.at("values").get<std::vector<float>>();
    return cyxwiz::Tensor(shape, values.data(), cyxwiz::DataType::Float32);
}

void CheckTensorExact(const cyxwiz::Tensor& actual, const json& expected) {
    REQUIRE(actual.Shape() == ShapeFromFixture(expected));
    REQUIRE(actual.GetDataType() == cyxwiz::DataType::Float32);
    const auto values = expected.at("values").get<std::vector<float>>();
    REQUIRE(actual.NumElements() == values.size());
    const float* data = actual.ReadData<float>();
    for (size_t index = 0; index < values.size(); ++index) {
        CHECK(data[index] == values[index]);
    }
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

#ifdef CYXWIZ_HAS_ARRAYFIRE
size_t dropout_host_sync_count = 0;
size_t dropout_fallback_count = 0;

void CountHostSync(const cyxwiz::ArrayFireHostSyncEvent&) {
    ++dropout_host_sync_count;
}

void CountFallback(const cyxwiz::ArrayFireNativeCpuFallbackEvent&) {
    ++dropout_fallback_count;
}

class ScopedArrayFireDeviceRestore {
public:
    ScopedArrayFireDeviceRestore()
        : backend_(af::getActiveBackend()), device_(af::getDevice()) {}

    ~ScopedArrayFireDeviceRestore() {
        try {
            af::setBackend(backend_);
            af::setDevice(device_);
        } catch (...) {
        }
    }

private:
    af::Backend backend_;
    int device_;
};

template <typename Dropout>
void CheckActiveDeviceResidency(Dropout& dropout) {
    cyxwiz::Tensor input = cyxwiz::Tensor::FromArrayRowMajor2D(
        af::constant(1.0f, 128, 64, f32));
    cyxwiz::Tensor grad_output = cyxwiz::Tensor::FromArrayRowMajor2D(
        af::constant(1.0f, 128, 64, f32));

    dropout_host_sync_count = 0;
    dropout_fallback_count = 0;
    const size_t host_bytes_before = cyxwiz::MemoryManager::GetAllocatedBytes();
    cyxwiz::Tensor output;
    cyxwiz::Tensor grad_input;
    {
        const cyxwiz::ScopedArrayFireFallbackPolicy strict(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
            &CountHostSync);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CountFallback);
        af::setSeed(3917);
        output = dropout.Forward(input);
        grad_input = dropout.Backward(grad_output);
        output.GetArrayRowMajor2D().eval();
        grad_input.GetArrayRowMajor2D().eval();
        af::sync();
    }

    CHECK(output.Shape() == input.Shape());
    CHECK(grad_input.Shape() == output.Shape());
    CHECK(dropout_host_sync_count == 0);
    CHECK(dropout_fallback_count == 0);
    CHECK(cyxwiz::MemoryManager::GetAllocatedBytes() == host_bytes_before);

    const float* output_data = output.ReadData<float>();
    const float* grad_data = grad_input.ReadData<float>();
    size_t unexpected_values = 0;
    size_t backward_mismatches = 0;
    for (size_t index = 0; index < output.NumElements(); ++index) {
        unexpected_values +=
            output_data[index] == 0.0f || output_data[index] == 2.0f ? 0 : 1;
        backward_mismatches += grad_data[index] == output_data[index] ? 0 : 1;
    }
    CHECK(unexpected_values == 0);
    CHECK(backward_mismatches == 0);
}
#endif

} // namespace

TEST_CASE("Dropout eval and boundary probabilities match PyTorch",
          "[dropout][pytorch][contract]") {
    const json fixture = LoadDropoutFixture();
    const cyxwiz::Tensor input = TensorFromFixture(fixture.at("input"));
    const cyxwiz::Tensor grad_output =
        TensorFromFixture(fixture.at("grad_output"));

    const auto check_eval = [&](auto& dropout) {
        dropout.SetTraining(false);
        CheckTensorExact(
            dropout.Forward(input), fixture.at("eval_expected").at("output"));
        CheckTensorExact(
            dropout.Backward(grad_output),
            fixture.at("eval_expected").at("grad_input"));
    };
    cyxwiz::DropoutLayer eval_layer(0.5f);
    check_eval(eval_layer);
    cyxwiz::DropoutModule eval_module(0.5f);
    check_eval(eval_module);

    for (const auto& boundary : fixture.at("boundary_cases")) {
        const float p = boundary.at("probability").get<float>();
        DYNAMIC_SECTION("p=" << p << " DropoutLayer") {
            cyxwiz::DropoutLayer dropout(p);
            CheckTensorExact(
                dropout.Forward(input), boundary.at("expected").at("output"));
            CheckTensorExact(
                dropout.Backward(grad_output),
                boundary.at("expected").at("grad_input"));
        }
        DYNAMIC_SECTION("p=" << p << " DropoutModule") {
            cyxwiz::DropoutModule dropout(p);
            CheckTensorExact(
                dropout.Forward(input), boundary.at("expected").at("output"));
            CheckTensorExact(
                dropout.Backward(grad_output),
                boundary.at("expected").at("grad_input"));
        }
    }

    CHECK_THROWS_AS(cyxwiz::DropoutLayer(-0.01f), std::invalid_argument);
    CHECK_THROWS_AS(cyxwiz::DropoutLayer(1.01f), std::invalid_argument);
    CHECK_THROWS_AS(cyxwiz::DropoutModule(-0.01f), std::invalid_argument);
    CHECK_THROWS_AS(cyxwiz::DropoutModule(1.01f), std::invalid_argument);
}

TEST_CASE("Dropout backward follows the exact forward mask contract",
          "[dropout][backward][contract]") {
    std::vector<float> input_values(64, 1.0f);
    std::vector<float> grad_values(64);
    for (size_t index = 0; index < grad_values.size(); ++index) {
        grad_values[index] = static_cast<float>(index + 1) * 0.125f;
    }
    const cyxwiz::Tensor input(
        {8, 8}, input_values.data(), cyxwiz::DataType::Float32);
    const cyxwiz::Tensor grad_output(
        {8, 8}, grad_values.data(), cyxwiz::DataType::Float32);

    const auto check = [&](auto& dropout) {
        CHECK_THROWS_AS(dropout.Backward(grad_output), std::logic_error);
#ifdef CYXWIZ_HAS_ARRAYFIRE
        af::setSeed(3923);
#endif
        const cyxwiz::Tensor output = dropout.Forward(input);
        dropout.SetTraining(false);
        const cyxwiz::Tensor grad_input = dropout.Backward(grad_output);
        const float* output_data = output.ReadData<float>();
        const float* grad_data = grad_input.ReadData<float>();
        size_t backward_mismatches = 0;
        for (size_t index = 0; index < input.NumElements(); ++index) {
            const float expected = output_data[index] == 0.0f
                ? 0.0f
                : grad_values[index] / 0.75f;
            backward_mismatches +=
                std::abs(grad_data[index] - expected) <= 1.0e-6f ? 0 : 1;
        }
        CHECK(backward_mismatches == 0);

        const std::vector<float> wrong_shape_values(64, 1.0f);
        const cyxwiz::Tensor wrong_shape(
            {4, 16}, wrong_shape_values.data(), cyxwiz::DataType::Float32);
        CHECK_THROWS_AS(dropout.Backward(wrong_shape), std::runtime_error);
        const cyxwiz::Tensor wrong_dtype({8, 8}, cyxwiz::DataType::Float64);
        CHECK_THROWS_AS(dropout.Backward(wrong_dtype), std::runtime_error);
    };

    cyxwiz::DropoutLayer layer(0.25f);
    check(layer);
    cyxwiz::DropoutModule module(0.25f);
    check(module);

    const cyxwiz::Tensor unsupported({8, 8}, cyxwiz::DataType::Float64);
    cyxwiz::DropoutLayer unsupported_layer(0.25f);
    CHECK_THROWS_AS(
        unsupported_layer.Forward(unsupported), std::runtime_error);
    cyxwiz::DropoutModule unsupported_module(0.25f);
    CHECK_THROWS_AS(
        unsupported_module.Forward(unsupported), std::runtime_error);
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
TEST_CASE("Dropout distribution matches PyTorch and backend seed replays",
          "[dropout][pytorch][random][replay]") {
    const ScopedArrayFireDeviceRestore restore;
    af::setBackend(AF_BACKEND_CPU);
    af::setDevice(0);
    const json fixture = LoadDropoutFixture();

    size_t case_index = 0;
    for (const auto& test_case : fixture.at("distribution_cases")) {
        const float p = test_case.at("probability").get<float>();
        const size_t count = test_case.at("sample_count").get<size_t>();
        const float scale =
            test_case.at("expected_keep_scale").get<float>();
        const unsigned long long seed = 9100 + case_index++;
        DYNAMIC_SECTION("p=" << p) {
            cyxwiz::Tensor input = cyxwiz::Tensor::FromSemanticArray(
                af::constant(1.0f, static_cast<dim_t>(count), f32), {count});
            cyxwiz::Tensor grad_output = cyxwiz::Tensor::FromSemanticArray(
                af::constant(1.0f, static_cast<dim_t>(count), f32), {count});

            af::setSeed(seed);
            cyxwiz::DropoutLayer dropout(p);
            const cyxwiz::Tensor output = dropout.Forward(input);
            const cyxwiz::Tensor grad_input = dropout.Backward(grad_output);
            const float* output_data = output.ReadData<float>();
            const float* grad_data = grad_input.ReadData<float>();

            size_t zeros = 0;
            size_t unexpected_values = 0;
            size_t backward_mismatches = 0;
            double sum = 0.0;
            double squared_sum = 0.0;
            for (size_t index = 0; index < count; ++index) {
                unexpected_values +=
                    output_data[index] == 0.0f ||
                            std::abs(output_data[index] - scale) <= 1.0e-5f
                        ? 0
                        : 1;
                backward_mismatches +=
                    grad_data[index] == output_data[index] ? 0 : 1;
                zeros += output_data[index] == 0.0f ? 1 : 0;
                sum += output_data[index];
                squared_sum += output_data[index] * output_data[index];
            }
            CHECK(unexpected_values == 0);
            CHECK(backward_mismatches == 0);
            const double mean = sum / static_cast<double>(count);
            const double variance =
                squared_sum / static_cast<double>(count) - mean * mean;
            const double zero_fraction =
                static_cast<double>(zeros) / static_cast<double>(count);
            const auto& tolerance = test_case.at("tolerance");
            CHECK(zero_fraction == Catch::Approx(
                test_case.at("theoretical").at("zero_fraction").get<double>())
                .margin(tolerance.at("zero_fraction").get<double>()));
            CHECK(mean == Catch::Approx(
                test_case.at("pytorch_observed").at("mean").get<double>())
                .margin(tolerance.at("mean").get<double>()));
            CHECK(variance == Catch::Approx(
                test_case.at("pytorch_observed").at("variance").get<double>())
                .margin(tolerance.at("variance").get<double>()));
            CHECK(test_case.at("pytorch_observed")
                      .at("backward_mask_mismatch_count").get<int>() == 0);

            af::setSeed(seed);
            cyxwiz::DropoutLayer replay(p);
            const cyxwiz::Tensor replay_output = replay.Forward(input);
            const float* replay_data = replay_output.ReadData<float>();
            size_t replay_mismatches = 0;
            for (size_t index = 0; index < count; ++index) {
                replay_mismatches +=
                    replay_data[index] == output_data[index] ? 0 : 1;
            }
            CHECK(replay_mismatches == 0);
        }
    }
}

TEST_CASE("Dropout remains resident on every installed supported route",
          "[dropout][arrayfire][residency][accelerator]") {
    const ScopedArrayFireDeviceRestore restore;
    const int available = af::getAvailableBackends();

    for (const af::Backend backend :
         {AF_BACKEND_CPU, AF_BACKEND_CUDA, AF_BACKEND_OPENCL}) {
        if ((available & static_cast<int>(backend)) == 0) {
            continue;
        }
        af::setBackend(backend);
        const int count = af::getDeviceCount();
        for (int device = 0; device < count; ++device) {
            DYNAMIC_SECTION(
                "layer backend=" << static_cast<int>(backend)
                                  << " device=" << device) {
                af::setDevice(device);
                cyxwiz::DropoutLayer dropout(0.5f);
                CheckActiveDeviceResidency(dropout);
            }
            DYNAMIC_SECTION(
                "module backend=" << static_cast<int>(backend)
                                   << " device=" << device) {
                af::setDevice(device);
                cyxwiz::DropoutModule dropout(0.5f);
                CheckActiveDeviceResidency(dropout);
            }
        }
    }
}

#ifndef NDEBUG
TEST_CASE("Dropout fallback is observable and strict policy rejects it",
          "[dropout][arrayfire][fallback]") {
    constexpr const char* force_fallback =
        "CYXWIZ_TEST_FORCE_ARRAYFIRE_FALLBACK";
    const std::vector<float> values(64, 1.0f);
    const cyxwiz::Tensor input(
        {8, 8}, values.data(), cyxwiz::DataType::Float32);

    cyxwiz::DropoutLayer compatible(0.5f);
    dropout_fallback_count = 0;
    {
        const ScopedEnvVar force(force_fallback, "DropoutLayer::Forward");
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver observer(
            &CountFallback);
        const cyxwiz::ScopedArrayFireFallbackPolicy policy(
            cyxwiz::ArrayFireFallbackPolicy::AllowNativeCpuFallback);
        const cyxwiz::Tensor output = compatible.Forward(input);
        CHECK(output.Shape() == input.Shape());
    }
    CHECK(dropout_fallback_count == 1);
    cyxwiz::Tensor compatible_output;
    {
        af::setSeed(3941);
        cyxwiz::DropoutLayer compatible_backward(0.5f);
        compatible_output = compatible_backward.Forward(input);
        const ScopedEnvVar force(force_fallback, "DropoutLayer::Backward");
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver observer(
            &CountFallback);
        const cyxwiz::ScopedArrayFireFallbackPolicy policy(
            cyxwiz::ArrayFireFallbackPolicy::AllowNativeCpuFallback);
        const cyxwiz::Tensor grad_input = compatible_backward.Backward(input);
        const float* output_data = compatible_output.ReadData<float>();
        const float* grad_data = grad_input.ReadData<float>();
        size_t mismatches = 0;
        for (size_t index = 0; index < input.NumElements(); ++index) {
            mismatches += output_data[index] == grad_data[index] ? 0 : 1;
        }
        CHECK(mismatches == 0);
    }
    CHECK(dropout_fallback_count == 2);

    cyxwiz::DropoutLayer strict(0.5f);
    {
        const ScopedEnvVar force(force_fallback, "DropoutLayer::Forward");
        const cyxwiz::ScopedArrayFireFallbackPolicy policy(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        CHECK_THROWS_AS(strict.Forward(input), std::runtime_error);
    }

    cyxwiz::DropoutLayer strict_backward(0.5f);
    (void)strict_backward.Forward(input);
    {
        const ScopedEnvVar force(force_fallback, "DropoutLayer::Backward");
        const cyxwiz::ScopedArrayFireFallbackPolicy policy(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        CHECK_THROWS_AS(strict_backward.Backward(input), std::runtime_error);
    }
}
#endif
#endif
