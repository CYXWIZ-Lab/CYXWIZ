#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "algorithms/arrayfire_backend_utils.h"

#include <cyxwiz/layers/pooling.h>
#include <cyxwiz/tensor.h>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

#include <cstdlib>
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

#ifdef CYXWIZ_HAS_ARRAYFIRE

size_t pooling_host_sync_count = 0;
size_t pooling_fallback_count = 0;
bool saw_pooling_cpu_path = false;
cyxwiz::ArrayFireNativeCpuFallbackEvent last_pooling_fallback;

void CountPoolingHostSync(const cyxwiz::ArrayFireHostSyncEvent& event) {
    ++pooling_host_sync_count;
    saw_pooling_cpu_path |=
        event.attribution_category == "layer_cpu_path";
}

void CountPoolingFallback(
    const cyxwiz::ArrayFireNativeCpuFallbackEvent& event) {
    ++pooling_fallback_count;
    last_pooling_fallback = event;
}

void ResetPoolingObservations() {
    pooling_host_sync_count = 0;
    pooling_fallback_count = 0;
    saw_pooling_cpu_path = false;
    last_pooling_fallback = {};
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

template <typename PoolLayer>
void CheckStrictForcedFallback(const char* forward_operation,
                               const char* backward_operation) {
    constexpr const char* force_fallback =
        "CYXWIZ_TEST_FORCE_ARRAYFIRE_FALLBACK";
    const std::vector<size_t> input_shape{2, 2, 1, 1};
    const std::vector<float> input_values{1.0f, 2.0f, 3.0f, 4.0f};

    PoolLayer forward_pool(2, 1, 1);
    const cyxwiz::Tensor forward_input =
        DeviceOnlyTensor(input_shape, input_values);
    ResetPoolingObservations();
    {
        const ScopedEnvVar force(force_fallback, forward_operation);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CountPoolingFallback);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
            &CountPoolingHostSync);
        const cyxwiz::ScopedArrayFireFallbackPolicy strict(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        CHECK_THROWS_AS(
            forward_pool.Forward(forward_input), std::runtime_error);
    }
    CHECK(pooling_fallback_count == 1);
    CHECK(pooling_host_sync_count == 0);
    CHECK(last_pooling_fallback.operation_name == forward_operation);
    CHECK(last_pooling_fallback.fallback_forbidden);

    PoolLayer backward_pool(2, 1, 1);
    const cyxwiz::Tensor output = backward_pool.Forward(forward_input);
    const cyxwiz::Tensor grad_output = DeviceOnlyOnes(output.Shape());
    ResetPoolingObservations();
    {
        const ScopedEnvVar force(force_fallback, backward_operation);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CountPoolingFallback);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
            &CountPoolingHostSync);
        const cyxwiz::ScopedArrayFireFallbackPolicy strict(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        CHECK_THROWS_AS(
            backward_pool.Backward(grad_output), std::runtime_error);
    }
    CHECK(pooling_fallback_count == 1);
    CHECK(pooling_host_sync_count == 0);
    CHECK(last_pooling_fallback.operation_name == backward_operation);
    CHECK(last_pooling_fallback.fallback_forbidden);
}

template <typename PoolLayer>
void CheckCompatibleForcedFallback(
    const char* forward_operation,
    const char* backward_operation,
    const std::vector<float>& expected_output,
    const std::vector<float>& expected_grad_input) {
    constexpr const char* force_fallback =
        "CYXWIZ_TEST_FORCE_ARRAYFIRE_FALLBACK";
    const std::vector<size_t> input_shape{2, 2, 1, 1};
    const std::vector<size_t> output_shape{3, 3, 1, 1};
    const std::vector<float> input_values{1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> grad_values{
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
    };

    PoolLayer forward_pool(2, 1, 1);
    const cyxwiz::Tensor forward_input =
        DeviceOnlyTensor(input_shape, input_values);
    cyxwiz::Tensor output;
    ResetPoolingObservations();
    {
        const ScopedEnvVar force(force_fallback, forward_operation);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CountPoolingFallback);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
            &CountPoolingHostSync);
        const cyxwiz::ScopedArrayFireFallbackPolicy compatible(
            cyxwiz::ArrayFireFallbackPolicy::AllowNativeCpuFallback);
        output = forward_pool.Forward(forward_input);
    }
    CHECK(pooling_fallback_count == 1);
    CHECK(pooling_host_sync_count >= 1);
    CHECK(saw_pooling_cpu_path);
    CHECK(last_pooling_fallback.operation_name == forward_operation);
    CHECK_FALSE(last_pooling_fallback.fallback_forbidden);
    CheckValues(output, output_shape, expected_output);

    PoolLayer backward_pool(2, 1, 1);
    (void)backward_pool.Forward(forward_input);
    const cyxwiz::Tensor grad_output =
        DeviceOnlyTensor(output_shape, grad_values);
    cyxwiz::Tensor grad_input;
    ResetPoolingObservations();
    {
        const ScopedEnvVar force(force_fallback, backward_operation);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CountPoolingFallback);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
            &CountPoolingHostSync);
        const cyxwiz::ScopedArrayFireFallbackPolicy compatible(
            cyxwiz::ArrayFireFallbackPolicy::AllowNativeCpuFallback);
        grad_input = backward_pool.Backward(grad_output);
    }
    CHECK(pooling_fallback_count == 1);
    CHECK(pooling_host_sync_count >= 1);
    CHECK(saw_pooling_cpu_path);
    CHECK(last_pooling_fallback.operation_name == backward_operation);
    CHECK_FALSE(last_pooling_fallback.fallback_forbidden);
    CheckValues(grad_input, input_shape, expected_grad_input);
}

void CheckGlobalCompatibleForcedFallback() {
    constexpr const char* force_fallback =
        "CYXWIZ_TEST_FORCE_ARRAYFIRE_FALLBACK";
    constexpr const char* forward_operation =
        "GlobalAvgPool2DLayer::Forward";
    constexpr const char* backward_operation =
        "GlobalAvgPool2DLayer::Backward";
    const std::vector<size_t> input_shape{2, 2, 2, 2};
    const std::vector<size_t> output_shape{2, 2};
    const std::vector<float> input_values{
        1.0f, 10.0f, -1.0f, 5.0f,
        2.0f, 20.0f, -2.0f, 1.0f,
        3.0f, 30.0f, -3.0f, 2.0f,
        4.0f, 40.0f, -4.0f, 3.0f,
    };
    const std::vector<float> grad_values{1.0f, 2.0f, 3.0f, 4.0f};

    const cyxwiz::Tensor input =
        DeviceOnlyTensor(input_shape, input_values);
    cyxwiz::GlobalAvgPool2DLayer forward_pool;
    cyxwiz::Tensor output;
    ResetPoolingObservations();
    {
        const ScopedEnvVar force(force_fallback, forward_operation);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CountPoolingFallback);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
            &CountPoolingHostSync);
        const cyxwiz::ScopedArrayFireFallbackPolicy compatible(
            cyxwiz::ArrayFireFallbackPolicy::AllowNativeCpuFallback);
        output = forward_pool.Forward(input);
    }
    CHECK(pooling_fallback_count == 1);
    CHECK(pooling_host_sync_count >= 1);
    CHECK(saw_pooling_cpu_path);
    CHECK(last_pooling_fallback.operation_name == forward_operation);
    CHECK_FALSE(last_pooling_fallback.fallback_forbidden);
    CheckValues(output, output_shape, {2.5f, 25.0f, -2.5f, 2.75f});

    cyxwiz::GlobalAvgPool2DLayer backward_pool;
    (void)backward_pool.Forward(input);
    const cyxwiz::Tensor grad_output =
        DeviceOnlyTensor(output_shape, grad_values);
    cyxwiz::Tensor grad_input;
    ResetPoolingObservations();
    {
        const ScopedEnvVar force(force_fallback, backward_operation);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CountPoolingFallback);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
            &CountPoolingHostSync);
        const cyxwiz::ScopedArrayFireFallbackPolicy compatible(
            cyxwiz::ArrayFireFallbackPolicy::AllowNativeCpuFallback);
        grad_input = backward_pool.Backward(grad_output);
    }
    CHECK(pooling_fallback_count == 1);
    CHECK(pooling_host_sync_count >= 1);
    CHECK(saw_pooling_cpu_path);
    CHECK(last_pooling_fallback.operation_name == backward_operation);
    CHECK_FALSE(last_pooling_fallback.fallback_forbidden);
    CheckValues(
        grad_input,
        input_shape,
        {
            0.25f, 0.5f, 0.75f, 1.0f,
            0.25f, 0.5f, 0.75f, 1.0f,
            0.25f, 0.5f, 0.75f, 1.0f,
            0.25f, 0.5f, 0.75f, 1.0f,
        });
}

void CheckGlobalStrictForcedFallback() {
    constexpr const char* force_fallback =
        "CYXWIZ_TEST_FORCE_ARRAYFIRE_FALLBACK";
    constexpr const char* forward_operation =
        "GlobalAvgPool2DLayer::Forward";
    constexpr const char* backward_operation =
        "GlobalAvgPool2DLayer::Backward";
    const std::vector<size_t> input_shape{2, 2, 1, 1};
    const std::vector<float> input_values{1.0f, 2.0f, 3.0f, 4.0f};
    const cyxwiz::Tensor input =
        DeviceOnlyTensor(input_shape, input_values);

    cyxwiz::GlobalAvgPool2DLayer forward_pool;
    ResetPoolingObservations();
    {
        const ScopedEnvVar force(force_fallback, forward_operation);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CountPoolingFallback);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
            &CountPoolingHostSync);
        const cyxwiz::ScopedArrayFireFallbackPolicy strict(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        CHECK_THROWS_AS(forward_pool.Forward(input), std::runtime_error);
    }
    CHECK(pooling_fallback_count == 1);
    CHECK(pooling_host_sync_count == 0);
    CHECK(last_pooling_fallback.operation_name == forward_operation);
    CHECK(last_pooling_fallback.fallback_forbidden);

    cyxwiz::GlobalAvgPool2DLayer backward_pool;
    const cyxwiz::Tensor output = backward_pool.Forward(input);
    const cyxwiz::Tensor grad_output = DeviceOnlyOnes(output.Shape());
    ResetPoolingObservations();
    {
        const ScopedEnvVar force(force_fallback, backward_operation);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CountPoolingFallback);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
            &CountPoolingHostSync);
        const cyxwiz::ScopedArrayFireFallbackPolicy strict(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        CHECK_THROWS_AS(
            backward_pool.Backward(grad_output), std::runtime_error);
    }
    CHECK(pooling_fallback_count == 1);
    CHECK(pooling_host_sync_count == 0);
    CHECK(last_pooling_fallback.operation_name == backward_operation);
    CHECK(last_pooling_fallback.fallback_forbidden);
}

#endif

} // namespace

TEST_CASE("MaxPool2D honors negative padding and overlapping gradients",
          "[pool][maxpool][correctness]") {
    const std::vector<float> input_values{
        -1.0f, -2.0f,
        -3.0f, -4.0f,
    };
    const cyxwiz::Tensor input(
        {2, 2, 1, 1}, input_values.data(), cyxwiz::DataType::Float32);
    cyxwiz::MaxPool2DLayer pool(2, 1, 1);

    const cyxwiz::Tensor output = pool.Forward(input);
    CheckValues(
        output,
        {3, 3, 1, 1},
        {
            -1.0f, -1.0f, -2.0f,
            -1.0f, -1.0f, -2.0f,
            -3.0f, -3.0f, -4.0f,
        });

    const std::vector<float> grad_values(9, 1.0f);
    const cyxwiz::Tensor grad_output(
        {3, 3, 1, 1}, grad_values.data(), cyxwiz::DataType::Float32);
    const cyxwiz::Tensor grad_input = pool.Backward(grad_output);
    CheckValues(grad_input, {2, 2, 1, 1}, {4.0f, 2.0f, 2.0f, 1.0f});
}

TEST_CASE("MaxPool2D uses row-major first-index tie behavior",
          "[pool][maxpool][correctness]") {
    const std::vector<float> input_values{0.0f, 5.0f, 5.0f, 0.0f};
    const cyxwiz::Tensor input(
        {2, 2, 1, 1}, input_values.data(), cyxwiz::DataType::Float32);
    cyxwiz::MaxPool2DLayer pool(2, 2, 0);

    CheckValues(pool.Forward(input), {1, 1, 1, 1}, {5.0f});
    const std::vector<float> grad_values{1.0f};
    const cyxwiz::Tensor grad_output(
        {1, 1, 1, 1}, grad_values.data(), cyxwiz::DataType::Float32);
    CheckValues(
        pool.Backward(grad_output),
        {2, 2, 1, 1},
        {0.0f, 1.0f, 0.0f, 0.0f});
}

TEST_CASE("MaxPool2D preserves [H, W, C, N] batch/channel layout",
          "[pool][maxpool][layout]") {
    const std::vector<float> input_values{
        1.0f, 10.0f, -1.0f, 5.0f,
        2.0f, 20.0f, -2.0f, 1.0f,
        3.0f, 30.0f, -3.0f, 2.0f,
        4.0f, 40.0f, -4.0f, 3.0f,
    };
    const cyxwiz::Tensor input(
        {2, 2, 2, 2}, input_values.data(), cyxwiz::DataType::Float32);
    cyxwiz::MaxPool2DLayer pool(2, 2, 0);
    CheckValues(
        pool.Forward(input),
        {1, 1, 2, 2},
        {4.0f, 40.0f, -1.0f, 5.0f});

    const std::vector<float> grad_values{1.0f, 2.0f, 3.0f, 4.0f};
    const cyxwiz::Tensor grad_output(
        {1, 1, 2, 2}, grad_values.data(), cyxwiz::DataType::Float32);
    CheckValues(
        pool.Backward(grad_output),
        {2, 2, 2, 2},
        {
            0.0f, 0.0f, 3.0f, 4.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
            1.0f, 2.0f, 0.0f, 0.0f,
        });
}

TEST_CASE("AvgPool2D counts padded zeros and accumulates overlap",
          "[pool][avgpool][correctness]") {
    const std::vector<float> input_values{1.0f, 2.0f, 3.0f, 4.0f};
    const cyxwiz::Tensor input(
        {2, 2, 1, 1}, input_values.data(), cyxwiz::DataType::Float32);
    cyxwiz::AvgPool2DLayer pool(2, 1, 1);

    CheckValues(
        pool.Forward(input),
        {3, 3, 1, 1},
        {
            0.25f, 0.75f, 0.5f,
            1.0f, 2.5f, 1.5f,
            0.75f, 1.75f, 1.0f,
        });

    const std::vector<float> grad_values{
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
    };
    const cyxwiz::Tensor grad_output(
        {3, 3, 1, 1}, grad_values.data(), cyxwiz::DataType::Float32);
    CheckValues(pool.Backward(grad_output), {2, 2, 1, 1}, {3.0f, 4.0f, 6.0f, 7.0f});
}

TEST_CASE("GlobalAvgPool2D preserves channel and batch layout",
          "[pool][global_avg_pool][correctness][layout]") {
    const std::vector<float> input_values{
        1.0f, 10.0f, -1.0f, 5.0f,
        2.0f, 20.0f, -2.0f, 1.0f,
        3.0f, 30.0f, -3.0f, 2.0f,
        4.0f, 40.0f, -4.0f, 3.0f,
    };
    const cyxwiz::Tensor input(
        {2, 2, 2, 2}, input_values.data(), cyxwiz::DataType::Float32);
    cyxwiz::GlobalAvgPool2DLayer pool;
    CheckValues(
        pool.Forward(input),
        {2, 2},
        {2.5f, 25.0f, -2.5f, 2.75f});

    const std::vector<float> grad_values{1.0f, 2.0f, 3.0f, 4.0f};
    const cyxwiz::Tensor grad_output(
        {2, 2}, grad_values.data(), cyxwiz::DataType::Float32);
    CheckValues(
        pool.Backward(grad_output),
        {2, 2, 2, 2},
        {
            0.25f, 0.5f, 0.75f, 1.0f,
            0.25f, 0.5f, 0.75f, 1.0f,
            0.25f, 0.5f, 0.75f, 1.0f,
            0.25f, 0.5f, 0.75f, 1.0f,
        });
}

TEST_CASE("Pooling validates forward state, exact gradient shape, and padding",
          "[pool][validation]") {
    const std::vector<float> input_values{1.0f, 2.0f, 3.0f, 4.0f};
    const cyxwiz::Tensor input(
        {2, 2, 1, 1}, input_values.data(), cyxwiz::DataType::Float32);
    const std::vector<float> one_value{1.0f};
    const cyxwiz::Tensor one_grad(
        {1, 1, 1, 1}, one_value.data(), cyxwiz::DataType::Float32);

    cyxwiz::MaxPool2DLayer max_pool(2, 2, 0);
    cyxwiz::AvgPool2DLayer avg_pool(2, 2, 0);
    CHECK_THROWS_AS(max_pool.Backward(one_grad), std::logic_error);
    CHECK_THROWS_AS(avg_pool.Backward(one_grad), std::logic_error);
    cyxwiz::GlobalAvgPool2DLayer global_pool;
    CHECK_THROWS_AS(global_pool.Backward(one_grad), std::logic_error);

    (void)max_pool.Forward(input);
    (void)avg_pool.Forward(input);
    const std::vector<float> malformed_values(2, 1.0f);
    const cyxwiz::Tensor malformed_grad(
        {1, 2, 1, 1}, malformed_values.data(), cyxwiz::DataType::Float32);
    CHECK_THROWS_AS(max_pool.Backward(malformed_grad), std::runtime_error);
    CHECK_THROWS_AS(avg_pool.Backward(malformed_grad), std::runtime_error);

    CHECK_THROWS_AS(cyxwiz::MaxPool2DLayer(2, 1, 2), std::invalid_argument);
    CHECK_THROWS_AS(cyxwiz::AvgPool2DLayer(2, 1, 2), std::invalid_argument);
}

#ifdef CYXWIZ_HAS_ARRAYFIRE

TEST_CASE("Pooling stays ArrayFire-resident under strict policy",
          "[pool][arrayfire][residency]") {
    const std::vector<size_t> input_shape{4, 4, 2, 2};
    std::vector<float> input_values(64);
    for (size_t index = 0; index < input_values.size(); ++index) {
        input_values[index] = static_cast<float>(index) - 20.0f;
    }
    const cyxwiz::Tensor input = DeviceOnlyTensor(input_shape, input_values);

    SECTION("MaxPool2D") {
        cyxwiz::MaxPool2DLayer pool(2, 1, 1);
        ResetPoolingObservations();
        cyxwiz::Tensor output;
        cyxwiz::Tensor grad_input;
        {
            const cyxwiz::ScopedArrayFireFallbackPolicy strict(
                cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
            const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
                &CountPoolingFallback);
            const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
                &CountPoolingHostSync);
            output = pool.Forward(input);
            const cyxwiz::Tensor grad_output = DeviceOnlyOnes(output.Shape());
            grad_input = pool.Backward(grad_output);
            output.GetSemanticArray().eval();
            grad_input.GetSemanticArray().eval();
            af::sync();
        }
        CHECK(pooling_fallback_count == 0);
        CHECK(pooling_host_sync_count == 0);
        CHECK(output.Shape() == std::vector<size_t>{5, 5, 2, 2});
        CHECK(grad_input.Shape() == input_shape);
    }

    SECTION("AvgPool2D") {
        cyxwiz::AvgPool2DLayer pool(2, 1, 1);
        ResetPoolingObservations();
        cyxwiz::Tensor output;
        cyxwiz::Tensor grad_input;
        {
            const cyxwiz::ScopedArrayFireFallbackPolicy strict(
                cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
            const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
                &CountPoolingFallback);
            const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
                &CountPoolingHostSync);
            output = pool.Forward(input);
            const cyxwiz::Tensor grad_output = DeviceOnlyOnes(output.Shape());
            grad_input = pool.Backward(grad_output);
            output.GetSemanticArray().eval();
            grad_input.GetSemanticArray().eval();
            af::sync();
        }
        CHECK(pooling_fallback_count == 0);
        CHECK(pooling_host_sync_count == 0);
        CHECK(output.Shape() == std::vector<size_t>{5, 5, 2, 2});
        CHECK(grad_input.Shape() == input_shape);
    }

    SECTION("GlobalAvgPool2D") {
        cyxwiz::GlobalAvgPool2DLayer pool;
        ResetPoolingObservations();
        cyxwiz::Tensor output;
        cyxwiz::Tensor grad_input;
        {
            const cyxwiz::ScopedArrayFireFallbackPolicy strict(
                cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
            const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
                &CountPoolingFallback);
            const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
                &CountPoolingHostSync);
            output = pool.Forward(input);
            const cyxwiz::Tensor grad_output = DeviceOnlyOnes(output.Shape());
            grad_input = pool.Backward(grad_output);
            output.GetSemanticArray().eval();
            grad_input.GetSemanticArray().eval();
            af::sync();
        }
        CHECK(pooling_fallback_count == 0);
        CHECK(pooling_host_sync_count == 0);
        CHECK(output.Shape() == std::vector<size_t>{2, 2});
        CHECK(grad_input.Shape() == input_shape);
    }
}

#ifndef NDEBUG

TEST_CASE("Pooling forced fallback is compatible and attributed",
          "[pool][arrayfire][fallback]") {
    CheckCompatibleForcedFallback<cyxwiz::MaxPool2DLayer>(
        "MaxPool2DLayer::Forward",
        "MaxPool2DLayer::Backward",
        {
            1.0f, 2.0f, 2.0f,
            3.0f, 4.0f, 4.0f,
            3.0f, 4.0f, 4.0f,
        },
        {1.0f, 5.0f, 11.0f, 28.0f});
    CheckCompatibleForcedFallback<cyxwiz::AvgPool2DLayer>(
        "AvgPool2DLayer::Forward",
        "AvgPool2DLayer::Backward",
        {
            0.25f, 0.75f, 0.5f,
            1.0f, 2.5f, 1.5f,
            0.75f, 1.75f, 1.0f,
        },
        {3.0f, 4.0f, 6.0f, 7.0f});
    CheckGlobalCompatibleForcedFallback();
}

TEST_CASE("Pooling strict policy rejects forced fallback before host sync",
          "[pool][arrayfire][fallback][policy]") {
    CheckStrictForcedFallback<cyxwiz::MaxPool2DLayer>(
        "MaxPool2DLayer::Forward", "MaxPool2DLayer::Backward");
    CheckStrictForcedFallback<cyxwiz::AvgPool2DLayer>(
        "AvgPool2DLayer::Forward", "AvgPool2DLayer::Backward");
    CheckGlobalStrictForcedFallback();
}

#endif
#endif
