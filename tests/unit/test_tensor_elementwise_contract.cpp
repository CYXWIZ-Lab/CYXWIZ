#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "algorithms/arrayfire_backend_utils.h"

#include <cyxwiz/device.h>
#include <cyxwiz/tensor.h>

#include <nlohmann/json.hpp>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

#include <cmath>
#include <functional>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;

json LoadElementwiseFixture() {
    std::ifstream stream(CYXWIZ_TRAINING_CORE_FIXTURE_PATH);
    REQUIRE(stream.is_open());
    json fixture;
    stream >> fixture;
    REQUIRE(fixture.value("schema_version", 0) == 1);
    REQUIRE(fixture.at("oracle").value("name", "") == "PyTorch");
    REQUIRE(fixture.at("oracle").value("version", "").rfind("2.10.0", 0) == 0);
    return fixture.at("cases").at("tensor_elementwise");
}

cyxwiz::DataType ParseDataType(const std::string& value) {
    if (value == "float32") return cyxwiz::DataType::Float32;
    if (value == "float64") return cyxwiz::DataType::Float64;
    if (value == "int32") return cyxwiz::DataType::Int32;
    if (value == "int64") return cyxwiz::DataType::Int64;
    if (value == "uint8") return cyxwiz::DataType::UInt8;
    throw std::runtime_error("unsupported elementwise fixture dtype: " + value);
}

template <typename T>
cyxwiz::Tensor TypedTensorFromFixture(const json& value,
                                      cyxwiz::DataType dtype) {
    const auto shape = value.at("shape").get<std::vector<size_t>>();
    const auto values = value.at("values").get<std::vector<T>>();
    return cyxwiz::Tensor(shape, values.data(), dtype);
}

cyxwiz::Tensor TensorFromFixture(const json& value,
                                 cyxwiz::DataType dtype) {
    switch (dtype) {
        case cyxwiz::DataType::Float32:
            return TypedTensorFromFixture<float>(value, dtype);
        case cyxwiz::DataType::Float64:
            return TypedTensorFromFixture<double>(value, dtype);
        case cyxwiz::DataType::Int32:
            return TypedTensorFromFixture<int32_t>(value, dtype);
        case cyxwiz::DataType::Int64:
            return TypedTensorFromFixture<int64_t>(value, dtype);
        case cyxwiz::DataType::UInt8:
            return TypedTensorFromFixture<uint8_t>(value, dtype);
    }
    throw std::runtime_error("unsupported elementwise Tensor dtype");
}

cyxwiz::Tensor ApplyElementwise(const cyxwiz::Tensor& input,
                                const json& test_case) {
    const std::string operation = test_case.at("operation").get<std::string>();
    const std::string rhs_kind = test_case.at("rhs_kind").get<std::string>();
    if (rhs_kind == "tensor") {
        const auto rhs_dtype =
            ParseDataType(test_case.at("rhs_dtype").get<std::string>());
        const cyxwiz::Tensor rhs =
            TensorFromFixture(test_case.at("rhs"), rhs_dtype);
        if (operation == "add") return input + rhs;
        if (operation == "subtract") return input - rhs;
        if (operation == "multiply") return input * rhs;
        if (operation == "divide") return input / rhs;
        if (operation == "pow") return input.Pow(rhs);
    }
    if (rhs_kind == "scalar") {
        const float scalar = test_case.at("scalar").get<float>();
        if (operation == "add") return input + scalar;
        if (operation == "subtract") return input - scalar;
        if (operation == "multiply") return input * scalar;
        if (operation == "divide") return input / scalar;
        if (operation == "pow") return input.Pow(scalar);
    }
    if (operation == "sqrt") return input.Sqrt();
    if (operation == "exp") return input.Exp();
    if (operation == "log") return input.Log();
    if (operation == "sin") return input.Sin();
    if (operation == "cos") return input.Cos();
    if (operation == "abs") return input.Abs();
    if (operation == "sign") return input.Sign();
    if (operation == "clip") {
        return input.Clip(test_case.at("minimum").get<float>(),
                          test_case.at("maximum").get<float>());
    }
    if (operation == "negate") return -input;
    throw std::runtime_error("unsupported elementwise fixture operation: " + operation);
}

template <typename T>
void CheckExact(const cyxwiz::Tensor& actual, const json& expected) {
    const auto values = expected.at("values").get<std::vector<T>>();
    const T* output = actual.ReadData<T>();
    REQUIRE(actual.NumElements() == values.size());
    for (size_t index = 0; index < values.size(); ++index) {
        REQUIRE(output[index] == values[index]);
    }
}

template <typename T>
void CheckReal(const cyxwiz::Tensor& actual,
               const json& expected,
               double atol,
               double rtol) {
    const auto values = expected.at("values").get<std::vector<T>>();
    const T* output = actual.ReadData<T>();
    REQUIRE(actual.NumElements() == values.size());
    for (size_t index = 0; index < values.size(); ++index) {
        REQUIRE(static_cast<double>(output[index]) ==
                Catch::Approx(static_cast<double>(values[index]))
                    .margin(atol).epsilon(rtol));
    }
}

void CheckElementwise(const cyxwiz::Tensor& actual, const json& test_case) {
    const auto dtype =
        ParseDataType(test_case.at("output_dtype").get<std::string>());
    const auto& expected = test_case.at("expected");
    REQUIRE(actual.GetDataType() == dtype);
    REQUIRE(actual.Shape() == expected.at("shape").get<std::vector<size_t>>());
    const double atol = test_case.at("tolerance").at("atol").get<double>();
    const double rtol = test_case.at("tolerance").at("rtol").get<double>();
    switch (dtype) {
        case cyxwiz::DataType::Float32: CheckReal<float>(actual, expected, atol, rtol); break;
        case cyxwiz::DataType::Float64: CheckReal<double>(actual, expected, atol, rtol); break;
        case cyxwiz::DataType::Int32: CheckExact<int32_t>(actual, expected); break;
        case cyxwiz::DataType::Int64: CheckExact<int64_t>(actual, expected); break;
        case cyxwiz::DataType::UInt8: CheckExact<uint8_t>(actual, expected); break;
    }
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
size_t g_elementwise_host_sync_count = 0;
size_t g_elementwise_fallback_count = 0;

void CountElementwiseHostSync(const cyxwiz::ArrayFireHostSyncEvent&) {
    ++g_elementwise_host_sync_count;
}

void CountElementwiseFallback(const cyxwiz::ArrayFireNativeCpuFallbackEvent&) {
    ++g_elementwise_fallback_count;
}

cyxwiz::Tensor DeviceTensorFromFixture(const json& value,
                                       cyxwiz::DataType dtype) {
    const cyxwiz::Tensor host = TensorFromFixture(value, dtype);
    if (host.NumElements() == 0) {
        return host;
    }
    return cyxwiz::Tensor::FromSemanticArray(
        host.GetSemanticArray(), host.Shape());
}

void CheckFixtureOnActiveDevice(const json& fixture, bool supports_fp64) {
    for (const json& test_case : fixture) {
        const std::string name = test_case.at("name").get<std::string>();
        CAPTURE(name);
        const auto input_dtype =
            ParseDataType(test_case.at("input_dtype").get<std::string>());
        const auto output_dtype =
            ParseDataType(test_case.at("output_dtype").get<std::string>());
        if (!supports_fp64 &&
            (input_dtype == cyxwiz::DataType::Float64 ||
             output_dtype == cyxwiz::DataType::Float64 ||
             (test_case.contains("rhs_dtype") &&
              ParseDataType(test_case.at("rhs_dtype").get<std::string>()) ==
                  cyxwiz::DataType::Float64))) {
            continue;
        }

        cyxwiz::Tensor input =
            DeviceTensorFromFixture(test_case.at("input"), input_dtype);
        json active_case = test_case;
        if (test_case.at("rhs_kind") == "tensor") {
            const auto rhs_dtype =
                ParseDataType(test_case.at("rhs_dtype").get<std::string>());
            cyxwiz::Tensor rhs =
                DeviceTensorFromFixture(test_case.at("rhs"), rhs_dtype);
            // ApplyElementwise constructs from JSON, so execute tensor cases
            // directly to retain the resident right-hand input.
            const std::string operation =
                test_case.at("operation").get<std::string>();
            g_elementwise_host_sync_count = 0;
            g_elementwise_fallback_count = 0;
            cyxwiz::Tensor actual;
            {
                const cyxwiz::ScopedArrayFireFallbackPolicy strict(
                    cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
                const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
                    &CountElementwiseHostSync);
                const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
                    &CountElementwiseFallback);
                if (operation == "add") actual = input + rhs;
                else if (operation == "subtract") actual = input - rhs;
                else if (operation == "multiply") actual = input * rhs;
                else if (operation == "divide") actual = input / rhs;
                else actual = input.Pow(rhs);
                if (actual.NumElements() != 0) actual.GetSemanticArray().eval();
            }
            REQUIRE(g_elementwise_host_sync_count == 0);
            REQUIRE(g_elementwise_fallback_count == 0);
            CheckElementwise(actual, test_case);
            continue;
        }

        g_elementwise_host_sync_count = 0;
        g_elementwise_fallback_count = 0;
        cyxwiz::Tensor actual;
        {
            const cyxwiz::ScopedArrayFireFallbackPolicy strict(
                cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
            const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
                &CountElementwiseHostSync);
            const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
                &CountElementwiseFallback);
            actual = ApplyElementwise(input, active_case);
            if (actual.NumElements() != 0) actual.GetSemanticArray().eval();
        }
        REQUIRE(g_elementwise_host_sync_count == 0);
        REQUIRE(g_elementwise_fallback_count == 0);
        CheckElementwise(actual, test_case);
    }
}
#endif

} // namespace

TEST_CASE("Tensor elementwise operations match generated PyTorch dtype broadcast and rank matrix",
          "[tensor][tensor_elementwise][pytorch]") {
    const json fixture = LoadElementwiseFixture();
#ifdef CYXWIZ_HAS_ARRAYFIRE
    const auto activation =
        cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(true);
    REQUIRE(activation.success);
    REQUIRE(activation.execution_validated);
    CheckFixtureOnActiveDevice(fixture, true);
#else
    for (const json& test_case : fixture) {
        CAPTURE(test_case.at("name").get<std::string>());
        const auto dtype =
            ParseDataType(test_case.at("input_dtype").get<std::string>());
        const cyxwiz::Tensor input =
            TensorFromFixture(test_case.at("input"), dtype);
        CheckElementwise(ApplyElementwise(input, test_case), test_case);
    }
#endif
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
TEST_CASE("Tensor elementwise matrix matches PyTorch on every installed accelerator",
          "[tensor][tensor_elementwise][accelerator]") {
    const json fixture = LoadElementwiseFixture();
    size_t exercised_routes = 0;
    for (const auto& device : cyxwiz::Device::GetAvailableDevices()) {
        if (!device.device_selectable ||
            (device.type != cyxwiz::DeviceType::CUDA &&
             device.type != cyxwiz::DeviceType::OPENCL)) {
            continue;
        }
        INFO("device type=" << static_cast<int>(device.type)
             << " id=" << device.device_id << " name=" << device.name);
        const auto activation =
            cyxwiz::Device(device.type, device.device_id).ActivateExact(true);
        REQUIRE(activation.success);
        REQUIRE(activation.execution_validated);
        REQUIRE(activation.effective_type == device.type);
        REQUIRE(activation.effective_device_id == device.device_id);
        CheckFixtureOnActiveDevice(
            fixture, !device.supports_fp64_known || device.supports_fp64);
        ++exercised_routes;
    }
    if (exercised_routes == 0) {
        WARN("No CUDA or OpenCL route was available for elementwise parity");
    }
    REQUIRE(cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(true).success);
}

// TOFIX140 F4: rank-five elementwise ops run on the device (flat view, with
// the broadcast collapse). Each result equals the same op on the equivalent
// rank-4 shapes, which take the semantic path.
TEST_CASE("Tensor rank-five elementwise ops run on the device",
          "[tensor][tensor_elementwise][arrayfire]") {
    REQUIRE(cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(true).success);
    const cyxwiz::Tensor left = cyxwiz::Tensor::RangeN(
        {1, 2, 1, 2, 2}, cyxwiz::DataType::Float32);
    const float right_values[] = {1.0f, 2.0f, 3.0f, 0.5f};
    const cyxwiz::Tensor right(
        {2, 1, 2, 1}, right_values, cyxwiz::DataType::Float32);
    const cyxwiz::Tensor left4 = left.Reshape({2, 1, 2, 2});
    right.GetRowMajorFlatArray().eval();

    using Op = std::function<cyxwiz::Tensor(const cyxwiz::Tensor&, const cyxwiz::Tensor&)>;
    const std::vector<std::pair<std::string, Op>> ops = {
        {"add", [](const auto& l, const auto& r) { return l + r; }},
        {"subtract", [](const auto& l, const auto& r) { return l - r; }},
        {"multiply", [](const auto& l, const auto& r) { return l * r; }},
        {"divide", [](const auto& l, const auto& r) { return l / r; }},
        {"pow tensor", [](const auto& l, const auto& r) { return l.Pow(r); }},
        {"add scalar", [](const auto& l, const auto&) { return l + 1.0f; }},
        {"subtract scalar", [](const auto& l, const auto&) { return l - 1.0f; }},
        {"multiply scalar", [](const auto& l, const auto&) { return l * 2.0f; }},
        {"divide scalar", [](const auto& l, const auto&) { return l / 2.0f; }},
        {"pow scalar", [](const auto& l, const auto&) { return l.Pow(2.0f); }},
        {"sqrt", [](const auto& l, const auto&) { return l.Sqrt(); }},
        {"exp", [](const auto& l, const auto&) { return l.Exp(); }},
        {"log", [](const auto& l, const auto&) { return l.Log(); }},
        {"sin", [](const auto& l, const auto&) { return l.Sin(); }},
        {"cos", [](const auto& l, const auto&) { return l.Cos(); }},
        {"abs", [](const auto& l, const auto&) { return (l - 3.0f).Abs(); }},
        {"sign", [](const auto& l, const auto&) { return (l - 3.0f).Sign(); }},
        {"clip", [](const auto& l, const auto&) { return l.Clip(1.0f, 4.0f); }},
        {"negate", [](const auto& l, const auto&) { return -l; }},
    };
    g_elementwise_fallback_count = 0;
    std::vector<cyxwiz::Tensor> rank5, rank4;
    {
        const cyxwiz::ScopedArrayFireFallbackPolicy strict(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver observer(
            &CountElementwiseFallback);
        for (const auto& [name, op] : ops) {
            rank5.push_back(op(left, right));
            rank4.push_back(op(left4, right));
        }
    }
    REQUIRE(g_elementwise_fallback_count == 0);
    for (size_t i = 0; i < ops.size(); ++i) {
        CAPTURE(ops[i].first);
        REQUIRE(rank5[i].Shape() == left.Shape());
        const float* a = rank5[i].ReadData<float>();
        const float* b = rank4[i].ReadData<float>();
        for (size_t k = 0; k < 8; ++k) {
            CAPTURE(k);
            REQUIRE(((std::isnan(a[k]) && std::isnan(b[k])) || a[k] == b[k]));
        }
    }
}
#endif

TEST_CASE("Tensor elementwise finite extremes and invalid domains follow PyTorch",
          "[tensor][tensor_elementwise][extreme]") {
    const float values[] = {
        -std::numeric_limits<float>::infinity(), -1.0f, -0.0f, 0.0f,
        1.0f, std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN()};
    const cyxwiz::Tensor input({7}, values, cyxwiz::DataType::Float32);
    const cyxwiz::Tensor root = input.Sqrt();
    const cyxwiz::Tensor logarithm = input.Log();
    const cyxwiz::Tensor sign = input.Sign();
    const cyxwiz::Tensor clipped = input.Clip(-0.5f, 0.5f);
    const cyxwiz::Tensor divided = input / 0.0f;
    const cyxwiz::Tensor sine = input.Sin();
    const cyxwiz::Tensor cosine = input.Cos();
    // PyTorch: sin/cos of +-inf and NaN are NaN; sin keeps the sign of -0.
    REQUIRE(std::isnan(sine.ReadData<float>()[0]));
    REQUIRE(std::isnan(sine.ReadData<float>()[5]));
    REQUIRE(std::isnan(sine.ReadData<float>()[6]));
    REQUIRE(std::signbit(sine.ReadData<float>()[2]));
    REQUIRE(sine.ReadData<float>()[2] == 0.0f);
    REQUIRE(std::isnan(cosine.ReadData<float>()[0]));
    REQUIRE(cosine.ReadData<float>()[3] == 1.0f);
    REQUIRE(std::isnan(cosine.ReadData<float>()[6]));
    REQUIRE(std::isnan(root.ReadData<float>()[0]));
    REQUIRE(std::isnan(root.ReadData<float>()[1]));
    REQUIRE(std::signbit(root.ReadData<float>()[2]));
    REQUIRE(std::isinf(logarithm.ReadData<float>()[2]));
    REQUIRE(logarithm.ReadData<float>()[2] < 0.0f);
    REQUIRE(sign.ReadData<float>()[0] == -1.0f);
    REQUIRE(sign.ReadData<float>()[6] == 0.0f);
    REQUIRE(clipped.ReadData<float>()[0] == -0.5f);
    REQUIRE(clipped.ReadData<float>()[5] == 0.5f);
    REQUIRE(std::isnan(clipped.ReadData<float>()[6]));
    REQUIRE(std::isinf(divided.ReadData<float>()[0]));
    REQUIRE(divided.ReadData<float>()[0] < 0.0f);
    REQUIRE(std::isnan(divided.ReadData<float>()[2]));

    const int32_t integer_values[] = {
        (std::numeric_limits<int32_t>::min)(),
        (std::numeric_limits<int32_t>::max)(), 2};
    const int32_t powers[] = {1, 1, 31};
    const cyxwiz::Tensor integers(
        {3}, integer_values, cyxwiz::DataType::Int32);
    const cyxwiz::Tensor exponents({3}, powers, cyxwiz::DataType::Int32);
    const int32_t one_values[] = {1, 1, 2};
    const int32_t multiplier_values[] = {1, 2, 2};
    const cyxwiz::Tensor integer_rhs(
        {3}, one_values, cyxwiz::DataType::Int32);
    const cyxwiz::Tensor integer_multiplier(
        {3}, multiplier_values, cyxwiz::DataType::Int32);
    REQUIRE((integers + integer_rhs).ReadData<int32_t>()[1] ==
            (std::numeric_limits<int32_t>::min)());
    REQUIRE((integers - integer_rhs).ReadData<int32_t>()[0] ==
            (std::numeric_limits<int32_t>::max)());
    REQUIRE((integers * integer_multiplier).ReadData<int32_t>()[1] == -2);
    REQUIRE((-integers).ReadData<int32_t>()[0] ==
            (std::numeric_limits<int32_t>::min)());
    REQUIRE(integers.Abs().ReadData<int32_t>()[0] ==
            (std::numeric_limits<int32_t>::min)());
    REQUIRE(integers.Pow(exponents).ReadData<int32_t>()[2] ==
            (std::numeric_limits<int32_t>::min)());
    REQUIRE(integers.Clip(2.0f, 1.0f).GetDataType() ==
            cyxwiz::DataType::Float32);
    REQUIRE(integers.Clip(2.0f, 1.0f).ReadData<float>()[0] == 1.0f);

    const int32_t numerator_values[] = {1, 0, -1};
    const int32_t denominator_values[] = {0, 0, 0};
    const cyxwiz::Tensor numerators(
        {3}, numerator_values, cyxwiz::DataType::Int32);
    const cyxwiz::Tensor denominators(
        {3}, denominator_values, cyxwiz::DataType::Int32);
    const cyxwiz::Tensor integer_division = numerators / denominators;
    REQUIRE(integer_division.GetDataType() == cyxwiz::DataType::Float32);
    REQUIRE(std::isinf(integer_division.ReadData<float>()[0]));
    REQUIRE(std::isnan(integer_division.ReadData<float>()[1]));
    REQUIRE(std::isinf(integer_division.ReadData<float>()[2]));
    REQUIRE(integer_division.ReadData<float>()[2] < 0.0f);

    REQUIRE_THROWS_AS(
        input + cyxwiz::Tensor::Ones({2}, cyxwiz::DataType::Float32),
        std::runtime_error);
    REQUIRE_THROWS_AS(
        input.Pow(cyxwiz::Tensor::Ones({2}, cyxwiz::DataType::Float32)),
        std::runtime_error);
}
