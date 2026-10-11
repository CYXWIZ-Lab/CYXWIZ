#include <catch2/catch_test_macros.hpp>

#include "algorithms/arrayfire_backend_utils.h"

#include <cyxwiz/device.h>
#include <cyxwiz/tensor.h>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <functional>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;

json LoadMaskFixture() {
    std::ifstream stream(CYXWIZ_TRAINING_CORE_FIXTURE_PATH);
    REQUIRE(stream.is_open());
    json fixture;
    stream >> fixture;
    REQUIRE(fixture.value("schema_version", 0) == 1);
    REQUIRE(fixture.at("oracle").value("name", "") == "PyTorch");
    REQUIRE(fixture.at("oracle").value("version", "").rfind("2.10.0", 0) == 0);
    return fixture.at("cases").at("tensor_masks");
}

cyxwiz::DataType ParseDataType(const std::string& value) {
    if (value == "float32") return cyxwiz::DataType::Float32;
    if (value == "float64") return cyxwiz::DataType::Float64;
    if (value == "int32") return cyxwiz::DataType::Int32;
    if (value == "int64") return cyxwiz::DataType::Int64;
    if (value == "uint8") return cyxwiz::DataType::UInt8;
    throw std::runtime_error("unsupported mask fixture dtype: " + value);
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
    throw std::runtime_error("unsupported mask Tensor dtype");
}

cyxwiz::Tensor ApplyTensorMask(const cyxwiz::Tensor& left,
                               const cyxwiz::Tensor& right,
                               const std::string& operation) {
    if (operation == "greater") return left > right;
    if (operation == "greater_equal") return left >= right;
    if (operation == "less") return left < right;
    if (operation == "less_equal") return left <= right;
    if (operation == "equal") return left == right;
    if (operation == "not_equal") return left != right;
    if (operation == "logical_and") return left && right;
    if (operation == "logical_or") return left || right;
    throw std::runtime_error("unsupported tensor mask operation: " + operation);
}

cyxwiz::Tensor ApplyScalarMask(const cyxwiz::Tensor& input,
                               float scalar,
                               const std::string& operation) {
    if (operation == "greater") return input > scalar;
    if (operation == "greater_equal") return input >= scalar;
    if (operation == "less") return input < scalar;
    if (operation == "less_equal") return input <= scalar;
    if (operation == "equal") return input == scalar;
    if (operation == "not_equal") return input != scalar;
    throw std::runtime_error("unsupported scalar mask operation: " + operation);
}

cyxwiz::Tensor ApplyMask(const cyxwiz::Tensor& input,
                         const json& test_case) {
    const std::string operation = test_case.at("operation").get<std::string>();
    const std::string rhs_kind = test_case.at("rhs_kind").get<std::string>();
    if (rhs_kind == "tensor") {
        const auto dtype =
            ParseDataType(test_case.at("rhs_dtype").get<std::string>());
        return ApplyTensorMask(
            input, TensorFromFixture(test_case.at("rhs"), dtype), operation);
    }
    if (rhs_kind == "scalar") {
        return ApplyScalarMask(
            input, test_case.at("scalar").get<float>(), operation);
    }
    if (operation == "logical_not") return !input;
    throw std::runtime_error("unsupported mask fixture operation: " + operation);
}

void CheckMask(const cyxwiz::Tensor& actual, const json& expected) {
    const auto shape = expected.at("shape").get<std::vector<size_t>>();
    const auto values = expected.at("values").get<std::vector<uint8_t>>();
    REQUIRE(actual.GetDataType() == cyxwiz::DataType::UInt8);
    REQUIRE(actual.Shape() == shape);
    REQUIRE(actual.NumElements() == values.size());
    const uint8_t* output = actual.ReadData<uint8_t>();
    for (size_t index = 0; index < values.size(); ++index) {
        REQUIRE(output[index] == values[index]);
    }
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
size_t g_mask_host_sync_count = 0;
size_t g_mask_fallback_count = 0;

void CountMaskHostSync(const cyxwiz::ArrayFireHostSyncEvent&) {
    ++g_mask_host_sync_count;
}

void CountMaskFallback(const cyxwiz::ArrayFireNativeCpuFallbackEvent&) {
    ++g_mask_fallback_count;
}

cyxwiz::Tensor DeviceTensorFromFixture(const json& value,
                                       cyxwiz::DataType dtype) {
    const cyxwiz::Tensor host = TensorFromFixture(value, dtype);
    if (host.NumElements() == 0) return host;
    return cyxwiz::Tensor::FromSemanticArray(
        host.GetSemanticArray(), host.Shape());
}

void CheckFixtureOnActiveDevice(const json& fixture, bool supports_fp64) {
    for (const json& test_case : fixture) {
        const std::string name = test_case.at("name").get<std::string>();
        CAPTURE(name);
        const auto input_dtype =
            ParseDataType(test_case.at("input_dtype").get<std::string>());
        if (!supports_fp64 &&
            (input_dtype == cyxwiz::DataType::Float64 ||
             (test_case.contains("rhs_dtype") &&
              ParseDataType(test_case.at("rhs_dtype").get<std::string>()) ==
                  cyxwiz::DataType::Float64))) {
            continue;
        }

        const cyxwiz::Tensor input =
            DeviceTensorFromFixture(test_case.at("input"), input_dtype);
        cyxwiz::Tensor actual;
        g_mask_host_sync_count = 0;
        g_mask_fallback_count = 0;
        {
            const cyxwiz::ScopedArrayFireFallbackPolicy strict(
                cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
            const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
                &CountMaskHostSync);
            const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
                &CountMaskFallback);
            if (test_case.at("rhs_kind") == "tensor") {
                const auto right_dtype = ParseDataType(
                    test_case.at("rhs_dtype").get<std::string>());
                const cyxwiz::Tensor right = DeviceTensorFromFixture(
                    test_case.at("rhs"), right_dtype);
                actual = ApplyTensorMask(
                    input, right,
                    test_case.at("operation").get<std::string>());
            } else {
                actual = ApplyMask(input, test_case);
            }
            if (actual.NumElements() != 0) actual.GetSemanticArray().eval();
        }
        REQUIRE(g_mask_host_sync_count == 0);
        REQUIRE(g_mask_fallback_count == 0);
        CheckMask(actual, test_case.at("expected"));
    }
}
#endif

} // namespace

TEST_CASE("Tensor masks match generated PyTorch dtype broadcast and rank matrix",
          "[tensor][tensor_mask][pytorch]") {
    const json fixture = LoadMaskFixture();
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
        CheckMask(
            ApplyMask(TensorFromFixture(test_case.at("input"), dtype), test_case),
            test_case.at("expected"));
    }
#endif
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
TEST_CASE("Tensor masks match PyTorch on every installed accelerator",
          "[tensor][tensor_mask][accelerator]") {
    const json fixture = LoadMaskFixture();
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
        WARN("No CUDA or OpenCL route was available for Tensor mask parity");
    }
    REQUIRE(cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(true).success);
}

// TOFIX140 F4: rank-five comparisons and logical ops run on the device; each
// result equals the same op on the equivalent rank-4 shapes.
TEST_CASE("Tensor rank-five masks run on the device",
          "[tensor][tensor_mask][arrayfire]") {
    REQUIRE(cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(true).success);
    const cyxwiz::Tensor left = cyxwiz::Tensor::RangeN(
        {1, 2, 1, 2, 2}, cyxwiz::DataType::Int32);
    const int32_t right_values[] = {0, 1, 0, 1};
    const cyxwiz::Tensor right(
        {2, 1, 2, 1}, right_values, cyxwiz::DataType::Int32);
    const cyxwiz::Tensor left4 = left.Reshape({2, 1, 2, 2});
    right.GetRowMajorFlatArray().eval();

    using Op = std::function<cyxwiz::Tensor(const cyxwiz::Tensor&, const cyxwiz::Tensor&)>;
    const std::vector<std::pair<std::string, Op>> ops = {
        {">", [](const auto& l, const auto& r) { return l > r; }},
        {">=", [](const auto& l, const auto& r) { return l >= r; }},
        {"<", [](const auto& l, const auto& r) { return l < r; }},
        {"<=", [](const auto& l, const auto& r) { return l <= r; }},
        {"==", [](const auto& l, const auto& r) { return l == r; }},
        {"!=", [](const auto& l, const auto& r) { return l != r; }},
        {"> scalar", [](const auto& l, const auto&) { return l > 2.5f; }},
        {">= scalar", [](const auto& l, const auto&) { return l >= 2.5f; }},
        {"< scalar", [](const auto& l, const auto&) { return l < 2.5f; }},
        {"<= scalar", [](const auto& l, const auto&) { return l <= 2.5f; }},
        {"== scalar", [](const auto& l, const auto&) { return l == 3.0f; }},
        {"!= scalar", [](const auto& l, const auto&) { return l != 3.0f; }},
        {"&&", [](const auto& l, const auto& r) { return l && r; }},
        {"||", [](const auto& l, const auto& r) { return l || r; }},
        {"!", [](const auto& l, const auto&) { return !l; }},
    };
    g_mask_fallback_count = 0;
    std::vector<cyxwiz::Tensor> rank5, rank4;
    {
        const cyxwiz::ScopedArrayFireFallbackPolicy strict(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver observer(
            &CountMaskFallback);
        for (const auto& [name, op] : ops) {
            rank5.push_back(op(left, right));
            rank4.push_back(op(left4, right));
        }
    }
    REQUIRE(g_mask_fallback_count == 0);
    for (size_t i = 0; i < ops.size(); ++i) {
        CAPTURE(ops[i].first);
        REQUIRE(rank5[i].Shape() == left.Shape());
        REQUIRE(rank5[i].GetDataType() == cyxwiz::DataType::UInt8);
        const uint8_t* a = rank5[i].ReadData<uint8_t>();
        const uint8_t* b = rank4[i].ReadData<uint8_t>();
        for (size_t k = 0; k < 8; ++k) REQUIRE(a[k] == b[k]);
    }
}
#endif

TEST_CASE("Tensor masks reject incompatible broadcast shapes",
          "[tensor][tensor_mask][invalid]") {
    const cyxwiz::Tensor left =
        cyxwiz::Tensor::Ones({2, 3}, cyxwiz::DataType::Float32);
    const cyxwiz::Tensor right =
        cyxwiz::Tensor::Ones({4}, cyxwiz::DataType::Int64);
    REQUIRE_THROWS_AS(left == right, std::runtime_error);
    REQUIRE_THROWS_AS(left && right, std::runtime_error);
}

TEST_CASE("Tensor masks preserve PyTorch special-value truth semantics",
          "[tensor][tensor_mask][extreme]") {
    const float values[] = {
        -std::numeric_limits<float>::infinity(), -0.0f, 0.0f,
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN()};
    const cyxwiz::Tensor input({5}, values, cyxwiz::DataType::Float32);
    const cyxwiz::Tensor equal = input == input;
    const cyxwiz::Tensor not_equal = input != input;
    const cyxwiz::Tensor logical_not = !input;
    const uint8_t* eq = equal.ReadData<uint8_t>();
    const uint8_t* ne = not_equal.ReadData<uint8_t>();
    const uint8_t* inverted = logical_not.ReadData<uint8_t>();
    REQUIRE(eq[0] == 1);
    REQUIRE(eq[1] == 1);
    REQUIRE(eq[2] == 1);
    REQUIRE(eq[3] == 1);
    REQUIRE(eq[4] == 0);
    REQUIRE(ne[4] == 1);
    REQUIRE(inverted[0] == 0);
    REQUIRE(inverted[1] == 1);
    REQUIRE(inverted[2] == 1);
    REQUIRE(inverted[3] == 0);
    REQUIRE(inverted[4] == 0);
}
