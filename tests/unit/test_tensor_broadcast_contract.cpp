#include <catch2/catch_test_macros.hpp>

#include "algorithms/arrayfire_backend_utils.h"

#include <cyxwiz/device.h>
#include <cyxwiz/tensor.h>

#include <arrayfire.h>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;

json LoadBroadcastFixture() {
    std::ifstream stream(CYXWIZ_TRAINING_CORE_FIXTURE_PATH);
    REQUIRE(stream.is_open());
    json fixture;
    stream >> fixture;
    REQUIRE(fixture.value("schema_version", 0) == 1);
    REQUIRE(fixture.at("oracle").value("name", "") == "PyTorch");
    REQUIRE(fixture.at("oracle").value("version", "").rfind("2.10.0", 0) == 0);
    return fixture.at("cases").at("tensor_broadcast");
}

cyxwiz::DataType ParseDataType(const std::string& value) {
    if (value == "float32") return cyxwiz::DataType::Float32;
    if (value == "float64") return cyxwiz::DataType::Float64;
    if (value == "int32") return cyxwiz::DataType::Int32;
    if (value == "int64") return cyxwiz::DataType::Int64;
    if (value == "uint8") return cyxwiz::DataType::UInt8;
    throw std::runtime_error("unsupported broadcast fixture dtype: " + value);
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
    throw std::runtime_error("unsupported broadcast Tensor dtype");
}

template <typename T>
void CheckTypedTensor(const cyxwiz::Tensor& actual, const json& expected) {
    const auto values = expected.at("values").get<std::vector<T>>();
    REQUIRE(actual.NumElements() == values.size());
    if (values.empty()) return;
    const T* output = actual.ReadData<T>();
    for (size_t index = 0; index < values.size(); ++index) {
        REQUIRE(output[index] == values[index]);
    }
}

void CheckTensor(const cyxwiz::Tensor& actual,
                 cyxwiz::DataType dtype,
                 const json& expected) {
    REQUIRE(actual.GetDataType() == dtype);
    REQUIRE(actual.Shape() ==
            expected.at("shape").get<std::vector<size_t>>());
    switch (dtype) {
        case cyxwiz::DataType::Float32:
            CheckTypedTensor<float>(actual, expected);
            return;
        case cyxwiz::DataType::Float64:
            CheckTypedTensor<double>(actual, expected);
            return;
        case cyxwiz::DataType::Int32:
            CheckTypedTensor<int32_t>(actual, expected);
            return;
        case cyxwiz::DataType::Int64:
            CheckTypedTensor<int64_t>(actual, expected);
            return;
        case cyxwiz::DataType::UInt8:
            CheckTypedTensor<uint8_t>(actual, expected);
            return;
    }
    throw std::runtime_error("unsupported broadcast expected dtype");
}

cyxwiz::Tensor ApplyMaterialization(const cyxwiz::Tensor& input,
                                    const std::vector<size_t>& target_shape,
                                    const std::string& operation) {
    if (operation == "expand") return input.Expand(target_shape);
    if (operation == "broadcast_to") return input.BroadcastTo(target_shape);
    throw std::runtime_error("unsupported broadcast operation: " + operation);
}

void CheckShapeCases(const json& fixture) {
    for (const json& test_case : fixture.at("shape_cases")) {
        const auto left = test_case.at("left").get<std::vector<size_t>>();
        const auto right = test_case.at("right").get<std::vector<size_t>>();
        const bool expected = test_case.at("broadcastable").get<bool>();
        CAPTURE(test_case.at("name").get<std::string>());
        REQUIRE(cyxwiz::Tensor::IsBroadcastable(left, right) == expected);
        REQUIRE(cyxwiz::Tensor::IsBroadcastable(right, left) == expected);
        if (expected) {
            const auto expected_shape =
                test_case.at("expected_shape").get<std::vector<size_t>>();
            REQUIRE(cyxwiz::Tensor::BroadcastShape(left, right) == expected_shape);
            REQUIRE(cyxwiz::Tensor::BroadcastShape(right, left) == expected_shape);
        } else {
            REQUIRE_THROWS_AS(
                cyxwiz::Tensor::BroadcastShape(left, right), std::runtime_error);
            REQUIRE_THROWS_AS(
                cyxwiz::Tensor::BroadcastShape(right, left), std::runtime_error);
        }
    }
}

size_t g_broadcast_host_sync_count = 0;
size_t g_broadcast_fallback_count = 0;

void CountBroadcastHostSync(const cyxwiz::ArrayFireHostSyncEvent&) {
    ++g_broadcast_host_sync_count;
}

void CountBroadcastFallback(const cyxwiz::ArrayFireNativeCpuFallbackEvent&) {
    ++g_broadcast_fallback_count;
}

cyxwiz::Tensor DeviceTensorFromFixture(const json& value,
                                       cyxwiz::DataType dtype) {
    const cyxwiz::Tensor host = TensorFromFixture(value, dtype);
    if (host.NumElements() == 0) return host;
    return cyxwiz::Tensor::FromSemanticArray(
        host.GetSemanticArray(), host.Shape());
}

void CheckMaterializationCasesOnActiveDevice(const json& fixture,
                                             bool supports_fp64) {
    for (const json& test_case : fixture.at("materialization_cases")) {
        const std::string name = test_case.at("name").get<std::string>();
        const auto dtype =
            ParseDataType(test_case.at("dtype").get<std::string>());
        if (!supports_fp64 && dtype == cyxwiz::DataType::Float64) continue;
        const auto target_shape =
            test_case.at("target_shape").get<std::vector<size_t>>();
        CAPTURE(name);

        for (const std::string operation : {"expand", "broadcast_to"}) {
            CAPTURE(operation);
            const cyxwiz::Tensor input =
                DeviceTensorFromFixture(test_case.at("input"), dtype);
            cyxwiz::Tensor actual;
            g_broadcast_host_sync_count = 0;
            g_broadcast_fallback_count = 0;
            {
                const cyxwiz::ScopedArrayFireFallbackPolicy strict(
                    cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
                const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
                    &CountBroadcastHostSync);
                const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
                    &CountBroadcastFallback);
                actual = ApplyMaterialization(input, target_shape, operation);
                if (actual.NumElements() != 0) actual.GetSemanticArray().eval();
            }
            REQUIRE(g_broadcast_host_sync_count == 0);
            REQUIRE(g_broadcast_fallback_count == 0);
            CheckTensor(actual, dtype, test_case.at("expected"));
        }
    }
}

} // namespace

TEST_CASE("Tensor broadcast helpers and materialization match PyTorch",
          "[tensor][tensor_broadcast][pytorch]") {
    const json fixture = LoadBroadcastFixture();
    CheckShapeCases(fixture);
    const auto activation =
        cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(true);
    REQUIRE(activation.success);
    REQUIRE(activation.execution_validated);
    CheckMaterializationCasesOnActiveDevice(fixture, true);
}

TEST_CASE("Tensor broadcast materialization matches PyTorch on every accelerator",
          "[tensor][tensor_broadcast][accelerator]") {
    const json fixture = LoadBroadcastFixture();
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
        CheckMaterializationCasesOnActiveDevice(
            fixture, !device.supports_fp64_known || device.supports_fp64);
        ++exercised_routes;
    }
    if (exercised_routes == 0) {
        WARN("No CUDA or OpenCL route was available for Tensor broadcast parity");
    }
    REQUIRE(cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(true).success);
}

// TOFIX140 F4: rank-five broadcasts run on the device. [1,2,1,2,1] ->
// [3,2,4,2,5] alternates broadcast and kept axes (five groups: the gather
// path); [1,2,1,1,3] -> [4,2,5,6,3] collapses to four (the tile path).
TEST_CASE("Tensor rank-five broadcast runs on the device",
          "[tensor][tensor_broadcast][arrayfire]") {
    REQUIRE(cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(true).success);
    const int32_t alternating_values[] = {0, 1, 2, 3};
    const cyxwiz::Tensor alternating({1, 2, 1, 2, 1}, alternating_values, cyxwiz::DataType::Int32);
    const int32_t collapsing_values[] = {0, 1, 2, 3, 4, 5};
    const cyxwiz::Tensor collapsing({1, 2, 1, 1, 3}, collapsing_values, cyxwiz::DataType::Int32);
    alternating.GetRowMajorFlatArray().eval();  // resident before the strict scope
    collapsing.GetRowMajorFlatArray().eval();

    const std::vector<size_t> alternating_target{3, 2, 4, 2, 5};
    const std::vector<size_t> collapsing_target{4, 2, 5, 6, 3};
    g_broadcast_host_sync_count = 0;
    g_broadcast_fallback_count = 0;
    cyxwiz::Tensor expanded, broadcast, identity, collapsed, sum;
    {
        const cyxwiz::ScopedArrayFireFallbackPolicy strict(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
            &CountBroadcastHostSync);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CountBroadcastFallback);
        expanded = alternating.Expand(alternating_target);
        broadcast = alternating.BroadcastTo(alternating_target);
        identity = alternating.Expand({1, 2, 1, 2, 1});
        collapsed = collapsing.BroadcastTo(collapsing_target);
        sum = alternating + collapsing.Reshape({1, 1, 1, 2, 3}).Expand({1, 2, 1, 2, 3});
    }
    REQUIRE(g_broadcast_host_sync_count == 0);
    REQUIRE(g_broadcast_fallback_count == 0);

    REQUIRE(expanded.Shape() == alternating_target);
    const int32_t* expanded_values = expanded.ReadData<int32_t>();
    const int32_t* broadcast_values = broadcast.ReadData<int32_t>();
    size_t index = 0;
    for (size_t a0 = 0; a0 < 3; ++a0)
        for (size_t a1 = 0; a1 < 2; ++a1)
            for (size_t a2 = 0; a2 < 4; ++a2)
                for (size_t a3 = 0; a3 < 2; ++a3)
                    for (size_t a4 = 0; a4 < 5; ++a4, ++index) {
                        const int32_t expected = static_cast<int32_t>(a1 * 2 + a3);
                        REQUIRE(expanded_values[index] == expected);
                        REQUIRE(broadcast_values[index] == expected);
                    }
    const int32_t* identity_values = identity.ReadData<int32_t>();
    for (size_t i = 0; i < 4; ++i) REQUIRE(identity_values[i] == static_cast<int32_t>(i));

    REQUIRE(collapsed.Shape() == collapsing_target);
    const int32_t* collapsed_values = collapsed.ReadData<int32_t>();
    index = 0;
    for (size_t a0 = 0; a0 < 4; ++a0)
        for (size_t a1 = 0; a1 < 2; ++a1)
            for (size_t a2 = 0; a2 < 5; ++a2)
                for (size_t a3 = 0; a3 < 6; ++a3)
                    for (size_t a4 = 0; a4 < 3; ++a4, ++index)
                        REQUIRE(collapsed_values[index] == static_cast<int32_t>(a1 * 3 + a4));

    // [1,2,1,2,1] + [1,2,1,2,3] -> [1,2,1,2,3]: elementwise add with a rank-five broadcast.
    REQUIRE(sum.Shape() == std::vector<size_t>{1, 2, 1, 2, 3});
    const int32_t* sum_values = sum.ReadData<int32_t>();
    index = 0;
    for (size_t a1 = 0; a1 < 2; ++a1)
        for (size_t a3 = 0; a3 < 2; ++a3)
            for (size_t a4 = 0; a4 < 3; ++a4, ++index)
                REQUIRE(sum_values[index] == static_cast<int32_t>(a1 * 2 + a3 + a3 * 3 + a4));
}

TEST_CASE("Tensor broadcast materialization rejects PyTorch-invalid shapes",
          "[tensor][tensor_broadcast][invalid]") {
    const json fixture = LoadBroadcastFixture();
    for (const json& test_case : fixture.at("invalid_materializations")) {
        CAPTURE(test_case.at("name").get<std::string>());
        const auto input_shape =
            test_case.at("input_shape").get<std::vector<size_t>>();
        const auto target_shape =
            test_case.at("target_shape").get<std::vector<size_t>>();
        const cyxwiz::Tensor input(input_shape, cyxwiz::DataType::Float32);
        REQUIRE_THROWS_AS(input.Expand(target_shape), std::runtime_error);
        REQUIRE_THROWS_AS(input.BroadcastTo(target_shape), std::runtime_error);
    }
}

TEST_CASE("Tensor broadcast shape overflow fails before compute",
          "[tensor][tensor_broadcast][invalid]") {
    const float value = 1.0f;
    const cyxwiz::Tensor scalar({}, &value, cyxwiz::DataType::Float32);
    const std::vector<size_t> target{
        (std::numeric_limits<size_t>::max)(), 2};
    g_broadcast_host_sync_count = 0;
    g_broadcast_fallback_count = 0;
    {
        const cyxwiz::ScopedArrayFireFallbackPolicy strict(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
            &CountBroadcastHostSync);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CountBroadcastFallback);
        REQUIRE_THROWS_AS(scalar.Expand(target), std::overflow_error);
        REQUIRE_THROWS_AS(scalar.BroadcastTo(target), std::overflow_error);
    }
    REQUIRE(g_broadcast_host_sync_count == 0);
    REQUIRE(g_broadcast_fallback_count == 0);
}
