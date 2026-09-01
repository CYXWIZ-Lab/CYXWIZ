#include <catch2/catch_test_macros.hpp>

#include "algorithms/arrayfire_backend_utils.h"

#include <cyxwiz/device.h>
#include <cyxwiz/tensor.h>

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

#ifdef CYXWIZ_HAS_ARRAYFIRE
size_t g_broadcast_host_sync_count = 0;
size_t g_broadcast_fallback_count = 0;
std::vector<std::string> g_broadcast_fallback_operations;

void CountBroadcastHostSync(const cyxwiz::ArrayFireHostSyncEvent&) {
    ++g_broadcast_host_sync_count;
}

void CountBroadcastFallback(
    const cyxwiz::ArrayFireNativeCpuFallbackEvent& event) {
    ++g_broadcast_fallback_count;
    g_broadcast_fallback_operations.push_back(event.operation_name);
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
#endif

} // namespace

TEST_CASE("Tensor broadcast helpers and materialization match PyTorch",
          "[tensor][tensor_broadcast][pytorch]") {
    const json fixture = LoadBroadcastFixture();
    CheckShapeCases(fixture);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    const auto activation =
        cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(true);
    REQUIRE(activation.success);
    REQUIRE(activation.execution_validated);
    CheckMaterializationCasesOnActiveDevice(fixture, true);
#else
    for (const json& test_case : fixture.at("materialization_cases")) {
        const auto dtype =
            ParseDataType(test_case.at("dtype").get<std::string>());
        const auto target_shape =
            test_case.at("target_shape").get<std::vector<size_t>>();
        for (const std::string operation : {"expand", "broadcast_to"}) {
            const cyxwiz::Tensor input =
                TensorFromFixture(test_case.at("input"), dtype);
            CheckTensor(
                ApplyMaterialization(input, target_shape, operation),
                dtype,
                test_case.at("expected"));
        }
    }
#endif
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
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

TEST_CASE("Tensor rank-five broadcast fallback is explicit and strict-safe",
          "[tensor][tensor_broadcast][arrayfire][fallback]") {
    REQUIRE(cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(true).success);
    const std::vector<size_t> input_shape{1, 2, 1, 2, 1};
    const std::vector<size_t> target_shape{3, 2, 4, 2, 5};
    auto make_input = [&]() {
        const int32_t values[] = {0, 1, 2, 3};
        return cyxwiz::Tensor(
            input_shape, values, cyxwiz::DataType::Int32);
    };

    const cyxwiz::Tensor compatibility_input = make_input();
    g_broadcast_fallback_count = 0;
    g_broadcast_fallback_operations.clear();
    cyxwiz::Tensor expanded;
    cyxwiz::Tensor broadcast;
    cyxwiz::Tensor identity_expand;
    cyxwiz::Tensor identity_broadcast;
    {
        const cyxwiz::ScopedArrayFireFallbackPolicy compatibility(
            cyxwiz::ArrayFireFallbackPolicy::AllowNativeCpuFallback);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver observer(
            &CountBroadcastFallback);
        expanded = compatibility_input.Expand(target_shape);
        broadcast = compatibility_input.BroadcastTo(target_shape);
        identity_expand = compatibility_input.Expand(input_shape);
        identity_broadcast = compatibility_input.BroadcastTo(input_shape);
    }
    REQUIRE(g_broadcast_fallback_count == 4);
    REQUIRE(g_broadcast_fallback_operations == std::vector<std::string>{
        "Tensor::Expand", "Tensor::BroadcastTo",
        "Tensor::Expand", "Tensor::BroadcastTo"});
    REQUIRE(expanded.Shape() == target_shape);
    REQUIRE(broadcast.Shape() == target_shape);
    REQUIRE(identity_expand.Shape() == input_shape);
    REQUIRE(identity_broadcast.Shape() == input_shape);
    const int32_t* expanded_values = expanded.ReadData<int32_t>();
    const int32_t* broadcast_values = broadcast.ReadData<int32_t>();
    size_t index = 0;
    for (size_t axis0 = 0; axis0 < 3; ++axis0) {
        for (size_t axis1 = 0; axis1 < 2; ++axis1) {
            for (size_t axis2 = 0; axis2 < 4; ++axis2) {
                for (size_t axis3 = 0; axis3 < 2; ++axis3) {
                    for (size_t axis4 = 0; axis4 < 5; ++axis4) {
                        const int32_t expected =
                            static_cast<int32_t>(axis1 * 2 + axis3);
                        REQUIRE(expanded_values[index] == expected);
                        REQUIRE(broadcast_values[index] == expected);
                        ++index;
                    }
                }
            }
        }
    }
    const int32_t* identity_expand_values =
        identity_expand.ReadData<int32_t>();
    const int32_t* identity_broadcast_values =
        identity_broadcast.ReadData<int32_t>();
    for (size_t value_index = 0; value_index < 4; ++value_index) {
        REQUIRE(identity_expand_values[value_index] ==
                static_cast<int32_t>(value_index));
        REQUIRE(identity_broadcast_values[value_index] ==
                static_cast<int32_t>(value_index));
    }

    const cyxwiz::Tensor strict_input = make_input();
    g_broadcast_host_sync_count = 0;
    g_broadcast_fallback_count = 0;
    g_broadcast_fallback_operations.clear();
    {
        const cyxwiz::ScopedArrayFireFallbackPolicy strict(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
            &CountBroadcastHostSync);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CountBroadcastFallback);
        REQUIRE_THROWS_AS(strict_input.Expand(target_shape), std::runtime_error);
        REQUIRE_THROWS_AS(
            strict_input.BroadcastTo(target_shape), std::runtime_error);
        REQUIRE_THROWS_AS(strict_input.Expand(input_shape), std::runtime_error);
        REQUIRE_THROWS_AS(
            strict_input.BroadcastTo(input_shape), std::runtime_error);
    }
    REQUIRE(g_broadcast_host_sync_count == 0);
    REQUIRE(g_broadcast_fallback_count == 4);
    REQUIRE(g_broadcast_fallback_operations == std::vector<std::string>{
        "Tensor::Expand", "Tensor::BroadcastTo",
        "Tensor::Expand", "Tensor::BroadcastTo"});
}
#endif

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
#ifdef CYXWIZ_HAS_ARRAYFIRE
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
#else
    REQUIRE_THROWS_AS(scalar.Expand(target), std::overflow_error);
    REQUIRE_THROWS_AS(scalar.BroadcastTo(target), std::overflow_error);
#endif
}
