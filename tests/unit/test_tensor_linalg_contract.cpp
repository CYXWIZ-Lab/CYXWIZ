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

json LoadLinalgFixture() {
    std::ifstream stream(CYXWIZ_TRAINING_CORE_FIXTURE_PATH);
    REQUIRE(stream.is_open());
    json fixture;
    stream >> fixture;
    REQUIRE(fixture.value("schema_version", 0) == 1);
    REQUIRE(fixture.at("oracle").value("name", "") == "PyTorch");
    REQUIRE(fixture.at("oracle").value("version", "").rfind("2.10.0", 0) == 0);
    return fixture.at("cases").at("tensor_linalg");
}

cyxwiz::DataType ParseDataType(const std::string& value) {
    if (value == "float32") return cyxwiz::DataType::Float32;
    if (value == "float64") return cyxwiz::DataType::Float64;
    if (value == "int32") return cyxwiz::DataType::Int32;
    if (value == "int64") return cyxwiz::DataType::Int64;
    if (value == "uint8") return cyxwiz::DataType::UInt8;
    throw std::runtime_error("unsupported linalg fixture dtype: " + value);
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
    throw std::runtime_error("unsupported linalg Tensor dtype");
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
    REQUIRE(actual.Shape() == expected.at("shape").get<std::vector<size_t>>());
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
    throw std::runtime_error("unsupported linalg expected dtype");
}

void CheckNativeParity(const json& fixture) {
    for (const json& test_case : fixture.at("dot_cases")) {
        const auto dtype =
            ParseDataType(test_case.at("dtype").get<std::string>());
        const cyxwiz::Tensor left =
            TensorFromFixture(test_case.at("left"), dtype);
        const cyxwiz::Tensor right =
            TensorFromFixture(test_case.at("right"), dtype);
        CAPTURE(test_case.at("name").get<std::string>());
        CheckTensor(left.Dot(right), dtype, test_case.at("expected"));
    }
    for (const json& test_case : fixture.at("batch_matmul_cases")) {
        const auto dtype =
            ParseDataType(test_case.at("dtype").get<std::string>());
        const cyxwiz::Tensor left =
            TensorFromFixture(test_case.at("left"), dtype);
        const cyxwiz::Tensor right =
            TensorFromFixture(test_case.at("right"), dtype);
        CAPTURE(test_case.at("name").get<std::string>());
        CheckTensor(left.BatchMatMul(right), dtype, test_case.at("expected"));
    }
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
size_t g_linalg_host_sync_count = 0;
size_t g_linalg_fallback_count = 0;
std::vector<std::string> g_linalg_fallback_operations;
std::vector<std::string> g_linalg_fallback_reasons;

void CountLinalgHostSync(const cyxwiz::ArrayFireHostSyncEvent&) {
    ++g_linalg_host_sync_count;
}

void CountLinalgFallback(
    const cyxwiz::ArrayFireNativeCpuFallbackEvent& event) {
    ++g_linalg_fallback_count;
    g_linalg_fallback_operations.push_back(event.operation_name);
    g_linalg_fallback_reasons.push_back(event.reason_code);
}

cyxwiz::Tensor DeviceTensorFromFixture(const json& value,
                                       cyxwiz::DataType dtype) {
    const cyxwiz::Tensor host = TensorFromFixture(value, dtype);
    if (host.NumElements() == 0) return host;
    return cyxwiz::Tensor::FromSemanticArray(
        host.GetSemanticArray(), host.Shape());
}

void CheckResidentCasesOnActiveDevice(const json& fixture, bool supports_fp64) {
    for (const json& test_case : fixture.at("dot_cases")) {
        const auto dtype =
            ParseDataType(test_case.at("dtype").get<std::string>());
        if (!supports_fp64 && dtype == cyxwiz::DataType::Float64) continue;
        const cyxwiz::Tensor left =
            DeviceTensorFromFixture(test_case.at("left"), dtype);
        const cyxwiz::Tensor right =
            DeviceTensorFromFixture(test_case.at("right"), dtype);
        CAPTURE(test_case.at("name").get<std::string>());
        g_linalg_host_sync_count = 0;
        g_linalg_fallback_count = 0;
        cyxwiz::Tensor actual;
        {
            const cyxwiz::ScopedArrayFireFallbackPolicy strict(
                cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
            const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
                &CountLinalgHostSync);
            const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
                &CountLinalgFallback);
            actual = left.Dot(right);
            if (actual.NumElements() != 0) actual.GetSemanticArray().eval();
        }
        REQUIRE(g_linalg_host_sync_count == 0);
        REQUIRE(g_linalg_fallback_count == 0);
        CheckTensor(actual, dtype, test_case.at("expected"));
    }

    for (const json& test_case : fixture.at("batch_matmul_cases")) {
        const auto dtype =
            ParseDataType(test_case.at("dtype").get<std::string>());
        if (dtype != cyxwiz::DataType::Float32 &&
            dtype != cyxwiz::DataType::Float64) {
            continue;
        }
        if (!supports_fp64 && dtype == cyxwiz::DataType::Float64) continue;
        const cyxwiz::Tensor left =
            DeviceTensorFromFixture(test_case.at("left"), dtype);
        const cyxwiz::Tensor right =
            DeviceTensorFromFixture(test_case.at("right"), dtype);
        CAPTURE(test_case.at("name").get<std::string>());
        g_linalg_host_sync_count = 0;
        g_linalg_fallback_count = 0;
        cyxwiz::Tensor actual;
        {
            const cyxwiz::ScopedArrayFireFallbackPolicy strict(
                cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
            const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
                &CountLinalgHostSync);
            const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
                &CountLinalgFallback);
            actual = left.BatchMatMul(right);
            if (actual.NumElements() != 0) actual.GetSemanticArray().eval();
        }
        REQUIRE(g_linalg_host_sync_count == 0);
        REQUIRE(g_linalg_fallback_count == 0);
        CheckTensor(actual, dtype, test_case.at("expected"));
    }
}

void CheckIntegerBatchFallback(const json& fixture) {
    for (const json& test_case : fixture.at("batch_matmul_cases")) {
        const auto dtype =
            ParseDataType(test_case.at("dtype").get<std::string>());
        if (dtype == cyxwiz::DataType::Float32 ||
            dtype == cyxwiz::DataType::Float64) {
            continue;
        }
        const cyxwiz::Tensor left =
            DeviceTensorFromFixture(test_case.at("left"), dtype);
        const cyxwiz::Tensor right =
            DeviceTensorFromFixture(test_case.at("right"), dtype);
        const bool requires_compute =
            left.NumElements() != 0 && right.NumElements() != 0;
        CAPTURE(test_case.at("name").get<std::string>());

        g_linalg_fallback_count = 0;
        g_linalg_fallback_operations.clear();
        g_linalg_fallback_reasons.clear();
        cyxwiz::Tensor actual;
        {
            const cyxwiz::ScopedArrayFireFallbackPolicy compatibility(
                cyxwiz::ArrayFireFallbackPolicy::AllowNativeCpuFallback);
            const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver observer(
                &CountLinalgFallback);
            actual = left.BatchMatMul(right);
        }
        REQUIRE(g_linalg_fallback_count == (requires_compute ? 1 : 0));
        if (requires_compute) {
            REQUIRE(g_linalg_fallback_operations ==
                    std::vector<std::string>{"Tensor::BatchMatMul"});
            REQUIRE(g_linalg_fallback_reasons ==
                    std::vector<std::string>{"unsupported_dtype"});
        }
        CheckTensor(actual, dtype, test_case.at("expected"));

        if (!requires_compute) continue;
        const cyxwiz::Tensor strict_left =
            DeviceTensorFromFixture(test_case.at("left"), dtype);
        const cyxwiz::Tensor strict_right =
            DeviceTensorFromFixture(test_case.at("right"), dtype);
        g_linalg_host_sync_count = 0;
        g_linalg_fallback_count = 0;
        {
            const cyxwiz::ScopedArrayFireFallbackPolicy strict(
                cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
            const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
                &CountLinalgHostSync);
            const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
                &CountLinalgFallback);
            REQUIRE_THROWS_AS(
                strict_left.BatchMatMul(strict_right), std::runtime_error);
        }
        REQUIRE(g_linalg_host_sync_count == 0);
        REQUIRE(g_linalg_fallback_count == 1);
    }
}
#endif

} // namespace

TEST_CASE("Tensor Dot and BatchMatMul match the PyTorch contract",
          "[tensor][tensor_linalg][pytorch]") {
    const json fixture = LoadLinalgFixture();
#ifdef CYXWIZ_HAS_ARRAYFIRE
    const auto activation =
        cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(true);
    REQUIRE(activation.success);
    REQUIRE(activation.execution_validated);
    CheckResidentCasesOnActiveDevice(fixture, true);
    CheckIntegerBatchFallback(fixture);
#else
    CheckNativeParity(fixture);
#endif
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
TEST_CASE("Tensor floating BatchMatMul has a resident execution boundary",
          "[tensor][tensor_linalg][arrayfire][batch_residency]") {
    const json fixture = LoadLinalgFixture();
    REQUIRE(cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(true).success);
    const json& test_case = fixture.at("batch_matmul_cases").at(0);
    const auto dtype = ParseDataType(test_case.at("dtype").get<std::string>());
    const cyxwiz::Tensor left =
        DeviceTensorFromFixture(test_case.at("left"), dtype);
    const cyxwiz::Tensor right =
        DeviceTensorFromFixture(test_case.at("right"), dtype);
    g_linalg_host_sync_count = 0;
    g_linalg_fallback_count = 0;
    cyxwiz::Tensor actual;
    {
        const cyxwiz::ScopedArrayFireFallbackPolicy strict(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
            &CountLinalgHostSync);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CountLinalgFallback);
        actual = left.BatchMatMul(right);
        actual.GetSemanticArray().eval();
    }
    REQUIRE(g_linalg_host_sync_count == 0);
    REQUIRE(g_linalg_fallback_count == 0);
    CheckTensor(actual, dtype, test_case.at("expected"));
}

TEST_CASE("Tensor integer BatchMatMul declares native compatibility",
          "[tensor][tensor_linalg][arrayfire][batch_fallback]") {
    const json fixture = LoadLinalgFixture();
    REQUIRE(cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(true).success);
    const json& test_case = fixture.at("batch_matmul_cases").at(12);
    const auto dtype = ParseDataType(test_case.at("dtype").get<std::string>());
    const cyxwiz::Tensor left =
        DeviceTensorFromFixture(test_case.at("left"), dtype);
    const cyxwiz::Tensor right =
        DeviceTensorFromFixture(test_case.at("right"), dtype);
    g_linalg_fallback_count = 0;
    g_linalg_fallback_operations.clear();
    g_linalg_fallback_reasons.clear();
    cyxwiz::Tensor actual;
    {
        const cyxwiz::ScopedArrayFireFallbackPolicy compatibility(
            cyxwiz::ArrayFireFallbackPolicy::AllowNativeCpuFallback);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver observer(
            &CountLinalgFallback);
        actual = left.BatchMatMul(right);
    }
    REQUIRE(g_linalg_fallback_count == 1);
    REQUIRE(g_linalg_fallback_operations ==
            std::vector<std::string>{"Tensor::BatchMatMul"});
    REQUIRE(g_linalg_fallback_reasons ==
            std::vector<std::string>{"unsupported_dtype"});
    CheckTensor(actual, dtype, test_case.at("expected"));
}

TEST_CASE("Tensor linalg stays resident on every supported accelerator route",
          "[tensor][tensor_linalg][accelerator]") {
    const json fixture = LoadLinalgFixture();
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
        CheckResidentCasesOnActiveDevice(
            fixture, !device.supports_fp64_known || device.supports_fp64);
        ++exercised_routes;
    }
    if (exercised_routes == 0) {
        WARN("No CUDA or OpenCL route was available for Tensor linalg parity");
    }
    REQUIRE(cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(true).success);
}
#endif

TEST_CASE("Tensor linalg rejects invalid PyTorch shapes and mixed dtypes",
          "[tensor][tensor_linalg][invalid]") {
    const json fixture = LoadLinalgFixture();
    for (const json& test_case : fixture.at("invalid_dot")) {
        CAPTURE(test_case.at("name").get<std::string>());
        const cyxwiz::Tensor left(
            test_case.at("left_shape").get<std::vector<size_t>>(),
            cyxwiz::DataType::Float32);
        const cyxwiz::Tensor right(
            test_case.at("right_shape").get<std::vector<size_t>>(),
            cyxwiz::DataType::Float32);
        REQUIRE_THROWS_AS(left.Dot(right), std::runtime_error);
    }
    for (const json& test_case : fixture.at("invalid_batch_matmul")) {
        CAPTURE(test_case.at("name").get<std::string>());
        const cyxwiz::Tensor left(
            test_case.at("left_shape").get<std::vector<size_t>>(),
            cyxwiz::DataType::Float32);
        const cyxwiz::Tensor right(
            test_case.at("right_shape").get<std::vector<size_t>>(),
            cyxwiz::DataType::Float32);
        REQUIRE_THROWS_AS(left.BatchMatMul(right), std::runtime_error);
    }

    const cyxwiz::Tensor float_vector({2}, cyxwiz::DataType::Float32);
    const cyxwiz::Tensor int_vector({2}, cyxwiz::DataType::Int32);
    REQUIRE_THROWS_AS(float_vector.Dot(int_vector), std::runtime_error);
    const cyxwiz::Tensor float_batch({1, 1, 2}, cyxwiz::DataType::Float32);
    const cyxwiz::Tensor int_batch({1, 2, 1}, cyxwiz::DataType::Int32);
    REQUIRE_THROWS_AS(float_batch.BatchMatMul(int_batch), std::runtime_error);
}

TEST_CASE("Tensor linalg output overflow fails before compute",
          "[tensor][tensor_linalg][invalid]") {
    const size_t maximum = (std::numeric_limits<size_t>::max)();
    const cyxwiz::Tensor dot_left({maximum, 0}, cyxwiz::DataType::Float32);
    const cyxwiz::Tensor dot_right({maximum, 0}, cyxwiz::DataType::Float32);
    const cyxwiz::Tensor batch_left(
        {maximum, 1, 0}, cyxwiz::DataType::Float32);
    const cyxwiz::Tensor batch_right(
        {maximum, 0, 2}, cyxwiz::DataType::Float32);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    g_linalg_host_sync_count = 0;
    g_linalg_fallback_count = 0;
    const cyxwiz::ScopedArrayFireFallbackPolicy strict(
        cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
    const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
        &CountLinalgHostSync);
    const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
        &CountLinalgFallback);
#endif
    REQUIRE_THROWS_AS(dot_left.Dot(dot_right), std::overflow_error);
    REQUIRE_THROWS_AS(
        batch_left.BatchMatMul(batch_right), std::overflow_error);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    REQUIRE(g_linalg_host_sync_count == 0);
    REQUIRE(g_linalg_fallback_count == 0);
#endif
}
