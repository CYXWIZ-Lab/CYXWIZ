#include <catch2/catch_test_macros.hpp>

#include "algorithms/arrayfire_backend_utils.h"

#include <cyxwiz/device.h>
#include <cyxwiz/memory_manager.h>
#include <cyxwiz/tensor.h>

#include <arrayfire.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;

json LoadFactoryFixture() {
    std::ifstream stream(CYXWIZ_TRAINING_CORE_FIXTURE_PATH);
    REQUIRE(stream.is_open());
    json fixture;
    stream >> fixture;
    REQUIRE(fixture.value("schema_version", 0) == 1);
    REQUIRE(fixture.at("oracle").value("name", "") == "PyTorch");
    REQUIRE(fixture.at("oracle").value("version", "").rfind("2.10.0", 0) == 0);
    return fixture.at("cases").at("tensor_factories");
}

cyxwiz::DataType ParseDataType(const std::string& value) {
    if (value == "float32") return cyxwiz::DataType::Float32;
    if (value == "float64") return cyxwiz::DataType::Float64;
    if (value == "int32") return cyxwiz::DataType::Int32;
    if (value == "int64") return cyxwiz::DataType::Int64;
    if (value == "uint8") return cyxwiz::DataType::UInt8;
    throw std::runtime_error("unsupported factory fixture dtype: " + value);
}

cyxwiz::Tensor ApplyFactory(const std::string& operation,
                            const std::vector<size_t>& shape,
                            cyxwiz::DataType dtype,
                            uint64_t seed = 39039) {
    if (operation == "zeros") return cyxwiz::Tensor::Zeros(shape, dtype);
    if (operation == "ones") return cyxwiz::Tensor::Ones(shape, dtype);
    if (operation == "random") return cyxwiz::Tensor::Random(shape, dtype);
    if (operation == "random_seeded") {
        return cyxwiz::Tensor::RandomSeeded(shape, seed, dtype);
    }
    if (operation == "range_n") return cyxwiz::Tensor::RangeN(shape, dtype);
    throw std::runtime_error("unsupported factory operation: " + operation);
}

template <typename T>
void CheckTypedExact(const cyxwiz::Tensor& actual, const json& expected) {
    const auto values = expected.at("values").get<std::vector<T>>();
    REQUIRE(actual.NumElements() == values.size());
    if (values.empty()) return;
    const T* output = actual.ReadData<T>();
    for (size_t index = 0; index < values.size(); ++index) {
        REQUIRE(output[index] == values[index]);
    }
}

void CheckExact(const cyxwiz::Tensor& actual,
                cyxwiz::DataType dtype,
                const json& expected) {
    REQUIRE(actual.GetDataType() == dtype);
    REQUIRE(actual.Shape() == expected.at("shape").get<std::vector<size_t>>());
    switch (dtype) {
        case cyxwiz::DataType::Float32:
            CheckTypedExact<float>(actual, expected);
            return;
        case cyxwiz::DataType::Float64:
            CheckTypedExact<double>(actual, expected);
            return;
        case cyxwiz::DataType::Int32:
            CheckTypedExact<int32_t>(actual, expected);
            return;
        case cyxwiz::DataType::Int64:
            CheckTypedExact<int64_t>(actual, expected);
            return;
        case cyxwiz::DataType::UInt8:
            CheckTypedExact<uint8_t>(actual, expected);
            return;
    }
    throw std::runtime_error("unsupported exact factory dtype");
}

template <typename T>
bool TensorValuesEqual(const cyxwiz::Tensor& left,
                       const cyxwiz::Tensor& right) {
    REQUIRE(left.Shape() == right.Shape());
    REQUIRE(left.GetDataType() == right.GetDataType());
    if (left.NumElements() == 0) return true;
    return std::memcmp(
               left.ReadData<T>(), right.ReadData<T>(), left.NumBytes()) == 0;
}

bool TensorValuesEqual(const cyxwiz::Tensor& left,
                       const cyxwiz::Tensor& right) {
    switch (left.GetDataType()) {
        case cyxwiz::DataType::Float32:
            return TensorValuesEqual<float>(left, right);
        case cyxwiz::DataType::Float64:
            return TensorValuesEqual<double>(left, right);
        case cyxwiz::DataType::Int32:
            return TensorValuesEqual<int32_t>(left, right);
        case cyxwiz::DataType::Int64:
            return TensorValuesEqual<int64_t>(left, right);
        case cyxwiz::DataType::UInt8:
            return TensorValuesEqual<uint8_t>(left, right);
    }
    throw std::runtime_error("unsupported replay dtype");
}

template <typename T>
void CheckFloatingDistribution(const cyxwiz::Tensor& tensor,
                               const json& contract) {
    const T* values = tensor.ReadData<T>();
    double sum = 0.0;
    double squared_sum = 0.0;
    bool all_in_range = true;
    const double lower =
        contract.at("floating_range").at(0).get<double>();
    const double upper =
        contract.at("floating_range").at(1).get<double>();
    for (size_t index = 0; index < tensor.NumElements(); ++index) {
        const double value = static_cast<double>(values[index]);
        all_in_range = all_in_range && value >= lower && value < upper;
        sum += value;
        squared_sum += value * value;
    }
    REQUIRE(all_in_range);
    const double count = static_cast<double>(tensor.NumElements());
    const double mean = sum / count;
    const double variance = squared_sum / count - mean * mean;
    REQUIRE(std::abs(mean - contract.at("theoretical_mean").get<double>()) <=
            contract.at("tolerance").at("mean").get<double>());
    REQUIRE(std::abs(
                variance - contract.at("theoretical_variance").get<double>()) <=
            contract.at("tolerance").at("variance").get<double>());
}

template <typename T>
void CheckIntegerRange(const cyxwiz::Tensor& tensor,
                       int64_t lower,
                       int64_t upper_exclusive) {
    const T* values = tensor.ReadData<T>();
    bool all_in_range = true;
    for (size_t index = 0; index < tensor.NumElements(); ++index) {
        const int64_t value = static_cast<int64_t>(values[index]);
        all_in_range =
            all_in_range && value >= lower && value < upper_exclusive;
    }
    REQUIRE(all_in_range);
}

void CheckRandomContract(const json& fixture) {
    const json& contract = fixture.at("random_contract");
    const size_t count = contract.at("sample_count").get<size_t>();
    for (const auto dtype : {
             cyxwiz::DataType::Float32,
             cyxwiz::DataType::Float64,
             cyxwiz::DataType::Int32,
             cyxwiz::DataType::Int64,
             cyxwiz::DataType::UInt8}) {
        CAPTURE(static_cast<int>(dtype));
        const cyxwiz::Tensor random = cyxwiz::Tensor::Random({count}, dtype);
        const cyxwiz::Tensor seeded = cyxwiz::Tensor::RandomSeeded(
            {count}, fixture.at("seeded_contract").at("same_seed").get<uint64_t>(), dtype);
        if (dtype == cyxwiz::DataType::Float32) {
            CheckFloatingDistribution<float>(random, contract);
            CheckFloatingDistribution<float>(seeded, contract);
        } else if (dtype == cyxwiz::DataType::Float64) {
            CheckFloatingDistribution<double>(random, contract);
            CheckFloatingDistribution<double>(seeded, contract);
        } else if (dtype == cyxwiz::DataType::Int32) {
            CheckIntegerRange<int32_t>(random, 0, 100);
            CheckIntegerRange<int32_t>(seeded, 0, 100);
        } else if (dtype == cyxwiz::DataType::Int64) {
            CheckIntegerRange<int64_t>(random, 0, 100);
            CheckIntegerRange<int64_t>(seeded, 0, 100);
        } else {
            CheckIntegerRange<uint8_t>(random, 0, 256);
            CheckIntegerRange<uint8_t>(seeded, 0, 256);
        }
    }
}

void CheckSeedReplay(const json& fixture) {
    const uint64_t same_seed =
        fixture.at("seeded_contract").at("same_seed").get<uint64_t>();
    const uint64_t different_seed =
        fixture.at("seeded_contract").at("different_seed").get<uint64_t>();
    for (const auto dtype : {
             cyxwiz::DataType::Float32,
             cyxwiz::DataType::Float64,
             cyxwiz::DataType::Int32,
             cyxwiz::DataType::Int64,
             cyxwiz::DataType::UInt8}) {
        CAPTURE(static_cast<int>(dtype));
        const cyxwiz::Tensor first =
            cyxwiz::Tensor::RandomSeeded({2, 1, 257}, same_seed, dtype);
        const cyxwiz::Tensor second =
            cyxwiz::Tensor::RandomSeeded({2, 1, 257}, same_seed, dtype);
        const cyxwiz::Tensor different =
            cyxwiz::Tensor::RandomSeeded({2, 1, 257}, different_seed, dtype);
        REQUIRE(first.Shape() == std::vector<size_t>{2, 1, 257});
        REQUIRE(TensorValuesEqual(first, second));
        REQUIRE_FALSE(TensorValuesEqual(first, different));
    }
}

size_t g_factory_host_sync_count = 0;
size_t g_factory_fallback_count = 0;

void CountFactoryHostSync(const cyxwiz::ArrayFireHostSyncEvent&) {
    ++g_factory_host_sync_count;
}

void CountFactoryFallback(const cyxwiz::ArrayFireNativeCpuFallbackEvent&) {
    ++g_factory_fallback_count;
}

void CheckFactoryCasesOnActiveDevice(const json& fixture, bool supports_fp64) {
    for (const json& test_case : fixture.at("exact_cases")) {
        const auto dtype =
            ParseDataType(test_case.at("dtype").get<std::string>());
        if (!supports_fp64 && dtype == cyxwiz::DataType::Float64) continue;
        const auto shape = test_case.at("shape").get<std::vector<size_t>>();
        const std::string operation =
            test_case.at("operation").get<std::string>();
        CAPTURE(test_case.at("name").get<std::string>());
        g_factory_host_sync_count = 0;
        g_factory_fallback_count = 0;
        cyxwiz::Tensor actual;
        {
            const cyxwiz::ScopedArrayFireFallbackPolicy strict(
                cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
            const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
                &CountFactoryHostSync);
            const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
                &CountFactoryFallback);
            actual = ApplyFactory(operation, shape, dtype);
            if (actual.NumElements() != 0) actual.GetSemanticArray().eval();
        }
        REQUIRE(g_factory_host_sync_count == 0);
        REQUIRE(g_factory_fallback_count == 0);
        CheckExact(actual, dtype, test_case.at("expected"));
    }

    for (const auto dtype : {
             cyxwiz::DataType::Float32,
             cyxwiz::DataType::Float64,
             cyxwiz::DataType::Int32,
             cyxwiz::DataType::Int64,
             cyxwiz::DataType::UInt8}) {
        if (!supports_fp64 && dtype == cyxwiz::DataType::Float64) continue;
        for (const std::string operation : {"random", "random_seeded"}) {
            CAPTURE(operation, static_cast<int>(dtype));
            g_factory_host_sync_count = 0;
            g_factory_fallback_count = 0;
            cyxwiz::Tensor actual;
            cyxwiz::Tensor replay;
            cyxwiz::Tensor different;
            {
                const cyxwiz::ScopedArrayFireFallbackPolicy strict(
                    cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
                const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
                    &CountFactoryHostSync);
                const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
                    &CountFactoryFallback);
                actual = ApplyFactory(operation, {2, 1, 257}, dtype);
                actual.GetSemanticArray().eval();
                if (operation == "random_seeded") {
                    replay = ApplyFactory(
                        operation, {2, 1, 257}, dtype, 39039);
                    different = ApplyFactory(
                        operation, {2, 1, 257}, dtype, 39040);
                    replay.GetSemanticArray().eval();
                    different.GetSemanticArray().eval();
                }
            }
            REQUIRE(actual.Shape() == std::vector<size_t>{2, 1, 257});
            REQUIRE(g_factory_host_sync_count == 0);
            REQUIRE(g_factory_fallback_count == 0);
            if (operation == "random_seeded") {
                REQUIRE(TensorValuesEqual(actual, replay));
                REQUIRE_FALSE(TensorValuesEqual(actual, different));
            }
        }
    }
}

} // namespace

TEST_CASE("Tensor exact factories match PyTorch shape dtype and values",
          "[tensor][tensor_factory][pytorch]") {
    const json fixture = LoadFactoryFixture();
    const auto activation =
        cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(true);
    REQUIRE(activation.success);
    REQUIRE(activation.execution_validated);
    CheckFactoryCasesOnActiveDevice(fixture, true);
    CheckRandomContract(fixture);
    CheckSeedReplay(fixture);
}

TEST_CASE("Tensor seeded and range factories are device-resident",
          "[tensor][tensor_factory][arrayfire][residency]") {
    REQUIRE(cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(true).success);
    const size_t before = cyxwiz::MemoryManager::GetAllocatedBytes();
    {
        const cyxwiz::Tensor seeded = cyxwiz::Tensor::RandomSeeded(
            {1024}, 39039, cyxwiz::DataType::Float32);
        const cyxwiz::Tensor range = cyxwiz::Tensor::RangeN(
            {1024}, cyxwiz::DataType::Float32);
        REQUIRE(cyxwiz::MemoryManager::GetAllocatedBytes() == before);
        seeded.GetSemanticArray().eval();
        range.GetSemanticArray().eval();
    }
    REQUIRE(cyxwiz::MemoryManager::GetAllocatedBytes() == before);
}

TEST_CASE("Tensor factories stay resident on every accelerator",
          "[tensor][tensor_factory][accelerator]") {
    const json fixture = LoadFactoryFixture();
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
        CheckFactoryCasesOnActiveDevice(
            fixture, !device.supports_fp64_known || device.supports_fp64);
        ++exercised_routes;
    }
    if (exercised_routes == 0) {
        WARN("No CUDA or OpenCL route was available for Tensor factory parity");
    }
    REQUIRE(cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(true).success);
}

// TOFIX140 F4: ranks above 4 live on the device as row-major flat arrays, so
// rank-five factories run on ArrayFire with no native fallback, even strict.
TEST_CASE("Tensor rank-five factories run on the device",
          "[tensor][tensor_factory][arrayfire]") {
    REQUIRE(cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(true).success);
    const std::vector<size_t> shape{2, 1, 3, 1, 4};
    g_factory_host_sync_count = 0;
    g_factory_fallback_count = 0;
    std::vector<cyxwiz::Tensor> outputs;
    {
        const cyxwiz::ScopedArrayFireFallbackPolicy strict(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
            &CountFactoryHostSync);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CountFactoryFallback);
        for (const std::string operation : {"zeros", "ones", "range_n", "random", "random_seeded"}) {
            outputs.push_back(ApplyFactory(operation, shape, cyxwiz::DataType::Int32));
            REQUIRE(outputs.back().Shape() == shape);
        }
    }
    REQUIRE(g_factory_host_sync_count == 0);
    REQUIRE(g_factory_fallback_count == 0);

    const auto* zeros = outputs[0].ReadData<int32_t>();
    const auto* ones = outputs[1].ReadData<int32_t>();
    const auto* range = outputs[2].ReadData<int32_t>();
    const auto* random = outputs[3].ReadData<int32_t>();
    for (size_t i = 0; i < 24; ++i) {
        REQUIRE(zeros[i] == 0);
        REQUIRE(ones[i] == 1);
        REQUIRE(range[i] == static_cast<int32_t>(i));  // row-major storage order
        REQUIRE(random[i] >= 0);
        REQUIRE(random[i] < 100);
    }
    // Draws land in storage order, so a seeded rank-five tensor holds the same
    // values as a flat one of the same seed and count.
    const cyxwiz::Tensor flat = cyxwiz::Tensor::RandomSeeded({24}, 7, cyxwiz::DataType::Float32);
    const cyxwiz::Tensor folded = cyxwiz::Tensor::RandomSeeded(shape, 7, cyxwiz::DataType::Float32);
    const auto* flat_values = flat.ReadData<float>();
    const auto* folded_values = folded.ReadData<float>();
    for (size_t i = 0; i < 24; ++i) REQUIRE(folded_values[i] == flat_values[i]);
    const cyxwiz::Tensor matrix = cyxwiz::Tensor::RandomSeeded({4, 6}, 7, cyxwiz::DataType::Float32);
    const auto* matrix_values = matrix.ReadData<float>();
    for (size_t i = 0; i < 24; ++i) REQUIRE(matrix_values[i] == flat_values[i]);
}

TEST_CASE("Tensor empty factories require no compute or fallback",
          "[tensor][tensor_factory][arrayfire]") {
    const std::vector<size_t> empty_rank_five{1, 0, 1, 1, 1};
    g_factory_host_sync_count = 0;
    g_factory_fallback_count = 0;
    {
        const cyxwiz::ScopedArrayFireFallbackPolicy strict(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
            &CountFactoryHostSync);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CountFactoryFallback);
        for (const std::string operation : {
                 "zeros", "ones", "random", "random_seeded", "range_n"}) {
            const cyxwiz::Tensor output = ApplyFactory(
                operation, empty_rank_five, cyxwiz::DataType::Float32);
            REQUIRE(output.Shape() == empty_rank_five);
            REQUIRE(output.NumElements() == 0);
        }
    }
    REQUIRE(g_factory_host_sync_count == 0);
    REQUIRE(g_factory_fallback_count == 0);
}

TEST_CASE("Tensor factory shape overflow fails before compute",
          "[tensor][tensor_factory][invalid]") {
    const std::vector<size_t> shape{
        (std::numeric_limits<size_t>::max)(), 2};
    g_factory_host_sync_count = 0;
    g_factory_fallback_count = 0;
    const cyxwiz::ScopedArrayFireFallbackPolicy strict(
        cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
    const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
        &CountFactoryHostSync);
    const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
        &CountFactoryFallback);
    for (const std::string operation : {
             "zeros", "ones", "random", "random_seeded", "range_n"}) {
        REQUIRE_THROWS_AS(
            ApplyFactory(operation, shape, cyxwiz::DataType::Float32),
            std::overflow_error);
    }
    REQUIRE(g_factory_host_sync_count == 0);
    REQUIRE(g_factory_fallback_count == 0);
}
