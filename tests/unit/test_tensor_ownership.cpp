#include <catch2/catch_test_macros.hpp>

#include "algorithms/arrayfire_backend_utils.h"

#include <cyxwiz/device.h>
#include <cyxwiz/memory_manager.h>
#include <cyxwiz/tensor.h>

#include <array>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace {

template<typename T>
struct TensorType;

template<>
struct TensorType<float> {
    static constexpr cyxwiz::DataType value = cyxwiz::DataType::Float32;
};

template<>
struct TensorType<double> {
    static constexpr cyxwiz::DataType value = cyxwiz::DataType::Float64;
};

template<>
struct TensorType<int32_t> {
    static constexpr cyxwiz::DataType value = cyxwiz::DataType::Int32;
};

template<>
struct TensorType<int64_t> {
    static constexpr cyxwiz::DataType value = cyxwiz::DataType::Int64;
};

template<>
struct TensorType<uint8_t> {
    static constexpr cyxwiz::DataType value = cyxwiz::DataType::UInt8;
};

template<typename T>
std::array<T, 6> Values() {
    return {
        static_cast<T>(1),
        static_cast<T>(2),
        static_cast<T>(3),
        static_cast<T>(4),
        static_cast<T>(5),
        static_cast<T>(6),
    };
}

template<typename T>
void CheckHostOwnership() {
    const size_t before = cyxwiz::MemoryManager::GetAllocatedBytes();
    {
        constexpr cyxwiz::DataType dtype = TensorType<T>::value;
        const auto values = Values<T>();
        cyxwiz::Tensor zeroed({2, 3}, dtype);
        cyxwiz::Tensor original({2, 3}, values.data(), dtype);

        REQUIRE(zeroed.Shape() == std::vector<size_t>{2, 3});
        REQUIRE(zeroed.NumDimensions() == 2);
        REQUIRE(zeroed.NumElements() == 6);
        REQUIRE(zeroed.NumBytes() == 6 * sizeof(T));
        REQUIRE(zeroed.GetDataType() == dtype);
        for (size_t index = 0; index < 6; ++index) {
            REQUIRE(zeroed.ReadData<T>()[index] == static_cast<T>(0));
            REQUIRE(original.ReadData<T>()[index] == values[index]);
        }

        cyxwiz::Tensor copied(original);
        cyxwiz::Tensor cloned = original.Clone();
        cyxwiz::Tensor assigned;
        assigned = original;
        cyxwiz::Tensor moved(std::move(copied));
        cyxwiz::Tensor move_assigned;
        move_assigned = std::move(assigned);

        cloned.MutableData<T>()[0] = static_cast<T>(42);
        REQUIRE(original.ReadData<T>()[0] == values[0]);
        REQUIRE(cloned.ReadData<T>()[0] == static_cast<T>(42));
        REQUIRE(moved.ReadData<T>()[5] == values[5]);
        REQUIRE(move_assigned.ReadData<T>()[4] == values[4]);

        copied = cyxwiz::Tensor({1}, dtype);
        assigned = cyxwiz::Tensor({1}, dtype);
        REQUIRE(copied.NumElements() == 1);
        REQUIRE(assigned.NumElements() == 1);
    }
    REQUIRE(cyxwiz::MemoryManager::GetAllocatedBytes() == before);
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
int g_ownership_host_sync_count = 0;
uint64_t g_ownership_host_sync_bytes = 0;

void CaptureOwnershipHostSync(
    const cyxwiz::ArrayFireHostSyncEvent& event) {
    ++g_ownership_host_sync_count;
    g_ownership_host_sync_bytes += event.bytes;
}

template<typename T>
void CheckDeviceOwnership() {
    constexpr cyxwiz::DataType dtype = TensorType<T>::value;
    const auto values = Values<T>();
    cyxwiz::Tensor host({2, 3}, values.data(), dtype);
    cyxwiz::Tensor original = cyxwiz::Tensor::FromSemanticArray(
        host.GetSemanticArray(), {2, 3});
    const size_t before_ownership =
        cyxwiz::MemoryManager::GetAllocatedBytes();

    g_ownership_host_sync_count = 0;
    g_ownership_host_sync_bytes = 0;
    cyxwiz::Tensor cloned;
    cyxwiz::Tensor moved;
    cyxwiz::Tensor move_assigned;
    {
        const cyxwiz::ScopedArrayFireHostSyncObserver observer(
            &CaptureOwnershipHostSync);
        cyxwiz::Tensor copied(original);
        cloned = original.Clone();
        cyxwiz::Tensor assigned;
        assigned = original;
        moved = std::move(copied);
        move_assigned = std::move(assigned);
    }

    REQUIRE(g_ownership_host_sync_count == 0);
    REQUIRE(g_ownership_host_sync_bytes == 0);
    REQUIRE(cyxwiz::MemoryManager::GetAllocatedBytes() == before_ownership);
    REQUIRE(cloned.Shape() == std::vector<size_t>{2, 3});
    REQUIRE(moved.Shape() == std::vector<size_t>{2, 3});
    REQUIRE(move_assigned.Shape() == std::vector<size_t>{2, 3});
    REQUIRE(cloned.GetDataType() == dtype);
    REQUIRE(moved.GetDataType() == dtype);
    REQUIRE(move_assigned.GetDataType() == dtype);

    g_ownership_host_sync_count = 0;
    g_ownership_host_sync_bytes = 0;
    {
        const cyxwiz::ScopedArrayFireHostSyncObserver observer(
            &CaptureOwnershipHostSync);
        cloned.MutableData<T>()[0] = static_cast<T>(42);
        REQUIRE(original.ReadData<T>()[0] == values[0]);
    }
    REQUIRE(g_ownership_host_sync_count == 2);
    REQUIRE(g_ownership_host_sync_bytes == 2 * original.NumBytes());
    REQUIRE(cloned.ReadData<T>()[0] == static_cast<T>(42));
}

template<typename T>
void CheckExplicitHostAccess() {
    constexpr cyxwiz::DataType dtype = TensorType<T>::value;
    const auto values = Values<T>();
    cyxwiz::Tensor host({2, 3}, values.data(), dtype);

    cyxwiz::Tensor readable = cyxwiz::Tensor::FromSemanticArray(
        host.GetSemanticArray(), {2, 3});
    const size_t before_read = cyxwiz::MemoryManager::GetAllocatedBytes();
    g_ownership_host_sync_count = 0;
    g_ownership_host_sync_bytes = 0;
    {
        const cyxwiz::ScopedArrayFireHostSyncObserver observer(
            &CaptureOwnershipHostSync);
        REQUIRE(readable.ReadData<T>()[3] == values[3]);
        REQUIRE(readable.GetSemanticArray().elements() == 6);
        REQUIRE(readable.ReadData<T>()[4] == values[4]);
    }
    REQUIRE(g_ownership_host_sync_count == 1);
    REQUIRE(g_ownership_host_sync_bytes == readable.NumBytes());
    REQUIRE(cyxwiz::MemoryManager::GetAllocatedBytes() ==
            before_read + readable.NumBytes());

    cyxwiz::Tensor writable = cyxwiz::Tensor::FromSemanticArray(
        host.GetSemanticArray(), {2, 3});
    g_ownership_host_sync_count = 0;
    g_ownership_host_sync_bytes = 0;
    af::array rebuilt_device;
    {
        const cyxwiz::ScopedArrayFireHostSyncObserver observer(
            &CaptureOwnershipHostSync);
        writable.MutableData<T>()[2] = static_cast<T>(42);
        rebuilt_device = writable.GetSemanticArray();
        rebuilt_device.eval();
    }
    REQUIRE(g_ownership_host_sync_count == 1);
    REQUIRE(g_ownership_host_sync_bytes == writable.NumBytes());

    cyxwiz::Tensor roundtrip = cyxwiz::Tensor::FromSemanticArray(
        rebuilt_device, {2, 3});
    REQUIRE(roundtrip.ReadData<T>()[2] == static_cast<T>(42));
    REQUIRE(host.ReadData<T>()[2] == values[2]);
}

template<typename T>
void CheckSemanticLayoutRoundTrip(const std::vector<size_t>& shape) {
    constexpr cyxwiz::DataType dtype = TensorType<T>::value;
    size_t element_count = 1;
    for (size_t dim : shape) {
        element_count *= dim;
    }
    std::vector<T> values(element_count);
    for (size_t index = 0; index < element_count; ++index) {
        values[index] = static_cast<T>(index + 1);
    }

    cyxwiz::Tensor host(shape, values.data(), dtype);
    cyxwiz::Tensor device_only;
    device_only.SetFromArray(host.GetArray());

    g_ownership_host_sync_count = 0;
    g_ownership_host_sync_bytes = 0;
    cyxwiz::Tensor roundtrip;
    {
        const cyxwiz::ScopedArrayFireHostSyncObserver observer(
            &CaptureOwnershipHostSync);
        af::array semantic = device_only.GetSemanticArray();
        for (size_t dim = 0; dim < shape.size(); ++dim) {
            REQUIRE(semantic.dims(static_cast<unsigned int>(dim)) ==
                    static_cast<dim_t>(shape[dim]));
        }
        roundtrip = cyxwiz::Tensor::FromSemanticArray(semantic, shape);
        af::array native = roundtrip.GetArray();
        native.eval();
    }

    REQUIRE(g_ownership_host_sync_count == 0);
    REQUIRE(g_ownership_host_sync_bytes == 0);
    REQUIRE(roundtrip.Shape() == shape);
    const T* output = roundtrip.ReadData<T>();
    for (size_t index = 0; index < element_count; ++index) {
        REQUIRE(output[index] == values[index]);
    }
}

template<typename T>
void CheckSemanticLayoutMatrix() {
    CheckSemanticLayoutRoundTrip<T>({6});
    CheckSemanticLayoutRoundTrip<T>({2, 3});
    CheckSemanticLayoutRoundTrip<T>({2, 2, 3});
    CheckSemanticLayoutRoundTrip<T>({2, 2, 2, 3});
}
#endif

} // namespace

TEST_CASE("Tensor ownership covers every supported dtype",
          "[tensor][tensor_ownership]") {
    CheckHostOwnership<float>();
    CheckHostOwnership<double>();
    CheckHostOwnership<int32_t>();
    CheckHostOwnership<int64_t>();
    CheckHostOwnership<uint8_t>();
}

TEST_CASE("Tensor zero-size and overflow metadata follows checked semantics",
          "[tensor][tensor_ownership]") {
    const size_t maximum = (std::numeric_limits<size_t>::max)();
    const cyxwiz::Tensor zero_after_large_dims(
        {maximum, 2, 0}, cyxwiz::DataType::UInt8);
    REQUIRE(zero_after_large_dims.NumElements() == 0);
    REQUIRE(zero_after_large_dims.NumBytes() == 0);
    REQUIRE(zero_after_large_dims.ReadData() == nullptr);

    REQUIRE_THROWS_AS(
        cyxwiz::Tensor({maximum, 2}, cyxwiz::DataType::UInt8),
        std::overflow_error);
    REQUIRE_THROWS_AS(
        cyxwiz::Tensor(
            {(maximum / sizeof(double)) + 1},
            cyxwiz::DataType::Float64),
        std::overflow_error);
    REQUIRE_THROWS_AS(
        cyxwiz::Tensor(
            {1}, static_cast<cyxwiz::DataType>(999)),
        std::runtime_error);
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
TEST_CASE("Tensor copy move and clone preserve device-only dtype state",
          "[tensor][tensor_ownership][arrayfire][host_sync]") {
    const auto activation =
        cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(true);
    REQUIRE(activation.success);
    REQUIRE(activation.execution_validated);

    CheckDeviceOwnership<float>();
    CheckDeviceOwnership<double>();
    CheckDeviceOwnership<int32_t>();
    CheckDeviceOwnership<int64_t>();
    CheckDeviceOwnership<uint8_t>();
}

TEST_CASE("Tensor explicit host access preserves reads and rebuilds mutations",
          "[tensor][tensor_host_access][arrayfire][host_sync]") {
    const auto activation =
        cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(true);
    REQUIRE(activation.success);
    REQUIRE(activation.execution_validated);

    CheckExplicitHostAccess<float>();
    CheckExplicitHostAccess<double>();
    CheckExplicitHostAccess<int32_t>();
    CheckExplicitHostAccess<int64_t>();
    CheckExplicitHostAccess<uint8_t>();
}

TEST_CASE("Tensor semantic layouts round-trip all dtypes without host sync",
          "[tensor][tensor_layout][arrayfire][host_sync]") {
    const auto activation =
        cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(true);
    REQUIRE(activation.success);
    REQUIRE(activation.execution_validated);

    CheckSemanticLayoutMatrix<float>();
    CheckSemanticLayoutMatrix<double>();
    CheckSemanticLayoutMatrix<int32_t>();
    CheckSemanticLayoutMatrix<int64_t>();
    CheckSemanticLayoutMatrix<uint8_t>();
}
#endif
