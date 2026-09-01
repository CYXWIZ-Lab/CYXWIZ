#include <catch2/catch_test_macros.hpp>

#include "algorithms/arrayfire_backend_utils.h"

#include <cyxwiz/device.h>
#include <cyxwiz/tensor.h>

#include <nlohmann/json.hpp>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

#include <cstdint>
#include <fstream>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;

json LoadIndexingFixture() {
    std::ifstream stream(CYXWIZ_TRAINING_CORE_FIXTURE_PATH);
    REQUIRE(stream.is_open());

    json fixture;
    stream >> fixture;
    REQUIRE(fixture.value("schema_version", 0) == 1);
    REQUIRE(fixture.at("oracle").value("name", "") == "PyTorch");
    const std::string oracle_version =
        fixture.at("oracle").value("version", "");
    REQUIRE(oracle_version.rfind("2.10.0", 0) == 0);
    return fixture.at("cases").at("tensor_indexing_f32");
}

std::vector<size_t> ShapeFromFixture(const json& value) {
    return value.at("shape").get<std::vector<size_t>>();
}

cyxwiz::Tensor TensorFromFixture(const json& value) {
    const auto shape = ShapeFromFixture(value);
    const auto values = value.at("values").get<std::vector<float>>();
    return cyxwiz::Tensor(shape, values.data(), cyxwiz::DataType::Float32);
}

void CheckTensor(const cyxwiz::Tensor& actual, const json& expected) {
    REQUIRE(actual.Shape() == ShapeFromFixture(expected));
    const auto expected_values =
        expected.at("values").get<std::vector<float>>();
    REQUIRE(actual.NumElements() == expected_values.size());
    const float* values = actual.ReadData<float>();
    for (size_t index = 0; index < expected_values.size(); ++index) {
        REQUIRE(values[index] == expected_values[index]);
    }
}

std::vector<size_t> RowMajorStrides(const std::vector<size_t>& shape) {
    std::vector<size_t> strides(shape.size(), 1);
    size_t stride = 1;
    for (size_t axis = shape.size(); axis > 0; --axis) {
        strides[axis - 1] = stride;
        stride *= shape[axis - 1];
    }
    return strides;
}

std::vector<size_t> GatherSourceIndices(
    const std::vector<size_t>& shape,
    size_t axis,
    const std::vector<size_t>& selected) {
    std::vector<size_t> output_shape = shape;
    output_shape[axis] = selected.size();
    const auto input_strides = RowMajorStrides(shape);
    const auto output_strides = RowMajorStrides(output_shape);
    const size_t element_count =
        std::accumulate(output_shape.begin(), output_shape.end(), size_t{1},
                        std::multiplies<size_t>());
    std::vector<size_t> source_indices(element_count, 0);
    std::vector<size_t> output_index(output_shape.size(), 0);

    for (size_t linear = 0; linear < element_count; ++linear) {
        size_t remainder = linear;
        for (size_t dim = 0; dim < output_shape.size(); ++dim) {
            output_index[dim] = remainder / output_strides[dim];
            remainder %= output_strides[dim];
        }
        output_index[axis] = selected[output_index[axis]];
        for (size_t dim = 0; dim < shape.size(); ++dim) {
            source_indices[linear] += output_index[dim] * input_strides[dim];
        }
    }
    return source_indices;
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
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

size_t g_indexing_host_sync_count = 0;
size_t g_indexing_fallback_count = 0;

void CountIndexingHostSync(const cyxwiz::ArrayFireHostSyncEvent&) {
    ++g_indexing_host_sync_count;
}

void CountIndexingFallback(
    const cyxwiz::ArrayFireNativeCpuFallbackEvent&) {
    ++g_indexing_fallback_count;
}

template<typename T, typename Operation>
void CheckDeviceGather(const std::vector<size_t>& shape,
                       size_t axis,
                       const std::vector<size_t>& selected,
                       Operation&& operation) {
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

    g_indexing_host_sync_count = 0;
    g_indexing_fallback_count = 0;
    cyxwiz::Tensor output;
    {
        const cyxwiz::ScopedArrayFireFallbackPolicy strict(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
            &CountIndexingHostSync);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CountIndexingFallback);
        output = operation(device_only);
        output.GetSemanticArray().eval();
    }

    REQUIRE(g_indexing_host_sync_count == 0);
    REQUIRE(g_indexing_fallback_count == 0);
    const auto source_indices = GatherSourceIndices(shape, axis, selected);
    const T* output_values = output.ReadData<T>();
    for (size_t index = 0; index < source_indices.size(); ++index) {
        REQUIRE(output_values[index] == values[source_indices[index]]);
    }
}

template<typename T>
void CheckDeviceIndexingMatrix() {
    CheckDeviceGather<T>(
        {6}, 0, {1, 3, 5},
        [](const cyxwiz::Tensor& input) { return input.Slice(0, 1, -1, 2); });
    CheckDeviceGather<T>(
        {3, 4}, 1, {1, 3},
        [](const cyxwiz::Tensor& input) { return input.Slice(-1, -3, -1, 2); });
    CheckDeviceGather<T>(
        {2, 3, 4}, 1, {0, 2},
        [](const cyxwiz::Tensor& input) { return input.Slice(1, 0, -1, 2); });
    CheckDeviceGather<T>(
        {2, 2, 3, 2}, 2, {1, 2},
        [](const cyxwiz::Tensor& input) { return input.Slice(2, 1, 3); });

    CheckDeviceGather<T>(
        {6}, 0, {5, 1, 1, 0},
        [](const cyxwiz::Tensor& input) {
            return input.IndexSelect(0, {5, 1, 1, 0});
        });
    CheckDeviceGather<T>(
        {3, 4}, 1, {3, 1, 0},
        [](const cyxwiz::Tensor& input) {
            return input.IndexSelect(-1, {3, 1, -4});
        });
    CheckDeviceGather<T>(
        {2, 3, 4}, 1, {2, 0},
        [](const cyxwiz::Tensor& input) {
            return input.IndexSelect(1, {2, 0});
        });
    CheckDeviceGather<T>(
        {2, 2, 3, 2}, 2, {2, 0, 2},
        [](const cyxwiz::Tensor& input) {
            return input.IndexSelect(2, {2, 0, 2});
        });
}

template<typename T>
void CheckScalarHostBoundary() {
    constexpr cyxwiz::DataType dtype = TensorType<T>::value;
    const std::vector<T> values = {
        static_cast<T>(1), static_cast<T>(2),
        static_cast<T>(3), static_cast<T>(4)};
    cyxwiz::Tensor host({2, 2}, values.data(), dtype);
    cyxwiz::Tensor device_only;
    device_only.SetFromArray(host.GetArray());

    g_indexing_host_sync_count = 0;
    {
        const cyxwiz::ScopedArrayFireHostSyncObserver observer(
            &CountIndexingHostSync);
        REQUIRE(device_only.At(1, 0) == 3.0f);
        REQUIRE(g_indexing_host_sync_count == 1);
        device_only.Set(0, 1, 9.0f);
        REQUIRE(g_indexing_host_sync_count == 1);
        device_only.GetSemanticArray().eval();
        REQUIRE(g_indexing_host_sync_count == 1);
    }
    REQUIRE(device_only.ReadData<T>()[1] == static_cast<T>(9));
}
#endif

} // namespace

TEST_CASE("Tensor slice and index select match generated PyTorch matrices",
          "[tensor][tensor_indexing][pytorch]") {
    const json fixture = LoadIndexingFixture();
    for (const json& test_case : fixture.at("slice")) {
        CAPTURE(test_case.at("name").get<std::string>());
        const cyxwiz::Tensor input = TensorFromFixture(test_case.at("input"));
        CheckTensor(
            input.Slice(test_case.at("dim").get<int>(),
                        test_case.at("start").get<int>(),
                        test_case.at("end").get<int>(),
                        test_case.at("step").get<int>()),
            test_case.at("expected"));
    }
    for (const json& test_case : fixture.at("index_select")) {
        CAPTURE(test_case.at("name").get<std::string>());
        const cyxwiz::Tensor input = TensorFromFixture(test_case.at("input"));
        CheckTensor(
            input.IndexSelect(
                test_case.at("dim").get<int>(),
                test_case.at("indices").get<std::vector<int>>()),
            test_case.at("expected"));
    }
}

TEST_CASE("Tensor indexing rejects invalid ranges before compute",
          "[tensor][tensor_indexing][errors]") {
    const cyxwiz::Tensor input =
        cyxwiz::Tensor::RangeN({2, 3, 4}, cyxwiz::DataType::Float32);
    const cyxwiz::Tensor empty_axis({2, 0, 3}, cyxwiz::DataType::Float32);

    REQUIRE_THROWS_AS(input.Slice(3, 0), std::runtime_error);
    REQUIRE_THROWS_AS(input.Slice(1, 0, -1, 0), std::runtime_error);
    REQUIRE_THROWS_AS(input.Slice(1, 0, -1, -1), std::runtime_error);
    REQUIRE_THROWS_AS(input.IndexSelect(3, {0}), std::runtime_error);
    REQUIRE_THROWS_AS(input.IndexSelect(1, {3}), std::out_of_range);
    REQUIRE_THROWS_AS(input.IndexSelect(1, {-4}), std::out_of_range);
    REQUIRE_THROWS_AS(empty_axis.IndexSelect(1, {0}), std::out_of_range);
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
TEST_CASE("Tensor gathers remain device resident for every dtype",
          "[tensor][tensor_indexing][arrayfire][host_sync]") {
    const auto activation =
        cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(true);
    REQUIRE(activation.success);
    REQUIRE(activation.execution_validated);

    CheckDeviceIndexingMatrix<float>();
    CheckDeviceIndexingMatrix<double>();
    CheckDeviceIndexingMatrix<int32_t>();
    CheckDeviceIndexingMatrix<int64_t>();
    CheckDeviceIndexingMatrix<uint8_t>();
}

TEST_CASE("Tensor empty gathers are metadata-only",
          "[tensor][tensor_indexing][arrayfire][empty]") {
    const auto activation =
        cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(true);
    REQUIRE(activation.success);

    const cyxwiz::Tensor input =
        cyxwiz::Tensor::RangeN({3, 4}, cyxwiz::DataType::Float32);
    g_indexing_host_sync_count = 0;
    g_indexing_fallback_count = 0;
    {
        const cyxwiz::ScopedArrayFireFallbackPolicy strict(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
            &CountIndexingHostSync);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CountIndexingFallback);
        const cyxwiz::Tensor slice = input.Slice(1, 3, 1);
        const cyxwiz::Tensor selected = input.IndexSelect(1, {});
        REQUIRE(slice.Shape() == std::vector<size_t>{3, 0});
        REQUIRE(selected.Shape() == std::vector<size_t>{3, 0});
    }
    REQUIRE(g_indexing_host_sync_count == 0);
    REQUIRE(g_indexing_fallback_count == 0);
}

TEST_CASE("Tensor rank-five indexing fallback is explicit",
          "[tensor][tensor_indexing][arrayfire][fallback]") {
    const cyxwiz::Tensor input =
        cyxwiz::Tensor::RangeN({1, 2, 1, 2, 2}, cyxwiz::DataType::Float32);

    g_indexing_fallback_count = 0;
    {
        const cyxwiz::ScopedArrayFireFallbackPolicy compatibility(
            cyxwiz::ArrayFireFallbackPolicy::AllowNativeCpuFallback);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver observer(
            &CountIndexingFallback);
        REQUIRE(input.Slice(3, 0, -1).Shape() == input.Shape());
        REQUIRE(input.IndexSelect(4, {1, 0}).Shape() == input.Shape());
    }
    REQUIRE(g_indexing_fallback_count == 2);

    g_indexing_fallback_count = 0;
    {
        const cyxwiz::ScopedArrayFireFallbackPolicy strict(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver observer(
            &CountIndexingFallback);
        REQUIRE_THROWS_AS(input.Slice(3, 0, -1), std::runtime_error);
        REQUIRE_THROWS_AS(input.IndexSelect(4, {1, 0}), std::runtime_error);
    }
    REQUIRE(g_indexing_fallback_count == 2);
}

TEST_CASE("Tensor scalar indexing uses an explicit bounded host boundary",
          "[tensor][tensor_indexing][arrayfire][host_boundary]") {
    const auto activation =
        cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(true);
    REQUIRE(activation.success);

    CheckScalarHostBoundary<float>();
    CheckScalarHostBoundary<double>();
    CheckScalarHostBoundary<int32_t>();
    CheckScalarHostBoundary<int64_t>();
    CheckScalarHostBoundary<uint8_t>();
}
#endif
