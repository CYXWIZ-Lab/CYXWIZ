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

json LoadPermuteFixture() {
    std::ifstream stream(CYXWIZ_TRAINING_CORE_FIXTURE_PATH);
    REQUIRE(stream.is_open());

    json fixture;
    stream >> fixture;
    REQUIRE(fixture.value("schema_version", 0) == 1);
    REQUIRE(fixture.at("oracle").value("name", "") == "PyTorch");
    const std::string oracle_version =
        fixture.at("oracle").value("version", "");
    REQUIRE(oracle_version.rfind("2.10.0", 0) == 0);
    return fixture.at("cases").at("tensor_permute_f32");
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

std::vector<size_t> PermutedSourceIndices(
    const std::vector<size_t>& shape,
    const std::vector<int>& dims) {
    std::vector<size_t> output_shape;
    output_shape.reserve(dims.size());
    for (int dim : dims) {
        output_shape.push_back(shape[static_cast<size_t>(dim)]);
    }

    const auto input_strides = RowMajorStrides(shape);
    const auto output_strides = RowMajorStrides(output_shape);
    const size_t element_count =
        std::accumulate(shape.begin(), shape.end(), size_t{1},
                        std::multiplies<size_t>());
    std::vector<size_t> source_indices(element_count, 0);
    std::vector<size_t> output_index(output_shape.size(), 0);
    std::vector<size_t> input_index(shape.size(), 0);

    for (size_t linear = 0; linear < element_count; ++linear) {
        size_t remainder = linear;
        for (size_t axis = 0; axis < output_shape.size(); ++axis) {
            output_index[axis] = remainder / output_strides[axis];
            remainder %= output_strides[axis];
            input_index[static_cast<size_t>(dims[axis])] = output_index[axis];
        }
        for (size_t axis = 0; axis < shape.size(); ++axis) {
            source_indices[linear] += input_index[axis] * input_strides[axis];
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

size_t g_permute_host_sync_count = 0;
size_t g_permute_fallback_count = 0;

void CountPermuteHostSync(const cyxwiz::ArrayFireHostSyncEvent&) {
    ++g_permute_host_sync_count;
}

void CountPermuteFallback(
    const cyxwiz::ArrayFireNativeCpuFallbackEvent&) {
    ++g_permute_fallback_count;
}

template<typename T, typename Operation>
void CheckDeviceReorder(const std::vector<size_t>& shape,
                        const std::vector<int>& expected_dims,
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

    g_permute_host_sync_count = 0;
    g_permute_fallback_count = 0;
    cyxwiz::Tensor output;
    {
        const cyxwiz::ScopedArrayFireFallbackPolicy strict(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
            &CountPermuteHostSync);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CountPermuteFallback);
        output = operation(device_only);
        output.GetSemanticArray().eval();
    }

    REQUIRE(g_permute_host_sync_count == 0);
    REQUIRE(g_permute_fallback_count == 0);
    const auto source_indices = PermutedSourceIndices(shape, expected_dims);
    const T* output_values = output.ReadData<T>();
    for (size_t index = 0; index < source_indices.size(); ++index) {
        REQUIRE(output_values[index] == values[source_indices[index]]);
    }
}

template<typename T>
void CheckDevicePermutation(const std::vector<size_t>& shape,
                            const std::vector<int>& dims) {
    CheckDeviceReorder<T>(
        shape, dims,
        [&dims](const cyxwiz::Tensor& input) { return input.Permute(dims); });
}

template<typename T>
void CheckDeviceTranspose(const std::vector<size_t>& shape,
                          int dim0,
                          int dim1) {
    std::vector<int> expected_dims(shape.size());
    std::iota(expected_dims.begin(), expected_dims.end(), 0);
    const int rank = static_cast<int>(shape.size());
    const int first = dim0 < 0 ? dim0 + rank : dim0;
    const int second = dim1 < 0 ? dim1 + rank : dim1;
    std::swap(expected_dims[static_cast<size_t>(first)],
              expected_dims[static_cast<size_t>(second)]);
    CheckDeviceReorder<T>(
        shape, expected_dims,
        [dim0, dim1](const cyxwiz::Tensor& input) {
            return input.Transpose(dim0, dim1);
        });
}

template<typename T>
void CheckDeviceDefaultTranspose() {
    CheckDeviceReorder<T>(
        {2, 3}, {1, 0},
        [](const cyxwiz::Tensor& input) { return input.Transpose(); });
}

template<typename T>
void CheckDevicePermutationMatrix() {
    CheckDevicePermutation<T>({2, 3}, {1, 0});
    CheckDevicePermutation<T>({2, 3, 4}, {2, 0, 1});
    CheckDevicePermutation<T>({2, 2, 3, 2}, {3, 1, 0, 2});
    CheckDeviceTranspose<T>({2, 3}, 0, 1);
    CheckDeviceTranspose<T>({2, 3, 4}, -1, 0);
    CheckDeviceTranspose<T>({2, 2, 3, 2}, 1, 3);
    CheckDeviceDefaultTranspose<T>();
}
#endif

} // namespace

TEST_CASE("Tensor permute matches generated PyTorch rank matrix",
          "[tensor][tensor_permute][pytorch]") {
    const json fixture = LoadPermuteFixture();
    for (const json& test_case : fixture) {
        CAPTURE(test_case.at("name").get<std::string>());
        const cyxwiz::Tensor input = TensorFromFixture(test_case.at("input"));
        CheckTensor(
            input.Permute(test_case.at("dims").get<std::vector<int>>()),
            test_case.at("expected"));
    }
}

TEST_CASE("Tensor permute and transpose reject invalid dimensions",
          "[tensor][tensor_permute][errors]") {
    const cyxwiz::Tensor rank4 =
        cyxwiz::Tensor::RangeN({2, 2, 3, 2}, cyxwiz::DataType::Float32);
    const cyxwiz::Tensor scalar({}, cyxwiz::DataType::Float32);

    REQUIRE_THROWS_AS(rank4.Permute({0, 1, 2}), std::runtime_error);
    REQUIRE_THROWS_AS(rank4.Permute({0, 1, 1, 3}), std::runtime_error);
    REQUIRE_THROWS_AS(rank4.Permute({0, 1, 2, 4}), std::runtime_error);
    REQUIRE_THROWS_AS(rank4.Permute({0, 1, 2, -5}), std::runtime_error);
    REQUIRE_THROWS_AS(scalar.Permute({0}), std::runtime_error);
    REQUIRE_THROWS_AS(rank4.Transpose(0, 4), std::runtime_error);
    REQUIRE_THROWS_AS(rank4.Transpose(-5, 1), std::runtime_error);
    REQUIRE_THROWS_AS(rank4.Transpose(), std::runtime_error);
    REQUIRE_THROWS_AS(scalar.Transpose(0, 0), std::runtime_error);
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
TEST_CASE("Tensor permute and transpose remain device resident for every dtype",
          "[tensor][tensor_permute][arrayfire][host_sync]") {
    const auto activation =
        cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(true);
    REQUIRE(activation.success);
    REQUIRE(activation.execution_validated);

    CheckDevicePermutationMatrix<float>();
    CheckDevicePermutationMatrix<double>();
    CheckDevicePermutationMatrix<int32_t>();
    CheckDevicePermutationMatrix<int64_t>();
    CheckDevicePermutationMatrix<uint8_t>();
}

TEST_CASE("Tensor permute rank-five compatibility fallback is explicit",
          "[tensor][tensor_permute][arrayfire][fallback]") {
    const cyxwiz::Tensor input =
        cyxwiz::Tensor::RangeN({1, 2, 1, 2, 2}, cyxwiz::DataType::Float32);

    g_permute_fallback_count = 0;
    cyxwiz::Tensor output;
    {
        const cyxwiz::ScopedArrayFireFallbackPolicy compatibility(
            cyxwiz::ArrayFireFallbackPolicy::AllowNativeCpuFallback);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver observer(
            &CountPermuteFallback);
        output = input.Permute({4, 1, 3, 0, 2});
    }
    REQUIRE(g_permute_fallback_count == 1);
    REQUIRE(output.Shape() == std::vector<size_t>{2, 2, 2, 1, 1});

    g_permute_fallback_count = 0;
    {
        const cyxwiz::ScopedArrayFireFallbackPolicy strict(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver observer(
            &CountPermuteFallback);
        REQUIRE_THROWS_AS(
            input.Permute({4, 1, 3, 0, 2}), std::runtime_error);
    }
    REQUIRE(g_permute_fallback_count == 1);
}
#endif
