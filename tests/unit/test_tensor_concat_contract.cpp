#include <catch2/catch_test_macros.hpp>

#include "tensor_index_helpers.h"
#include "algorithms/arrayfire_backend_utils.h"

#include <cyxwiz/device.h>
#include <cyxwiz/tensor.h>

#include <nlohmann/json.hpp>

#include <arrayfire.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;

json LoadConcatFixture() {
    std::ifstream stream(CYXWIZ_TRAINING_CORE_FIXTURE_PATH);
    REQUIRE(stream.is_open());

    json fixture;
    stream >> fixture;
    REQUIRE(fixture.value("schema_version", 0) == 1);
    REQUIRE(fixture.at("oracle").value("name", "") == "PyTorch");
    REQUIRE(fixture.at("oracle").value("version", "").rfind("2.10.0", 0) == 0);
    return fixture.at("cases").at("tensor_concat_f32");
}

std::vector<size_t> ShapeFromFixture(const json& value) {
    return value.at("shape").get<std::vector<size_t>>();
}

cyxwiz::Tensor TensorFromFixture(const json& value) {
    const auto shape = ShapeFromFixture(value);
    const auto values = value.at("values").get<std::vector<float>>();
    return cyxwiz::Tensor(shape, values.data(), cyxwiz::DataType::Float32);
}

std::vector<cyxwiz::Tensor> TensorsFromFixture(const json& values) {
    std::vector<cyxwiz::Tensor> tensors;
    tensors.reserve(values.size());
    for (const json& value : values) {
        tensors.push_back(TensorFromFixture(value));
    }
    return tensors;
}

void CheckTensor(const cyxwiz::Tensor& actual, const json& expected) {
    REQUIRE(actual.Shape() == ShapeFromFixture(expected));
    const auto expected_values = expected.at("values").get<std::vector<float>>();
    REQUIRE(actual.NumElements() == expected_values.size());
    const float* values = actual.ReadData<float>();
    for (size_t index = 0; index < expected_values.size(); ++index) {
        REQUIRE(values[index] == expected_values[index]);
    }
}

void CheckTensorVector(const std::vector<cyxwiz::Tensor>& actual,
                       const json& expected) {
    REQUIRE(actual.size() == expected.size());
    for (size_t index = 0; index < actual.size(); ++index) {
        CheckTensor(actual[index], expected[index]);
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

size_t g_concat_host_sync_count = 0;
size_t g_concat_fallback_count = 0;

void CountConcatHostSync(const cyxwiz::ArrayFireHostSyncEvent&) {
    ++g_concat_host_sync_count;
}

void CountConcatFallback(const cyxwiz::ArrayFireNativeCpuFallbackEvent&) {
    ++g_concat_fallback_count;
}

size_t ElementCount(const std::vector<size_t>& shape) {
    return std::accumulate(
        shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
}

template<typename T>
cyxwiz::Tensor DeviceTensor(const std::vector<size_t>& shape, size_t offset) {
    std::vector<T> values(ElementCount(shape));
    for (size_t index = 0; index < values.size(); ++index) {
        values[index] = static_cast<T>(offset + index + 1);
    }
    cyxwiz::Tensor host(shape, values.data(), TensorType<T>::value);
    cyxwiz::Tensor device;
    device.SetFromSemanticArray(host.GetSemanticArray(), shape);
    return device;
}

template<typename T>
void CheckResidentCat(const std::vector<std::vector<size_t>>& shapes,
                      size_t axis) {
    std::vector<cyxwiz::Tensor> inputs;
    std::vector<std::vector<T>> source_values;
    size_t offset = 0;
    for (const auto& shape : shapes) {
        const size_t count = ElementCount(shape);
        std::vector<T> values(count);
        for (size_t index = 0; index < count; ++index) {
            values[index] = static_cast<T>(offset + index + 1);
        }
        source_values.push_back(values);
        inputs.push_back(DeviceTensor<T>(shape, offset));
        offset += count;
    }

    g_concat_host_sync_count = 0;
    g_concat_fallback_count = 0;
    cyxwiz::Tensor output;
    {
        const cyxwiz::ScopedArrayFireFallbackPolicy strict(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
            &CountConcatHostSync);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CountConcatFallback);
        output = cyxwiz::Tensor::Cat(inputs, static_cast<int>(axis));
        output.GetSemanticArray().eval();
    }
    REQUIRE(g_concat_host_sync_count == 0);
    REQUIRE(g_concat_fallback_count == 0);

    std::vector<size_t> output_shape = shapes.front();
    output_shape[axis] = 0;
    for (const auto& shape : shapes) {
        output_shape[axis] += shape[axis];
    }
    REQUIRE(output.Shape() == output_shape);

    const auto output_strides = RowMajorStrides(output_shape);
    std::vector<std::vector<size_t>> input_strides;
    for (const auto& shape : shapes) {
        input_strides.push_back(RowMajorStrides(shape));
    }
    const T* actual = output.ReadData<T>();
    std::vector<size_t> coordinate(output_shape.size(), 0);
    for (size_t linear = 0; linear < output.NumElements(); ++linear) {
        size_t remainder = linear;
        for (size_t dim = 0; dim < output_shape.size(); ++dim) {
            coordinate[dim] = remainder / output_strides[dim];
            remainder %= output_strides[dim];
        }
        size_t input_index = 0;
        size_t axis_offset = coordinate[axis];
        while (axis_offset >= shapes[input_index][axis]) {
            axis_offset -= shapes[input_index][axis];
            ++input_index;
        }
        coordinate[axis] = axis_offset;
        size_t source_linear = 0;
        for (size_t dim = 0; dim < coordinate.size(); ++dim) {
            source_linear += coordinate[dim] * input_strides[input_index][dim];
        }
        REQUIRE(actual[linear] == source_values[input_index][source_linear]);
    }
}

template<typename T>
void CheckResidentStack(const std::vector<size_t>& shape, size_t axis) {
    const size_t count = ElementCount(shape);
    const std::vector<cyxwiz::Tensor> inputs = {
        DeviceTensor<T>(shape, 0),
        DeviceTensor<T>(shape, count)};

    g_concat_host_sync_count = 0;
    g_concat_fallback_count = 0;
    cyxwiz::Tensor output;
    {
        const cyxwiz::ScopedArrayFireFallbackPolicy strict(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
            &CountConcatHostSync);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CountConcatFallback);
        output = cyxwiz::Tensor::Stack(inputs, static_cast<int>(axis));
        output.GetSemanticArray().eval();
    }
    REQUIRE(g_concat_host_sync_count == 0);
    REQUIRE(g_concat_fallback_count == 0);

    std::vector<size_t> output_shape = shape;
    output_shape.insert(output_shape.begin() + static_cast<ptrdiff_t>(axis), 2);
    REQUIRE(output.Shape() == output_shape);
    const auto output_strides = RowMajorStrides(output_shape);
    const auto input_strides = RowMajorStrides(shape);
    const T* actual = output.ReadData<T>();
    std::vector<size_t> output_coordinate(output_shape.size(), 0);
    for (size_t linear = 0; linear < output.NumElements(); ++linear) {
        size_t remainder = linear;
        for (size_t dim = 0; dim < output_shape.size(); ++dim) {
            output_coordinate[dim] = remainder / output_strides[dim];
            remainder %= output_strides[dim];
        }
        const size_t tensor_index = output_coordinate[axis];
        size_t source_linear = 0;
        for (size_t dim = 0, source_dim = 0;
             dim < output_coordinate.size(); ++dim) {
            if (dim != axis) {
                source_linear +=
                    output_coordinate[dim] * input_strides[source_dim++];
            }
        }
        REQUIRE(actual[linear] == static_cast<T>(
            tensor_index * count + source_linear + 1));
    }
}

template<typename T>
void CheckResidentSplitChunk() {
    cyxwiz::Tensor input = DeviceTensor<T>({2, 3, 4}, 0);
    g_concat_host_sync_count = 0;
    g_concat_fallback_count = 0;
    std::vector<cyxwiz::Tensor> split;
    std::vector<cyxwiz::Tensor> chunks;
    {
        const cyxwiz::ScopedArrayFireFallbackPolicy strict(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
            &CountConcatHostSync);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CountConcatFallback);
        split = input.Split(2, 2);
        chunks = input.Chunk(2, 1);
        for (const cyxwiz::Tensor& output : split) {
            output.GetSemanticArray().eval();
        }
        for (const cyxwiz::Tensor& output : chunks) {
            output.GetSemanticArray().eval();
        }
    }
    REQUIRE(g_concat_host_sync_count == 0);
    REQUIRE(g_concat_fallback_count == 0);
    REQUIRE(split.size() == 2);
    REQUIRE(chunks.size() == 2);
    REQUIRE(split[0].Shape() == std::vector<size_t>{2, 3, 2});
    REQUIRE(split[1].Shape() == std::vector<size_t>{2, 3, 2});
    REQUIRE(chunks[0].Shape() == std::vector<size_t>{2, 2, 4});
    REQUIRE(chunks[1].Shape() == std::vector<size_t>{2, 1, 4});
    REQUIRE(split[0].ReadData<T>()[0] == static_cast<T>(1));
    REQUIRE(split[0].ReadData<T>()[11] == static_cast<T>(22));
    REQUIRE(split[1].ReadData<T>()[0] == static_cast<T>(3));
    REQUIRE(split[1].ReadData<T>()[11] == static_cast<T>(24));
    REQUIRE(chunks[0].ReadData<T>()[0] == static_cast<T>(1));
    REQUIRE(chunks[0].ReadData<T>()[15] == static_cast<T>(20));
    REQUIRE(chunks[1].ReadData<T>()[0] == static_cast<T>(9));
    REQUIRE(chunks[1].ReadData<T>()[7] == static_cast<T>(24));
}

template<typename T>
void CheckResidentConcatMatrix() {
    CheckResidentCat<T>({{2}, {3}}, 0);
    CheckResidentCat<T>({{2, 1}, {2, 2}}, 1);
    CheckResidentCat<T>({{2, 1, 2}, {2, 2, 2}}, 1);
    CheckResidentCat<T>({{1, 2, 1, 2}, {1, 2, 2, 2}}, 2);

    CheckResidentStack<T>({}, 0);
    CheckResidentStack<T>({3}, 1);
    CheckResidentStack<T>({2, 3}, 1);
    CheckResidentStack<T>({2, 2, 2}, 3);
    CheckResidentSplitChunk<T>();
}

} // namespace

TEST_CASE("Tensor concat family matches generated PyTorch matrices",
          "[tensor][tensor_concat][pytorch]") {
    const json fixture = LoadConcatFixture();
    for (const json& test_case : fixture.at("cat")) {
        CAPTURE(test_case.at("name").get<std::string>());
        CheckTensor(
            cyxwiz::Tensor::Cat(
                TensorsFromFixture(test_case.at("inputs")),
                test_case.at("dim").get<int>()),
            test_case.at("expected"));
    }
    for (const json& test_case : fixture.at("stack")) {
        CAPTURE(test_case.at("name").get<std::string>());
        CheckTensor(
            cyxwiz::Tensor::Stack(
                TensorsFromFixture(test_case.at("inputs")),
                test_case.at("dim").get<int>()),
            test_case.at("expected"));
    }
    for (const json& test_case : fixture.at("split_size")) {
        CAPTURE(test_case.at("name").get<std::string>());
        const cyxwiz::Tensor input = TensorFromFixture(test_case.at("input"));
        CheckTensorVector(
            input.Split(test_case.at("split_size").get<int>(),
                        test_case.at("dim").get<int>()),
            test_case.at("expected"));
    }
    for (const json& test_case : fixture.at("split_sizes")) {
        CAPTURE(test_case.at("name").get<std::string>());
        const cyxwiz::Tensor input = TensorFromFixture(test_case.at("input"));
        CheckTensorVector(
            input.Split(test_case.at("sizes").get<std::vector<int>>(),
                        test_case.at("dim").get<int>()),
            test_case.at("expected"));
    }
    for (const json& test_case : fixture.at("chunk")) {
        CAPTURE(test_case.at("name").get<std::string>());
        const cyxwiz::Tensor input = TensorFromFixture(test_case.at("input"));
        CheckTensorVector(
            input.Chunk(test_case.at("chunks").get<int>(),
                        test_case.at("dim").get<int>()),
            test_case.at("expected"));
    }
}

TEST_CASE("Tensor concat family rejects invalid contracts before compute",
          "[tensor][tensor_concat][errors]") {
    const cyxwiz::Tensor scalar({}, cyxwiz::DataType::Float32);
    const cyxwiz::Tensor matrix =
        cyxwiz::Tensor::Ones({2, 2}, cyxwiz::DataType::Float32);
    const cyxwiz::Tensor bad_shape =
        cyxwiz::Tensor::Ones({2, 3}, cyxwiz::DataType::Float32);
    const cyxwiz::Tensor bad_dtype =
        cyxwiz::Tensor::Ones({2, 2}, cyxwiz::DataType::Int32);

    REQUIRE_THROWS_AS(cyxwiz::Tensor::Cat({}, 0), std::runtime_error);
    REQUIRE_THROWS_AS(cyxwiz::Tensor::Cat({scalar, scalar}, 0), std::runtime_error);
    REQUIRE_THROWS_AS(cyxwiz::Tensor::Cat({matrix, bad_shape}, 0), std::runtime_error);
    REQUIRE_THROWS_AS(cyxwiz::Tensor::Cat({matrix, bad_dtype}, 0), std::runtime_error);
    REQUIRE_THROWS_AS(cyxwiz::Tensor::Stack({}, 0), std::runtime_error);
    REQUIRE_THROWS_AS(cyxwiz::Tensor::Stack({matrix, bad_shape}, 0), std::runtime_error);
    REQUIRE_THROWS_AS(cyxwiz::Tensor::Stack({matrix, bad_dtype}, 0), std::runtime_error);
    REQUIRE_THROWS_AS(matrix.Split(0, 0), std::runtime_error);
    REQUIRE_THROWS_AS(matrix.Split({1, -1, 2}, 1), std::runtime_error);
    REQUIRE_THROWS_AS(matrix.Split(std::vector<int>{1}, 1), std::runtime_error);
    REQUIRE_THROWS_AS(matrix.Chunk(0, 0), std::runtime_error);
}

TEST_CASE("Tensor cat and stack remain device resident for every dtype",
          "[tensor][tensor_concat][arrayfire][host_sync]") {
    const auto activation =
        cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(true);
    REQUIRE(activation.success);
    REQUIRE(activation.execution_validated);

    CheckResidentConcatMatrix<float>();
    CheckResidentConcatMatrix<double>();
    CheckResidentConcatMatrix<int32_t>();
    CheckResidentConcatMatrix<int64_t>();
    CheckResidentConcatMatrix<uint8_t>();
}

// TOFIX140 F4: rank-five Cat / Stack run on the device ([after, extent,
// before] flat views joined on ArrayFire dim 1).
TEST_CASE("Tensor rank-five concat runs on the device",
          "[tensor][tensor_concat][arrayfire]") {
    REQUIRE(cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(true).success);
    const std::vector<size_t> shape{2, 3, 2, 2, 3};
    const cyxwiz::Tensor first = cyxwiz::Tensor::RangeN(shape, cyxwiz::DataType::Float32);
    const cyxwiz::Tensor second = first + 1000.0f;
    const std::vector<size_t> shape4{2, 3, 2, 3};
    const cyxwiz::Tensor first4 = cyxwiz::Tensor::RangeN(shape4, cyxwiz::DataType::Float32);
    const cyxwiz::Tensor second4 = first4 + 1000.0f;
    g_concat_fallback_count = 0;
    for (int axis = 0; axis < 5; ++axis) {
        CAPTURE(axis);
        cyxwiz::Tensor cat, stack;
        {
            const cyxwiz::ScopedArrayFireFallbackPolicy strict(
                cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
            const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver observer(
                &CountConcatFallback);
            cat = cyxwiz::Tensor::Cat({first, second}, axis);
            stack = cyxwiz::Tensor::Stack({first4, second4}, axis);
        }
        std::vector<size_t> cat_shape = shape;
        cat_shape[static_cast<size_t>(axis)] *= 2;
        REQUIRE(cat.Shape() == cat_shape);
        const float* cat_values = cat.ReadData<float>();
        for (size_t i = 0; i < cat.NumElements(); ++i) {
            std::vector<size_t> c = tensor_test::Unravel(i, cat_shape);
            const size_t extent = shape[static_cast<size_t>(axis)];
            const bool from_second = c[static_cast<size_t>(axis)] >= extent;
            if (from_second) c[static_cast<size_t>(axis)] -= extent;
            REQUIRE(cat_values[i] == static_cast<float>(tensor_test::Ravel(c, shape)) + (from_second ? 1000.0f : 0.0f));
        }
        std::vector<size_t> stack_shape = shape4;
        stack_shape.insert(stack_shape.begin() + axis, 2);
        REQUIRE(stack.Shape() == stack_shape);
        const float* stack_values = stack.ReadData<float>();
        for (size_t i = 0; i < stack.NumElements(); ++i) {
            std::vector<size_t> c = tensor_test::Unravel(i, stack_shape);
            const size_t which = c[static_cast<size_t>(axis)];
            c.erase(c.begin() + axis);
            REQUIRE(stack_values[i] == static_cast<float>(tensor_test::Ravel(c, shape4)) + (which == 1 ? 1000.0f : 0.0f));
        }
    }
    REQUIRE(g_concat_fallback_count == 0);
}
