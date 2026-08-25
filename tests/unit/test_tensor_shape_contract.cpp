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
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;

json LoadShapeFixture() {
    std::ifstream stream(CYXWIZ_TRAINING_CORE_FIXTURE_PATH);
    REQUIRE(stream.is_open());

    json fixture;
    stream >> fixture;
    REQUIRE(fixture.value("schema_version", 0) == 1);
    REQUIRE(fixture.at("oracle").value("name", "") == "PyTorch");
    const std::string oracle_version =
        fixture.at("oracle").value("version", "");
    REQUIRE(oracle_version.rfind("2.10.0", 0) == 0);
    return fixture.at("cases").at("tensor_shape_semantics_f32");
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
    REQUIRE(actual.GetDataType() == cyxwiz::DataType::Float32);

    const auto expected_values =
        expected.at("values").get<std::vector<float>>();
    REQUIRE(actual.NumElements() == expected_values.size());
    const float* actual_values = actual.ReadData<float>();
    for (size_t index = 0; index < expected_values.size(); ++index) {
        REQUIRE(actual_values[index] == expected_values[index]);
    }
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

size_t g_shape_host_sync_count = 0;
size_t g_shape_fallback_count = 0;

void CountShapeHostSync(const cyxwiz::ArrayFireHostSyncEvent&) {
    ++g_shape_host_sync_count;
}

void CountShapeFallback(
    const cyxwiz::ArrayFireNativeCpuFallbackEvent&) {
    ++g_shape_fallback_count;
}

template<typename T>
void CheckDeviceShapeMatrix() {
    constexpr cyxwiz::DataType dtype = TensorType<T>::value;
    std::vector<T> values(6);
    for (size_t index = 0; index < values.size(); ++index) {
        values[index] = static_cast<T>(index + 1);
    }

    cyxwiz::Tensor host({1, 2, 1, 3}, values.data(), dtype);
    cyxwiz::Tensor device_only;
    device_only.SetFromArray(host.GetArray());

    g_shape_host_sync_count = 0;
    g_shape_fallback_count = 0;
    cyxwiz::Tensor reshaped;
    cyxwiz::Tensor viewed;
    cyxwiz::Tensor squeezed;
    cyxwiz::Tensor squeezed_dim;
    cyxwiz::Tensor squeeze_noop;
    cyxwiz::Tensor unsqueezed;
    cyxwiz::Tensor flattened;
    {
        const cyxwiz::ScopedArrayFireFallbackPolicy strict(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
            &CountShapeHostSync);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CountShapeFallback);

        reshaped = device_only.Reshape({3, 2});
        viewed = device_only.View({2, 3});
        squeezed = device_only.Squeeze();
        squeezed_dim = device_only.Squeeze(-2);
        squeeze_noop = device_only.Squeeze(1);
        unsqueezed = squeezed.Unsqueeze(0);
        flattened = device_only.Flatten(1, 3);

        reshaped.GetSemanticArray().eval();
        viewed.GetSemanticArray().eval();
        squeezed.GetSemanticArray().eval();
        squeezed_dim.GetSemanticArray().eval();
        squeeze_noop.GetSemanticArray().eval();
        unsqueezed.GetSemanticArray().eval();
        flattened.GetSemanticArray().eval();
    }

    REQUIRE(g_shape_host_sync_count == 0);
    REQUIRE(g_shape_fallback_count == 0);
    REQUIRE(reshaped.Shape() == std::vector<size_t>{3, 2});
    REQUIRE(viewed.Shape() == std::vector<size_t>{2, 3});
    REQUIRE(squeezed.Shape() == std::vector<size_t>{2, 3});
    REQUIRE(squeezed_dim.Shape() == std::vector<size_t>{1, 2, 3});
    REQUIRE(squeeze_noop.Shape() == std::vector<size_t>{1, 2, 1, 3});
    REQUIRE(unsqueezed.Shape() == std::vector<size_t>{1, 2, 3});
    REQUIRE(flattened.Shape() == std::vector<size_t>{1, 6});
    REQUIRE(flattened.ReadData<T>()[5] == values[5]);
}
#endif

} // namespace

TEST_CASE("Tensor shape operations match generated PyTorch semantics",
          "[tensor][tensor_shape][pytorch]") {
    const json fixture = LoadShapeFixture();
    const json& expected = fixture.at("expected");
    const cyxwiz::Tensor input = TensorFromFixture(fixture.at("input"));

    CheckTensor(input.Reshape({3, 2}), expected.at("reshape"));
    CheckTensor(input.View({2, 3}), expected.at("view"));
    CheckTensor(input.Squeeze(), expected.at("squeeze_all"));
    CheckTensor(
        input.Squeeze(-1), expected.at("squeeze_last_non_singleton"));
    CheckTensor(
        input.Squeeze(1), expected.at("squeeze_middle_non_singleton"));
    CheckTensor(
        input.Squeeze(-2), expected.at("squeeze_negative_singleton"));
    CheckTensor(input.Unsqueeze(0), expected.at("unsqueeze_front"));
    CheckTensor(input.Unsqueeze(-1), expected.at("unsqueeze_back"));
    CheckTensor(input.Flatten(), expected.at("flatten_all"));
    CheckTensor(input.Flatten(1, 2), expected.at("flatten_middle"));

    const cyxwiz::Tensor scalar = TensorFromFixture(fixture.at("scalar"));
    CheckTensor(scalar.Squeeze(), expected.at("scalar_squeeze"));
    CheckTensor(scalar.Squeeze(-1), expected.at("scalar_squeeze_dim"));
    CheckTensor(scalar.Unsqueeze(-1), expected.at("scalar_unsqueeze"));
    CheckTensor(scalar.Flatten(), expected.at("scalar_flatten"));

    const cyxwiz::Tensor empty = TensorFromFixture(fixture.at("empty"));
    CheckTensor(empty.Reshape({0, 6}), expected.at("empty_reshape"));
    CheckTensor(empty.Flatten(1, 2), expected.at("empty_flatten"));
}

TEST_CASE("Tensor shape operations reject invalid dimensions and products",
          "[tensor][tensor_shape][errors]") {
    const cyxwiz::Tensor input =
        cyxwiz::Tensor::RangeN({1, 2, 1, 3}, cyxwiz::DataType::Float32);
    const cyxwiz::Tensor scalar({}, cyxwiz::DataType::Float32);

    REQUIRE_THROWS_AS(input.Reshape({5}), std::runtime_error);
    REQUIRE_THROWS_AS(input.View({2, 4}), std::runtime_error);
    REQUIRE_THROWS_AS(
        input.Reshape({(std::numeric_limits<size_t>::max)(), 2}),
        std::overflow_error);
    REQUIRE_THROWS_AS(input.Squeeze(-5), std::runtime_error);
    REQUIRE_THROWS_AS(input.Squeeze(4), std::runtime_error);
    REQUIRE_THROWS_AS(scalar.Squeeze(-2), std::runtime_error);
    REQUIRE_THROWS_AS(scalar.Squeeze(1), std::runtime_error);
    REQUIRE_THROWS_AS(input.Unsqueeze(-6), std::runtime_error);
    REQUIRE_THROWS_AS(input.Unsqueeze(5), std::runtime_error);
    REQUIRE_THROWS_AS(scalar.Unsqueeze(-2), std::runtime_error);
    REQUIRE_THROWS_AS(scalar.Unsqueeze(1), std::runtime_error);
    REQUIRE_THROWS_AS(input.Flatten(3, 1), std::runtime_error);
    REQUIRE_THROWS_AS(input.Flatten(-5, -1), std::runtime_error);
    REQUIRE_THROWS_AS(input.Flatten(0, 4), std::runtime_error);

    const cyxwiz::Tensor empty({0}, cyxwiz::DataType::Float32);
    const std::vector<size_t> zero_product_shape = {
        (std::numeric_limits<size_t>::max)(), 2, 0};
    const cyxwiz::Tensor zero_product = empty.Reshape(zero_product_shape);
    REQUIRE(zero_product.Shape() == zero_product_shape);
    REQUIRE(zero_product.NumElements() == 0);
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
TEST_CASE("Tensor shape operations remain device resident for every dtype",
          "[tensor][tensor_shape][arrayfire][host_sync]") {
    const auto activation =
        cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(true);
    REQUIRE(activation.success);
    REQUIRE(activation.execution_validated);

    CheckDeviceShapeMatrix<float>();
    CheckDeviceShapeMatrix<double>();
    CheckDeviceShapeMatrix<int32_t>();
    CheckDeviceShapeMatrix<int64_t>();
    CheckDeviceShapeMatrix<uint8_t>();
}
#endif
