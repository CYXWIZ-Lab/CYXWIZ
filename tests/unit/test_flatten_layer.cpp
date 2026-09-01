#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "algorithms/arrayfire_backend_utils.h"

#include <cyxwiz/layers/flatten.h>
#include <cyxwiz/memory_manager.h>
#include <cyxwiz/sequential.h>
#include <cyxwiz/tensor.h>

#include <nlohmann/json.hpp>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;

json LoadFlattenFixture() {
    std::ifstream stream(CYXWIZ_TRAINING_CORE_FIXTURE_PATH);
    REQUIRE(stream.is_open());

    json fixture;
    stream >> fixture;
    REQUIRE(fixture.value("schema_version", 0) == 1);
    REQUIRE(fixture.at("oracle").value("name", "") == "PyTorch");
    REQUIRE(fixture.at("oracle").value("device", "") == "cpu");
    REQUIRE(!fixture.at("oracle").value("version", "").empty());
    return fixture.at("cases").at("flatten_forward_backward_f32");
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
    for (size_t i = 0; i < expected_values.size(); ++i) {
        CHECK(actual_values[i] == Catch::Approx(expected_values[i]));
    }
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
size_t g_flatten_host_sync_count = 0;
size_t g_flatten_fallback_count = 0;

void CountFlattenHostSync(const cyxwiz::ArrayFireHostSyncEvent&) {
    ++g_flatten_host_sync_count;
}

void CountFlattenFallback(
    const cyxwiz::ArrayFireNativeCpuFallbackEvent&) {
    ++g_flatten_fallback_count;
}

cyxwiz::Tensor DeviceRowMajor3D(const std::vector<float>& values) {
    af::array row_major_storage(4, 3, 2, values.data());
    return cyxwiz::Tensor::FromArrayRowMajor3D(
        af::reorder(row_major_storage, 2, 1, 0));
}

cyxwiz::Tensor DeviceRowMajor2D(const std::vector<float>& values) {
    af::array row_major_storage(12, 2, values.data());
    return cyxwiz::Tensor::FromArrayRowMajor2D(
        af::transpose(row_major_storage));
}

class ScopedArrayFireDeviceRestore {
public:
    ScopedArrayFireDeviceRestore()
        : backend_(af::getActiveBackend()), device_(af::getDevice()) {}

    ~ScopedArrayFireDeviceRestore() {
        try {
            af::setBackend(backend_);
            af::setDevice(device_);
        } catch (...) {
        }
    }

private:
    af::Backend backend_;
    int device_;
};

void CheckActiveDeviceResidency() {
    std::vector<float> values(24);
    std::vector<float> gradients(24);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i) - 7.5f;
        gradients[i] = static_cast<float>(i) * 0.25f - 1.25f;
    }

    cyxwiz::Tensor input = DeviceRowMajor3D(values);
    cyxwiz::Tensor grad_output = DeviceRowMajor2D(gradients);
    cyxwiz::FlattenModule module(1);

    g_flatten_host_sync_count = 0;
    g_flatten_fallback_count = 0;
    const size_t host_bytes_before = cyxwiz::MemoryManager::GetAllocatedBytes();
    cyxwiz::Tensor output;
    cyxwiz::Tensor grad_input;
    {
        const cyxwiz::ScopedArrayFireFallbackPolicy strict(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_sync_observer(
            &CountFlattenHostSync);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CountFlattenFallback);

        output = module.Forward(input);
        grad_input = module.Backward(grad_output);
        (void)output.GetArrayRowMajor2D();
        (void)grad_input.GetArrayRowMajor3D();
        af::sync();
    }

    CHECK(output.Shape() == std::vector<size_t>{2, 12});
    CHECK(grad_input.Shape() == std::vector<size_t>{2, 3, 4});
    CHECK(g_flatten_host_sync_count == 0);
    CHECK(g_flatten_fallback_count == 0);
    CHECK(cyxwiz::MemoryManager::GetAllocatedBytes() == host_bytes_before);

    CheckTensor(output, json{{"shape", {2, 12}}, {"values", values}});
    CheckTensor(
        grad_input,
        json{{"shape", {2, 3, 4}}, {"values", gradients}});
}
#endif

} // namespace

TEST_CASE("FlattenLayer matches PyTorch forward and backward",
          "[flatten][pytorch]") {
    const auto cases = LoadFlattenFixture();
    REQUIRE(cases.is_array());

    for (const auto& test_case : cases) {
        if (test_case.at("parameters").at("start_dim").get<int>() != 1) {
            continue;
        }
        INFO("case=" << test_case.at("name").get<std::string>());

        cyxwiz::FlattenLayer layer;
        cyxwiz::Tensor input = TensorFromFixture(test_case.at("input"));
        CheckTensor(layer.Forward(input), test_case.at("expected").at("output"));

        cyxwiz::Tensor grad_output =
            TensorFromFixture(test_case.at("grad_output"));
        CheckTensor(
            layer.Backward(grad_output),
            test_case.at("expected").at("grad_input"));
    }
}

TEST_CASE("FlattenModule matches configured PyTorch start dimensions",
          "[flatten][pytorch][sequential]") {
    const auto cases = LoadFlattenFixture();
    REQUIRE(cases.is_array());

    for (const auto& test_case : cases) {
        INFO("case=" << test_case.at("name").get<std::string>());
        const int start_dim =
            test_case.at("parameters").at("start_dim").get<int>();
        cyxwiz::FlattenModule module(start_dim);

        cyxwiz::Tensor input = TensorFromFixture(test_case.at("input"));
        CheckTensor(
            module.Forward(input), test_case.at("expected").at("output"));

        cyxwiz::Tensor grad_output =
            TensorFromFixture(test_case.at("grad_output"));
        CheckTensor(
            module.Backward(grad_output),
            test_case.at("expected").at("grad_input"));
    }
}

TEST_CASE("Flatten rejects invalid forward and backward contracts",
          "[flatten][contract]") {
    const std::vector<float> values(24, 1.0f);
    const std::vector<float> wrong_grad_values(24, 2.0f);
    const cyxwiz::Tensor input({2, 3, 4}, values.data());
    const cyxwiz::Tensor wrong_grad({4, 6}, wrong_grad_values.data());
    const cyxwiz::Tensor wrong_dtype_grad(
        {2, 12}, cyxwiz::DataType::Float64);

    cyxwiz::FlattenLayer layer;
    CHECK_THROWS_AS(layer.Backward(wrong_grad), std::logic_error);
    (void)layer.Forward(input);
    CHECK_THROWS_AS(layer.Backward(wrong_grad), std::runtime_error);
    CHECK_THROWS_AS(layer.Backward(wrong_dtype_grad), std::runtime_error);

    const cyxwiz::Tensor rank_one({24}, values.data());
    CHECK_THROWS_AS(layer.Forward(rank_one), std::runtime_error);

    cyxwiz::FlattenModule invalid_module(3);
    CHECK_THROWS_AS(invalid_module.Forward(input), std::runtime_error);
}

TEST_CASE("Flatten preserves every supported tensor dtype exactly",
          "[flatten][dtype]") {
    const auto check = []<typename T>(
                           const std::vector<T>& values,
                           cyxwiz::DataType dtype) {
        cyxwiz::FlattenModule module(1);
        const cyxwiz::Tensor input({2, 2, 2}, values.data(), dtype);
        const cyxwiz::Tensor output = module.Forward(input);
        REQUIRE(output.Shape() == std::vector<size_t>{2, 4});
        REQUIRE(output.GetDataType() == dtype);
        REQUIRE(output.NumBytes() == values.size() * sizeof(T));
        CHECK(std::memcmp(
                  output.ReadData(), values.data(), output.NumBytes()) == 0);

        const cyxwiz::Tensor restored = module.Backward(output);
        REQUIRE(restored.Shape() == std::vector<size_t>{2, 2, 2});
        REQUIRE(restored.GetDataType() == dtype);
        CHECK(std::memcmp(
                  restored.ReadData(), values.data(), restored.NumBytes()) == 0);
    };

    check(std::vector<float>{1, -2, 3, -4, 5, -6, 7, -8},
          cyxwiz::DataType::Float32);
    check(std::vector<double>{1, -2, 3, -4, 5, -6, 7, -8},
          cyxwiz::DataType::Float64);
    check(std::vector<int32_t>{1, -2, 3, -4, 5, -6, 7, -8},
          cyxwiz::DataType::Int32);
    check(std::vector<int64_t>{1, -2, 3, -4, 5, -6, 7, -8},
          cyxwiz::DataType::Int64);
    check(std::vector<uint8_t>{1, 2, 3, 4, 5, 6, 7, 8},
          cyxwiz::DataType::UInt8);
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
TEST_CASE("Flatten forward and backward remain ArrayFire-resident",
          "[flatten][arrayfire][residency]") {
    CheckActiveDeviceResidency();
}

TEST_CASE("Flatten residency passes every installed CUDA and OpenCL device",
          "[flatten][arrayfire][accelerator]") {
    const ScopedArrayFireDeviceRestore restore;
    const int available = af::getAvailableBackends();
    size_t exercised = 0;

    for (const af::Backend backend : {AF_BACKEND_CUDA, AF_BACKEND_OPENCL}) {
        if ((available & static_cast<int>(backend)) == 0) {
            continue;
        }
        af::setBackend(backend);
        const int device_count = af::getDeviceCount();
        for (int device = 0; device < device_count; ++device) {
            DYNAMIC_SECTION(
                "backend=" << static_cast<int>(backend)
                            << " device=" << device) {
                af::setDevice(device);
                CheckActiveDeviceResidency();
                ++exercised;
            }
        }
    }

    if (exercised == 0) {
        WARN("No CUDA or OpenCL accelerator is installed");
    }
}
#endif
