#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "algorithms/arrayfire_backend_utils.h"

#include <cyxwiz/device.h>
#include <cyxwiz/tensor.h>

#include <nlohmann/json.hpp>

#include <arrayfire.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;

json LoadReductionFixture() {
    std::ifstream stream(CYXWIZ_TRAINING_CORE_FIXTURE_PATH);
    REQUIRE(stream.is_open());

    json fixture;
    stream >> fixture;
    REQUIRE(fixture.value("schema_version", 0) == 1);
    REQUIRE(fixture.at("oracle").value("name", "") == "PyTorch");
    REQUIRE(fixture.at("oracle").value("version", "").rfind("2.10.0", 0) == 0);
    return fixture.at("cases").at("tensor_reductions");
}

cyxwiz::DataType ParseDataType(const std::string& value) {
    if (value == "float32") return cyxwiz::DataType::Float32;
    if (value == "float64") return cyxwiz::DataType::Float64;
    if (value == "int32") return cyxwiz::DataType::Int32;
    if (value == "int64") return cyxwiz::DataType::Int64;
    if (value == "uint8") return cyxwiz::DataType::UInt8;
    throw std::runtime_error("unsupported reduction fixture dtype: " + value);
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
    throw std::runtime_error("unsupported reduction Tensor dtype");
}

cyxwiz::Tensor ApplyReduction(const cyxwiz::Tensor& input,
                              const json& test_case) {
    const std::string operation = test_case.at("operation").get<std::string>();
    const bool global = test_case.at("dim").is_null();
    const bool keepdim = test_case.at("keepdim").get<bool>();
    const int64_t correction = test_case.at("correction").get<int64_t>();
    if (global) {
        if (operation == "torch.sum") return input.Sum();
        if (operation == "torch.mean") return input.Mean();
        if (operation == "torch.max") return input.Max();
        if (operation == "torch.min") return input.Min();
        if (operation == "torch.prod") return input.Prod();
        if (operation == "torch.var") return input.VarWithCorrection(correction);
        if (operation == "torch.std") return input.StdWithCorrection(correction);
    } else {
        const int dim = test_case.at("dim").get<int>();
        if (operation == "torch.sum") return input.Sum(dim, keepdim);
        if (operation == "torch.mean") return input.Mean(dim, keepdim);
        if (operation == "torch.max") return input.Max(dim, keepdim);
        if (operation == "torch.min") return input.Min(dim, keepdim);
        if (operation == "torch.prod") return input.Prod(dim, keepdim);
        if (operation == "torch.var") return input.Var(dim, keepdim, correction);
        if (operation == "torch.std") return input.Std(dim, keepdim, correction);
    }
    throw std::runtime_error("unsupported reduction fixture operation: " + operation);
}

template <typename T>
void CheckExactValues(const cyxwiz::Tensor& actual, const json& expected) {
    const auto values = expected.at("values").get<std::vector<T>>();
    const T* output = actual.ReadData<T>();
    REQUIRE(actual.NumElements() == values.size());
    for (size_t index = 0; index < values.size(); ++index) {
        REQUIRE(output[index] == values[index]);
    }
}

template <typename T>
void CheckRealValues(const cyxwiz::Tensor& actual,
                     const json& expected,
                     double absolute_tolerance,
                     double relative_tolerance) {
    const auto values = expected.at("values").get<std::vector<T>>();
    const T* output = actual.ReadData<T>();
    REQUIRE(actual.NumElements() == values.size());
    for (size_t index = 0; index < values.size(); ++index) {
        REQUIRE(static_cast<double>(output[index]) ==
                Catch::Approx(static_cast<double>(values[index]))
                    .margin(absolute_tolerance)
                    .epsilon(relative_tolerance));
    }
}

void CheckReduction(const cyxwiz::Tensor& actual, const json& test_case) {
    const auto& expected = test_case.at("expected");
    const cyxwiz::DataType dtype =
        ParseDataType(test_case.at("output_dtype").get<std::string>());
    REQUIRE(actual.GetDataType() == dtype);
    REQUIRE(actual.Shape() == expected.at("shape").get<std::vector<size_t>>());
    const double absolute_tolerance =
        test_case.at("tolerance").at("atol").get<double>();
    const double relative_tolerance =
        test_case.at("tolerance").at("rtol").get<double>();
    switch (dtype) {
        case cyxwiz::DataType::Float32:
            CheckRealValues<float>(actual, expected, absolute_tolerance,
                                   relative_tolerance);
            break;
        case cyxwiz::DataType::Float64:
            CheckRealValues<double>(actual, expected, absolute_tolerance,
                                    relative_tolerance);
            break;
        case cyxwiz::DataType::Int32:
            CheckExactValues<int32_t>(actual, expected);
            break;
        case cyxwiz::DataType::Int64:
            CheckExactValues<int64_t>(actual, expected);
            break;
        case cyxwiz::DataType::UInt8:
            CheckExactValues<uint8_t>(actual, expected);
            break;
    }
}

size_t g_reduction_host_sync_count = 0;
size_t g_reduction_fallback_count = 0;

void CountReductionHostSync(const cyxwiz::ArrayFireHostSyncEvent&) {
    ++g_reduction_host_sync_count;
}

void CountReductionFallback(const cyxwiz::ArrayFireNativeCpuFallbackEvent&) {
    ++g_reduction_fallback_count;
}

cyxwiz::Tensor DeviceTensorFromFixture(const json& value,
                                       cyxwiz::DataType dtype) {
    cyxwiz::Tensor host = TensorFromFixture(value, dtype);
    return cyxwiz::Tensor::FromSemanticArray(
        host.GetSemanticArray(), host.Shape());
}

void CheckReductionFixtureOnActiveDevice(const json& fixture,
                                         bool supports_fp64) {
    for (const json& test_case : fixture) {
        const std::string name = test_case.at("name").get<std::string>();
        CAPTURE(name);
        const cyxwiz::DataType input_dtype =
            ParseDataType(test_case.at("input_dtype").get<std::string>());
        if (input_dtype == cyxwiz::DataType::Float64 && !supports_fp64) {
            continue;
        }
        const cyxwiz::Tensor input =
            DeviceTensorFromFixture(test_case.at("input"), input_dtype);
        g_reduction_host_sync_count = 0;
        g_reduction_fallback_count = 0;
        cyxwiz::Tensor actual;
        {
            const cyxwiz::ScopedArrayFireFallbackPolicy strict(
                cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
            const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
                &CountReductionHostSync);
            const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
                &CountReductionFallback);
            actual = ApplyReduction(input, test_case);
            actual.GetSemanticArray().eval();
        }
        REQUIRE(g_reduction_host_sync_count == 0);
        REQUIRE(g_reduction_fallback_count == 0);
        CheckReduction(actual, test_case);
    }
}

} // namespace

TEST_CASE("Tensor reductions match generated PyTorch dtype rank and correction matrix",
          "[tensor][tensor_reduction][pytorch]") {
    const json fixture = LoadReductionFixture();
    const auto activation =
        cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(true);
    REQUIRE(activation.success);
    REQUIRE(activation.execution_validated);

    CheckReductionFixtureOnActiveDevice(fixture, true);
}

TEST_CASE("Tensor reductions match PyTorch on every installed accelerator",
          "[tensor][tensor_reduction][accelerator]") {
    const json fixture = LoadReductionFixture();
    const auto devices = cyxwiz::Device::GetAvailableDevices();
    size_t exercised_routes = 0;
    for (const auto& device : devices) {
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
        REQUIRE(activation.requested_type == device.type);
        REQUIRE(activation.effective_type == device.type);
        REQUIRE(activation.requested_device_id == device.device_id);
        REQUIRE(activation.effective_device_id == device.device_id);
        CheckReductionFixtureOnActiveDevice(
            fixture, !device.supports_fp64_known || device.supports_fp64);
        ++exercised_routes;
    }
    if (exercised_routes == 0) {
        WARN("No CUDA or OpenCL route was available for reduction parity");
    }
    const auto restore =
        cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(true);
    REQUIRE(restore.success);
}

TEST_CASE("Tensor reduction empty and scalar domains match PyTorch semantics",
          "[tensor][tensor_reduction][empty]") {
    const cyxwiz::Tensor empty({2, 0, 3}, cyxwiz::DataType::Float32);
    const cyxwiz::Tensor sum = empty.Sum(1);
    const cyxwiz::Tensor product = empty.Prod(-2, true);
    const cyxwiz::Tensor mean = empty.Mean(1);
    const cyxwiz::Tensor variance = empty.Var(1, false, 0);
    const cyxwiz::Tensor deviation = empty.Std(1, false, 0);
    REQUIRE(sum.Shape() == std::vector<size_t>{2, 3});
    REQUIRE(product.Shape() == std::vector<size_t>{2, 1, 3});
    for (size_t index = 0; index < sum.NumElements(); ++index) {
        REQUIRE(sum.ReadData<float>()[index] == 0.0f);
        REQUIRE(product.ReadData<float>()[index] == 1.0f);
        REQUIRE(std::isnan(mean.ReadData<float>()[index]));
        REQUIRE(std::isnan(variance.ReadData<float>()[index]));
        REQUIRE(std::isnan(deviation.ReadData<float>()[index]));
    }
    REQUIRE_THROWS_AS(empty.Max(1), std::runtime_error);
    REQUIRE_THROWS_AS(empty.Min(1), std::runtime_error);

    const cyxwiz::Tensor globally_empty({0}, cyxwiz::DataType::Float64);
    REQUIRE(globally_empty.Sum().Shape().empty());
    REQUIRE(globally_empty.Sum().ReadData<double>()[0] == 0.0);
    REQUIRE(globally_empty.Prod().ReadData<double>()[0] == 1.0);
    REQUIRE(std::isnan(globally_empty.Mean().ReadData<double>()[0]));
    REQUIRE(std::isnan(globally_empty.Var().ReadData<double>()[0]));
    REQUIRE(std::isnan(globally_empty.Std().ReadData<double>()[0]));
    REQUIRE_THROWS_AS(globally_empty.Max(), std::runtime_error);
    REQUIRE_THROWS_AS(globally_empty.Min(), std::runtime_error);

    const float scalar_value = 3.5f;
    const cyxwiz::Tensor scalar({}, &scalar_value, cyxwiz::DataType::Float32);
    REQUIRE(scalar.Mean(0, true).Shape().empty());
    REQUIRE(scalar.Sum(-1).ReadData<float>()[0] == scalar_value);
    REQUIRE_THROWS_AS(scalar.Sum(1), std::runtime_error);
}

TEST_CASE("Tensor variance correction matches PyTorch singular domains",
          "[tensor][tensor_reduction][correction]") {
    const float values[] = {1.0f, 2.0f, 3.0f};
    const cyxwiz::Tensor input({3}, values, cyxwiz::DataType::Float32);
    REQUIRE(input.VarWithCorrection(-1).ReadData<float>()[0] ==
            Catch::Approx(0.5f));
    REQUIRE(input.VarWithCorrection(1).ReadData<float>()[0] ==
            Catch::Approx(1.0f));
    REQUIRE(std::isinf(input.VarWithCorrection(3).ReadData<float>()[0]));
}

TEST_CASE("Tensor global empty reductions remain device resident",
          "[tensor][tensor_reduction][arrayfire][empty][residency]") {
    const auto activation =
        cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(true);
    REQUIRE(activation.success);
    const cyxwiz::Tensor empty({0}, cyxwiz::DataType::Float32);

    g_reduction_host_sync_count = 0;
    g_reduction_fallback_count = 0;
    std::vector<cyxwiz::Tensor> outputs;
    {
        const cyxwiz::ScopedArrayFireFallbackPolicy strict(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_observer(
            &CountReductionHostSync);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CountReductionFallback);
        outputs = {
            empty.Sum(), empty.Prod(), empty.Mean(), empty.Var(), empty.Std()};
        for (const auto& output : outputs) {
            output.GetSemanticArray().eval();
        }
    }

    REQUIRE(g_reduction_host_sync_count == 0);
    REQUIRE(g_reduction_fallback_count == 0);
    REQUIRE(outputs[0].ReadData<float>()[0] == 0.0f);
    REQUIRE(outputs[1].ReadData<float>()[0] == 1.0f);
    REQUIRE(std::isnan(outputs[2].ReadData<float>()[0]));
    REQUIRE(std::isnan(outputs[3].ReadData<float>()[0]));
    REQUIRE(std::isnan(outputs[4].ReadData<float>()[0]));
}

// TOFIX140 F4: rank-five reductions run on the device (row-major flat view,
// reduced as [after, axis, before]). Every axis, with and without keepdim,
// against a host reference over the same values.
TEST_CASE("Tensor rank-five reductions run on the device",
          "[tensor][tensor_reduction][arrayfire]") {
    REQUIRE(cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(true).success);
    const std::vector<size_t> shape{2, 3, 1, 2, 4};
    const cyxwiz::Tensor input =
        cyxwiz::Tensor::RandomSeeded(shape, 140, cyxwiz::DataType::Float32) * 2.0f + 0.5f;
    std::vector<double> values(input.NumElements());
    {
        const float* host = input.ReadData<float>();
        for (size_t i = 0; i < values.size(); ++i) values[i] = host[i];
    }
    input.GetRowMajorFlatArray().eval();

    struct Reference { double sum = 0, prod = 1, max = -1e300, min = 1e300, mean = 0, var = 0; };
    const auto reduce = [](const std::vector<double>& group) {
        Reference r;
        for (double v : group) { r.sum += v; r.prod *= v; r.max = (std::max)(r.max, v); r.min = (std::min)(r.min, v); }
        r.mean = r.sum / static_cast<double>(group.size());
        for (double v : group) r.var += (v - r.mean) * (v - r.mean);
        r.var /= static_cast<double>(group.size());
        return r;
    };
    const auto check = [](const cyxwiz::Tensor& actual, const std::vector<double>& expected, double tolerance) {
        REQUIRE(actual.NumElements() == expected.size());
        const float* host = actual.ReadData<float>();
        for (size_t i = 0; i < expected.size(); ++i) {
            CAPTURE(i);
            REQUIRE(host[i] == Catch::Approx(expected[i]).margin(tolerance).epsilon(1e-5));
        }
    };

    g_reduction_fallback_count = 0;
    {
        const cyxwiz::ScopedArrayFireFallbackPolicy strict(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver observer(
            &CountReductionFallback);
        const Reference all = reduce(values);
        REQUIRE(input.Sum().Shape().empty());
        check(input.Sum(), {all.sum}, 1e-4);
        check(input.Mean(), {all.mean}, 1e-5);
        check(input.Max(), {all.max}, 0.0);
        check(input.Min(), {all.min}, 0.0);
        check(input.Prod(), {all.prod}, 1e-3);
        check(input.Var(), {all.var}, 1e-5);
        check(input.Std(), {std::sqrt(all.var)}, 1e-5);

        for (int axis = 0; axis < 5; ++axis) {
            CAPTURE(axis);
            // Row-major: index = (before * extent + k) * after + a.
            size_t before = 1, after = 1;
            for (int i = 0; i < axis; ++i) before *= shape[static_cast<size_t>(i)];
            for (size_t i = static_cast<size_t>(axis) + 1; i < shape.size(); ++i) after *= shape[i];
            const size_t extent = shape[static_cast<size_t>(axis)];
            std::vector<double> sum, mean, max, min, prod, var, std_dev;
            for (size_t b = 0; b < before; ++b) {
                for (size_t a = 0; a < after; ++a) {
                    std::vector<double> group;
                    for (size_t k = 0; k < extent; ++k) group.push_back(values[(b * extent + k) * after + a]);
                    const Reference r = reduce(group);
                    sum.push_back(r.sum); mean.push_back(r.mean); max.push_back(r.max); min.push_back(r.min);
                    prod.push_back(r.prod); var.push_back(r.var); std_dev.push_back(std::sqrt(r.var));
                }
            }
            for (bool keepdim : {false, true}) {
                CAPTURE(keepdim);
                std::vector<size_t> expected_shape = shape;
                if (keepdim) expected_shape[static_cast<size_t>(axis)] = 1;
                else expected_shape.erase(expected_shape.begin() + axis);
                REQUIRE(input.Sum(axis, keepdim).Shape() == expected_shape);
                check(input.Sum(axis, keepdim), sum, 1e-5);
                check(input.Mean(axis, keepdim), mean, 1e-5);
                check(input.Max(axis, keepdim), max, 0.0);
                check(input.Min(axis, keepdim), min, 0.0);
                check(input.Prod(axis, keepdim), prod, 1e-5);
                check(input.Var(axis, keepdim), var, 1e-5);
                check(input.Std(axis, keepdim), std_dev, 1e-5);
            }
        }
    }
    REQUIRE(g_reduction_fallback_count == 0);
}
