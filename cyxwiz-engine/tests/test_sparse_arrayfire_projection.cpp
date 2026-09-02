#include <cyxwiz/layers/linear.h>
#include <cyxwiz/sequential.h>
#include <cyxwiz/tensor.h>

#include "algorithms/arrayfire_backend_utils.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

#ifdef CYXWIZ_HAS_ARRAYFIRE

struct SparseLinearFixture {
    std::vector<size_t> input_shape;
    std::vector<float> input_dense;
    std::vector<int32_t> row_offsets;
    std::vector<int32_t> column_indices;
    std::vector<float> values;
    std::vector<size_t> weight_shape;
    std::vector<float> weight;
    std::vector<float> bias;
    std::vector<float> grad_output;
    std::vector<float> expected_output;
    std::vector<float> expected_weight_grad;
    std::vector<float> expected_bias_grad;
};

SparseLinearFixture LoadPyTorchFixture(const char* executable_path) {
    const auto path = std::filesystem::path(executable_path).parent_path() /
        "computation_truth_fixtures" / "sparse_linear_pytorch.json";
    std::ifstream input(path);
    Check(input.good(), "missing PyTorch sparse Linear fixture: " + path.string());
    const auto document = nlohmann::json::parse(input);
    Check(document.at("reference").at("framework") == "PyTorch",
          "sparse Linear fixture must identify PyTorch as its oracle");
    const auto& test_case = document.at("case");
    SparseLinearFixture fixture;
    fixture.input_shape =
        test_case.at("input_shape").get<std::vector<size_t>>();
    fixture.input_dense =
        test_case.at("input_dense").get<std::vector<float>>();
    fixture.row_offsets =
        test_case.at("row_offsets").get<std::vector<int32_t>>();
    fixture.column_indices =
        test_case.at("column_indices").get<std::vector<int32_t>>();
    fixture.values = test_case.at("values").get<std::vector<float>>();
    fixture.weight_shape =
        test_case.at("weight_shape").get<std::vector<size_t>>();
    fixture.weight = test_case.at("weight").get<std::vector<float>>();
    fixture.bias = test_case.at("bias").get<std::vector<float>>();
    fixture.grad_output =
        test_case.at("grad_output").get<std::vector<float>>();
    fixture.expected_output =
        test_case.at("expected_output").get<std::vector<float>>();
    fixture.expected_weight_grad =
        test_case.at("expected_weight_grad").get<std::vector<float>>();
    fixture.expected_bias_grad =
        test_case.at("expected_bias_grad").get<std::vector<float>>();
    return fixture;
}

struct BackendRequest {
    af::Backend backend;
    const char* name;
};

std::optional<BackendRequest> ParseBackend(const std::string& argument) {
    if (argument == "cpu") {
        return BackendRequest{AF_BACKEND_CPU, "cpu"};
    }
    if (argument == "cuda") {
        return BackendRequest{AF_BACKEND_CUDA, "cuda"};
    }
    if (argument == "opencl") {
        return BackendRequest{AF_BACKEND_OPENCL, "opencl"};
    }
    if (argument == "oneapi") {
        return BackendRequest{AF_BACKEND_ONEAPI, "oneapi"};
    }
    return std::nullopt;
}

void CheckMatrix(const af::array& actual,
                 dim_t rows,
                 dim_t columns,
                 const std::vector<float>& expected,
                 const std::string& message) {
    Check(actual.dims(0) == rows && actual.dims(1) == columns,
          message + " shape mismatch");
    Check(actual.type() == f32, message + " dtype mismatch");

    std::vector<float> host(expected.size());
    actual.host(host.data());
    for (size_t index = 0; index < expected.size(); ++index) {
        Check(std::fabs(host[index] - expected[index]) < 1e-6f,
              message + " value mismatch at column-major index " +
                  std::to_string(index));
    }
}

void CheckTensor(const cyxwiz::Tensor& actual,
                 const std::vector<size_t>& shape,
                 const std::vector<float>& expected,
                 const std::string& message) {
    Check(actual.Shape() == shape, message + " shape mismatch");
    const float* values = actual.ReadData<float>();
    for (size_t index = 0; index < expected.size(); ++index) {
        Check(std::fabs(values[index] - expected[index]) < 1e-5f,
              message + " value mismatch at row-major index " +
                  std::to_string(index));
    }
}

void TestSparseForwardAndWeightGradient(const BackendRequest& request) {
    af::setBackend(request.backend);
    af::setDevice(0);

    // X = [[1, 0, 2, 0],
    //      [0, 3, 0, 4]]
    const std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<int> row_offsets = {0, 2, 4};
    const std::vector<int> column_indices = {0, 2, 1, 3};
    const af::array sparse_features = af::sparse(
        2, 4, 4, values.data(), row_offsets.data(), column_indices.data(),
        f32, AF_STORAGE_CSR, afHost);
    Check(sparse_features.issparse(), "feature matrix should remain sparse");

    // W has shape [features, outputs]. ArrayFire host arrays are column-major.
    // W = [[1, 5], [2, 6], [3, 7], [4, 8]]
    const std::vector<float> weights = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
    };
    const af::array weight_view(4, 2, weights.data());
    const af::array forward = af::matmul(
        sparse_features, weight_view, AF_MAT_NONE, AF_MAT_NONE);
    CheckMatrix(forward, 2, 2, {7.0f, 22.0f, 19.0f, 50.0f},
                "sparse forward projection");

    // dY = [[1, 2], [3, 4]]. X^T @ dY produces [features, outputs].
    const std::vector<float> output_gradient = {1.0f, 3.0f, 2.0f, 4.0f};
    const af::array output_gradient_array(2, 2, output_gradient.data());
    const af::array weight_gradient = af::matmul(
        sparse_features, output_gradient_array, AF_MAT_TRANS, AF_MAT_NONE);
    CheckMatrix(weight_gradient, 4, 2,
                {1.0f, 9.0f, 2.0f, 12.0f,
                 2.0f, 12.0f, 4.0f, 16.0f},
                "sparse weight gradient");

    af::sync();
}

void TestLinearAndSequentialSparseParity(
    const BackendRequest& request,
    const SparseLinearFixture& fixture) {
    af::setBackend(request.backend);
    af::setDevice(0);

    Check(fixture.input_shape.size() == 2 &&
              fixture.weight_shape.size() == 2 &&
              fixture.input_shape[1] == fixture.weight_shape[1] &&
              fixture.bias.size() == fixture.weight_shape[0],
          "PyTorch sparse Linear fixture shape contract mismatch");
    const size_t rows = fixture.input_shape[0];
    const size_t in_features = fixture.input_shape[1];
    const size_t out_features = fixture.weight_shape[0];
    const cyxwiz::LinearSparseCsrBatchView sparse_view{
        rows,
        in_features,
        fixture.values.size(),
        fixture.row_offsets.data(),
        fixture.column_indices.data(),
        fixture.values.data()};
    const cyxwiz::Tensor dense_input(
        fixture.input_shape,
        fixture.input_dense.data(),
        cyxwiz::DataType::Float32);
    const cyxwiz::Tensor weight(
        fixture.weight_shape,
        fixture.weight.data(),
        cyxwiz::DataType::Float32);
    const cyxwiz::Tensor bias(
        {out_features}, fixture.bias.data(), cyxwiz::DataType::Float32);
    const cyxwiz::Tensor output_gradient(
        {rows, out_features},
        fixture.grad_output.data(),
        cyxwiz::DataType::Float32);

    const cyxwiz::ScopedArrayFireFallbackPolicy strict_policy(
        cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);

    cyxwiz::LinearLayer layer(in_features, out_features, true);
    layer.SetParameters({{"weight", weight}, {"bias", bias}});
    const cyxwiz::Tensor dense_output = layer.Forward(dense_input);
    (void)layer.Backward(output_gradient);
    const auto dense_gradients = layer.GetGradients();

    const cyxwiz::Tensor sparse_output = layer.ForwardSparseCsr(sparse_view);
    layer.BackwardSparseCsr(sparse_view, output_gradient);
    const auto sparse_gradients = layer.GetGradients();

    const std::vector<size_t> output_shape = {rows, out_features};
    CheckTensor(dense_output, output_shape, fixture.expected_output,
                "dense Linear PyTorch reference output");
    CheckTensor(sparse_output, output_shape, fixture.expected_output,
                "sparse Linear PyTorch reference output");
    CheckTensor(dense_gradients.at("weight"), fixture.weight_shape,
                fixture.expected_weight_grad,
                "dense Linear PyTorch weight gradient");
    CheckTensor(sparse_gradients.at("weight"), fixture.weight_shape,
                fixture.expected_weight_grad,
                "sparse Linear PyTorch weight gradient");
    CheckTensor(sparse_gradients.at("bias"), {out_features},
                fixture.expected_bias_grad,
                "sparse Linear bias gradient");

    cyxwiz::SequentialModel model;
    model.Add<cyxwiz::LinearModule>(in_features, out_features, true);
    Check(model.SupportsSparseCsrInput(),
          "Linear-first SequentialModel must advertise sparse support");
    model.SetParameters({
        {"layer0.weight", weight},
        {"layer0.bias", bias},
    });
    const cyxwiz::Tensor sequential_output =
        model.ForwardSparseCsr(sparse_view);
    model.BackwardSparseCsr(sparse_view, output_gradient);
    CheckTensor(sequential_output, output_shape, fixture.expected_output,
                "Sequential sparse output");
    CheckTensor(model.GetGradients().at("layer0.weight"),
                fixture.weight_shape,
                fixture.expected_weight_grad,
                "Sequential sparse weight gradient");

    const std::vector<int32_t> empty_offsets(rows + 1, 0);
    const cyxwiz::LinearSparseCsrBatchView empty_view{
        rows,
        in_features,
        0,
        empty_offsets.data(),
        nullptr,
        nullptr};
    const auto empty_output = layer.ForwardSparseCsr(empty_view);
    std::vector<float> expected_empty_output(rows * out_features);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t output = 0; output < out_features; ++output) {
            expected_empty_output[row * out_features + output] =
                fixture.bias[output];
        }
    }
    CheckTensor(empty_output, output_shape, expected_empty_output,
                "all-empty CSR forward output");
    layer.BackwardSparseCsr(empty_view, output_gradient);
    CheckTensor(layer.GetGradients().at("weight"), fixture.weight_shape,
                std::vector<float>(fixture.weight.size(), 0.0f),
                "all-empty CSR weight gradient");

    const std::vector<int32_t> malformed_offsets = {0, 2, 1};
    const cyxwiz::LinearSparseCsrBatchView malformed_view{
        2,
        in_features,
        1,
        malformed_offsets.data(),
        fixture.column_indices.data(),
        fixture.values.data()};
    bool rejected_malformed = false;
    try {
        (void)layer.ForwardSparseCsr(malformed_view);
    } catch (const std::invalid_argument&) {
        rejected_malformed = true;
    }
    Check(rejected_malformed,
          "Linear sparse projection must reject malformed CSR offsets");
}

#endif

} // namespace

int main(int argc, char** argv) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    const std::string backend_name = argc == 2 ? argv[1] : "cpu";
    const auto request = ParseBackend(backend_name);
    if (!request || argc > 2) {
        std::cerr << "usage: test_sparse_arrayfire_projection "
                     "[cpu|cuda|opencl|oneapi]\n";
        return 2;
    }

    try {
        const auto fixture = LoadPyTorchFixture(argv[0]);
        const int available = af::getAvailableBackends();
        if ((available & static_cast<int>(request->backend)) == 0) {
            std::cout << "SKIP: ArrayFire " << request->name
                      << " backend is not installed\n";
            return 0;
        }
        TestSparseForwardAndWeightGradient(*request);
        TestLinearAndSequentialSparseParity(*request, fixture);
    } catch (const af::exception& error) {
        std::cerr << "FAIL: ArrayFire " << request->name
                  << " sparse projection failed: " << error.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: sparse Linear parity fixture failed: "
                  << error.what() << '\n';
        return 1;
    }
    std::cout << "ArrayFire " << request->name
              << " sparse projection contract tests passed\n";
#else
    std::cout << "SKIP: ArrayFire is not available in this build\n";
#endif
    return 0;
}
