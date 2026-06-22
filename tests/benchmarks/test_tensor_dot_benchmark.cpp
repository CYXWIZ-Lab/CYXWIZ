#include "cyxwiz/tensor.h"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace {

constexpr size_t kBatch = 1024;
constexpr size_t kFeatures = 512;
constexpr int kWarmupIterations = 3;
constexpr int kMeasuredIterations = 20;

std::vector<float> MakeData(float scale) {
    std::vector<float> values(kBatch * kFeatures);
    for (size_t i = 0; i < values.size(); i++) {
        values[i] = scale * static_cast<float>((i % 97) + 1) / 97.0f;
    }
    return values;
}

double RunRowWiseDotBenchmark(const cyxwiz::Tensor& left,
                              const cyxwiz::Tensor& right,
                              int iterations,
                              float& checksum) {
    const auto start = std::chrono::steady_clock::now();
    checksum = 0.0f;

    for (int i = 0; i < iterations; i++) {
        cyxwiz::Tensor result = left.Dot(right);
        if (result.Shape() != std::vector<size_t>{kBatch, 1}) {
            throw std::runtime_error("TensorDot benchmark: unexpected result shape");
        }
        const float* out = result.Data<float>();
        checksum += out[0];
        checksum += out[kBatch - 1];
    }

    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

} // namespace

int main() {
    const std::vector<float> left_data = MakeData(1.0f);
    const std::vector<float> right_data = MakeData(0.5f);

    cyxwiz::Tensor left_host({kBatch, kFeatures}, left_data.data(), cyxwiz::DataType::Float32);
    cyxwiz::Tensor right_host({kBatch, kFeatures}, right_data.data(), cyxwiz::DataType::Float32);

#ifdef CYXWIZ_HAS_ARRAYFIRE
    cyxwiz::Tensor left =
        cyxwiz::Tensor::FromArrayRowMajor2D(left_host.GetArrayRowMajor2D());
    cyxwiz::Tensor right =
        cyxwiz::Tensor::FromArrayRowMajor2D(right_host.GetArrayRowMajor2D());
    const char* backend = "ArrayFire row-major 2D";
#else
    const cyxwiz::Tensor& left = left_host;
    const cyxwiz::Tensor& right = right_host;
    const char* backend = "CPU fallback";
#endif

    float warmup_checksum = 0.0f;
    RunRowWiseDotBenchmark(left, right, kWarmupIterations, warmup_checksum);

    float checksum = 0.0f;
    const double elapsed_ms =
        RunRowWiseDotBenchmark(left, right, kMeasuredIterations, checksum);

    if (!std::isfinite(checksum)) {
        throw std::runtime_error("TensorDot benchmark: checksum is not finite");
    }

    std::cout << "TensorDot row-wise benchmark\n";
    std::cout << "backend=" << backend << "\n";
    std::cout << "shape=[" << kBatch << "," << kFeatures << "]\n";
    std::cout << "iterations=" << kMeasuredIterations << "\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "total_ms=" << elapsed_ms << "\n";
    std::cout << "avg_ms=" << (elapsed_ms / static_cast<double>(kMeasuredIterations)) << "\n";
    std::cout << "checksum=" << checksum << "\n";

    return 0;
}
