#include "cyxwiz/tensor.h"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace {

constexpr size_t kBatch = 1024;
constexpr size_t kFeatures = 512;
constexpr int kWarmupIterations = 3;
constexpr int kMeasuredIterations = 20;

enum class FanInOp {
    Add,
    Multiply,
    Average,
    Concatenate
};

const char* OpName(FanInOp op) {
    switch (op) {
        case FanInOp::Add: return "add";
        case FanInOp::Multiply: return "multiply";
        case FanInOp::Average: return "average";
        case FanInOp::Concatenate: return "concatenate";
    }
    return "unknown";
}

std::vector<float> MakeData(float scale) {
    std::vector<float> values(kBatch * kFeatures);
    for (size_t i = 0; i < values.size(); i++) {
        values[i] = scale * static_cast<float>((i % 113) + 1) / 113.0f;
    }
    return values;
}

cyxwiz::Tensor RunFanInOp(const cyxwiz::Tensor& left,
                          const cyxwiz::Tensor& right,
                          FanInOp op) {
    switch (op) {
        case FanInOp::Add:
            return left + right;
        case FanInOp::Multiply:
            return left * right;
        case FanInOp::Average:
            return (left + right) * 0.5f;
        case FanInOp::Concatenate:
            return cyxwiz::Tensor::Cat({left, right}, 1);
    }
    throw std::runtime_error("Fan-in benchmark: unsupported op");
}

double RunTensorBenchmark(const cyxwiz::Tensor& left,
                          const cyxwiz::Tensor& right,
                          FanInOp op,
                          int iterations,
                          float& checksum) {
    const auto start = std::chrono::steady_clock::now();
    checksum = 0.0f;

    const std::vector<size_t> expected_shape =
        op == FanInOp::Concatenate
            ? std::vector<size_t>{kBatch, kFeatures * 2}
            : std::vector<size_t>{kBatch, kFeatures};

    for (int i = 0; i < iterations; i++) {
        cyxwiz::Tensor result = RunFanInOp(left, right, op);
        if (result.Shape() != expected_shape) {
            throw std::runtime_error("Fan-in benchmark: unexpected result shape");
        }
        const float* out = result.Data<float>();
        checksum += out[0];
        checksum += out[result.NumElements() - 1];
    }

    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double RunCpuReferenceBenchmark(const std::vector<float>& left,
                                const std::vector<float>& right,
                                FanInOp op,
                                int iterations,
                                float& checksum) {
    std::vector<float> output(op == FanInOp::Concatenate
                                  ? kBatch * kFeatures * 2
                                  : kBatch * kFeatures);
    const auto start = std::chrono::steady_clock::now();
    checksum = 0.0f;

    for (int iteration = 0; iteration < iterations; iteration++) {
        if (op == FanInOp::Concatenate) {
            for (size_t row = 0; row < kBatch; row++) {
                const size_t src_offset = row * kFeatures;
                const size_t dst_offset = row * kFeatures * 2;
                for (size_t col = 0; col < kFeatures; col++) {
                    output[dst_offset + col] = left[src_offset + col];
                    output[dst_offset + kFeatures + col] = right[src_offset + col];
                }
            }
        } else {
            for (size_t i = 0; i < left.size(); i++) {
                switch (op) {
                    case FanInOp::Add:
                        output[i] = left[i] + right[i];
                        break;
                    case FanInOp::Multiply:
                        output[i] = left[i] * right[i];
                        break;
                    case FanInOp::Average:
                        output[i] = (left[i] + right[i]) * 0.5f;
                        break;
                    case FanInOp::Concatenate:
                        break;
                }
            }
        }
        checksum += output.front();
        checksum += output.back();
    }

    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void PrintBenchmarkLine(const cyxwiz::Tensor& left,
                        const cyxwiz::Tensor& right,
                        const std::vector<float>& left_data,
                        const std::vector<float>& right_data,
                        FanInOp op) {
    float warmup_checksum = 0.0f;
    RunTensorBenchmark(left, right, op, kWarmupIterations, warmup_checksum);

    float checksum = 0.0f;
    const double elapsed_ms =
        RunTensorBenchmark(left, right, op, kMeasuredIterations, checksum);
    float cpu_checksum = 0.0f;
    const double cpu_elapsed_ms =
        RunCpuReferenceBenchmark(left_data, right_data, op, kMeasuredIterations, cpu_checksum);

    if (!std::isfinite(checksum) || !std::isfinite(cpu_checksum)) {
        throw std::runtime_error("Fan-in benchmark: checksum is not finite");
    }

    std::cout << "op=" << OpName(op)
              << " total_ms=" << elapsed_ms
              << " avg_ms=" << (elapsed_ms / static_cast<double>(kMeasuredIterations))
              << " checksum=" << checksum
              << " cpu_reference_total_ms=" << cpu_elapsed_ms
              << " cpu_reference_avg_ms="
              << (cpu_elapsed_ms / static_cast<double>(kMeasuredIterations))
              << " cpu_reference_checksum=" << cpu_checksum
              << "\n";
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

    std::cout << "Tensor fan-in benchmark\n";
    std::cout << "backend=" << backend << "\n";
    std::cout << "shape=[" << kBatch << "," << kFeatures << "]\n";
    std::cout << "iterations=" << kMeasuredIterations << "\n";
    std::cout << std::fixed << std::setprecision(3);

    PrintBenchmarkLine(left, right, left_data, right_data, FanInOp::Add);
    PrintBenchmarkLine(left, right, left_data, right_data, FanInOp::Multiply);
    PrintBenchmarkLine(left, right, left_data, right_data, FanInOp::Average);
    PrintBenchmarkLine(left, right, left_data, right_data, FanInOp::Concatenate);

    return 0;
}
