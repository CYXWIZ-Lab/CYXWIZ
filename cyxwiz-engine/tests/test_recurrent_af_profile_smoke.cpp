#include "cyxwiz/layers/recurrent.h"
#include "cyxwiz/recurrent_cuda_placement.h"
#include "cyxwiz/tensor.h"

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

std::string ActiveArrayFireBackendName() {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        switch (af::getActiveBackend()) {
            case AF_BACKEND_CPU: return "ArrayFire CPU";
            case AF_BACKEND_CUDA: return "ArrayFire CUDA";
            case AF_BACKEND_OPENCL: return "ArrayFire OpenCL";
            default: return "ArrayFire unknown";
        }
    } catch (const af::exception& e) {
        return std::string("ArrayFire unavailable: ") + e.what();
    }
#else
    return "ArrayFire not compiled";
#endif
}

cyxwiz::Tensor MakeInput(size_t batch_size,
                         size_t seq_len,
                         size_t input_size) {
    cyxwiz::Tensor input(std::vector<size_t>{batch_size, seq_len, input_size});
    float* data = input.Data<float>();
    for (size_t i = 0; i < input.NumElements(); ++i) {
        data[i] = std::sin(static_cast<float>(i % 29) * 0.17f) * 0.1f;
    }
    return input;
}

cyxwiz::Tensor MakeOnesLike(const cyxwiz::Tensor& reference) {
    cyxwiz::Tensor output(reference.Shape());
    float* data = output.Data<float>();
    for (size_t i = 0; i < output.NumElements(); ++i) {
        data[i] = 1.0f;
    }
    return output;
}

double ElapsedMilliseconds(
    const std::chrono::steady_clock::time_point& start,
    const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int EnvPositiveIntOrDefault(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return fallback;
    }
    try {
        const int parsed = std::stoi(value);
        return parsed > 0 ? parsed : fallback;
    } catch (const std::exception&) {
        return fallback;
    }
}

size_t EnvPositiveSizeOrDefault(const char* name, size_t fallback) {
    return static_cast<size_t>(
        EnvPositiveIntOrDefault(name, static_cast<int>(fallback)));
}

struct TimingStats {
    double min_ms = 0.0;
    double median_ms = 0.0;
    double max_ms = 0.0;
    double mean_ms = 0.0;
};

struct LstmProfileShape {
    size_t batch_size = 8;
    size_t seq_len = 8;
    size_t input_size = 16;
    int hidden_size = 8;
    int num_layers = 1;
};

TimingStats SummarizeTimings(std::vector<double> values) {
    Check(!values.empty(), "timing summary requires at least one value");
    std::sort(values.begin(), values.end());

    double sum = 0.0;
    for (double value : values) {
        sum += value;
    }

    TimingStats stats;
    stats.min_ms = values.front();
    stats.max_ms = values.back();
    stats.mean_ms = sum / static_cast<double>(values.size());
    const size_t mid = values.size() / 2;
    stats.median_ms = (values.size() % 2 == 0)
        ? 0.5 * (values[mid - 1] + values[mid])
        : values[mid];
    return stats;
}

void PrintStatsLine(const char* name, const TimingStats& stats) {
    std::cout << "  " << name
              << "_min_ms=" << stats.min_ms
              << " " << name << "_median_ms=" << stats.median_ms
              << " " << name << "_max_ms=" << stats.max_ms
              << " " << name << "_mean_ms=" << stats.mean_ms
              << "\n";
}

void RunLstmProfileShape(const LstmProfileShape& shape,
                         int measured_runs,
                         int warmup_runs,
                         int backward_eval_interval,
                         const char* scenario) {
    constexpr bool kBidirectional = false;
    const bool default_shape =
        shape.batch_size == 8 &&
        shape.seq_len == 8 &&
        shape.input_size == 16 &&
        shape.hidden_size == 8 &&
        shape.num_layers == 1;

    cyxwiz::RecurrentCudaPlacementRequest request;
    request.kind = cyxwiz::RecurrentLayerKind::LSTM;
    request.batch_size = shape.batch_size;
    request.seq_len = shape.seq_len;
    request.input_size = shape.input_size;
    request.hidden_size = shape.hidden_size;
    request.num_layers = shape.num_layers;
    request.bidirectional = kBidirectional;
    request.return_sequences = true;
    const auto decision = cyxwiz::EvaluateRecurrentCudaPlacement(request);

    if (default_shape) {
        Check(decision.reason_code ==
                  cyxwiz::RecurrentCudaPlacementReason::ArrayFireCudaAllowedByEstimator,
              "small LSTM smoke shape should stay ArrayFire-CUDA eligible by policy");
        Check(decision.should_attempt_arrayfire_cuda,
              "small LSTM smoke shape should attempt ArrayFire CUDA when CUDA is active");
    }

    const int total_runs = warmup_runs + measured_runs;

    cyxwiz::Tensor input =
        MakeInput(shape.batch_size, shape.seq_len, shape.input_size);

    std::vector<double> forward_ms;
    std::vector<double> backward_ms;
    forward_ms.reserve(static_cast<size_t>(measured_runs));
    backward_ms.reserve(static_cast<size_t>(measured_runs));

    for (int run = 0; run < total_runs; ++run) {
        cyxwiz::LSTMLayer lstm(
            static_cast<int>(shape.input_size),
            shape.hidden_size,
            shape.num_layers,
            true,
            kBidirectional,
            0.0f);

        lstm.ResetState();
        const auto forward_start = std::chrono::steady_clock::now();
        cyxwiz::Tensor output = lstm.Forward(input);
        const auto forward_end = std::chrono::steady_clock::now();

        Check(output.Shape().size() == 3,
              "LSTM forward output should be rank-3 for sequence input");
        Check(output.Shape()[0] == shape.batch_size,
              "LSTM forward should preserve batch dimension");
        Check(output.Shape()[2] == static_cast<size_t>(shape.hidden_size),
              "LSTM forward should emit hidden_size features for single-direction LSTM");

        cyxwiz::Tensor grad_output = MakeOnesLike(output);
        const auto backward_start = std::chrono::steady_clock::now();
        cyxwiz::Tensor grad_input = lstm.Backward(grad_output);
        const auto backward_end = std::chrono::steady_clock::now();

        Check(grad_input.Shape() == input.Shape(),
              "LSTM backward should return an input-shaped gradient");

        if (run >= warmup_runs) {
            forward_ms.push_back(ElapsedMilliseconds(forward_start, forward_end));
            backward_ms.push_back(ElapsedMilliseconds(backward_start, backward_end));
        }
    }

    const TimingStats forward_stats = SummarizeTimings(forward_ms);
    const TimingStats backward_stats = SummarizeTimings(backward_ms);

    std::cout << "recurrent_af_profile_smoke\n"
              << "  backend=" << ActiveArrayFireBackendName() << "\n"
              << "  scenario=" << scenario << "\n"
              << "  batch_size=" << shape.batch_size
              << " seq_len=" << shape.seq_len
              << " input_size=" << shape.input_size
              << " hidden_size=" << shape.hidden_size
              << " layers=" << shape.num_layers
              << " bidirectional=false\n"
              << "  placement_reason=" << decision.reason_code
              << " estimated_formal_parameter_bytes="
              << decision.estimated_formal_parameter_bytes
              << " limit_bytes="
              << decision.formal_parameter_limit_bytes << "\n"
              << "  measured_runs=" << measured_runs
              << " warmup_runs=" << warmup_runs
              << " backward_eval_interval=" << backward_eval_interval
              << " env=CYXWIZ_RECURRENT_PROFILE_RUNS"
              << "\n"
              << "  shape_env=CYXWIZ_RECURRENT_PROFILE_BATCH,"
                 "CYXWIZ_RECURRENT_PROFILE_SEQ,"
                 "CYXWIZ_RECURRENT_PROFILE_INPUT,"
                 "CYXWIZ_RECURRENT_PROFILE_HIDDEN,"
                 "CYXWIZ_RECURRENT_PROFILE_LAYERS\n"
              << "  hotspot_candidates=input_projection,per_step_gate_math,"
                 "af_to_tensor_cache_materialization,cpu_backward\n";
    PrintStatsLine("forward", forward_stats);
    PrintStatsLine("backward", backward_stats);
}

void RunLstmProfileSmoke() {
    const int measured_runs =
        EnvPositiveIntOrDefault("CYXWIZ_RECURRENT_PROFILE_RUNS", 1);
    const int warmup_runs = measured_runs > 1
        ? EnvPositiveIntOrDefault("CYXWIZ_RECURRENT_PROFILE_WARMUP", 1)
        : 0;
    const int backward_eval_interval =
        EnvPositiveIntOrDefault("CYXWIZ_LSTM_AF_BACKWARD_EVAL_INTERVAL", 1);

    if (std::getenv("CYXWIZ_RECURRENT_PROFILE_MATRIX")) {
        const std::vector<LstmProfileShape> shapes = {
            {8, 8, 16, 8, 1},
            {16, 16, 32, 16, 1},
            {16, 32, 32, 16, 1},
        };
        for (const auto& shape : shapes) {
            RunLstmProfileShape(
                shape,
                measured_runs,
                warmup_runs,
                backward_eval_interval,
                "lstm_forward_backward_matrix");
        }
        return;
    }

    LstmProfileShape shape;
    shape.batch_size =
        EnvPositiveSizeOrDefault("CYXWIZ_RECURRENT_PROFILE_BATCH", 8);
    shape.seq_len =
        EnvPositiveSizeOrDefault("CYXWIZ_RECURRENT_PROFILE_SEQ", 8);
    shape.input_size =
        EnvPositiveSizeOrDefault("CYXWIZ_RECURRENT_PROFILE_INPUT", 16);
    shape.hidden_size =
        EnvPositiveIntOrDefault("CYXWIZ_RECURRENT_PROFILE_HIDDEN", 8);
    shape.num_layers =
        EnvPositiveIntOrDefault("CYXWIZ_RECURRENT_PROFILE_LAYERS", 1);

    RunLstmProfileShape(
        shape,
        measured_runs,
        warmup_runs,
        backward_eval_interval,
        "lstm_forward_backward_single");
}

} // namespace

int main() {
    try {
        RunLstmProfileSmoke();
    } catch (const std::exception& e) {
        std::cerr << "FAIL: recurrent AF profile smoke threw: "
                  << e.what() << "\n";
        return 1;
    }

    std::cout << "Recurrent AF profile smoke passed\n";
    return 0;
}
