#include "cyxwiz/layers/recurrent.h"
#include "cyxwiz/recurrent_cuda_placement.h"
#include "cyxwiz/tensor.h"

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
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

void RunLstmProfileSmoke() {
    constexpr size_t kBatchSize = 8;
    constexpr size_t kSeqLen = 8;
    constexpr size_t kInputSize = 16;
    constexpr int kHiddenSize = 8;
    constexpr int kNumLayers = 1;
    constexpr bool kBidirectional = false;

    cyxwiz::RecurrentCudaPlacementRequest request;
    request.kind = cyxwiz::RecurrentLayerKind::LSTM;
    request.batch_size = kBatchSize;
    request.seq_len = kSeqLen;
    request.input_size = kInputSize;
    request.hidden_size = kHiddenSize;
    request.num_layers = kNumLayers;
    request.bidirectional = kBidirectional;
    request.return_sequences = true;
    const auto decision = cyxwiz::EvaluateRecurrentCudaPlacement(request);

    Check(decision.reason_code ==
              cyxwiz::RecurrentCudaPlacementReason::ArrayFireCudaAllowedByEstimator,
          "small LSTM smoke shape should stay ArrayFire-CUDA eligible by policy");
    Check(decision.should_attempt_arrayfire_cuda,
          "small LSTM smoke shape should attempt ArrayFire CUDA when CUDA is active");

    cyxwiz::LSTMLayer lstm(
        static_cast<int>(kInputSize),
        kHiddenSize,
        kNumLayers,
        true,
        kBidirectional,
        0.0f);
    cyxwiz::Tensor input = MakeInput(kBatchSize, kSeqLen, kInputSize);

    lstm.ResetState();
    const auto forward_start = std::chrono::steady_clock::now();
    cyxwiz::Tensor output = lstm.Forward(input);
    const auto forward_end = std::chrono::steady_clock::now();

    Check(output.Shape().size() == 3,
          "LSTM forward output should be rank-3 for sequence input");
    Check(output.Shape()[0] == kBatchSize,
          "LSTM forward should preserve batch dimension");
    Check(output.Shape()[2] == static_cast<size_t>(kHiddenSize),
          "LSTM forward should emit hidden_size features for single-direction LSTM");

    cyxwiz::Tensor grad_output = MakeOnesLike(output);
    const auto backward_start = std::chrono::steady_clock::now();
    cyxwiz::Tensor grad_input = lstm.Backward(grad_output);
    const auto backward_end = std::chrono::steady_clock::now();

    Check(grad_input.Shape() == input.Shape(),
          "LSTM backward should return an input-shaped gradient");

    std::cout << "recurrent_af_profile_smoke\n"
              << "  backend=" << ActiveArrayFireBackendName() << "\n"
              << "  scenario=lstm_forward_backward_small\n"
              << "  batch_size=" << kBatchSize
              << " seq_len=" << kSeqLen
              << " input_size=" << kInputSize
              << " hidden_size=" << kHiddenSize
              << " layers=" << kNumLayers
              << " bidirectional=false\n"
              << "  placement_reason=" << decision.reason_code
              << " estimated_formal_parameter_bytes="
              << decision.estimated_formal_parameter_bytes
              << " limit_bytes="
              << decision.formal_parameter_limit_bytes << "\n"
              << "  forward_ms="
              << ElapsedMilliseconds(forward_start, forward_end)
              << " backward_ms="
              << ElapsedMilliseconds(backward_start, backward_end)
              << "\n"
              << "  hotspot_candidates=input_projection,per_step_gate_math,"
                 "af_to_tensor_cache_materialization,cpu_backward\n";
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
