#pragma once

#include <algorithm>
#include <cstddef>
#include <sstream>
#include <string>

namespace cyxwiz {

enum class RecurrentLayerKind {
    LSTM,
    GRU
};

struct RecurrentCudaPlacementRequest {
    RecurrentLayerKind kind = RecurrentLayerKind::GRU;
    size_t batch_size = 0;
    size_t seq_len = 0;
    size_t input_size = 0;
    size_t hidden_size = 0;
    size_t num_layers = 1;
    bool bidirectional = false;
    bool return_sequences = false;
};

struct RecurrentCudaPlacementDecision {
    bool should_attempt_arrayfire_cuda = true;
    size_t estimated_formal_parameter_bytes = 0;
    size_t formal_parameter_limit_bytes = 4096;
    std::string layer_name;
    std::string reason_code;
    std::string expected_backend = "ArrayFire CUDA";
    std::string fallback_backend = "CPU";
    std::string reason;
};

inline const char* RecurrentKindName(RecurrentLayerKind kind) {
    switch (kind) {
        case RecurrentLayerKind::LSTM:
            return "LSTM";
        case RecurrentLayerKind::GRU:
            return "GRU";
    }
    return "Recurrent";
}

inline size_t EstimateArrayFireRecurrentFormalParameterBytes(
    const RecurrentCudaPlacementRequest& request) {
    const size_t hidden = std::max<size_t>(1, request.hidden_size);
    const size_t directions = request.bidirectional ? 2 : 1;
    const size_t layers = std::max<size_t>(1, request.num_layers);

    // Conservative model for ArrayFire CUDA JIT recurrent expressions.
    // This estimates the generated kernel formal-parameter pressure, not
    // normal VRAM usage. It is calibrated from observed NVRTC overflows and
    // intentionally errs on CPU placement when close to CUDA's 4096-byte
    // formal parameter limit.
    const size_t base_bytes = 3000;
    const size_t per_hidden_bytes =
        request.kind == RecurrentLayerKind::GRU ? 24 : 56;
    const size_t direction_bytes = (directions - 1) * 384;
    const size_t layer_bytes = (layers - 1) * 192;
    const size_t sequence_bytes = request.return_sequences ? 96 : 0;

    return base_bytes +
           hidden * per_hidden_bytes * directions +
           direction_bytes +
           layer_bytes +
           sequence_bytes;
}

inline RecurrentCudaPlacementDecision EvaluateRecurrentCudaPlacement(
    const RecurrentCudaPlacementRequest& request) {
    RecurrentCudaPlacementDecision decision;
    decision.layer_name = RecurrentKindName(request.kind);

    if (request.kind == RecurrentLayerKind::GRU) {
        decision.should_attempt_arrayfire_cuda = false;
        decision.reason_code = request.bidirectional
            ? "gru_bidirectional_arrayfire_cuda_disabled"
            : "gru_arrayfire_cuda_probe_required";
        decision.expected_backend = "CPU";
        std::ostringstream msg;
        msg << decision.layer_name
            << " recurrent step is expected to run on CPU instead of "
            << "ArrayFire CUDA for this shape: batch_size="
            << request.batch_size
            << ", seq_len=" << request.seq_len
            << ", input_size=" << request.input_size
            << ", hidden_size=" << request.hidden_size
            << ", layers=" << request.num_layers
            << ", bidirectional="
            << (request.bidirectional ? "true" : "false")
            << ", return_sequences="
            << (request.return_sequences ? "true" : "false")
            << ". The current ArrayFire CUDA GRU recurrent loop is "
            << "probe-required and conservatively CPU-routed because real "
            << "sentiment training produced CUDA generated-kernel formal "
            << "parameter overflows even for single-direction GRU shapes. "
            << "This is separate from VRAM capacity. Training can continue, "
            << "but this GRU recurrent step will run on CPU until a fused/"
            << "native CUDA recurrent kernel or exact backend probe is "
            << "available.";
        decision.reason = msg.str();
        return decision;
    }

    if (request.kind == RecurrentLayerKind::LSTM && request.bidirectional) {
        decision.should_attempt_arrayfire_cuda = false;
        decision.reason_code = "lstm_bidirectional_cuda_jit_param_overflow_risk";
        decision.expected_backend = "CPU";
        std::ostringstream msg;
        msg << decision.layer_name
            << " recurrent step is expected to run on CPU instead of "
            << "ArrayFire CUDA for this shape: batch_size="
            << request.batch_size
            << ", seq_len=" << request.seq_len
            << ", input_size=" << request.input_size
            << ", hidden_size=" << request.hidden_size
            << ", layers=" << request.num_layers
            << ", bidirectional=true"
            << ", return_sequences="
            << (request.return_sequences ? "true" : "false")
            << ". The current ArrayFire CUDA bidirectional LSTM path can "
            << "exceed CUDA's 4096-byte generated-kernel formal parameter "
            << "limit even at small hidden sizes. Training can continue, "
            << "but this bidirectional LSTM recurrent step will run on CPU "
            << "until the GPU path has dedicated correctness and timeout "
            << "coverage.";
        decision.reason = msg.str();
        return decision;
    }

    decision.estimated_formal_parameter_bytes =
        EstimateArrayFireRecurrentFormalParameterBytes(request);
    decision.should_attempt_arrayfire_cuda =
        decision.estimated_formal_parameter_bytes <=
        decision.formal_parameter_limit_bytes;

    if (!decision.should_attempt_arrayfire_cuda) {
        decision.reason_code = "cuda_jit_param_overflow_risk";
        decision.expected_backend = "CPU";
        std::ostringstream msg;
        msg << decision.layer_name
            << " recurrent step is expected to run on CPU instead of "
            << "ArrayFire CUDA for this shape: batch_size="
            << request.batch_size
            << ", seq_len=" << request.seq_len
            << ", input_size=" << request.input_size
            << ", hidden_size=" << request.hidden_size
            << ", layers=" << request.num_layers
            << ", bidirectional="
            << (request.bidirectional ? "true" : "false")
            << ", return_sequences="
            << (request.return_sequences ? "true" : "false")
            << ". Estimated CUDA kernel formal-parameter use is "
            << decision.estimated_formal_parameter_bytes
            << " bytes, above the "
            << decision.formal_parameter_limit_bytes
            << "-byte CUDA limit for generated kernels. Training can "
            << "continue, but this recurrent step may be slower on CPU.";
        decision.reason = msg.str();
    } else {
        decision.reason_code = "arrayfire_cuda_allowed_by_estimator";
    }

    return decision;
}

} // namespace cyxwiz
