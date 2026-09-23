#pragma once

#include "api_export.h"

#include <algorithm>
#include <cstddef>
#include <sstream>
#include <string>

namespace cyxwiz {

enum class RecurrentLayerKind {
    LSTM,
    GRU
};

// tofix67 slice 5: name of the recurrent staged ArrayFire execution plan.
// The LSTM ArrayFire forward materializes per timestep (input projection,
// recurrent projection, gate combination, state update), which bounds CUDA
// JIT fusion below the formal-parameter limit for estimator-approved
// shapes. Placement explanations reference this name so evidence and
// support bundles can identify the exact plan; bump the version suffix when
// the staging boundaries change.
//
// tofix67 slice 7 benchmark verdict (2026-09-22, GTX 1050 Ti, see
// track67.md): the estimator caps CUDA eligibility at hidden<=~19, and at
// those sizes the staged CUDA path measured 1.4-2.7x SLOWER than the
// native CPU forward — staged ArrayFire CUDA has no winning shape on this
// class of hardware, which meets the ticket's native-provider graduation
// criteria (unsafe beyond the cap, slower within it). On the ArrayFire CPU
// backend the v1 boundaries win up to 5.1x over native at hidden>=64, so
// the staging itself is not the bottleneck and eval-boundary pruning was
// rejected by evidence. A native/fused recurrent provider is the only
// route to real GPU recurrent training; building it awaits the dependency
// decision (cuDNN vs custom kernels) per the ticket's non-goals.
inline constexpr const char* RecurrentStagedArrayFirePlanName =
    "recurrent_timestep_materialization_v1";

// Test-only: force ShouldUseArrayFireRecurrentForward to route to the
// native CPU recurrent path, so staged-ArrayFire vs native parity can be
// exercised in one process. Never set in production code.
CYXWIZ_API void SetForceNativeRecurrentForwardForTesting(bool force);

struct RecurrentCudaPlacementRequest {
    RecurrentLayerKind kind = RecurrentLayerKind::GRU;
    size_t batch_size = 0;
    size_t seq_len = 0;
    size_t input_size = 0;
    size_t hidden_size = 0;
    size_t num_layers = 1;
    bool bidirectional = false;
    bool return_sequences = false;
    bool deep_preflight = false;
    size_t preflight_timeout_ms = 2000;
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

namespace RecurrentCudaPlacementReason {
inline constexpr const char* ArrayFireCudaAllowedByEstimator =
    "arrayfire_cuda_allowed_by_estimator";
inline constexpr const char* CudaJitParamOverflowRisk =
    "cuda_jit_param_overflow_risk";
inline constexpr const char* GruArrayFireCudaProbeRequired =
    "gru_arrayfire_cuda_probe_required";
inline constexpr const char* GruBidirectionalArrayFireCudaDisabled =
    "gru_bidirectional_arrayfire_cuda_disabled";
inline constexpr const char* LstmBidirectionalCudaJitParamOverflowRisk =
    "lstm_bidirectional_cuda_jit_param_overflow_risk";
} // namespace RecurrentCudaPlacementReason

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
            ? RecurrentCudaPlacementReason::GruBidirectionalArrayFireCudaDisabled
            : RecurrentCudaPlacementReason::GruArrayFireCudaProbeRequired;
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
        decision.reason_code =
            RecurrentCudaPlacementReason::LstmBidirectionalCudaJitParamOverflowRisk;
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
        decision.reason_code =
            RecurrentCudaPlacementReason::CudaJitParamOverflowRisk;
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
        decision.reason_code =
            RecurrentCudaPlacementReason::ArrayFireCudaAllowedByEstimator;
    }

    return decision;
}

} // namespace cyxwiz
