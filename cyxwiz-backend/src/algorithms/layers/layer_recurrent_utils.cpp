#include "layer_recurrent_utils.h"

#include <atomic>

namespace cyxwiz {

namespace {
std::atomic<bool> g_force_native_recurrent_forward_for_testing{false};
} // namespace

void SetForceNativeRecurrentForwardForTesting(bool force) {
    g_force_native_recurrent_forward_for_testing.store(force);
}

namespace recurrent_utils_detail {
bool IsNativeRecurrentForwardForcedForTesting() {
    return g_force_native_recurrent_forward_for_testing.load();
}
} // namespace recurrent_utils_detail

} // namespace cyxwiz

#ifdef CYXWIZ_HAS_ARRAYFIRE

#include "cyxwiz/backend_placement_observation.h"
#include "cyxwiz/debug_hooks.h"

#include <algorithm>
#include <atomic>

#include <arrayfire.h>
#include <spdlog/spdlog.h>

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

namespace cyxwiz {

namespace {

std::atomic<bool> g_disable_lstm_arrayfire_cuda_after_failure{false};
std::atomic<bool> g_disable_gru_arrayfire_cuda_after_failure{false};

std::atomic<bool>& RecurrentFailureDisableFlag(RecurrentLayerKind kind) {
    return kind == RecurrentLayerKind::LSTM
        ? g_disable_lstm_arrayfire_cuda_after_failure
        : g_disable_gru_arrayfire_cuda_after_failure;
}

} // namespace

std::string BuildRecurrentFormalParameterOverflowFallbackMessage(
    const char* layer_name) {
    return std::string("ArrayFire ") + layer_name +
           " hit CUDA generated-kernel formal-parameter overflow "
           "(reason=" +
           BackendFallbackReasonName(BackendFallbackReason::CudaJitParamOverflow) +
           "); falling back to native CPU recurrent path. This is separate "
           "from VRAM capacity.";
}

void DisableArrayFireCudaRecurrentAfterFailure(
    RecurrentLayerKind kind,
    const char* layer_name,
    size_t batch_size,
    size_t seq_len,
    size_t input_size,
    int hidden_size,
    int num_layers,
    bool bidirectional,
    const char* error_message) {
    const bool is_param_overflow =
        IsCudaJitFormalParameterOverflow(error_message);
    const BackendFallbackReason reason = is_param_overflow
        ? BackendFallbackReason::CudaJitParamOverflow
        : ClassifyArrayFireBackendFallbackReason(error_message);

    RecurrentCudaPlacementRequest request;
    request.kind = kind;
    request.batch_size = batch_size;
    request.seq_len = seq_len;
    request.input_size = input_size;
    request.hidden_size = static_cast<size_t>(std::max(1, hidden_size));
    request.num_layers = static_cast<size_t>(std::max(1, num_layers));
    request.bidirectional = bidirectional;
    request.return_sequences = false;

    // Record evidence for EVERY failure class, not only param overflow —
    // the compiler routes around any known-unsafe recurrent key
    // (tofix67 slice 6). Only overflow flips the process-wide disable
    // latch; other reasons may be transient device conditions.
    RecordRecurrentCudaPlacementObservation(
        request,
        BackendFallbackReasonName(reason),
        BackendPlacementObservationSource::RuntimeFallback,
        is_param_overflow
            ? std::string(layer_name) +
                  " runtime ArrayFire CUDA forward failed with "
                  "generated-kernel formal-parameter overflow. This "
                  "observation should make future compiler preflight route "
                  "the same recurrent shape to CPU before training starts."
            : std::string(layer_name) +
                  " runtime ArrayFire CUDA forward failed (reason=" +
                  BackendFallbackReasonName(reason) +
                  "). This observation should make future compiler "
                  "preflight route the same recurrent shape to CPU before "
                  "training starts.");

    if (!is_param_overflow) {
        return;
    }

    auto& disabled = RecurrentFailureDisableFlag(kind);
    if (!disabled.exchange(true)) {
        const std::string disable_message =
            std::string(layer_name) +
            " ArrayFire CUDA recurrent path hit CUDA generated-kernel "
            "formal-parameter overflow (reason=" +
            BackendFallbackReasonName(BackendFallbackReason::CudaJitParamOverflow) +
            "). Disabling this recurrent CUDA path "
            "for the rest of the process and using the native CPU recurrent "
            "path directly for later batches. This is separate from VRAM "
            "capacity.";
        BackendDebugHooks::EmitDebugEvent(layer_name, disable_message);
        spdlog::warn("{}", disable_message);
    }
}

bool ShouldUseArrayFireRecurrentForward(
    RecurrentLayerKind kind,
    size_t batch_size,
    size_t seq_len,
    size_t input_size,
    int hidden_size,
    int num_layers,
    bool bidirectional) {
    if (recurrent_utils_detail::IsNativeRecurrentForwardForcedForTesting()) {
        return false;
    }
    try {
        if (af::getActiveBackend() != AF_BACKEND_CUDA) {
            return true;
        }
    } catch (const af::exception&) {
        return true;
    }

    auto& disabled_after_failure = RecurrentFailureDisableFlag(kind);
    if (disabled_after_failure.load()) {
        static std::atomic<bool> warned_disabled_lstm{false};
        static std::atomic<bool> warned_disabled_gru{false};
        std::atomic<bool>& warned =
            kind == RecurrentLayerKind::LSTM ? warned_disabled_lstm : warned_disabled_gru;
        if (!warned.exchange(true)) {
            spdlog::warn(
                "CUDA recurrent placement: ArrayFire {} forward is disabled "
                "after a previous CUDA generated-kernel formal-parameter "
                "overflow; runtime is using the native CPU recurrent path "
                "directly for this process.",
                RecurrentKindName(kind));
        }
        return false;
    }

    RecurrentCudaPlacementRequest request;
    request.kind = kind;
    request.batch_size = batch_size;
    request.seq_len = seq_len;
    request.input_size = input_size;
    request.hidden_size = static_cast<size_t>(std::max(1, hidden_size));
    request.num_layers = static_cast<size_t>(std::max(1, num_layers));
    request.bidirectional = bidirectional;
    request.return_sequences = false;

    const auto decision = EvaluateRecurrentCudaPlacement(request);
    if (decision.should_attempt_arrayfire_cuda) {
        return true;
    }

    static std::atomic<bool> warned_lstm{false};
    static std::atomic<bool> warned_gru{false};
    std::atomic<bool>& warned =
        kind == RecurrentLayerKind::LSTM ? warned_lstm : warned_gru;
    if (!warned.exchange(true)) {
        const std::string reason =
            "CUDA recurrent preflight: skipping ArrayFire " +
            decision.layer_name + " forward. " + decision.reason +
            " Runtime is using the native CPU recurrent path directly to avoid "
            "repeated failed GPU compiles. To keep this layer on GPU, reduce "
            "hidden_size/sequence length/bidirectionality, or replace this "
            "path with a fused native recurrent CUDA kernel.";
        BackendDebugHooks::EmitDebugEvent(decision.layer_name + "Layer", reason);
        spdlog::warn("{}", reason);
    }
    return false;
}

} // namespace cyxwiz

#endif
