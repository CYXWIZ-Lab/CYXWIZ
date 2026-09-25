#pragma once

#include "api_export.h"
#include "backend_fallback_reason.h"
#include "device_type.h"
#include "tensor.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Native neural-network provider contract (tofix68 phase 2), implementing
// the boundary in backend_provider_boundary_design.md:
//  - operation-level boundary; CyxWiz Tensors are the interchange type,
//    explicit host copies in v1;
//  - per-run, per-family, compile-time binding — providers never make
//    placement policy, they answer capability queries;
//  - CyxWiz keeps autograd, optimizers, and checkpoints;
//  - one registry, one taxonomy (backend_fallback_reason.h), evidence via
//    the existing placement-observation spine.
// Variant enums (activation, gated FFN, position strategy) mirror the
// tofix112 property authority; extend them there first.

namespace cyxwiz {

enum class NeuralOp {
    RnnForward,
    RnnBackward,
    LstmForward,
    LstmBackward,
    GruForward,
    GruBackward,
    AttentionForward,
    AttentionBackward,
    // Device-resident interface check (tofix112 phase 5b): outputs[0] =
    // 2 * inputs[0], elementwise. Exercises caller-owned device memory on the
    // caller's queue without any attention math.
    DeviceProbe,
};

CYXWIZ_API const char* NeuralOpName(NeuralOp op);

enum class NeuralActivation {
    None,
    Relu,
    Gelu,
    Swish,
    Elu,
    Selu,
    SquaredRelu,
    Tanh,
    Sigmoid,
};

enum class NeuralFfnType {
    DenseMlp,
    GatedMlp,
};

enum class NeuralGateActivation {
    None,
    Glu,
    ReGlu,
    GeGlu,
    SwiGlu,
    LiGlu,
};

enum class NeuralPositionStrategy {
    None,
    ExternalAdditive,
    LearnedAbsolute,
    SinusoidalAbsolute,
    Rope,
    PartialRope,
    RelativeBias,
    Alibi,
};

// Device family a provider serves and a request targets (device-keyed
// dispatch rule, tofix68 2026-09-23). Dispatch keys on the run's SELECTED
// device, never on hardware presence: a CUDA provider is never chosen for
// a run the user pointed at OpenCL, oneAPI, or CPU, even when a CUDA
// device is installed. The default (CPU) target selects no provider —
// callers fill it from the run, and an unfilled request fails closed.
struct NeuralDeviceTarget {
    DeviceType platform = DeviceType::CPU;
    int device_id = 0;
};

// The run's current target: the active ArrayFire backend and device (the
// same source ExecutionDeviceContext reads); CPU when ArrayFire is absent
// or the query fails.
CYXWIZ_API NeuralDeviceTarget CaptureCurrentNeuralDeviceTarget();
// Stable lowercase names ("cpu", "cuda", "opencl", "oneapi", ...) matching
// CurrentArrayFireBackendName's vocabulary.
CYXWIZ_API const char* NeuralDevicePlatformName(DeviceType platform);

// Compact typed descriptor. Never carries ArrayFire expression trees, raw
// user-controlled device pointers, or Python objects. Capability answers
// are per FULL tuple — a provider that supports LstmForward with Relu but
// not Selu says so here, and the compiler routes truthfully.
struct NeuralOpRequest {
    NeuralOp op = NeuralOp::LstmForward;
    // The run's selected device; the registry consults only providers
    // whose Platform() serves it (see NeuralDeviceTarget).
    NeuralDeviceTarget target;
    bool training = false;
    DataType dtype = DataType::Float32;
    size_t batch = 0;
    size_t seq = 0;
    size_t input = 0;
    size_t hidden = 0;
    size_t layers = 1;
    size_t directions = 1;
    size_t heads = 0;  // attention ops only
    NeuralActivation activation = NeuralActivation::None;
    NeuralFfnType ffn_type = NeuralFfnType::DenseMlp;
    NeuralGateActivation gate_activation = NeuralGateActivation::None;
    NeuralPositionStrategy position_strategy = NeuralPositionStrategy::None;
    bool deterministic = false;
    // Incremented by CyxWiz on any weight mutation; providers may cache
    // device-resident parameter mirrors keyed by this token.
    uint64_t parameter_version = 0;
    // v2 device-resident execution: buffers are the caller's device memory
    // (ArrayFire arrays) and work runs on the caller's queue. Providers
    // answer capability for this mode separately from host (v1) execution.
    bool device_resident = false;
    size_t elements = 0;  // DeviceProbe only
    // Attention (device-resident) contract, tofix112 phase 5b:
    //   inputs  Q [head_dim, seq, batch, heads]            (head_dim fastest)
    //           K, V [head_dim, kv_seq, batch, kv_heads]
    //           slopes [heads] when position_strategy == Alibi
    //   outputs O [head_dim, seq, batch, heads], LSE [seq, batch, heads]
    //           (natural-log sum of exp of the final scores per query row)
    // Scores: s = scale * q.k; soft-capped (cap * tanh(s / cap)) when
    // logit_softcap > 0; + slope * (key_pos - query_pos) for ALiBi; keys past
    // the query (causal) or `sliding_window`+ positions back are excluded.
    // Query i sits at position query_offset + i (KV-cached decoding).
    // AttentionBackward: inputs Q, K, V, O, dO, LSE[, slopes]; outputs dQ
    // [head_dim, seq, batch, heads], dK and dV [head_dim, kv_seq, batch,
    // kv_heads] (summed over the query heads sharing a kv head), and a
    // workspace delta [seq, batch, heads] = rowsum(dO * O).
    size_t kv_heads = 0;
    size_t head_dim = 0;
    size_t kv_seq = 0;
    size_t query_offset = 0;
    bool causal = false;
    size_t sliding_window = 0;
    float logit_softcap = 0.0f;
    float softmax_scale = 0.0f;
    // Attention dropout (applied when training): probabilities are kept with
    // 1 - p and scaled by 1/(1-p); keep/drop is a hash of (dropout_seed,
    // ((b * heads + h) * seq + i) * kv_seq + j), so forward and backward with
    // the same seed use the same mask. Softmax statistics (LSE) are pre-dropout.
    float attention_dropout = 0.0f;
    uint64_t dropout_seed = 0;
};

// Host-side CyxWiz-owned buffers, explicit copies at the boundary (v1).
// Per-op slot meaning is documented with each op's execution contract when
// it lights up (phase 3+); the skeleton executes nothing.
struct NeuralOpBuffers {
    std::vector<const Tensor*> inputs;
    std::vector<const Tensor*> weights;
    std::vector<Tensor*> outputs;
    std::vector<Tensor*> gradients;
};

// v2 device-resident execution (tofix112 phase 5b). The caller's device
// queue: ArrayFire's own CUDA stream, or its OpenCL context/queue/device, so
// provider kernels are ordered with ArrayFire work without host syncs.
struct NeuralDeviceQueue {
    DeviceType platform = DeviceType::CPU;
    int native_device = -1;        // CUDA ordinal (OpenCL: unused)
    void* cuda_stream = nullptr;   // cudaStream_t
    void* cl_context = nullptr;    // cl_context (not retained)
    void* cl_queue = nullptr;      // cl_command_queue (not retained)
    void* cl_device = nullptr;     // cl_device_id
};

// One caller-owned device buffer, locked for the duration of the call.
// CUDA: a float* device pointer; OpenCL: a cl_mem. Layout per op contract.
struct NeuralDeviceBuffer {
    void* handle = nullptr;
    size_t elements = 0;
};

struct NeuralDeviceOpBuffers {
    NeuralDeviceQueue queue;
    std::vector<NeuralDeviceBuffer> inputs;
    std::vector<NeuralDeviceBuffer> outputs;
};

struct NeuralCapability {
    bool supported = false;
    BackendFallbackReason reason =
        BackendFallbackReason::NvidiaProviderUnsupportedContract;
    std::string detail;
};

struct NeuralResourceEstimate {
    size_t workspace_bytes = 0;
    size_t reserve_bytes = 0;
};

struct NeuralOpStatus {
    bool ok = false;
    BackendFallbackReason reason =
        BackendFallbackReason::NvidiaProviderExecutionFailed;
    std::string detail;
};

class CYXWIZ_API INeuralNetworkProvider {
public:
    virtual ~INeuralNetworkProvider() = default;

    virtual const char* ProviderId() const = 0;
    // Device family this provider serves. The registry filters on it
    // BEFORE asking capability; a provider is never consulted for a
    // request targeting another family.
    virtual DeviceType Platform() const = 0;
    // Provider semver plus toolkit identity, e.g.
    // "cyxwiz.nvidia-cublas-cell 0.1.0 / CUDA 12.6 / cuBLAS 12".
    virtual std::string Version() const = 0;
    virtual NeuralCapability QueryCapability(
        const NeuralOpRequest& request) const = 0;
    virtual NeuralResourceEstimate EstimateResources(
        const NeuralOpRequest& request) const = 0;
    virtual NeuralOpStatus Execute(const NeuralOpRequest& request,
                                   NeuralOpBuffers& buffers) = 0;
    // v2: execute on caller-owned device buffers on buffers.queue. Enqueue
    // only - no host synchronization; the caller's queue orders later work.
    // Providers that do not implement it answer unsupported for requests
    // with device_resident=true and keep this default.
    virtual NeuralOpStatus ExecuteDevice(const NeuralOpRequest& request,
                                         NeuralDeviceOpBuffers& buffers) {
        (void)request;
        (void)buffers;
        NeuralOpStatus status;
        status.reason = BackendFallbackReason::NvidiaProviderUnsupportedContract;
        status.detail = "device-resident execution is not implemented by this provider";
        return status;
    }
};

// Runs a device-resident op on CyxWiz Tensors that live in ArrayFire memory:
// locks each tensor's device buffer, captures ArrayFire's native queue for
// the active backend, calls provider.ExecuteDevice, then unlocks. Output
// tensors are allocated here with the given shapes (Float32) and returned
// through `outputs`. No host copies or synchronization.
CYXWIZ_API NeuralOpStatus ExecuteNeuralOpOnDevice(
    INeuralNetworkProvider& provider, const NeuralOpRequest& request,
    const std::vector<const Tensor*>& inputs,
    const std::vector<std::vector<size_t>>& output_shapes,
    std::vector<Tensor>& outputs);

// Test-only: makes the registry answer as if no providers were registered
// (List() empty, FindSupporting() null), so AF-vs-native oracle tests and
// legacy placement-policy tests keep measuring what they measure. Never
// set in production code.
CYXWIZ_API void SetNeuralProvidersDisabledForTesting(bool disabled);

#ifdef CYXWIZ_HAS_NVIDIA_DNN_PROVIDER
// Test-only: synchronizes the device and reports free/total device memory
// so repeated-run leak checks (tofix68 validation gate) can run from a
// test binary that does not link the CUDA runtime. Never used by
// placement or execution logic.
CYXWIZ_API bool NvidiaProviderDeviceMemoryForTesting(size_t& free_bytes,
                                                     size_t& total_bytes);
#endif

#ifdef CYXWIZ_HAS_OPENCL_DNN_PROVIDER
// OpenCL tenant retention floor (owner ruling 2026-09-23): requests whose
// hidden size is below this value are declined with
// OpenclProviderBelowRetentionFloor so they stay on the portable path. The
// tenant measured 0.82x of native CPU at hidden=8 and 2.4x at hidden=16;
// the CUDA tenant has no floor because it is never slower (GRU ruling).
CYXWIZ_API size_t OpenclProviderRetentionFloorHidden();
// Test-only: override the floor (0 disables it) so small-tuple parity tests
// keep exercising the provider math. Never set in production code.
CYXWIZ_API void SetOpenclProviderRetentionFloorForTesting(size_t hidden_floor);
#endif

// The single provider registry (no parallel provider lists anywhere else).
// Built-in providers self-register on first access when their build flag
// is enabled AND their runtime probe succeeds; an empty registry is the
// normal state of a portable build.
class CYXWIZ_API NeuralProviderRegistry {
public:
    static NeuralProviderRegistry& Instance();

    void Register(std::shared_ptr<INeuralNetworkProvider> provider);
    std::vector<std::shared_ptr<INeuralNetworkProvider>> List() const;
    // Registered providers whose Platform() serves the target's device
    // family (device-keyed dispatch: stage 1 of lookup).
    std::vector<std::shared_ptr<INeuralNetworkProvider>> ListServing(
        const NeuralDeviceTarget& target) const;
    // First provider SERVING request.target that reports Supported for the
    // exact tuple (stage 2), or nullptr — the caller (compiler placement,
    // layer dispatch) owns the fallback decision. Providers on other
    // device families are never consulted.
    std::shared_ptr<INeuralNetworkProvider> FindSupporting(
        const NeuralOpRequest& request) const;

    NeuralProviderRegistry(const NeuralProviderRegistry&) = delete;
    NeuralProviderRegistry& operator=(const NeuralProviderRegistry&) = delete;

private:
    NeuralProviderRegistry() = default;
    struct State;
    State& GetState() const;
};

} // namespace cyxwiz
