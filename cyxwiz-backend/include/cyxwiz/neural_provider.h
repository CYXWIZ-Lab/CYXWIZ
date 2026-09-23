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
};

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
