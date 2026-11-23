// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
#pragma once

// How a layer reaches a native neural provider (TOFIX232).
//  - One request builder per op family. The layer builds its run-time request
//    with it, and the Engine's placement builds the same request from the
//    compiled shapes, so the placement and the run ask the same question.
//  - One dispatch: find the provider, run it, report the outcome. A run can
//    observe every outcome (ran, declined, failed) and can make a provider
//    failure stop the run instead of falling back to ArrayFire.

#include "api_export.h"
#include "neural_provider.h"

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>

namespace cyxwiz {

// ---- Request builders -------------------------------------------------------

// LSTM / GRU / RNN. Weights as the layer keeps them; batch-first input
// [batch, seq, input]. A backward op is a training request.
struct RecurrentProviderShape {
    size_t batch = 0;
    size_t seq = 0;
    size_t input = 0;
    size_t hidden = 0;
    size_t layers = 1;
    NeuralActivation activation = NeuralActivation::None;  // RNN: Tanh or Relu
};
CYXWIZ_API NeuralOpRequest RecurrentProviderRequest(NeuralOp op, const NeuralDeviceTarget& target,
                                                    const RecurrentProviderShape& shape);
// Whether a recurrent layer asks a provider at all: batch-first, one
// direction per call (a bidirectional layer asks for each direction's
// single-layer child), and no dropout between stacked layers while training.
// The layer also asks only when it starts from a zero state.
CYXWIZ_API bool RecurrentLayerAsksProvider(bool batch_first, bool training, float dropout,
                                           size_t layers);

// Conv2D on [H, W, Cin, N] (NeuralOpRequest's conv contract).
struct Conv2dProviderShape {
    size_t batch = 0;
    size_t height = 0;
    size_t width = 0;
    size_t in_channels = 0;
    size_t out_channels = 0;
    size_t kernel = 0;
    size_t stride = 1;
    size_t padding = 0;
    bool bias = false;
};
CYXWIZ_API NeuralOpRequest Conv2dProviderRequest(NeuralOp op, const NeuralDeviceTarget& target,
                                                 const Conv2dProviderShape& shape);

// Fused attention (NeuralOpRequest's attention contract). Only a layer whose
// mask is declared (causal, sliding window or none) asks.
struct AttentionProviderShape {
    size_t batch = 0;
    size_t seq = 0;
    size_t kv_seq = 0;
    size_t heads = 0;
    size_t kv_heads = 0;
    size_t head_dim = 0;
    size_t query_offset = 0;
    bool causal = false;
    size_t sliding_window = 0;
    float logit_softcap = 0.0f;
    float softmax_scale = 0.0f;
    bool alibi = false;
    bool training = false;
    float attention_dropout = 0.0f;  // applied only when training
    uint64_t dropout_seed = 0;
};
CYXWIZ_API NeuralOpRequest AttentionProviderRequest(NeuralOp op, const NeuralDeviceTarget& target,
                                                    const AttentionProviderShape& shape);

// ---- Dispatch, its observer and its failure policy --------------------------

enum class NeuralDispatchOutcome {
    Ran,       // the provider ran the op
    Declined,  // a provider serves the device but not this request: ArrayFire runs it on the same device
    Failed,    // the provider was chosen and failed
};
CYXWIZ_API const char* NeuralDispatchOutcomeName(NeuralDispatchOutcome outcome);

struct NeuralDispatchEvent {
    std::string provider_id;
    std::string provider_version;
    std::string op;      // NeuralOpName
    std::string layer;   // the layer family asking ("LSTM", "Conv2D", ...)
    std::string device;  // NeuralDevicePlatformName of the request's target
    NeuralDispatchOutcome outcome = NeuralDispatchOutcome::Ran;
    std::string reason;  // BackendFallbackReasonName, for Declined and Failed
    std::string detail;
};

// Called on the thread that dispatches (as the host-sync observer is).
using NeuralDispatchObserver = void (*)(const NeuralDispatchEvent& event);

class CYXWIZ_API ScopedNeuralDispatchObserver {
public:
    explicit ScopedNeuralDispatchObserver(NeuralDispatchObserver observer);
    ~ScopedNeuralDispatchObserver();
    ScopedNeuralDispatchObserver(const ScopedNeuralDispatchObserver&) = delete;
    ScopedNeuralDispatchObserver& operator=(const ScopedNeuralDispatchObserver&) = delete;

private:
    NeuralDispatchObserver previous_;
};

// What a provider failure does on this thread. FallBack (the default): the
// layer runs its ArrayFire path. Fail (a Strict run): the dispatch throws
// NeuralProviderRouteError, so the run stops naming the provider and the op.
enum class NeuralProviderFailurePolicy {
    FallBack,
    Fail,
};

class CYXWIZ_API ScopedNeuralProviderFailurePolicy {
public:
    explicit ScopedNeuralProviderFailurePolicy(NeuralProviderFailurePolicy policy);
    ~ScopedNeuralProviderFailurePolicy();
    ScopedNeuralProviderFailurePolicy(const ScopedNeuralProviderFailurePolicy&) = delete;
    ScopedNeuralProviderFailurePolicy& operator=(const ScopedNeuralProviderFailurePolicy&) = delete;

private:
    NeuralProviderFailurePolicy previous_;
};

CYXWIZ_API NeuralProviderFailurePolicy CurrentNeuralProviderFailurePolicy();

// Thrown by DispatchNeuralProviderOp under NeuralProviderFailurePolicy::Fail.
class NeuralProviderRouteError : public std::runtime_error {
public:
    explicit NeuralProviderRouteError(NeuralDispatchEvent event)
        : std::runtime_error("provider_route_changed: " + event.provider_id + " " + event.op +
                             " failed in " + event.layer + " (" + event.reason + "): " + event.detail),
          event_(std::move(event)) {}
    const NeuralDispatchEvent& Event() const { return event_; }

private:
    NeuralDispatchEvent event_;
};

// Runs `execute` on the first provider that serves request.target and
// supports the request. Returns true when it ran.
//  - No provider serves the device: false; nothing is reported.
//  - Every serving provider declines: false; Declined is reported.
//  - execute returns !ok: Failed is reported and logged; under the Fail
//    policy NeuralProviderRouteError is thrown, otherwise false.
// The caller runs its ArrayFire path whenever this returns false.
using NeuralProviderExecute = std::function<NeuralOpStatus(INeuralNetworkProvider& provider)>;
CYXWIZ_API bool DispatchNeuralProviderOp(const NeuralOpRequest& request, const char* layer,
                                         const NeuralProviderExecute& execute);

} // namespace cyxwiz
