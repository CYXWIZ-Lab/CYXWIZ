// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
#pragma once

// How a layer reaches a native neural provider (TOFIX232).
//  - One request builder per op family. The layer builds its run-time request
//    with it, and the Engine's placement builds the same request from the
//    compiled shapes, so the placement and the run ask the same question.
//  - One dispatch: find the provider, run it, report the outcome. A run can
//    observe every outcome (ran, declined, failed, not served) and can make a
//    provider failure stop the run instead of falling back to ArrayFire.

#include "api_export.h"
#include "neural_provider.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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
// Whether a provider serves the backward of this recurrent forward request on
// its device. A training layer runs its forward on a provider only then: a
// provider forward whose backward is declined would recompute the batch on
// ArrayFire, a route the placement (which asks for both) did not declare.
CYXWIZ_API bool RecurrentProviderServesBackward(const NeuralOpRequest& forward);

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
    // No provider serves the request's device family, or the request or the
    // thread is off the run's bound device (target_mismatch): ArrayFire runs
    // the op. Reported once per (layer, op, device family, cause) in an
    // observer's scope, not per call, so a target mismatch is reported even
    // after the op was reported not served. The reason (provider_not_built,
    // provider_not_installed, provider_abi_mismatch, provider_disabled,
    // provider_not_offered, the family's *_provider_unavailable, or
    // target_mismatch) and its words are in `detail`.
    NotServed,
};
CYXWIZ_API const char* NeuralDispatchOutcomeName(NeuralDispatchOutcome outcome);

struct NeuralDispatchEvent {
    std::string provider_id;       // empty for NotServed: there is no provider
    std::string provider_version;
    std::string op;      // NeuralOpName
    std::string layer;   // the layer family asking ("LSTM", "Conv2D", ...)
    std::string device;  // NeuralDevicePlatformName of the request's target
    NeuralDispatchOutcome outcome = NeuralDispatchOutcome::Ran;
    std::string reason;  // BackendFallbackReasonName, for every outcome but Ran
    std::string detail;
};

// Called on the thread that dispatches (as the host-sync observer is).
using NeuralDispatchObserver = void (*)(const NeuralDispatchEvent& event);

// An observer's scope on this thread: a run, or a test. It is also the scope
// of "once" for NotServed: each scope hears of an unserved (layer, op, device
// family) one time per cause (no provider, target mismatch), however often
// the op is asked for. A thread with no observer scope logs a target
// mismatch once per (op, cause) for the process.
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

// What ran where, counted: every dispatch outcome on this thread while the
// tally lives, per (layer, op). It is an observer scope of its own and passes
// each event on to the observer that was set before it, so a run's trace
// still sees everything. A trainer uses it to say how its ops computed; a
// test uses it to require a path (CYXWIZ_EXPECT_PATH).
class CYXWIZ_API NeuralDispatchTally {
public:
    struct Entry {
        std::string layer;        // "PPO", "Adam", "fused MLP", "LSTM", ...
        std::string op;           // NeuralOpName
        std::string device;       // NeuralDevicePlatformName
        std::string provider_id;  // the provider that ran or answered last; empty when none serves the device
        uint64_t ran = 0;
        uint64_t declined = 0;
        uint64_t failed = 0;
        uint64_t not_served = 0;  // reports, not calls: NotServed is reported once per scope
        std::string reason;       // of the last outcome that was not Ran
        std::string detail;
    };

    NeuralDispatchTally();
    ~NeuralDispatchTally();
    NeuralDispatchTally(const NeuralDispatchTally&) = delete;
    NeuralDispatchTally& operator=(const NeuralDispatchTally&) = delete;

    const std::vector<Entry>& Entries() const;
    // The entry of (layer, op), or null when it was never dispatched.
    const Entry* Find(const char* layer, NeuralOp op) const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

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

// Whether `e` is a NeuralProviderRouteError. Decided inside the backend, so
// code in another module (a trainer's fallback handler) gets the right answer
// where type identity does not cross a shared-library boundary.
CYXWIZ_API bool IsNeuralProviderRouteError(const std::exception& e);

// Inside a run (ScopedNeuralRunTarget): whether the request's device and the
// thread's ArrayFire device are both the device the run bound. When not, no
// provider may be asked (it would read tensors that are not where it runs):
// false, logged as a warning and reported to the observer as NotServed with
// reason target_mismatch, once per (layer, op, device family) and scope (with
// no observer scope, once per op for the process). Outside a run: true.
// Cost: outside a run, one thread-local read. Inside a run, the thread's
// ArrayFire device (two ArrayFire queries) is read at the thread's first call
// in the run scope and remembered until the scope ends; later calls compare
// the request with the bound device only. A thread is not read again in the
// middle of a scope, so code that moves a thread's ArrayFire device inside a
// run opens a ScopedNeuralRunTarget of its own (PolicyDeviceScope does). A
// mismatch is read again on every call.
CYXWIZ_API bool CheckNeuralRunTarget(const NeuralOpRequest& request, const char* layer);

// Runs `execute` on the first provider that serves request.target and
// supports the request. Returns true when it ran.
//  - Inside a run, the request or the thread is not on the run's bound
//    device: false; NotServed (target_mismatch) is reported once, as
//    CheckNeuralRunTarget says.
//  - No provider serves the device: false; NotServed is reported, once per
//    (layer, op, device family) in the observer's scope, with the reason
//    WhyNoNeuralProviderServes gives. Without an observer nothing is built.
//  - Every serving provider declines: false; Declined is reported.
//  - execute returns !ok: Failed is reported and logged; under the Fail
//    policy NeuralProviderRouteError is thrown, otherwise false.
// The caller runs its ArrayFire path whenever this returns false.
using NeuralProviderExecute = std::function<NeuralOpStatus(INeuralNetworkProvider& provider)>;
CYXWIZ_API bool DispatchNeuralProviderOp(const NeuralOpRequest& request, const char* layer,
                                         const NeuralProviderExecute& execute);

// For a caller that asks the registry once and then never dispatches when no
// provider takes the request (FusedMlp::For): reports to the thread's
// observer why not, as the dispatch would (NotServed once per scope, or
// Declined with the serving provider's reason). Nothing when a provider
// supports the request, and nothing without an observer.
CYXWIZ_API void ReportNeuralProviderProbe(const NeuralOpRequest& request, const char* layer);

} // namespace cyxwiz
