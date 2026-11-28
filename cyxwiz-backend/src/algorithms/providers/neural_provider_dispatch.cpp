#include "cyxwiz/neural_provider_dispatch.h"

#include "neural_run_target_internal.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace cyxwiz {

namespace {

thread_local NeuralDispatchObserver g_dispatch_observer = nullptr;
thread_local NeuralProviderFailurePolicy g_failure_policy = NeuralProviderFailurePolicy::FallBack;

// Why an op was not served, as part of the "once" key, so one cause never
// hides another: no provider serves the device family (its reason, from
// WhyNoNeuralProviderServes, is a fact of the process and does not change
// within a run), or the request or the thread is off the run's bound device
// (target_mismatch).
enum class NotServedCause { NoProvider, TargetMismatch };

// What each observer scope on this thread has already heard is not served:
// one list per ScopedNeuralDispatchObserver, innermost last. A scope is a run
// (or a tally inside it), so "once per (layer, op, device family, cause) and
// run". A list holds a handful of entries: a run asks for a few ops.
struct NotServedKey {
    NeuralOp op;
    DeviceType platform;
    std::string layer;
    NotServedCause cause;
};
thread_local std::vector<std::vector<NotServedKey>> g_not_served_reported;

// Outside every observer scope (a thread no run observes, which can still
// hold a run target) there is no scope to remember in: a cause is logged once
// per (op, cause) for the process.
std::mutex g_unscoped_mutex;
std::vector<std::pair<NeuralOp, NotServedCause>> g_unscoped_reported;

bool IsBackwardRecurrentOp(NeuralOp op) {
    return op == NeuralOp::LstmBackward || op == NeuralOp::GruBackward || op == NeuralOp::RnnBackward;
}

void Notify(const NeuralDispatchEvent& event) {
    if (g_dispatch_observer) g_dispatch_observer(event);
}

// A provider's Version(), built once per provider object on this thread: an
// observed run builds an event for every call (an RL run makes thousands a
// second), and a Version() may format (the NVIDIA provider's uses a stream).
// Keyed by ownership, so a provider registered later at the same address is
// never given another's version.
const std::string& ProviderVersion(const std::shared_ptr<INeuralNetworkProvider>& provider) {
    thread_local std::vector<std::pair<std::weak_ptr<INeuralNetworkProvider>, std::string>> versions;
    for (const auto& [known, version] : versions)
        if (!known.owner_before(provider) && !provider.owner_before(known)) return version;
    versions.erase(std::remove_if(versions.begin(), versions.end(),
                                  [](const auto& entry) { return entry.first.expired(); }),
                   versions.end());
    versions.emplace_back(provider, provider->Version());
    return versions.back().second;
}

NeuralDispatchEvent MakeEvent(const NeuralOpRequest& request, const char* layer,
                              const std::shared_ptr<INeuralNetworkProvider>& provider, NeuralDispatchOutcome outcome) {
    NeuralDispatchEvent event;
    if (provider) {
        event.provider_id = provider->ProviderId();
        event.provider_version = ProviderVersion(provider);
    }
    event.op = NeuralOpName(request.op);
    event.layer = layer ? layer : "";
    event.device = NeuralDevicePlatformName(request.target.platform);
    event.outcome = outcome;
    return event;
}

// Whether this scope hears of (layer, op, device family) not being served
// for `cause` for the first time; it is remembered from then on. Outside
// every scope: whether the process hears of (op, cause) for the first time.
bool FirstInScope(const NeuralOpRequest& request, const char* layer, NotServedCause cause) {
    if (g_not_served_reported.empty()) {
        std::lock_guard<std::mutex> lock(g_unscoped_mutex);
        for (const auto& [op, known] : g_unscoped_reported)
            if (op == request.op && known == cause) return false;
        g_unscoped_reported.emplace_back(request.op, cause);
        return true;
    }
    auto& reported = g_not_served_reported.back();
    const char* name = layer ? layer : "";
    for (const auto& key : reported)
        if (key.op == request.op && key.platform == request.target.platform && key.layer == name &&
            key.cause == cause)
            return false;
    reported.push_back({request.op, request.target.platform, name, cause});
    return true;
}

// No provider serves the request's device family: tells the observer why,
// the first time its scope asks for this (layer, op, device family). Called
// only with an observer set, so an unobserved call builds nothing.
void ReportNotServedOnce(const NeuralOpRequest& request, const char* layer) {
    if (!FirstInScope(request, layer, NotServedCause::NoProvider)) return;
    const NeuralProviderAbsence absence = WhyNoNeuralProviderServes(request.target.platform, request.op);
    NeuralDispatchEvent event = MakeEvent(request, layer, nullptr, NeuralDispatchOutcome::NotServed);
    event.reason = BackendFallbackReasonName(absence.reason);
    event.detail = absence.Sentence();
    // The path taken, once per op and run: Info when a provider is missing
    // and can be supplied; Debug for the designed ArrayFire path (the CPU)
    if (absence.fixable)
        spdlog::info("{} {} runs on ArrayFire on {}: {} ({})", event.layer, event.op, event.device, event.detail,
                     event.reason);
    else
        spdlog::debug("{} {} runs on ArrayFire on {}: {} ({})", event.layer, event.op, event.device, event.detail,
                      event.reason);
    Notify(event);
}

std::string TargetWords(const NeuralDeviceTarget& target) {
    return std::string(NeuralDevicePlatformName(target.platform)) + " device " + std::to_string(target.device_id);
}

void ReportDeclined(const NeuralOpRequest& request, const char* layer,
                    const std::shared_ptr<INeuralNetworkProvider>& serving) {
    const NeuralCapability capability = serving->QueryCapability(request);
    NeuralDispatchEvent event = MakeEvent(request, layer, serving, NeuralDispatchOutcome::Declined);
    event.reason = BackendFallbackReasonName(capability.reason);
    event.detail = capability.detail;
    Notify(event);
}

// The innermost tally of this thread, and the function it observes with
thread_local NeuralDispatchTally::Impl* g_tally = nullptr;
void TallyObserver(const NeuralDispatchEvent& event);

}  // namespace

NeuralOpRequest RecurrentProviderRequest(NeuralOp op, const NeuralDeviceTarget& target,
                                         const RecurrentProviderShape& shape) {
    NeuralOpRequest request;
    request.op = op;
    request.target = target;
    // The forward math is the same in training; the backward is the training op.
    request.training = IsBackwardRecurrentOp(op);
    request.dtype = DataType::Float32;
    request.batch = shape.batch;
    request.seq = shape.seq;
    request.input = shape.input;
    request.hidden = shape.hidden;
    request.layers = shape.layers;
    request.directions = 1;
    request.activation = shape.activation;
    return request;
}

bool RecurrentLayerAsksProvider(bool batch_first, bool training, float dropout, size_t layers) {
    return batch_first && !(training && dropout > 0.0f && layers > 1);
}

bool RecurrentProviderServesBackward(const NeuralOpRequest& forward) {
    NeuralOp backward;
    switch (forward.op) {
        case NeuralOp::LstmForward: backward = NeuralOp::LstmBackward; break;
        case NeuralOp::GruForward: backward = NeuralOp::GruBackward; break;
        case NeuralOp::RnnForward: backward = NeuralOp::RnnBackward; break;
        default: return false;
    }
    const RecurrentProviderShape shape{forward.batch, forward.seq,    forward.input,
                                       forward.hidden, forward.layers, forward.activation};
    return NeuralProviderRegistry::Instance().FindSupporting(RecurrentProviderRequest(backward, forward.target, shape)) !=
           nullptr;
}

NeuralOpRequest Conv2dProviderRequest(NeuralOp op, const NeuralDeviceTarget& target,
                                      const Conv2dProviderShape& shape) {
    NeuralOpRequest request;
    request.op = op;
    request.target = target;
    request.device_resident = true;
    request.dtype = DataType::Float32;
    request.batch = shape.batch;
    request.input = shape.in_channels;
    request.hidden = shape.out_channels;
    request.conv_height = shape.height;
    request.conv_width = shape.width;
    request.conv_kernel = shape.kernel;
    request.conv_stride = shape.stride;
    request.conv_padding = shape.padding;
    request.conv_bias = shape.bias;
    return request;
}

NeuralOpRequest AttentionProviderRequest(NeuralOp op, const NeuralDeviceTarget& target,
                                         const AttentionProviderShape& shape) {
    NeuralOpRequest request;
    request.op = op;
    request.target = target;
    request.device_resident = true;
    request.dtype = DataType::Float32;
    request.training = op == NeuralOp::AttentionBackward || shape.training;
    request.batch = shape.batch;
    request.heads = shape.heads;
    request.kv_heads = shape.kv_heads;
    request.seq = shape.seq;
    request.kv_seq = shape.kv_seq;
    request.head_dim = shape.head_dim;
    request.query_offset = shape.query_offset;
    request.causal = shape.causal;
    request.sliding_window = shape.sliding_window;
    request.logit_softcap = shape.logit_softcap;
    request.softmax_scale = shape.softmax_scale;
    request.position_strategy = shape.alibi ? NeuralPositionStrategy::Alibi : NeuralPositionStrategy::None;
    if (request.training && shape.attention_dropout > 0.0f) {
        request.attention_dropout = shape.attention_dropout;
        request.dropout_seed = shape.dropout_seed;
    }
    return request;
}

const char* NeuralDispatchOutcomeName(NeuralDispatchOutcome outcome) {
    switch (outcome) {
        case NeuralDispatchOutcome::Ran: return "ran";
        case NeuralDispatchOutcome::Declined: return "declined";
        case NeuralDispatchOutcome::Failed: return "failed";
        case NeuralDispatchOutcome::NotServed: return "not_served";
    }
    return "unknown";
}

ScopedNeuralDispatchObserver::ScopedNeuralDispatchObserver(NeuralDispatchObserver observer)
    : previous_(g_dispatch_observer) {
    g_dispatch_observer = observer;
    g_not_served_reported.emplace_back();  // this scope has heard nothing yet
}

ScopedNeuralDispatchObserver::~ScopedNeuralDispatchObserver() {
    g_dispatch_observer = previous_;
    if (!g_not_served_reported.empty()) g_not_served_reported.pop_back();
}

struct NeuralDispatchTally::Impl {
    std::vector<Entry> entries;
    Impl* outer = nullptr;  // the tally this one is nested in
    NeuralDispatchObserver previous = nullptr;

    void Record(const NeuralDispatchEvent& event) {
        Entry* entry = nullptr;
        for (auto& known : entries)
            if (known.op == event.op && known.layer == event.layer && known.device == event.device) entry = &known;
        if (!entry) {
            entries.emplace_back();
            entry = &entries.back();
            entry->layer = event.layer;
            entry->op = event.op;
            entry->device = event.device;
        }
        if (!event.provider_id.empty() || event.outcome == NeuralDispatchOutcome::NotServed)
            entry->provider_id = event.provider_id;
        switch (event.outcome) {
            case NeuralDispatchOutcome::Ran: ++entry->ran; return;
            case NeuralDispatchOutcome::Declined: ++entry->declined; break;
            case NeuralDispatchOutcome::Failed: ++entry->failed; break;
            case NeuralDispatchOutcome::NotServed: ++entry->not_served; break;
        }
        entry->reason = event.reason;
        entry->detail = event.detail;
    }
};

namespace {

void TallyObserver(const NeuralDispatchEvent& event) {
    NeuralDispatchTally::Impl* tally = g_tally;
    if (!tally) return;
    tally->Record(event);
    if (tally->previous == &TallyObserver) {
        // The tally around this one: it records, then passes on in its turn
        g_tally = tally->outer;
        TallyObserver(event);
        g_tally = tally;
    } else if (tally->previous) {
        tally->previous(event);
    }
}

}  // namespace

NeuralDispatchTally::NeuralDispatchTally() : impl_(std::make_unique<Impl>()) {
    impl_->outer = g_tally;
    impl_->previous = g_dispatch_observer;
    g_tally = impl_.get();
    g_dispatch_observer = &TallyObserver;
    g_not_served_reported.emplace_back();
}

NeuralDispatchTally::~NeuralDispatchTally() {
    g_dispatch_observer = impl_->previous;
    g_tally = impl_->outer;
    if (!g_not_served_reported.empty()) g_not_served_reported.pop_back();
}

const std::vector<NeuralDispatchTally::Entry>& NeuralDispatchTally::Entries() const {
    return impl_->entries;
}

const NeuralDispatchTally::Entry* NeuralDispatchTally::Find(const char* layer, NeuralOp op) const {
    const char* name = NeuralOpName(op);
    for (const auto& entry : impl_->entries)
        if (entry.op == name && entry.layer == (layer ? layer : "")) return &entry;
    return nullptr;
}

ScopedNeuralProviderFailurePolicy::ScopedNeuralProviderFailurePolicy(NeuralProviderFailurePolicy policy)
    : previous_(g_failure_policy) {
    g_failure_policy = policy;
}

ScopedNeuralProviderFailurePolicy::~ScopedNeuralProviderFailurePolicy() {
    g_failure_policy = previous_;
}

NeuralProviderFailurePolicy CurrentNeuralProviderFailurePolicy() {
    return g_failure_policy;
}

bool IsNeuralProviderRouteError(const std::exception& e) {
    return dynamic_cast<const NeuralProviderRouteError*>(&e) != nullptr;
}

bool CheckNeuralRunTarget(const NeuralOpRequest& request, const char* layer) {
    NeuralDeviceTarget bound;
    if (!BoundNeuralRunTarget(bound)) return true;  // outside a run: the request's own target
    const bool request_on_it = SameNeuralDeviceTarget(request.target, bound);
    // The thread's ArrayFire device is read once per run scope on a thread,
    // at its first call: the Engine binds a run's device once, and code that
    // moves a thread to another device opens a scope of its own
    if (request_on_it && detail::NeuralRunThreadVerified()) return true;
    NeuralDeviceTarget thread;
    std::string error;
    const bool known = QueryArrayFireNeuralTarget(thread, error);
    const bool thread_on_it = known && SameNeuralDeviceTarget(thread, bound);
    if (thread_on_it) detail::MarkNeuralRunThreadVerified();
    if (thread_on_it && request_on_it) return true;
    if (!FirstInScope(request, layer, NotServedCause::TargetMismatch)) return false;
    // A provider would read tensors that are not where it runs: none is
    // asked, ArrayFire runs the op where its data is
    std::string detail = "the run is bound to " + TargetWords(bound);
    if (!known) detail += ", and ArrayFire's active device could not be read (" + error + ")";
    else if (!thread_on_it) detail += ", the thread's ArrayFire device is " + TargetWords(thread);
    if (!request_on_it) detail += ", the request names " + TargetWords(request.target);
    const NeuralProviderAbsence absence =
        DescribeNeuralProviderAbsence(BackendFallbackReason::TargetMismatch, request.target.platform);
    NeuralDispatchEvent event = MakeEvent(request, layer, nullptr, NeuralDispatchOutcome::NotServed);
    event.reason = BackendFallbackReasonName(BackendFallbackReason::TargetMismatch);
    event.detail = absence.name + ": " + detail + ". " + absence.fix;
    spdlog::warn("{} {}: no neural provider is asked ({}): {}", event.layer, event.op, event.reason, event.detail);
    Notify(event);
    return false;
}

bool DispatchNeuralProviderOp(const NeuralOpRequest& request, const char* layer,
                              const NeuralProviderExecute& execute) {
    if (!CheckNeuralRunTarget(request, layer)) return false;
    auto& registry = NeuralProviderRegistry::Instance();
    const auto provider = registry.FindSupporting(request);
    if (!provider) {
        const auto serving = registry.ListServing(request.target);
        if (!serving.empty()) {
            ReportDeclined(request, layer, serving.front());
        } else if (g_dispatch_observer) {
            // Unobserved, this costs what it did before: nothing is built
            ReportNotServedOnce(request, layer);
        }
        return false;
    }
    const NeuralOpStatus status = execute(*provider);
    if (status.ok) {
        // The common case, every batch: no event built unless someone observes.
        if (g_dispatch_observer) Notify(MakeEvent(request, layer, provider, NeuralDispatchOutcome::Ran));
        return true;
    }
    NeuralDispatchEvent event = MakeEvent(request, layer, provider, NeuralDispatchOutcome::Failed);
    event.reason = BackendFallbackReasonName(status.reason);
    event.detail = status.detail;
    Notify(event);
    if (g_failure_policy == NeuralProviderFailurePolicy::Fail) {
        throw NeuralProviderRouteError(std::move(event));
    }
    spdlog::warn("{}: provider {} {} failed ({}): {}; running the ArrayFire path", event.layer,
                 event.provider_id, event.op, event.reason, event.detail);
    return false;
}

void ReportNeuralProviderProbe(const NeuralOpRequest& request, const char* layer) {
    if (!CheckNeuralRunTarget(request, layer)) return;  // reported as target_mismatch
    if (!g_dispatch_observer) return;
    auto& registry = NeuralProviderRegistry::Instance();
    if (registry.FindSupporting(request)) return;
    const auto serving = registry.ListServing(request.target);
    if (serving.empty()) ReportNotServedOnce(request, layer);
    else ReportDeclined(request, layer, serving.front());
}

} // namespace cyxwiz
