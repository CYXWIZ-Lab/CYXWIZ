#include "rl_common.h"

#include <cyxwiz/fused_mlp.h>
#include <cyxwiz/neural_provider_dispatch.h>
#include <cyxwiz/sequential.h>

#include "../training_trace_collector.h"

namespace cyxwiz::rl {

PolicyRunScopes::PolicyRunScopes(DeviceType device_type, int device_id, ExecutionPolicy policy,
                                 NeuralDispatchObserver observer)
    : target_(NeuralDeviceTarget{device_type, device_id}),
      observer_(observer),
      failure_policy_(policy == ExecutionPolicy::Strict ? NeuralProviderFailurePolicy::Fail
                                                        : NeuralProviderFailurePolicy::FallBack) {}

std::unique_ptr<FusedMlp> FuseNetwork(IExecutableModel& model, FusedMlp::Refusal& refusal) {
    SequentialModel* sequential = model.AsSequentialModel();
    if (!sequential) {
        refusal = FusedMlp::Refusal{};
        refusal.text = "not a sequential network";
        return nullptr;
    }
    return FusedMlp::For(*sequential, refusal);
}

NetworkRoute FusedNetworkRoute(const FusedMlp& fused) {
    NetworkRoute route;
    route.provider_id = fused.ProviderId();
    route.text = "fused small-network kernels (" + route.provider_id + ")";
    return route;
}

NetworkRoute ArrayFireNetworkRoute(const FusedMlp::Refusal& refusal, const std::string& which) {
    NetworkRoute route;
    route.reason = refusal.reason;
    // A missing provider is the same for every network; another reason is one network's own
    route.detail = refusal.no_provider || which.empty() ? refusal.text : which + ": " + refusal.text;
    if (refusal.no_provider && refusal.absence.fixable) route.missing_provider = refusal.absence.Sentence();
    route.text = route.detail.empty() ? std::string("ArrayFire operations") : "ArrayFire operations (" + route.detail + ")";
    return route;
}

namespace {
constexpr const char* kLatchedWords = "turned off for this run after a provider failure";
}  // namespace

NetworkRoute FailedNetworkRoute(const std::exception& e) {
    NetworkRoute route;
    route.reason = BackendFallbackReasonName(BackendFallbackReason::ProviderLatched);
    route.detail = std::string("the fused kernels failed: ") + e.what() + "; " + kLatchedWords;
    route.text = "ArrayFire operations (" + route.detail + ")";
    return route;
}

void MarkProviderLatched(ComputePathPart& part) {
    const std::string failure =
        part.reason.empty() ? part.detail : part.reason + (part.detail.empty() ? std::string() : ": " + part.detail);
    part.provider_id.clear();  // ArrayFire from then on, whatever ran before
    part.mixed = false;
    part.reason = BackendFallbackReasonName(BackendFallbackReason::ProviderLatched);
    part.detail = (failure.empty() ? std::string() : failure + "; ") + kLatchedWords;
}

ComputePathPart NetworksPart(const NetworkRoute& route) {
    ComputePathPart part;
    part.what = "networks";
    part.provider_id = route.provider_id;
    part.reason = route.reason;
    part.detail = route.detail;
    part.arrayfire = "ArrayFire operations";
    return part;
}

ComputePathPart DispatchedPart(const char* what, const NeuralDispatchTally& tally, const char* layer, NeuralOp op,
                               DeviceType device, const std::string& unasked) {
    ComputePathPart part;
    part.what = what;
    const NeuralDispatchTally::Entry* entry = tally.Find(layer, op, device);
    if (!entry) {
        part.detail = unasked;
        return part;
    }
    const uint64_t elsewhere = entry->declined + entry->failed + entry->not_served;
    if (entry->ran > 0) {
        part.provider_id = entry->provider_id;
        part.mixed = elsewhere > 0;
    }
    if (elsewhere > 0) {
        part.reason = entry->reason;
        part.detail = entry->detail.empty() ? entry->reason : entry->detail;
    }
    return part;
}

std::optional<ComputePathPart> AdamPart(const NeuralDispatchTally& tally, DeviceType device, const AdamRoute& adam) {
    if (adam.together && !tally.Find("Adam", NeuralOp::AdamStep, device)) return std::nullopt;
    ComputePathPart part = DispatchedPart("Adam", tally, "Adam", NeuralOp::AdamStep, device, adam.why);
    part.arrayfire = "ArrayFire, per tensor";
    if (!adam.together) {  // per tensor from now on, whatever ran before
        const NeuralDispatchTally::Entry* entry = tally.Find("Adam", NeuralOp::AdamStep, device);
        if (entry && entry->failed > 0) {
            MarkProviderLatched(part);
        } else {
            part.provider_id.clear();
            part.mixed = false;
            if (part.detail.empty()) part.detail = adam.why;
        }
    }
    return part;
}

namespace {

// Where a part runs, in words; its reason left out when the networks gave the same one
std::string Where(const ComputePathPart& part, const NetworkRoute& route) {
    const bool said = part.detail.empty() || part.detail == route.detail || part.detail == route.missing_provider;
    const std::string why = said ? std::string() : " (" + part.detail + ")";
    if (part.provider_id.empty()) return part.arrayfire + why;
    return part.mixed ? part.provider_id + ", some calls on ArrayFire" + why : part.provider_id;
}

}  // namespace

std::string FormatComputePath(const NetworkRoute& route, const std::vector<ComputePathPart>& ops) {
    std::string text = route.text;
    for (size_t i = 0; i < ops.size();) {
        // Neighbours that run the same way share one entry
        const std::string where = Where(ops[i], route);
        std::string names = ops[i].what;
        size_t j = i + 1;
        for (; j < ops.size() && Where(ops[j], route) == where; ++j) names += ", " + ops[j].what;
        text += "; " + names + ": " + where;
        i = j;
    }
    return text;
}

std::vector<ComputePathPart> ComputePathParts(const NetworkRoute& route, const std::vector<ComputePathPart>& ops) {
    std::vector<ComputePathPart> parts{NetworksPart(route)};
    parts.insert(parts.end(), ops.begin(), ops.end());
    return parts;
}

void RethrowProviderRouteError(const std::exception& e) {
    if (IsNeuralProviderRouteError(e)) throw;  // the exception being handled
}

}  // namespace cyxwiz::rl
