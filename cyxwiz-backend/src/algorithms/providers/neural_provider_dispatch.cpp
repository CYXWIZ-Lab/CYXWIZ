#include "cyxwiz/neural_provider_dispatch.h"

#include <spdlog/spdlog.h>

namespace cyxwiz {

namespace {

thread_local NeuralDispatchObserver g_dispatch_observer = nullptr;
thread_local NeuralProviderFailurePolicy g_failure_policy = NeuralProviderFailurePolicy::FallBack;

bool IsBackwardRecurrentOp(NeuralOp op) {
    return op == NeuralOp::LstmBackward || op == NeuralOp::GruBackward || op == NeuralOp::RnnBackward;
}

void Notify(const NeuralDispatchEvent& event) {
    if (g_dispatch_observer) g_dispatch_observer(event);
}

NeuralDispatchEvent MakeEvent(const NeuralOpRequest& request, const char* layer,
                              const INeuralNetworkProvider& provider, NeuralDispatchOutcome outcome) {
    NeuralDispatchEvent event;
    event.provider_id = provider.ProviderId();
    event.provider_version = provider.Version();
    event.op = NeuralOpName(request.op);
    event.layer = layer ? layer : "";
    event.device = NeuralDevicePlatformName(request.target.platform);
    event.outcome = outcome;
    return event;
}

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
    }
    return "unknown";
}

ScopedNeuralDispatchObserver::ScopedNeuralDispatchObserver(NeuralDispatchObserver observer)
    : previous_(g_dispatch_observer) {
    g_dispatch_observer = observer;
}

ScopedNeuralDispatchObserver::~ScopedNeuralDispatchObserver() {
    g_dispatch_observer = previous_;
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

bool DispatchNeuralProviderOp(const NeuralOpRequest& request, const char* layer,
                              const NeuralProviderExecute& execute) {
    auto& registry = NeuralProviderRegistry::Instance();
    const auto provider = registry.FindSupporting(request);
    if (!provider) {
        const auto serving = registry.ListServing(request.target);
        if (!serving.empty()) {
            const NeuralCapability capability = serving.front()->QueryCapability(request);
            NeuralDispatchEvent event =
                MakeEvent(request, layer, *serving.front(), NeuralDispatchOutcome::Declined);
            event.reason = BackendFallbackReasonName(capability.reason);
            event.detail = capability.detail;
            Notify(event);
        }
        return false;
    }
    const NeuralOpStatus status = execute(*provider);
    if (status.ok) {
        Notify(MakeEvent(request, layer, *provider, NeuralDispatchOutcome::Ran));
        return true;
    }
    NeuralDispatchEvent event = MakeEvent(request, layer, *provider, NeuralDispatchOutcome::Failed);
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

} // namespace cyxwiz
