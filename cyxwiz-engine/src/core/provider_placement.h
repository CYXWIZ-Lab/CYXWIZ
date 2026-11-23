#pragma once

// Which layers a native neural provider runs (TOFIX232). The requests are
// built with the backend's request builders (cyxwiz/neural_provider_dispatch.h),
// the same ones each layer uses at run time, from the compiled shapes; the
// registry is asked for the device given (the run's bound device for the
// executable plan, the active device for the Compile report).

#include "execution_device_context.h"
#include "graph_compiler.h"
#include "spatial_layer_shapes.h"
#include "spatial_sequential_head.h"
#include "transformer_configuration_policy.h"

#include <cyxwiz/neural_provider.h>
#include <cyxwiz/neural_provider_dispatch.h>

#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz {

// The device a run bound, as a provider target.
inline NeuralDeviceTarget NeuralTargetOfExecutionContext(const ExecutionDeviceContext& context) {
    NeuralDeviceTarget target;
    target.device_id = context.effective_device_id;
    if (context.effective_backend == "arrayfire_cuda") target.platform = DeviceType::CUDA;
    else if (context.effective_backend == "arrayfire_opencl") target.platform = DeviceType::OPENCL;
    else if (context.effective_backend == "arrayfire_oneapi") target.platform = DeviceType::ONEAPI;
    else target.platform = DeviceType::CPU;
    return target;
}

namespace provider_placement_detail {

inline std::string Param(const CompiledLayer& layer, const char* key) {
    const auto it = layer.parameters.find(key);
    return it == layer.parameters.end() ? std::string() : it->second;
}

inline size_t SizeParam(const CompiledLayer& layer, const char* key, size_t fallback) {
    try {
        const std::string value = Param(layer, key);
        if (value.empty()) return fallback;
        const int parsed = std::stoi(value);
        return parsed > 0 ? static_cast<size_t>(parsed) : fallback;
    } catch (...) {
        return fallback;
    }
}

inline bool BoolParam(const CompiledLayer& layer, const char* key) {
    const std::string value = Param(layer, key);
    return value == "true" || value == "1";
}

inline void AddPair(std::vector<NeuralOpRequest>& requests, NeuralOp forward, NeuralOp backward,
                    const NeuralDeviceTarget& target, const RecurrentProviderShape& shape) {
    requests.push_back(RecurrentProviderRequest(forward, target, shape));
    requests.push_back(RecurrentProviderRequest(backward, target, shape));
}

// LSTM / GRU / RNN as ModelBuilder builds them: batch-first, a zero starting
// state and no dropout between stacked layers. A bidirectional layer runs
// each direction and level as a single-layer child (deeper levels read 2 *
// hidden features).
inline std::vector<NeuralOpRequest> RecurrentRequests(const TrainingConfiguration& config,
                                                      const CompiledLayer& layer,
                                                      const NeuralDeviceTarget& target) {
    if (layer.input_shape.size() != 2) return {};
    NeuralOp forward = NeuralOp::LstmForward;
    NeuralOp backward = NeuralOp::LstmBackward;
    RecurrentProviderShape shape;
    if (layer.type == gui::NodeType::GRU) {
        forward = NeuralOp::GruForward;
        backward = NeuralOp::GruBackward;
    } else if (layer.type == gui::NodeType::RNN) {
        forward = NeuralOp::RnnForward;
        backward = NeuralOp::RnnBackward;
        shape.activation = Param(layer, "nonlinearity") == "relu" ? NeuralActivation::Relu : NeuralActivation::Tanh;
    }
    shape.batch = static_cast<size_t>(std::max(1, config.batch_size));
    shape.seq = layer.input_shape[0];
    shape.input = layer.input_shape[1];
    shape.hidden = SizeParam(layer, "hidden_size", 128);
    const size_t levels = SizeParam(layer, "num_layers", 1);
    std::vector<NeuralOpRequest> requests;
    if (!BoolParam(layer, "bidirectional")) {
        shape.layers = levels;
        AddPair(requests, forward, backward, target, shape);
        return requests;
    }
    shape.layers = 1;
    AddPair(requests, forward, backward, target, shape);
    if (levels > 1) {
        shape.input = 2 * shape.hidden;
        AddPair(requests, forward, backward, target, shape);
    }
    return requests;
}

inline std::optional<std::vector<size_t>> SpatialInputShape(const TrainingConfiguration& config, size_t index) {
    const auto& layer = config.layers[index];
    if (layer.input_shape.size() == 3) return layer.input_shape;
    try {
        const auto head = ResolveSpatialSequentialHead(config);
        if (head && index < head->input_shapes.size() && head->input_shapes[index].size() == 3)
            return head->input_shapes[index];
    } catch (const std::invalid_argument&) {
    }
    return std::nullopt;
}

inline std::vector<NeuralOpRequest> Conv2dRequests(const TrainingConfiguration& config, size_t index,
                                                   const NeuralDeviceTarget& target) {
    const auto& layer = config.layers[index];
    const auto in = SpatialInputShape(config, index);
    if (!in) return {};
    spatial::Geometry g;
    try {
        g = spatial::ResolveGeometry(layer.type, layer.parameters, (*in)[2]);
    } catch (const std::invalid_argument&) {
        return {};
    }
    Conv2dProviderShape shape;
    shape.batch = static_cast<size_t>(std::max(1, config.batch_size));
    shape.height = (*in)[0];
    shape.width = (*in)[1];
    shape.in_channels = (*in)[2];
    shape.out_channels = static_cast<size_t>(g.channels_out);
    shape.kernel = static_cast<size_t>(g.kernel);
    shape.stride = static_cast<size_t>(g.stride);
    shape.padding = static_cast<size_t>(g.padding);
    shape.bias = g.use_bias;
    return {Conv2dProviderRequest(NeuralOp::Conv2dForward, target, shape),
            Conv2dProviderRequest(NeuralOp::Conv2dBackward, target, shape)};
}

// MultiHeadAttention (its module declares a causal or no mask) and a
// decoder-only TransformerDecoder's self-attention (causal).
inline std::vector<NeuralOpRequest> AttentionRequests(const TrainingConfiguration& config,
                                                      const CompiledLayer& layer,
                                                      const NeuralDeviceTarget& target) {
    if (layer.input_shape.size() != 2) return {};
    TransformerConfiguration c;
    if (ResolveTransformerConfiguration(layer.type, layer.parameters, c)) return {};
    const size_t embed = layer.input_shape[1];
    if (c.num_heads == 0 || embed % c.num_heads != 0) return {};
    const bool decoder = layer.type == gui::NodeType::TransformerDecoder;
    AttentionProviderShape shape;
    shape.batch = static_cast<size_t>(std::max(1, config.batch_size));
    shape.seq = layer.input_shape[0];
    shape.kv_seq = shape.seq;
    shape.heads = c.num_heads;
    shape.kv_heads = c.num_kv_heads > 0 ? c.num_kv_heads : c.num_heads;
    shape.head_dim = embed / c.num_heads;
    shape.causal = decoder || c.causal;
    shape.sliding_window = shape.causal ? c.sliding_window : 0;
    shape.logit_softcap = c.attn_logit_softcap;
    shape.softmax_scale = 1.0f / std::sqrt(static_cast<float>(shape.head_dim));
    shape.alibi = c.position_encoding == "alibi";
    shape.training = true;
    shape.attention_dropout = c.dropout;
    shape.dropout_seed = 1;
    return {AttentionProviderRequest(NeuralOp::AttentionForward, target, shape),
            AttentionProviderRequest(NeuralOp::AttentionBackward, target, shape)};
}

}  // namespace provider_placement_detail

// The provider requests a training run of config.layers[index] makes (forward
// and backward); empty when the layer never asks a provider or its shapes are
// not known at compile time.
inline std::vector<NeuralOpRequest> ProviderRequestsForLayer(const TrainingConfiguration& config, size_t index,
                                                             const NeuralDeviceTarget& target) {
    if (index >= config.layers.size()) return {};
    const auto& layer = config.layers[index];
    switch (layer.type) {
        case gui::NodeType::LSTM:
        case gui::NodeType::GRU:
        case gui::NodeType::RNN:
            return provider_placement_detail::RecurrentRequests(config, layer, target);
        case gui::NodeType::Conv2D:
            return provider_placement_detail::Conv2dRequests(config, index, target);
        case gui::NodeType::MultiHeadAttention:
        case gui::NodeType::TransformerDecoder:
            return provider_placement_detail::AttentionRequests(config, layer, target);
        default:
            return {};
    }
}

struct LayerProviderSelection {
    // The provider that serves every request; null: ArrayFire runs the layer.
    std::shared_ptr<INeuralNetworkProvider> provider;
    // Why, for the placement's explanation (empty when no provider is registered).
    std::string note;
};

inline LayerProviderSelection SelectLayerProvider(const std::vector<NeuralOpRequest>& requests) {
    LayerProviderSelection selection;
    if (requests.empty()) return selection;
    auto& registry = NeuralProviderRegistry::Instance();
    const auto providers = registry.List();
    if (providers.empty()) return selection;
    const NeuralDeviceTarget& target = requests.front().target;
    const auto serving = registry.ListServing(target);
    if (serving.empty()) {
        std::string registered;
        for (const auto& provider : providers) {
            if (!registered.empty()) registered += ", ";
            registered += std::string(provider->ProviderId()) + ": " + NeuralDevicePlatformName(provider->Platform());
        }
        selection.note = " Native provider(s) registered (" + registered +
                         ") do not serve the run's target device (" + NeuralDevicePlatformName(target.platform) +
                         "); portable path retained.";
        return selection;
    }
    std::shared_ptr<INeuralNetworkProvider> chosen;
    std::string ops;
    for (const auto& request : requests) {
        const auto provider = registry.FindSupporting(request);
        if (!provider) {
            const auto capability = serving.front()->QueryCapability(request);
            selection.note = std::string(" Native provider (") + serving.front()->ProviderId() + ") declined " +
                             NeuralOpName(request.op) + ": " + capability.detail + ".";
            return selection;
        }
        if (!chosen) chosen = provider;
        const std::string op = NeuralOpName(request.op);
        if (ops.find(op) == std::string::npos) ops += (ops.empty() ? "" : ", ") + op;
    }
    selection.provider = chosen;
    selection.note = " Native provider selected for this exact contract (" + ops + "): " + chosen->Version() + ".";
    return selection;
}

// The Compile report's entry: names the provider when one serves every
// request of the layer; otherwise records why not. A provider failure runs
// the layer's ArrayFire path in a Compatibility run.
inline void ApplyProviderPlacement(BackendPlacementEntry& placement, const std::vector<NeuralOpRequest>& requests) {
    const LayerProviderSelection selection = SelectLayerProvider(requests);
    placement.explanation += selection.note;
    if (!selection.provider) return;
    placement.expected_backend = std::string("native provider ") + selection.provider->ProviderId();
    placement.fallback_backend = "ArrayFire";
    placement.status = BackendPlacementStatus::Gpu;
    placement.reason_code = BackendPlacementReason::NativeProviderSelected;
}

} // namespace cyxwiz
