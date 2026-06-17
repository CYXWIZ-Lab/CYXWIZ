#pragma once

#include "graph_compiler.h"
#include "pipeline_runtime_capabilities.h"

#include <string>

namespace cyxwiz::backend_placement {

enum class LayerCapabilityKind {
    ArrayFireTensor,
    Recurrent,
    TimeDistributedSequenceWrapper,
    UnsupportedSequentialModelLayer,
    Unclassified
};

struct LayerCapability {
    LayerCapabilityKind kind = LayerCapabilityKind::Unclassified;
    const char* type_name = "Layer";
};

inline const char* LayerTypeName(gui::NodeType type) {
    switch (type) {
        case gui::NodeType::Dense: return "Dense";
        case gui::NodeType::Conv2D: return "Conv2D";
        case gui::NodeType::MaxPool2D: return "MaxPool2D";
        case gui::NodeType::AvgPool2D: return "AvgPool2D";
        case gui::NodeType::GlobalMaxPool: return "GlobalMaxPool";
        case gui::NodeType::GlobalAvgPool: return "GlobalAvgPool";
        case gui::NodeType::Flatten: return "Flatten";
        case gui::NodeType::Reshape: return "Reshape";
        case gui::NodeType::View: return "View";
        case gui::NodeType::Permute: return "Permute";
        case gui::NodeType::Squeeze: return "Squeeze";
        case gui::NodeType::Unsqueeze: return "Unsqueeze";
        case gui::NodeType::TensorBroadcastTo: return "TensorBroadcastTo";
        case gui::NodeType::TensorExpand: return "TensorExpand";
        case gui::NodeType::TensorIndexSelect: return "TensorIndexSelect";
        case gui::NodeType::TensorAbs: return "TensorAbs";
        case gui::NodeType::TensorExp: return "TensorExp";
        case gui::NodeType::TensorLog: return "TensorLog";
        case gui::NodeType::TensorSqrt: return "TensorSqrt";
        case gui::NodeType::TensorSign: return "TensorSign";
        case gui::NodeType::TensorPow: return "TensorPow";
        case gui::NodeType::TensorClip: return "TensorClip";
        case gui::NodeType::TensorCompare: return "TensorCompare";
        case gui::NodeType::TensorLogicalMask: return "TensorLogicalMask";
        case gui::NodeType::TensorSum: return "TensorSum";
        case gui::NodeType::TensorMean: return "TensorMean";
        case gui::NodeType::TensorMax: return "TensorMax";
        case gui::NodeType::TensorMin: return "TensorMin";
        case gui::NodeType::TensorProd: return "TensorProd";
        case gui::NodeType::TensorVar: return "TensorVar";
        case gui::NodeType::TensorStd: return "TensorStd";
        case gui::NodeType::Dropout: return "Dropout";
        case gui::NodeType::BatchNorm: return "BatchNorm";
        case gui::NodeType::ConvTranspose2D: return "ConvTranspose2D";
        case gui::NodeType::Upsample: return "Upsample";
        case gui::NodeType::PixelShuffle: return "PixelShuffle";
        case gui::NodeType::PolicyNetwork: return "PolicyNetwork";
        case gui::NodeType::ValueNetwork: return "ValueNetwork";
        case gui::NodeType::Embedding: return "Embedding";
        case gui::NodeType::LSTM: return "LSTM";
        case gui::NodeType::GRU: return "GRU";
        case gui::NodeType::RNN: return "RNN";
        case gui::NodeType::Bidirectional: return "Bidirectional";
        case gui::NodeType::TimeDistributed: return "TimeDistributed";
        case gui::NodeType::ReLU: return "ReLU";
        case gui::NodeType::LeakyReLU: return "LeakyReLU";
        case gui::NodeType::ELU: return "ELU";
        case gui::NodeType::GELU: return "GELU";
        case gui::NodeType::Swish: return "Swish";
        case gui::NodeType::Mish: return "Mish";
        case gui::NodeType::Sigmoid: return "Sigmoid";
        case gui::NodeType::Tanh: return "Tanh";
        case gui::NodeType::Softmax: return "Softmax";
        default: return "Layer";
    }
}

inline bool IsRecurrentLayer(gui::NodeType type) {
    return type == gui::NodeType::GRU ||
           type == gui::NodeType::LSTM;
}

inline bool IsTimeDistributedSequenceWrapper(gui::NodeType type) {
    return type == gui::NodeType::TimeDistributed;
}

inline bool IsKnownArrayFireTensorLayer(gui::NodeType type) {
    switch (type) {
        case gui::NodeType::Dense:
        case gui::NodeType::Conv2D:
        case gui::NodeType::MaxPool2D:
        case gui::NodeType::AvgPool2D:
        case gui::NodeType::GlobalMaxPool:
        case gui::NodeType::GlobalAvgPool:
        case gui::NodeType::Flatten:
        case gui::NodeType::Reshape:
        case gui::NodeType::View:
        case gui::NodeType::Permute:
        case gui::NodeType::Squeeze:
        case gui::NodeType::Unsqueeze:
        case gui::NodeType::TensorBroadcastTo:
        case gui::NodeType::TensorExpand:
        case gui::NodeType::TensorIndexSelect:
        case gui::NodeType::TensorAbs:
        case gui::NodeType::TensorExp:
        case gui::NodeType::TensorLog:
        case gui::NodeType::TensorSqrt:
        case gui::NodeType::TensorSign:
        case gui::NodeType::TensorPow:
        case gui::NodeType::TensorClip:
        case gui::NodeType::TensorCompare:
        case gui::NodeType::TensorLogicalMask:
        case gui::NodeType::TensorSum:
        case gui::NodeType::TensorMean:
        case gui::NodeType::TensorMax:
        case gui::NodeType::TensorMin:
        case gui::NodeType::TensorProd:
        case gui::NodeType::TensorVar:
        case gui::NodeType::TensorStd:
        case gui::NodeType::Dropout:
        case gui::NodeType::BatchNorm:
        case gui::NodeType::ConvTranspose2D:
        case gui::NodeType::Upsample:
        case gui::NodeType::PixelShuffle:
        case gui::NodeType::Embedding:
        case gui::NodeType::ReLU:
        case gui::NodeType::LeakyReLU:
        case gui::NodeType::ELU:
        case gui::NodeType::GELU:
        case gui::NodeType::Swish:
        case gui::NodeType::Mish:
        case gui::NodeType::Sigmoid:
        case gui::NodeType::Tanh:
        case gui::NodeType::Softmax:
            return true;
        default:
            return false;
    }
}

inline LayerCapability ClassifyLayer(gui::NodeType type) {
    LayerCapability capability;
    capability.type_name = LayerTypeName(type);
    if (IsPipelineUnsupportedSequentialModelLayer(type)) {
        capability.kind = LayerCapabilityKind::UnsupportedSequentialModelLayer;
    } else if (IsKnownArrayFireTensorLayer(type)) {
        capability.kind = LayerCapabilityKind::ArrayFireTensor;
    } else if (IsRecurrentLayer(type)) {
        capability.kind = LayerCapabilityKind::Recurrent;
    } else if (IsTimeDistributedSequenceWrapper(type)) {
        capability.kind = LayerCapabilityKind::TimeDistributedSequenceWrapper;
    } else {
        capability.kind = LayerCapabilityKind::Unclassified;
    }
    return capability;
}

inline BackendPlacementEntry BuildArrayFireTensorPlacement(
    const CompiledLayer& layer) {
    BackendPlacementEntry placement;
    placement.node_id = layer.node_id;
    placement.node_name = layer.name;
    placement.node_type = LayerTypeName(layer.type);
    placement.requested_backend = "auto";
    placement.expected_backend = "ArrayFire active backend";
    placement.fallback_backend = "CPU";
    placement.status = BackendPlacementStatus::Gpu;
    placement.reason_code = BackendPlacementReason::ArrayFireTensorOpCapable;
    placement.explanation =
        std::string(placement.node_type) +
        " is compiled as a standard tensor/model layer. The runtime will "
        "execute it on the active ArrayFire backend when that backend is "
        "available for the selected device and dtype.";
    placement.suggested_action = "No action needed.";
    return placement;
}

inline BackendPlacementEntry BuildUnsupportedSequentialModelPlacement(
    const CompiledLayer& layer,
    const char* reason) {
    BackendPlacementEntry placement;
    placement.node_id = layer.node_id;
    placement.node_name = layer.name;
    placement.node_type = LayerTypeName(layer.type);
    placement.requested_backend = "auto";
    placement.expected_backend = BackendPlacementStatus::Unsupported;
    placement.status = BackendPlacementStatus::Unsupported;
    placement.reason_code =
        BackendPlacementReason::UnsupportedSequentialModelLayer;
    placement.explanation =
        std::string(placement.node_type) +
        " cannot be placed on a backend because it is not supported by "
        "ModelBuilder/SequentialModel yet. " +
        (reason != nullptr ? reason : "");
    placement.suggested_action =
        "Replace this node with a supported layer or keep it disconnected from "
        "the selected training path until backend support lands.";
    return placement;
}

inline BackendPlacementEntry BuildUnclassifiedPlacement(
    const CompiledLayer& layer) {
    BackendPlacementEntry placement;
    placement.node_id = layer.node_id;
    placement.node_name = layer.name;
    placement.node_type = LayerTypeName(layer.type);
    placement.requested_backend = "auto";
    placement.expected_backend = BackendPlacementStatus::Unknown;
    placement.fallback_backend = "CPU";
    placement.status = BackendPlacementStatus::Unknown;
    placement.reason_code = BackendPlacementReason::BackendCapabilityUnclassified;
    placement.explanation =
        std::string(placement.node_type) +
        " is compiled, but the compiler does not yet have a precise backend "
        "capability rule for this node type. Runtime will execute through the "
        "existing model path and may use the active backend or CPU fallback.";
    placement.suggested_action =
        "No action needed unless training is slow; this node type should be "
        "classified in the backend capability registry.";
    return placement;
}

inline BackendPlacementEntry BuildTimeDistributedSequenceWrapperPlacement(
    const CompiledLayer& layer) {
    BackendPlacementEntry placement;
    placement.node_id = layer.node_id;
    placement.node_name = layer.name;
    placement.node_type = LayerTypeName(layer.type);
    placement.requested_backend = "auto";
    placement.expected_backend = BackendPlacementStatus::Unknown;
    placement.fallback_backend = "CPU";
    placement.status = BackendPlacementStatus::Unknown;
    placement.reason_code = BackendPlacementReason::TimeDistributedSequenceWrapper;
    placement.explanation =
        "TimeDistributed is a recognized sequence wrapper that applies a dense "
        "projection across time steps. Backend placement is intentionally "
        "reported as unknown until the compiler can classify the wrapper and "
        "its inner projection as one precise device contract.";
    placement.suggested_action =
        "No action needed for compile support. For performance-sensitive "
        "sequence models, validate runtime placement and add a dedicated "
        "TimeDistributed backend contract before relying on GPU residency.";
    return placement;
}

} // namespace cyxwiz::backend_placement
