#pragma once

#include "graph_compiler.h"
#include "pipeline_runtime_capabilities.h"
#include "cyxwiz/backend_placement_observation.h"

#include <string>

namespace cyxwiz::backend_placement {

enum class LayerCapabilityKind {
    ArrayFireTensor,
    CpuBackedModelLayer,
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
        case gui::NodeType::Conv1D: return "Conv1D";
        case gui::NodeType::Conv2D: return "Conv2D";
        case gui::NodeType::Conv3D: return "Conv3D";
        case gui::NodeType::DepthwiseConv2D: return "DepthwiseConv2D";
        case gui::NodeType::MaxPool2D: return "MaxPool2D";
        case gui::NodeType::AvgPool2D: return "AvgPool2D";
        case gui::NodeType::GlobalMaxPool: return "GlobalMaxPool";
        case gui::NodeType::GlobalAvgPool: return "GlobalAvgPool";
        case gui::NodeType::AdaptiveAvgPool: return "AdaptiveAvgPool";
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
        case gui::NodeType::LayerNorm: return "LayerNorm";
        case gui::NodeType::GroupNorm: return "GroupNorm";
        case gui::NodeType::InstanceNorm: return "InstanceNorm";
        case gui::NodeType::MultiHeadAttention: return "MultiHeadAttention";
        case gui::NodeType::SelfAttention: return "SelfAttention";
        case gui::NodeType::CrossAttention: return "CrossAttention";
        case gui::NodeType::LinearAttention: return "LinearAttention";
        case gui::NodeType::ConvTranspose2D: return "ConvTranspose2D";
        case gui::NodeType::Upsample: return "Upsample";
        case gui::NodeType::PixelShuffle: return "PixelShuffle";
        case gui::NodeType::PolicyNetwork: return "PolicyNetwork";
        case gui::NodeType::ValueNetwork: return "ValueNetwork";
        case gui::NodeType::Embedding: return "Embedding";
        case gui::NodeType::TransformerEncoder: return "TransformerEncoder";
        case gui::NodeType::PositionalEncoding: return "PositionalEncoding";
        case gui::NodeType::TransformerDecoder: return "TransformerDecoder";
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
        case gui::NodeType::Add: return "Add";
        case gui::NodeType::Multiply: return "Multiply";
        case gui::NodeType::Average: return "Average";
        case gui::NodeType::Concatenate: return "Concatenate";
        case gui::NodeType::TensorDot: return "TensorDot";
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

inline bool IsKnownCpuBackedModelLayer(gui::NodeType type) {
    return type == gui::NodeType::LayerNorm ||
           type == gui::NodeType::MultiHeadAttention ||
           type == gui::NodeType::TransformerEncoder ||
           type == gui::NodeType::TransformerDecoder ||
           type == gui::NodeType::PositionalEncoding;
}

inline LayerCapability ClassifyLayer(gui::NodeType type) {
    LayerCapability capability;
    capability.type_name = LayerTypeName(type);
    if (IsPipelineUnsupportedSequentialModelLayer(type)) {
        capability.kind = LayerCapabilityKind::UnsupportedSequentialModelLayer;
    } else if (IsKnownCpuBackedModelLayer(type)) {
        capability.kind = LayerCapabilityKind::CpuBackedModelLayer;
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

inline BackendPlacementEntry BuildCpuBackedModelLayerPlacement(
    const CompiledLayer& layer) {
    BackendPlacementEntry placement;
    placement.node_id = layer.node_id;
    placement.node_name = layer.name;
    placement.node_type = LayerTypeName(layer.type);
    placement.requested_backend = "auto";
    placement.expected_backend = "CPU";
    placement.fallback_backend = "CPU";
    placement.status = BackendPlacementStatus::Cpu;
    placement.reason_code = BackendPlacementReason::GraphRuntimeCpuBacked;
    placement.explanation =
        std::string(placement.node_type) +
        " is supported by ModelBuilder/SequentialModel, but the current "
        "module implementation is CPU-backed. Training is correct, but this "
        "layer should not be counted as GPU-resident until a focused ArrayFire "
        "implementation and residency/parity test are added.";
    if (layer.type == gui::NodeType::MultiHeadAttention) {
        placement.suggested_action =
            "No correctness action needed for single-input self-attention. "
            "For performance-sensitive transformer graphs, add and test a "
            "focused ArrayFire MultiHeadAttention path before claiming GPU "
            "support. Keep connected Key/Value/Context cross-attention "
            "blocked until its graph/runtime/export contract is proven.";
    } else if (layer.type == gui::NodeType::LayerNorm) {
        placement.suggested_action =
            "No correctness action needed. For performance-sensitive "
            "transformer graphs, add and test an ArrayFire LayerNorm path "
            "before claiming GPU support.";
    } else {
        placement.suggested_action =
            "No correctness action needed. For performance-sensitive "
            "transformer graphs, add and test a focused ArrayFire "
            "implementation plus residency/parity coverage for this layer "
            "before claiming GPU support.";
    }
    return placement;
}

inline size_t PlacementSizeParam(const CompiledLayer& layer,
                                 const char* key,
                                 size_t fallback) {
    const auto it = layer.parameters.find(key);
    if (it == layer.parameters.end()) {
        return fallback;
    }
    try {
        return static_cast<size_t>(std::stoull(it->second));
    } catch (...) {
        return fallback;
    }
}

inline bool IsActivationPlacementLayer(gui::NodeType type) {
    switch (type) {
        case gui::NodeType::ReLU:
        case gui::NodeType::Sigmoid:
        case gui::NodeType::Tanh:
            return true;
        default:
            return false;
    }
}

inline std::string BuildArrayFireTensorPlacementShapeSignature(
    const CompiledLayer& layer) {
    if (layer.type == gui::NodeType::Dense) {
        return BuildDensePlacementShapeSignature(
            layer.input_shape,
            static_cast<size_t>(layer.units));
    }
    if (layer.type == gui::NodeType::Embedding) {
        return BuildEmbeddingPlacementShapeSignature(
            PlacementSizeParam(layer, "num_embeddings", 0),
            PlacementSizeParam(layer, "embedding_dim", 0),
            layer.input_shape,
            "int32");
    }
    if (IsActivationPlacementLayer(layer.type)) {
        return BuildActivationPlacementShapeSignature(
            layer.input_shape,
            "float32");
    }
    return BuildTensorLayerPlacementShapeSignature(
        layer.input_shape,
        layer.output_shape);
}

inline const char* BuildArrayFireTensorPlacementObservationDtype(
    const CompiledLayer& layer) {
    return layer.type == gui::NodeType::Embedding ? "int32" : "float32";
}

inline BackendPlacementEntry BuildArrayFireTensorPlacement(
    const CompiledLayer& layer) {
    const std::string layer_type_name = LayerTypeName(layer.type);
    const std::string shape_signature =
        BuildArrayFireTensorPlacementShapeSignature(layer);
    const std::string observation_dtype =
        BuildArrayFireTensorPlacementObservationDtype(layer);
    BackendPlacementObservation cached_observation;
    const bool cached_fallback =
        TryGetBackendPlacementObservationForActiveDevice(
            layer_type_name,
            "cuda",
            observation_dtype,
            shape_signature,
            cached_observation);
    BackendPlacementEntry placement;
    placement.node_id = layer.node_id;
    placement.node_name = layer.name;
    placement.node_type = layer_type_name;
    placement.requested_backend = "auto";
    placement.expected_backend = cached_fallback
        ? "CPU"
        : "ArrayFire active backend";
    placement.fallback_backend = "CPU";
    placement.status = cached_fallback
        ? BackendPlacementStatus::Cpu
        : BackendPlacementStatus::Gpu;
    placement.reason_code = cached_fallback
        ? cached_observation.reason_code
        : BackendPlacementReason::ArrayFireTensorOpCapable;
    if (cached_fallback) {
        placement.observation_source = cached_observation.source;
        placement.observation_device = cached_observation.device;
        placement.observation_dtype = cached_observation.dtype;
        placement.observation_shape_signature =
            cached_observation.shape_signature;
        placement.observation_detail = cached_observation.detail;
        placement.observation_timestamp = cached_observation.timestamp;
        placement.observation_probe_outcome =
            cached_observation.probe_outcome;
        placement.observation_probe_scope = cached_observation.probe_scope;
        placement.explanation =
            std::string(placement.node_type) +
            " is expected to run on native CPU because a previous runtime fallback "
            "observation for this exact backend/device/dtype/shape reported "
            "a backend failure (reason=" + cached_observation.reason_code +
            ", source=" + cached_observation.source + "). Device: " +
            cached_observation.device + ". Shape signature: " +
            cached_observation.shape_signature + ".";
        placement.suggested_action =
            "Training can continue. Inspect the fallback reason and reduce "
            "batch size/features or use the ArrayFire CPU backend for this "
            "tensor-layer shape until the backend path is fixed.";
    } else {
        placement.explanation =
            std::string(placement.node_type) +
            " is compiled as a standard tensor/model layer. The runtime can "
            "execute supported dtype/shape paths on the active ArrayFire backend "
            "and may use a recorded native CPU fallback when the dtype or shape "
            "is unsupported, or a backend operation fails.";
        placement.suggested_action = "No action needed.";
    }
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
        "existing model path and may use the active backend or a recorded "
        "native CPU fallback.";
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
        "TimeDistributed Dense is a concrete shared linear projection across "
        "time steps, not a generic inner-layer wrapper. Placement remains "
        "unknown because its rank-3 to rank-2 reshape boundary and inner "
        "Linear execution are not yet proven as one device-resident contract.";
    placement.suggested_action =
        "No action needed for compile support. For performance-sensitive "
        "sequence models, validate the reshape and Linear runtime placement "
        "before relying on GPU residency.";
    return placement;
}

inline bool IsMixedArrayFireGraphRuntimeOp(gui::NodeType type) {
    switch (type) {
        case gui::NodeType::Add:
        case gui::NodeType::Multiply:
        case gui::NodeType::Average:
        case gui::NodeType::Concatenate:
        case gui::NodeType::TensorDot:
        case gui::NodeType::TensorCompare:
        case gui::NodeType::TensorLogicalMask:
            return true;
        default:
            return false;
    }
}

inline bool IsCpuBackedGraphRuntimeOp(gui::NodeType) {
    return false;
}

inline BackendPlacementEntry BuildGraphRuntimePlacement(
    const CompiledGraphNode& node) {
    BackendPlacementEntry placement;
    placement.node_id = node.node_id;
    placement.node_name = node.name;
    placement.node_type = LayerTypeName(node.type);
    placement.requested_backend = "auto";
    placement.fallback_backend = "CPU";

    if (IsMixedArrayFireGraphRuntimeOp(node.type)) {
        placement.expected_backend = "ArrayFire tensor primitive when supported";
        placement.status = BackendPlacementStatus::Mixed;
        placement.reason_code = BackendPlacementReason::GraphRuntimeArrayFireMixed;
        placement.explanation =
            std::string(placement.node_type) +
            " is a graph-runtime tensor op. It can use ArrayFire-backed "
            "Tensor primitives for supported dtypes/shapes, but it keeps CPU "
            "fallback for unsupported dtypes, CPU-only builds, or backend "
            "failures.";
        if (node.type == gui::NodeType::TensorDot) {
            placement.explanation +=
                " Current TensorDot ArrayFire coverage is Float32/Float64 "
                "forward for 1D vector dot and 2D row-wise dot; graph backward "
                "still uses the graph-executable gradient path.";
        } else if (node.type == gui::NodeType::Add) {
            placement.explanation +=
                " Current Add ArrayFire coverage includes Float32/Float64 "
                "row-major 2D elementwise addition, including shared-input "
                "and independent graph fan-in residency coverage.";
        } else if (node.type == gui::NodeType::Multiply) {
            placement.explanation +=
                " Current Multiply ArrayFire coverage includes Float32/Float64 "
                "row-major 2D elementwise multiplication, including shared-input "
                "and independent graph fan-in residency coverage.";
        } else if (node.type == gui::NodeType::Average) {
            placement.explanation +=
                " Current Average ArrayFire coverage reuses Float32/Float64 "
                "row-major 2D addition and scalar scaling, including "
                "shared-input and independent graph fan-in residency coverage.";
        } else if (node.type == gui::NodeType::TensorCompare) {
            placement.explanation +=
                " Current TensorCompare ArrayFire coverage is Float32/Float64 "
                "forward for tensor and scalar comparisons with matching "
                "tensor dtypes.";
        } else if (node.type == gui::NodeType::Concatenate) {
            placement.explanation +=
                " Current Concatenate ArrayFire coverage is Float32/Float64 "
                "2D concatenation through the row-major tensor bridge.";
        } else if (node.type == gui::NodeType::TensorLogicalMask) {
            placement.explanation +=
                " Current TensorLogicalMask ArrayFire coverage is "
                "Float32/Float64 matching-dtype tensor logical operations and "
                "unary logical not.";
        }
        placement.suggested_action =
            "No correctness action needed. Treat performance as workload-specific "
            "and use focused benchmarks before making a GPU speed claim.";
        return placement;
    }

    if (IsCpuBackedGraphRuntimeOp(node.type)) {
        placement.expected_backend = "CPU";
        placement.status = BackendPlacementStatus::Cpu;
        placement.reason_code = BackendPlacementReason::GraphRuntimeCpuBacked;
        placement.explanation =
            std::string(placement.node_type) +
            " is executable as a graph-runtime op, but the current primitive "
            "path is CPU-backed and may materialize host data.";
        placement.suggested_action =
            "No correctness action needed. Add a focused ArrayFire primitive "
            "and residency test before claiming GPU support for this graph op.";
        return placement;
    }

    placement.expected_backend = BackendPlacementStatus::Unknown;
    placement.status = BackendPlacementStatus::Unknown;
    placement.reason_code = BackendPlacementReason::BackendCapabilityUnclassified;
    placement.explanation =
        std::string(placement.node_type) +
        " is recorded as a graph-runtime op, but no backend placement rule "
        "exists for it yet.";
    placement.suggested_action =
        "Classify this graph op before relying on any GPU/fallback claim.";
    return placement;
}

} // namespace cyxwiz::backend_placement
