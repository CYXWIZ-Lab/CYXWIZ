#pragma once

#include "graph_compiler.h"
#include "dense_activation_configuration_policy.h"
#include "pipeline_runtime_capabilities.h"
#include "cyxwiz/gpu_execution_modes.h"
#include "cyxwiz/neural_provider.h"

#include <string>
#include <algorithm>
#include <cmath>

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
        case gui::NodeType::TimeDistributed: return "TimeDistributed";
        case gui::NodeType::ReLU: return "ReLU";
        case gui::NodeType::LeakyReLU: return "LeakyReLU";
        case gui::NodeType::ELU: return "ELU";
        case gui::NodeType::SELU: return "SELU";
        case gui::NodeType::PReLU: return "PReLU";
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
        case gui::NodeType::TensorBatchMatMul: return "TensorBatchMatMul";
        default: return "Layer";
    }
}

inline bool IsRecurrentLayer(gui::NodeType type) {
    return type == gui::NodeType::GRU ||
           type == gui::NodeType::LSTM ||
           type == gui::NodeType::RNN;
}

inline bool IsTimeDistributedSequenceWrapper(gui::NodeType type) {
    return type == gui::NodeType::TimeDistributed;
}

inline bool IsKnownArrayFireTensorLayer(gui::NodeType type) {
    switch (type) {
        case gui::NodeType::Dense:
        case gui::NodeType::LinearAttention:  // Tensor ops (BatchMatMul, Clip, Exp) only
        case gui::NodeType::Conv1D:
        case gui::NodeType::Conv2D:
        case gui::NodeType::Conv3D:  // sparse gather + matmul
        case gui::NodeType::DepthwiseConv2D:
        case gui::NodeType::MaxPool2D:
        case gui::NodeType::AvgPool2D:
        case gui::NodeType::GlobalMaxPool:
        case gui::NodeType::GlobalAvgPool:
        case gui::NodeType::AdaptiveAvgPool:
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
        case gui::NodeType::GroupNorm:
        case gui::NodeType::InstanceNorm:
        case gui::NodeType::ConvTranspose2D:
        case gui::NodeType::Upsample:
        case gui::NodeType::PixelShuffle:
        case gui::NodeType::Embedding:
        case gui::NodeType::LayerNorm:
        case gui::NodeType::MultiHeadAttention:
        case gui::NodeType::CrossAttention:
        case gui::NodeType::TransformerEncoder:
        case gui::NodeType::TransformerDecoder:
        case gui::NodeType::PositionalEncoding:
        case gui::NodeType::ReLU:
        case gui::NodeType::LeakyReLU:
        case gui::NodeType::ELU:
        case gui::NodeType::SELU:
        case gui::NodeType::PReLU:
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

// Compiler shapes exclude the batch dimension. A TimeDistributed wrapper over a
// [sequence, features] input runs as a tensor layer.
inline LayerCapability ClassifyLayer(const CompiledLayer& layer) {
    auto result = ClassifyLayer(layer.type);
    const bool nonempty = !layer.input_shape.empty() &&
        std::all_of(layer.input_shape.begin(), layer.input_shape.end(), [](size_t d) { return d > 0; });
    if (layer.type == gui::NodeType::TimeDistributed && nonempty && layer.input_shape.size() == 2) {
        result.kind = LayerCapabilityKind::ArrayFireTensor;
    }
    return result;
}

// tofix67 slice 4: the declared execution mode a placement builder stamps on
// its entries is the owning strategy from cyxwiz/gpu_execution_modes.h. The
// ArrayFire-tensor and graph-runtime builders serve families that all declare
// direct ArrayFire (cross-checked by the [gpu_execution] unit tests).
inline void StampDeclaredExecutionMode(BackendPlacementEntry& placement,
                                       cyxwiz::GpuExecutionMode mode) {
    placement.declared_execution_mode = cyxwiz::GpuExecutionModeName(mode);
    placement.explanation +=
        std::string(" Declared execution mode: ") +
        placement.declared_execution_mode + ".";
}

inline BackendPlacementEntry BuildArrayFireTensorPlacement(
    const CompiledLayer& layer) {
    BackendPlacementEntry placement;
    placement.node_id = layer.node_id;
    placement.node_name = layer.name;
    placement.node_type = LayerTypeName(layer.type);
    placement.requested_backend = "auto";
    placement.expected_backend = "ArrayFire active backend";
    placement.status = BackendPlacementStatus::Gpu;
    placement.reason_code = BackendPlacementReason::ArrayFireTensorOpCapable;
    // Model layers run on ArrayFire only (TOFIX140); the Tensor* graph primitives
    // go through core Tensor operations, which still keep a native CPU fallback.
    const bool tensor_primitive = placement.node_type.rfind("Tensor", 0) == 0;
    placement.fallback_backend = tensor_primitive ? "CPU" : "none";
    placement.explanation =
        std::string(placement.node_type) +
        (tensor_primitive
             ? " runs on the active ArrayFire backend for supported dtype/shape paths; "
               "the core Tensor operation may use its native CPU fallback when the "
               "dtype or shape is unsupported, or a backend operation fails."
             : " runs on the active ArrayFire backend (the selected device; ArrayFire's "
               "CPU backend when the CPU is selected). There is no native CPU fallback: "
               "a backend failure stops the run with the device error.");
    placement.suggested_action = "No action needed.";
    StampDeclaredExecutionMode(placement,
                               cyxwiz::GpuExecutionMode::DirectArrayFire);
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
    StampDeclaredExecutionMode(placement, cyxwiz::GpuExecutionMode::Unsupported);
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

// tofix68 placement wiring: consult the neural provider registry with the
// EXACT tuple the run needs. Selection rewrites the decision fields
// (expected backend, status, reason) and names the provider; refusal is
// recorded in the explanation so the debugger shows the provider's verdict
// instead of silence. The declared_execution_mode field is NOT touched —
// it records the family's declared strategy, while provider selection is a
// per-key decision (slice-4 semantics). No placement change when no
// provider is registered (portable builds / no CUDA device).
//
// Device-keyed dispatch (tofix68 rule, 2026-09-23): request.target carries
// the run's SELECTED device; only providers serving that device family are
// consulted, so a CUDA provider is never selected for an OpenCL/oneAPI/CPU
// run on a machine that happens to have a CUDA device. The placement label
// is derived from the provider, never hard-coded.
inline void ApplyNativeProviderPlacement(
    BackendPlacementEntry& placement,
    const cyxwiz::NeuralOpRequest& request) {
    auto& registry = cyxwiz::NeuralProviderRegistry::Instance();
    const auto providers = registry.List();
    if (providers.empty()) {
        return;
    }
    const auto serving = registry.ListServing(request.target);
    if (serving.empty()) {
        std::string registered;
        for (const auto& provider : providers) {
            if (!registered.empty()) {
                registered += ", ";
            }
            registered += std::string(provider->ProviderId()) + ": " +
                          cyxwiz::NeuralDevicePlatformName(
                              provider->Platform());
        }
        placement.explanation +=
            std::string(" Native provider(s) registered (") + registered +
            ") do not serve the run's target device (" +
            cyxwiz::NeuralDevicePlatformName(request.target.platform) +
            "); portable path retained.";
        return;
    }
    if (const auto provider = registry.FindSupporting(request)) {
        placement.expected_backend =
            std::string("native provider ") + provider->ProviderId();
        placement.fallback_backend = "CPU";
        placement.status = BackendPlacementStatus::Gpu;
        placement.reason_code = BackendPlacementReason::NativeProviderSelected;
        placement.explanation +=
            std::string(" Native provider selected for this exact "
                        "contract: ") +
            provider->Version() + ".";
        return;
    }
    const auto capability = serving.front()->QueryCapability(request);
    placement.explanation +=
        std::string(" Native provider (") + serving.front()->ProviderId() +
        ") declined this contract: " + capability.detail + ".";
}

inline bool IsMixedArrayFireGraphRuntimeOp(gui::NodeType type) {
    switch (type) {
        case gui::NodeType::Add:
        case gui::NodeType::Multiply:
        case gui::NodeType::Average:
        case gui::NodeType::Concatenate:
        case gui::NodeType::Split:
        case gui::NodeType::TensorDot:
        case gui::NodeType::TensorBatchMatMul:
        case gui::NodeType::TensorCompare:
        case gui::NodeType::TensorLogicalMask:
            return true;
        default:
            return false;
    }
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
        } else if (node.type == gui::NodeType::Split) {
            placement.explanation +=
                " Split slices its input into two tensors with Tensor::Split "
                "and concatenates the branch gradients back in backward.";
        } else if (node.type == gui::NodeType::TensorLogicalMask) {
            placement.explanation +=
                " Current TensorLogicalMask ArrayFire coverage is "
                "Float32/Float64 matching-dtype tensor logical operations and "
                "unary logical not.";
        }
        placement.suggested_action =
            "No correctness action needed. Treat performance as workload-specific "
            "and use focused benchmarks before making a GPU speed claim.";
        StampDeclaredExecutionMode(
            placement,
            cyxwiz::DeclaredGpuExecutionMode(
                cyxwiz::GpuOperationFamily::GraphTensorOp));
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
