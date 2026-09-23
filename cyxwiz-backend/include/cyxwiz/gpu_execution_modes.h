#pragma once

namespace cyxwiz {

// Operator-owned GPU execution modes (tofix67 slice 4, architecture
// decision 2). Every ArrayFire-first operation family declares how it
// executes on a GPU-selected backend; the engine's placement layer consumes
// the declaration when building placement entries. This table is the
// audited registry of record — change a family's mode only with parity and
// benchmark evidence (ticket validation rules).
enum class GpuExecutionMode {
    DirectArrayFire,
    StagedArrayFire,
    NativeProvider,
    Cpu,
    Unsupported,
};

constexpr const char* GpuExecutionModeName(GpuExecutionMode mode) {
    switch (mode) {
    case GpuExecutionMode::DirectArrayFire:
        return "direct_arrayfire";
    case GpuExecutionMode::StagedArrayFire:
        return "staged_arrayfire";
    case GpuExecutionMode::NativeProvider:
        return "native_provider";
    case GpuExecutionMode::Cpu:
        return "cpu";
    case GpuExecutionMode::Unsupported:
        return "unsupported";
    }
    return "unsupported";
}

// The ticket's initial operation families. Granularity follows the audit in
// tofix67: a family groups operations that share one execution strategy and
// one shape-signature convention.
enum class GpuOperationFamily {
    Recurrent,
    DenseMatmul,
    EmbeddingIndexing,
    AttentionNormalization,
    ConvolutionPooling,
    Activation,
    Loss,
    Optimizer,
    Reduction,
    Transform,
    Evaluation,
    GraphTensorOp,
};

constexpr const char* GpuOperationFamilyName(GpuOperationFamily family) {
    switch (family) {
    case GpuOperationFamily::Recurrent:
        return "recurrent";
    case GpuOperationFamily::DenseMatmul:
        return "dense_matmul";
    case GpuOperationFamily::EmbeddingIndexing:
        return "embedding_indexing";
    case GpuOperationFamily::AttentionNormalization:
        return "attention_normalization";
    case GpuOperationFamily::ConvolutionPooling:
        return "convolution_pooling";
    case GpuOperationFamily::Activation:
        return "activation";
    case GpuOperationFamily::Loss:
        return "loss";
    case GpuOperationFamily::Optimizer:
        return "optimizer";
    case GpuOperationFamily::Reduction:
        return "reduction";
    case GpuOperationFamily::Transform:
        return "transform";
    case GpuOperationFamily::Evaluation:
        return "evaluation";
    case GpuOperationFamily::GraphTensorOp:
        return "graph_tensor_op";
    }
    return "unclassified";
}

// 2026-09-22 initial audit (tofix67 implementation-order step 3). Modes
// reflect what the code does TODAY, not aspirations:
//
// - Recurrent: staged. The LSTM ArrayFire path materializes per timestep
//   (de-facto staging) behind the formal-parameter-byte estimator; GRU and
//   bidirectional LSTM are policy-routed to CPU per exact key. Slice 5
//   names the staged plan; a native provider is step 8.
// - DenseMatmul / EmbeddingIndexing / ConvolutionPooling / Activation /
//   Loss / Optimizer / Reduction / Transform / GraphTensorOp: direct
//   ArrayFire calls with recorded native-CPU fallback per exact key.
// - AttentionNormalization (MultiHeadAttention, Transformer encoder/decoder,
//   PositionalEncoding, LayerNorm): CPU-backed per the part_f contract; the
//   strict-shape ArrayFire variants stay evidence-gated until residency is
//   claimed.
// - Evaluation: native CPU by design — the host-vector evaluation owner is
//   enforced ArrayFire-free by source scan.
constexpr GpuExecutionMode DeclaredGpuExecutionMode(
    GpuOperationFamily family) {
    switch (family) {
    case GpuOperationFamily::Recurrent:
        return GpuExecutionMode::StagedArrayFire;
    case GpuOperationFamily::DenseMatmul:
        return GpuExecutionMode::DirectArrayFire;
    case GpuOperationFamily::EmbeddingIndexing:
        return GpuExecutionMode::DirectArrayFire;
    case GpuOperationFamily::AttentionNormalization:
        return GpuExecutionMode::Cpu;
    case GpuOperationFamily::ConvolutionPooling:
        return GpuExecutionMode::DirectArrayFire;
    case GpuOperationFamily::Activation:
        return GpuExecutionMode::DirectArrayFire;
    case GpuOperationFamily::Loss:
        return GpuExecutionMode::DirectArrayFire;
    case GpuOperationFamily::Optimizer:
        return GpuExecutionMode::DirectArrayFire;
    case GpuOperationFamily::Reduction:
        return GpuExecutionMode::DirectArrayFire;
    case GpuOperationFamily::Transform:
        return GpuExecutionMode::DirectArrayFire;
    case GpuOperationFamily::Evaluation:
        return GpuExecutionMode::Cpu;
    case GpuOperationFamily::GraphTensorOp:
        return GpuExecutionMode::DirectArrayFire;
    }
    return GpuExecutionMode::Unsupported;
}

} // namespace cyxwiz
