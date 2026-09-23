#include <catch2/catch_test_macros.hpp>

#include <cyxwiz/gpu_execution_modes.h>

#include <string>

TEST_CASE("GPU execution mode and family names pin the declared spellings",
          "[gpu_execution][taxonomy]") {
    using namespace cyxwiz;

    CHECK(std::string(GpuExecutionModeName(
              GpuExecutionMode::DirectArrayFire)) == "direct_arrayfire");
    CHECK(std::string(GpuExecutionModeName(
              GpuExecutionMode::StagedArrayFire)) == "staged_arrayfire");
    CHECK(std::string(GpuExecutionModeName(
              GpuExecutionMode::NativeProvider)) == "native_provider");
    CHECK(std::string(GpuExecutionModeName(GpuExecutionMode::Cpu)) == "cpu");
    CHECK(std::string(GpuExecutionModeName(
              GpuExecutionMode::Unsupported)) == "unsupported");

    CHECK(std::string(GpuOperationFamilyName(
              GpuOperationFamily::Recurrent)) == "recurrent");
    CHECK(std::string(GpuOperationFamilyName(
              GpuOperationFamily::AttentionNormalization)) ==
          "attention_normalization");
    CHECK(std::string(GpuOperationFamilyName(
              GpuOperationFamily::GraphTensorOp)) == "graph_tensor_op");
}

TEST_CASE("Declared execution modes record the 2026-09-22 audit truth",
          "[gpu_execution][taxonomy]") {
    using namespace cyxwiz;

    // Changing any of these is an execution-strategy change and requires the
    // ticket's parity + benchmark evidence, not a drive-by edit.
    CHECK(DeclaredGpuExecutionMode(GpuOperationFamily::Recurrent) ==
          GpuExecutionMode::StagedArrayFire);
    CHECK(DeclaredGpuExecutionMode(
              GpuOperationFamily::AttentionNormalization) ==
          GpuExecutionMode::Cpu);
    CHECK(DeclaredGpuExecutionMode(GpuOperationFamily::Evaluation) ==
          GpuExecutionMode::Cpu);

    // Families served by the engine's ArrayFire-tensor and graph-runtime
    // placement builders must all declare direct ArrayFire — those builders
    // stamp direct_arrayfire on their entries, and this is the cross-check
    // that keeps the stamp consistent with the family table.
    for (const auto family : {GpuOperationFamily::DenseMatmul,
                              GpuOperationFamily::EmbeddingIndexing,
                              GpuOperationFamily::ConvolutionPooling,
                              GpuOperationFamily::Activation,
                              GpuOperationFamily::Loss,
                              GpuOperationFamily::Optimizer,
                              GpuOperationFamily::Reduction,
                              GpuOperationFamily::Transform,
                              GpuOperationFamily::GraphTensorOp}) {
        INFO("family: " << GpuOperationFamilyName(family));
        CHECK(DeclaredGpuExecutionMode(family) ==
              GpuExecutionMode::DirectArrayFire);
    }

    // No family declares a native provider yet — that graduation is gated
    // behind staged-plan benchmarks (ticket implementation-order step 8).
    for (const auto family : {GpuOperationFamily::Recurrent,
                              GpuOperationFamily::DenseMatmul,
                              GpuOperationFamily::EmbeddingIndexing,
                              GpuOperationFamily::AttentionNormalization,
                              GpuOperationFamily::ConvolutionPooling,
                              GpuOperationFamily::Activation,
                              GpuOperationFamily::Loss,
                              GpuOperationFamily::Optimizer,
                              GpuOperationFamily::Reduction,
                              GpuOperationFamily::Transform,
                              GpuOperationFamily::Evaluation,
                              GpuOperationFamily::GraphTensorOp}) {
        INFO("family: " << GpuOperationFamilyName(family));
        CHECK(DeclaredGpuExecutionMode(family) !=
              GpuExecutionMode::NativeProvider);
    }
}
