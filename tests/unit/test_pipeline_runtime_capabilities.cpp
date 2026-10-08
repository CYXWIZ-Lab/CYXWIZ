#include <catch2/catch_test_macros.hpp>
#include <string>

#include "../../cyxwiz-engine/src/core/pipeline_runtime_capabilities.h"

TEST_CASE("Training capability registry allows the CNN stack (TOFIX140 A1)",
          "[pipeline][capabilities][spatial]") {
    using cyxwiz::PipelineTrainingBackendSupportMode;
    using gui::NodeType;
    // Compiled, built and trained on [H,W,C,N]; PyTorch parity in
    // cyxwiz-engine/tests/computation_truth (spatial_layers_pytorch_parity).
    for (const NodeType type : {NodeType::Conv2D, NodeType::MaxPool2D, NodeType::AvgPool2D,
                                NodeType::ConvTranspose2D, NodeType::Upsample,
                                NodeType::PixelShuffle, NodeType::GroupNorm,
                                NodeType::InstanceNorm, NodeType::GlobalAvgPool, NodeType::Conv1D,
                                NodeType::GlobalMaxPool, NodeType::AdaptiveAvgPool}) {
        CAPTURE(static_cast<int>(type));
        const auto support = cyxwiz::ResolvePipelineTrainingBackendSupport(type);
        REQUIRE(support.mode == PipelineTrainingBackendSupportMode::Allowed);
        CHECK(support.compile_supported);
        CHECK(support.training_supported);
        REQUIRE(support.reason != nullptr);
    }
    // The ones without an owner stay blocked.
    for (const NodeType type : {NodeType::Conv3D, NodeType::DepthwiseConv2D}) {
        CAPTURE(static_cast<int>(type));
        CHECK(cyxwiz::ResolvePipelineTrainingBackendSupport(type).mode ==
              PipelineTrainingBackendSupportMode::UnsupportedSequentialModelLayer);
    }
}

TEST_CASE("Training capability registry exposes tested causal LM building blocks",
          "[pipeline][capabilities][language_model]") {
    using cyxwiz::PipelineTrainingBackendSupportMode;
    using cyxwiz::ResolvePipelineTrainingBackendSupport;
    using gui::NodeType;

    const NodeType supported_nodes[] = {
        NodeType::Embedding,
        NodeType::PositionalEncoding,
        NodeType::TransformerEncoder,
        NodeType::TransformerDecoder,
        NodeType::MultiHeadAttention,
        NodeType::TimeDistributed
    };

    for (NodeType node_type : supported_nodes) {
        const auto support = ResolvePipelineTrainingBackendSupport(node_type);
        REQUIRE(support.mode == PipelineTrainingBackendSupportMode::Allowed);
        REQUIRE(support.compile_supported);
        REQUIRE(support.training_supported);
        REQUIRE(support.reason != nullptr);
    }

    REQUIRE(cyxwiz::IsPipelineSupportedTrainingRoleNode(
        NodeType::CrossEntropyLoss));
}

TEST_CASE("Training capability registry keeps unsupported attention variants blocked",
          "[pipeline][capabilities][language_model]") {
    using cyxwiz::PipelineTrainingBackendSupportMode;
    using cyxwiz::ResolvePipelineTrainingBackendSupport;
    using gui::NodeType;

    const NodeType unsupported_nodes[] = {
        NodeType::CrossAttention,
        NodeType::LinearAttention
    };

    for (NodeType node_type : unsupported_nodes) {
        const auto support = ResolvePipelineTrainingBackendSupport(node_type);
        REQUIRE(support.mode ==
                PipelineTrainingBackendSupportMode::UnsupportedSequentialModelLayer);
        REQUIRE_FALSE(support.compile_supported);
        REQUIRE_FALSE(support.training_supported);
        REQUIRE(support.reason != nullptr);
    }
}

TEST_CASE("Simple RNN is a supported trainable model layer after the Studio wiring",
          "[pipeline_runtime_capabilities][recurrent]") {
    const auto support = cyxwiz::ResolvePipelineTrainingBackendSupport(gui::NodeType::RNN);
    REQUIRE(support.mode == cyxwiz::PipelineTrainingBackendSupportMode::Allowed);
    REQUIRE(support.compile_supported);
    REQUIRE(support.training_supported);
    REQUIRE(cyxwiz::IsPipelineSupportedTrainingBackendNode(gui::NodeType::RNN));
    REQUIRE_FALSE(cyxwiz::IsPipelineUnsupportedSequentialModelLayer(gui::NodeType::RNN));
}
