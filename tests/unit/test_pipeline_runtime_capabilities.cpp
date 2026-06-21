#include <catch2/catch_test_macros.hpp>

#include "../../cyxwiz-engine/src/core/pipeline_runtime_capabilities.h"

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

TEST_CASE("Training capability registry keeps standalone attention blocked",
          "[pipeline][capabilities][language_model]") {
    using cyxwiz::PipelineTrainingBackendSupportMode;
    using cyxwiz::ResolvePipelineTrainingBackendSupport;
    using gui::NodeType;

    const NodeType unsupported_nodes[] = {
        NodeType::MultiHeadAttention,
        NodeType::SelfAttention,
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
