#include <catch2/catch_test_macros.hpp>
#include <string>

#include "../../cyxwiz-engine/src/core/pipeline_runtime_capabilities.h"

TEST_CASE("Upsample mode-specific backend evidence does not promote Studio training support",
          "[pipeline][capabilities][upsample]") {
    const auto support = cyxwiz::ResolvePipelineTrainingBackendSupport(gui::NodeType::Upsample);
    REQUIRE(support.mode == cyxwiz::PipelineTrainingBackendSupportMode::UnsupportedSequentialModelLayer);
    REQUIRE_FALSE(support.compile_supported);
    REQUIRE_FALSE(support.training_supported);
    REQUIRE(support.reason != nullptr);
    const std::string reason(support.reason);
    CHECK(reason.find("ArrayFire-first nearest") != std::string::npos);
    CHECK(reason.find("nearest/bilinear") != std::string::npos);
    CHECK(reason.find("exact ModelBuilder construction") != std::string::npos);
    CHECK(reason.find("spatial batch-layout") != std::string::npos);
    CHECK(reason.find("observed native fallback") != std::string::npos);
    CHECK(reason.find("ModelBuilder") != std::string::npos);
    CHECK(reason.find("Studio training workflow") != std::string::npos);
}

TEST_CASE("PixelShuffle backend evidence does not promote Studio training support",
          "[pipeline][capabilities][pixelshuffle]") {
    const auto support = cyxwiz::ResolvePipelineTrainingBackendSupport(gui::NodeType::PixelShuffle);
    REQUIRE(support.mode == cyxwiz::PipelineTrainingBackendSupportMode::UnsupportedSequentialModelLayer);
    REQUIRE_FALSE(support.compile_supported);
    REQUIRE_FALSE(support.training_supported);
    REQUIRE(support.reason != nullptr);
    const std::string reason(support.reason);
    CHECK(reason.find("ArrayFire-first") != std::string::npos);
    CHECK(reason.find("exact ModelBuilder construction") != std::string::npos);
    CHECK(reason.find("spatial batch-layout") != std::string::npos);
    CHECK(reason.find("ModelBuilder") != std::string::npos);
    CHECK(reason.find("Studio training workflow") != std::string::npos);
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

TEST_CASE("Simple RNN is a supported trainable model layer after the Studio wiring",
          "[pipeline_runtime_capabilities][recurrent]") {
    const auto support = cyxwiz::ResolvePipelineTrainingBackendSupport(gui::NodeType::RNN);
    REQUIRE(support.mode == cyxwiz::PipelineTrainingBackendSupportMode::Allowed);
    REQUIRE(support.compile_supported);
    REQUIRE(support.training_supported);
    REQUIRE(cyxwiz::IsPipelineSupportedTrainingBackendNode(gui::NodeType::RNN));
    REQUIRE_FALSE(cyxwiz::IsPipelineUnsupportedSequentialModelLayer(gui::NodeType::RNN));
}
