#include <catch2/catch_test_macros.hpp>
#include <string>
#include <set>

#include "../../cyxwiz-engine/src/core/pipeline_runtime_capabilities.h"
#include "../../cyxwiz-engine/src/core/data_input_formats.h"

TEST_CASE("Data Input runtime formats follow the production source catalog",
          "[pipeline][capabilities][data_input]") {
    const std::set<std::string> expected{
        "auto", "csv", "tsv", "json", "jsonl", "parquet", "feather", "arrow", "ipc", "zip_text"};
    int checked = 0;
    for (const auto& capability : cyxwiz::ResolvePipelineAllowedParameterValues("DataInput")) {
        if (std::string(capability.parameter_name) != "type" &&
            std::string(capability.parameter_name) != "file_type") continue;
        const std::set<std::string> actual(capability.allowed_values.begin(), capability.allowed_values.end());
        CHECK(actual == expected);
        for (const char* name : capability.allowed_values)
            CHECK(cyxwiz::data_input::IsExecutable(name));
        ++checked;
    }
    CHECK(checked == 2);
    CHECK_FALSE(cyxwiz::data_input::IsExecutable("hdf5"));
    CHECK_FALSE(cyxwiz::data_input::IsExecutable("h5"));
}

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
                                NodeType::GlobalMaxPool, NodeType::AdaptiveAvgPool,
                                NodeType::DepthwiseConv2D, NodeType::Conv3D}) {
        CAPTURE(static_cast<int>(type));
        const auto support = cyxwiz::ResolvePipelineTrainingBackendSupport(type);
        REQUIRE(support.mode == PipelineTrainingBackendSupportMode::Allowed);
        CHECK(support.compile_supported);
        CHECK(support.training_supported);
        REQUIRE(support.reason != nullptr);
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

TEST_CASE("Linear Attention is a supported trainable model layer (TOFIX140 Group C)",
          "[pipeline_runtime_capabilities][attention]") {
    const auto support = cyxwiz::ResolvePipelineTrainingBackendSupport(gui::NodeType::LinearAttention);
    REQUIRE(support.mode == cyxwiz::PipelineTrainingBackendSupportMode::Allowed);
    REQUIRE(support.compile_supported);
    REQUIRE(support.training_supported);
    REQUIRE_FALSE(cyxwiz::IsPipelineUnsupportedSequentialModelLayer(gui::NodeType::LinearAttention));
}

TEST_CASE("Cross Attention is a supported trainable model layer (TOFIX140 Group C)",
          "[pipeline_runtime_capabilities][attention]") {
    const auto support = cyxwiz::ResolvePipelineTrainingBackendSupport(gui::NodeType::CrossAttention);
    REQUIRE(support.mode == cyxwiz::PipelineTrainingBackendSupportMode::Allowed);
    REQUIRE(support.compile_supported);
    REQUIRE(support.training_supported);
    REQUIRE_FALSE(cyxwiz::IsPipelineUnsupportedSequentialModelLayer(gui::NodeType::CrossAttention));
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
