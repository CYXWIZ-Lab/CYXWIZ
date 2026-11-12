#pragma once

// Freezing a layer: its weights stay as they are while the rest of the model
// trains (fine-tuning a pretrained backbone). One place for the setting's
// names and for which node types hold weights, shared by the node settings,
// the model builder and the canvas actions.

#include "graph_model.h"

namespace cyxwiz::frozen {

// Node settings.
inline constexpr const char* kFrozen = "frozen";                      // "true" | "false"
inline constexpr const char* kStatistics = "frozen_statistics";       // BatchNorm: "keep" | "update"
inline constexpr const char* kStatisticsKeep = "keep";
inline constexpr const char* kStatisticsUpdate = "update";

// The layer nodes that hold weights a run trains, and so can be frozen.
// (Embedding has its own "Freeze Loaded Weights" for a table loaded from a file.)
inline bool LayerNodeHoldsWeights(gui::NodeType type) {
    switch (type) {
        case gui::NodeType::Dense:
        case gui::NodeType::Conv1D:
        case gui::NodeType::Conv2D:
        case gui::NodeType::Conv3D:
        case gui::NodeType::DepthwiseConv2D:
        case gui::NodeType::ConvTranspose2D:
        case gui::NodeType::BatchNorm:
        case gui::NodeType::LayerNorm:
        case gui::NodeType::GroupNorm:
        case gui::NodeType::InstanceNorm:
        case gui::NodeType::SqueezeExcitation:
        case gui::NodeType::LSTM:
        case gui::NodeType::GRU:
        case gui::NodeType::RNN:
        case gui::NodeType::MultiHeadAttention:
        case gui::NodeType::CrossAttention:
        case gui::NodeType::LinearAttention:
        case gui::NodeType::TransformerEncoder:
        case gui::NodeType::TransformerDecoder:
            return true;
        default:
            return false;
    }
}

// Only BatchNorm keeps running statistics, so only it has the second setting.
inline bool LayerNodeHasFrozenStatistics(gui::NodeType type) { return type == gui::NodeType::BatchNorm; }

}  // namespace cyxwiz::frozen
