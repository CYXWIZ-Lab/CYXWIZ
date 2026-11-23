#pragma once

#include "graph_model.h"

namespace cyxwiz {

// The built-in loss nodes: one list for the compiler, the code export and the
// debugger (the export's own copy left the metric-learning losses out, so such
// a graph exported a cross-entropy criterion).
inline bool IsBuiltInLossNodeType(gui::NodeType type) {
    switch (type) {
        case gui::NodeType::MSELoss:
        case gui::NodeType::CrossEntropyLoss:
        case gui::NodeType::FocalLoss:
        case gui::NodeType::BCELoss:
        case gui::NodeType::BCEWithLogits:
        case gui::NodeType::L1Loss:
        case gui::NodeType::SmoothL1Loss:
        case gui::NodeType::HuberLoss:
        case gui::NodeType::NLLLoss:
        case gui::NodeType::SoftDiceLoss:
        case gui::NodeType::TverskyLoss:
        case gui::NodeType::JaccardLoss:
        case gui::NodeType::TripletLoss:
        case gui::NodeType::ContrastiveLoss:
        case gui::NodeType::CosineEmbeddingLoss:
            return true;
        default:
            return false;
    }
}

}  // namespace cyxwiz
