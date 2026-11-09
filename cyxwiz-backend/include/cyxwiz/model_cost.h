// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
#pragma once

// Multiply-adds of one sample through a layer (TOFIX129 D11), counted the
// way PyTorch's torch.utils.flop_counter counts FLOPs, halved (one
// multiply-add = two FLOPs): matrix products only; attention as its plain
// matrix products; recurrent layers by their weight matrices times the
// sequence length; pooling, activations, normalization and reshapes count
// nothing. The Engine's Model > Summary (from a compiled graph) and
// pycyxwiz.model.summary (from a Sequential's modules) both fill a
// LayerCostInput and count here, so the two agree; checked against PyTorch
// (the Engine's model_summary truth test).

#include "cyxwiz/api_export.h"

#include <cstddef>
#include <string>
#include <vector>

namespace cyxwiz {

enum class LayerCostKind {
    None,             // no multiply-adds (pooling, activations, norms, reshapes)
    Dense,            // also a time-distributed dense: each position separately
    Conv1D,
    Conv2D,
    DepthwiseConv2D,
    Conv3D,
    ConvTranspose2D,
    LSTM,
    GRU,
    RNN,
    Attention,        // multi-head self-attention
    TransformerBlock, // self-attention + feed-forward
    LinearAttention
};

// One layer, for one sample. Shapes in CyxWiz's layout: [features],
// [length, width], [H, W, C], [D, H, W, C].
struct CYXWIZ_API LayerCostInput {
    LayerCostKind kind = LayerCostKind::None;
    std::vector<size_t> input, output;
    long long kernel = 0;            // convolutions: the (square / cubic) kernel side
    long long hidden = 0;            // recurrent: hidden size
    long long stacked = 1;           // recurrent: layers stacked in this one
    bool bidirectional = false;      // recurrent
    long long heads = 1;             // attention
    long long kv_heads = 0;          // attention: key / value heads (0 = heads)
    long long feedforward = 0;       // transformer block: feed-forward width
    bool gated = false;              // transformer block: a gated (3-matrix) feed-forward
};

struct CYXWIZ_API LayerCost {
    bool counted = false;  // false: why not is in `formula`
    long long value = 0;   // multiply-adds of one sample
    std::string formula;   // "26·26·32 · 3·3·1 = 194,688"
};

CYXWIZ_API LayerCost CountMultiplyAdds(const LayerCostInput& layer);

}  // namespace cyxwiz
