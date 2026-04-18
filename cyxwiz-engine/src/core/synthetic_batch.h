#pragma once

#include "graph_compiler.h"
#include <cyxwiz/tensor.h>
#include <cstdint>

namespace cyxwiz {

// A single synthetic batch (batch=1) used by DebugExecutor to stress-test
// graph wiring. Features + labels are generated deterministically from
// `seed` so repeated invocations produce identical numbers — debug runs
// should be reproducible.
struct SyntheticBatch {
    Tensor features;
    Tensor labels;
};

// Build a synthetic batch matching what a real DataLoader would hand to
// TrainingExecutor's forward path. Dispatch is driven by
// `config.preprocessing_domain` for features and `config.loss_type` for
// labels. Shapes per domain:
//   Tabular     -> [1, input_size] float32 in [0, 1)
//   Text        -> [1, seq_len] int64 token IDs in
//                  [0, num_embeddings)  (seq_len = config.input_size)
//   TimeSeries  -> [1, input_size] float32 in [0, 1)   (tabular fallback in v1)
//   Image       -> [1, input_size] float32 in [0, 1)   (tabular fallback in v1)
//   Audio       -> [1, input_size] float32 in [0, 1)   (tabular fallback in v1)
//
// Labels per loss_type:
//   CrossEntropy / NLL         -> int64 [1]             in [0, num_classes)
//   BCE / BCEWithLogits        -> float32 [1, num_classes] in {0, 1}
//   MSE / L1 / SmoothL1 / Huber -> float32 [1, output_size] in [0, 1)
SyntheticBatch MakeSyntheticBatch(const TrainingConfiguration& config,
                                  uint32_t seed = 1337);

} // namespace cyxwiz
