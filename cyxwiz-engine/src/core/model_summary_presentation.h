#pragma once

// The Model window's Summary tab (TOFIX129 D11): every layer of the graph as
// the compiler built it - output shape, learnable parameters (counted from
// the built model), multiply-adds a sample and activation memory - with the
// totals and a few words of advice. The one counter of the Engine: it reads
// the same compile as the Properties panel's AS COMPILED card. Pure data in
// and out: no ImGui, no registry, no backend calls.

#include "compiled_node_presentation.h"
#include "graph_compiler.h"

#include <cyxwiz/model_cost.h>

#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

struct ModelSummaryInputs {
    LiveCompileState state = LiveCompileState::NotCompiled;
    const TrainingConfiguration* config = nullptr;  // nullptr before the first compile
    std::map<size_t, long long> layer_parameters;   // per compiled layer index
    bool parameters_counted = false;
    bool parameters_counting = false;
    // Display name of each node's type ("Conv2D", "LSTM"), by node id.
    std::map<int, std::string> type_labels;
    // Batch for the activation total; 0 = the graph's batch size.
    size_t batch = 0;
};

// Multiply-adds of one compiled layer for one sample: the node's settings
// as the SDK counter's numbers (cyxwiz/model_cost.h, as PyTorch's
// torch.utils.flop_counter counts, halved); element-wise layers 0 with
// words. `formula` says why not when !counted.
LayerCost CountLayerMultiplyAdds(const CompiledLayer& layer);

struct ModelSummaryRow {
    int node_id = -1;
    std::string name;          // the node's name on the canvas
    std::string kind;          // its type ("Conv2D")
    std::string output;        // "[26, 26, 32]"
    std::vector<size_t> output_shape;
    bool parameters_known = false;
    long long parameters = 0;
    LayerCost multiply_adds;
    double share = 0.0;        // of the model's multiply-adds, 0..1
    unsigned long long activation_bytes = 0;  // its output, one sample, float32
    std::string how;           // how its numbers follow from the shapes
};

struct ModelSummaryCard {
    std::string title;
    std::string text;
};

struct ModelSummary {
    bool has_model = false;
    std::string empty_text;    // why there is nothing to show
    std::string status;        // "Compiled", "Compile failed", ...
    CompiledStatusKind kind = CompiledStatusKind::None;
    std::string status_note;

    std::vector<ModelSummaryRow> rows;
    size_t batch = 1;
    long long parameters = 0;
    bool parameters_known = false;
    long long multiply_adds = 0;
    size_t uncounted_layers = 0;  // layers whose multiply-adds are not known
    unsigned long long activation_bytes_per_sample = 0;

    // The four tiles: value and the line under it.
    std::string parameters_text, parameters_note;
    std::string multiply_adds_text, multiply_adds_note;
    std::string weights_text, weights_note;
    std::string activations_text, activations_note;

    std::vector<ModelSummaryCard> advice;
};

ModelSummary BuildModelSummary(const ModelSummaryInputs& inputs);

// "2.63 M", "194.7 K", "980"
std::string FormatModelCount(long long value);

// The summary as aligned text (Copy as text) and as CSV (Export CSV).
std::string ModelSummaryText(const ModelSummary& summary);
std::string ModelSummaryCsv(const ModelSummary& summary);

}  // namespace cyxwiz
