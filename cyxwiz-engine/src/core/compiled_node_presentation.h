#pragma once

// The "AS COMPILED" card in the Properties panel (TOFIX123): what the graph
// compiler built for the selected node - its role, per-sample and batch
// shapes, memory, learnable parameters and its own compiler issues. Pure
// data in and out: no ImGui, no registry, no backend calls, so every screen
// that shows compiled facts renders the same words.

#include "graph_compiler.h"

#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

enum class LiveCompileState { NotCompiled, Compiling, Compiled, Failed };

struct CompiledNodeInputs {
    LiveCompileState state = LiveCompileState::NotCompiled;
    // The latest compile of the graph on the canvas; nullptr before the first.
    const TrainingConfiguration* config = nullptr;
    bool data_loaded = false;
    // Learnable parameters per compiled layer index (from building the model
    // the compile describes); empty until counted.
    std::map<size_t, long long> layer_parameters;
    bool parameters_counted = false;   // the count finished: a missing layer has 0
    bool parameters_counting = false;
    // Display name of the node's type ("LSTM", "Concatenate", ...).
    std::string type_label;
};

enum class CompiledStatusKind { Ok, Failed, Pending, None };

struct CompiledNodeIssue {
    bool error = false;
    std::string message;
};

struct CompiledNodeCard {
    std::string status;  // "Compiled" / "Compile failed" / "Compiling..." / "Not compiled yet"
    CompiledStatusKind kind = CompiledStatusKind::None;
    std::string status_note;
    std::string role;
    int select_node_id = -1;  // e.g. the Concatenate a fused Embedding lives on
    std::string select_label;
    std::vector<CompiledNodeIssue> issues;  // this node's own compiler issues

    bool has_shapes = false;
    size_t batch = 0;
    std::string input_sample, input_batch, output_sample, output_batch;
    std::string input_note;
    std::string output_memory;
    std::string parameters;         // "3,002,048", "0", "counting..."
    std::string parameter_memory;
    std::string parameter_note;
    std::string no_shapes_text;     // why there are no shapes, when there are none

    std::vector<std::pair<std::string, std::string>> details;
};

CompiledNodeCard BuildCompiledNodeCard(const std::vector<gui::MLNode>& nodes,
                                       const std::vector<gui::NodeLink>& links,
                                       int node_id,
                                       const CompiledNodeInputs& inputs);

// "3,002,048"
std::string FormatCount(long long value);
// "1.36 MB", "773.3 KB", "0 bytes" (1024-based, float32 elements in bytes).
std::string FormatBytes(unsigned long long bytes);
// "[96, 116]"
std::string FormatShape(const std::vector<size_t>& shape);

// Learnable parameters per compiled layer index from a built model's
// parameter names ("layer<i>.<rest>" -> counts summed per i).
std::map<size_t, long long> CountParametersPerLayer(
    const std::vector<std::pair<std::string, long long>>& named_counts);

}  // namespace cyxwiz
