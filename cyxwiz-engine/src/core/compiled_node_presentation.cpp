#include "compiled_node_presentation.h"

#include "sequence_fusion_presentation.h"

#include <algorithm>
#include <cstdio>

namespace cyxwiz {

namespace {

constexpr unsigned long long kBytesPerElement = 4;  // float32 activations and weights

const gui::MLNode* NodeById(const std::vector<gui::MLNode>& nodes, int id) {
    for (const auto& node : nodes) {
        if (node.id == id) return &node;
    }
    return nullptr;
}

std::string Param(const std::map<std::string, std::string>& params, const char* key) {
    const auto it = params.find(key);
    return it == params.end() ? std::string{} : it->second;
}

long long ParseCount(const std::string& value) {
    try {
        return value.empty() ? 0 : std::stoll(value);
    } catch (...) {
        return 0;
    }
}

unsigned long long Elements(const std::vector<size_t>& shape) {
    if (shape.empty()) return 0;
    unsigned long long count = 1;
    for (size_t d : shape) count *= d;
    return count;
}

std::vector<size_t> WithBatch(size_t batch, std::vector<size_t> shape) {
    shape.insert(shape.begin(), batch);
    return shape;
}

bool IsFusionLayer(const CompiledLayer& layer) {
    return layer.type == gui::NodeType::Concatenate &&
           Param(layer.parameters, "sequence_feature_fusion") == "true";
}

// Fill the shape and parameter rows of the card.
void SetShapes(CompiledNodeCard& card, size_t batch, const std::vector<size_t>& in,
               const std::vector<size_t>& out) {
    card.has_shapes = true;
    card.batch = batch;
    card.input_sample = FormatShape(in);
    card.output_sample = FormatShape(out);
    card.input_batch = FormatShape(WithBatch(batch, in));
    card.output_batch = FormatShape(WithBatch(batch, out));
    card.output_memory = FormatBytes(Elements(WithBatch(batch, out)) * kBytesPerElement);
}

void SetParameters(CompiledNodeCard& card, const CompiledNodeInputs& inputs, bool known, long long count) {
    if (!known) {
        card.parameters = inputs.parameters_counting ? "counting..." : "not counted";
        card.parameter_memory = "-";
        return;
    }
    card.parameters = FormatCount(count);
    card.parameter_memory = FormatBytes(static_cast<unsigned long long>(count) * kBytesPerElement);
}

}  // namespace

std::string FormatCount(long long value) {
    const bool negative = value < 0;
    std::string digits = std::to_string(negative ? -value : value);
    std::string out;
    const size_t lead = digits.size() % 3;
    for (size_t i = 0; i < digits.size(); ++i) {
        if (i != 0 && i >= lead && (i - lead) % 3 == 0) out += ',';
        out += digits[i];
    }
    return negative ? "-" + out : out;
}

std::string FormatBytes(unsigned long long bytes) {
    char buffer[32];
    if (bytes < 1024ULL) {
        std::snprintf(buffer, sizeof(buffer), "%llu bytes", bytes);
    } else if (bytes < 1024ULL * 1024ULL) {
        std::snprintf(buffer, sizeof(buffer), "%.1f KB", static_cast<double>(bytes) / 1024.0);
    } else if (bytes < 1024ULL * 1024ULL * 1024ULL) {
        std::snprintf(buffer, sizeof(buffer), "%.2f MB", static_cast<double>(bytes) / (1024.0 * 1024.0));
    } else {
        std::snprintf(buffer, sizeof(buffer), "%.2f GB",
                      static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
    }
    return buffer;
}

std::string FormatShape(const std::vector<size_t>& shape) {
    std::string text = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) text += ", ";
        text += std::to_string(shape[i]);
    }
    return text + "]";
}

std::map<size_t, long long> CountParametersPerLayer(
    const std::vector<std::pair<std::string, long long>>& named_counts) {
    std::map<size_t, long long> per_layer;
    for (const auto& [name, count] : named_counts) {
        if (name.rfind("layer", 0) != 0) continue;
        size_t end = 5;
        while (end < name.size() && name[end] >= '0' && name[end] <= '9') ++end;
        if (end == 5) continue;
        per_layer[static_cast<size_t>(std::stoull(name.substr(5, end - 5)))] += count;
    }
    return per_layer;
}

CompiledNodeCard BuildCompiledNodeCard(const std::vector<gui::MLNode>& nodes,
                                       const std::vector<gui::NodeLink>& links,
                                       int node_id,
                                       const CompiledNodeInputs& inputs) {
    CompiledNodeCard card;
    const gui::MLNode* node = NodeById(nodes, node_id);
    switch (inputs.state) {
        case LiveCompileState::NotCompiled:
            card.status = "Not compiled yet";
            card.kind = CompiledStatusKind::None;
            card.status_note = "The graph compiles in the background after an edit.";
            break;
        case LiveCompileState::Compiling:
            card.status = "Compiling...";
            card.kind = CompiledStatusKind::Pending;
            card.status_note = "Showing the previous compile until this one finishes.";
            break;
        case LiveCompileState::Compiled:
            card.status = "Compiled";
            card.kind = CompiledStatusKind::Ok;
            break;
        case LiveCompileState::Failed:
            card.status = "Compile failed";
            card.kind = CompiledStatusKind::Failed;
            break;
    }
    const TrainingConfiguration* config = inputs.config;
    if (!node || !config) {
        if (!node) card.no_shapes_text = "This node no longer exists.";
        else card.no_shapes_text = "No compile yet: shapes and parameters appear once the graph compiles.";
        return card;
    }

    if (inputs.state == LiveCompileState::Compiled || inputs.state == LiveCompileState::Failed) {
        card.status_note = inputs.data_loaded
            ? "Recompiled after your last edit. Data loaded."
            : "Data not loaded: shapes come from the saved data contract.";
    }

    size_t errors = 0;
    for (const auto& issue : config->issues) {
        if (issue.node_id != node_id || issue.level == IssueLevel::Info) continue;
        const bool error = issue.level == IssueLevel::Error;
        errors += error ? 1 : 0;
        card.issues.push_back({error, issue.message});
    }
    // A failed compile keeps its status on every node; a node without its own
    // error still shows the shapes that compile gave it.
    if (!config->is_valid) {
        card.status = "Compile failed";
        card.kind = CompiledStatusKind::Failed;
        card.status_note = errors > 0
            ? std::to_string(errors) + (errors == 1 ? " error" : " errors") +
                  " on this node. Shapes appear once the graph compiles."
            : "The graph has errors on other nodes; this node's shapes are from the same compile.";
    }

    const size_t batch = config->batch_size > 0 ? static_cast<size_t>(config->batch_size) : 1;
    const size_t layer_count = config->layers.size();
    const auto layer_it = std::find_if(config->layers.begin(), config->layers.end(),
                                       [&](const CompiledLayer& layer) { return layer.node_id == node_id; });

    // Word / POS Embedding folded into the fused layer.
    const CompiledLayer* fused = nullptr;
    size_t fused_index = 0;
    for (size_t i = 0; i < layer_count; ++i) {
        if (IsFusionLayer(config->layers[i])) {
            fused = &config->layers[i];
            fused_index = i;
        }
    }
    std::string fused_role;  // "word" / "POS" for an Embedding inside the fused layer
    if (fused && node->type == gui::NodeType::Embedding) {
        for (const auto& verdict : AnalyzeSequenceFusion(nodes, links, {}, true)) {
            if (verdict.concat_node_id != fused->node_id) continue;
            if (verdict.word_node_id == node_id) fused_role = "word";
            if (verdict.pos_node_id == node_id) fused_role = "POS";
        }
    }

    if (layer_it != config->layers.end()) {
        const size_t index = static_cast<size_t>(layer_it - config->layers.begin());
        const CompiledLayer& layer = *layer_it;
        const std::string position = "Layer " + std::to_string(index + 1) + " of " + std::to_string(layer_count);
        if (IsFusionLayer(layer)) {
            std::string parts;
            for (const auto& verdict : AnalyzeSequenceFusion(nodes, links, {}, true)) {
                if (verdict.concat_node_id != node_id) continue;
                const auto* word = NodeById(nodes, verdict.word_node_id);
                const auto* pos = NodeById(nodes, verdict.pos_node_id);
                if (word && pos) parts = " (with " + word->name + " and " + pos->name + ")";
            }
            card.role = position + ": word + POS fusion" + parts;
            std::vector<size_t> packed = layer.input_shape;
            packed.push_back(2);
            SetShapes(card, batch, packed, layer.output_shape);
            card.input_note = "Input: word and POS ids packed per token.";
            const long long word = ParseCount(Param(layer.parameters, "word_num_embeddings")) *
                                   ParseCount(Param(layer.parameters, "word_embedding_dim"));
            const long long pos = ParseCount(Param(layer.parameters, "pos_num_embeddings")) *
                                  ParseCount(Param(layer.parameters, "pos_embedding_dim"));
            card.parameter_note = "Word " + FormatCount(ParseCount(Param(layer.parameters, "word_num_embeddings"))) +
                                  " x " + Param(layer.parameters, "word_embedding_dim") + " (" + FormatCount(word) +
                                  ") + POS " + FormatCount(ParseCount(Param(layer.parameters, "pos_num_embeddings"))) +
                                  " x " + Param(layer.parameters, "pos_embedding_dim") + " (" + FormatCount(pos) +
                                  "). Sizes follow the data at launch.";
            card.details.emplace_back("Compiled layer", "SequenceFeatureFusion");
        } else {
            card.role = position + ": " + (inputs.type_label.empty() ? layer.name : inputs.type_label);
            SetShapes(card, batch, layer.input_shape, layer.output_shape);
            card.details.emplace_back("Compiled layer", inputs.type_label.empty() ? layer.name : inputs.type_label);
        }
        const auto counted = inputs.layer_parameters.find(index);
        SetParameters(card, inputs, counted != inputs.layer_parameters.end() || inputs.parameters_counted,
                      counted == inputs.layer_parameters.end() ? 0 : counted->second);
        card.details.emplace_back("Layer index", std::to_string(index));
        for (const auto& [key, value] : layer.parameters) {
            if (key == "sequence_feature_fusion" || value.empty()) continue;
            card.details.emplace_back(key, value);
        }
    } else if (!fused_role.empty()) {
        const auto* concat = NodeById(nodes, fused->node_id);
        const std::string prefix = fused_role == "word" ? "word_" : "pos_";
        card.role = "Part of layer " + std::to_string(fused_index + 1) +
                    " (word + POS fusion); the layer is built on " + (concat ? concat->name : "the Concatenate");
        card.select_node_id = fused->node_id;
        card.select_label = "Select " + (concat ? concat->name : std::string("the Concatenate"));
        const long long vocab = ParseCount(Param(fused->parameters, (prefix + "num_embeddings").c_str()));
        const long long dim = ParseCount(Param(fused->parameters, (prefix + "embedding_dim").c_str()));
        std::vector<size_t> out = fused->input_shape;
        out.push_back(static_cast<size_t>(std::max<long long>(dim, 0)));
        SetShapes(card, batch, fused->input_shape, out);
        card.input_note = fused_role == "word"
            ? "Input: word ids (Concatenate Input 1)."
            : "Input: POS ids (Concatenate Input 2).";
        SetParameters(card, inputs, true, vocab * dim);
        card.parameter_note = "Vocabulary " + FormatCount(vocab) + " x " + std::to_string(dim) +
                              " (set from the data at launch).";
        card.details.emplace_back("Compiled into", "layer " + std::to_string(fused_index) + " SequenceFeatureFusion");
        card.details.emplace_back("Role", fused_role == "word" ? "word ids (Input 1)" : "POS ids (Input 2)");
    } else if (node_id == config->data_source_node_id) {
        card.role = "Data source for the model";
        if (!config->input_shape.empty()) {
            card.has_shapes = true;
            card.batch = batch;
            card.output_sample = FormatShape(config->input_shape);
            card.output_batch = FormatShape(WithBatch(batch, config->input_shape));
            card.output_memory = FormatBytes(Elements(WithBatch(batch, config->input_shape)) * kBytesPerElement);
            card.input_sample = card.input_batch = "-";
            card.input_note = "Output: one model input per sample.";
            SetParameters(card, inputs, true, 0);
        }
    } else if (node_id == config->loss_node_id) {
        card.role = "Loss: " + (inputs.type_label.empty() ? node->name : inputs.type_label);
    } else if (node_id == config->optimizer_node_id) {
        card.role = "Optimizer: " + (inputs.type_label.empty() ? node->name : inputs.type_label);
    } else {
        const bool on_path = std::any_of(config->graph_plan.nodes.begin(), config->graph_plan.nodes.end(),
                                         [&](const CompiledGraphNode& n) { return n.node_id == node_id; });
        card.role = on_path ? "On the training path; not a model layer" : "Not on the training path";
    }

    if (errors > 0) {
        card.has_shapes = false;
        card.no_shapes_text =
            "No shapes or parameters: the compiler rejected this node. The same reason shows in the Compile "
            "popup and on its links.";
    } else if (!card.has_shapes && card.no_shapes_text.empty()) {
        card.no_shapes_text = "This node has no tensor shape of its own in the compiled model.";
    }
    if (inputs.state == LiveCompileState::Compiling && !card.has_shapes && card.no_shapes_text.empty()) {
        card.no_shapes_text = "Compiling...";
    }
    return card;
}

}  // namespace cyxwiz
