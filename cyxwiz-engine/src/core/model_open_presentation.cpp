#include "model_open_presentation.h"

#include "compiled_node_presentation.h"

#include <cstdio>

namespace cyxwiz::model_open {

namespace {

std::string FormatLine(const ProbeResult& probe) {
    std::string name;
    switch (probe.format) {
        case ModelFormat::CyxModel: name = "CyxWiz package"; break;
        case ModelFormat::Safetensors: name = "Weights only (safetensors)"; break;
        default: name = GetFormatName(probe.format); break;
    }
    // safetensors names itself as its version; say it once.
    if (!probe.format_version.empty() && probe.format_version != "safetensors")
        name += " (" + probe.format_version + ")";
    return name;
}

std::string Trained(const ProbeResult& probe) {
    if (!probe.epochs_trained) return {};
    const int epochs = *probe.epochs_trained;
    std::string text = FormatCompiledCount(epochs) + (epochs == 1 ? " epoch" : " epochs");
    char buffer[64];
    if (probe.final_accuracy) {
        std::snprintf(buffer, sizeof(buffer), ", accuracy %.2f%%", *probe.final_accuracy * 100.0);
        text += buffer;
    }
    if (probe.final_loss) {
        std::snprintf(buffer, sizeof(buffer), ", loss %.4f", static_cast<double>(*probe.final_loss));
        text += buffer;
    }
    return text;
}

std::string Contains(const ProbeResult& probe) {
    std::vector<std::string> parts;
    if (probe.has_graph) parts.emplace_back("graph");
    parts.emplace_back("weights");
    if (probe.has_training_history) parts.emplace_back("training history");
    if (probe.has_optimizer_state) parts.emplace_back("optimizer state");
    if (probe.has_tokenizer) parts.emplace_back("tokenizer");
    if (probe.has_vocabulary) parts.emplace_back("vocabulary");
    std::string text;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) text += i + 1 == parts.size() ? " and " : ", ";
        text += parts[i];
    }
    return text;
}

}  // namespace

std::string LayersLine(const ProbeResult& probe, size_t limit) {
    std::string text;
    const size_t shown = probe.layer_names.size() < limit ? probe.layer_names.size() : limit;
    for (size_t i = 0; i < shown; ++i) {
        const std::string& name = probe.layer_names[i];
        if (i > 0) text += ", ";
        text += name;
        const auto shape = probe.layer_shapes.find(name);
        if (shape == probe.layer_shapes.end()) continue;
        text += " [";
        for (size_t d = 0; d < shape->second.size(); ++d) {
            if (d > 0) text += ", ";
            text += std::to_string(shape->second[d]);
        }
        text += "]";
    }
    if (probe.layer_names.size() > shown)
        text += ", and " + FormatCompiledCount(static_cast<long long>(probe.layer_names.size() - shown)) + " more";
    return text;
}

View BuildView(const ProbeResult& probe) {
    View v;
    if (!probe.valid) {
        v.error = probe.error_message.empty() ? "The file is not a model this build can read." : probe.error_message;
        v.open_disabled_reason = "The file could not be read";
        return v;
    }
    v.readable = true;
    v.format_line = FormatLine(probe);
    v.rows.emplace_back("Name", probe.model_name.empty() ? "Not named" : probe.model_name);
    if (!probe.author.empty()) v.rows.emplace_back("Author", probe.author);
    if (!probe.description.empty()) v.rows.emplace_back("Description", probe.description);
    v.rows.emplace_back("Parameters", FormatCompiledCount(probe.num_parameters) + " in " +
                                          FormatCompiledCount(probe.num_layers) +
                                          (probe.num_layers == 1 ? " layer" : " layers"));
    const std::string trained = Trained(probe);
    if (!trained.empty()) v.rows.emplace_back("Trained", trained);
    v.rows.emplace_back("Contains", Contains(probe));
    v.rows.emplace_back("Size", FormatCompiledBytes(probe.file_size));
    v.layers = LayersLine(probe);
    v.can_open_graph = probe.has_graph;
    if (v.can_open_graph) {
        v.note = "Opens the graph in Studio, without its weights: train it, or load one of its training checkpoints "
                 "with Train > Test > Load Checkpoint for Testing.";
    } else {
        v.open_disabled_reason = "This file has no graph";
        v.note = "This file holds weights only; it has no graph to open.";
    }
    return v;
}

}  // namespace cyxwiz::model_open
