#include "model_summary_presentation.h"

#include "spatial_layer_shapes.h"
#include "transformer_configuration_policy.h"

#include <algorithm>
#include <cstdio>
#include <exception>

namespace cyxwiz {

namespace {

constexpr unsigned long long kBytesPerElement = 4;  // float32

long long Elements(const std::vector<size_t>& shape) {
    if (shape.empty()) return 0;
    long long count = 1;
    for (size_t d : shape) count *= static_cast<long long>(d);
    return count;
}

std::string Param(const std::map<std::string, std::string>& params, const char* key) {
    const auto it = params.find(key);
    return it == params.end() ? std::string{} : it->second;
}

long long IntParam(const std::map<std::string, std::string>& params, const char* key, long long fallback) {
    const std::string text = Param(params, key);
    if (text.empty()) return fallback;
    try {
        return std::stoll(text);
    } catch (...) {
        return fallback;
    }
}

bool BoolParam(const std::map<std::string, std::string>& params, const char* key) {
    const std::string text = Param(params, key);
    return text == "true" || text == "1" || text == "True";
}

std::string Count(long long value) { return FormatCompiledCount(value); }

LayerCost NotCounted(std::string why) { return {false, 0, std::move(why)}; }

bool IsZeroCostLayer(gui::NodeType type) {
    switch (type) {
        case gui::NodeType::Dense:
        case gui::NodeType::TimeDistributed:
        case gui::NodeType::Conv1D:
        case gui::NodeType::Conv2D:
        case gui::NodeType::Conv3D:
        case gui::NodeType::DepthwiseConv2D:
        case gui::NodeType::ConvTranspose2D:
        case gui::NodeType::LSTM:
        case gui::NodeType::GRU:
        case gui::NodeType::RNN:
        case gui::NodeType::MultiHeadAttention:
        case gui::NodeType::TransformerEncoder:
        case gui::NodeType::TransformerDecoder:
        case gui::NodeType::LinearAttention:
        case gui::NodeType::CrossAttention:
        case gui::NodeType::PluginCustom:
            return false;
        default:
            return true;
    }
}

std::string ZeroCostWords(gui::NodeType type) {
    switch (type) {
        case gui::NodeType::Embedding:
            return "A table lookup: no multiply-adds.";
        case gui::NodeType::MaxPool2D:
        case gui::NodeType::AvgPool2D:
        case gui::NodeType::GlobalAvgPool:
        case gui::NodeType::GlobalMaxPool:
        case gui::NodeType::AdaptiveAvgPool:
            return "Pooling: no multiply-adds counted.";
        case gui::NodeType::Flatten:
        case gui::NodeType::Reshape:
        case gui::NodeType::View:
        case gui::NodeType::Permute:
        case gui::NodeType::Squeeze:
        case gui::NodeType::Unsqueeze:
            return "Rearranges the values: no multiply-adds.";
        case gui::NodeType::Dropout:
            return "Training only: drops units at random; nothing at test time.";
        case gui::NodeType::BatchNorm:
        case gui::NodeType::LayerNorm:
        case gui::NodeType::GroupNorm:
        case gui::NodeType::InstanceNorm:
            return "Normalizes each value: element-wise, no multiply-adds counted.";
        default:
            return "Element-wise: no multiply-adds counted.";
    }
}

}  // namespace

LayerCost CountLayerMultiplyAdds(const CompiledLayer& layer) {
    if (IsZeroCostLayer(layer.type)) return {true, 0, ZeroCostWords(layer.type)};
    // The node's settings as the SDK counter's numbers; the arithmetic is
    // CountMultiplyAdds (shared with pycyxwiz.model.summary).
    LayerCostInput cost;
    cost.input = layer.input_shape;
    cost.output = layer.output_shape;
    const auto kernel = [&](size_t channels_axis) {
        const auto& shape = layer.input_shape;
        if (shape.size() <= channels_axis) return 0LL;
        return static_cast<long long>(spatial::ResolveGeometry(layer.type, layer.parameters, shape[channels_axis]).kernel);
    };
    try {
        switch (layer.type) {
            case gui::NodeType::Dense:
            case gui::NodeType::TimeDistributed:
                cost.kind = LayerCostKind::Dense;
                break;
            case gui::NodeType::Conv2D:
                cost.kind = LayerCostKind::Conv2D;
                cost.kernel = kernel(2);
                break;
            case gui::NodeType::DepthwiseConv2D:
                cost.kind = LayerCostKind::DepthwiseConv2D;
                cost.kernel = kernel(2);
                break;
            case gui::NodeType::Conv1D:
                cost.kind = LayerCostKind::Conv1D;
                cost.kernel = kernel(1);
                break;
            case gui::NodeType::Conv3D:
                cost.kind = LayerCostKind::Conv3D;
                cost.kernel = kernel(3);
                break;
            case gui::NodeType::ConvTranspose2D:
                cost.kind = LayerCostKind::ConvTranspose2D;
                cost.kernel = kernel(2);
                break;
            case gui::NodeType::LSTM:
            case gui::NodeType::GRU:
            case gui::NodeType::RNN:
                cost.kind = layer.type == gui::NodeType::LSTM  ? LayerCostKind::LSTM
                            : layer.type == gui::NodeType::GRU ? LayerCostKind::GRU
                                                               : LayerCostKind::RNN;
                cost.hidden = std::max<long long>(1, IntParam(layer.parameters, "hidden_size", 128));
                cost.stacked = std::max<long long>(1, IntParam(layer.parameters, "num_layers", 1));
                cost.bidirectional = BoolParam(layer.parameters, "bidirectional");
                break;
            case gui::NodeType::MultiHeadAttention:
            case gui::NodeType::TransformerEncoder:
            case gui::NodeType::TransformerDecoder: {
                const long long width = layer.input_shape.empty() ? 0 : static_cast<long long>(layer.input_shape.back());
                if (width <= 0) return NotCounted("its input has no sequence shape yet");
                TransformerConfiguration configuration;
                configuration.model_width = static_cast<size_t>(width);
                if (const auto error = ResolveTransformerConfiguration(layer.type, layer.parameters, configuration))
                    return NotCounted(*error);
                cost.kind = layer.type == gui::NodeType::MultiHeadAttention ? LayerCostKind::Attention
                                                                            : LayerCostKind::TransformerBlock;
                cost.heads = static_cast<long long>(configuration.num_heads);
                cost.kv_heads = static_cast<long long>(configuration.num_kv_heads);
                cost.feedforward = static_cast<long long>(configuration.feedforward_width);
                cost.gated = configuration.ffn_type == "gated";
                break;
            }
            case gui::NodeType::LinearAttention:
                cost.kind = LayerCostKind::LinearAttention;
                cost.heads = std::max<long long>(1, IntParam(layer.parameters, "num_heads", 8));
                break;
            case gui::NodeType::CrossAttention:
                return NotCounted("Depends on the Key length, which the compile does not record.");
            case gui::NodeType::PluginCustom:
                return NotCounted("An extension node: its cost is its own code's.");
            default:
                return {true, 0, ZeroCostWords(layer.type)};
        }
    } catch (const std::exception& error) {
        return NotCounted(error.what());
    }
    return CountMultiplyAdds(cost);
}

std::string FormatModelCount(long long value) {
    char buffer[32];
    const double v = static_cast<double>(value);
    if (value >= 1000000000LL) std::snprintf(buffer, sizeof(buffer), "%.2f G", v / 1e9);
    else if (value >= 1000000LL) std::snprintf(buffer, sizeof(buffer), "%.2f M", v / 1e6);
    else if (value >= 10000LL) std::snprintf(buffer, sizeof(buffer), "%.1f K", v / 1e3);
    else return FormatCompiledCount(value);
    return buffer;
}

ModelSummary BuildModelSummary(const ModelSummaryInputs& inputs) {
    ModelSummary summary;
    const TrainingConfiguration* config = inputs.config;
    switch (inputs.state) {
        case LiveCompileState::NotCompiled:
            summary.status = "Not compiled yet";
            break;
        case LiveCompileState::Compiling:
            summary.status = "Compiling...";
            summary.kind = CompiledStatusKind::Pending;
            summary.status_note = "Showing the previous compile until this one finishes.";
            break;
        case LiveCompileState::Compiled:
            summary.status = "Compiled";
            summary.kind = CompiledStatusKind::Ok;
            break;
        case LiveCompileState::Failed:
            summary.status = "Compile failed";
            summary.kind = CompiledStatusKind::Failed;
            break;
    }
    if (!config) {
        summary.empty_text = "Open a training graph: the summary reads its compile, as Properties does.";
        return summary;
    }
    if (config->layers.empty()) {
        summary.empty_text = "This graph has no model layers yet: add layers between the data and the loss.";
        return summary;
    }
    if (!config->is_valid) {
        summary.status = "Compile failed";
        summary.kind = CompiledStatusKind::Failed;
        summary.status_note = "Layers the compiler rejected have no shapes; the Compile popup (F7) lists why.";
    }
    summary.has_model = true;
    summary.batch = inputs.batch > 0 ? inputs.batch
                                     : (config->batch_size > 0 ? static_cast<size_t>(config->batch_size) : 1);

    // The data the model reads.
    if (!config->input_shape.empty()) {
        ModelSummaryRow row;
        row.node_id = config->data_source_node_id;
        row.name = "Data";
        row.kind = "Input";
        row.output_shape = config->input_shape;
        row.output = FormatCompiledShape(config->input_shape);
        row.parameters_known = true;
        row.multiply_adds = {true, 0, ""};
        row.activation_bytes = static_cast<unsigned long long>(Elements(config->input_shape)) * kBytesPerElement;
        row.how = "One sample as the model reads it: " + row.output + ", " +
                  Count(Elements(config->input_shape)) + " values.";
        summary.rows.push_back(std::move(row));
    }

    summary.parameters_known = inputs.parameters_counted;
    // Layers and the graph ops between them (Add, Concatenate, Split), in
    // the compiler's order.
    std::map<int, size_t> plan_order;
    for (size_t i = 0; i < config->graph_plan.nodes.size(); ++i) plan_order[config->graph_plan.nodes[i].node_id] = i;
    struct Entry {
        size_t order = 0;
        long long layer = -1;  // index in config->layers, or -1 for a graph op
        int op_node_id = -1;
    };
    std::vector<Entry> entries;
    size_t last_order = 0;
    for (size_t i = 0; i < config->layers.size(); ++i) {
        const auto found = plan_order.find(config->layers[i].node_id);
        if (found != plan_order.end()) last_order = found->second;
        entries.push_back({last_order, static_cast<long long>(i), -1});
    }
    for (const auto& [node_id, shapes] : config->graph_op_output_shapes) {
        const auto found = plan_order.find(node_id);
        if (found != plan_order.end()) entries.push_back({found->second, -1, node_id});
    }
    std::stable_sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) { return a.order < b.order; });

    for (const Entry& entry : entries) {
        if (entry.layer < 0) {
            const CompiledGraphNode* op = nullptr;
            for (const auto& node : config->graph_plan.nodes)
                if (node.node_id == entry.op_node_id) op = &node;
            const auto& shapes = config->graph_op_output_shapes.at(entry.op_node_id);
            ModelSummaryRow row;
            row.node_id = entry.op_node_id;
            row.name = op ? op->name : "Graph op";
            const auto label = inputs.type_labels.find(entry.op_node_id);
            row.kind = label != inputs.type_labels.end() ? label->second : row.name;
            row.output_shape = shapes.empty() ? std::vector<size_t>{} : shapes.front();
            for (size_t s = 0; s < shapes.size(); ++s)
                row.output += (s ? " + " : "") + FormatCompiledShape(shapes[s]);
            row.parameters_known = true;
            row.multiply_adds = {true, 0, "Joins or splits branches: no multiply-adds counted."};
            for (const auto& shape : shapes)
                row.activation_bytes += static_cast<unsigned long long>(Elements(shape)) * kBytesPerElement;
            row.how = row.output + " out. " + row.multiply_adds.formula;
            summary.rows.push_back(std::move(row));
            continue;
        }
        const size_t i = static_cast<size_t>(entry.layer);
        const CompiledLayer& layer = config->layers[i];
        ModelSummaryRow row;
        row.node_id = layer.node_id;
        row.name = layer.name;
        const auto label = inputs.type_labels.find(layer.node_id);
        row.kind = label != inputs.type_labels.end() ? label->second : layer.name;
        row.output_shape = layer.output_shape;
        row.output = layer.output_shape.empty() ? "-" : FormatCompiledShape(layer.output_shape);
        const auto counted = inputs.layer_parameters.find(i);
        row.parameters_known = counted != inputs.layer_parameters.end() || inputs.parameters_counted;
        row.parameters = counted == inputs.layer_parameters.end() ? 0 : counted->second;
        row.multiply_adds = CountLayerMultiplyAdds(layer);
        row.activation_bytes = static_cast<unsigned long long>(Elements(layer.output_shape)) * kBytesPerElement;

        std::string how;
        if (!layer.input_shape.empty() && !layer.output_shape.empty())
            how = FormatCompiledShape(layer.input_shape) + " in, " + row.output + " out. ";
        if (row.parameters_known && row.parameters > 0) how += "Parameters " + Count(row.parameters) + ". ";
        if (row.multiply_adds.counted && row.multiply_adds.value > 0)
            how += "Multiply-adds " + row.multiply_adds.formula + ".";
        else
            how += row.multiply_adds.formula;
        row.how = how;

        summary.parameters += row.parameters;
        if (row.multiply_adds.counted) summary.multiply_adds += row.multiply_adds.value;
        else ++summary.uncounted_layers;
        summary.rows.push_back(std::move(row));
    }
    for (auto& row : summary.rows) {
        summary.activation_bytes_per_sample += row.activation_bytes;
        if (summary.multiply_adds > 0 && row.multiply_adds.counted)
            row.share = static_cast<double>(row.multiply_adds.value) / static_cast<double>(summary.multiply_adds);
    }

    // Tiles.
    if (summary.parameters_known) {
        summary.parameters_text = Count(summary.parameters);
        summary.parameters_note = "learnable, counted from the built model";
        summary.weights_text = FormatCompiledBytes(static_cast<unsigned long long>(summary.parameters) * kBytesPerElement);
        summary.weights_note = "float32; " +
                               FormatCompiledBytes(static_cast<unsigned long long>(summary.parameters) * 2ULL) +
                               " as float16";
    } else {
        summary.parameters_text = inputs.parameters_counting ? "counting..." : "-";
        summary.parameters_note = inputs.parameters_counting ? "building the model to count them"
                                                             : "counted once the model builds";
        summary.weights_text = "-";
        summary.weights_note = summary.parameters_note;
    }
    summary.multiply_adds_text = FormatModelCount(summary.multiply_adds);
    char flops[64];
    std::snprintf(flops, sizeof(flops), "%s FLOPs (2 per multiply-add)",
                  FormatModelCount(2 * summary.multiply_adds).c_str());
    summary.multiply_adds_note = flops;
    if (summary.uncounted_layers > 0)
        summary.multiply_adds_note = "without " + std::to_string(summary.uncounted_layers) + " layer" +
                                     (summary.uncounted_layers == 1 ? "" : "s") + " not counted";
    summary.activations_text =
        FormatCompiledBytes(summary.activation_bytes_per_sample * static_cast<unsigned long long>(summary.batch));
    summary.activations_note = FormatCompiledBytes(summary.activation_bytes_per_sample) +
                               " a sample, kept for the backward pass (batch " + std::to_string(summary.batch) + ")";

    // Advice: where the work and the weights are.
    const ModelSummaryRow* busiest = nullptr;
    const ModelSummaryRow* heaviest = nullptr;
    for (const auto& row : summary.rows) {
        if (row.multiply_adds.counted && (!busiest || row.multiply_adds.value > busiest->multiply_adds.value))
            busiest = &row;
        if (row.parameters_known && (!heaviest || row.parameters > heaviest->parameters)) heaviest = &row;
    }
    const auto percent = [](double fraction) { return std::to_string(static_cast<int>(fraction * 100.0 + 0.5)) + " %"; };
    if (busiest && busiest->share >= 0.5) {
        ModelSummaryCard card;
        card.title = busiest->name + " does " + percent(busiest->share) + " of the work";
        card.text = "Fewer filters or units there, a stride of 2, or a pooling before it cuts the multiply-adds most.";
        summary.advice.push_back(std::move(card));
    }
    if (summary.parameters_known && heaviest && summary.parameters > 0) {
        const double share = static_cast<double>(heaviest->parameters) / static_cast<double>(summary.parameters);
        if (share >= 0.5) {
            ModelSummaryCard card;
            card.title = heaviest->name + " holds " + percent(share) + " of the parameters";
            card.text = heaviest->kind == "Dense" ? "A smaller layer, or pooling before Flatten (fewer inputs), shrinks the model most."
                              : "A smaller layer there shrinks the model most.";
            summary.advice.push_back(std::move(card));
        }
    }
    return summary;
}

std::string ModelSummaryText(const ModelSummary& summary) {
    std::vector<std::vector<std::string>> table = {{"#", "Layer", "Type", "Output", "Parameters", "Multiply-adds"}};
    for (size_t i = 0; i < summary.rows.size(); ++i) {
        const auto& row = summary.rows[i];
        table.push_back({std::to_string(i + 1), row.name, row.kind, row.output,
                         row.parameters_known ? Count(row.parameters) : "-",
                         row.multiply_adds.counted ? Count(row.multiply_adds.value) : "not counted"});
    }
    std::vector<size_t> widths(table[0].size(), 0);
    for (const auto& line : table)
        for (size_t c = 0; c < line.size(); ++c) widths[c] = std::max(widths[c], line[c].size());
    std::string text;
    for (const auto& line : table) {
        for (size_t c = 0; c < line.size(); ++c) {
            const bool number = c == 0 || c >= 4;
            const std::string pad(widths[c] - line[c].size(), ' ');
            text += number ? pad + line[c] : line[c] + pad;
            text += c + 1 < line.size() ? "  " : "\n";
        }
    }
    text += "Parameters: " + summary.parameters_text + " (" + summary.parameters_note + ")\n";
    text += "Multiply-adds a sample: " + Count(summary.multiply_adds) + " (" + summary.multiply_adds_note + ")\n";
    text += "Weights: " + summary.weights_text + " (" + summary.weights_note + ")\n";
    text += "Activations: " + summary.activations_text + " (" + summary.activations_note + ")\n";
    return text;
}

std::string ModelSummaryCsv(const ModelSummary& summary) {
    const auto quote = [](const std::string& value) {
        std::string out = "\"";
        for (char c : value) out += c == '"' ? std::string("\"\"") : std::string(1, c);
        return out + "\"";
    };
    std::string csv = "index,layer,type,output_shape,parameters,multiply_adds,activation_bytes\n";
    for (size_t i = 0; i < summary.rows.size(); ++i) {
        const auto& row = summary.rows[i];
        csv += std::to_string(i + 1) + "," + quote(row.name) + "," + quote(row.kind) + "," + quote(row.output) + "," +
               (row.parameters_known ? std::to_string(row.parameters) : "") + "," +
               (row.multiply_adds.counted ? std::to_string(row.multiply_adds.value) : "") + "," +
               std::to_string(row.activation_bytes) + "\n";
    }
    return csv;
}

}  // namespace cyxwiz
