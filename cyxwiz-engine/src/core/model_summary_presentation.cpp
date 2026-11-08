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

// "26·26·32" from a shape.
std::string Product(const std::vector<size_t>& shape) {
    std::string text;
    for (size_t i = 0; i < shape.size(); ++i) text += (i ? "·" : "") + std::to_string(shape[i]);
    return text.empty() ? "1" : text;
}

LayerMultiplyAdds Counted(long long value, std::string formula) {
    return {true, value, std::move(formula) + " = " + Count(value)};
}

LayerMultiplyAdds NotCounted(std::string why) { return {false, 0, std::move(why)}; }

// A sample of a sequence layer: [length, width]; a flat [width] is one step.
std::pair<long long, long long> LengthAndWidth(const std::vector<size_t>& shape) {
    if (shape.size() >= 2) return {static_cast<long long>(shape[shape.size() - 2]), static_cast<long long>(shape.back())};
    if (shape.size() == 1) return {1, static_cast<long long>(shape[0])};
    return {0, 0};
}

LayerMultiplyAdds Recurrent(const CompiledLayer& layer, long long gates, const char* gate_words) {
    const auto [length, width] = LengthAndWidth(layer.input_shape);
    if (length <= 0 || width <= 0) return NotCounted("its input has no sequence shape yet");
    const long long hidden = std::max<long long>(1, IntParam(layer.parameters, "hidden_size", 128));
    const long long stacked = std::max<long long>(1, IntParam(layer.parameters, "num_layers", 1));
    const long long directions = BoolParam(layer.parameters, "bidirectional") ? 2 : 1;
    // Each step and direction: gates x hidden x (input + hidden); a stacked
    // layer reads the one below (hidden x directions).
    const long long first = gates * hidden * (width + hidden);
    const long long upper = gates * hidden * (hidden * directions + hidden);
    const long long value = length * directions * (first + (stacked - 1) * upper);
    std::string formula = std::to_string(length) + " steps · " + (directions == 2 ? "2 directions · " : "") +
                          gate_words + " · " + std::to_string(hidden) + " · (" + std::to_string(width) + " + " +
                          std::to_string(hidden) + ")";
    if (stacked > 1)
        formula += " + " + std::to_string(stacked - 1) + " stacked layer" + (stacked > 2 ? "s" : "") + " on " +
                   std::to_string(hidden * directions);
    return Counted(value, formula);
}

// Self-attention projections and scores: Q and the output E x E, K and V
// E x (E · kv heads / heads), scores and weighted sum 2 · L² · E.
LayerMultiplyAdds Attention(const CompiledLayer& layer, long long feedforward_factor) {
    const auto [length, width] = LengthAndWidth(layer.input_shape);
    if (length <= 0 || width <= 0) return NotCounted("its input has no sequence shape yet");
    TransformerConfiguration configuration;
    configuration.model_width = static_cast<size_t>(width);
    if (const auto error = ResolveTransformerConfiguration(layer.type, layer.parameters, configuration))
        return NotCounted(*error);
    const long long heads = std::max<long long>(1, static_cast<long long>(configuration.num_heads));
    const long long kv_heads = configuration.num_kv_heads > 0 ? static_cast<long long>(configuration.num_kv_heads) : heads;
    const long long kv_width = width * kv_heads / heads;
    const long long projections = length * (2 * width * width + 2 * width * kv_width);
    const long long scores = 2 * length * length * width;
    long long value = projections + scores;
    std::string formula = std::to_string(length) + " · (2·" + std::to_string(width) + "² + 2·" +
                          std::to_string(width) + "·" + std::to_string(kv_width) + ") projections + 2·" +
                          std::to_string(length) + "²·" + std::to_string(width) + " attention";
    if (feedforward_factor > 0) {
        const bool gated = configuration.ffn_type == "gated";
        const long long ff = static_cast<long long>(configuration.feedforward_width);
        const long long factor = gated ? 3 : 2;
        value += factor * length * width * ff;
        formula += " + " + std::to_string(factor) + "·" + std::to_string(length) + "·" + std::to_string(width) + "·" +
                   std::to_string(ff) + (gated ? " gated feed-forward" : " feed-forward");
    }
    return Counted(value, formula);
}

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

LayerMultiplyAdds CountLayerMultiplyAdds(const CompiledLayer& layer) {
    const auto& in = layer.input_shape;
    const auto& out = layer.output_shape;
    if (IsZeroCostLayer(layer.type)) return {true, 0, ZeroCostWords(layer.type)};
    try {
        switch (layer.type) {
            case gui::NodeType::Dense:
            case gui::NodeType::TimeDistributed: {
                // Each output value is a dot product over the input features
                // of its position: a sequence keeps its positions.
                const long long outputs = Elements(out);
                const long long units = out.empty() ? 0 : static_cast<long long>(out.back());
                if (outputs <= 0 || units <= 0 || in.empty()) return NotCounted("its shapes are not known yet");
                const long long positions = outputs / units;
                const long long features = Elements(in) / std::max<long long>(1, positions);
                std::string formula = (positions > 1 ? std::to_string(positions) + " positions · " : "") +
                                      std::to_string(features) + " · " + std::to_string(units);
                return Counted(outputs * features, formula);
            }
            case gui::NodeType::Conv2D:
            case gui::NodeType::DepthwiseConv2D: {
                if (in.size() != 3 || out.size() != 3) return NotCounted("its input has no image shape yet");
                const auto g = spatial::ResolveGeometry(layer.type, layer.parameters, in[2]);
                const long long per_output =
                    static_cast<long long>(g.kernel) * g.kernel *
                    (layer.type == gui::NodeType::DepthwiseConv2D ? 1 : static_cast<long long>(in[2]));
                const std::string window = std::to_string(g.kernel) + "·" + std::to_string(g.kernel) +
                                           (layer.type == gui::NodeType::DepthwiseConv2D ? "" : "·" + std::to_string(in[2]));
                return Counted(Elements(out) * per_output, Product(out) + " · " + window);
            }
            case gui::NodeType::Conv1D: {
                if (in.size() != 2 || out.size() != 2) return NotCounted("its input has no sequence shape yet");
                const auto g = spatial::ResolveGeometry(layer.type, layer.parameters, in[1]);
                return Counted(Elements(out) * g.kernel * static_cast<long long>(in[1]),
                               Product(out) + " · " + std::to_string(g.kernel) + "·" + std::to_string(in[1]));
            }
            case gui::NodeType::Conv3D: {
                if (in.size() != 4 || out.size() != 4) return NotCounted("its input has no volume shape yet");
                const auto g = spatial::ResolveGeometry(layer.type, layer.parameters, in[3]);
                const long long k = g.kernel;
                return Counted(Elements(out) * k * k * k * static_cast<long long>(in[3]),
                               Product(out) + " · " + std::to_string(k) + "·" + std::to_string(k) + "·" +
                                   std::to_string(k) + "·" + std::to_string(in[3]));
            }
            case gui::NodeType::ConvTranspose2D: {
                // Every input value is spread over a k x k window of each output channel.
                if (in.size() != 3 || out.size() != 3) return NotCounted("its input has no image shape yet");
                const auto g = spatial::ResolveGeometry(layer.type, layer.parameters, in[2]);
                const long long per_input = static_cast<long long>(g.kernel) * g.kernel * static_cast<long long>(out[2]);
                return Counted(Elements(in) * per_input, Product(in) + " · " + std::to_string(g.kernel) + "·" +
                                                             std::to_string(g.kernel) + "·" + std::to_string(out[2]));
            }
            case gui::NodeType::LSTM:
                return Recurrent(layer, 4, "4 gates");
            case gui::NodeType::GRU:
                return Recurrent(layer, 3, "3 gates");
            case gui::NodeType::RNN:
                return Recurrent(layer, 1, "1 gate");
            case gui::NodeType::MultiHeadAttention:
                return Attention(layer, 0);
            case gui::NodeType::TransformerEncoder:
            case gui::NodeType::TransformerDecoder:
                return Attention(layer, 1);
            case gui::NodeType::LinearAttention: {
                // Projections 4·L·E², then per head phi(K)ᵀV and phi(Q)·(KᵀV): 2·L·E·d.
                const auto [length, width] = LengthAndWidth(in);
                if (length <= 0 || width <= 0) return NotCounted("its input has no sequence shape yet");
                const long long heads = std::max<long long>(1, IntParam(layer.parameters, "num_heads", 8));
                const long long head_width = width / heads;
                const long long value = 4 * length * width * width + 2 * length * width * head_width;
                return Counted(value, "4·" + std::to_string(length) + "·" + std::to_string(width) + "² projections + 2·" +
                                          std::to_string(length) + "·" + std::to_string(width) + "·" +
                                          std::to_string(head_width) + " kernel attention");
            }
            case gui::NodeType::CrossAttention:
                return NotCounted("Depends on the Key length, which the compile does not record.");
            case gui::NodeType::PluginCustom:
                return NotCounted("An extension node: its cost is its own code's.");
            default:
                break;
        }
    } catch (const std::exception& error) {
        return NotCounted(error.what());
    }
    return {true, 0, ZeroCostWords(layer.type)};
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
