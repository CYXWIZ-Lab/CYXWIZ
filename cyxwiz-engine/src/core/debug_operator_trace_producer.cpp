#include "debug_operator_trace_producer.h"

#include "debug_graph_trace_executor.h"
#include "debug_operator_trace_adapter.h"
#include "error_codes.h"
#include "node_executors/text_tokenizer_operator.h"

#include <algorithm>
#include <chrono>
#include <iterator>
#include <map>
#include <queue>
#include <unordered_set>
#include <vector>

namespace cyxwiz {

namespace {

bool IsDataInputNode(const gui::MLNode& node) {
    return node.type == gui::NodeType::DataInput ||
           node.type == gui::NodeType::DatasetInput;
}

std::string DatasetNameForNode(const gui::MLNode& node) {
    auto dataset_name = node.parameters.find("dataset_name");
    if (dataset_name != node.parameters.end() && !dataset_name->second.empty()) {
        return dataset_name->second;
    }

    auto legacy_dataset = node.parameters.find("dataset");
    if (legacy_dataset != node.parameters.end() && !legacy_dataset->second.empty()) {
        return legacy_dataset->second;
    }

    return {};
}

bool IsFoldedTextConfigNode(gui::NodeType type) {
    return type == gui::NodeType::TextVocabulary ||
           type == gui::NodeType::TextPadding;
}

bool IsSupportedTraceOperator(gui::NodeType type) {
    return type == gui::NodeType::TextTokenizer;
}

std::string NodeTypeName(gui::NodeType type) {
    switch (type) {
        case gui::NodeType::Dense:
            return "Dense";
        case gui::NodeType::DataInput:
            return "DataInput";
        case gui::NodeType::DatasetInput:
            return "DatasetInput";
        case gui::NodeType::TextTokenizer:
            return "TextTokenizer";
        case gui::NodeType::TextVocabulary:
            return "TextVocabulary";
        case gui::NodeType::TextPadding:
            return "TextPadding";
        default:
            return std::to_string(static_cast<int>(type));
    }
}

const gui::MLNode* FindNodeById(
    int id,
    const std::vector<gui::MLNode>& nodes) {
    for (const auto& node : nodes) {
        if (node.id == id) {
            return &node;
        }
    }
    return nullptr;
}

const gui::MLNode* FindDataInput(
    const std::vector<gui::MLNode>& nodes,
    const std::string& source_dataset_name) {
    if (!source_dataset_name.empty()) {
        for (const auto& node : nodes) {
            if (IsDataInputNode(node) &&
                DatasetNameForNode(node) == source_dataset_name) {
                return &node;
            }
        }
        return nullptr;
    }

    for (const auto& node : nodes) {
        if (IsDataInputNode(node)) {
            return &node;
        }
    }
    return nullptr;
}

std::vector<const gui::MLNode*> DownstreamNodes(
    int from_node_id,
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links) {
    std::vector<const gui::MLNode*> downstream;
    for (const auto& link : links) {
        if (link.from_node != from_node_id) {
            continue;
        }
        if (const auto* node = FindNodeById(link.to_node, nodes)) {
            downstream.push_back(node);
        }
    }
    return downstream;
}

bool HasReachableSupportedTraceOperator(
    int node_id,
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    std::unordered_set<int>& visiting) {
    if (!visiting.insert(node_id).second) {
        return false;
    }

    const gui::MLNode* node = FindNodeById(node_id, nodes);
    if (!node) {
        return false;
    }
    if (IsSupportedTraceOperator(node->type)) {
        return true;
    }
    if (!IsDataInputNode(*node) && !IsFoldedTextConfigNode(node->type)) {
        return false;
    }

    for (const auto& link : links) {
        if (link.from_node != node_id) {
            continue;
        }
        if (HasReachableSupportedTraceOperator(
                link.to_node, nodes, links, visiting)) {
            return true;
        }
    }

    return false;
}

bool HasReachableTraceCycle(
    int node_id,
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    std::unordered_set<int>& visiting,
    std::unordered_set<int>& visited,
    int& cycle_node_id) {
    if (visiting.find(node_id) != visiting.end()) {
        cycle_node_id = node_id;
        return true;
    }
    if (visited.find(node_id) != visited.end()) {
        return false;
    }

    visiting.insert(node_id);
    for (const auto& link : links) {
        if (link.from_node != node_id) {
            continue;
        }

        std::unordered_set<int> supported_visit;
        if (!HasReachableSupportedTraceOperator(
                link.to_node, nodes, links, supported_visit)) {
            continue;
        }

        if (HasReachableTraceCycle(
                link.to_node, nodes, links, visiting, visited,
                cycle_node_id)) {
            return true;
        }
    }

    visiting.erase(node_id);
    visited.insert(node_id);
    return false;
}

bool ValidateLinearTraceOperatorPath(
    const gui::MLNode& data_input,
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    std::string& error) {
    std::unordered_set<int> cycle_visiting;
    std::unordered_set<int> cycle_checked;
    int cycle_node_id = -1;
    if (HasReachableTraceCycle(
            data_input.id, nodes, links, cycle_visiting, cycle_checked,
            cycle_node_id)) {
        const gui::MLNode* node = FindNodeById(cycle_node_id, nodes);
        const std::string node_name =
            node ? node->name : std::to_string(cycle_node_id);
        error = "DebugOperatorTraceProducer: cyclic graph path involving node '" +
                node_name +
                "' is not supported by operator-backed debugger tracing";
        return false;
    }

    std::queue<int> queue;
    std::unordered_set<int> visited;
    queue.push(data_input.id);
    visited.insert(data_input.id);

    while (!queue.empty()) {
        const int node_id = queue.front();
        queue.pop();

        std::vector<int> trace_relevant_children;
        for (const auto& link : links) {
            if (link.from_node != node_id) {
                continue;
            }

            std::unordered_set<int> visiting;
            if (HasReachableSupportedTraceOperator(
                    link.to_node, nodes, links, visiting)) {
                trace_relevant_children.push_back(link.to_node);
            }

            if (visited.insert(link.to_node).second) {
                queue.push(link.to_node);
            }
        }

        if (trace_relevant_children.size() > 1) {
            const gui::MLNode* node = FindNodeById(node_id, nodes);
            const std::string node_name =
                node ? node->name : std::to_string(node_id);
            error = "DebugOperatorTraceProducer: branched operator trace paths from node '" +
                    node_name +
                    "' are not supported by the first TextTokenizer trace slice";
            return false;
        }
    }

    return true;
}

std::vector<size_t> TableShape(const std::shared_ptr<arrow::Table>& table) {
    if (!table) {
        return {};
    }
    return {
        static_cast<size_t>(table->num_rows()),
        static_cast<size_t>(table->num_columns())
    };
}

DebugTraceRecord BuildGraphWarningTrace(
    const std::string& run_id,
    const std::shared_ptr<arrow::Table>& input,
    const std::string& message,
    const std::string& error_code = errors::Runtime::ExecutionFailed) {
    auto trace = DebugNodeTraceContract::Make(
        run_id,
        -1,
        "DebugOperatorTraceProducer",
        "OperatorTrace",
        "OperatorTrace",
        DebugTraceRole::Warning,
        TableShape(input),
        {},
        "arrow::Table",
        "CPU",
        "warning");
    trace.payload["trace_producer"] = "DebugOperatorTraceProducer";
    trace.payload["operator_backed"] = false;
    trace.payload["message"] = message;
    trace.payload["error_code"] = error_code;
    DebugNodeTraceContract::AttachDiagnosticContext(
        trace,
        "graph_walk",
        "DebugOperatorTraceProducer",
        "cyxwiz-engine/src/core/debug_operator_trace_producer.cpp",
        "cyxwiz::DebugOperatorTraceProducer::TracePreprocessingGraph");
    DebugNodeTraceContract::AddWarning(
        trace,
        message,
        error_code);
    return trace;
}

struct DebugTableWindow {
    std::shared_ptr<arrow::Table> table;
    size_t source_rows = 0;
    size_t source_columns = 0;
    size_t selected_sample_index = 0;
    size_t row_offset = 0;
    size_t row_count = 0;
    size_t row_limit = 0;
    bool bounded = false;
    bool selected_sample_clamped = false;
    bool selected_sample_available = false;
};

DebugTableWindow BuildDebugTableWindow(
    const std::shared_ptr<arrow::Table>& source_table,
    size_t selected_sample_index,
    size_t max_debug_rows) {
    DebugTableWindow window;
    window.table = source_table;
    window.selected_sample_index = selected_sample_index;
    window.row_limit = max_debug_rows;

    if (!source_table) {
        return window;
    }

    window.source_rows = static_cast<size_t>(std::max<int64_t>(
        0, source_table->num_rows()));
    window.source_columns = static_cast<size_t>(std::max<int>(
        0, source_table->num_columns()));
    if (window.source_rows == 0) {
        return window;
    }

    window.selected_sample_available = selected_sample_index < window.source_rows;
    window.selected_sample_clamped = !window.selected_sample_available;
    window.row_offset = std::min(selected_sample_index, window.source_rows - 1);
    const size_t remaining_rows = window.source_rows - window.row_offset;
    window.row_count = max_debug_rows == 0
        ? remaining_rows
        : std::min(max_debug_rows, remaining_rows);
    window.bounded = window.row_offset != 0 || window.row_count != window.source_rows;
    if (window.bounded) {
        window.table = source_table->Slice(
            static_cast<int64_t>(window.row_offset),
            static_cast<int64_t>(window.row_count));
    } else {
        window.row_count = window.source_rows;
    }
    return window;
}

void AnnotateTraceWindow(DebugTraceRecord& trace,
                         const DebugTableWindow& window,
                         const std::string& source_dataset_name,
                         const gui::MLNode* source_node = nullptr) {
    if (!source_dataset_name.empty()) {
        trace.payload["source_dataset_name"] = source_dataset_name;
    }
    if (source_node) {
        trace.payload["source_node_id"] = source_node->id;
        trace.payload["source_node_name"] = source_node->name;
        trace.payload["source_node_type"] = NodeTypeName(source_node->type);
        const std::string node_dataset_name = DatasetNameForNode(*source_node);
        if (!node_dataset_name.empty()) {
            trace.payload["source_node_dataset_name"] = node_dataset_name;
        }
    }
    trace.payload["source_rows"] = window.source_rows;
    trace.payload["source_columns"] = window.source_columns;
    trace.payload["selected_sample_index"] = window.selected_sample_index;
    trace.payload["debug_row_offset"] = window.row_offset;
    trace.payload["debug_row_count"] = window.row_count;
    trace.payload["debug_row_limit"] = window.row_limit;
    trace.payload["bounded_debug_table"] = window.bounded;
    trace.payload["selected_sample_clamped"] = window.selected_sample_clamped;
    trace.payload["selected_sample_available"] = window.selected_sample_available;
}

void AnnotateTraceWindow(std::vector<DebugTraceRecord>& traces,
                         const DebugTableWindow& window,
                         const std::string& source_dataset_name,
                         const gui::MLNode* source_node = nullptr) {
    for (auto& trace : traces) {
        AnnotateTraceWindow(trace, window, source_dataset_name, source_node);
    }
}

std::vector<std::string> FoldTextConfigParams(
    const gui::MLNode& config_node,
    std::map<std::string, std::string>& params) {
    std::vector<std::string> contributed_keys;
    const auto mark_contributed = [&contributed_keys](const std::string& key) {
        if (std::find(contributed_keys.begin(), contributed_keys.end(), key) ==
            contributed_keys.end()) {
            contributed_keys.push_back(key);
        }
    };
    const auto set_param = [&params, &mark_contributed](
        const std::string& key,
        const std::string& value) {
        params[key] = value;
        mark_contributed(key);
    };

    if (config_node.type == gui::NodeType::TextVocabulary) {
        auto min_freq = config_node.parameters.find("min_freq");
        if (min_freq != config_node.parameters.end()) {
            set_param("min_word_freq", min_freq->second);
        }
        auto min_word_freq = config_node.parameters.find("min_word_freq");
        if (min_word_freq != config_node.parameters.end()) {
            set_param("min_word_freq", min_word_freq->second);
        }
        auto max_vocab = config_node.parameters.find("max_vocab_size");
        if (max_vocab != config_node.parameters.end()) {
            set_param("max_vocab_size", max_vocab->second);
        }
        auto vocab_file = config_node.parameters.find("vocab_file");
        if (vocab_file != config_node.parameters.end()) {
            params["vocab_file"] = vocab_file->second;
            params["vocab_build_if_missing"] = "true";
            mark_contributed("vocab_file_configured");
            mark_contributed("vocab_build_if_missing");
        }
    } else if (config_node.type == gui::NodeType::TextPadding) {
        auto max_length = config_node.parameters.find("max_length");
        if (max_length != config_node.parameters.end()) {
            set_param("max_length", max_length->second);
        }
        auto pad_value = config_node.parameters.find("pad_value");
        if (pad_value != config_node.parameters.end()) {
            set_param("pad_value", pad_value->second);
        }
    }

    return contributed_keys;
}

nlohmann::json BuildTextTokenizerConfigPayload(
    const std::map<std::string, std::string>& params) {
    nlohmann::json config = nlohmann::json::object();
    const auto add_if_present = [&params, &config](const std::string& key) {
        auto it = params.find(key);
        if (it != params.end()) {
            config[key] = it->second;
        }
    };

    add_if_present("text_col");
    add_if_present("label_col");
    add_if_present("tokenizer_type");
    add_if_present("max_length");
    add_if_present("lowercase");
    add_if_present("min_word_freq");
    add_if_present("max_vocab_size");
    add_if_present("pad_value");
    add_if_present("vocab_build_if_missing");
    config["vocab_file_configured"] =
        params.find("vocab_file") != params.end() &&
        !params.at("vocab_file").empty();
    return config;
}

std::map<std::string, std::string> BuildTextTokenizerParams(
    const gui::MLNode& tokenizer_node,
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    nlohmann::json* folded_config_nodes = nullptr) {
    auto params = tokenizer_node.parameters;
    if (folded_config_nodes) {
        *folded_config_nodes = nlohmann::json::array();
    }

    std::queue<int> queue;
    std::unordered_set<int> visited;
    queue.push(tokenizer_node.id);
    visited.insert(tokenizer_node.id);

    while (!queue.empty()) {
        const int node_id = queue.front();
        queue.pop();

        for (const auto& link : links) {
            if (link.from_node != node_id ||
                !visited.insert(link.to_node).second) {
                continue;
            }

            const gui::MLNode* child = FindNodeById(link.to_node, nodes);
            if (!child || !IsFoldedTextConfigNode(child->type)) {
                continue;
            }

            const auto contributed_keys = FoldTextConfigParams(*child, params);
            if (folded_config_nodes && !contributed_keys.empty()) {
                folded_config_nodes->push_back({
                    {"node_id", child->id},
                    {"node_name", child->name},
                    {"node_type", NodeTypeName(child->type)},
                    {"contributed_keys", contributed_keys},
                });
            }
            queue.push(child->id);
        }
    }

    return params;
}

} // namespace

std::vector<DebugTraceRecord>
DebugOperatorTraceProducer::TracePreprocessingGraph(
    const std::string& run_id,
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    const std::shared_ptr<arrow::Table>& source_table,
    const std::string& source_dataset_name,
    size_t selected_sample_index,
    size_t max_debug_rows) const {
    std::vector<DebugTraceRecord> traces;
    const DebugTableWindow debug_window = BuildDebugTableWindow(
        source_table, selected_sample_index, max_debug_rows);

    const gui::MLNode* data_input = FindDataInput(nodes, source_dataset_name);
    if (!data_input) {
        std::string message = "Operator-backed debugger tracing skipped because the graph has no DataInput or DatasetInput source node.";
        if (!source_dataset_name.empty()) {
            message = "Operator-backed debugger tracing skipped because dataset '" +
                source_dataset_name +
                "' does not match any DataInput/DatasetInput node in the graph.";
        }
        auto warning = BuildGraphWarningTrace(
            run_id, debug_window.table, message, errors::Runtime::InputDatasetMissing);
        AnnotateTraceWindow(warning, debug_window, source_dataset_name);
        return {std::move(warning)};
    }

    if (!debug_window.table) {
        std::string message = "Operator-backed debugger tracing skipped because no Arrow source table was available.";
        if (!source_dataset_name.empty()) {
            message = "Operator-backed debugger tracing skipped because Arrow dataset '" +
                source_dataset_name + "' did not provide a source table.";
        }
        auto warning = BuildGraphWarningTrace(
            run_id, debug_window.table, message, errors::Runtime::InputDatasetMissing);
        warning.payload["diagnostic_phase"] = "data_source";
        AnnotateTraceWindow(warning, debug_window, source_dataset_name, data_input);
        return {std::move(warning)};
    }

    std::unordered_set<int> supported_visit;
    const bool has_supported_trace_operator =
        HasReachableSupportedTraceOperator(
            data_input->id, nodes, links, supported_visit);
    std::string graph_shape_error;
    if (has_supported_trace_operator &&
        !ValidateLinearTraceOperatorPath(
            *data_input, nodes, links, graph_shape_error)) {
        auto warning = BuildWarningTrace(
            run_id, *data_input, debug_window.table, graph_shape_error);
        warning.payload["diagnostic_phase"] = "graph_walk";
        AnnotateTraceWindow(warning, debug_window, source_dataset_name, data_input);
        return {std::move(warning)};
    }

    std::map<int, std::shared_ptr<arrow::Table>> table_by_node;
    table_by_node[data_input->id] = debug_window.table;

    std::queue<int> queue;
    std::unordered_set<int> visited;
    queue.push(data_input->id);
    visited.insert(data_input->id);

    const bool has_downstream_nodes = !DownstreamNodes(
        data_input->id, nodes, links).empty();

    while (!queue.empty()) {
        const int node_id = queue.front();
        queue.pop();
        const auto table_it = table_by_node.find(node_id);
        const auto input_table = table_it == table_by_node.end()
            ? std::shared_ptr<arrow::Table>{}
            : table_it->second;

        for (const gui::MLNode* child : DownstreamNodes(node_id, nodes, links)) {
            if (!visited.insert(child->id).second) {
                continue;
            }

            if (child->type == gui::NodeType::TextTokenizer) {
                std::shared_ptr<arrow::Table> output_table;
                auto child_traces = TraceTextTokenizer(
                    run_id, *child, nodes, links, input_table, &output_table);
                if (!child_traces.empty() &&
                    child_traces.back().status == "ok" && output_table) {
                    table_by_node[child->id] = output_table;
                    queue.push(child->id);
                }
                AnnotateTraceWindow(
                    child_traces, debug_window, source_dataset_name, data_input);
                traces.insert(traces.end(),
                              std::make_move_iterator(child_traces.begin()),
                              std::make_move_iterator(child_traces.end()));
                continue;
            }

            if (!IsFoldedTextConfigNode(child->type)) {
                auto warning = BuildWarningTrace(
                    run_id,
                    *child,
                    input_table,
                    "Operator-backed debugger tracing currently supports TextTokenizer only.",
                    errors::Runtime::UnsupportedNode);
                warning.payload["diagnostic_phase"] = "unsupported_operator";
                AnnotateTraceWindow(warning, debug_window, source_dataset_name, data_input);
                traces.push_back(std::move(warning));
                continue;
            }

            table_by_node[child->id] = input_table;
            queue.push(child->id);
        }
    }

    if (traces.empty() && has_downstream_nodes) {
        auto warning = BuildWarningTrace(
            run_id,
            *data_input,
            debug_window.table,
            "Operator-backed debugger tracing skipped because only folded text configuration nodes were reachable without a TextTokenizer operator.");
        warning.payload["diagnostic_phase"] = "graph_walk";
        AnnotateTraceWindow(warning, debug_window, source_dataset_name, data_input);
        traces.push_back(std::move(warning));
    }

    return traces;
}

std::vector<DebugTraceRecord>
DebugOperatorTraceProducer::TraceTextTokenizer(
    const std::string& run_id,
    const gui::MLNode& node,
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    const std::shared_ptr<arrow::Table>& input,
    std::shared_ptr<arrow::Table>* output) const {
    if (!input) {
        return {BuildWarningTrace(
            run_id,
            node,
            input,
            "TextTokenizer debugger trace skipped because no Arrow source table was available.")};
    }

    TextTokenizerOperator op;
    std::string error;
    nlohmann::json folded_config_nodes = nlohmann::json::array();
    const auto params = BuildTextTokenizerParams(
        node, nodes, links, &folded_config_nodes);
    const auto effective_config = BuildTextTokenizerConfigPayload(params);
    const bool folded_config_applied = !folded_config_nodes.empty();
    if (!op.Configure(params, error)) {
        auto trace = BuildWarningTrace(run_id, node, input, error);
        trace.payload["diagnostic_phase"] = "configure";
        trace.payload["effective_text_tokenizer_config"] = effective_config;
        trace.payload["folded_text_config_applied"] = folded_config_applied;
        trace.payload["folded_text_config_nodes"] = folded_config_nodes;
        return {std::move(trace)};
    }

    const auto start = std::chrono::steady_clock::now();
    auto applied = op.Apply(input);
    const auto end = std::chrono::steady_clock::now();
    const auto duration_ms =
        std::chrono::duration<float, std::milli>(end - start).count();

    if (!applied.ok()) {
        auto trace = BuildWarningTrace(
            run_id,
            node,
            input,
            applied.status().ToString());
        trace.payload["diagnostic_phase"] = "apply";
        trace.payload["effective_text_tokenizer_config"] = effective_config;
        trace.payload["folded_text_config_applied"] = folded_config_applied;
        trace.payload["folded_text_config_nodes"] = folded_config_nodes;
        return {std::move(trace)};
    }

    auto output_table = applied.ValueOrDie();
    if (output) {
        *output = output_table;
    }

    DebugOperatorTraceAdapter adapter;
    DebugGraphTraceExecutor executor;
    auto step = adapter.BuildStep(
        node.id,
        node.name,
        "TextTokenizer",
        input,
        output_table,
        duration_ms);
    step.payload["trace_producer"] = "DebugOperatorTraceProducer";
    step.payload["operator_backed"] = true;
    step.payload["vocab_size"] = op.GetLastVocabSize();
    step.payload["effective_text_tokenizer_config"] = effective_config;
    step.payload["folded_text_config_applied"] = folded_config_applied;
    step.payload["folded_text_config_nodes"] = folded_config_nodes;
    auto traces = executor.TraceSteps(run_id, {std::move(step)});
    for (auto& trace : traces) {
        DebugNodeTraceContract::AttachDiagnosticContext(
            trace,
            "operator_transform",
            "DebugOperatorTraceProducer",
            "cyxwiz-engine/src/core/debug_operator_trace_producer.cpp",
            "cyxwiz::DebugOperatorTraceProducer::TraceTextTokenizer");
    }
    return traces;
}

DebugTraceRecord DebugOperatorTraceProducer::BuildWarningTrace(
    const std::string& run_id,
    const gui::MLNode& node,
    const std::shared_ptr<arrow::Table>& input,
    const std::string& message,
    const std::string& error_code) const {
    auto trace = DebugNodeTraceContract::Make(
        run_id,
        node.id,
        node.name,
        NodeTypeName(node.type),
        "OperatorTrace",
        DebugTraceRole::Warning,
        TableShape(input),
        {},
        "arrow::Table",
        "CPU",
        "warning");
    trace.payload["trace_producer"] = "DebugOperatorTraceProducer";
    trace.payload["operator_backed"] = false;
    trace.payload["message"] = message;
    trace.payload["error_code"] = error_code;
    DebugNodeTraceContract::AttachDiagnosticContext(
        trace,
        "operator_trace",
        "DebugOperatorTraceProducer",
        "cyxwiz-engine/src/core/debug_operator_trace_producer.cpp",
        "cyxwiz::DebugOperatorTraceProducer::BuildWarningTrace");
    DebugNodeTraceContract::AddWarning(
        trace,
        message,
        error_code);
    return trace;
}

} // namespace cyxwiz
