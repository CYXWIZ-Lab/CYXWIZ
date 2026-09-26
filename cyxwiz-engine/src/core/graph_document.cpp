#include "graph_document.h"

#include "data_input_parameters.h"
#include "graph_node_factory.h"
#include "pipeline_runtime_capabilities.h"
#include "../gui/node_import_guardrails.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <stdexcept>

namespace gui {

namespace {

bool TryReadSerializedNodeType(const nlohmann::json& node_json,
                                      NodeType& node_type) {
    if (!node_json.contains("type") || !node_json["type"].is_number_integer()) {
        spdlog::error("Serialized node '{}' is missing an integer node type",
                      node_json.value("name", "<unnamed>"));
        return false;
    }

    const int type_value = node_json["type"].get<int>();
    if (type_value < 0 || type_value >= static_cast<int>(NodeType::Unknown)) {
        spdlog::error("Serialized node '{}' has unsupported node type id {}",
                      node_json.value("name", "<unnamed>"),
                      type_value);
        return false;
    }

    node_type = static_cast<NodeType>(type_value);
    return true;
}

bool HasParamValue(const std::map<std::string, std::string>& params,
                          const std::string& key) {
    auto it = params.find(key);
    return it != params.end() && !it->second.empty();
}

void CopyLegacyParamIfMissing(std::map<std::string, std::string>& params,
                                     const std::string& canonical_key,
                                     const std::string& legacy_key,
                                     bool prefer_legacy = false) {
    if ((!prefer_legacy && HasParamValue(params, canonical_key)) ||
        !HasParamValue(params, legacy_key)) {
        return;
    }
    params[canonical_key] = params[legacy_key];
}

std::string FirstCsvToken(const std::string& value) {
    const size_t comma = value.find(',');
    std::string token = value.substr(0, comma);
    const size_t start = token.find_first_not_of(" \t");
    if (start == std::string::npos) {
        return "";
    }
    const size_t end = token.find_last_not_of(" \t");
    return token.substr(start, end - start + 1);
}

void CopyLegacyColumnIfMissing(std::map<std::string, std::string>& params,
                                      const std::string& canonical_key,
                                      const std::string& legacy_key,
                                      bool prefer_legacy = false) {
    if ((!prefer_legacy && HasParamValue(params, canonical_key)) ||
        !HasParamValue(params, legacy_key)) {
        return;
    }
    const std::string token = FirstCsvToken(params[legacy_key]);
    if (!token.empty()) {
        params[canonical_key] = token;
    }
}

bool ResolveSavedGraphLinkPins(const nlohmann::json& link_json,
                                      const MLNode* from_node,
                                      const MLNode* to_node,
                                      bool preserve_legacy_data_validator_outputs,
                                      bool preserve_legacy_evaluation_table_inputs,
                                      bool preserve_legacy_classical_tree_pins,
                                      NodeLink& link) {
    if (!from_node || !to_node) {
        spdlog::warn(
            "Skipping saved graph link {} ({} -> {}): referenced node is missing",
            link.id,
            link.from_node,
            link.to_node);
        return false;
    }

    if (preserve_legacy_evaluation_table_inputs &&
        detail::IsLegacySplitInputEvaluationNode(to_node->type)) {
        spdlog::warn(
            "Skipping saved graph link {} ({} -> {}): legacy '{}' used split "
            "prediction/label inputs that cannot be inferred as the required "
            "Dataset table; reconnect one table containing both configured columns",
            link.id,
            link.from_node,
            link.to_node,
            to_node->name);
        return false;
    }

    if (preserve_legacy_classical_tree_pins &&
        detail::IsLegacySplitInputTreeTrainer(to_node->type) &&
        link_json.contains("to_pin_index") &&
        link_json["to_pin_index"].is_number_integer() &&
        link_json["to_pin_index"].get<int>() == 1) {
        spdlog::warn(
            "Skipping saved graph link {} ({} -> {}): legacy '{}' used a "
            "separate Labels input that cannot be inferred as a target column "
            "inside the required Dataset table; reconnect one table and set "
            "target_col",
            link.id,
            link.from_node,
            link.to_node,
            to_node->name);
        return false;
    }

    if (preserve_legacy_evaluation_table_inputs &&
        from_node->type == NodeType::ROCCurveNode &&
        link_json.contains("from_pin_index") &&
        link_json["from_pin_index"].is_number_integer() &&
        link_json["from_pin_index"].get<int>() == 1) {
        spdlog::warn(
            "Skipping saved graph link {} ({} -> {}): legacy ROC AUC output "
            "is now the 'auc' column in the Curve table",
            link.id,
            link.from_node,
            link.to_node);
        return false;
    }

    int from_pin_index = 0;
    const bool legacy_data_validator_source =
        preserve_legacy_data_validator_outputs &&
        from_node->type == NodeType::DataValidator;
    const bool legacy_classical_tree_source =
        preserve_legacy_classical_tree_pins &&
        detail::IsLegacySplitInputTreeTrainer(from_node->type);
    const bool source_pin_resolved = legacy_data_validator_source
        ? detail::ResolveLegacyDataValidatorOutputPinIndex(
              link_json, from_pin_index)
        : legacy_classical_tree_source
            ? detail::ResolveLegacyClassicalTreeOutputPinIndex(
                  link_json, from_pin_index)
            : detail::ResolveSerializedPinIndex(
                  link_json, "from_pin_index", from_node->outputs.size(),
                  from_pin_index);
    if (!source_pin_resolved) {
        const std::string saved_index = link_json.contains("from_pin_index")
            ? link_json["from_pin_index"].dump()
            : "legacy default 0";
        if (legacy_data_validator_source) {
            spdlog::warn(
                "Skipping saved graph link {} ({} -> {}): legacy DataValidator "
                "output index {} is not a runtime artifact; only Issues "
                "(legacy index 2) can be migrated",
                link.id,
                link.from_node,
                link.to_node,
                saved_index);
            return false;
        }
        if (legacy_classical_tree_source) {
            spdlog::warn(
                "Skipping saved graph link {} ({} -> {}): legacy tree Model "
                "output index {} was not a runtime artifact; only Predictions "
                "(legacy index 1) can be migrated to the Dataset output",
                link.id,
                link.from_node,
                link.to_node,
                saved_index);
            return false;
        }
        spdlog::warn(
            "Skipping saved graph link {} ({} -> {}): source pin index {} is invalid "
            "for node '{}' ({} outputs)",
            link.id,
            link.from_node,
            link.to_node,
            saved_index,
            from_node->name,
            from_node->outputs.size());
        return false;
    }

    int to_pin_index = 0;
    if (!detail::ResolveSerializedPinIndex(
            link_json, "to_pin_index", to_node->inputs.size(), to_pin_index)) {
        const std::string saved_index = link_json.contains("to_pin_index")
            ? link_json["to_pin_index"].dump()
            : "legacy default 0";
        spdlog::warn(
            "Skipping saved graph link {} ({} -> {}): target pin index {} is invalid "
            "for node '{}' ({} inputs)",
            link.id,
            link.from_node,
            link.to_node,
            saved_index,
            to_node->name,
            to_node->inputs.size());
        return false;
    }

    link.from_pin = from_node->outputs[from_pin_index].id;
    link.to_pin = to_node->inputs[to_pin_index].id;
    return true;
}

}  // namespace

bool RejectDenseEncodedSequencePlaceholder(const MLNode& node,
                                                  const std::string& source) {
    std::string matched_marker;
    if (!detail::IsDenseEncodedSequencePlaceholder(node, matched_marker)) {
        return false;
    }

    spdlog::error("{} node '{}' is encoded as Dense but matches sequence/NER "
                  "placeholder marker '{}'; import requires a first-class "
                  "supported node type instead of erasing the original identity",
                  source,
                  node.name,
                  matched_marker);
    return true;
}

void MigrateLegacyNodeParameters(NodeType type,
                                 std::map<std::string, std::string>& params,
                                 bool prefer_legacy) {
    cyxwiz::CanonicalizePipelineParameterAliases(
        type, params, prefer_legacy);
    switch (type) {
        case NodeType::DataInput:
            cyxwiz::MigrateDataInputFormatAliases(params);
            break;
        case NodeType::ExportCSV:
        case NodeType::ExportParquet:
        case NodeType::ExportJSON:
        case NodeType::DataOutput:
            // Saved before path_base existed: relative outputs meant the exports
            // folder. Make that explicit so reopening never moves an output.
            if (params.find("path_base") == params.end()) params["path_base"] = "exports";
            break;
        case NodeType::TimeSeriesWindow:
            CopyLegacyColumnIfMissing(params, "value_col", "target_column", prefer_legacy);
            CopyLegacyColumnIfMissing(params, "value_col", "column", prefer_legacy);
            CopyLegacyColumnIfMissing(params, "value_col", "columns", prefer_legacy);
            CopyLegacyParamIfMissing(params, "input_width", "window_size", prefer_legacy);
            CopyLegacyParamIfMissing(params, "shift", "forecast_horizon", prefer_legacy);
            break;

        case NodeType::TimeSeriesFeatures:
            CopyLegacyColumnIfMissing(params, "value_col", "columns", prefer_legacy);
            CopyLegacyParamIfMissing(params, "lag_values", "lag_features", prefer_legacy);
            CopyLegacyParamIfMissing(params, "lag_values", "lag_periods", prefer_legacy);
            CopyLegacyParamIfMissing(params, "rolling_windows", "rolling_window", prefer_legacy);
            CopyLegacyParamIfMissing(params, "rolling_aggregations", "rolling_features", prefer_legacy);
            break;

        case NodeType::LogTransform:
        case NodeType::Differencing:
            CopyLegacyColumnIfMissing(params, "value_col", "column", prefer_legacy);
            CopyLegacyColumnIfMissing(params, "value_col", "columns", prefer_legacy);
            CopyLegacyColumnIfMissing(params, "value_col", "target_column", prefer_legacy);
            break;

        default:
            break;
    }
}

}  // namespace gui

namespace cyxwiz {

bool BuildGraphDocument(const nlohmann::json& document, const nlohmann::json& content,
                        const GraphDocumentLoadOptions& options, GraphDocument& out, std::string& error) {
    using namespace gui;
    out = {};
    try {
        const bool preserve_legacy_data_boundary = detail::PreserveLegacyDataBoundaryPins(document);
        const bool preserve_legacy_data_validator_outputs = detail::PreserveLegacyDataValidatorOutputs(document);
        const bool preserve_legacy_evaluation_table_inputs = detail::PreserveLegacyEvaluationTableInputs(document);
        const bool preserve_legacy_classical_tree_pins = detail::PreserveLegacyClassicalTreeTablePins(document);
        if (preserve_legacy_data_boundary) {
            spdlog::warn(
                "Loading an unversioned/legacy data boundary without changing its pins or links. Use the Data Split migration action to adopt Dataset v2 explicitly.");
        }

        // CreateGraphNode owns the pin contract; saved ids replace its node ids.
        int next_node_id = 1;
        int next_pin_id = 1;
        int max_node_id = 0;
        int max_link_id = 0;
        out.nodes.reserve(content["nodes"].size());
        for (const auto& node_json : content["nodes"]) {
            NodeType node_type = NodeType::Unknown;
            if (!TryReadSerializedNodeType(node_json, node_type)) {
                error = "node '" + node_json.value("name", std::string("<unnamed>")) + "' has an unsupported type";
                return false;
            }

            const int saved_node_id = node_json.at("id").get<int>();
            const std::string saved_node_name = node_json.at("name").get<std::string>();
            MLNode node = CreateGraphNode(node_type, saved_node_name, next_node_id, next_pin_id);
            node.id = saved_node_id;
            node.name = saved_node_name;
            node.description = node_json.value("description", std::string{});
            if (node_json.contains("parameters")) {
                node.parameters = node_json["parameters"].get<std::map<std::string, std::string>>();
            }
            MigrateLegacyNodeParameters(node.type, node.parameters);
            if (RejectDenseEncodedSequencePlaceholder(node, "Saved graph")) {
                error = "node '" + node.name + "' is a Dense standing in for a sequence placeholder";
                return false;
            }

            if (node.type == NodeType::DataInput || node.type == NodeType::DataSplit ||
                node.type == NodeType::DataLoader) {
                RebuildDataBoundaryPins(node, preserve_legacy_data_boundary, next_pin_id);
            }
            // Before links are restored by pin index: a multi-input SQL step
            // needs its named input pins to exist.
            (void)SyncSqlStepInputPins(node, next_pin_id);

            max_node_id = std::max(max_node_id, node.id);
            out.nodes.push_back(std::move(node));
        }

        if (options.restore_extra_pins) options.restore_extra_pins(out.nodes, next_pin_id);

        const auto find_loaded_node = [&out](int node_id) -> const MLNode* {
            const auto it = std::find_if(out.nodes.begin(), out.nodes.end(),
                                         [node_id](const MLNode& node) { return node.id == node_id; });
            return it == out.nodes.end() ? nullptr : &*it;
        };

        out.links.reserve(content["links"].size());
        for (const auto& link_json : content["links"]) {
            NodeLink link;
            link.id = link_json.at("id").get<int>();
            link.from_node = link_json.at("from_node").get<int>();
            link.to_node = link_json.at("to_node").get<int>();
            if (!ResolveSavedGraphLinkPins(link_json, find_loaded_node(link.from_node),
                                           find_loaded_node(link.to_node), preserve_legacy_data_validator_outputs,
                                           preserve_legacy_evaluation_table_inputs,
                                           preserve_legacy_classical_tree_pins, link)) {
                if (options.strict_links) {
                    throw std::runtime_error("invalid link " + std::to_string(link.id) +
                                             " (its pins do not exist on the loaded nodes)");
                }
                continue;
            }
            if (link_json.contains("link_type")) {
                link.type = static_cast<LinkType>(link_json["link_type"].get<int>());
            }
            max_link_id = std::max(max_link_id, link.id);
            out.links.push_back(link);
        }

        out.next_node_id = max_node_id + 1;
        out.next_link_id = max_link_id + 1;
        out.next_pin_id = next_pin_id;
        return true;
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
}

bool ParseGraphDocument(const std::string& json_text, GraphDocument& out, std::string& error) {
    out = {};
    nlohmann::json root;
    try {
        root = nlohmann::json::parse(json_text);
    } catch (const std::exception& e) {
        error = std::string("the graph is not valid JSON: ") + e.what();
        return false;
    }
    if (!root.is_object() || !root.contains("nodes") || !root["nodes"].is_array() || !root.contains("links") ||
        !root["links"].is_array()) {
        error = "the graph has no nodes and links arrays";
        return false;
    }
    if (root.contains("subgraphs")) {
        error = "the graph contains subgraphs; expand them in the Engine before running it here";
        return false;
    }
    for (const auto& node : root["nodes"]) {
        if (node.value("type", -1) == static_cast<int>(gui::NodeType::Subgraph)) {
            error = "the graph contains subgraphs; expand them in the Engine before running it here";
            return false;
        }
    }
    if (!BuildGraphDocument(root, root, {}, out, error)) {
        error = "the graph could not be loaded: " + error;
        return false;
    }
    return true;
}

}  // namespace cyxwiz
