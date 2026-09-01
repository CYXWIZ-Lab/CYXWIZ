// File I/O Module for Node Editor
// This module contains all file operations for the visual node editor:
// - Graph save/load functionality (JSON serialization)
// - Cross-platform native file dialogs
// - Code export functionality (PyTorch, TensorFlow, Keras, PyCyxWiz)

#include "node_editor.h"
#include "node_type_import_registry.h"
#include "../core/data_input_parameters.h"
#include "../core/pipeline_runtime_capabilities.h"
#include "node_import_guardrails.h"
#include "../core/file_dialogs.h"
#include "../core/project_manager.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <unordered_map>
#include <spdlog/spdlog.h>

namespace gui {

// ========== Pattern to Graph Conversion ==========

// Resolve legacy pattern/import names through one shared authority.
static NodeType StringToNodeType(const std::string& type_str) {
    const auto resolved = ResolveNodeTypeImportName(type_str);
    if (resolved.has_value()) {
        return *resolved;
    }

    spdlog::warn("Unknown node type '{}'", type_str);
    return NodeType::Unknown;
}

static bool TryReadSerializedNodeType(const nlohmann::json& node_json,
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

static bool RejectDenseEncodedSequencePlaceholder(const MLNode& node,
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

static bool HasParamValue(const std::map<std::string, std::string>& params,
                          const std::string& key) {
    auto it = params.find(key);
    return it != params.end() && !it->second.empty();
}

static void CopyLegacyParamIfMissing(std::map<std::string, std::string>& params,
                                     const std::string& canonical_key,
                                     const std::string& legacy_key,
                                     bool prefer_legacy = false) {
    if ((!prefer_legacy && HasParamValue(params, canonical_key)) ||
        !HasParamValue(params, legacy_key)) {
        return;
    }
    params[canonical_key] = params[legacy_key];
}

static std::string FirstCsvToken(const std::string& value) {
    const size_t comma = value.find(',');
    std::string token = value.substr(0, comma);
    const size_t start = token.find_first_not_of(" \t");
    if (start == std::string::npos) {
        return "";
    }
    const size_t end = token.find_last_not_of(" \t");
    return token.substr(start, end - start + 1);
}

static void CopyLegacyColumnIfMissing(std::map<std::string, std::string>& params,
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

static void MigrateLegacyNodeParameters(NodeType type,
                                        std::map<std::string, std::string>& params,
                                        bool prefer_legacy = false) {
    cyxwiz::CanonicalizePipelineParameterAliases(
        type, params, prefer_legacy);
    switch (type) {
        case NodeType::DataInput:
            cyxwiz::MigrateDataInputFormatAliases(params);
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

bool NodeEditor::LoadPatternAsGraph(const nlohmann::json& j) {
    // Clear existing graph
    ClearGraph();

    const auto& tmpl = j["template"];

    // Map string IDs to integer IDs
    std::unordered_map<std::string, int> id_map;
    int next_id = 1;

    // Load nodes from template
    if (tmpl.contains("nodes") && tmpl["nodes"].is_array()) {
        for (const auto& node_json : tmpl["nodes"]) {
            std::string str_id = node_json.value("id", "");
            std::string type_str = node_json.value("type", "Dense");
            std::string name = node_json.value("name", type_str);

            // Convert string type to NodeType enum
            NodeType node_type = StringToNodeType(type_str);
            if (node_type == NodeType::Unknown) {
                spdlog::error("Cannot load pattern node '{}' with unknown type '{}'",
                              name,
                              type_str);
                ClearGraph();
                return false;
            }
            MLNode early_node;
            early_node.type = node_type;
            early_node.name = name;
            if (RejectDenseEncodedSequencePlaceholder(early_node, "Pattern")) {
                ClearGraph();
                return false;
            }

            // Create node with proper pins
            MLNode node = CreateNode(node_type, name);
            node.id = next_id;
            id_map[str_id] = next_id;
            next_id++;

            // Parse position (two formats supported: pos_x/pos_y or pos: [x, y])
            float pos_x = node_json.value("pos_x", 0.0f);
            float pos_y = node_json.value("pos_y", 0.0f);

            if (node_json.contains("pos") && node_json["pos"].is_array() && node_json["pos"].size() >= 2) {
                pos_x = node_json["pos"][0].get<float>();
                pos_y = node_json["pos"][1].get<float>();
            }

            // Apply any node parameters (substitute pattern parameters with defaults)
            if (node_json.contains("params") && node_json["params"].is_object()) {
                for (auto& [key, value] : node_json["params"].items()) {
                    std::string param_value;
                    if (value.is_string()) {
                        param_value = value.get<std::string>();
                        // Handle pattern parameter references like "$hidden1_size"
                        if (!param_value.empty() && param_value[0] == '$') {
                            // Find the default value for this parameter
                            std::string param_name = param_value.substr(1);
                            if (j.contains("parameters") && j["parameters"].is_array()) {
                                for (const auto& p : j["parameters"]) {
                                    if (p.value("name", "") == param_name) {
                                        param_value = p.value("default_value", param_value);
                                        break;
                                    }
                                }
                            }
                        }
                    } else if (value.is_number_integer()) {
                        param_value = std::to_string(value.get<int>());
                    } else if (value.is_number_float()) {
                        param_value = std::to_string(value.get<float>());
                    } else if (value.is_boolean()) {
                        param_value = value.get<bool>() ? "true" : "false";
                    }
                    node.parameters[key] = param_value;
                }
            }
            MigrateLegacyNodeParameters(node.type, node.parameters, true);
            if (RejectDenseEncodedSequencePlaceholder(node, "Pattern")) {
                ClearGraph();
                return false;
            }

            nodes_.push_back(node);

            // Queue position
            pending_positions_[node.id] = ImVec2(pos_x, pos_y);
        }
    }

    pending_positions_frames_ = 3;

    // Load links from template
    int link_id = 1;
    if (tmpl.contains("links") && tmpl["links"].is_array()) {
        for (const auto& link_json : tmpl["links"]) {
            std::string from_str = link_json.value("from", "");
            std::string to_str = link_json.value("to", "");

            auto from_it = id_map.find(from_str);
            auto to_it = id_map.find(to_str);

            if (from_it == id_map.end() || to_it == id_map.end()) {
                spdlog::warn("Link references unknown node: {} -> {}", from_str, to_str);
                continue;
            }

            int from_node_id = from_it->second;
            int to_node_id = to_it->second;

            const MLNode* from_node = FindNodeById(from_node_id);
            const MLNode* to_node = FindNodeById(to_node_id);

            if (!from_node || !to_node) {
                spdlog::warn("Could not find nodes for link: {} -> {}", from_str, to_str);
                continue;
            }

            // Get pin indices (default to first pin)
            int from_pin_idx = link_json.value("from_pin", 0);
            int to_pin_idx = link_json.value("to_pin", 0);

            if (from_pin_idx < 0 ||
                from_pin_idx >= static_cast<int>(from_node->outputs.size())) {
                spdlog::warn(
                    "Skipping pattern link {} -> {}: source pin index {} is out of range for node '{}' ({} outputs)",
                    from_str,
                    to_str,
                    from_pin_idx,
                    from_node->name,
                    from_node->outputs.size());
                continue;
            }
            if (to_pin_idx < 0 ||
                to_pin_idx >= static_cast<int>(to_node->inputs.size())) {
                spdlog::warn(
                    "Skipping pattern link {} -> {}: target pin index {} is out of range for node '{}' ({} inputs)",
                    from_str,
                    to_str,
                    to_pin_idx,
                    to_node->name,
                    to_node->inputs.size());
                continue;
            }

            // Create link using actual pin IDs
            NodeLink link;
            link.id = link_id++;
            link.from_pin = from_node->outputs[from_pin_idx].id;
            link.to_pin = to_node->inputs[to_pin_idx].id;
            link.from_node = from_node_id;
            link.to_node = to_node_id;
            link.type = LinkType::TensorFlow;

            links_.push_back(link);
        }
    }

    // Update next IDs
    next_node_id_ = next_id;
    next_link_id_ = link_id;

    std::string name = j.value("name", "Imported Pattern");
    spdlog::info("Loaded pattern '{}' as graph ({} nodes, {} links)",
                 name, nodes_.size(), links_.size());

    return true;
}

// ========== Graph Save/Load Implementation ==========

void NodeEditor::RebuildDataBoundaryPins(
    MLNode& node,
    bool legacy_contract) {
    if (node.type != NodeType::DataInput &&
        node.type != NodeType::DataSplit &&
        node.type != NodeType::DataLoader) {
        return;
    }

    node.inputs.clear();
    node.outputs.clear();
    const auto add_pin = [&](std::vector<NodePin>& pins,
                             PinType type,
                             const char* name,
                             bool is_input,
                             bool required,
                             const char* description) {
        NodePin pin;
        pin.id = next_pin_id_++;
        pin.type = type;
        pin.name = name;
        pin.is_input = is_input;
        pin.is_required = required;
        pin.description = description;
        pins.push_back(std::move(pin));
    };

    if (legacy_contract) {
        node.parameters["data_boundary_pin_contract"] = "legacy.v1";
        node.parameters["data_boundary_migration_required"] = "true";
        if (node.type == NodeType::DataInput) {
            add_pin(node.outputs, PinType::Tensor, "Data", false, true,
                    "Preserved legacy feature output. Migrate the graph data boundary to use Dataset pins.");
            add_pin(node.outputs, PinType::Labels, "Labels", false, false,
                    "Preserved legacy label output. Migrate explicitly before saving as the Dataset contract.");
        } else if (node.type == NodeType::DataSplit) {
            add_pin(node.inputs, PinType::Tensor, "Data", true, true,
                    "Preserved legacy feature input.");
            add_pin(node.inputs, PinType::Labels, "Labels", true, true,
                    "Preserved legacy label input.");
            add_pin(node.outputs, PinType::Tensor, "Train Data", false, true,
                    "Preserved legacy training feature output.");
            add_pin(node.outputs, PinType::Labels, "Train Labels", false, false,
                    "Preserved legacy training label output.");
            add_pin(node.outputs, PinType::Tensor, "Val Data", false, false,
                    "Preserved legacy validation feature output.");
            add_pin(node.outputs, PinType::Labels, "Val Labels", false, false,
                    "Preserved legacy validation label output.");
            add_pin(node.outputs, PinType::Tensor, "Test Data", false, false,
                    "Preserved legacy held-out feature output.");
            add_pin(node.outputs, PinType::Labels, "Test Labels", false, false,
                    "Preserved legacy held-out label output.");
        } else {
            add_pin(node.inputs, PinType::Tensor, "Data", true, true,
                    "Preserved legacy unbatched feature input.");
            add_pin(node.inputs, PinType::Labels, "Labels", true, true,
                    "Preserved legacy unbatched label input.");
            add_pin(node.outputs, PinType::Tensor, "Data", false, true,
                    "Batched feature tensor.");
            add_pin(node.outputs, PinType::Labels, "Labels", false, false,
                    "Batched labels for the loss target.");
        }
        return;
    }

    node.parameters["data_boundary_pin_contract"] = "dataset.v2";
    node.parameters.erase("data_boundary_migration_required");
    if (node.type == NodeType::DataInput) {
        add_pin(node.outputs, PinType::Dataset, "Dataset", false, true,
                "Loaded Dataset asset. Connect it to a named Data Split role input.");
    } else if (node.type == NodeType::DataSplit) {
        add_pin(node.inputs, PinType::Dataset, "Training Dataset", true, true,
                "Required Training source. Missing roles are derived only from this Dataset.");
        add_pin(node.inputs, PinType::Dataset, "Validation Dataset", true, false,
                "Optional external Validation/Dev Dataset preserved in full.");
        add_pin(node.inputs, PinType::Dataset, "Test Dataset", true, false,
                "Optional external held-out Test Dataset preserved in full.");
        add_pin(node.outputs, PinType::Dataset, "Partitions", false, true,
                "Resolved Train/Validation/Test partitions and manifest.");
    } else {
        add_pin(node.inputs, PinType::Dataset, "Partitions", true, true,
                "Resolved partition contract from Data Split.");
        add_pin(node.outputs, PinType::Tensor, "Data", false, true,
                "Batched feature tensor for the model.");
        add_pin(node.outputs, PinType::Labels, "Labels", false, false,
                "Batched labels for supervised loss targets.");
    }
}

bool NodeEditor::HasLegacyDataBoundary() const {
    for (const auto& node : nodes_) {
        auto it = node.parameters.find("data_boundary_pin_contract");
        if (it != node.parameters.end() && it->second == "legacy.v1") {
            return true;
        }
    }
    return false;
}

DataBoundaryMigrationResult NodeEditor::MigrateLegacyDataBoundary() {
    DataBoundaryMigrationResult result;
    if (!HasLegacyDataBoundary()) {
        result.message = "The graph already uses the Dataset v2 boundary.";
        return result;
    }

    const auto pin_index = [](const std::vector<NodePin>& pins, int pin_id) {
        for (size_t i = 0; i < pins.size(); ++i) {
            if (pins[i].id == pin_id) return static_cast<int>(i);
        }
        return -1;
    };
    const auto is_legacy = [](const MLNode* node) {
        if (!node) return false;
        auto it = node->parameters.find("data_boundary_pin_contract");
        return it != node->parameters.end() && it->second == "legacy.v1";
    };

    std::unordered_map<int, int> split_to_loader;
    for (const auto& link : links_) {
        const auto* from = FindNodeById(link.from_node);
        const auto* to = FindNodeById(link.to_node);
        if (!from || !to || from->type != NodeType::DataSplit ||
            to->type != NodeType::DataLoader || !is_legacy(from) ||
            !is_legacy(to)) {
            continue;
        }
        if (pin_index(from->outputs, link.from_pin) == 0 &&
            pin_index(to->inputs, link.to_pin) == 0) {
            if (split_to_loader.count(from->id) > 0 &&
                split_to_loader[from->id] != to->id) {
                result.message = "Migration is blocked: legacy Data Split '" +
                    from->name + "' feeds more than one Data Loader.";
                return result;
            }
            split_to_loader[from->id] = to->id;
        }
    }

    std::unordered_map<int, int> input_to_loader;
    for (const auto& link : links_) {
        const auto* from = FindNodeById(link.from_node);
        const auto* to = FindNodeById(link.to_node);
        if (!from || !to || from->type != NodeType::DataInput ||
            to->type != NodeType::DataSplit || !is_legacy(from) ||
            !is_legacy(to)) {
            continue;
        }
        if (pin_index(from->outputs, link.from_pin) == 0 &&
            pin_index(to->inputs, link.to_pin) == 0 &&
            split_to_loader.count(to->id) > 0) {
            input_to_loader[from->id] = split_to_loader[to->id];
        }
    }

    enum class LinkAction { Keep, RemoveDuplicate, RerouteToLoaderLabels };
    struct PlannedLink {
        NodeLink link;
        int from_index = -1;
        int to_index = -1;
        LinkAction action = LinkAction::Keep;
        int reroute_loader_id = -1;
    };
    std::vector<PlannedLink> plan;
    plan.reserve(links_.size());

    for (const auto& link : links_) {
        const auto* from = FindNodeById(link.from_node);
        const auto* to = FindNodeById(link.to_node);
        if (!from || !to) {
            result.message = "Migration is blocked: a graph link references a missing node.";
            return result;
        }
        PlannedLink item;
        item.link = link;
        item.from_index = pin_index(from->outputs, link.from_pin);
        item.to_index = pin_index(to->inputs, link.to_pin);
        if (item.from_index < 0 || item.to_index < 0) {
            result.message = "Migration is blocked: link " +
                std::to_string(link.id) + " references an unknown pin.";
            return result;
        }

        if (is_legacy(from) && from->type == NodeType::DataInput &&
            item.from_index == 0 &&
            !(is_legacy(to) && to->type == NodeType::DataSplit &&
              item.to_index == 0)) {
            result.message = "Migration is blocked: legacy Data Input features from '" +
                from->name + "' bypass Data Split. Insert/preserve an explicit Split + Loader boundary before migration.";
            return result;
        }
        if (is_legacy(from) && from->type == NodeType::DataSplit &&
            item.from_index == 0 &&
            !(is_legacy(to) && to->type == NodeType::DataLoader &&
              item.to_index == 0)) {
            result.message = "Migration is blocked: legacy Training Data from Data Split '" +
                from->name + "' bypasses the Data Loader.";
            return result;
        }
        if (is_legacy(to) && to->type == NodeType::DataSplit &&
            item.to_index == 0 &&
            !(is_legacy(from) && from->type == NodeType::DataInput &&
              item.from_index == 0)) {
            result.message = "Migration is blocked: legacy Data Split '" +
                to->name + "' has a non-standard Data source.";
            return result;
        }
        if (is_legacy(to) && to->type == NodeType::DataLoader &&
            item.to_index == 0 &&
            !(is_legacy(from) && from->type == NodeType::DataSplit &&
              item.from_index == 0)) {
            result.message = "Migration is blocked: legacy Data Loader '" +
                to->name + "' does not receive Training Data from a legacy Data Split.";
            return result;
        }

        if (is_legacy(from) && from->type == NodeType::DataSplit &&
            item.from_index >= 2) {
            result.message = "Migration is blocked: legacy Data Split '" +
                from->name + "' has a connected Val/Test canvas branch. "
                "Disconnect or preserve that graph until the branch is redesigned around runtime partitions.";
            return result;
        }

        if (is_legacy(from) && from->type == NodeType::DataInput &&
            item.from_index == 1) {
            if (is_legacy(to) && to->type == NodeType::DataSplit &&
                item.to_index == 1) {
                item.action = LinkAction::RemoveDuplicate;
            } else if (input_to_loader.count(from->id) > 0) {
                item.action = LinkAction::RerouteToLoaderLabels;
                item.reroute_loader_id = input_to_loader[from->id];
            } else {
                result.message = "Migration is blocked: legacy label output from Data Input '" +
                    from->name + "' cannot be mapped to a unique Data Loader Labels output.";
                return result;
            }
        }

        if (is_legacy(from) && from->type == NodeType::DataSplit &&
            item.from_index == 1) {
            auto loader_it = split_to_loader.find(from->id);
            if (loader_it == split_to_loader.end()) {
                result.message = "Migration is blocked: legacy Data Split '" +
                    from->name + "' has no unique downstream Data Loader.";
                return result;
            }
            if (is_legacy(to) && to->type == NodeType::DataLoader &&
                to->id == loader_it->second && item.to_index == 1) {
                item.action = LinkAction::RemoveDuplicate;
            } else {
                item.action = LinkAction::RerouteToLoaderLabels;
                item.reroute_loader_id = loader_it->second;
            }
        }

        if (is_legacy(to) && to->type == NodeType::DataSplit &&
            item.to_index == 1 && item.action != LinkAction::RemoveDuplicate) {
            result.message = "Migration is blocked: legacy Data Split Labels input has a non-standard source.";
            return result;
        }
        if (is_legacy(to) && to->type == NodeType::DataLoader &&
            item.to_index == 1 && item.action != LinkAction::RemoveDuplicate) {
            result.message = "Migration is blocked: legacy Data Loader Labels input has a non-standard source.";
            return result;
        }
        plan.push_back(std::move(item));
    }

    SaveUndoState();
    for (auto& node : nodes_) {
        if (is_legacy(&node)) {
            RebuildDataBoundaryPins(node, false);
            ++result.nodes_migrated;
        }
    }

    std::vector<NodeLink> migrated_links;
    migrated_links.reserve(plan.size());
    for (auto& item : plan) {
        if (item.action == LinkAction::RemoveDuplicate) {
            ++result.links_removed;
            continue;
        }
        auto* from = FindNodeById(item.link.from_node);
        auto* to = FindNodeById(item.link.to_node);
        if (item.action == LinkAction::RerouteToLoaderLabels) {
            from = FindNodeById(item.reroute_loader_id);
            item.link.from_node = item.reroute_loader_id;
            item.from_index = 1;
            ++result.links_rerouted;
        }
        if (!from || !to || item.from_index < 0 || item.to_index < 0 ||
            item.from_index >= static_cast<int>(from->outputs.size()) ||
            item.to_index >= static_cast<int>(to->inputs.size())) {
            result.message = "Migration failed while applying the validated link plan.";
            Undo();
            return result;
        }
        item.link.from_pin = from->outputs[item.from_index].id;
        item.link.to_pin = to->inputs[item.to_index].id;
        migrated_links.push_back(item.link);
    }
    links_ = std::move(migrated_links);
    RebuildPinLookup();
    ClearValidationState();

    result.success = true;
    result.message = "Migrated " + std::to_string(result.nodes_migrated) +
        " data-boundary nodes; removed " +
        std::to_string(result.links_removed) +
        " duplicate legacy label links and rerouted " +
        std::to_string(result.links_rerouted) + " label targets.";
    spdlog::info("Track70 data-boundary migration: {}", result.message);
    return result;
}

bool NodeEditor::SaveGraph(const std::string& filepath) {
    using json = nlohmann::json;

    try {
        json j;
        // CyxWiz Studio: Update to v2.1 format with annotations
        j["version"] = "2.1";
        j["data_boundary_version"] = HasLegacyDataBoundary()
            ? detail::kLegacyDataBoundaryVersion
            : detail::kCurrentDataBoundaryVersion;
        j["data_validator_contract_version"] =
            detail::kCurrentDataValidatorContractVersion;
        j["evaluation_table_contract_version"] =
            detail::kCurrentEvaluationTableContractVersion;
        j["classical_tree_table_contract_version"] =
            detail::kCurrentClassicalTreeTableContractVersion;
        j["framework"] = static_cast<int>(selected_framework_);
        j["execution_mode"] = static_cast<int>(execution_mode_);  // Save execution mode

        // CyxWiz Studio: Save workflow description
        j["workflow_description"] = std::string(workflow_description_);

        // CyxWiz Studio: Save canvas annotations
        json annotations_array = json::array();
        for (const auto& annotation : annotations_) {
            json ann_json;
            ann_json["id"] = annotation.id;
            ann_json["title"] = annotation.title;
            ann_json["content"] = annotation.content;
            ann_json["pos_x"] = annotation.position.x;
            ann_json["pos_y"] = annotation.position.y;
            ann_json["width"] = annotation.size.x;
            ann_json["height"] = annotation.size.y;
            ann_json["color"] = annotation.color;
            ann_json["minimized"] = annotation.is_minimized;
            annotations_array.push_back(ann_json);
        }
        j["annotations"] = annotations_array;

        // CyxWiz Studio: Save node groups
        json groups_array = json::array();
        for (const auto& group : groups_) {
            json group_json;
            group_json["id"] = group.id;
            group_json["name"] = group.name;
            group_json["description"] = group.description;
            group_json["node_ids"] = group.node_ids;
            group_json["color_r"] = group.color.x;
            group_json["color_g"] = group.color.y;
            group_json["color_b"] = group.color.z;
            group_json["color_a"] = group.color.w;
            group_json["collapsed"] = group.collapsed;
            group_json["padding"] = group.padding;
            groups_array.push_back(group_json);
        }
        j["groups"] = groups_array;

        // Serialize nodes
        json nodes_array = json::array();
        for (const auto& node : nodes_) {
            json node_json;
            node_json["id"] = node.id;
            node_json["type"] = static_cast<int>(node.type);
            node_json["name"] = node.name;
            node_json["description"] = node.description;
            node_json["parameters"] = node.parameters;

            // Unified Canvas Phase 7: Save node category for better organization
            node_json["category"] = static_cast<int>(node.category);

            // Save node position
            auto it = cached_node_positions_.find(node.id);
            ImVec2 pos = (it != cached_node_positions_.end()) ? it->second : ImVec2(0,0);
            node_json["pos_x"] = pos.x;
            node_json["pos_y"] = pos.y;

            nodes_array.push_back(node_json);
        }
        j["nodes"] = nodes_array;

        // Serialize links with pin indices for multi-pin support
        json links_array = json::array();
        for (const auto& link : links_) {
            json link_json;
            link_json["id"] = link.id;
            link_json["from_node"] = link.from_node;
            link_json["from_pin"] = link.from_pin;
            link_json["to_node"] = link.to_node;
            link_json["to_pin"] = link.to_pin;

            // Save pin indices for proper multi-pin node support
            const MLNode* from_node = FindNodeById(link.from_node);
            const MLNode* to_node = FindNodeById(link.to_node);

            int from_pin_index = 0;
            if (from_node) {
                for (size_t i = 0; i < from_node->outputs.size(); ++i) {
                    if (from_node->outputs[i].id == link.from_pin) {
                        from_pin_index = static_cast<int>(i);
                        break;
                    }
                }
            }

            int to_pin_index = 0;
            if (to_node) {
                for (size_t i = 0; i < to_node->inputs.size(); ++i) {
                    if (to_node->inputs[i].id == link.to_pin) {
                        to_pin_index = static_cast<int>(i);
                        break;
                    }
                }
            }

            link_json["from_pin_index"] = from_pin_index;
            link_json["to_pin_index"] = to_pin_index;

            // Save link type for skip connection visualization
            link_json["link_type"] = static_cast<int>(link.type);

            links_array.push_back(link_json);
        }
        j["links"] = links_array;

        // Write to file
        std::ofstream file(filepath);
        if (!file.is_open()) {
            spdlog::error("Failed to open file for writing: {}", filepath);
            return false;
        }

        file << j.dump(4);  // Pretty print with 4-space indent
        current_file_path_ = filepath;
        spdlog::info("Graph saved to: {}", filepath);
        return true;

    } catch (const std::exception& e) {
        spdlog::error("Error saving graph: {}", e.what());
        return false;
    }
}

static bool ResolveSavedGraphLinkPins(const nlohmann::json& link_json,
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

bool NodeEditor::LoadGraphJson(const nlohmann::json& graph_json,
                               const std::string& source_description) {
    if (!graph_json.is_object() ||
        !graph_json.contains("nodes") || !graph_json["nodes"].is_array() ||
        !graph_json.contains("links") || !graph_json["links"].is_array()) {
        spdlog::error("Cannot load graph from {}: expected nodes and links arrays",
                      source_description);
        return false;
    }

    struct CounterRollback {
        int& node_id;
        int& pin_id;
        int saved_node_id;
        int saved_pin_id;
        bool active = true;

        CounterRollback(int& node_id_ref, int& pin_id_ref)
            : node_id(node_id_ref),
              pin_id(pin_id_ref),
              saved_node_id(node_id_ref),
              saved_pin_id(pin_id_ref) {}

        void Restore() {
            if (!active) return;
            node_id = saved_node_id;
            pin_id = saved_pin_id;
            active = false;
        }

        ~CounterRollback() { Restore(); }
    } counters(next_node_id_, next_pin_id_);

    try {
        // CreateNode owns the pin contract, so use it to build a complete
        // replacement graph without touching the live node/link containers.
        next_node_id_ = 1;
        next_pin_id_ = 1;

        std::vector<MLNode> loaded_nodes;
        std::vector<NodeLink> loaded_links;
        std::map<int, ImVec2> loaded_positions;
        std::vector<NodeGroup> loaded_groups;
        std::vector<CanvasAnnotation> loaded_annotations;
        int loaded_next_group_id = 1;
        int loaded_next_annotation_id = 1;
        int max_node_id = 0;
        int max_link_id = 0;

        const bool preserve_legacy_data_boundary =
            detail::PreserveLegacyDataBoundaryPins(graph_json);
        const bool preserve_legacy_data_validator_outputs =
            detail::PreserveLegacyDataValidatorOutputs(graph_json);
        const bool preserve_legacy_evaluation_table_inputs =
            detail::PreserveLegacyEvaluationTableInputs(graph_json);
        const bool preserve_legacy_classical_tree_pins =
            detail::PreserveLegacyClassicalTreeTablePins(graph_json);
        if (preserve_legacy_data_boundary) {
            spdlog::warn(
                "Loading an unversioned/legacy data boundary without changing its pins or links. Use the Data Split migration action to adopt Dataset v2 explicitly.");
        }

        CodeFramework loaded_framework = selected_framework_;
        if (graph_json.contains("framework")) {
            loaded_framework = static_cast<CodeFramework>(
                graph_json["framework"].get<int>());
        }

        ExecutionMode loaded_execution_mode = ExecutionMode::CodeGeneration;
        if (graph_json.contains("execution_mode")) {
            loaded_execution_mode = static_cast<ExecutionMode>(
                graph_json["execution_mode"].get<int>());
        }
        const std::string loaded_workflow_description =
            graph_json.value("workflow_description", std::string{});

        if (graph_json.contains("annotations")) {
            if (!graph_json["annotations"].is_array()) {
                throw std::runtime_error("annotations must be an array");
            }
            for (const auto& annotation_json : graph_json["annotations"]) {
                CanvasAnnotation annotation;
                annotation.id = annotation_json.value("id", loaded_next_annotation_id);
                annotation.title = annotation_json.value("title", "");
                annotation.content = annotation_json.value("content", "");
                annotation.position.x = annotation_json.value("pos_x", 0.0f);
                annotation.position.y = annotation_json.value("pos_y", 0.0f);
                annotation.size.x = annotation_json.value("width", 200.0f);
                annotation.size.y = annotation_json.value("height", 100.0f);
                annotation.color = annotation_json.value(
                    "color", static_cast<ImU32>(IM_COL32(255, 255, 200, 255)));
                annotation.is_minimized = annotation_json.value("minimized", false);
                loaded_annotations.push_back(std::move(annotation));
                loaded_next_annotation_id = std::max(
                    loaded_next_annotation_id,
                    loaded_annotations.back().id + 1);
            }
        }

        if (graph_json.contains("groups")) {
            if (!graph_json["groups"].is_array()) {
                throw std::runtime_error("groups must be an array");
            }
            for (const auto& group_json : graph_json["groups"]) {
                NodeGroup group;
                group.id = group_json.value("id", loaded_next_group_id);
                group.name = group_json.value("name", "Group");
                group.description = group_json.value("description", "");
                if (group_json.contains("node_ids")) {
                    group.node_ids = group_json["node_ids"].get<std::vector<int>>();
                }
                group.color.x = group_json.value("color_r", 0.2f);
                group.color.y = group_json.value("color_g", 0.3f);
                group.color.z = group_json.value("color_b", 0.4f);
                group.color.w = group_json.value("color_a", 0.3f);
                group.collapsed = group_json.value("collapsed", false);
                group.padding = group_json.value("padding", 20.0f);
                loaded_groups.push_back(std::move(group));
                loaded_next_group_id = std::max(
                    loaded_next_group_id,
                    loaded_groups.back().id + 1);
            }
        }

        loaded_nodes.reserve(graph_json["nodes"].size());
        for (const auto& node_json : graph_json["nodes"]) {
            NodeType node_type = NodeType::Unknown;
            if (!TryReadSerializedNodeType(node_json, node_type)) {
                return false;
            }

            const int saved_node_id = node_json.at("id").get<int>();
            const std::string saved_node_name =
                node_json.at("name").get<std::string>();
            MLNode node = CreateNode(node_type, saved_node_name);
            node.id = saved_node_id;
            node.name = saved_node_name;
            node.description = node_json.value("description", std::string{});
            if (node_json.contains("parameters")) {
                node.parameters = node_json["parameters"].get<
                    std::map<std::string, std::string>>();
            }
            MigrateLegacyNodeParameters(node.type, node.parameters);
            if (RejectDenseEncodedSequencePlaceholder(node, "Saved graph")) {
                return false;
            }

            if (node.type == NodeType::DataInput ||
                node.type == NodeType::DataSplit ||
                node.type == NodeType::DataLoader) {
                RebuildDataBoundaryPins(node, preserve_legacy_data_boundary);
            }

            max_node_id = std::max(max_node_id, node.id);
            if (node_json.contains("pos_x") && node_json.contains("pos_y")) {
                loaded_positions[node.id] = ImVec2(
                    node_json["pos_x"].get<float>(),
                    node_json["pos_y"].get<float>());
            }
            loaded_nodes.push_back(std::move(node));
        }

        const auto find_loaded_node = [&loaded_nodes](int node_id) -> const MLNode* {
            const auto it = std::find_if(
                loaded_nodes.begin(), loaded_nodes.end(),
                [node_id](const MLNode& node) { return node.id == node_id; });
            return it == loaded_nodes.end() ? nullptr : &*it;
        };

        loaded_links.reserve(graph_json["links"].size());
        for (const auto& link_json : graph_json["links"]) {
            NodeLink link;
            link.id = link_json.at("id").get<int>();
            link.from_node = link_json.at("from_node").get<int>();
            link.to_node = link_json.at("to_node").get<int>();
            if (!ResolveSavedGraphLinkPins(
                    link_json,
                    find_loaded_node(link.from_node),
                    find_loaded_node(link.to_node),
                    preserve_legacy_data_validator_outputs,
                    preserve_legacy_evaluation_table_inputs,
                    preserve_legacy_classical_tree_pins,
                    link)) {
                continue;
            }
            if (link_json.contains("link_type")) {
                link.type = static_cast<LinkType>(link_json["link_type"].get<int>());
            }
            max_link_id = std::max(max_link_id, link.id);
            loaded_links.push_back(link);
        }

        const int loaded_next_pin_id = next_pin_id_;
        counters.Restore();

        // Commit only after the complete replacement graph has been built.
        ClearGraph();
        nodes_ = std::move(loaded_nodes);
        links_ = std::move(loaded_links);
        groups_ = std::move(loaded_groups);
        annotations_ = std::move(loaded_annotations);
        pending_positions_ = std::move(loaded_positions);
        pending_positions_frames_ = pending_positions_.empty() ? 0 : 3;
        cached_node_positions_.clear();
        next_node_id_ = max_node_id + 1;
        next_pin_id_ = loaded_next_pin_id;
        next_link_id_ = max_link_id + 1;
        next_group_id_ = loaded_next_group_id;
        next_annotation_id_ = loaded_next_annotation_id;
        selected_framework_ = loaded_framework;
        execution_mode_ = loaded_execution_mode;
        SetWorkflowDescription(loaded_workflow_description);
        RebuildPinLookup();

        spdlog::info(
            "Graph loaded from {} ({} nodes, {} links, execution mode {})",
            source_description,
            nodes_.size(),
            links_.size(),
            static_cast<int>(execution_mode_));
        return true;
    } catch (const std::exception& error) {
        spdlog::error("Error loading graph from {}: {}",
                      source_description,
                      error.what());
        return false;
    }
}

bool NodeEditor::LoadGraph(const std::string& filepath) {
    using json = nlohmann::json;

    spdlog::info("Loading graph from: {}", filepath);
    try {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            spdlog::error("Failed to open file for reading: {}", filepath);
            return false;
        }

        json graph_json;
        file >> graph_json;

        if (graph_json.contains("template") &&
            graph_json["template"].is_object() &&
            graph_json["template"].contains("nodes")) {
            spdlog::info("Detected pattern template format, converting to graph format");
            return LoadPatternAsGraph(graph_json);
        }

        if (!LoadGraphJson(graph_json, filepath)) {
            return false;
        }
        current_file_path_ = filepath;
        return true;
    } catch (const std::exception& error) {
        spdlog::error("Error loading graph from {}: {}", filepath, error.what());
        return false;
    }
}

std::string NodeEditor::GetGraphJson() const {
    using json = nlohmann::json;

    try {
        json j;
        j["version"] = "1.0";
        j["data_boundary_version"] = HasLegacyDataBoundary()
            ? detail::kLegacyDataBoundaryVersion
            : detail::kCurrentDataBoundaryVersion;
        j["data_validator_contract_version"] =
            detail::kCurrentDataValidatorContractVersion;
        j["evaluation_table_contract_version"] =
            detail::kCurrentEvaluationTableContractVersion;
        j["classical_tree_table_contract_version"] =
            detail::kCurrentClassicalTreeTableContractVersion;
        j["framework"] = static_cast<int>(selected_framework_);

        // Serialize nodes
        json nodes_array = json::array();
        for (const auto& node : nodes_) {
            json node_json;
            node_json["id"] = node.id;
            node_json["type"] = static_cast<int>(node.type);
            node_json["name"] = node.name;
            node_json["description"] = node.description;
            node_json["parameters"] = node.parameters;

            // Save node position
            auto it = cached_node_positions_.find(node.id);
            ImVec2 pos = (it != cached_node_positions_.end()) ? it->second : ImVec2(0,0);
            node_json["pos_x"] = pos.x;
            node_json["pos_y"] = pos.y;

            nodes_array.push_back(node_json);
        }
        j["nodes"] = nodes_array;

        // Serialize links with pin indices for multi-pin support
        json links_array = json::array();
        for (const auto& link : links_) {
            json link_json;
            link_json["id"] = link.id;
            link_json["from_node"] = link.from_node;
            link_json["from_pin"] = link.from_pin;
            link_json["to_node"] = link.to_node;
            link_json["to_pin"] = link.to_pin;

            // Save pin indices for proper multi-pin node support
            const MLNode* from_node = FindNodeById(link.from_node);
            const MLNode* to_node = FindNodeById(link.to_node);

            int from_pin_index = 0;
            if (from_node) {
                for (size_t i = 0; i < from_node->outputs.size(); ++i) {
                    if (from_node->outputs[i].id == link.from_pin) {
                        from_pin_index = static_cast<int>(i);
                        break;
                    }
                }
            }

            int to_pin_index = 0;
            if (to_node) {
                for (size_t i = 0; i < to_node->inputs.size(); ++i) {
                    if (to_node->inputs[i].id == link.to_pin) {
                        to_pin_index = static_cast<int>(i);
                        break;
                    }
                }
            }

            link_json["from_pin_index"] = from_pin_index;
            link_json["to_pin_index"] = to_pin_index;

            // Save link type for skip connection visualization
            link_json["link_type"] = static_cast<int>(link.type);

            links_array.push_back(link_json);
        }
        j["links"] = links_array;

        return j.dump(4);  // Pretty print with 4-space indent

    } catch (const std::exception& e) {
        spdlog::error("Error serializing graph: {}", e.what());
        return "";
    }
}

bool NodeEditor::LoadGraphFromString(const std::string& json_string) {
    using json = nlohmann::json;

    if (json_string.empty()) {
        spdlog::error("Cannot load graph from empty JSON string");
        return false;
    }

    try {
        return LoadGraphJson(json::parse(json_string), "JSON string");
    } catch (const std::exception& error) {
        spdlog::error("Error parsing graph JSON string: {}", error.what());
        return false;
    }
}

// ========== Cross-Platform File Dialogs ==========

void NodeEditor::ShowSaveDialog() {
    auto& project = cyxwiz::ProjectManager::Instance();
    const std::string default_path =
        project.HasActiveProject() ? project.GetCyxGraphsPath() : std::string();
    auto result = cyxwiz::FileDialogs::SaveGraph(
        default_path.empty() ? nullptr : default_path.c_str());
    if (result) {
        if (SaveGraph(*result)) {
            spdlog::info("Graph successfully saved");
        }
    }
}

void NodeEditor::ShowLoadDialog() {
    auto& project = cyxwiz::ProjectManager::Instance();
    const std::string default_path =
        project.HasActiveProject() ? project.GetCyxGraphsPath() : std::string();
    auto result = cyxwiz::FileDialogs::OpenGraph(
        default_path.empty() ? nullptr : default_path.c_str());
    if (result) {
        if (LoadGraph(*result)) {
            spdlog::info("Graph successfully loaded");
        }
    }
}

// ========== Code Export Implementation ==========

void NodeEditor::ExportCodeToFile() {
    // Validate graph first
    std::string error_message;
    if (!ValidateGraph(error_message)) {
        spdlog::error("Cannot export code: {}", error_message);
        // TODO: Show error dialog to user
        return;
    }

    // Generate code
    auto sorted_ids = TopologicalSort();
    if (sorted_ids.empty()) {
        spdlog::error("Failed to sort graph for code generation");
        return;
    }

    std::string code;
    std::string extension = ".py";
    std::string framework_name;

    switch (selected_framework_) {
        case CodeFramework::PyTorch:
            code = GeneratePyTorchCode(sorted_ids);
            framework_name = "PyTorch";
            break;
        case CodeFramework::TensorFlow:
            code = GenerateTensorFlowCode(sorted_ids);
            framework_name = "TensorFlow";
            break;
        case CodeFramework::Keras:
            code = GenerateKerasCode(sorted_ids);
            framework_name = "Keras";
            break;
        case CodeFramework::PyCyxWiz:
            code = GeneratePyCyxWizCode(sorted_ids);
            framework_name = "PyCyxWiz";
            break;
    }

    // Build the code with header and footer
    std::string header = "# Neural Network Model Generated by CyxWiz\n";
    header += "# Framework: " + framework_name + "\n";
    header += "# Generated on: " + std::string(__DATE__) + " " + std::string(__TIME__) + "\n\n";

    std::string full_code = header + code;

    // Save to file - will be called from ShowExportDialog
    return;
}

void NodeEditor::ShowExportDialog() {
    // Validate graph first
    std::string error_message;
    if (!ValidateGraph(error_message)) {
        spdlog::error("Cannot export code: {}", error_message);
        return;
    }

    // Generate code
    auto sorted_ids = TopologicalSort();
    if (sorted_ids.empty()) {
        spdlog::error("Failed to sort graph for code generation");
        return;
    }

    std::string code;
    std::string framework_name;

    switch (selected_framework_) {
        case CodeFramework::PyTorch:
            code = GeneratePyTorchCode(sorted_ids);
            framework_name = "PyTorch";
            break;
        case CodeFramework::TensorFlow:
            code = GenerateTensorFlowCode(sorted_ids);
            framework_name = "TensorFlow";
            break;
        case CodeFramework::Keras:
            code = GenerateKerasCode(sorted_ids);
            framework_name = "Keras";
            break;
        case CodeFramework::PyCyxWiz:
            code = GeneratePyCyxWizCode(sorted_ids);
            framework_name = "PyCyxWiz";
            break;
    }

    // Build the code with header
    std::string header = "# Neural Network Model Generated by CyxWiz\n";
    header += "# Framework: " + framework_name + "\n";
    header += "# Generated on: " + std::string(__DATE__) + " " + std::string(__TIME__) + "\n\n";

    std::string full_code = header + code;

    // Default filename based on framework
    std::string default_name = "model_" + framework_name + ".py";

    // Show cross-platform save dialog
    auto result = cyxwiz::FileDialogs::SaveScript();
    if (result) {
        std::ofstream file(*result);
        if (file.is_open()) {
            file << full_code;
            file.close();
            spdlog::info("Code exported successfully to: {}", *result);
        } else {
            spdlog::error("Failed to open file for writing: {}", *result);
        }
    }
}

} // namespace gui
