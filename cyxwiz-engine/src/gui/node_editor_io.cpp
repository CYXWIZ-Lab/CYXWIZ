// File I/O Module for Node Editor
// This module contains all file operations for the visual node editor:
// - Graph save/load functionality (JSON serialization)
// - Cross-platform native file dialogs
// - Code export functionality (PyTorch, TensorFlow, Keras, PyCyxWiz)

#include "node_editor.h"
#include "../core/graph_document.h"
#include "../core/graph_node_factory.h"
#include "subgraph_document.h"
#include "../core/data_input_parameters.h"
#include "../core/pipeline_runtime_capabilities.h"
#include "node_import_guardrails.h"
#include "patterns/pattern_library.h"
#include "../core/file_dialogs.h"
#include "../core/project_manager.h"
#include "../core/project_data_path.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <unordered_map>
#include <spdlog/spdlog.h>

namespace gui {

void NodeEditor::DetachFromFile() {
    current_file_path_.clear();
    saved_fingerprint_ = 0;
    fingerprint_baseline_frames_ = 0;
    unsaved_ = true;
}

void NodeEditor::DetachFromExample() {
    const std::string& project_root = cyxwiz::ProjectManager::Instance().GetProjectRoot();
    for (auto& node : nodes_) {
        if (node.type != NodeType::DataInput) continue;
        for (const char* key : {"file_path", "folder_path", "labels_csv", "audio_labels_csv"}) {
            const auto it = node.parameters.find(key);
            if (it == node.parameters.end() || it->second.empty() || std::filesystem::path(it->second).is_absolute())
                continue;
            const std::string resolved = cyxwiz::ResolveProjectDataPath(it->second, project_root);
            std::error_code ec;
            if (!std::filesystem::exists(resolved, ec)) continue;  // the Data Input says it is missing
            it->second = cyxwiz::MakeProjectRelativePath(
                std::filesystem::absolute(resolved, ec).lexically_normal().generic_string(), project_root);
        }
    }
    DetachFromFile();
}

bool NodeEditor::LoadPatternAsGraph(const std::string& filepath) {
    // The Pattern Browser's instantiation: parameter defaults (in node types
    // too), every node type and link checked first, a pattern that is not
    // ready refused, the user's node defaults applied.
    auto& library = patterns::PatternLibrary::Instance();
    patterns::Pattern pattern;
    std::vector<MLNode> nodes;
    std::vector<NodeLink> links;
    int next_node_id = 1;
    int next_link_id = 1;
    std::string error;
    const auto create = [this](NodeType type, const std::string& name) {
        MLNode node = CreateNode(type, name);
        ApplyUserNodeDefaults(node);
        return node;
    };
    if (!library.ReadPatternFile(filepath, pattern, &error) ||
        !library.InstantiatePatternWithCreator(pattern, {}, nodes, links, next_node_id, next_link_id, ImVec2(0, 0),
                                               create, &error)) {
        spdlog::error("Cannot open pattern {} as a graph: {}", filepath, error);
        return false;
    }
    if (!CanReplaceGraph("load a graph")) return false;
    ClearGraph();
    // A template is a new graph, not the file it came from.
    DetachFromFile();
    InsertPattern(nodes, links);
    ForgetUndoHistory();
    spdlog::info("Loaded pattern '{}' as graph ({} nodes, {} links)", pattern.name, nodes_.size(), links_.size());
    return true;
}

// ========== Graph Save/Load Implementation ==========

void NodeEditor::RebuildDataBoundaryPins(
    MLNode& node,
    bool legacy_contract) {
    gui::RebuildDataBoundaryPins(node, legacy_contract, next_pin_id_);
}

bool NodeEditor::HasLegacyDataBoundary() const {
    for (const auto& data : subgraphs_) for (const auto& node : data.internal_nodes) {
        const auto it = node.parameters.find("data_boundary_pin_contract");
        if (it != node.parameters.end() && it->second == "legacy.v1") return true;
    }
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
    if (!subgraphs_.empty()) {
        result.message = "Migrate data boundaries before encapsulating nodes in subgraphs.";
        return result;
    }
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

// The graph as it is saved (also the unsaved-changes fingerprint).
nlohmann::json NodeEditor::GraphDocumentJson() {
    using json = nlohmann::json;
    json j = detail::CreateSerializedGraphDocument("2.1", HasLegacyDataBoundary());
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

    detail::WriteEditorGraphContent(j, nodes_, links_, subgraphs_, cached_node_positions_);
    return j;
}

bool NodeEditor::SaveGraph(const std::string& filepath) {
    using json = nlohmann::json;

    try {
        json j = GraphDocumentJson();

        // Write to file
        std::ofstream file(filepath);
        if (!file.is_open()) {
            spdlog::error("Failed to open file for writing: {}", filepath);
            return false;
        }

        file << j.dump(4);  // Pretty print with 4-space indent
        current_file_path_ = filepath;
        saved_fingerprint_ = FingerprintOf(j);
        unsaved_ = false;
        cyxwiz::SetGraphDataSearchDirectoryFromGraphFile(filepath);
        spdlog::info("Graph saved to: {}", filepath);
        return true;

    } catch (const std::exception& e) {
        spdlog::error("Error saving graph: {}", e.what());
        return false;
    }
}

bool NodeEditor::LoadGraphJson(const nlohmann::json& graph_json,
                               const std::string& source_description) {
    if (!CanReplaceGraph("replace the graph")) return false;
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
        const auto flat_graph = detail::FlattenSubgraphDocument(graph_json);
        // CreateNode owns the pin contract, so use it to build a complete
        // replacement graph without touching the live node/link containers.
        next_node_id_ = 1;
        next_pin_id_ = 1;

        std::map<int, ImVec2> loaded_positions;
        std::vector<NodeGroup> loaded_groups;
        std::vector<CanvasAnnotation> loaded_annotations;
        int loaded_next_group_id = 1;
        int loaded_next_annotation_id = 1;

        cyxwiz::CodeFramework loaded_framework = selected_framework_;
        if (graph_json.contains("framework")) {
            loaded_framework = static_cast<cyxwiz::CodeFramework>(
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
                    "color", DefaultAnnotationColour());
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

        cyxwiz::GraphDocumentLoadOptions load_options;
        load_options.restore_extra_pins = [&graph_json](std::vector<MLNode>& nodes, int& next_pin_id) {
            detail::RestoreSubgraphPins(graph_json, nodes, next_pin_id);
        };
        load_options.strict_links = graph_json.contains("subgraphs");
        cyxwiz::GraphDocument document;
        std::string load_error;
        if (!cyxwiz::BuildGraphDocument(graph_json, flat_graph, load_options, document, load_error)) {
            spdlog::error("Cannot load graph from {}: {}", source_description, load_error);
            return false;
        }
        std::vector<MLNode> loaded_nodes = std::move(document.nodes);
        std::vector<NodeLink> loaded_links = std::move(document.links);
        next_pin_id_ = document.next_pin_id;
        const int max_node_id = document.next_node_id - 1;
        const int max_link_id = document.next_link_id - 1;
        for (const auto& node_json : flat_graph["nodes"]) {
            if (!node_json.contains("pos_x") || !node_json.contains("pos_y")) continue;
            const int node_id = node_json.at("id").get<int>();
            loaded_positions[node_id] = ImVec2(
                node_json["pos_x"].get<float>(),
                node_json["pos_y"].get<float>());
        }
        for (auto& node : loaded_nodes) {
            const auto pos = loaded_positions.find(node.id);
            if (pos == loaded_positions.end()) continue;
            node.initial_pos_x = pos->second.x;
            node.initial_pos_y = pos->second.y;
            node.has_initial_position = true;
        }

        auto loaded_subgraphs = detail::RestoreSubgraphContents(graph_json, loaded_nodes, loaded_links);

        const int loaded_next_pin_id = next_pin_id_;
        counters.Restore();

        // Commit only after the complete replacement graph has been built.
        if (!CanReplaceGraph("replace the graph")) return false;
        ClearGraph();
        nodes_ = std::move(loaded_nodes);
        subgraphs_ = std::move(loaded_subgraphs);
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
        ForgetUndoHistory();

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
    if (!CanReplaceGraph("load a graph")) return false;
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
            // A template example names its data relative to its own folder
            // (or a parent, e.g. examples/datasets/...), as a saved graph does.
            cyxwiz::SetGraphDataSearchDirectoryFromGraphFile(filepath);
            return LoadPatternAsGraph(filepath);
        }

        // Before the nodes load: their data inputs resolve relative paths
        // against this graph's folder when no project holds them.
        cyxwiz::SetGraphDataSearchDirectoryFromGraphFile(filepath);
        if (!LoadGraphJson(graph_json, filepath)) {
            return false;
        }
        fingerprint_baseline_frames_ = 3;
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
        json j = detail::CreateSerializedGraphDocument("1.0", HasLegacyDataBoundary());
        j["framework"] = static_cast<int>(selected_framework_);
        j["execution_mode"] = static_cast<int>(execution_mode_);

        detail::WriteEditorGraphContent(j, nodes_, links_, subgraphs_, cached_node_positions_);

        return j.dump(4);  // Pretty print with 4-space indent

    } catch (const std::exception& e) {
        spdlog::error("Error serializing graph: {}", e.what());
        return "";
    }
}

bool NodeEditor::LoadGraphFromString(const std::string& json_string) {
    if (!CanReplaceGraph("load a graph")) return false;
    using json = nlohmann::json;

    if (json_string.empty()) {
        spdlog::error("Cannot load graph from empty JSON string");
        return false;
    }

    try {
        // An imported graph is new: Save asks for a file name.
        const bool loaded = LoadGraphJson(json::parse(json_string), "JSON string");
        if (loaded) DetachFromFile();
        return loaded;
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

void NodeEditor::SaveFromStudio() {
    if (nodes_.empty()) {
        save_refusal_pending_ = true;
        return;
    }
    SaveCurrentGraph();
}

void NodeEditor::SaveAsFromStudio() {
    if (nodes_.empty()) {
        save_refusal_pending_ = true;
        return;
    }
    ShowSaveDialog();
}

std::string NodeEditor::GraphDisplayName() const {
    return current_file_path_.empty() ? std::string("Untitled graph")
                                      : std::filesystem::path(current_file_path_).filename().string();
}

bool NodeEditor::SaveCurrentGraph() {
    if (nodes_.empty()) return true;
    if (current_file_path_.empty()) {
        ShowSaveDialog();
        return !current_file_path_.empty();
    }
    return SaveGraph(current_file_path_);
}

void NodeEditor::ShowLoadDialog() {
    if (!CanReplaceGraph("load a graph")) return;
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

void NodeEditor::ExportGraphCopy() {
    if (nodes_.empty()) return;
    auto& project = cyxwiz::ProjectManager::Instance();
    const std::string start = project.HasActiveProject() ? project.GetCyxGraphsPath() : std::string();
    const std::string name = GetGraphStem() + " copy.cyxgraph";
    const auto path = cyxwiz::FileDialogs::SaveFile("Export a copy of the graph", {{"CyxWiz Graphs", "cyxgraph"}},
                                                    start.empty() ? nullptr : start.c_str(), name.c_str());
    if (!path) return;
    try {
        std::ofstream file(*path, std::ios::binary | std::ios::trunc);
        if (!file || !(file << GraphDocumentJson().dump(4))) {
            spdlog::error("Could not write the copy of the graph to {}", *path);
            return;
        }
        // The open graph stays on its own file; the copy is not the graph's file.
        spdlog::info("Wrote a copy of the graph to {}", *path);
    } catch (const std::exception& e) {
        spdlog::error("Could not write the copy of the graph to {}: {}", *path, e.what());
    }
}

} // namespace gui
