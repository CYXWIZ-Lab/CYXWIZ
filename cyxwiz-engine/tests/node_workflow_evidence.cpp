#include "node_workflow_evidence.h"

#include "../src/core/node_metadata.h"
#include "../src/gui/node_type_import_registry.h"

#include <exception>
#include <fstream>
#include <limits>
#include <set>
#include <string_view>
#include <utility>

#include <nlohmann/json.hpp>

namespace cyxwiz::test {
namespace {

using Json = nlohmann::json;

bool ReadJson(const std::filesystem::path& path,
              Json& document,
              std::string& error) {
    std::ifstream input(path, std::ios::in | std::ios::binary);
    if (!input) {
        error = "cannot open " + path.generic_string();
        return false;
    }
    try {
        input >> document;
        return true;
    } catch (const std::exception& exception) {
        error = "cannot parse " + path.generic_string() + ": " +
                exception.what();
        return false;
    }
}

bool IsSafeRepositoryRelativePath(const std::filesystem::path& path) {
    if (path.empty() || path.is_absolute()) return false;
    for (const auto& component : path) {
        if (component == "..") return false;
    }
    return true;
}

bool ParseValidationLevel(std::string_view value,
                          WorkflowValidationLevel& level) {
    if (value == "authored_graph_only") {
        level = WorkflowValidationLevel::AuthoredGraphOnly;
        return true;
    }
    if (value == "automated_contract") {
        level = WorkflowValidationLevel::AutomatedContract;
        return true;
    }
    if (value == "live_release") {
        level = WorkflowValidationLevel::LiveRelease;
        return true;
    }
    return false;
}

const Json* FindGraphNodes(const Json& graph) {
    const auto nodes = graph.find("nodes");
    if (nodes != graph.end() && nodes->is_array()) return &*nodes;
    const auto graph_template = graph.find("template");
    if (graph_template == graph.end() || !graph_template->is_object()) {
        return nullptr;
    }
    const auto template_nodes = graph_template->find("nodes");
    return template_nodes != graph_template->end() &&
                   template_nodes->is_array()
               ? &*template_nodes
               : nullptr;
}

std::string RequiredString(const Json& object,
                           std::string_view field,
                           std::vector<std::string>& errors,
                           std::string_view context) {
    const auto found = object.find(std::string(field));
    if (found == object.end() || !found->is_string() ||
        found->get_ref<const std::string&>().empty()) {
        errors.push_back(std::string(context) + " has no non-empty " +
                         std::string(field));
        return {};
    }
    return found->get<std::string>();
}

} // namespace

const char* WorkflowValidationLevelName(WorkflowValidationLevel level) {
    switch (level) {
        case WorkflowValidationLevel::AuthoredGraphOnly:
            return "authored_graph_only";
        case WorkflowValidationLevel::AutomatedContract:
            return "automated_contract";
        case WorkflowValidationLevel::LiveRelease:
            return "live_release";
    }
    return "unknown";
}

NodeWorkflowEvidenceCatalog LoadNodeWorkflowEvidenceCatalog(
    const std::filesystem::path& repository_root,
    const std::filesystem::path& manifest_path,
    const std::vector<const NodeMetadata*>& registered_nodes) {
    NodeWorkflowEvidenceCatalog catalog;
    std::set<int> registered_identities;
    for (const auto* metadata : registered_nodes) {
        if (metadata != nullptr) {
            registered_identities.insert(static_cast<int>(metadata->type));
        }
    }

    Json manifest;
    std::string read_error;
    if (!ReadJson(manifest_path, manifest, read_error)) {
        catalog.errors.push_back(std::move(read_error));
        return catalog;
    }
    const auto schema_version = manifest.find("schema_version");
    if (!manifest.is_object() || schema_version == manifest.end() ||
        !schema_version->is_number_integer() ||
        schema_version->get<int>() != 1) {
        catalog.errors.push_back(
            "workflow evidence manifest must use schema_version 1");
        return catalog;
    }
    const auto workflows = manifest.find("workflows");
    if (workflows == manifest.end() || !workflows->is_array()) {
        catalog.errors.push_back(
            "workflow evidence manifest has no workflows array");
        return catalog;
    }

    std::set<std::string> workflow_ids;
    std::set<std::string> workflow_paths;
    for (std::size_t manifest_index = 0;
         manifest_index < workflows->size(); ++manifest_index) {
        const Json& entry = (*workflows)[manifest_index];
        const std::string context =
            "workflow[" + std::to_string(manifest_index) + "]";
        if (!entry.is_object()) {
            catalog.errors.push_back(context + " is not an object");
            continue;
        }

        NodeWorkflowArtifactReference artifact;
        artifact.id = RequiredString(entry, "id", catalog.errors, context);
        artifact.relative_path =
            RequiredString(entry, "path", catalog.errors, context);
        artifact.workflow_lane =
            RequiredString(entry, "workflow_lane", catalog.errors, context);
        artifact.objective =
            RequiredString(entry, "objective", catalog.errors, context);
        const std::string validation =
            RequiredString(entry, "validation_level", catalog.errors, context);
        if (artifact.id.empty() || artifact.relative_path.empty() ||
            artifact.workflow_lane.empty() || artifact.objective.empty() ||
            validation.empty()) {
            continue;
        }
        if (!workflow_ids.insert(artifact.id).second) {
            catalog.errors.push_back("duplicate workflow evidence id: " +
                                     artifact.id);
            continue;
        }
        if (!workflow_paths.insert(artifact.relative_path).second) {
            catalog.errors.push_back("duplicate workflow evidence path: " +
                                     artifact.relative_path);
            continue;
        }
        if (!ParseValidationLevel(validation, artifact.validation_level)) {
            catalog.errors.push_back("unsupported validation level for " +
                                     artifact.id + ": " + validation);
            continue;
        }

        const std::filesystem::path relative_path(artifact.relative_path);
        if (!IsSafeRepositoryRelativePath(relative_path) ||
            relative_path.extension() != ".cyxgraph") {
            catalog.errors.push_back("unsafe or non-cyxgraph workflow path: " +
                                     artifact.relative_path);
            continue;
        }
        const std::filesystem::path graph_path =
            (repository_root / relative_path).lexically_normal();
        Json graph;
        if (!ReadJson(graph_path, graph, read_error)) {
            catalog.errors.push_back(std::move(read_error));
            continue;
        }
        const Json* graph_nodes = FindGraphNodes(graph);
        if (graph_nodes == nullptr || graph_nodes->empty()) {
            catalog.errors.push_back("workflow has no graph nodes: " +
                                     artifact.relative_path);
            continue;
        }
        const auto graph_name = graph.find("name");
        artifact.title = graph_name != graph.end() && graph_name->is_string() &&
                                 !graph_name->get_ref<const std::string&>().empty()
                             ? graph_name->get<std::string>()
                             : artifact.id;

        std::set<int> artifact_node_identities;
        for (std::size_t node_index = 0; node_index < graph_nodes->size();
             ++node_index) {
            const Json& node = (*graph_nodes)[node_index];
            const std::string node_context = artifact.relative_path +
                " node[" + std::to_string(node_index) + "]";
            if (!node.is_object() || !node.contains("type")) {
                catalog.errors.push_back(node_context + " has no type");
                continue;
            }

            int identity = std::numeric_limits<int>::min();
            const Json& type = node["type"];
            if (type.is_number_unsigned()) {
                const auto numeric_identity =
                    type.get<unsigned long long>();
                if (numeric_identity >
                    static_cast<unsigned long long>(
                        std::numeric_limits<int>::max())) {
                    catalog.errors.push_back(node_context +
                                             " type is outside int range");
                    continue;
                }
                identity = static_cast<int>(numeric_identity);
            } else if (type.is_number_integer()) {
                const auto numeric_identity = type.get<long long>();
                if (numeric_identity < std::numeric_limits<int>::min() ||
                    numeric_identity > std::numeric_limits<int>::max()) {
                    catalog.errors.push_back(node_context +
                                             " type is outside int range");
                    continue;
                }
                identity = static_cast<int>(numeric_identity);
            } else if (type.is_string()) {
                const std::string import_name = type.get<std::string>();
                const auto resolved =
                    gui::ResolveNodeTypeImportName(import_name);
                if (!resolved.has_value()) {
                    catalog.errors.push_back(node_context +
                                             " uses unresolved import name " +
                                             import_name);
                    continue;
                }
                identity = static_cast<int>(*resolved);
            } else {
                catalog.errors.push_back(node_context +
                                         " type is not an integer or string");
                continue;
            }

            if (registered_identities.count(identity) == 0) {
                catalog.errors.push_back(
                    node_context + " resolves to unregistered identity " +
                    std::to_string(identity));
                continue;
            }
            artifact_node_identities.insert(identity);
        }
        if (artifact_node_identities.empty()) {
            catalog.errors.push_back("workflow resolves no registered nodes: " +
                                     artifact.relative_path);
            continue;
        }

        const std::size_t artifact_index = catalog.artifacts.size();
        catalog.artifacts.push_back(std::move(artifact));
        for (const int identity : artifact_node_identities) {
            catalog.artifact_indices_by_node[identity].push_back(
                artifact_index);
        }
    }

    return catalog;
}

} // namespace cyxwiz::test
