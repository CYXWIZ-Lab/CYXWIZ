#pragma once

#include <cstddef>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace cyxwiz {
struct NodeMetadata;
}

namespace cyxwiz::test {

enum class WorkflowValidationLevel {
    AuthoredGraphOnly,
    AutomatedContract,
    LiveRelease,
};

struct NodeWorkflowArtifactReference {
    std::string id;
    std::string relative_path;
    std::string title;
    std::string workflow_lane;
    std::string objective;
    WorkflowValidationLevel validation_level =
        WorkflowValidationLevel::AuthoredGraphOnly;
};

struct NodeWorkflowEvidenceCatalog {
    std::vector<NodeWorkflowArtifactReference> artifacts;
    std::map<int, std::vector<std::size_t>> artifact_indices_by_node;
    std::vector<std::string> errors;
};

const char* WorkflowValidationLevelName(WorkflowValidationLevel level);

NodeWorkflowEvidenceCatalog LoadNodeWorkflowEvidenceCatalog(
    const std::filesystem::path& repository_root,
    const std::filesystem::path& manifest_path,
    const std::vector<const NodeMetadata*>& registered_nodes);

} // namespace cyxwiz::test
