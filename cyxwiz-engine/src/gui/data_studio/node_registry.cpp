#include "node_registry.h"
#include "../../core/pipeline_runtime_capabilities.h"
#include <spdlog/spdlog.h>
#include <algorithm>

namespace cyxwiz {

NodeRegistry& NodeRegistry::Instance() {
    static NodeRegistry instance;
    return instance;
}

NodeRegistry::NodeRegistry() {
    RegisterBuiltInNodes();
    spdlog::info("[Data Studio] NodeRegistry initialized with {} node types",
                 node_types_.size());
}

void NodeRegistry::RegisterNodeType(const NodeType& node_type) {
    const auto support = ResolvePipelineRuntimeSupport(node_type.type_id);
    if (!support.pipeline_executor_supported) {
        const char* reason = support.fail_closed_reason;
        spdlog::warn("[Data Studio] Skipping unsupported node type '{}': {}",
                     node_type.type_id,
                     reason != nullptr ? reason : "not supported by PipelineExecutor");
        return;
    }

    // Check for duplicates
    if (HasNodeType(node_type.type_id)) {
        spdlog::warn("[Data Studio] Node type already registered: {}", node_type.type_id);
        return;
    }

    node_types_.push_back(node_type);
    spdlog::debug("[Data Studio] Registered node type: {} ({})",
                  node_type.display_name, node_type.type_id);
}

const std::vector<NodeRegistry::NodeType>& NodeRegistry::GetAllNodeTypes() const {
    return node_types_;
}

std::vector<NodeRegistry::NodeType> NodeRegistry::GetNodeTypesByCategory(
    const std::string& category) const {

    std::vector<NodeType> result;
    for (const auto& node_type : node_types_) {
        if (node_type.category == category) {
            result.push_back(node_type);
        }
    }
    return result;
}

std::vector<std::string> NodeRegistry::GetCategories() const {
    std::vector<std::string> categories;
    for (const auto& node_type : node_types_) {
        if (std::find(categories.begin(), categories.end(), node_type.category) == categories.end()) {
            categories.push_back(node_type.category);
        }
    }
    return categories;
}

const NodeRegistry::NodeType* NodeRegistry::FindNodeType(const std::string& type_id) const {
    auto it = std::find_if(node_types_.begin(), node_types_.end(),
                          [&type_id](const NodeType& nt) { return nt.type_id == type_id; });

    return (it != node_types_.end()) ? &(*it) : nullptr;
}

bool NodeRegistry::HasNodeType(const std::string& type_id) const {
    return FindNodeType(type_id) != nullptr;
}

} // namespace cyxwiz
