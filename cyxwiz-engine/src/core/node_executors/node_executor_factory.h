#pragma once

#include "node_executor.h"
#include "kmeans_executor.h"
#include "../../gui/node_editor.h"  // For NodeType enum
#include <memory>
#include <unordered_map>
#include <set>
#include <vector>

namespace cyxwiz {

/**
 * NodeExecutorFactory - Creates and manages node executors
 *
 * Maps NodeType enum values to their corresponding executor implementations.
 * Caches executor instances for reuse.
 */
class NodeExecutorFactory {
public:
    static NodeExecutorFactory& Instance() {
        static NodeExecutorFactory instance;
        return instance;
    }

    /**
     * Get or create an executor for the given node type
     * Returns nullptr if no executor exists for this node type
     */
    INodeExecutor* GetExecutor(gui::NodeType type);

    /**
     * Check if an executor exists for the given node type
     */
    bool HasExecutor(gui::NodeType type) const;

    /**
     * Get list of all supported node types
     */
    const std::vector<gui::NodeType>& GetSupportedTypes() const;

    /**
     * Clear all cached executors
     */
    void ClearCache();

private:
    NodeExecutorFactory();

    void RegisterSupportedTypes();
    std::unique_ptr<INodeExecutor> CreateExecutor(gui::NodeType type);

    // Member variables
    std::unordered_map<gui::NodeType, std::unique_ptr<INodeExecutor>> executors_;
    std::set<gui::NodeType> supported_types_;
    std::vector<gui::NodeType> supported_type_list_;
};

// Inline implementations
inline NodeExecutorFactory::NodeExecutorFactory() {
    RegisterSupportedTypes();
}

inline INodeExecutor* NodeExecutorFactory::GetExecutor(gui::NodeType type) {
    auto it = executors_.find(type);
    if (it != executors_.end()) {
        return it->second.get();
    }

    // Create new executor
    auto executor = CreateExecutor(type);
    if (executor) {
        auto* ptr = executor.get();
        executors_[type] = std::move(executor);
        return ptr;
    }

    return nullptr;
}

inline bool NodeExecutorFactory::HasExecutor(gui::NodeType type) const {
    return supported_types_.count(type) > 0;
}

inline const std::vector<gui::NodeType>& NodeExecutorFactory::GetSupportedTypes() const {
    return supported_type_list_;
}

inline void NodeExecutorFactory::ClearCache() {
    executors_.clear();
}

inline void NodeExecutorFactory::RegisterSupportedTypes() {
    // Analytics/Clustering
    supported_types_.insert(gui::NodeType::KMeansCluster);
    // TODO: Add more as executors are implemented
    // supported_types_.insert(gui::NodeType::DBSCANCluster);
    // supported_types_.insert(gui::NodeType::HierarchicalCluster);
    // supported_types_.insert(gui::NodeType::PCANode);
    // supported_types_.insert(gui::NodeType::CorrelationMatrix);

    // Build list for iteration
    supported_type_list_.assign(supported_types_.begin(), supported_types_.end());
}

inline std::unique_ptr<INodeExecutor> NodeExecutorFactory::CreateExecutor(gui::NodeType type) {
    switch (type) {
        case gui::NodeType::KMeansCluster:
            return std::make_unique<KMeansExecutor>();

        // TODO: Add more executor types as they are implemented
        // case gui::NodeType::DBSCANCluster:
        //     return std::make_unique<DBSCANExecutor>();
        // case gui::NodeType::PCANode:
        //     return std::make_unique<PCAExecutor>();

        default:
            return nullptr;
    }
}

} // namespace cyxwiz
