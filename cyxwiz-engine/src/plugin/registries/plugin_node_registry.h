#pragma once

#include "../interfaces/i_node_provider.h"
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <functional>
#include <optional>

namespace cyxwiz::plugin {

class PluginNodeRegistry {
public:
    static PluginNodeRegistry& Instance();

    PluginNodeRegistry(const PluginNodeRegistry&) = delete;
    PluginNodeRegistry& operator=(const PluginNodeRegistry&) = delete;

    // Register all node types from a provider
    void Register(const std::string& plugin_id, INodeProvider* provider);

    // Remove all registrations for a plugin
    void RemoveByPlugin(const std::string& plugin_id);

    // Query
    std::vector<PluginNodeTypeInfo> GetAllNodeTypes() const;
    // Returns (qualified_name, info) pairs for search catalog integration
    std::vector<std::pair<std::string, PluginNodeTypeInfo>> GetAllNodeTypesWithNames() const;
    bool HasNodeType(const std::string& qualified_name) const;   // "plugin_id:type_name"
    const PluginNodeTypeInfo* GetNodeTypeInfo(const std::string& qualified_name) const;
    std::optional<PluginNodeTypeInfo> GetNodeTypeInfoCopy(const std::string& qualified_name) const;

    // Code generation
    std::string GenerateCode(
        const std::string& qualified_name,
        const std::map<std::string, std::string>& parameters,
        const std::string& framework
    ) const;

    size_t GetCount() const;

private:
    PluginNodeRegistry() = default;

    struct NodeEntry {
        std::string plugin_id;
        PluginNodeTypeInfo info;
        INodeProvider* provider = nullptr;   // non-owning
    };

    mutable std::mutex mutex_;
    std::map<std::string, NodeEntry> nodes_;  // qualified_name -> entry
};

} // namespace cyxwiz::plugin
