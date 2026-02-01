#include "plugin_node_registry.h"
#include <spdlog/spdlog.h>

namespace cyxwiz::plugin {

PluginNodeRegistry& PluginNodeRegistry::Instance() {
    static PluginNodeRegistry instance;
    return instance;
}

void PluginNodeRegistry::Register(const std::string& plugin_id, INodeProvider* provider) {
    if (!provider) return;

    // NOTE: This calls GetNodeTypes() across the DLL boundary, returning a
    // DLL-allocated std::vector. Prefer RegisterDirect() from plugin_manager.cpp
    // where we can deep-copy strings before the DLL vector is destroyed.
    auto types = provider->GetNodeTypes();
    std::lock_guard lock(mutex_);

    for (auto& info : types) {
        std::string qname = plugin_id + ":" + info.type_name;
        if (nodes_.count(qname)) {
            spdlog::warn("PluginNodeRegistry: Duplicate node type {}, skipping", qname);
            continue;
        }
        spdlog::info("PluginNodeRegistry: Registered node {} ({})", info.display_name, qname);
        nodes_[qname] = NodeEntry{plugin_id, std::move(info), provider};
    }
}

void PluginNodeRegistry::RegisterDirect(const std::string& plugin_id,
                                         PluginNodeTypeInfo&& info,
                                         INodeProvider* provider) {
    std::lock_guard lock(mutex_);
    std::string qname = plugin_id + ":" + info.type_name;
    if (nodes_.count(qname)) {
        spdlog::warn("PluginNodeRegistry: Duplicate node type {}, skipping", qname);
        return;
    }
    spdlog::info("PluginNodeRegistry: Registered node {} ({})", info.display_name, qname);
    nodes_[qname] = NodeEntry{plugin_id, std::move(info), provider};
}

void PluginNodeRegistry::RemoveByPlugin(const std::string& plugin_id) {
    std::lock_guard lock(mutex_);
    for (auto it = nodes_.begin(); it != nodes_.end();) {
        if (it->second.plugin_id == plugin_id)
            it = nodes_.erase(it);
        else
            ++it;
    }
}

std::vector<PluginNodeTypeInfo> PluginNodeRegistry::GetAllNodeTypes() const {
    std::lock_guard lock(mutex_);
    std::vector<PluginNodeTypeInfo> result;
    result.reserve(nodes_.size());
    for (const auto& [_, entry] : nodes_)
        result.push_back(entry.info);
    return result;
}

std::vector<std::pair<std::string, PluginNodeTypeInfo>> PluginNodeRegistry::GetAllNodeTypesWithNames() const {
    std::lock_guard lock(mutex_);
    std::vector<std::pair<std::string, PluginNodeTypeInfo>> result;
    result.reserve(nodes_.size());
    for (const auto& [qname, entry] : nodes_)
        result.emplace_back(qname, entry.info);
    return result;
}

bool PluginNodeRegistry::HasNodeType(const std::string& qualified_name) const {
    std::lock_guard lock(mutex_);
    return nodes_.count(qualified_name) > 0;
}

const PluginNodeTypeInfo* PluginNodeRegistry::GetNodeTypeInfo(const std::string& qualified_name) const {
    std::lock_guard lock(mutex_);
    auto it = nodes_.find(qualified_name);
    return it != nodes_.end() ? &it->second.info : nullptr;
}

std::optional<PluginNodeTypeInfo> PluginNodeRegistry::GetNodeTypeInfoCopy(const std::string& qualified_name) const {
    std::lock_guard lock(mutex_);
    auto it = nodes_.find(qualified_name);
    if (it != nodes_.end())
        return it->second.info;
    return std::nullopt;
}

DynamicPinResult PluginNodeRegistry::ResolveDynamicPins(
    const std::string& qualified_name,
    const std::map<std::string, std::string>& parameters
) const {
    std::lock_guard lock(mutex_);
    auto it = nodes_.find(qualified_name);
    if (it == nodes_.end() || !it->second.provider) return {};

    try {
        return it->second.provider->ResolveDynamicPins(it->second.info.type_name, parameters);
    } catch (const std::exception& e) {
        spdlog::error("PluginNodeRegistry: ResolveDynamicPins failed for {}: {}", qualified_name, e.what());
        return {};
    }
}

std::string PluginNodeRegistry::GenerateCode(
    const std::string& qualified_name,
    const std::map<std::string, std::string>& parameters,
    const std::string& framework
) const {
    std::lock_guard lock(mutex_);
    auto it = nodes_.find(qualified_name);
    if (it == nodes_.end() || !it->second.provider) return "";

    try {
        return it->second.provider->GenerateCode(it->second.info.type_name, parameters, framework);
    } catch (const std::exception& e) {
        spdlog::error("PluginNodeRegistry: Code generation failed for {}: {}", qualified_name, e.what());
        return "";
    }
}

size_t PluginNodeRegistry::GetCount() const {
    std::lock_guard lock(mutex_);
    return nodes_.size();
}

} // namespace cyxwiz::plugin
