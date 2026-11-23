#include "plugin_simulation_registry.h"

#include "../../core/extension_node_registry.h"
#include "../security/safe_execute.h"

#include <spdlog/spdlog.h>

#include <algorithm>

namespace cyxwiz::plugin {

SimulationRunLease::~SimulationRunLease() {
    for (auto it = begun_.rbegin(); it != begun_.rend(); ++it) {
        auto* provider = it->provider;
        const auto result = security::SafeExecute(it->plugin_id, "EndGraphRun", [provider]() { provider->EndGraphRun(); });
        if (!result.success) spdlog::warn("Plugin {} failed to end its graph run: {}", it->plugin_id, result.error_message);
    }
}

PluginSimulationRegistry& PluginSimulationRegistry::Instance() {
    static PluginSimulationRegistry instance;
    return instance;
}

void PluginSimulationRegistry::Register(const std::string& plugin_id, const PluginSimulationNodeInfo& info,
                                        ISimulationProvider* provider) {
    if (!provider || info.node_type.empty()) return;
    Registered entry;
    entry.plugin_id = plugin_id;
    entry.info.node_type = std::string(info.node_type.c_str());
    entry.info.panel_id = std::string(info.panel_id.c_str());
    entry.provider = provider;
    const std::string key = MakeExtensionTypeId(plugin_id, entry.info.node_type);
    std::lock_guard lock(mutex_);
    if (nodes_.count(key)) {
        spdlog::warn("PluginSimulationRegistry: Duplicate simulation node {}, skipping", key);
        return;
    }
    nodes_[key] = std::move(entry);
}

void PluginSimulationRegistry::RemoveByPlugin(const std::string& plugin_id) {
    std::lock_guard lock(mutex_);
    for (auto it = nodes_.begin(); it != nodes_.end();) {
        it = it->second.plugin_id == plugin_id ? nodes_.erase(it) : std::next(it);
    }
}

bool PluginSimulationRegistry::IsSimulationNode(const std::string& type_id) const {
    std::lock_guard lock(mutex_);
    return nodes_.count(type_id) > 0;
}

size_t PluginSimulationRegistry::GetCount() const {
    std::lock_guard lock(mutex_);
    return nodes_.size();
}

std::shared_ptr<SimulationRunLease> PluginSimulationRegistry::BeginGraphRun(const std::vector<::gui::MLNode>& nodes,
                                                                            std::string& reason) {
    struct Plugin {
        ISimulationProvider* provider = nullptr;
        std::vector<PluginSimulationGraphNode> nodes;
    };
    std::map<std::string, Plugin> plugins;
    std::shared_ptr<SimulationRunLease> lease(new SimulationRunLease());
    {
        std::lock_guard lock(mutex_);
        for (const auto& node : nodes) {
            if (node.type != ::gui::NodeType::PluginCustom) continue;
            const auto it = nodes_.find(node.extension_type_id);
            if (it == nodes_.end()) {
                reason = "'" + node.name +
                         "' cannot run in a graph simulation: its plugin is not loaded, or it does not run this node "
                         "in a simulation. Remove the node or load its plugin.";
                return nullptr;
            }
            auto& plugin = plugins[it->second.plugin_id];
            plugin.provider = it->second.provider;
            plugin.nodes.push_back({node.id, it->second.info.node_type, node.parameters});
            const auto panel = std::make_pair(it->second.plugin_id, it->second.info.panel_id);
            if (!panel.second.empty() && std::find(lease->panels_.begin(), lease->panels_.end(), panel) == lease->panels_.end())
                lease->panels_.push_back(panel);
        }
    }
    for (const auto& node : nodes) {
        if (node.type != ::gui::NodeType::PluginCustom) continue;
        if (auto library = ExtensionNodeRegistry::Instance().ProviderLease(node.extension_type_id))
            lease->library_leases_.push_back(std::move(library));
    }

    for (const auto& [plugin_id, plugin] : plugins) {
        PluginSimulationBeginResult begin;
        auto* provider = plugin.provider;
        const auto& graph_nodes = plugin.nodes;
        const auto result = security::SafeExecute(plugin_id, "BeginGraphRun", [&begin, provider, &graph_nodes]() {
            begin = provider->BeginGraphRun(graph_nodes);
        });
        if (!result.success) {
            reason = "The plugin " + plugin_id + " failed to start the run: " + result.error_message;
            return nullptr;  // the lease ends the plugins that had begun
        }
        if (!begin.ok) {
            reason = begin.reason.empty() ? "The plugin " + plugin_id + " cannot start the run."
                                          : std::string(begin.reason.c_str());
            return nullptr;
        }
        lease->begun_.push_back({plugin_id, provider});
        lease->plugin_ids_.insert(plugin_id);
    }
    return lease;
}

} // namespace cyxwiz::plugin
