#include "plugin_command_registry.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <exception>

namespace cyxwiz::plugin {

PluginCommandRegistry& PluginCommandRegistry::Instance() {
    static PluginCommandRegistry instance;
    return instance;
}

void PluginCommandRegistry::Register(const std::string& plugin_id, const PluginCommandInfo& info,
                                     ICommandProvider* provider) {
    if (!provider || info.command_id.empty() || info.label.empty()) return;
    const std::string key = plugin_id + ":" + std::string(info.command_id.c_str());
    std::lock_guard lock(mutex_);
    if (commands_.count(key)) {
        spdlog::warn("PluginCommandRegistry: Duplicate command {}, skipping", key);
        return;
    }
    Registered entry;
    entry.plugin_id = plugin_id;
    entry.info.command_id = std::string(info.command_id.c_str());
    entry.info.menu = std::string(info.menu.c_str());
    entry.info.group = std::string(info.group.c_str());
    entry.info.label = std::string(info.label.c_str());
    entry.info.hint = std::string(info.hint.c_str());
    entry.info.checkable = info.checkable;
    entry.info.replaced_by_graph_run = info.replaced_by_graph_run;
    entry.provider = provider;
    commands_[key] = std::move(entry);
    order_.push_back(key);
}

void PluginCommandRegistry::RemoveByPlugin(const std::string& plugin_id) {
    std::lock_guard lock(mutex_);
    for (auto it = commands_.begin(); it != commands_.end();) {
        if (it->second.plugin_id == plugin_id) {
            order_.erase(std::remove(order_.begin(), order_.end(), it->first), order_.end());
            it = commands_.erase(it);
        } else {
            ++it;
        }
    }
}

std::vector<PluginCommandRegistry::Entry> PluginCommandRegistry::Snapshot() const {
    std::lock_guard lock(mutex_);
    std::vector<Entry> out;
    out.reserve(order_.size());
    for (const auto& key : order_) {
        const auto& c = commands_.at(key);
        Entry e{key, c.info, {}};
        try {
            const PluginCommandState state = c.provider->GetCommandState(c.info.command_id);
            e.state.enabled = state.enabled;
            e.state.disabled_reason = std::string(state.disabled_reason.c_str());
            e.state.checked = state.checked;
        } catch (const std::exception& ex) {
            e.state.enabled = false;
            e.state.disabled_reason = std::string("The plugin failed to report this command: ") + ex.what();
        }
        out.push_back(std::move(e));
    }
    return out;
}

bool PluginCommandRegistry::Run(const std::string& key) {
    ICommandProvider* provider = nullptr;
    std::string command_id;
    {
        std::lock_guard lock(mutex_);
        const auto it = commands_.find(key);
        if (it == commands_.end()) return false;
        provider = it->second.provider;
        command_id = it->second.info.command_id;
    }
    // Run outside the lock: the command may load or unload things that
    // change the registry
    try {
        provider->RunCommand(command_id);
        return true;
    } catch (const std::exception& ex) {
        spdlog::error("PluginCommandRegistry: {} threw: {}", key, ex.what());
        return false;
    }
}

size_t PluginCommandRegistry::GetCount() const {
    std::lock_guard lock(mutex_);
    return commands_.size();
}

} // namespace cyxwiz::plugin
