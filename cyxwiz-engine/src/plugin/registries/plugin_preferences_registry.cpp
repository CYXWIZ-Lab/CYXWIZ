#include "plugin_preferences_registry.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <exception>

namespace cyxwiz::plugin {

PluginPreferencesRegistry& PluginPreferencesRegistry::Instance() {
    static PluginPreferencesRegistry instance;
    return instance;
}

void PluginPreferencesRegistry::Register(const std::string& plugin_id, const PluginPreferencesSection& section,
                                         IPreferencesProvider* provider) {
    if (!provider || section.section_id.empty() || section.title.empty()) return;
    const std::string key = plugin_id + ":" + std::string(section.section_id.c_str());
    std::lock_guard lock(mutex_);
    if (sections_.count(key)) {
        spdlog::warn("PluginPreferencesRegistry: Duplicate section {}, skipping", key);
        return;
    }
    Registered entry;
    entry.plugin_id = plugin_id;
    entry.section.section_id = std::string(section.section_id.c_str());
    entry.section.tab = std::string(section.tab.c_str());
    entry.section.title = std::string(section.title.c_str());
    entry.section.hint = std::string(section.hint.c_str());
    entry.provider = provider;
    sections_[key] = std::move(entry);
    order_.push_back(key);
}

void PluginPreferencesRegistry::RemoveByPlugin(const std::string& plugin_id) {
    std::lock_guard lock(mutex_);
    for (auto it = sections_.begin(); it != sections_.end();) {
        if (it->second.plugin_id == plugin_id) {
            order_.erase(std::remove(order_.begin(), order_.end(), it->first), order_.end());
            it = sections_.erase(it);
        } else {
            ++it;
        }
    }
}

std::vector<PluginPreferencesRegistry::Entry> PluginPreferencesRegistry::SectionsFor(
    const std::string& tab, const std::vector<std::string>& known_tabs, const std::string& fallback_tab) const {
    std::lock_guard lock(mutex_);
    std::vector<Entry> out;
    for (const auto& key : order_) {
        const auto& s = sections_.at(key).section;
        const bool known = std::find(known_tabs.begin(), known_tabs.end(), s.tab) != known_tabs.end();
        if ((known ? s.tab : fallback_tab) == tab) out.push_back({key, s});
    }
    return out;
}

bool PluginPreferencesRegistry::Render(const std::string& key) {
    IPreferencesProvider* provider = nullptr;
    std::string section_id;
    {
        std::lock_guard lock(mutex_);
        const auto it = sections_.find(key);
        if (it == sections_.end()) return false;
        provider = it->second.provider;
        section_id = it->second.section.section_id;
    }
    try {
        provider->RenderPreferences(section_id);
        return true;
    } catch (const std::exception& ex) {
        spdlog::error("PluginPreferencesRegistry: RenderPreferences({}) threw: {}", key, ex.what());
        return false;
    }
}

size_t PluginPreferencesRegistry::GetCount() const {
    std::lock_guard lock(mutex_);
    return sections_.size();
}

}  // namespace cyxwiz::plugin
