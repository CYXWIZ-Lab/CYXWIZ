#pragma once

// Preferences sections from plugins (IPreferencesProvider, plugin API 1.4).
// The Preferences dialog draws the sections of each tab at its end. Keyed
// "<plugin id>:<section id>".

#include "../interfaces/i_preferences_provider.h"

#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace cyxwiz::plugin {

class PluginPreferencesRegistry {
public:
    static PluginPreferencesRegistry& Instance();

    PluginPreferencesRegistry(const PluginPreferencesRegistry&) = delete;
    PluginPreferencesRegistry& operator=(const PluginPreferencesRegistry&) = delete;

    // The section is copied into the Engine's own strings (it comes from the
    // plugin's library).
    void Register(const std::string& plugin_id, const PluginPreferencesSection& section,
                  IPreferencesProvider* provider);
    void RemoveByPlugin(const std::string& plugin_id);

    struct Entry {
        std::string key;  // "<plugin id>:<section id>"
        PluginPreferencesSection section;
    };
    // The sections drawn in `tab`, in registration order. A section whose tab
    // the Engine does not have (known_tabs) belongs to `fallback_tab`.
    std::vector<Entry> SectionsFor(const std::string& tab, const std::vector<std::string>& known_tabs,
                                   const std::string& fallback_tab) const;

    // Draws one section's settings; false when no section has that key or
    // the plugin threw (logged).
    bool Render(const std::string& key);

    size_t GetCount() const;

private:
    PluginPreferencesRegistry() = default;

    struct Registered {
        std::string plugin_id;
        PluginPreferencesSection section;
        IPreferencesProvider* provider = nullptr;  // non-owning
    };

    mutable std::mutex mutex_;
    std::vector<std::string> order_;
    std::map<std::string, Registered> sections_;
};

}  // namespace cyxwiz::plugin
