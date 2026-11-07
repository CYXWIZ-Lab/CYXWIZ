#pragma once

// Menu commands from plugins (ICommandProvider, plugin API 1.4). The menu
// bar lists them by the menu they name, asks their state each frame and runs
// the one the user chooses. Keyed "<plugin id>:<command id>".

#include "../interfaces/i_command_provider.h"

#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace cyxwiz::plugin {

class PluginCommandRegistry {
public:
    static PluginCommandRegistry& Instance();

    PluginCommandRegistry(const PluginCommandRegistry&) = delete;
    PluginCommandRegistry& operator=(const PluginCommandRegistry&) = delete;

    // The info is copied into the Engine's own strings (it comes from the
    // plugin's library).
    void Register(const std::string& plugin_id, const PluginCommandInfo& info, ICommandProvider* provider);
    void RemoveByPlugin(const std::string& plugin_id);

    struct Entry {
        std::string key;  // "<plugin id>:<command id>"
        PluginCommandInfo info;
        PluginCommandState state;
    };
    // Every command with its current state, in registration order. A
    // provider that throws reports the command disabled with the reason.
    std::vector<Entry> Snapshot() const;

    // Runs the command; false when no command has that key or it threw.
    bool Run(const std::string& key);

    size_t GetCount() const;

private:
    PluginCommandRegistry() = default;

    struct Registered {
        std::string plugin_id;
        PluginCommandInfo info;
        ICommandProvider* provider = nullptr;  // non-owning
    };

    mutable std::mutex mutex_;
    std::vector<std::string> order_;
    std::map<std::string, Registered> commands_;
};

} // namespace cyxwiz::plugin
