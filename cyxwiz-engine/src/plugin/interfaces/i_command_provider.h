// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
#pragma once

// Commands a plugin adds to the Engine's menus (plugin API 1.4). The Engine
// finds them through IPlugin::QueryInterface("ICommandProvider") when the
// plugin starts (UIModify permission), lists them in the menu they name and
// in the Command Palette, asks their state while a menu is drawn and runs one
// when the user chooses it. Everything is called on the UI thread.

#include <string>
#include <vector>

namespace cyxwiz::plugin {

struct PluginCommandInfo {
    std::string command_id;  // unique within the plugin ("run")
    // The top menu, by its title ("Simulation"). A title the Engine has no
    // menu for puts the command in Apps, under a heading of that title.
    std::string menu;
    std::string group;  // commands of one group sit together, a separator between groups
    std::string label;  // "Run Simulation"; end it with "..." when it opens a dialog
    std::string hint;   // one sentence for the status bar and the Command Palette
    bool checkable = false;
};

struct PluginCommandState {
    bool enabled = true;
    std::string disabled_reason;  // shown on hover while disabled
    bool checked = false;         // checkable commands
};

class ICommandProvider {
public:
    virtual ~ICommandProvider() = default;

    virtual std::vector<PluginCommandInfo> GetCommands() = 0;
    // Asked every frame a menu or the palette shows the command: keep it cheap.
    virtual PluginCommandState GetCommandState(const std::string& command_id) = 0;
    // The user chose the command.
    virtual void RunCommand(const std::string& command_id) = 0;
};

} // namespace cyxwiz::plugin
