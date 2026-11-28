#pragma once

// MuJoCo's warnings and errors in the Engine log. With no handler MuJoCo
// appends them to MUJOCO_LOG.TXT in the working folder (and prints them); the
// handlers below route them through the plugin's context instead, so MuJoCo
// writes no file in any run.
//
// Warnings: the first of each kind goes to the Engine log (numbers in the text
// do not make a new kind); repeats are counted and summed up once when the
// handlers are removed. Errors: logged, then the process ends, as MuJoCo's
// own handler does (MuJoCo cannot continue after an error).

//
// The plugin's own lines (LogInfo, LogWarn, LogError below) take the same
// route, for every part of the plugin that holds no context of its own. The
// plugin does not log through spdlog: on Linux a plugin's spdlog is not the
// Engine's, so its lines would reach only the terminal. Without a context
// (before OnLoad, after OnUnload, in tests) a line goes to stderr.

#include <fmt/format.h>

#include <string>
#include <utility>

#include "plugin/plugin_context.h"

namespace cyxwiz::plugin::mujoco {

// Call first in OnLoad. `ctx` must stay valid until RemoveMuJoCoLogHandlers.
void InstallMuJoCoLogHandlers(PluginContext& ctx);

// Call last in OnUnload: logs the repeat count and puts MuJoCo's previous
// handlers back, so no pointer into an unloaded plugin remains.
void RemoveMuJoCoLogHandlers();

enum class PluginLogLevel { Info, Warn, Error };

// One line in the Engine log through the plugin's context.
void PluginLog(PluginLogLevel level, const std::string& message);

template <typename... Args>
void LogInfo(fmt::format_string<Args...> format, Args&&... args) {
    PluginLog(PluginLogLevel::Info, fmt::format(format, std::forward<Args>(args)...));
}

template <typename... Args>
void LogWarn(fmt::format_string<Args...> format, Args&&... args) {
    PluginLog(PluginLogLevel::Warn, fmt::format(format, std::forward<Args>(args)...));
}

template <typename... Args>
void LogError(fmt::format_string<Args...> format, Args&&... args) {
    PluginLog(PluginLogLevel::Error, fmt::format(format, std::forward<Args>(args)...));
}

}  // namespace cyxwiz::plugin::mujoco
