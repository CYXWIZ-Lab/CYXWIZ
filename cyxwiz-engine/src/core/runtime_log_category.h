#pragma once

// The category of an Engine log line (TOFIX159 B4). Lines logged through the
// Engine's own spdlog logger name their subsystem at the start
// ("TrainingExecutor: ...", "PluginManager: ...", "Project opened: ..."); one
// table here turns that into a category, so the Logs tab's Category column
// and filters (category = "training") mean something. Lines from other
// loggers (a plugin's) keep their logger name as the category, in the store's
// form (lower case, other characters as _).

#include <string>
#include <string_view>

namespace cyxwiz {

// engine, compute, device, ui, plugin, python, project, account, network,
// training, data, graph; "system" when no rule matches.
std::string ClassifyLogCategory(std::string_view logger_name, std::string_view message);

}  // namespace cyxwiz
