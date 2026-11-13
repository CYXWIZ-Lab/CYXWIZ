#pragma once

// The Engine's PluginUi (plugin API 1.4): plugin panels draw the Engine's own
// buttons and theme colours through it (plugin/plugin_ui.h).

#include "../plugin/plugin_ui.h"

namespace cyxwiz::ui {

// The one instance, handed to plugins by EnginePluginContext::GetUi().
plugin::PluginUi& PluginUiKit();

}  // namespace cyxwiz::ui
