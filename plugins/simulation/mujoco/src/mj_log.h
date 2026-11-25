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

#include "plugin/plugin_context.h"

namespace cyxwiz::plugin::mujoco {

// Call first in OnLoad. `ctx` must stay valid until RemoveMuJoCoLogHandlers.
void InstallMuJoCoLogHandlers(PluginContext& ctx);

// Call last in OnUnload: logs the repeat count and puts MuJoCo's previous
// handlers back, so no pointer into an unloaded plugin remains.
void RemoveMuJoCoLogHandlers();

}  // namespace cyxwiz::plugin::mujoco
