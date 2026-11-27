// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
#pragma once

// Backend lifecycle for host programs (the Engine, tests, tools): which logger
// the backend logs through. Plugins never call it; they log through their
// plugin context.
//
// Public lifecycle functions are declared in cyxwiz.h:
//   - cyxwiz::Initialize()
//   - cyxwiz::Shutdown()
//   - cyxwiz::GetVersionString()

#include <memory>

#include "api_export.h"

namespace spdlog {
class logger;
}

namespace cyxwiz {

// Makes the backend log through the host program's logger, so its lines reach
// the same sinks (log file, Console panel) at the same level. Where spdlog is
// linked statically (Linux, macOS) the backend has its own copy of spdlog,
// kept private so other libraries cannot bind to it; without this call its
// lines would go only to its own default stdout logger. Call once, after the
// host sets up its default logger. A null logger is ignored.
CYXWIZ_API void UseHostLogger(std::shared_ptr<spdlog::logger> logger);

} // namespace cyxwiz
