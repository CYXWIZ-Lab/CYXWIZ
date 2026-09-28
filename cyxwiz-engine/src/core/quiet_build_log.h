#pragma once

// Info logging of graph compiles and model builds, silenced on the current
// thread inside a QuietModelBuildScope (model_builder.h). The background
// compile of the canvas runs after every edit (TOFIX123) and must not fill
// the Console; compiles for Compile/Train/Debug log as before.

#include <spdlog/spdlog.h>

namespace cyxwiz {
inline thread_local bool g_quiet_build_log = false;
}  // namespace cyxwiz

#define CYXWIZ_BUILDER_INFO(...)     do {         if (!::cyxwiz::g_quiet_build_log) spdlog::info(__VA_ARGS__);     } while (0)
