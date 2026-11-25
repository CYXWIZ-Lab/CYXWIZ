#include "core/user_folders.h"

#include "backend_pack_platform.h"
#include "core/engine_config.h"
#include "runtime_layout.h"

namespace cyxwiz::core {

bool IsInstalledRun() {
    if (!user_folders_detail::EnvironmentPath("CYXWIZ_ACTIVE_RUNTIME_ROOT").empty()) return true;
    const auto exe_dir = ExecutableDirectory();
    return !exe_dir.empty() &&
           runtime::IsInstalledBaseExecutable(exe_dir / std::string(runtime::CurrentEngineExecutableName()));
}

RunFileLocations CurrentRunFileLocations() {
    return RunFileLocationsFor(IsInstalledRun(), ExecutableDirectory());
}

std::filesystem::path EngineLogPath() {
    return UserFile(CurrentRunFileLocations().log_dir, "engine_log.txt");
}

}  // namespace cyxwiz::core
