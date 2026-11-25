#include "core/user_folders.h"

#include "backend_pack_platform.h"
#include "core/engine_config.h"
#include "runtime_layout.h"

namespace cyxwiz::core {

std::filesystem::path InstalledRuntimeRoot() {
    const auto exe_dir = ExecutableDirectory();
    return runtime::InstalledRuntimeRoot(
        exe_dir.empty() ? std::filesystem::path{} : exe_dir / std::string(runtime::CurrentEngineExecutableName()),
        ActiveRuntimeRootFromEnvironment());
}

bool IsInstalledRun() { return !InstalledRuntimeRoot().empty(); }

RunFileLocations CurrentRunFileLocations() {
    return RunFileLocationsFor(IsInstalledRun(), ExecutableDirectory());
}

std::filesystem::path EngineLogPath() {
    return UserFile(CurrentRunFileLocations().log_dir, "engine_log.txt");
}

}  // namespace cyxwiz::core
