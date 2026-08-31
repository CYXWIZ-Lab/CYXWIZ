#include "python_sandbox.h"

#include <algorithm>
#include <filesystem>

namespace scripting {

namespace {
PythonSandbox::ExecutionResult DisabledResult() {
    PythonSandbox::ExecutionResult result{};
    result.success = false;
    result.error_message =
        "Python scripting is unavailable in this Engine build";
    return result;
}
}  // namespace

PythonSandbox::PythonSandbox()
    : initialized_(false), initial_memory_(0), monitoring_active_(false) {}
PythonSandbox::PythonSandbox(const Config& config)
    : config_(config), initialized_(false), initial_memory_(0),
      monitoring_active_(false) {}
PythonSandbox::~PythonSandbox() = default;

PythonSandbox::ExecutionResult PythonSandbox::Execute(const std::string&) {
    return DisabledResult();
}

PythonSandbox::ExecutionResult PythonSandbox::ExecuteFile(const std::string&) {
    return DisabledResult();
}

void PythonSandbox::SetConfig(const Config& config) { config_ = config; }

bool PythonSandbox::IsModuleAllowed(const std::string& module_name) const {
    return config_.allowed_modules.contains(module_name);
}

bool PythonSandbox::IsBuiltinAllowed(const std::string& builtin_name) const {
    return !config_.blocked_builtins.contains(builtin_name);
}

bool PythonSandbox::IsPathAllowed(const std::string& path) const {
    if (config_.allowed_directory.empty()) return config_.allow_file_read;
    std::error_code error;
    const auto allowed = std::filesystem::weakly_canonical(
        config_.allowed_directory, error);
    if (error) return false;
    const auto candidate = std::filesystem::weakly_canonical(path, error);
    if (error) return false;
    return candidate == allowed ||
        std::mismatch(
            allowed.begin(), allowed.end(), candidate.begin(), candidate.end())
                .first == allowed.end();
}

}  // namespace scripting
