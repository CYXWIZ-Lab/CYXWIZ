#include "engine_runtime_ownership.h"
#include "runtime_layout.h"

namespace cyxwiz::runtime {
bool AcquirePackagedEngineOwnership(
    const std::filesystem::path &executable,
    const std::filesystem::path &environment_runtime,
    RuntimeOperationLock &ownership, std::string &error) {
  std::error_code code;
  const auto actual_executable = std::filesystem::canonical(executable, code);
  if (code) {
    error = "Cannot resolve the running Engine executable: " + code.message();
    return false;
  }
  const auto base_parent = actual_executable.parent_path().parent_path();
  auto root = environment_runtime;
  if (root.empty()) {
    if (base_parent.filename() != "base")
      return true;
    root = base_parent.parent_path();
  }
  if (ownership.AcquireEngineUse(root, error) !=
          RuntimeOperationLockStatus::Acquired ||
      !ownership.RestrictInheritedEngineOwnership(error))
    return false;
  ActiveRuntime active;
  if (!ResolveActiveRuntime(root, active, error))
    return false;
  const bool matches = std::filesystem::equivalent(
      actual_executable, active.engine_executable, code);
  if (code || !matches) {
    error = "This Engine executable is not the active installed package";
    return false;
  }
  return true;
}
} // namespace cyxwiz::runtime
