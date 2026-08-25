#pragma once

#include "backend_pack_installer_platform.h"

#include <cstddef>
#include <functional>
#include <string>

namespace cyxwiz::installer {

struct InstallerPlanExecutionProgress {
  std::size_t completed_steps = 0;
  std::size_t total_steps = 0;
  std::string activity;
};

struct InstallerPlanExecutionResult {
  bool succeeded = false;
  std::string message;
};

using InstallerPlanExecutionObserver =
    std::function<void(const InstallerPlanExecutionProgress &)>;

InstallerPlanExecutionResult ExecuteInstallerPlan(
    BackendPackInstallerPlatform &platform, const BackendPackInstallerPlan &plan,
    const InstallerPlanExecutionObserver &observer = {});

} // namespace cyxwiz::installer
