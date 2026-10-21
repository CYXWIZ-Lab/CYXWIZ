#pragma once

#include "runtime_layout.h"

namespace cyxwiz::runtime {

enum class RuntimeInstallationCondition { Fresh, Active, RecoveryRequired };

struct RuntimeInstallationInspection {
  RuntimeInstallationCondition condition =
      RuntimeInstallationCondition::RecoveryRequired;
  ActiveRuntimeState active;
  std::string message;
};

// Read-only structural preflight, not a component-hash or device verification.
// Does not authorize repair, deletion, or activation of an incomplete runtime.
RuntimeInstallationInspection
InspectRuntimeInstallation(const std::filesystem::path &runtime_root);

} // namespace cyxwiz::runtime
