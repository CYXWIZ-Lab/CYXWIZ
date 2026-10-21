#pragma once

#include "backend_pack_installer.h"

namespace cyxwiz::runtime {

// Internal commit boundary: caller holds a RuntimeMutationLease and has already
// verified the exact staged payload. No GUI/CLI may call this without exclusive
// installation ownership and an execution-active check covering other
// processes.
BackendPackInstallResult
PublishBaseRepair(const std::filesystem::path &runtime_root,
                  const std::filesystem::path &staging_root,
                  const ActiveRuntimeState &expected,
                  const BackendPackExecutionActiveCheck &execution_active,
                  const BackendPackInstallCheckpointHook &checkpoint,
                  const std::atomic<bool> &cancelled);

} // namespace cyxwiz::runtime
