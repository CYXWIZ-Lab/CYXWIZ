#pragma once
#include "runtime_operation_lock.h"

namespace cyxwiz::runtime {
// Developer builds without installed layout/environment need no product lock.
bool AcquirePackagedEngineOwnership(
    const std::filesystem::path &executable,
    const std::filesystem::path &environment_runtime,
    RuntimeOperationLock &ownership, std::string &error);
} // namespace cyxwiz::runtime
