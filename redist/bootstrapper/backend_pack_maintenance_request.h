#pragma once

#include "runtime_layout.h"

#include <cstdint>
#include <filesystem>
#include <string>

namespace cyxwiz::runtime {

enum class BackendPackMaintenanceAction {
    Remove,
    Rollback
};

struct BackendPackMaintenanceRequest {
    BackendPackMaintenanceAction action =
        BackendPackMaintenanceAction::Remove;
    std::string runtime_set_id;
    std::uint64_t runtime_generation = 0;
    std::string backend;
    std::string pack_id;
};

enum class BackendPackMaintenanceApplyStatus {
    Applied,
    NoRequest,
    InvalidRequest,
    StaleRequest,
    Failed
};

struct BackendPackMaintenanceApplyResult {
    BackendPackMaintenanceApplyStatus status =
        BackendPackMaintenanceApplyStatus::InvalidRequest;
    std::string message;
};

std::filesystem::path BackendPackMaintenanceRequestPath(
    const std::filesystem::path& runtime_root);

bool LoadBackendPackMaintenanceRequest(
    const std::filesystem::path& runtime_root,
    BackendPackMaintenanceRequest& output,
    std::string& error);

bool QueueBackendPackMaintenanceRequest(
    const std::filesystem::path& runtime_root,
    const BackendPackMaintenanceRequest& request,
    std::string& error);

bool HasValidBackendPackRollback(
    const std::filesystem::path& runtime_root,
    const ActiveRuntimeState& active,
    std::string& error);

BackendPackMaintenanceApplyResult ApplyPendingBackendPackMaintenance(
    const std::filesystem::path& runtime_root,
    const ActiveRuntimeState& launched_runtime);

const char* BackendPackMaintenanceActionName(
    BackendPackMaintenanceAction action);
const char* BackendPackMaintenanceApplyStatusName(
    BackendPackMaintenanceApplyStatus status);

}  // namespace cyxwiz::runtime
