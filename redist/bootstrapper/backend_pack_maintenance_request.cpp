#include "backend_pack_maintenance_request.h"

#include "backend_pack_remover.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <fstream>
#include <set>

#include <nlohmann/json.hpp>

namespace cyxwiz::runtime {
namespace {

using Json = nlohmann::json;

bool IsIdentifier(const std::string& value) {
    if (value.empty() || value.size() > 128 ||
        !std::isalnum(static_cast<unsigned char>(value.front()))) {
        return false;
    }
    return std::all_of(value.begin(), value.end(), [](unsigned char value) {
        return std::islower(value) || std::isdigit(value) || value == '.' ||
               value == '_' || value == '-';
    });
}

bool IsOptionalBackend(const std::string& backend) {
    return backend == "cuda" || backend == "opencl" ||
           backend == "oneapi";
}

bool SameIdentity(
    const ActiveRuntimeState& state,
    const BackendPackMaintenanceRequest& request) {
    return state.runtime_set_id == request.runtime_set_id &&
           state.generation == request.runtime_generation;
}

bool HasPack(
    const ActiveRuntimeState& state,
    const std::string& backend,
    const std::string& pack_id) {
    return std::any_of(
        state.packs.begin(), state.packs.end(), [&](const auto& pack) {
            return pack.backend == backend && pack.pack_id == pack_id;
        });
}

bool ValidateRequest(
    const BackendPackMaintenanceRequest& request,
    std::string& error) {
    if (!IsIdentifier(request.runtime_set_id) ||
        request.runtime_generation == 0) {
        error = "Maintenance request runtime identity is invalid";
        return false;
    }
    if (request.action == BackendPackMaintenanceAction::Remove) {
        if (!IsOptionalBackend(request.backend) ||
            !IsIdentifier(request.pack_id)) {
            error = "Removal request needs an optional backend and pack identity";
            return false;
        }
    } else if (!request.backend.empty() || !request.pack_id.empty()) {
        error = "Rollback request must not name an individual pack";
        return false;
    }
    return true;
}

Json RequestDocument(const BackendPackMaintenanceRequest& request) {
    return {
        {"schema_version", 1},
        {"action", BackendPackMaintenanceActionName(request.action)},
        {"runtime_set_id", request.runtime_set_id},
        {"runtime_generation", request.runtime_generation},
        {"backend", request.backend},
        {"pack_id", request.pack_id}};
}

bool ReadDocument(
    const std::filesystem::path& path,
    Json& output,
    std::string& error) {
    std::error_code filesystem_error;
    const auto status = std::filesystem::symlink_status(path, filesystem_error);
    if (filesystem_error || !std::filesystem::is_regular_file(status) ||
        std::filesystem::is_symlink(status)) {
        error = "Pending maintenance request is missing or unsafe";
        return false;
    }
    const auto size = std::filesystem::file_size(path, filesystem_error);
    if (filesystem_error || size == 0 || size > 64 * 1024) {
        error = "Pending maintenance request has an invalid size";
        return false;
    }
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        error = "Cannot open pending maintenance request";
        return false;
    }
    try {
        output = Json::parse(stream, nullptr, true, true);
    } catch (const std::exception& exception) {
        error = std::string("Cannot parse pending maintenance request: ") +
                exception.what();
        return false;
    }
    return true;
}

bool ParseDocument(
    const Json& document,
    BackendPackMaintenanceRequest& output,
    std::string& error) {
    if (!document.is_object()) {
        error = "Pending maintenance request must be an object";
        return false;
    }
    const std::set<std::string> expected{
        "schema_version", "action", "runtime_set_id",
        "runtime_generation", "backend", "pack_id"};
    std::set<std::string> observed;
    for (const auto& [key, value] : document.items()) {
        (void)value;
        observed.insert(key);
    }
    if (observed != expected ||
        !document["schema_version"].is_number_unsigned() ||
        document["schema_version"].get<std::uint64_t>() != 1 ||
        !document["action"].is_string() ||
        !document["runtime_set_id"].is_string() ||
        !document["runtime_generation"].is_number_unsigned() ||
        !document["backend"].is_string() ||
        !document["pack_id"].is_string()) {
        error = "Pending maintenance request schema is invalid";
        return false;
    }
    const auto action = document["action"].get<std::string>();
    if (action == "remove") {
        output.action = BackendPackMaintenanceAction::Remove;
    } else if (action == "rollback") {
        output.action = BackendPackMaintenanceAction::Rollback;
    } else {
        error = "Pending maintenance action is unsupported";
        return false;
    }
    output.runtime_set_id = document["runtime_set_id"].get<std::string>();
    output.runtime_generation =
        document["runtime_generation"].get<std::uint64_t>();
    output.backend = document["backend"].get<std::string>();
    output.pack_id = document["pack_id"].get<std::string>();
    return ValidateRequest(output, error);
}

}  // namespace

std::filesystem::path BackendPackMaintenanceRequestPath(
    const std::filesystem::path& runtime_root) {
    return runtime_root / "pending-backend-maintenance.json";
}

bool LoadBackendPackMaintenanceRequest(
    const std::filesystem::path& runtime_root,
    BackendPackMaintenanceRequest& output,
    std::string& error) {
    output = {};
    error.clear();
    if (runtime_root.empty() || !runtime_root.is_absolute()) {
        error = "Runtime root must be absolute";
        return false;
    }
    Json document;
    return ReadDocument(
               BackendPackMaintenanceRequestPath(runtime_root),
               document, error) &&
           ParseDocument(document, output, error);
}

bool HasValidBackendPackRollback(
    const std::filesystem::path& runtime_root,
    const ActiveRuntimeState& active,
    std::string& error) {
    ActiveRuntimeState rollback;
    const auto path = runtime_root / "rollback" / active.runtime_set_id /
                      "previous-active-runtime.json";
    if (!LoadActiveRuntimeState(path, rollback, error)) return false;
    if (rollback.runtime_set_id != active.runtime_set_id ||
        rollback.base_pack_id != active.base_pack_id) {
        error = "Rollback state belongs to a different runtime set or base";
        return false;
    }
    ActiveRuntime resolved;
    return ResolveRuntimeState(runtime_root, rollback, resolved, error);
}

bool QueueBackendPackMaintenanceRequest(
    const std::filesystem::path& runtime_root,
    const BackendPackMaintenanceRequest& request,
    std::string& error) {
    error.clear();
    if (runtime_root.empty() || !runtime_root.is_absolute() ||
        !ValidateRequest(request, error)) {
        if (error.empty()) error = "Runtime root must be absolute";
        return false;
    }
    ActiveRuntimeState active;
    if (!LoadActiveRuntimeState(
            runtime_root / "active-runtime.json", active, error) ||
        !SameIdentity(active, request)) {
        if (error.empty()) {
            error = "Active runtime changed before maintenance was queued";
        }
        return false;
    }
    if (request.action == BackendPackMaintenanceAction::Remove &&
        !HasPack(active, request.backend, request.pack_id)) {
        error = "The exact backend pack is not active in the requested runtime";
        return false;
    }
    if (request.action == BackendPackMaintenanceAction::Rollback &&
        !HasValidBackendPackRollback(runtime_root, active, error)) {
        error = "No valid previous runtime state is available: " + error;
        return false;
    }

    const auto destination = BackendPackMaintenanceRequestPath(runtime_root);
    std::error_code filesystem_error;
    if (std::filesystem::exists(destination, filesystem_error) ||
        filesystem_error) {
        error = filesystem_error
            ? "Cannot inspect the pending maintenance request"
            : "A backend maintenance action is already queued";
        return false;
    }
    const auto temporary = destination.string() + ".tmp-" +
        std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count());
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        if (!stream) {
            error = "Cannot create pending maintenance request";
            return false;
        }
        stream << RequestDocument(request).dump(2) << '\n';
        stream.flush();
        if (!stream) {
            error = "Cannot write pending maintenance request";
            stream.close();
            std::filesystem::remove(temporary, filesystem_error);
            return false;
        }
    }
    std::filesystem::rename(temporary, destination, filesystem_error);
    if (filesystem_error) {
        std::filesystem::remove(temporary, filesystem_error);
        error = "Cannot publish pending maintenance request atomically";
        return false;
    }
    return true;
}

BackendPackMaintenanceApplyResult ApplyPendingBackendPackMaintenance(
    const std::filesystem::path& runtime_root,
    const ActiveRuntimeState& launched_runtime) {
    const auto request_path = BackendPackMaintenanceRequestPath(runtime_root);
    std::error_code filesystem_error;
    if (!std::filesystem::exists(request_path, filesystem_error)) {
        return {filesystem_error
                    ? BackendPackMaintenanceApplyStatus::InvalidRequest
                    : BackendPackMaintenanceApplyStatus::NoRequest,
                filesystem_error
                    ? "Cannot inspect pending backend maintenance"
                    : "No backend maintenance action is queued"};
    }
    BackendPackMaintenanceRequest request;
    std::string error;
    if (!LoadBackendPackMaintenanceRequest(runtime_root, request, error)) {
        return {BackendPackMaintenanceApplyStatus::InvalidRequest, error};
    }
    ActiveRuntimeState current;
    if (!SameIdentity(launched_runtime, request) ||
        !LoadActiveRuntimeState(
            runtime_root / "active-runtime.json", current, error) ||
        !SameIdentity(current, request)) {
        return {BackendPackMaintenanceApplyStatus::StaleRequest,
                error.empty()
                    ? "Pending maintenance identity no longer matches the launched runtime"
                    : error};
    }

    std::string message;
    bool applied = false;
    if (request.action == BackendPackMaintenanceAction::Remove) {
        BackendPackRemover remover(runtime_root);
        const auto result = remover.Remove(request.backend, request.pack_id);
        applied = result.status == BackendPackRemovalStatus::Removed ||
                  result.status == BackendPackRemovalStatus::AlreadyAbsent;
        message = result.message;
    } else {
        BackendPackStateService state_service(runtime_root);
        const auto result = state_service.Rollback();
        applied = result.status == BackendPackStateStatus::Completed;
        message = result.message;
    }
    if (!applied) {
        return {BackendPackMaintenanceApplyStatus::Failed, std::move(message)};
    }
    std::filesystem::remove(request_path, filesystem_error);
    if (filesystem_error) {
        return {BackendPackMaintenanceApplyStatus::Failed,
                "Maintenance completed but the pending request could not be cleared"};
    }
    return {BackendPackMaintenanceApplyStatus::Applied, std::move(message)};
}

const char* BackendPackMaintenanceActionName(
    BackendPackMaintenanceAction action) {
    return action == BackendPackMaintenanceAction::Remove
        ? "remove"
        : "rollback";
}

const char* BackendPackMaintenanceApplyStatusName(
    BackendPackMaintenanceApplyStatus status) {
    switch (status) {
        case BackendPackMaintenanceApplyStatus::Applied: return "applied";
        case BackendPackMaintenanceApplyStatus::NoRequest: return "no_request";
        case BackendPackMaintenanceApplyStatus::InvalidRequest:
            return "invalid_request";
        case BackendPackMaintenanceApplyStatus::StaleRequest:
            return "stale_request";
        case BackendPackMaintenanceApplyStatus::Failed: return "failed";
    }
    return "unknown";
}

}  // namespace cyxwiz::runtime
