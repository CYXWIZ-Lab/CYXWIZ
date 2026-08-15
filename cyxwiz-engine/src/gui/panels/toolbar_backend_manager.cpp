#include "toolbar.h"

#include "../icons.h"
#include "../../core/async_task_manager.h"
#include "../../core/backend_pack_catalog_adapter.h"
#include "../../core/backend_pack_manager_model.h"
#include "../../core/backend_pack_qualification_adapter.h"
#include "../../core/compute_runtime_paths.h"
#include "../../core/route_qualification_snapshot.h"
#include "../../core/window_manager.h"
#include "backend_pack_maintenance_request.h"
#include "backend_pack_lifecycle_service.h"
#include "backend_pack_platform.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cyxwiz/cyxwiz.h>
#include <imgui.h>

namespace cyxwiz {
namespace {

const char* PackBackendName(DeviceType type) {
    switch (type) {
        case DeviceType::CPU: return "CPU base";
        case DeviceType::CUDA: return "CUDA";
        case DeviceType::OPENCL: return "OpenCL";
        case DeviceType::ONEAPI: return "oneAPI";
        default: return "Unknown";
    }
}

std::string PackBackendId(DeviceType type) {
    switch (type) {
        case DeviceType::CPU: return "cpu";
        case DeviceType::CUDA: return "cuda";
        case DeviceType::OPENCL: return "opencl";
        case DeviceType::ONEAPI: return "oneapi";
        default: return "unknown";
    }
}

bool RenderActionButton(
    const char* label,
    const BackendPackActionDecision& decision) {
    if (!decision.enabled) ImGui::BeginDisabled();
    const bool clicked = ImGui::SmallButton(label);
    if (!decision.enabled) ImGui::EndDisabled();
    if (!decision.enabled && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip("%s", decision.reason.c_str());
    }
    return clicked && decision.enabled;
}

std::string JoinOrUnavailable(const std::vector<std::string>& values) {
    if (values.empty()) return "Unavailable from current signed catalog";
    std::string result;
    for (const auto& value : values) {
        if (!result.empty()) result += ", ";
        result += value;
    }
    return result;
}

std::filesystem::path ActiveRuntimeRoot() {
    const char* value = std::getenv("CYXWIZ_ACTIVE_RUNTIME_ROOT");
    return value && *value ? std::filesystem::path(value)
                           : std::filesystem::path{};
}

std::filesystem::path BackendPackInstallerPath() {
    std::filesystem::path path = core::WindowManager::GetExecutablePath();
    path.replace_filename(
        runtime::CurrentBackendPackInstallerExecutableName());
    return path;
}

std::filesystem::path BackendPackManagerPath() {
    std::filesystem::path path = core::WindowManager::GetExecutablePath();
#ifdef _WIN32
    path.replace_filename("cyxwiz-installer.exe");
#else
    path.replace_filename("cyxwiz-installer");
#endif
    return path;
}

std::string CurrentUtc() {
    const auto now = std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now());
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif
    std::ostringstream stream;
    stream << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return stream.str();
}

std::shared_ptr<runtime::BackendPackLifecycleService> CreatePackService(
    const std::filesystem::path& runtime_root,
    const std::shared_ptr<RouteQualificationService>& qualification_service,
    std::string& error) {
    auto trust = runtime::BackendPackTrustStore::Load(
        runtime_root / "trust" / "trusted-keys.json", error);
    if (!trust) return {};
    std::filesystem::path probe = core::WindowManager::GetExecutablePath();
#ifdef _WIN32
    probe.replace_filename("cyxwiz-route-probe.exe");
#else
    probe.replace_filename("cyxwiz-route-probe");
#endif
    BackendPackQualificationAdapterOptions options;
    options.runtime_root = runtime_root;
    options.probe_executable = probe;
    options.cache_path = GetRouteQualificationCachePath();
    auto verifier = runtime::BackendPackMetadataVerifier(
        std::move(*trust), GetVersionString(),
        std::string(runtime::CurrentBackendPackPlatformId()),
        std::string(runtime::CurrentBackendPackArchitectureId()));
    return std::make_shared<runtime::BackendPackLifecycleService>(
        runtime_root, std::move(verifier),
        runtime::BackendPackExecutionActiveCheck{},
        CreateBackendPackQualificationHook(
            qualification_service, std::move(options)));
}

bool SameRuntimeIdentity(
    const runtime::ActiveRuntimeState& active,
    const RuntimeQualificationIdentity& identity) {
    if (active.runtime_set_id != identity.runtime_set_id ||
        active.generation != identity.generation ||
        active.base_pack_id != identity.base_pack_id ||
        active.packs.size() != identity.backend_packs.size()) {
        return false;
    }
    for (const auto& pack : active.packs) {
        const auto match = std::find_if(
            identity.backend_packs.begin(), identity.backend_packs.end(),
            [&](const auto& candidate) {
                return PackBackendId(candidate.type) == pack.backend &&
                       candidate.pack_id == pack.pack_id;
            });
        if (match == identity.backend_packs.end()) return false;
    }
    return true;
}

}  // namespace

bool ToolbarPanel::RenderBackendManagerSection(bool training_active) {
    const auto qualification_task =
        route_qualification_task_id_ == 0
            ? std::shared_ptr<AsyncTask>{}
            : AsyncTaskManager::Instance().GetTask(
                  route_qualification_task_id_);
    const bool qualification_running =
        qualification_task &&
        (qualification_task->GetState() == TaskState::Pending ||
         qualification_task->GetState() == TaskState::Running);
    std::string identity_error;
    const auto identity =
        ReadActiveRuntimeQualificationIdentity(identity_error);
    const auto runtime_root = ActiveRuntimeRoot();
    const auto installer_manager = BackendPackManagerPath();
    const bool installer_manager_available =
        !runtime_root.empty() && runtime_root.is_absolute() &&
        std::filesystem::is_regular_file(installer_manager);
    runtime::ActiveRuntimeState next_runtime;
    std::string next_runtime_error;
    const bool next_runtime_available =
        !runtime_root.empty() && runtime_root.is_absolute() &&
        runtime::LoadActiveRuntimeState(
            runtime_root / "active-runtime.json", next_runtime,
            next_runtime_error);
    if (runtime_root != backend_pack_runtime_root_) {
        backend_pack_runtime_root_ = runtime_root;
        backend_pack_lifecycle_service_.reset();
        backend_pack_catalog_records_.clear();
        backend_pack_catalog_loaded_ = false;
        backend_pack_catalog_available_ = false;
        backend_pack_catalog_message_.clear();
    }
    if (next_runtime_available && !backend_pack_catalog_loaded_) {
        backend_pack_catalog_loaded_ = true;
        backend_pack_catalog_available_ = false;
        backend_pack_catalog_records_.clear();
        std::string catalog_error;
        if (!backend_pack_lifecycle_service_) {
            backend_pack_lifecycle_service_ = CreatePackService(
                runtime_root, route_qualification_service_, catalog_error);
        }
        runtime::VerifiedBackendPackCatalogSnapshot snapshot;
        if (backend_pack_lifecycle_service_ &&
            backend_pack_lifecycle_service_->ReadCatalogSnapshot(
                CurrentUtc(), snapshot, catalog_error)) {
            backend_pack_catalog_records_ = BuildBackendPackCatalogRecords(
                snapshot, next_runtime);
            backend_pack_catalog_available_ = true;
            backend_pack_catalog_message_ =
                "Signed catalog " + snapshot.catalog.catalog_id +
                " verified; " +
                std::to_string(snapshot.records.size()) +
                " optional pack(s) published.";
        } else {
            backend_pack_catalog_message_ = catalog_error.empty()
                ? "No current signed backend-pack catalog is available."
                : catalog_error;
        }
    }
    const bool current_matches_next =
        next_runtime_available && identity.has_value() &&
        SameRuntimeIdentity(next_runtime, *identity);
    const auto evidence = GetRouteQualificationSnapshot();
    const bool evidence_matches_runtime =
        current_matches_next && evidence.has_value() &&
        evidence->runtime_set_id == identity->runtime_set_id &&
        evidence->runtime_generation == identity->generation &&
        evidence->base_pack_id == identity->base_pack_id;

    std::vector<BackendPackManagerRecord> records =
        backend_pack_catalog_records_;
    if (next_runtime_available) {
        const auto add_record = [&](DeviceType type, const std::string& pack_id) {
            const auto existing = std::find_if(
                records.begin(), records.end(), [&](const auto& record) {
                    return record.installed_pack_id == pack_id ||
                           record.pack_id == pack_id;
                });
            if (existing != records.end()) return;
            BackendPackManagerRecord record;
            record.backend = PackBackendId(type);
            record.pack_id = pack_id;
            record.installed_pack_id = pack_id;
            record.installed = true;
            record.active = true;
            records.push_back(std::move(record));
        };
        add_record(DeviceType::CPU, next_runtime.base_pack_id);
        for (const auto& pack : next_runtime.packs) {
            const auto type = pack.backend == "cuda"
                ? DeviceType::CUDA
                : pack.backend == "opencl"
                    ? DeviceType::OPENCL
                    : DeviceType::ONEAPI;
            add_record(type, pack.pack_id);
        }
    }
    if (evidence_matches_runtime) {
        for (auto& record : records) {
            for (const auto& route : evidence->routes) {
                const auto installed_id = record.installed_pack_id.empty()
                    ? record.pack_id : record.installed_pack_id;
                if (PackBackendId(route.type) != record.backend ||
                    route.pack_id != installed_id) {
                    continue;
                }
                record.qualification_evidence_available = true;
                record.training_authorized =
                    record.training_authorized || route.certified;
            }
        }
    }

    runtime::BackendPackMaintenanceRequest pending_request;
    std::string pending_error;
    const bool maintenance_pending =
        next_runtime_available &&
        runtime::LoadBackendPackMaintenanceRequest(
            runtime_root, pending_request, pending_error);
    std::error_code pending_filesystem_error;
    const bool pending_file_present =
        next_runtime_available && std::filesystem::exists(
            runtime::BackendPackMaintenanceRequestPath(runtime_root),
            pending_filesystem_error);
    backend_pack_maintenance_queued_ = pending_file_present;
    std::string rollback_error;
    const bool rollback_available =
        next_runtime_available &&
        runtime::HasValidBackendPackRollback(
            runtime_root, next_runtime, rollback_error);

    BackendPackManagerContext context;
    context.packaged_runtime = next_runtime_available;
    context.operation_running = qualification_running;
    context.training_active = training_active;
    // Maintenance is queued for the bootstrapper so the running process
    // never mutates loaded DLLs.
    context.catalog_available = backend_pack_catalog_available_;
    context.delivery_available = installer_manager_available;
    context.repair_available =
        next_runtime_available &&
        std::filesystem::is_regular_file(BackendPackInstallerPath());
    context.maintenance_available = next_runtime_available;
    context.maintenance_identity_matches = current_matches_next;
    context.maintenance_pending = pending_file_present;
    context.rollback_available = rollback_available;

    bool verify_requested = false;
    if (!ImGui::CollapsingHeader(
            ICON_FA_CUBES " Backend Manager",
            ImGuiTreeNodeFlags_DefaultOpen)) {
        return false;
    }

    ImGui::TextDisabled(
        "Browse backend packs here. Device selection below remains unchanged until you click OK.");
    ImGui::Spacing();
    ImGui::TextDisabled(
        "Package selection and downloads open in the standalone CyxWiz Installer. This view reports active runtime and qualification truth.");

    if (!next_runtime_error.empty()) {
        ImGui::TextColored(
            ImVec4(1.0f, 0.55f, 0.35f, 1.0f),
            "%s %s", ICON_FA_TRIANGLE_EXCLAMATION,
            next_runtime_error.c_str());
    } else if (!next_runtime_available) {
        ImGui::TextDisabled(
            "Development runtime: signed backend-pack installation and maintenance are unavailable.");
    } else {
        ImGui::TextDisabled(
            "Next launch: runtime set %s, generation %llu%s",
            next_runtime.runtime_set_id.c_str(),
            static_cast<unsigned long long>(next_runtime.generation),
            current_matches_next ? " (current process)" : " (restart pending)");
    }
    if (!identity_error.empty()) {
        ImGui::TextColored(
            ImVec4(1.0f, 0.55f, 0.35f, 1.0f),
            "%s Current process identity: %s",
            ICON_FA_TRIANGLE_EXCLAMATION, identity_error.c_str());
    }
    ImGui::TextColored(
        backend_pack_catalog_available_
            ? ImVec4(0.45f, 0.8f, 1.0f, 1.0f)
            : ImVec4(1.0f, 0.75f, 0.35f, 1.0f),
        "%s %s", ICON_FA_CIRCLE_INFO,
        backend_pack_catalog_message_.empty()
            ? "Signed catalog has not been loaded."
            : backend_pack_catalog_message_.c_str());
    if (next_runtime_available &&
        ImGui::SmallButton("Refresh signed catalog")) {
        backend_pack_catalog_loaded_ = false;
    }
    ImGui::SameLine();
    ImGui::BeginDisabled(!installer_manager_available);
    if (ImGui::SmallButton("Open CyxWiz Installer...")) {
        core::WindowManager::LaunchExecutable(
            installer_manager.string(),
            {"--runtime-root", runtime_root.string()});
    }
    ImGui::EndDisabled();
    if (!installer_manager_available &&
        ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip(
            "The standalone installer is available in packaged desktop builds");
    }
    if (maintenance_pending) {
        ImGui::TextColored(
            ImVec4(0.45f, 0.8f, 1.0f, 1.0f),
            "%s Queued for exit: %s%s%s",
            ICON_FA_CLOCK,
            runtime::BackendPackMaintenanceActionName(
                pending_request.action),
            pending_request.pack_id.empty() ? "" : " ",
            pending_request.pack_id.c_str());
    } else if (pending_file_present) {
        ImGui::TextColored(
            ImVec4(1.0f, 0.45f, 0.35f, 1.0f),
            "%s Pending maintenance request is invalid: %s",
            ICON_FA_TRIANGLE_EXCLAMATION, pending_error.c_str());
    }

    if (records.empty()) {
        ImGui::TextDisabled("No packaged backend-pack inventory is available.");
    } else if (ImGui::BeginTable(
                   "BackendPackManagerTable", 5,
                   ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                       ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Pack", ImGuiTableColumnFlags_WidthStretch, 1.4f);
        ImGui::TableSetupColumn("State", ImGuiTableColumnFlags_WidthFixed, 82.0f);
        ImGui::TableSetupColumn("Local verification", ImGuiTableColumnFlags_WidthStretch, 1.2f);
        ImGui::TableSetupColumn("Catalog", ImGuiTableColumnFlags_WidthFixed, 78.0f);
        ImGui::TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthStretch, 2.4f);
        ImGui::TableHeadersRow();
        for (const auto& record : records) {
            ImGui::PushID(record.pack_id.c_str());
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", record.backend == "cpu"
                                   ? PackBackendName(DeviceType::CPU)
                                   : record.backend.c_str());
            ImGui::TextDisabled("%s", record.pack_id.c_str());
            if (!record.installed_pack_id.empty() &&
                record.installed_pack_id != record.pack_id) {
                ImGui::TextDisabled(
                    "Installed: %s", record.installed_pack_id.c_str());
            }
            ImGui::TableNextColumn();
            const char* state = record.active
                ? (current_matches_next ? "Active" : "Next launch")
                : (!record.installed_pack_id.empty()
                       ? "Update available" : "Available");
            ImGui::TextColored(
                record.active
                    ? ImVec4(0.35f, 0.95f, 0.45f, 1.0f)
                    : ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                "%s", state);
            ImGui::TableNextColumn();
            if (!record.qualification_evidence_available) {
                ImGui::TextDisabled("Needs verification");
            } else if (record.training_authorized) {
                ImGui::TextColored(
                    ImVec4(0.35f, 0.95f, 0.45f, 1.0f), "Passed");
            } else {
                ImGui::TextColored(
                    ImVec4(1.0f, 0.7f, 0.3f, 1.0f), "Not authorized");
            }
            ImGui::TableNextColumn();
            ImGui::TextDisabled(
                "%s", BackendPackCatalogSupportName(record.catalog_support));
            ImGui::TableNextColumn();
            const auto details = EvaluateBackendPackAction(
                BackendPackAction::Details, context, &record);
            if (RenderActionButton("Details", details)) {
                backend_pack_details_id_ =
                    backend_pack_details_id_ == record.pack_id
                        ? std::string{}
                        : record.pack_id;
            }
            ImGui::SameLine();
            const auto verify = EvaluateBackendPackAction(
                BackendPackAction::Verify, context, &record);
            if (RenderActionButton("Verify", verify)) {
                verify_requested = true;
            }
            ImGui::SameLine();
            const auto install = EvaluateBackendPackAction(
                BackendPackAction::Install, context, &record);
            if (RenderActionButton("Install", install)) {
                core::WindowManager::LaunchExecutable(
                    installer_manager.string(),
                    {"--runtime-root", runtime_root.string(),
                     "--select", record.pack_id});
            }
            ImGui::SameLine();
            const auto repair = EvaluateBackendPackAction(
                BackendPackAction::Repair, context, &record);
            if (RenderActionButton("Repair", repair)) {
                backend_pack_maintenance_action_ = 2;
                backend_pack_maintenance_backend_ = record.backend;
                backend_pack_maintenance_pack_id_ = record.pack_id;
                show_backend_pack_maintenance_confirm_ = true;
            }
            ImGui::SameLine();
            const auto update = EvaluateBackendPackAction(
                BackendPackAction::Update, context, &record);
            if (RenderActionButton("Update", update)) {
                core::WindowManager::LaunchExecutable(
                    installer_manager.string(),
                    {"--runtime-root", runtime_root.string(),
                     "--select", record.pack_id});
            }
            ImGui::SameLine();
            if (RenderActionButton(
                    "Remove", EvaluateBackendPackAction(
                        BackendPackAction::Remove, context, &record))) {
                backend_pack_maintenance_action_ = 0;
                backend_pack_maintenance_backend_ = record.backend;
                backend_pack_maintenance_pack_id_ =
                    record.installed_pack_id.empty()
                        ? record.pack_id : record.installed_pack_id;
                show_backend_pack_maintenance_confirm_ = true;
            }
            ImGui::PopID();
        }
        ImGui::EndTable();
    }

    if (!backend_pack_details_id_.empty()) {
        const auto selected = std::find_if(
            records.begin(), records.end(), [&](const auto& record) {
                return record.pack_id == backend_pack_details_id_;
            });
        if (selected != records.end()) {
            ImGui::SeparatorText("Pack details");
            ImGui::BulletText("Pack ID: %s", selected->pack_id.c_str());
            ImGui::BulletText(
                "Download size: %s",
                FormatBackendPackByteSize(
                    selected->download_size_bytes).c_str());
            ImGui::BulletText(
                "License: %s", JoinOrUnavailable(selected->licenses).c_str());
            ImGui::BulletText(
                "Provider requirement: %s",
                JoinOrUnavailable(
                    selected->provider_requirements).c_str());
            ImGui::BulletText(
                "Catalog support: %s",
                BackendPackCatalogSupportName(selected->catalog_support));
            ImGui::TextWrapped(
                "Local verification: %s",
                selected->qualification_evidence_available
                    ? (selected->training_authorized
                           ? "Current evidence authorizes at least one exact route."
                           : "Current evidence does not authorize normal training.")
                    : "Required after install and before normal training selection.");
        }
    }

    const auto rollback = EvaluateBackendPackAction(
        BackendPackAction::Rollback, context);
    if (RenderActionButton(ICON_FA_ROTATE_LEFT " Rollback", rollback)) {
        backend_pack_maintenance_action_ = 1;
        backend_pack_maintenance_backend_.clear();
        backend_pack_maintenance_pack_id_.clear();
        show_backend_pack_maintenance_confirm_ = true;
    }
    if (backend_pack_maintenance_queued_) {
        ImGui::SameLine();
        if (ImGui::Button("Exit and Apply")) {
            if (exit_callback_) exit_callback_();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(
                "The bootstrapper applies the exact queued action after the Engine exits");
        }
    }
    if (!backend_pack_maintenance_message_.empty()) {
        ImGui::TextWrapped(
            "%s", backend_pack_maintenance_message_.c_str());
    }

    if (show_backend_pack_maintenance_confirm_) {
        ImGui::OpenPopup("Confirm Backend Maintenance");
        show_backend_pack_maintenance_confirm_ = false;
    }
    ImGui::SetNextWindowPos(
        ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing,
        ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSizeConstraints(
        ImVec2(460.0f, 0.0f), ImVec2(640.0f, 420.0f));
    if (ImGui::IsPopupOpen("Confirm Backend Maintenance")) {
        ImGui::SetNextWindowFocus();
    }
    if (ImGui::BeginPopupModal(
            "Confirm Backend Maintenance", nullptr,
            ImGuiWindowFlags_AlwaysAutoResize)) {
        const bool removing = backend_pack_maintenance_action_ == 0;
        const bool repairing = backend_pack_maintenance_action_ == 2;
        ImGui::TextColored(
            ImVec4(1.0f, 0.75f, 0.35f, 1.0f),
            "%s %s", ICON_FA_TRIANGLE_EXCLAMATION,
            removing ? "Remove backend pack after exit?"
                : repairing ? "Repair backend pack after exit?"
                            : "Restore the previous runtime after exit?");
        ImGui::Separator();
        if (removing || repairing) {
            ImGui::BulletText(
                "Pack: %s / %s",
                backend_pack_maintenance_backend_.c_str(),
                backend_pack_maintenance_pack_id_.c_str());
            if (repairing) {
                ImGui::TextWrapped(
                    "The signed helper will replace the complete immutable pack, verify its candidate routes, and reactivate it only if local qualification passes.");
            }
        } else {
            ImGui::BulletText("Action: Roll back the complete runtime set");
        }
        ImGui::BulletText(
            "Runtime: %s generation %llu",
            next_runtime.runtime_set_id.c_str(),
            static_cast<unsigned long long>(next_runtime.generation));
        ImGui::TextWrapped(
            "The device candidate below will not change. The bootstrapper validates the exact runtime identity and applies this action only after the Engine has exited.");
        ImGui::Spacing();
        if (ImGui::Button("Queue for Exit")) {
            runtime::BackendPackMaintenanceRequest request;
            request.action = removing
                ? runtime::BackendPackMaintenanceAction::Remove
                : repairing
                    ? runtime::BackendPackMaintenanceAction::Repair
                    : runtime::BackendPackMaintenanceAction::Rollback;
            request.runtime_set_id = next_runtime.runtime_set_id;
            request.runtime_generation = next_runtime.generation;
            request.backend = backend_pack_maintenance_backend_;
            request.pack_id = backend_pack_maintenance_pack_id_;
            std::string queue_error;
            backend_pack_maintenance_queued_ =
                runtime::QueueBackendPackMaintenanceRequest(
                    runtime_root, request, queue_error);
            backend_pack_maintenance_message_ =
                backend_pack_maintenance_queued_
                    ? "Maintenance queued. Exit the Engine to apply it safely."
                    : "Maintenance was not queued: " + queue_error;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) {
            backend_pack_maintenance_backend_.clear();
            backend_pack_maintenance_pack_id_.clear();
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    return verify_requested;
}

}  // namespace cyxwiz
