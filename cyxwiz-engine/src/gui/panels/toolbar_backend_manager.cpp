#include "toolbar.h"

#include "../icons.h"
#include "../../core/async_task_manager.h"
#include "../../core/backend_pack_manager_model.h"
#include "../../core/route_qualification_snapshot.h"

#include <algorithm>
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
    const auto evidence = GetRouteQualificationSnapshot();
    const bool evidence_matches_runtime =
        identity.has_value() && evidence.has_value() &&
        evidence->runtime_set_id == identity->runtime_set_id &&
        evidence->runtime_generation == identity->generation &&
        evidence->base_pack_id == identity->base_pack_id;

    std::vector<BackendPackManagerRecord> records;
    if (identity.has_value()) {
        const auto add_record = [&](DeviceType type, const std::string& pack_id) {
            BackendPackManagerRecord record;
            record.backend = PackBackendId(type);
            record.pack_id = pack_id;
            record.installed = true;
            record.active = true;
            if (evidence_matches_runtime) {
                for (const auto& route : evidence->routes) {
                    if (route.type != type || route.pack_id != pack_id) continue;
                    record.qualification_evidence_available = true;
                    record.training_authorized =
                        record.training_authorized || route.certified;
                }
            }
            records.push_back(std::move(record));
        };
        add_record(DeviceType::CPU, identity->base_pack_id);
        for (const auto& pack : identity->backend_packs) {
            add_record(pack.type, pack.pack_id);
        }
    }

    BackendPackManagerContext context;
    context.packaged_runtime = identity.has_value();
    context.operation_running = qualification_running;
    context.training_active = training_active;
    // Delivery and maintenance become available only when Preferences is
    // supplied a current signed catalog adapter and the lifecycle service.
    context.catalog_available = false;
    context.delivery_available = false;
    context.maintenance_available = false;
    context.rollback_available = false;

    bool verify_requested = false;
    if (!ImGui::CollapsingHeader(
            ICON_FA_CUBES " Backend Manager",
            ImGuiTreeNodeFlags_DefaultOpen)) {
        return false;
    }

    ImGui::TextDisabled(
        "Browse backend packs here. Device selection below remains unchanged until you click OK.");
    ImGui::Spacing();
    ImGui::Text("Installer choice");
    if (ImGui::RadioButton(
            "Recommended", backend_pack_install_choice_ == 0)) {
        backend_pack_install_choice_ = 0;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton(
            "CPU only", backend_pack_install_choice_ == 1)) {
        backend_pack_install_choice_ = 1;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton(
            "Custom backend packs", backend_pack_install_choice_ == 2)) {
        backend_pack_install_choice_ = 2;
    }
    ImGui::TextDisabled(
        "Recommended never authorizes a route by installation alone; every optional route still needs local verification.");

    if (!identity_error.empty()) {
        ImGui::TextColored(
            ImVec4(1.0f, 0.55f, 0.35f, 1.0f),
            "%s %s", ICON_FA_TRIANGLE_EXCLAMATION,
            identity_error.c_str());
    } else if (!identity.has_value()) {
        ImGui::TextDisabled(
            "Development runtime: signed backend-pack installation and maintenance are unavailable.");
    } else {
        ImGui::TextDisabled(
            "Runtime set %s, generation %llu",
            identity->runtime_set_id.c_str(),
            static_cast<unsigned long long>(identity->generation));
    }
    ImGui::TextColored(
        ImVec4(1.0f, 0.75f, 0.35f, 1.0f),
        "%s No current signed catalog is connected. Install, repair, update, remove, and rollback stay disabled.",
        ICON_FA_CIRCLE_INFO);

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
            ImGui::TableNextColumn();
            ImGui::TextColored(
                record.active
                    ? ImVec4(0.35f, 0.95f, 0.45f, 1.0f)
                    : ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                "%s", record.active ? "Active" : "Installed");
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
            RenderActionButton(
                "Repair", EvaluateBackendPackAction(
                    BackendPackAction::Repair, context, &record));
            ImGui::SameLine();
            RenderActionButton(
                "Update", EvaluateBackendPackAction(
                    BackendPackAction::Update, context, &record));
            ImGui::SameLine();
            RenderActionButton(
                "Remove", EvaluateBackendPackAction(
                    BackendPackAction::Remove, context, &record));
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
    RenderActionButton(ICON_FA_ROTATE_LEFT " Rollback", rollback);
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    return verify_requested;
}

}  // namespace cyxwiz
