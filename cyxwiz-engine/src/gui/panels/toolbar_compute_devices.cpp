// Preferences > Compute devices: one card per physical device with a row per
// route (tofix119 C). Data comes from the shared presentation model so the
// installer's Verification tab shows the same statuses and wording.
#include "toolbar.h"

#include "../icons.h"
#include "../ui_buttons.h"
#include "../../core/backend_pack_manager_model.h"
#include "../../core/compute_device_presentation.h"
#include "../../core/route_qualification_service.h"
#include "../../core/route_recommendation.h"

#include <algorithm>
#include <string>
#include <vector>

#include <imgui.h>

namespace cyxwiz {
namespace {

struct StatusStyle {
    ImVec4 color;
    const char* icon;
};

StatusStyle StyleFor(ComputeRouteStatus status) {
    switch (status) {
        case ComputeRouteStatus::Verified:
            return {ImVec4(0.24f, 0.84f, 0.55f, 1.0f), ICON_FA_CIRCLE_CHECK};
        case ComputeRouteStatus::Failed:
            return {ImVec4(0.96f, 0.63f, 0.29f, 1.0f), ICON_FA_TRIANGLE_EXCLAMATION};
        case ComputeRouteStatus::NotSupported:
            return {ImVec4(1.0f, 0.48f, 0.45f, 1.0f), ICON_FA_CIRCLE_XMARK};
        case ComputeRouteStatus::NeedsDriverUpdate:
            return {ImVec4(0.90f, 0.75f, 0.29f, 1.0f), ICON_FA_WRENCH};
        case ComputeRouteStatus::Verifying:
            return {ImVec4(0.70f, 0.65f, 1.0f, 1.0f), ICON_FA_SPINNER};
        case ComputeRouteStatus::NotVerifiedYet:
            break;
    }
    return {ImVec4(0.64f, 0.69f, 0.78f, 1.0f), ICON_FA_CLOCK};
}

const char* BackendKey(DeviceType type) {
    switch (type) {
        case DeviceType::CPU: return "cpu";
        case DeviceType::CUDA: return "cuda";
        case DeviceType::OPENCL: return "opencl";
        case DeviceType::ONEAPI: return "oneapi";
        default: return "";
    }
}

void RenderStatus(ComputeRouteStatus status) {
    const auto style = StyleFor(status);
    ImGui::TextColored(style.color, "%s %s", style.icon,
                       ComputeRouteStatusName(status));
}

}  // namespace

void ToolbarPanel::RenderComputeDeviceCards(
    const ComputeDeviceCardsContext& context) {
    const auto snapshot = GetRouteQualificationSnapshot();
    std::string verifying_backend;
    int verifying_device = -1;
    if (context.verification_running && route_qualification_service_) {
        const auto progress = route_qualification_service_->GetProgress();
        verifying_backend = progress.backend;
        verifying_device = progress.device_id;
    }

    std::vector<ComputeRouteInput> inputs;
    std::vector<size_t> input_index;
    for (size_t i = 0; i < cached_devices_.size(); ++i) {
        const auto& cached = cached_devices_[i];
        ComputeRouteInput input;
        input.type = static_cast<DeviceType>(cached.type);
        input.device_id = cached.device_id;

        DeviceInfo device;
        device.type = input.type;
        device.device_id = cached.device_id;
        device.name = cached.name;
        device.name_known = !cached.name.empty();
        device.name_is_fallback =
            cached.name_is_fallback && !cached.name_from_qualification;
        device.memory_total = cached.memory_total;
        device.memory_available = cached.memory_available;
        device.memory_total_known = cached.memory_total_known;
        device.memory_available_known = cached.memory_available_known;
        device.kind = static_cast<DeviceKind>(cached.kind);
        device.identity_confidence =
            static_cast<DeviceIdentityConfidence>(cached.identity_confidence);
        device.provider = cached.provider;
        device.provider_known = cached.provider_known;
        device.driver_version = cached.driver_version;
        device.driver_version_known = cached.driver_version_known;
        device.hardware_vendor_id = cached.hardware_vendor_id;
        device.hardware_vendor_id_known = cached.hardware_vendor_id_known;
        device.pci_location_known = cached.pci_location_known;
        device.pci_domain = cached.pci_domain;
        device.pci_bus = cached.pci_bus;
        device.pci_device = cached.pci_device;
        device.pci_function = cached.pci_function;
        device.physical_fingerprint = cached.physical_fingerprint;
        device.physical_fingerprint_known = cached.physical_fingerprint_known;
        device.metadata_status =
            static_cast<DeviceMetadataStatus>(cached.metadata_status);
        device.device_selectable = cached.device_selectable;
        device.execution_validated = cached.execution_validated;
        input.device = device;

        if (cached.qualification_evidence_available && snapshot.has_value()) {
            for (const auto& record : snapshot->routes) {
                if (record.type == input.type &&
                    record.device_id == input.device_id) {
                    input.evidence = record;
                    break;
                }
            }
        }

        if (input.type == DeviceType::CPU) {
            input.pack_label = "Engine base";
        } else {
            for (const auto& pack : backend_pack_catalog_records_) {
                if (pack.backend == BackendKey(input.type)) {
                    input.pack_label =
                        (pack.backend == "cuda"     ? std::string("CUDA")
                         : pack.backend == "opencl" ? std::string("OpenCL")
                                                    : std::string("oneAPI")) +
                        " pack";
                    if (pack.download_size_bytes > 0) {
                        input.pack_label += " · " +
                            FormatBackendPackByteSize(pack.download_size_bytes);
                    }
                    break;
                }
            }
        }

        ComputeRouteSelectionState selection;
        selection.active = context.is_active && context.is_active(i);
        selection.next_run = context.is_pending && context.is_pending(i);
        selection.saved = context.is_saved && context.is_saved(i);
        selection.selected = selected_device_index_ == static_cast<int>(i);
        selection.training_authorized = cached.training_authorized;
        selection.training_authorization = RouteTrainingAuthorizationStatusName(
            static_cast<RouteTrainingAuthorizationStatus>(
                cached.training_authorization_status));
        selection.qualification_message = cached.qualification_message;
        selection.authorization_message = cached.training_authorization_message;
        input.selection = selection;

        input.verifying = verifying_device == cached.device_id &&
            verifying_backend == BackendKey(input.type);
        input.verification_allowed =
            !context.training_active && !context.verification_running;
        inputs.push_back(std::move(input));
        input_index.push_back(i);
    }

    std::optional<ComputeFastestRoute> fastest;
    if (const auto recommended = RecommendFastestVerifiedRoute(snapshot)) {
        fastest = ComputeFastestRoute{recommended->type, recommended->device_id,
                                      recommended->median_iteration_ms};
    }
    const auto cards = BuildComputeDeviceCards(inputs, fastest);

    // Legend: one vocabulary everywhere.
    ImGui::TextDisabled("Status:");
    for (const auto status :
         {ComputeRouteStatus::Verified, ComputeRouteStatus::NotVerifiedYet,
          ComputeRouteStatus::Failed, ComputeRouteStatus::NotSupported,
          ComputeRouteStatus::NeedsDriverUpdate}) {
        ImGui::SameLine();
        const auto style = StyleFor(status);
        ImGui::TextColored(style.color, "%s", style.icon);
        ImGui::SameLine(0.0f, 4.0f);
        ImGui::TextDisabled("%s", ComputeRouteStatusName(status));
    }
    ImGui::Spacing();

    const auto index_of = [&](const ComputeRouteView& route) -> int {
        for (size_t k = 0; k < inputs.size(); ++k) {
            if (inputs[k].type == route.type &&
                inputs[k].device_id == route.device_id) {
                return static_cast<int>(input_index[k]);
            }
        }
        return -1;
    };
    const float actions_width =
        ui::ButtonWidth("Verify again", ui::ButtonSize::Small) +
        ui::ButtonWidth("Details", ui::ButtonSize::Small) +
        ImGui::GetStyle().ItemSpacing.x * 2.0f;

    for (const auto& card : cards) {
        ImGui::PushID(card.key.c_str());
        ImGui::BeginChild("card", ImVec2(0.0f, 0.0f),
                          ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY |
                              ImGuiChildFlags_AlwaysUseWindowPadding);
        ImGui::TextColored(ImVec4(0.64f, 0.69f, 0.78f, 1.0f), "%s", ICON_FA_MICROCHIP);
        ImGui::SameLine();
        ImGui::Text("%s", card.title.c_str());
        ImGui::SameLine();
        ImGui::TextDisabled("%s", card.subtitle.c_str());
        if (!card.recommended_route.empty()) {
            ImGui::TextColored(ImVec4(0.81f, 0.78f, 1.0f, 1.0f),
                               "%s Recommended for training: %s", ICON_FA_STAR,
                               card.recommended_route.c_str());
        } else {
            ImGui::TextDisabled("No recommended route yet");
        }
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextDisabled("%s", card.recommendation_reason.c_str());
        ImGui::PopTextWrapPos();
        ImGui::Separator();

        for (const auto& route : card.routes) {
            const int index = index_of(route);
            const std::string row_key = card.key + "/" + route.route_label + "/" +
                std::to_string(route.device_id);
            ImGui::PushID(row_key.c_str());
            if (ImGui::BeginTable("route", 4,
                                  ImGuiTableFlags_SizingStretchProp |
                                      (route.recommended ? ImGuiTableFlags_RowBg : 0))) {
                ImGui::TableSetupColumn("route", ImGuiTableColumnFlags_WidthStretch, 1.2f);
                ImGui::TableSetupColumn("status", ImGuiTableColumnFlags_WidthStretch, 1.3f);
                ImGui::TableSetupColumn("summary", ImGuiTableColumnFlags_WidthStretch, 2.8f);
                ImGui::TableSetupColumn("actions", ImGuiTableColumnFlags_WidthFixed,
                                        actions_width);
                ImGui::TableNextRow();

                ImGui::TableNextColumn();
                const bool can_select = route.selectable && index >= 0 &&
                    !context.training_active;
                if (!can_select) ImGui::BeginDisabled();
                if (ImGui::RadioButton("##select",
                                       index >= 0 && selected_device_index_ == index) &&
                    can_select) {
                    context.request_device(static_cast<size_t>(index));
                }
                if (!can_select) ImGui::EndDisabled();
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                    ImGui::SetTooltip("Use this route for the next training run (applied with OK)");
                }
                ImGui::SameLine();
                ImGui::Text("%s", route.route_label.c_str());
                if (!route.pack_label.empty()) {
                    ImGui::TextDisabled("%s", route.pack_label.c_str());
                }

                ImGui::TableNextColumn();
                RenderStatus(route.status);

                ImGui::TableNextColumn();
                ImGui::PushTextWrapPos(0.0f);
                ImGui::TextDisabled("%s", route.summary.c_str());
                if (!route.badges.empty()) {
                    std::string badges;
                    for (const auto& badge : route.badges) {
                        badges += (badges.empty() ? "" : "  ·  ") + badge;
                    }
                    ImGui::TextColored(ImVec4(0.70f, 0.65f, 1.0f, 1.0f), "%s",
                                       badges.c_str());
                }
                ImGui::PopTextWrapPos();

                ImGui::TableNextColumn();
                if (route.can_verify) {
                    if (ui::SecondaryButton(route.verify_label.c_str()) && index >= 0) {
                        pending_route_verify_index_ = index;
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Verify only this route; other results are kept");
                    }
                    ImGui::SameLine();
                }
                const bool open = compute_route_details_open_.count(row_key) > 0;
                if (ui::LinkButton(open ? "Hide" : "Details")) {
                    if (open) compute_route_details_open_.erase(row_key);
                    else compute_route_details_open_.insert(row_key);
                }
                ImGui::EndTable();
            }
            if (compute_route_details_open_.count(row_key) > 0 &&
                ImGui::BeginTable("details", 4,
                                  ImGuiTableFlags_SizingStretchProp |
                                      ImGuiTableFlags_BordersOuter |
                                      ImGuiTableFlags_PadOuterX)) {
                ImGui::TableSetupColumn("k1", ImGuiTableColumnFlags_WidthStretch, 0.8f);
                ImGui::TableSetupColumn("v1", ImGuiTableColumnFlags_WidthStretch, 1.6f);
                ImGui::TableSetupColumn("k2", ImGuiTableColumnFlags_WidthStretch, 0.8f);
                ImGui::TableSetupColumn("v2", ImGuiTableColumnFlags_WidthStretch, 1.6f);
                for (size_t d = 0; d < route.details.size(); ++d) {
                    if (d % 2 == 0) ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::TextDisabled("%s", route.details[d].first.c_str());
                    ImGui::TableNextColumn();
                    ImGui::PushTextWrapPos(0.0f);
                    ImGui::TextUnformatted(route.details[d].second.c_str());
                    ImGui::PopTextWrapPos();
                }
                ImGui::EndTable();
            }
            ImGui::PopID();
        }
        ImGui::EndChild();
        ImGui::PopID();
        ImGui::Spacing();
    }
}

}  // namespace cyxwiz
