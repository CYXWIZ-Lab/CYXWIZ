#include "permission_dialog.h"
#include <imgui.h>
#include <spdlog/spdlog.h>

namespace cyxwiz::plugin::security {

bool PermissionDialog::RequestApproval(
    const PluginManifest& manifest,
    const std::vector<PluginPermission>& undecided_perms)
{
    if (undecided_perms.empty()) return false;

    PendingApproval pending;
    pending.plugin_id = manifest.id;
    pending.plugin_name = manifest.name;
    pending.plugin_version = manifest.version.ToString();
    pending.plugin_author = manifest.author;
    pending.permissions = undecided_perms;
    pending.allowed.resize(undecided_perms.size(), false);

    current_ = std::move(pending);
    open_requested_ = true;

    spdlog::info("PermissionDialog: Requesting approval for plugin '{}' ({} permissions)",
                 manifest.name, undecided_perms.size());
    return true;
}

bool PermissionDialog::Render() {
    if (!current_.has_value() || current_->resolved) return false;

    if (open_requested_) {
        ImGui::OpenPopup("Plugin Permissions###PluginPermDialog");
        open_requested_ = false;
    }

    bool decision_made = false;

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(480, 0), ImGuiCond_Appearing);

    if (ImGui::BeginPopupModal("Plugin Permissions###PluginPermDialog", nullptr,
                                ImGuiWindowFlags_AlwaysAutoResize)) {
        auto& pending = *current_;

        // Header
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "Plugin Permission Request");
        ImGui::Separator();
        ImGui::Spacing();

        // Plugin info
        ImGui::Text("Plugin: %s", pending.plugin_name.c_str());
        if (!pending.plugin_author.empty())
            ImGui::Text("Author: %s", pending.plugin_author.c_str());
        ImGui::Text("Version: %s", pending.plugin_version.c_str());
        ImGui::Spacing();

        ImGui::TextWrapped("This plugin requests the following permissions:");
        ImGui::Spacing();

        // Permission checkboxes
        for (size_t i = 0; i < pending.permissions.size(); ++i) {
            auto perm = pending.permissions[i];
            const char* name = PermissionStore::PermissionDisplayName(perm);
            const char* desc = PermissionStore::PermissionDescription(perm);

            // std::vector<bool> uses proxy refs; need a temp bool for ImGui
            bool checked = pending.allowed[i];
            ImGui::Checkbox(name, &checked);
            pending.allowed[i] = checked;
            ImGui::SameLine();
            ImGui::TextDisabled("(?)");
            if (ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                ImGui::TextUnformatted(desc);
                ImGui::EndTooltip();
            }
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Buttons
        float button_width = 140.0f;
        float spacing = ImGui::GetStyle().ItemSpacing.x;
        float total = button_width * 2 + spacing;
        ImGui::SetCursorPosX((ImGui::GetWindowWidth() - total) * 0.5f);

        if (ImGui::Button("Allow Selected", ImVec2(button_width, 0))) {
            pending.accepted = true;
            pending.resolved = true;
            decision_made = true;
            spdlog::info("PermissionDialog: User allowed selected permissions for '{}'",
                         pending.plugin_name);
            ImGui::CloseCurrentPopup();
        }

        ImGui::SameLine();

        if (ImGui::Button("Deny All", ImVec2(button_width, 0))) {
            pending.accepted = false;
            pending.resolved = true;
            // Set all to denied
            std::fill(pending.allowed.begin(), pending.allowed.end(), false);
            decision_made = true;
            spdlog::info("PermissionDialog: User denied all permissions for '{}'",
                         pending.plugin_name);
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }

    if (decision_made && callback_) {
        callback_(*current_);
    }

    return decision_made;
}

} // namespace cyxwiz::plugin::security
