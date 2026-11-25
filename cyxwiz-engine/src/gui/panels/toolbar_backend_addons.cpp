// Preferences > Installed packs > Add-ons: the same add-on card as the CyxWiz
// Installer's Individual components (core/addon_pack_presentation). Install,
// repair and uninstall open the installer, which runs the signed pack helper
// (--plugin-pack-id, with --repair, --remove-pack-id); the Engine never
// changes pack files itself.

#include "toolbar.h"

#include "../icons.h"
#include "../pack_tone_colour.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../../core/addon_pack_presentation.h"
#include "../../core/window_manager.h"

#include <imgui.h>

#include <algorithm>
#include <set>

namespace cyxwiz {

void ToolbarPanel::RenderBackendAddOns(
    const std::vector<BackendPackManagerRecord>& records,
    const std::filesystem::path& installer,
    const std::filesystem::path& runtime_root, bool installer_available) {
    // The Engine runs from an installation; a CPU Engine update shown here
    // is applied by the Installer, so the cards use the Engine's words.
    const auto cards = BuildAddOnCards(records, {}, AddOnCardContext::EngineManager,
                                       CyxWizInstallerMode::Maintenance);
    if (cards.empty()) return;
    const ui::Tokens& t = ui::CurrentTokens();
    const auto& style = ImGui::GetStyle();
    constexpr const char* kNoInstaller =
        "The standalone installer is available in packaged desktop builds";

    ImGui::Spacing();
    ImGui::TextUnformatted("Add-ons");
    ImGui::SameLine();
    ImGui::TextDisabled("Optional features. They need no device check.");
    for (const auto& card : cards) {
        ImGui::PushID(card.key.c_str());
        ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_panel);  // a card: tinted, no outline
        ImGui::BeginChild("addon", ImVec2(0.0f, 0.0f),
                          ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding,
                          ImGuiWindowFlags_NoScrollbar);
        const float right = ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x;
        ImGui::TextColored(t.accent_text, "%s", ICON_FA_CUBE);
        ImGui::SameLine();
        ImGui::TextUnformatted(card.title.c_str());
        const float indent = ImGui::GetItemRectMin().x - ImGui::GetWindowPos().x -
                             style.WindowPadding.x;
        const float status_width = ImGui::CalcTextSize(card.status_text.c_str()).x;
        ImGui::SameLine();
        ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(), right - status_width));
        ImGui::TextColored(ui::PackToneColour(card.tone, t), "%s", card.status_text.c_str());

        ImGui::Indent(indent);
        ImGui::PushTextWrapPos(0.0f);
        if (card.status != AddOnCardStatus::NotPublished) {
            ImGui::TextDisabled("%s", card.description.c_str());
        }
        if (!card.facts.empty()) ImGui::TextDisabled("%s", card.facts.c_str());
        if (!card.explanation.empty()) {
            const bool warn = card.status == AddOnCardStatus::NeedsEngineUpdate ||
                              card.status == AddOnCardStatus::NeedsReinstall ||
                              card.status == AddOnCardStatus::UpdateBlocked ||
                              card.status == AddOnCardStatus::Damaged;
            if (warn) ImGui::PushStyleColor(ImGuiCol_Text, t.warning);
            ImGui::TextUnformatted(card.explanation.c_str());
            if (warn) ImGui::PopStyleColor();
        }
        if (!card.licence.empty()) ImGui::TextDisabled("%s", card.licence.c_str());
        if (!card.action_hint.empty()) {
            ImGui::TextColored(t.warning, "%s", card.action_hint.c_str());
        }
        if (!card.error.empty()) ImGui::TextColored(t.error, "%s", card.error.c_str());
        if (!card.files_note.empty()) ImGui::TextDisabled("%s", card.files_note.c_str());
        ImGui::PopTextWrapPos();

        // Row actions: the one that applies, then Details.
        bool action_drawn = false;
        if (card.can_install) {
            const char* label = card.status == AddOnCardStatus::UpdateAvailable
                ? "Update add-on..."
                : card.status == AddOnCardStatus::NeedsReinstall ? "Reinstall add-on..."
                                                                 : "Install add-on...";
            if (ui::SecondaryButton(label, installer_available, kNoInstaller,
                                    ui::ButtonSize::Regular)) {
                core::WindowManager::LaunchExecutable(
                    installer.string(),
                    {"--runtime-root", runtime_root.string(), "--select", card.pack_id});
            }
            if (installer_available && ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Opens the CyxWiz Installer with this add-on selected.");
            }
            action_drawn = true;
        }
        if (card.can_repair) {
            if (action_drawn) ImGui::SameLine();
            if (ui::SecondaryButton("Repair add-on...", installer_available, kNoInstaller,
                                    ui::ButtonSize::Regular)) {
                core::WindowManager::LaunchExecutable(
                    installer.string(),
                    {"--runtime-root", runtime_root.string(), "--select", card.pack_id});
            }
            if (installer_available && ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Opens the CyxWiz Installer with this add-on selected for repair.");
            }
            action_drawn = true;
        }
        if (card.can_uninstall) {
            if (action_drawn) ImGui::SameLine();
            if (ui::DangerButton("Uninstall add-on...", installer_available, kNoInstaller,
                                 ui::ButtonSize::Regular)) {
                core::WindowManager::LaunchExecutable(
                    installer.string(),
                    {"--runtime-root", runtime_root.string(), "--uninstall-pack",
                     card.installed_pack_id});
            }
            if (installer_available && ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Opens the CyxWiz Installer to confirm. The Engine stops loading it at "
                    "its next start.");
            }
            action_drawn = true;
        }
        if (action_drawn) ImGui::SameLine();
        const bool details_open = backend_pack_details_id_ == card.key;
        if (ui::LinkButton(details_open ? "Hide" : "Details")) {
            backend_pack_details_id_ = details_open ? std::string{} : card.key;
        }
        if (details_open &&
            ImGui::BeginTable("details", 4, ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("key1", ImGuiTableColumnFlags_WidthStretch, 1.0f);
            ImGui::TableSetupColumn("value1", ImGuiTableColumnFlags_WidthStretch, 2.0f);
            ImGui::TableSetupColumn("key2", ImGuiTableColumnFlags_WidthStretch, 1.0f);
            ImGui::TableSetupColumn("value2", ImGuiTableColumnFlags_WidthStretch, 2.0f);
            for (std::size_t index = 0; index < card.details.size(); ++index) {
                if (index % 2 == 0) ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextDisabled("%s", card.details[index].first.c_str());
                ImGui::TableNextColumn();
                ImGui::PushTextWrapPos(0.0f);
                ImGui::TextUnformatted(card.details[index].second.c_str());
                ImGui::PopTextWrapPos();
            }
            ImGui::EndTable();
        }
        ImGui::Unindent(indent);
        ImGui::EndChild();
        ImGui::PopStyleColor();
        ImGui::PopID();
    }
}

}  // namespace cyxwiz
