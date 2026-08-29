#include "installer_removal_view.h"

#include "gui/icons.h"

#include <imgui.h>

namespace cyxwiz::installer::gui {
namespace {

constexpr ImVec4 kDanger = ImVec4(1.0f, 0.38f, 0.43f, 1.0f);

} // namespace

InstallerRemovalViewAction RenderInstallerRemovalControl(
    InstallerRemovalViewState &view_state,
    const InstallerProductRemovalState &removal,
    bool operation_running) {
  InstallerRemovalViewAction action = InstallerRemovalViewAction::None;
  if (removal.requires_stable_host) {
    ImGui::BeginDisabled(operation_running);
    if (ImGui::Button("Open installed manager to uninstall",
                      ImVec2(-1.0f, 34.0f))) {
      action = InstallerRemovalViewAction::OpenInstalledManager;
    }
    ImGui::EndDisabled();
    if (!removal.message.empty()) {
      ImGui::TextDisabled("%s", removal.message.c_str());
    }
    return action;
  }
  ImGui::BeginDisabled(operation_running || !removal.available);
  if (ImGui::Button(ICON_FA_TRASH " Uninstall CyxWiz",
                    ImVec2(-1.0f, 34.0f))) {
    view_state.acknowledged = false;
    ImGui::OpenPopup("Remove CyxWiz");
  }
  ImGui::EndDisabled();
  if (!removal.message.empty()) {
    ImGui::TextDisabled("%s", removal.message.c_str());
  }

  ImGui::SetNextWindowSize(ImVec2(560.0f, 0.0f), ImGuiCond_Appearing);
  if (!ImGui::BeginPopupModal("Remove CyxWiz", nullptr,
                              ImGuiWindowFlags_AlwaysAutoResize)) {
    return action;
  }
  ImGui::TextColored(kDanger, "%s Uninstall this CyxWiz installation?",
                     ICON_FA_TRIANGLE_EXCLAMATION);
  ImGui::TextWrapped(
      "The Engine, Installer, CPU runtime, and every installed backend pack "
      "under this installation root will be permanently removed.");
  const auto root = removal.install_root.u8string();
  ImGui::Spacing();
  ImGui::TextDisabled("Installation root");
  ImGui::TextWrapped("%s", reinterpret_cast<const char *>(root.c_str()));
  ImGui::Spacing();
  ImGui::TextWrapped(
      "Projects and datasets stored outside this folder are not removed.");
  ImGui::Checkbox("I understand that this cannot be undone",
                  &view_state.acknowledged);
  ImGui::BeginDisabled(!view_state.acknowledged || operation_running);
  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.62f, 0.12f, 0.17f, 1.0f));
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                        ImVec4(0.76f, 0.16f, 0.22f, 1.0f));
  if (ImGui::Button("Uninstall", ImVec2(175.0f, 36.0f))) {
    action = InstallerRemovalViewAction::RemoveProduct;
    ImGui::CloseCurrentPopup();
  }
  ImGui::PopStyleColor(2);
  ImGui::EndDisabled();
  ImGui::SameLine();
  if (ImGui::Button("Cancel", ImVec2(110.0f, 36.0f))) {
    view_state.acknowledged = false;
    ImGui::CloseCurrentPopup();
  }
  ImGui::EndPopup();
  return action;
}

} // namespace cyxwiz::installer::gui
