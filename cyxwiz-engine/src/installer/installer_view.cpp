#include "installer_view.h"

#include "core/installer_pack_presentation.h"
#include "gui/icons.h"

#include <imgui.h>

#include <algorithm>
#include <cstring>
#include <string_view>
#include <vector>

namespace cyxwiz::installer::gui {
namespace {

const ImVec4 kAccent = InstallerBrandAccent();
constexpr ImVec4 kSuccess = ImVec4(0.25f, 0.82f, 0.56f, 1.0f);
constexpr ImVec4 kWarning = ImVec4(1.0f, 0.68f, 0.28f, 1.0f);
constexpr ImVec4 kDanger = ImVec4(1.0f, 0.38f, 0.43f, 1.0f);

const char *BackendName(std::string_view backend) {
  if (backend == "cpu")
    return "CPU Engine";
  if (backend == "cuda")
    return "NVIDIA CUDA";
  if (backend == "opencl")
    return "OpenCL";
  if (backend == "oneapi")
    return "Intel oneAPI";
  return "Compute backend";
}

const char *BackendIcon(std::string_view backend) {
  if (backend == "cpu")
    return ICON_FA_MICROCHIP;
  if (backend == "cuda")
    return ICON_FA_BOLT;
  if (backend == "opencl")
    return ICON_FA_LAYER_GROUP;
  if (backend == "oneapi")
    return ICON_FA_CUBES;
  return ICON_FA_SERVER;
}

const char *BackendDescription(std::string_view backend) {
  if (backend == "cpu") {
    return "Required CyxWiz Engine and ArrayFire CPU runtime.";
  }
  if (backend == "cuda") {
    return "Accelerated execution for supported NVIDIA GPUs.";
  }
  if (backend == "opencl") {
    return "Portable acceleration through a compatible OpenCL provider.";
  }
  if (backend == "oneapi") {
    return "Qualified Intel GPU and accelerator execution.";
  }
  return "Optional signed compute package.";
}

ImVec4 PackToneColor(InstallerPackPresentationTone tone) {
  switch (tone) {
  case InstallerPackPresentationTone::Success:
    return kSuccess;
  case InstallerPackPresentationTone::Warning:
    return kWarning;
  case InstallerPackPresentationTone::Danger:
    return kDanger;
  case InstallerPackPresentationTone::Accent:
    return kAccent;
  case InstallerPackPresentationTone::Neutral:
    return ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled);
  }
  return kAccent;
}

std::string Join(const std::vector<std::string> &values) {
  if (values.empty())
    return "No external provider requirement";
  std::string result;
  for (const auto &value : values) {
    if (!result.empty())
      result += ", ";
    result += value;
  }
  return result;
}

std::vector<std::string>
SelectedPackIds(const std::set<std::string> &selected) {
  return {selected.begin(), selected.end()};
}

void CenteredDisabledWrappedText(const char *text, float wrap_width,
                                 float area_width) {
  if (!text || *text == '\0')
    return;
  ImGui::PushStyleColor(ImGuiCol_Text,
                        ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
  const char *line_start = text;
  while (*line_start != '\0') {
    while (*line_start == ' ')
      ++line_start;
    if (*line_start == '\0')
      break;

    const char *line_end = line_start;
    const char *scan = line_start;
    while (*scan != '\0') {
      const char *word_end = scan;
      while (*word_end != '\0' && *word_end != ' ')
        ++word_end;
      const float candidate_width =
          ImGui::CalcTextSize(line_start, word_end).x;
      if (line_end != line_start && candidate_width > wrap_width)
        break;
      line_end = word_end;
      scan = word_end;
      while (*scan == ' ')
        ++scan;
    }

    const float line_width = ImGui::CalcTextSize(line_start, line_end).x;
    ImGui::SetCursorPosX(std::max(ImGui::GetStyle().WindowPadding.x,
                                  (area_width - line_width) * 0.5f));
    ImGui::TextUnformatted(line_start, line_end);
    line_start = line_end;
  }
  ImGui::PopStyleColor();
}

bool WorkloadCard(const char *id, const char *icon, const char *title,
                  const char *description, bool selected, bool enabled,
                  const ImVec2 &size) {
  ImGui::PushID(id);
  ImGui::PushStyleColor(ImGuiCol_ChildBg,
                        selected ? ImVec4(0.19f, 0.15f, 0.32f, 1.0f)
                                 : ImVec4(0.055f, 0.115f, 0.19f, 1.0f));
  ImGui::BeginChild("card", size, ImGuiChildFlags_None,
                    ImGuiWindowFlags_NoScrollbar |
                        ImGuiWindowFlags_NoScrollWithMouse);
  ImGui::BeginDisabled(!enabled);
  ImGui::Dummy(ImVec2(0.0f, 12.0f));
  const auto &style = ImGui::GetStyle();
  const float heading_width = ImGui::CalcTextSize(icon).x +
                              style.ItemInnerSpacing.x +
                              ImGui::CalcTextSize(title).x;
  ImGui::SetCursorPosX(std::max(style.WindowPadding.x,
                                (size.x - heading_width) * 0.5f));
  ImGui::TextColored(
      selected ? kAccent : ImGui::GetStyleColorVec4(ImGuiCol_Text), "%s", icon);
  ImGui::SameLine(0.0f, style.ItemInnerSpacing.x);
  ImGui::TextUnformatted(title);
  ImGui::Dummy(ImVec2(0.0f, 8.0f));
  const float description_width = std::max(1.0f, size.x - 38.0f);
  CenteredDisabledWrappedText(description, description_width, size.x);
  const bool clicked = enabled && ImGui::IsWindowHovered() &&
                       ImGui::IsMouseClicked(ImGuiMouseButton_Left);
  ImGui::EndDisabled();
  ImGui::EndChild();
  ImGui::PopStyleColor();
  ImGui::PopID();
  return clicked;
}

void RenderWorkloads(InstallerViewState &state,
                     const InstallerCatalogState &catalog,
                     bool operation_running) {
  ImGui::Text("Choose an installation profile");
  ImGui::TextDisabled("The CPU Engine is always included. Optional backends "
                      "are activated only after local verification.");
  ImGui::Dummy(ImVec2(0.0f, 12.0f));

  constexpr float kWorkloadGutter = 14.0f;
  constexpr float kCardGap = 14.0f;
  const float available_width = ImGui::GetContentRegionAvail().x;
  const float card_width = std::clamp(
      (available_width - kWorkloadGutter * 2.0f - kCardGap) / 2.0f, 220.0f,
      310.0f);
  const float row_width = card_width * 2.0f + kCardGap;
  const ImVec2 size(card_width, 128.0f);
  ImGui::SetCursorPosX(
      ImGui::GetCursorPosX() +
      std::max(kWorkloadGutter, (available_width - row_width) * 0.5f));
  if (WorkloadCard("recommended", ICON_FA_GAUGE_HIGH, "Recommended",
                   "Best comparable verified configuration, or CPU when no "
                   "optional route qualifies.",
                   state.choice == BackendPackInstallChoice::Recommended,
                   !operation_running, size)) {
    state.choice = BackendPackInstallChoice::Recommended;
  }
  ImGui::SameLine(0.0f, kCardGap);
  if (WorkloadCard(
          "cpu", ICON_FA_MICROCHIP, "CPU only",
          "Smallest reliable installation for every supported machine.",
          state.choice == BackendPackInstallChoice::CpuOnly, !operation_running,
          size)) {
    state.choice = BackendPackInstallChoice::CpuOnly;
  }
  ImGui::Dummy(ImVec2(0.0f, 14.0f));
  ImGui::Text("What will be installed");
  for (const auto &record : catalog.records) {
    const bool included =
        record.backend == "cpu" ||
        (state.choice == BackendPackInstallChoice::Recommended &&
         IsBackendPackRecommended(record)) ||
        (state.choice == BackendPackInstallChoice::Custom &&
         state.custom_selection.contains(record.pack_id));
    if (!included)
      continue;
    const auto presentation = BuildInstallerPackPresentation(record);
    ImGui::BulletText("%s %s  %s", BackendIcon(record.backend),
                      BackendName(record.backend),
                      presentation.status.c_str());
  }
}

void RenderComponents(InstallerViewState &state,
                      const InstallerCatalogState &catalog,
                      bool operation_running) {
  ImGui::Text("Individual components");
  ImGui::TextDisabled("Availability comes from the verified catalog. Driver "
                      "presence is confirmed during qualification.");
  ImGui::Spacing();

  if (catalog.records.empty()) {
    ImGui::TextColored(kWarning, "%s No components are available",
                       ICON_FA_TRIANGLE_EXCLAMATION);
    ImGui::TextWrapped("%s", catalog.message.c_str());
    return;
  }

  for (const auto &record : catalog.records) {
    ImGui::PushID(record.pack_id.c_str());
    ImGui::PushStyleColor(ImGuiCol_ChildBg,
                          ImVec4(0.055f, 0.115f, 0.19f, 1.0f));
    const float component_height = record.backend == "cuda"
                                       ? (record.installed ? 222.0f : 184.0f)
                                       : (record.installed ? 192.0f : 154.0f);
    ImGui::BeginChild("component", ImVec2(0.0f, component_height),
                      ImGuiChildFlags_None, ImGuiWindowFlags_NoScrollbar);
    const bool selectable = IsBackendPackSelectableForInstaller(record);
    const auto presentation = BuildInstallerPackPresentation(record);
    bool selected = record.backend == "cpu" ||
                    (record.installed && !record.update_available) ||
                    state.custom_selection.contains(record.pack_id);
    const bool can_select = selectable &&
                            (!record.installed || record.update_available);
    ImGui::BeginDisabled(!can_select || operation_running);
    if (ImGui::Checkbox("##selected", &selected)) {
      state.choice = BackendPackInstallChoice::Custom;
      if (selected)
        state.custom_selection.insert(record.pack_id);
      else
        state.custom_selection.erase(record.pack_id);
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::TextColored(kAccent, "%s", BackendIcon(record.backend));
    ImGui::SameLine();
    ImGui::TextUnformatted(BackendName(record.backend));
    if (record.backend == "cpu") {
      ImGui::SameLine();
      ImGui::TextDisabled("Required");
    }
    if (IsBackendPackRecommended(record)) {
      ImGui::SameLine();
      ImGui::TextColored(kSuccess, "Recommended");
    }
    ImGui::SameLine(ImGui::GetWindowWidth() - 175.0f);
    ImGui::TextColored(PackToneColor(presentation.tone), "%s",
                       presentation.status.c_str());
    ImGui::Indent(29.0f);
    ImGui::TextDisabled("%s", BackendDescription(record.backend));
    const auto size = FormatBackendPackByteSize(record.download_size_bytes);
    const auto providers = Join(record.provider_requirements);
    ImGui::TextDisabled("Version %s  |  %s  |  Requires: %s",
                        record.package_version.empty()
                            ? "not published"
                            : record.package_version.c_str(),
                        size.c_str(), providers.c_str());
    ImGui::TextWrapped("%s", presentation.explanation.c_str());
    if (record.backend == "cuda" &&
        !catalog.cuda_prerequisite.message.empty()) {
      ImGui::TextColored(
          catalog.cuda_prerequisite.device_available ? kSuccess : kWarning,
          "%s", catalog.cuda_prerequisite.message.c_str());
    }
    if (!presentation.action.empty()) {
      ImGui::PushStyleColor(ImGuiCol_Text, kWarning);
      ImGui::TextWrapped("%s", presentation.action.c_str());
      ImGui::PopStyleColor();
    }
    if (!record.delivery_metadata_error.empty()) {
      ImGui::TextColored(kDanger, "%s", record.delivery_metadata_error.c_str());
    }
    if (record.installed && record.backend != "cpu" &&
        !record.update_available) {
      ImGui::BeginDisabled(operation_running);
      if (ImGui::Button(ICON_FA_TRASH " Uninstall component",
                        ImVec2(190.0f, 30.0f))) {
        state.pending_pack_removal_id = record.installed_pack_id;
        state.pack_removal_confirmation_requested = true;
      }
      ImGui::EndDisabled();
    } else if (record.update_available) {
      ImGui::TextColored(
          kWarning,
          "Select this component to review and install its signed update.");
    }
    ImGui::Unindent(29.0f);
    ImGui::EndChild();
    ImGui::PopStyleColor();
    ImGui::PopID();
  }

  for (const std::string_view backend : {"cuda", "opencl", "oneapi"}) {
    const bool published = std::any_of(
        catalog.records.begin(), catalog.records.end(),
        [&](const BackendPackManagerRecord &record) {
          return record.backend == backend;
        });
    if (published)
      continue;
    ImGui::PushID(backend.data());
    ImGui::PushStyleColor(ImGuiCol_ChildBg,
                          ImVec4(0.055f, 0.115f, 0.19f, 1.0f));
    const float component_height =
        backend == "cuda" ? 166.0f : 126.0f;
    ImGui::BeginChild("component-unavailable",
                      ImVec2(0.0f, component_height),
                      ImGuiChildFlags_None, ImGuiWindowFlags_NoScrollbar);
    bool selected = false;
    ImGui::BeginDisabled();
    ImGui::Checkbox("##selected", &selected);
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::TextColored(kAccent, "%s", BackendIcon(backend));
    ImGui::SameLine();
    ImGui::TextUnformatted(BackendName(backend));
    ImGui::SameLine(ImGui::GetWindowWidth() - 175.0f);
    ImGui::TextColored(kWarning, "Not published");
    ImGui::Indent(29.0f);
    ImGui::TextDisabled("%s", BackendDescription(backend));
    ImGui::TextWrapped(
        "No signed %s package is published for this platform and release.",
        BackendName(backend));
    if (backend == "cuda" &&
        !catalog.cuda_prerequisite.message.empty()) {
      ImGui::TextColored(
          catalog.cuda_prerequisite.device_available ? kSuccess : kWarning,
          "%s", catalog.cuda_prerequisite.message.c_str());
    }
    ImGui::Unindent(29.0f);
    ImGui::EndChild();
    ImGui::PopStyleColor();
    ImGui::PopID();
  }
}

void RenderPackRemovalConfirmation(InstallerViewState &state,
                                   bool operation_running,
                                   InstallerViewAction &action) {
  if (state.pack_removal_confirmation_requested) {
    ImGui::OpenPopup("Uninstall backend component");
    state.pack_removal_confirmation_requested = false;
  }
  ImGui::SetNextWindowSize(ImVec2(520.0f, 0.0f), ImGuiCond_Appearing);
  if (!ImGui::BeginPopupModal("Uninstall backend component", nullptr,
                              ImGuiWindowFlags_AlwaysAutoResize)) {
    return;
  }
  ImGui::TextWrapped(
      "CyxWiz will deactivate this compute route, remove its versioned "
      "package files, and clear its downloaded package cache.");
  ImGui::Spacing();
  ImGui::TextDisabled("Component: %s",
                      state.pending_pack_removal_id.c_str());
  ImGui::Spacing();
  ImGui::BeginDisabled(operation_running ||
                       state.pending_pack_removal_id.empty());
  if (ImGui::Button("Uninstall", ImVec2(175.0f, 36.0f))) {
    action.kind = InstallerViewActionKind::ApplyPlan;
    action.plan.valid = true;
    action.plan.remove_pack_ids = {state.pending_pack_removal_id};
    state.custom_selection.erase(state.pending_pack_removal_id);
    state.choice = BackendPackInstallChoice::Custom;
    state.pending_pack_removal_id.clear();
    ImGui::CloseCurrentPopup();
  }
  ImGui::EndDisabled();
  ImGui::SameLine();
  if (ImGui::Button("Keep component", ImVec2(175.0f, 36.0f))) {
    state.pending_pack_removal_id.clear();
    ImGui::CloseCurrentPopup();
  }
  ImGui::EndPopup();
}

void RenderLocation(InstallerViewState &state,
                    const InstallerCatalogState &catalog,
                    const CyxWizInstallLocation &install_location,
                    bool operation_running, InstallerViewAction &action) {
  ImGui::Text("Installation location");
  ImGui::TextDisabled("CyxWiz uses versioned runtimes and never modifies the "
                      "global loader path.");
  ImGui::Spacing();
  if (catalog.mode == CyxWizInstallerMode::FreshInstall) {
    ImGui::SetNextItemWidth(-140.0f);
    if (ImGui::InputText("##install-location", state.install_path_text.data(),
                         state.install_path_text.size())) {
      state.install_location_dirty = true;
    }
    ImGui::SameLine();
    ImGui::BeginDisabled(operation_running);
    if (ImGui::Button("Use location")) {
      action.kind = InstallerViewActionKind::UseInstallLocation;
      action.install_root = std::filesystem::path(std::u8string(
          reinterpret_cast<const char8_t *>(state.install_path_text.data())));
      action.scope = state.install_scope;
    }
    ImGui::EndDisabled();

    int scope = static_cast<int>(state.install_scope);
    ImGui::BeginDisabled(operation_running);
    ImGui::RadioButton("Install for me", &scope, 0);
    ImGui::SameLine();
    ImGui::RadioButton("Install for all users", &scope, 1);
    ImGui::EndDisabled();
    const auto next_scope = static_cast<CyxWizInstallScope>(scope);
    if (next_scope != state.install_scope) {
      state.install_scope = next_scope;
      const auto default_root = DefaultCyxWizInstallRoot(next_scope);
      const auto value = default_root.u8string();
      if (value.size() < state.install_path_text.size()) {
        state.install_path_text.fill('\0');
        std::memcpy(state.install_path_text.data(), value.data(), value.size());
      }
      state.install_location_dirty = true;
      state.install_location_message =
          next_scope == CyxWizInstallScope::AllUsers
              ? "System-wide installation requires platform authorization"
              : "Current-user installation is the recommended least-privilege "
                "choice";
      action.kind = InstallerViewActionKind::UseInstallLocation;
      action.install_root = default_root;
      action.scope = next_scope;
    }
  } else {
    const auto value = install_location.install_root.u8string();
    ImGui::TextUnformatted(reinterpret_cast<const char *>(value.c_str()));
    ImGui::TextDisabled(
        "The location remains fixed while modifying this installation.");
  }
  ImGui::Spacing();
  ImGui::TextColored(state.install_location_dirty ? kWarning : kSuccess,
                     "%s %s",
                     state.install_location_dirty ? ICON_FA_TRIANGLE_EXCLAMATION
                                                  : ICON_FA_CIRCLE_CHECK,
                     state.install_location_message.c_str());
}

void RenderVerificationSummary(const InstallerVerificationSummary &summary) {
  if (!summary.evidence_available)
    return;
  ImGui::Text("Verification results");
  ImGui::TextWrapped("%s", summary.headline.c_str());
  ImGui::TextDisabled("%s", summary.performance_message.c_str());
  if (!summary.evidence_matches_runtime || summary.routes.empty())
    return;

  if (ImGui::BeginTable("verification", 4,
                        ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_RowBg |
                            ImGuiTableFlags_Resizable)) {
    ImGui::TableSetupColumn("Route");
    ImGui::TableSetupColumn("Result");
    ImGui::TableSetupColumn("Reason");
    ImGui::TableSetupColumn("Benchmark");
    ImGui::TableHeadersRow();
    for (const auto &route : summary.routes) {
      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0);
      ImGui::Text("%s device %d", route.backend.c_str(), route.device_id);
      if (!route.display_name.empty()) {
        ImGui::TextDisabled("%s", route.display_name.c_str());
      }
      ImGui::TableSetColumnIndex(1);
      ImGui::TextUnformatted(
          InstallerRouteVerificationStatusName(route.status));
      ImGui::TableSetColumnIndex(2);
      ImGui::TextWrapped("%s", route.reason.c_str());
      if (!route.recommended_action.empty()) {
        ImGui::TextDisabled("%s", route.recommended_action.c_str());
      }
      ImGui::TableSetColumnIndex(3);
      if (route.benchmark_available) {
        ImGui::Text("%.3f ms", route.benchmark_median_iteration_ms);
        if (route.best_measured) {
          ImGui::TextColored(kSuccess, "Best measured");
        }
      } else {
        ImGui::TextDisabled("Not available");
      }
    }
    ImGui::EndTable();
  }
}

void RenderHeader(const InstallerCatalogState &catalog,
                  const std::string &platform_name, bool operation_running,
                  const InstallerVisualAssets &assets,
  InstallerViewAction &action) {
  if (assets.logo_texture != 0) {
    constexpr float avatar_diameter = 88.0f;
    const ImVec2 avatar_size(avatar_diameter, avatar_diameter);
    ImGui::Dummy(avatar_size);
    ImGui::GetWindowDrawList()->AddImageRounded(
        assets.logo_texture, ImGui::GetItemRectMin(), ImGui::GetItemRectMax(),
        ImVec2(0.0f, 0.0f), ImVec2(1.0f, 1.0f), IM_COL32_WHITE,
        avatar_diameter * 0.5f);
    ImGui::SameLine();
  }
  ImGui::BeginGroup();
  if (assets.heading_font)
    ImGui::PushFont(assets.heading_font);
  ImGui::TextUnformatted("CyxWiz Installer");
  if (assets.heading_font)
    ImGui::PopFont();
  ImGui::TextDisabled("%s  |  %s", platform_name.c_str(),
                      catalog.mode == CyxWizInstallerMode::FreshInstall
                          ? "Fresh installation"
                          : "Modify installation");
  ImGui::EndGroup();

  const float button_width = 154.0f;
  ImGui::SameLine(ImGui::GetWindowWidth() - button_width - 26.0f);
  ImGui::BeginGroup();
  ImGui::TextColored(
      catalog.available ? kSuccess : kWarning, "%s %s",
      catalog.available ? ICON_FA_SHIELD : ICON_FA_TRIANGLE_EXCLAMATION,
      catalog.available ? "Catalog verified" : "Catalog unavailable");
  ImGui::BeginDisabled(operation_running);
  const char *refresh_label =
      catalog.mode == CyxWizInstallerMode::Maintenance
          ? ICON_FA_ROTATE " Check for updates"
          : ICON_FA_ROTATE " Refresh catalog";
  if (ImGui::Button(refresh_label,
                    ImVec2(button_width, 0.0f))) {
    action.kind = InstallerViewActionKind::RefreshCatalog;
  }
  ImGui::EndDisabled();
  ImGui::EndGroup();
}

void RenderSummary(InstallerViewState &state,
                   const InstallerCatalogState &catalog,
                   const CyxWizInstallLocation &install_location,
                   const InstallerProductRemovalState &product_removal,
                   bool operation_running, bool operation_cancellable,
                   const std::string &operation_message,
                   const InstallerPlanExecutionProgress &operation_progress,
                   InstallerViewAction &action) {
  const auto selection = ResolveBackendPackInstallerSelection(
      state.choice, catalog.records, SelectedPackIds(state.custom_selection));
  const auto plan =
      BuildBackendPackInstallerPlan(selection, catalog.records, catalog.mode);

  ImGui::TextColored(kAccent, "%s Installation summary", ICON_FA_CUBES);
  ImGui::Spacing();
  ImGui::TextDisabled("Profile");
  ImGui::TextUnformatted(
      state.choice == BackendPackInstallChoice::Recommended ? "Recommended"
      : state.choice == BackendPackInstallChoice::CpuOnly   ? "CPU only"
                                                            : "Custom");
  ImGui::Spacing();
  ImGui::TextDisabled("Components");
  ImGui::BulletText("%s", plan.update_base
                                  ? "CPU Engine (update available)"
                                  : "CPU Engine (required)");
  for (const auto &pack_id : selection.pack_ids) {
    const auto record =
        std::find_if(catalog.records.begin(), catalog.records.end(),
                     [&](const BackendPackManagerRecord &candidate) {
                       return candidate.pack_id == pack_id;
                     });
    if (record != catalog.records.end()) {
      ImGui::BulletText("%s", BackendName(record->backend));
    }
  }
  ImGui::Spacing();
  ImGui::TextDisabled("Required download");
  const auto size = FormatBackendPackByteSize(plan.download_size_bytes);
  ImGui::TextUnformatted(plan.download_size_bytes == 0 ? "No download required"
                                                       : size.c_str());
  ImGui::Spacing();
  ImGui::TextDisabled("Install location");
  const auto root = install_location.install_root.u8string();
  ImGui::TextWrapped("%s", reinterpret_cast<const char *>(root.c_str()));
  ImGui::Spacing();
  ImGui::TextWrapped("%s", plan.message.c_str());

  if (operation_running) {
    const float progress = std::clamp(
        operation_progress.overall_fraction, 0.0f, 1.0f);
    std::string progress_label = "Preparing";
    if (operation_progress.total_steps != 0) {
      progress_label = std::to_string(
                           static_cast<int>(progress * 100.0f)) +
                       "% overall";
    }
    ImGui::ProgressBar(progress, ImVec2(-1.0f, 0.0f),
                       progress_label.c_str());
    if (operation_progress.package_count != 0) {
      std::string package_text =
          "Package " + std::to_string(operation_progress.package_index) +
          " of " + std::to_string(operation_progress.package_count);
      if (!operation_progress.package_id.empty()) {
        package_text += ": " + operation_progress.package_id;
      }
      ImGui::TextWrapped("%s", package_text.c_str());
    }
    if (operation_progress.phase_count != 0) {
      std::string phase_text =
          "Phase " + std::to_string(operation_progress.phase_index) +
          " of " + std::to_string(operation_progress.phase_count);
      if (!operation_progress.phase_label.empty()) {
        phase_text += ": " + operation_progress.phase_label;
      }
      ImGui::TextUnformatted(phase_text.c_str());
    }
    if (operation_progress.total_bytes != 0) {
      const auto completed =
          operation_progress.completed_bytes == 0
              ? std::string("0 B")
              : FormatBackendPackByteSize(
                    operation_progress.completed_bytes);
      const auto total =
          FormatBackendPackByteSize(operation_progress.total_bytes);
      ImGui::TextDisabled("%s / %s", completed.c_str(), total.c_str());
    }
    if (operation_progress.component_count != 0) {
      ImGui::TextDisabled(
          "Files: %zu of %zu", operation_progress.component_index,
          operation_progress.component_count);
    }
    if (!operation_progress.activity.empty()) {
      ImGui::TextWrapped("%s", operation_progress.activity.c_str());
    }
  }
  if (!operation_message.empty()) {
    ImGui::TextWrapped("%s", operation_message.c_str());
  }

  const bool has_changes = plan.install_base || plan.update_base ||
                           !plan.pack_ids.empty() ||
                           !plan.deactivate_backends.empty();
  const bool can_apply = plan.valid && has_changes &&
                         (plan.pack_ids.empty() || catalog.available) &&
                         !state.install_location_dirty &&
                         install_location.valid && !operation_running;
  const bool maintenance =
      catalog.mode == CyxWizInstallerMode::Maintenance;
  const bool show_launch = maintenance || state.install_completed;
  const float footer_height = maintenance
                                  ? (has_changes ? 228.0f : 184.0f)
                                  : (state.install_completed ? 160.0f
                                                             : 116.0f);
  ImGui::SetCursorPosY(std::max(
      ImGui::GetCursorPosY(), ImGui::GetWindowHeight() - footer_height));
  if (state.install_completed) {
    ImGui::TextColored(kSuccess, "%s Installation complete",
                       ICON_FA_CIRCLE_CHECK);
  }
  if (show_launch) {
    ImGui::BeginDisabled(operation_running);
    if (ImGui::Button(
            state.engine_launched ? "Launch CyxWiz again" : "Launch CyxWiz",
            ImVec2(-1.0f, 38.0f))) {
      action.kind = InstallerViewActionKind::LaunchEngine;
    }
    ImGui::EndDisabled();
  }
  if (operation_running && operation_cancellable) {
    ImGui::BeginDisabled(state.cancellation_requested);
    constexpr float kCancellationButtonWidth = 210.0f;
    const float cancellation_offset = std::max(
        0.0f,
        (ImGui::GetContentRegionAvail().x - kCancellationButtonWidth) * 0.5f);
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + cancellation_offset);
    if (ImGui::Button(
            state.cancellation_requested ? "Cancelling safely..."
                                         : "Cancel installation",
            ImVec2(kCancellationButtonWidth, 38.0f))) {
      action.kind = InstallerViewActionKind::CancelOperation;
    }
    ImGui::EndDisabled();
  } else if (has_changes) {
    ImGui::BeginDisabled(!can_apply);
    const char *review_label = plan.update_base
                                   ? ICON_FA_DOWNLOAD " Review & update"
                                   : ICON_FA_DOWNLOAD " Review & install";
    if (ImGui::Button(review_label, ImVec2(-1.0f, 38.0f))) {
      state.pending_plan = plan;
      state.review_requested = true;
    }
    ImGui::EndDisabled();
  }
  if (maintenance) {
    const auto removal_action = RenderInstallerRemovalControl(
        state.removal, product_removal, operation_running);
    if (removal_action == InstallerRemovalViewAction::RemoveProduct) {
      action.kind = InstallerViewActionKind::RemoveProduct;
    } else if (removal_action ==
               InstallerRemovalViewAction::OpenInstalledManager) {
      action.kind = InstallerViewActionKind::OpenInstalledManager;
    }
  }
  if (ImGui::Button("Close", ImVec2(-1.0f, 32.0f))) {
    action.kind = InstallerViewActionKind::Close;
  }
}

void RenderCloseConfirmation(InstallerViewState &state,
                             bool operation_cancellable,
                             InstallerViewAction &action) {
  if (state.close_confirmation_requested) {
    ImGui::OpenPopup("Installation in progress");
    state.close_confirmation_requested = false;
  }
  ImGui::SetNextWindowSize(ImVec2(520.0f, 0.0f), ImGuiCond_Appearing);
  if (!ImGui::BeginPopupModal("Installation in progress", nullptr,
                              ImGuiWindowFlags_AlwaysAutoResize)) {
    return;
  }
  ImGui::TextWrapped(
      "CyxWiz is still applying installation changes. Closing now must "
      "first cancel the operation and wait for safe cleanup.");
  ImGui::Spacing();
  if (operation_cancellable) {
    if (ImGui::Button("Cancel installation and close",
                      ImVec2(250.0f, 36.0f))) {
      action.kind = InstallerViewActionKind::CancelAndClose;
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
  }
  if (ImGui::Button("Continue installation", ImVec2(190.0f, 36.0f))) {
    ImGui::CloseCurrentPopup();
  }
  ImGui::EndPopup();
}

void RenderReviewPopup(InstallerViewState &state,
                       const InstallerCatalogState &catalog,
                       const CyxWizInstallLocation &install_location,
                       bool operation_running, InstallerViewAction &action) {
  ImGui::SetNextWindowSize(ImVec2(570.0f, 0.0f), ImGuiCond_Appearing);
  if (!ImGui::BeginPopupModal("Review installation", nullptr,
                              ImGuiWindowFlags_AlwaysAutoResize)) {
    return;
  }
  ImGui::TextColored(kAccent, "%s Ready to apply", ICON_FA_SHIELD_HALVED);
  ImGui::TextWrapped("CyxWiz will verify every signature and locally qualify "
                     "each compute route before activation.");
  ImGui::Spacing();
  if (state.pending_plan.install_base) {
    const auto root = install_location.install_root.u8string();
    ImGui::Text("Install required CPU Engine:");
    ImGui::BulletText("%s", state.pending_plan.base_pack_id.c_str());
    ImGui::TextWrapped("Location: %s",
                       reinterpret_cast<const char *>(root.c_str()));
  }
  if (state.pending_plan.update_base) {
    ImGui::Text("Update CyxWiz Engine/CPU base:");
    ImGui::BulletText("%s", state.pending_plan.base_pack_id.c_str());
    ImGui::TextDisabled(
        "The previous complete runtime remains available for rollback.");
  }
  if (!state.pending_plan.pack_ids.empty()) {
    ImGui::Text("Acquire and locally qualify:");
    bool installs_cuda = false;
    for (const auto &pack_id : state.pending_plan.pack_ids) {
      ImGui::BulletText("%s", pack_id.c_str());
      const auto record = std::find_if(
          catalog.records.begin(), catalog.records.end(),
          [&](const BackendPackManagerRecord &candidate) {
            return candidate.pack_id == pack_id;
          });
      installs_cuda = installs_cuda ||
                      (record != catalog.records.end() &&
                       record->backend == "cuda");
    }
    if (installs_cuda && catalog.cuda_prerequisite.device_available) {
      ImGui::TextWrapped(
          "NVIDIA driver prerequisite: already satisfied. CyxWiz will "
          "install only its signed app-local CUDA backend pack.");
    }
  }
  if (!state.pending_plan.deactivate_backends.empty()) {
    ImGui::Text("Deactivate optional routes while keeping their files:");
    for (const auto &backend : state.pending_plan.deactivate_backends) {
      ImGui::BulletText("%s", backend.c_str());
    }
  }
  const auto size =
      FormatBackendPackByteSize(state.pending_plan.download_size_bytes);
  ImGui::Text("Package data: %s",
              state.pending_plan.download_size_bytes == 0 ? "None"
                                                          : size.c_str());
  ImGui::TextDisabled(
      "A package that fails verification or qualification remains inactive.");
  ImGui::Checkbox("Launch CyxWiz when installation completes",
                  &state.launch_after_install);
  ImGui::BeginDisabled(operation_running);
  if (ImGui::Button("Apply changes", ImVec2(170.0f, 36.0f))) {
    action.kind = InstallerViewActionKind::ApplyPlan;
    action.plan = state.pending_plan;
    action.launch_after_install = state.launch_after_install;
    ImGui::CloseCurrentPopup();
  }
  ImGui::SameLine();
  if (ImGui::Button("Cancel", ImVec2(110.0f, 36.0f))) {
    ImGui::CloseCurrentPopup();
  }
  ImGui::EndDisabled();
  ImGui::EndPopup();
}

} // namespace

InstallerViewAction RenderInstallerView(
    InstallerViewState &state, const InstallerCatalogState &catalog,
    const CyxWizInstallLocation &install_location,
    const InstallerProductRemovalState &product_removal,
    const std::string &platform_name, bool operation_running,
    bool operation_cancellable,
    const std::string &operation_message,
    const InstallerPlanExecutionProgress &operation_progress,
    const InstallerVisualAssets &assets) {
  InstallerViewAction action;
  const ImGuiViewport *viewport = ImGui::GetMainViewport();
  ImGui::SetNextWindowPos(viewport->WorkPos);
  ImGui::SetNextWindowSize(viewport->WorkSize);
  ImGui::Begin("CyxWiz Installer", nullptr,
               ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                   ImGuiWindowFlags_NoSavedSettings);

  RenderHeader(catalog, platform_name, operation_running, assets, action);
  if (!catalog.message.empty()) {
    ImGui::TextDisabled("%s", catalog.message.c_str());
  }
  ImGui::Dummy(ImVec2(0.0f, 6.0f));

  constexpr float kSummaryWidth = 310.0f;
  constexpr float kColumnGap = 24.0f;
  const float content_width =
      std::max(320.0f, ImGui::GetContentRegionAvail().x - kSummaryWidth -
                           kColumnGap);
  ImGui::BeginChild("main-content", ImVec2(content_width, 0.0f));
  if (ImGui::BeginTabBar("installer-pages")) {
    if (ImGui::BeginTabItem("Workloads")) {
      state.page = 0;
      RenderWorkloads(state, catalog, operation_running);
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Individual components")) {
      state.page = 1;
      RenderComponents(state, catalog, operation_running);
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Installation location")) {
      state.page = 2;
      RenderLocation(state, catalog, install_location, operation_running,
                     action);
      ImGui::EndTabItem();
    }
    if (catalog.verification.evidence_available &&
        ImGui::BeginTabItem("Verification")) {
      state.page = 3;
      RenderVerificationSummary(catalog.verification);
      ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
  }
  ImGui::EndChild();
  ImGui::SameLine(0.0f, kColumnGap);
  ImGui::BeginChild("summary", ImVec2(0.0f, 0.0f), ImGuiChildFlags_None);
  RenderSummary(state, catalog, install_location, product_removal,
                operation_running, operation_cancellable,
                operation_message, operation_progress, action);
  ImGui::EndChild();

  if (state.review_requested) {
    ImGui::OpenPopup("Review installation");
    state.review_requested = false;
  }
  RenderReviewPopup(state, catalog, install_location, operation_running,
                    action);
  RenderPackRemovalConfirmation(state, operation_running, action);
  RenderCloseConfirmation(state, operation_cancellable, action);
  ImGui::End();
  return action;
}

} // namespace cyxwiz::installer::gui
