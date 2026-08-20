#include "console_workbench.h"
#include "icons.h"

#include <imgui.h>

#include <optional>
#include <stdexcept>

namespace gui {

namespace {

const char *UnavailableReason(ConsoleSessionKind kind) {
  switch (kind) {
  case ConsoleSessionKind::PythonRepl:
    return "Python REPL migration is scheduled for the next Console slice.";
  case ConsoleSessionKind::AgentLlm:
    return "";
  case ConsoleSessionKind::CommandPrompt:
  case ConsoleSessionKind::PowerShell:
  case ConsoleSessionKind::GitBash:
#ifdef _WIN32
    return "";
#else
    return "PowerShell, Git Bash, and Command Prompt sessions are available on "
           "Windows only.";
#endif
  case ConsoleSessionKind::SystemShell:
#ifdef _WIN32
    return "The system shell profile is provided by the native PowerShell, "
           "Git Bash, and Command Prompt profiles on Windows.";
#else
    return "";
#endif
  case ConsoleSessionKind::Commands:
  case ConsoleSessionKind::Logs:
    return "";
  }
  return "This Console session is unavailable.";
}

} // namespace

ConsoleSessionKind ConsoleWorkbench::ActiveKind() const {
  const auto *active = sessions_.ActiveSession();
  if (!active) {
    throw std::logic_error("Console has no active session");
  }
  return active->kind;
}

bool ConsoleWorkbench::HasActiveSession() const {
  return sessions_.ActiveSession() != nullptr;
}

std::optional<std::uint64_t> ConsoleWorkbench::ActiveSessionId() const {
  return sessions_.ActiveSessionId();
}

std::string_view ConsoleWorkbench::ActiveProjectRoot() const {
  const auto *active = sessions_.ActiveSession();
  return active ? std::string_view(active->project_root) : std::string_view{};
}

const std::vector<ConsoleSessionEntry> &ConsoleWorkbench::Sessions() const {
  return sessions_.Sessions();
}

ConsoleSessionCreateResult
ConsoleWorkbench::ActivateSession(ConsoleSessionKind kind) {
  const auto &profile = ConsoleSessionModel::Profile(kind);
  auto result = sessions_.CreateOrActivate(
      kind, profile.requires_project ? project_root_ : std::string{});
  navigation_selection_pending_ = static_cast<bool>(result);
  if (result)
    RequestFocus();
  return result;
}

ConsoleSessionCreateResult
ConsoleWorkbench::EnsureSession(ConsoleSessionKind kind) {
  const auto &profile = ConsoleSessionModel::Profile(kind);
  return sessions_.Ensure(kind, profile.requires_project ? project_root_
                                                         : std::string{});
}

bool ConsoleWorkbench::MarkUnread(std::uint64_t session_id) {
  return sessions_.SetUnread(session_id);
}

bool ConsoleWorkbench::ConsumeFocusRequest() {
  if (focus_request_pending_ && focus_request_delay_frames_ > 0) {
    --focus_request_delay_frames_;
    return false;
  }
  const bool requested = focus_request_pending_;
  focus_request_pending_ = false;
  return requested;
}

void ConsoleWorkbench::RequestFocus() {
  focus_request_pending_ = true;
  // Popup and tab focus restoration runs at the end of the activation frame.
  // Apply content focus on the next frame so it remains authoritative.
  focus_request_delay_frames_ = 1;
}

void ConsoleWorkbench::SetProjectRoot(std::string project_root) {
  const auto previous_active = sessions_.ActiveSessionId();
  if (!project_root_.empty() && project_root_ != project_root) {
    sessions_.CloseProjectScopedSessions(project_root_);
  }
  project_root_ = std::move(project_root);
  navigation_selection_pending_ = true;
  if (previous_active != sessions_.ActiveSessionId()) {
    RequestFocus();
  }
}

void ConsoleWorkbench::CloseProject(std::string_view project_root) {
  const auto previous_active = sessions_.ActiveSessionId();
  sessions_.CloseProjectScopedSessions(project_root);
  if (project_root.empty() || project_root_ == project_root) {
    project_root_.clear();
  }
  navigation_selection_pending_ = true;
  if (previous_active != sessions_.ActiveSessionId())
    RequestFocus();
}

void ConsoleWorkbench::RenderCommandBar() {
  const ImGuiStyle &style = ImGui::GetStyle();
  const float button_size = ImGui::GetFrameHeight();
  const float controls_width = button_size;
  if (ImGui::BeginTable("ConsoleCommandBar", 2,
                        ImGuiTableFlags_SizingStretchProp)) {
    ImGui::TableSetupColumn("Session navigation",
                            ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Controls", ImGuiTableColumnFlags_WidthFixed,
                            controls_width);
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    RenderSessionNavigation();

    ImGui::TableSetColumnIndex(1);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 5.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);
    ImGui::PushStyleColor(ImGuiCol_Button, style.Colors[ImGuiCol_FrameBg]);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                          style.Colors[ImGuiCol_HeaderHovered]);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                          style.Colors[ImGuiCol_HeaderActive]);
    ImGui::PushStyleColor(ImGuiCol_Border, style.Colors[ImGuiCol_Separator]);
    if (ImGui::Button(ICON_FA_PLUS "##ConsoleAddSession",
                      ImVec2(button_size, button_size))) {
      ImGui::OpenPopup("ConsoleAddSessionPopup");
    }
    ImGui::PopStyleColor(4);
    ImGui::PopStyleVar(2);
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("New Console session");
    }

    RenderAddSessionPopup();
    ImGui::EndTable();
  }
}

void ConsoleWorkbench::RenderSessionNavigation() {
  if (sessions_.Sessions().empty()) {
    ImGui::TextDisabled("No active session");
    return;
  }

  if (!ImGui::BeginTabBar("ConsoleSessionNavigation",
                          ImGuiTabBarFlags_FittingPolicyScroll)) {
    return;
  }
  const bool applying_pending_selection = navigation_selection_pending_;
  const auto authoritative_session_id = sessions_.ActiveSessionId();
  std::optional<std::uint64_t> rendered_session_to_activate;
  std::optional<std::uint64_t> session_to_close;
  for (const auto &session : sessions_.Sessions()) {
    const bool active = authoritative_session_id == session.id;
    const ImGuiTabItemFlags flags = applying_pending_selection && active
                                        ? ImGuiTabItemFlags_SetSelected
                                        : ImGuiTabItemFlags_None;
    const std::string label =
        session.title + "##ConsoleSession" + std::to_string(session.id);
    bool keep_open = true;
    if (ImGui::BeginTabItem(label.c_str(), &keep_open, flags)) {
      // BeginTabItem can still expose the previously selected tab before
      // ImGui reaches a later SetSelected item. During programmatic
      // navigation the model is authoritative for this synchronization frame.
      if (!applying_pending_selection && !active)
        rendered_session_to_activate = session.id;
      ImGui::EndTabItem();
    }
    if (!keep_open)
      session_to_close = session.id;
  }
  navigation_selection_pending_ = false;
  ImGui::EndTabBar();

  if (rendered_session_to_activate &&
      sessions_.Activate(*rendered_session_to_activate)) {
    RequestFocus();
  }

  if (session_to_close) {
    const auto previous_active = sessions_.ActiveSessionId();
    sessions_.Close(*session_to_close);
    navigation_selection_pending_ = true;
    if (previous_active != sessions_.ActiveSessionId())
      RequestFocus();
  }
}

void ConsoleWorkbench::RenderAddSessionPopup() {
  if (!ImGui::BeginPopup("ConsoleAddSessionPopup"))
    return;

  ImGui::TextDisabled("New Console Session");
  ImGui::Separator();
  const auto &profiles = ConsoleSessionModel::Profiles();
  bool shell_profiles_started = false;
  for (const auto &profile : profiles) {
    if (!IsVisible(profile.kind))
      continue;
    const bool shell_profile =
        profile.kind == ConsoleSessionKind::PowerShell ||
        profile.kind == ConsoleSessionKind::GitBash ||
        profile.kind == ConsoleSessionKind::CommandPrompt ||
        profile.kind == ConsoleSessionKind::SystemShell;
    if (shell_profile && !shell_profiles_started) {
      ImGui::Separator();
      shell_profiles_started = true;
    }

    const bool implemented = IsImplemented(profile.kind);
    const bool project_available =
        !profile.requires_project || !project_root_.empty();
    ImGui::BeginDisabled(!implemented || !project_available);
    if (ImGui::MenuItem(profile.title.data())) {
      (void)ActivateSession(profile.kind);
    }
    ImGui::EndDisabled();
    if ((!implemented || !project_available) &&
        ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
      ImGui::SetTooltip("%s", project_available
                                  ? UnavailableReason(profile.kind)
                                  : "Open a project to create this session.");
    }
  }
  ImGui::EndPopup();
}

} // namespace gui
