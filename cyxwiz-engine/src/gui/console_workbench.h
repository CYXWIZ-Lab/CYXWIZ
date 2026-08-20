#pragma once

#include "console_session_model.h"

#include <cstdint>

namespace gui {

class ConsoleWorkbench {
public:
  static constexpr bool IsImplemented(ConsoleSessionKind kind) {
#ifdef _WIN32
    return kind == ConsoleSessionKind::PythonRepl ||
           kind == ConsoleSessionKind::AgentLlm ||
           kind == ConsoleSessionKind::Logs ||
           kind == ConsoleSessionKind::Commands ||
           kind == ConsoleSessionKind::CommandPrompt ||
           kind == ConsoleSessionKind::PowerShell ||
           kind == ConsoleSessionKind::GitBash;
#else
    return kind == ConsoleSessionKind::PythonRepl ||
           kind == ConsoleSessionKind::AgentLlm ||
           kind == ConsoleSessionKind::Logs ||
           kind == ConsoleSessionKind::Commands ||
           kind == ConsoleSessionKind::SystemShell;
#endif
  }

  static constexpr bool IsVisible(ConsoleSessionKind kind) {
#ifdef _WIN32
    return kind != ConsoleSessionKind::SystemShell;
#else
    return kind != ConsoleSessionKind::CommandPrompt &&
           kind != ConsoleSessionKind::PowerShell &&
           kind != ConsoleSessionKind::GitBash;
#endif
  }

  ConsoleSessionKind ActiveKind() const;
  bool HasActiveSession() const;
  std::optional<std::uint64_t> ActiveSessionId() const;
  std::string_view ActiveProjectRoot() const;
  const std::vector<ConsoleSessionEntry> &Sessions() const;
  ConsoleSessionCreateResult ActivateSession(ConsoleSessionKind kind);
  ConsoleSessionCreateResult EnsureSession(ConsoleSessionKind kind);
  bool MarkUnread(std::uint64_t session_id);
  bool ConsumeFocusRequest();
  void SetProjectRoot(std::string project_root);
  void CloseProject(std::string_view project_root);

  void RenderCommandBar();

private:
  void RequestFocus();
  void RenderSessionNavigation();
  void RenderAddSessionPopup();

  ConsoleSessionModel sessions_;
  std::string project_root_;
  bool navigation_selection_pending_ = true;
  bool focus_request_pending_ = true;
  std::uint8_t focus_request_delay_frames_ = 0;
};

} // namespace gui
