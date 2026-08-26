#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace gui {

enum class ConsoleSessionKind : std::uint8_t {
  PythonRepl,
  AgentLlm,
  Commands,
  Logs,
  PowerShell,
  GitBash,
  CommandPrompt,
  SystemShell
};

struct ConsoleSessionProfile {
  ConsoleSessionKind kind = ConsoleSessionKind::Logs;
  std::string_view title;
  bool singleton = true;
  bool requires_project = false;
};

struct ConsoleSessionEntry {
  std::uint64_t id = 0;
  ConsoleSessionKind kind = ConsoleSessionKind::Logs;
  std::string title;
  std::string project_root;
  bool unread = false;
};

enum class ConsoleSessionCreateError : std::uint8_t {
  None,
  ProjectRequired,
  ProjectMismatch
};

struct ConsoleSessionCreateResult {
  std::optional<std::uint64_t> session_id;
  ConsoleSessionCreateError error = ConsoleSessionCreateError::None;
  bool created = false;

  explicit operator bool() const {
    return session_id.has_value() && error == ConsoleSessionCreateError::None;
  }
};

class ConsoleSessionModel {
public:
  ConsoleSessionModel();

  static const std::array<ConsoleSessionProfile, 8> &Profiles();
  static const ConsoleSessionProfile &Profile(ConsoleSessionKind kind);

  ConsoleSessionCreateResult CreateOrActivate(ConsoleSessionKind kind,
                                              std::string project_root = {});
  ConsoleSessionCreateResult Ensure(ConsoleSessionKind kind,
                                    std::string project_root = {});
  bool Activate(std::uint64_t session_id);
  bool Close(std::uint64_t session_id);
  std::size_t CloseProjectScopedSessions(std::string_view project_root = {});
  bool SetUnread(std::uint64_t session_id, bool unread = true);

  const std::vector<ConsoleSessionEntry> &Sessions() const { return sessions_; }
  std::optional<std::uint64_t> ActiveSessionId() const {
    return active_session_id_;
  }
  const ConsoleSessionEntry *ActiveSession() const;
  const ConsoleSessionEntry *Find(std::uint64_t session_id) const;

private:
  ConsoleSessionCreateResult Create(ConsoleSessionKind kind,
                                    std::string project_root, bool activate);
  ConsoleSessionEntry *FindMutable(std::uint64_t session_id);
  std::string BuildTitle(ConsoleSessionKind kind);
  void SelectFallbackActiveSession();

  std::vector<ConsoleSessionEntry> sessions_;
  std::optional<std::uint64_t> active_session_id_;
  std::uint64_t next_session_id_ = 1;
  std::uint32_t next_command_prompt_number_ = 1;
  std::uint32_t next_power_shell_number_ = 1;
  std::uint32_t next_git_bash_number_ = 1;
  std::uint32_t next_system_shell_number_ = 1;
};

} // namespace gui
