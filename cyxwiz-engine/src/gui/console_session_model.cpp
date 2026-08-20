#include "console_session_model.h"

#include <algorithm>
#include <stdexcept>

namespace gui {

namespace {

constexpr std::array<ConsoleSessionProfile, 8> kProfiles{{
    {ConsoleSessionKind::PythonRepl, "Python REPL", true, true},
    {ConsoleSessionKind::AgentLlm, "Agent LLM", true, true},
    {ConsoleSessionKind::Commands, "Commands", true, false},
    {ConsoleSessionKind::Logs, "Logs", true, false},
    {ConsoleSessionKind::PowerShell, "PowerShell", false, true},
    {ConsoleSessionKind::GitBash, "Git Bash", false, true},
    {ConsoleSessionKind::CommandPrompt, "Command Prompt", false, true},
    {ConsoleSessionKind::SystemShell, "Shell", false, true},
}};

} // namespace

ConsoleSessionModel::ConsoleSessionModel() {
  (void)CreateOrActivate(ConsoleSessionKind::Logs);
}

const std::array<ConsoleSessionProfile, 8> &ConsoleSessionModel::Profiles() {
  return kProfiles;
}

const ConsoleSessionProfile &
ConsoleSessionModel::Profile(ConsoleSessionKind kind) {
  const auto found = std::find_if(
      kProfiles.begin(), kProfiles.end(),
      [kind](const auto &profile) { return profile.kind == kind; });
  if (found == kProfiles.end()) {
    throw std::invalid_argument("Unknown Console session kind");
  }
  return *found;
}

ConsoleSessionCreateResult
ConsoleSessionModel::CreateOrActivate(ConsoleSessionKind kind,
                                      std::string project_root) {
  return Create(kind, std::move(project_root), true);
}

ConsoleSessionCreateResult
ConsoleSessionModel::Ensure(ConsoleSessionKind kind, std::string project_root) {
  return Create(kind, std::move(project_root), false);
}

ConsoleSessionCreateResult ConsoleSessionModel::Create(ConsoleSessionKind kind,
                                                       std::string project_root,
                                                       bool activate) {
  const auto &profile = Profile(kind);
  if (profile.requires_project && project_root.empty()) {
    return {std::nullopt, ConsoleSessionCreateError::ProjectRequired, false};
  }

  if (profile.singleton) {
    const auto existing = std::find_if(
        sessions_.begin(), sessions_.end(),
        [kind](const auto &session) { return session.kind == kind; });
    if (existing != sessions_.end()) {
      if (profile.requires_project && existing->project_root != project_root) {
        return {std::nullopt, ConsoleSessionCreateError::ProjectMismatch,
                false};
      }
      if (activate)
        Activate(existing->id);
      return {existing->id, ConsoleSessionCreateError::None, false};
    }
  }

  ConsoleSessionEntry entry;
  entry.id = next_session_id_++;
  entry.kind = kind;
  entry.title = BuildTitle(kind);
  entry.project_root = std::move(project_root);
  sessions_.push_back(std::move(entry));
  if (activate)
    Activate(sessions_.back().id);
  return {sessions_.back().id, ConsoleSessionCreateError::None, true};
}

bool ConsoleSessionModel::Activate(std::uint64_t session_id) {
  auto *session = FindMutable(session_id);
  if (!session)
    return false;
  active_session_id_ = session_id;
  session->unread = false;
  return true;
}

bool ConsoleSessionModel::Close(std::uint64_t session_id) {
  const auto found = std::find_if(
      sessions_.begin(), sessions_.end(),
      [session_id](const auto &session) { return session.id == session_id; });
  if (found == sessions_.end())
    return false;

  const bool closed_active = active_session_id_ == session_id;
  sessions_.erase(found);
  if (closed_active)
    SelectFallbackActiveSession();
  return true;
}

std::size_t
ConsoleSessionModel::CloseProjectScopedSessions(std::string_view project_root) {
  const auto previous_size = sessions_.size();
  sessions_.erase(std::remove_if(sessions_.begin(), sessions_.end(),
                                 [project_root](const auto &session) {
                                   const auto &profile = Profile(session.kind);
                                   return profile.requires_project &&
                                          (project_root.empty() ||
                                           session.project_root ==
                                               project_root);
                                 }),
                  sessions_.end());

  if (active_session_id_ && !Find(*active_session_id_)) {
    SelectFallbackActiveSession();
  }
  return previous_size - sessions_.size();
}

bool ConsoleSessionModel::SetUnread(std::uint64_t session_id, bool unread) {
  auto *session = FindMutable(session_id);
  if (!session)
    return false;
  session->unread = unread && active_session_id_ != session_id;
  return true;
}

const ConsoleSessionEntry *ConsoleSessionModel::ActiveSession() const {
  return active_session_id_ ? Find(*active_session_id_) : nullptr;
}

const ConsoleSessionEntry *
ConsoleSessionModel::Find(std::uint64_t session_id) const {
  const auto found = std::find_if(
      sessions_.begin(), sessions_.end(),
      [session_id](const auto &session) { return session.id == session_id; });
  return found == sessions_.end() ? nullptr : &*found;
}

ConsoleSessionEntry *
ConsoleSessionModel::FindMutable(std::uint64_t session_id) {
  const auto found = std::find_if(
      sessions_.begin(), sessions_.end(),
      [session_id](const auto &session) { return session.id == session_id; });
  return found == sessions_.end() ? nullptr : &*found;
}

std::string ConsoleSessionModel::BuildTitle(ConsoleSessionKind kind) {
  if (kind == ConsoleSessionKind::CommandPrompt) {
    return "Command Prompt " + std::to_string(next_command_prompt_number_++);
  }
  if (kind == ConsoleSessionKind::PowerShell) {
    return "PowerShell " + std::to_string(next_power_shell_number_++);
  }
  if (kind == ConsoleSessionKind::GitBash) {
    return "Git Bash " + std::to_string(next_git_bash_number_++);
  }
  if (kind == ConsoleSessionKind::SystemShell) {
    return "Shell " + std::to_string(next_system_shell_number_++);
  }
  return std::string(Profile(kind).title);
}

void ConsoleSessionModel::SelectFallbackActiveSession() {
  if (sessions_.empty()) {
    active_session_id_.reset();
    return;
  }
  active_session_id_ = sessions_.front().id;
  sessions_.front().unread = false;
}

} // namespace gui
