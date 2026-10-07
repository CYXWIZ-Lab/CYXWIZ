#include "console.h"
#include "../core/async_task_manager.h"
#include "../core/engine_config.h"
#include "../core/file_dialogs.h"
#include "../core/project_manager.h"
#include "../core/runtime_console_commands.h"
#include "../core/runtime_log_store.h"
#include "panels/agent_llm_session.h"
#include "panels/python_repl_session.h"
#include "panels/local_shell_session.h"
#include "ui_buttons.h"
#include "separate_windows.h"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <imgui.h>
#include <iomanip>
#include <mutex>
#include <spdlog/spdlog.h>
#include <sstream>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#else
#include <array>
#include <cstdio>
#include <memory>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {
std::mutex g_console_mutex;

std::string
FormatLogTimestamp(const std::chrono::system_clock::time_point &timestamp) {
  const auto time = std::chrono::system_clock::to_time_t(timestamp);
  std::tm utc{};
#ifdef _WIN32
  gmtime_s(&utc, &time);
#else
  gmtime_r(&time, &utc);
#endif
  const auto milliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          timestamp.time_since_epoch()) %
      1000;

  std::ostringstream output;
  // Keep the full UTC date in every console row. A time-only prefix made rows
  // appear to disagree with ISO timestamps embedded in runtime event messages.
  output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%S") << '.' << std::setfill('0')
         << std::setw(3) << milliseconds.count() << 'Z';
  return output.str();
}

void ShowHelpTooltip(const char *text,
                     ImGuiHoveredFlags flags = ImGuiHoveredFlags_DelayNormal) {
  if (ImGui::IsItemHovered(flags)) {
    ImGui::SetTooltip("%s", text);
  }
}

std::string FormatRuntimeLogRow(const cyxwiz::RuntimeLogEvent &event) {
  std::ostringstream output;
  output << '#' << event.sequence << ' '
         << FormatLogTimestamp(event.timestamp_utc)
         << " level=" << cyxwiz::RuntimeLogLevelName(event.level)
         << " category=" << event.category;
  if (!event.source.empty())
    output << " source=" << event.source;
  if (!event.primary_error_code.empty()) {
    output << " code=" << event.primary_error_code;
  }
  if (!event.run_id.empty())
    output << " run=" << event.run_id;
  if (event.task_id != 0)
    output << " task=" << event.task_id;
  if (!event.backend.empty())
    output << " backend=" << event.backend;
  if (event.device_id >= 0)
    output << " device_id=" << event.device_id;
  output << " | " << event.message;
  return output.str();
}

std::string FormatRuntimeLogDetails(const cyxwiz::RuntimeLogEvent &event) {
  std::ostringstream output;
  output << FormatRuntimeLogRow(event) << '\n';
  if (!event.event_name.empty()) {
    output << "event=" << event.event_name << '\n';
  }
  if (event.node_id >= 0)
    output << "node_id=" << event.node_id << '\n';
  if (!event.device_name.empty()) {
    output << "device_name=" << event.device_name << '\n';
  }
  if (!event.dataset_name.empty()) {
    output << "dataset=" << event.dataset_name << '\n';
  }
  if (!event.diagnostic_phase.empty()) {
    output << "diagnostic_phase=" << event.diagnostic_phase << '\n';
  }
  if (!event.component.empty()) {
    output << "component=" << event.component << '\n';
  }
  for (const auto &issue_code : event.issue_codes) {
    output << "issue_code=" << issue_code << '\n';
  }
  for (const auto &[key, value] : event.details) {
    output << key << '=' << value << '\n';
  }
  return output.str();
}

#ifdef _WIN32
std::string QuoteWindowsArgument(const std::string &argument) {
  if (!argument.empty() &&
      argument.find_first_of(" \t\n\v\"") == std::string::npos) {
    return argument;
  }

  std::string quoted = "\"";
  size_t backslashes = 0;
  for (const char current : argument) {
    if (current == '\\') {
      ++backslashes;
      continue;
    }
    if (current == '"') {
      quoted.append(backslashes * 2 + 1, '\\');
      quoted.push_back('"');
    } else {
      quoted.append(backslashes, '\\');
      quoted.push_back(current);
    }
    backslashes = 0;
  }
  quoted.append(backslashes * 2, '\\');
  quoted.push_back('"');
  return quoted;
}
#endif
} // namespace

namespace gui {

struct Console::RuntimeLogExportTaskState {
  std::mutex mutex;
  bool running = false;
  bool success = false;
  std::string message;
  std::filesystem::path destination;  // written file, for Show in folder
};

Console::Console()
    : agent_llm_(std::make_unique<cyxwiz::AgentLlmSession>()),
      python_repl_(std::make_unique<cyxwiz::PythonReplSession>()),
      scroll_to_bottom_(false), show_window_(true), auto_scroll_(true),
      inspector_export_task_state_(
          std::make_shared<RuntimeLogExportTaskState>()),
      truth_provider_(cyxwiz::CreateEngineRuntimeTruthProvider()),
      command_service_(std::make_unique<cyxwiz::RuntimeConsoleCommandService>(
          cyxwiz::RuntimeLogStore::Instance(), truth_provider_.get())),
      show_copy_notification_(false), copy_notification_time_(0.0f) {
  memset(input_buf_, 0, sizeof(input_buf_));
  {
    // main.cpp writes engine_log.txt into the working directory it sets to
    // the Engine folder before the UI starts.
    std::error_code ec;
    log_file_path_ = std::filesystem::absolute("engine_log.txt", ec);
  }
  LoadSavedInspectorFilters();
  StartInspectorWorker();
}

Console::~Console() {
  // A running pip command is asked to stop (it terminates its process) and
  // any output still queued for this console is dropped.
  cyxwiz::AsyncTaskManager::Instance().CancelOwnedBy(task_owner_token_);
  local_shell_sessions_.clear();
  agent_llm_.reset();
  python_repl_.reset();
  StopInspectorWorker();
}

void Console::SetScriptingEngine(
    std::shared_ptr<scripting::ScriptingEngine> scripting_engine) {
  python_repl_->SetScriptingEngine(std::move(scripting_engine));
}

void Console::SetAssistantCommandHandler(
    std::function<cyxwiz::plugin::AssistantCommandResponse(
        const cyxwiz::plugin::AssistantCommandRequest &)>
        handler) {
  agent_llm_->SetCommandHandler(std::move(handler));
}

void Console::SetProjectRoot(std::string project_root) {
  local_shell_sessions_.clear();
  agent_llm_->ResetProjectState();
  python_repl_->ResetProjectState();
  workbench_.SetProjectRoot(std::move(project_root));
}

void Console::CloseProject(std::string_view project_root) {
  local_shell_sessions_.clear();
  agent_llm_->ResetProjectState();
  python_repl_->ResetProjectState();
  workbench_.CloseProject(project_root);
}

bool Console::ActivatePythonRepl() {
  show_window_ = true;
  return static_cast<bool>(
      workbench_.ActivateSession(ConsoleSessionKind::PythonRepl));
}

void Console::EndScriptOutput(const std::string &source, bool success,
                              bool cancelled, double seconds) {
  python_repl_->EndScriptOutput(source, success, cancelled, seconds);
}

void Console::AppendScriptOutput(const std::string &source,
                                 const std::string &text, bool is_error) {
  python_repl_->AppendScriptOutput(source, text, is_error);
  const auto session = workbench_.EnsureSession(ConsoleSessionKind::PythonRepl);
  if (session) {
    workbench_.MarkUnread(*session.session_id);
  }
}

void Console::Render() {
  UpdateLogsProblemBadge();
  if (!show_window_)
    return;

  ::gui::NextWindowMayLeave("Console");
  const bool expanded = ImGui::Begin("Console", &show_window_);
  ::gui::TabMenu("Console", &show_window_);
  if (expanded) {
    workbench_.RenderCommandBar();
    PruneLocalShellSessions();
    ImGui::Separator();
    const bool request_focus = workbench_.ConsumeFocusRequest();
    if (request_focus)
      ImGui::SetWindowFocus();
    RenderActiveSession(request_focus);
    scroll_to_bottom_.store(false, std::memory_order_relaxed);
  }
  ImGui::End();
}

void Console::RenderActiveSession(bool request_focus) {
  if (!workbench_.HasActiveSession()) {
    ImGui::TextDisabled("No active session. Use + to add Logs or Commands.");
    return;
  }

  switch (workbench_.ActiveKind()) {
  case ConsoleSessionKind::Logs:
    RenderLogsSession(request_focus);
    return;
  case ConsoleSessionKind::Commands:
    RenderCommandsSession(request_focus);
    return;
  case ConsoleSessionKind::PythonRepl:
    if (request_focus)
      python_repl_->RequestInputFocus();
    python_repl_->RenderContent();
    return;
  case ConsoleSessionKind::AgentLlm:
    if (request_focus)
      agent_llm_->RequestInputFocus();
    agent_llm_->RenderContent(workbench_.ActiveProjectRoot());
    return;
  case ConsoleSessionKind::CommandPrompt:
  case ConsoleSessionKind::PowerShell:
  case ConsoleSessionKind::GitBash:
  case ConsoleSessionKind::SystemShell:
    RenderLocalShellSession(request_focus);
    return;
  }
}

void Console::RenderLocalShellSession(bool request_focus) {
  const auto session_id = workbench_.ActiveSessionId();
  if (!session_id) {
    ImGui::TextDisabled("No active local shell session.");
    return;
  }

  auto &session = local_shell_sessions_[*session_id];
  if (!session) {
    cyxwiz::LocalShellKind kind = cyxwiz::LocalShellKind::PowerShell;
    if (workbench_.ActiveKind() == ConsoleSessionKind::CommandPrompt) {
      kind = cyxwiz::LocalShellKind::CommandPrompt;
    } else if (workbench_.ActiveKind() == ConsoleSessionKind::GitBash) {
      kind = cyxwiz::LocalShellKind::GitBash;
    } else if (workbench_.ActiveKind() == ConsoleSessionKind::SystemShell) {
      kind = cyxwiz::LocalShellKind::SystemShell;
    }
    session = std::make_unique<cyxwiz::LocalShellSession>(
        kind, std::filesystem::path(workbench_.ActiveProjectRoot()));
  }
  if (request_focus)
    session->RequestInputFocus();
  session->RenderContent();
}

void Console::PruneLocalShellSessions() {
  for (auto iterator = local_shell_sessions_.begin();
       iterator != local_shell_sessions_.end();) {
    const bool session_exists =
        std::any_of(workbench_.Sessions().begin(), workbench_.Sessions().end(),
                    [session_id = iterator->first](const auto &session) {
                      return session.id == session_id;
                    });
    if (session_exists) {
      ++iterator;
    } else {
      iterator = local_shell_sessions_.erase(iterator);
    }
  }
}

bool Console::IsRuntimeLogExportRunning() const {
  std::lock_guard<std::mutex> lock(inspector_export_task_state_->mutex);
  return inspector_export_task_state_->running;
}

void Console::RenderRuntimeLogExportStatus() {
  bool running = false;
  bool success = false;
  std::string message;
  std::filesystem::path destination;
  {
    std::lock_guard<std::mutex> lock(inspector_export_task_state_->mutex);
    running = inspector_export_task_state_->running;
    success = inspector_export_task_state_->success;
    message = inspector_export_task_state_->message;
    destination = inspector_export_task_state_->destination;
  }
  if (running) {
    ImGui::TextDisabled("Exporting frozen runtime-log slice...");
  } else if (!message.empty()) {
    ImGui::PushStyleColor(ImGuiCol_Text, success
                                             ? ImVec4(0.24f, 0.84f, 0.55f, 1.0f)
                                             : ImVec4(1.0f, 0.48f, 0.45f, 1.0f));
    ImGui::PushTextWrapPos(ImGui::GetContentRegionAvail().x * 0.75f +
                           ImGui::GetCursorPosX());
    ImGui::TextUnformatted(message.c_str());
    ImGui::PopTextWrapPos();
    ImGui::PopStyleColor();
    if (success && !destination.empty()) {
      ImGui::SameLine();
      if (cyxwiz::ui::LinkButton("Show in folder##export") &&
          !OpenPathWithSystem(destination, true)) {
        SetLogsNotice("Could not open the export folder");
      }
    }
    ImGui::SameLine();
    if (cyxwiz::ui::LinkButton("Dismiss##export")) {
      std::lock_guard<std::mutex> lock(inspector_export_task_state_->mutex);
      inspector_export_task_state_->message.clear();
      inspector_export_task_state_->destination.clear();
    }
  }
}

void Console::RenderRuntimeLogExportDialog() {
  if (inspector_export_popup_requested_) {
    ImGui::OpenPopup("Export runtime logs");
    inspector_export_popup_requested_ = false;
  }

  ImGui::SetNextWindowSize(ImVec2(760.0f, 680.0f), ImGuiCond_Appearing);
  if (!ImGui::BeginPopupModal("Export runtime logs", nullptr,
                              ImGuiWindowFlags_NoCollapse)) {
    return;
  }

  const auto frozen_result = inspector_export_result_;
  if (!frozen_result) {
    ImGui::TextDisabled("No frozen runtime-log result is available.");
    if (cyxwiz::ui::SecondaryButton("Close"))
      ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
    return;
  }

  const auto &source_events = frozen_result->query.events;
  const auto selected = std::find_if(
      source_events.begin(), source_events.end(), [this](const auto &event) {
        return event.sequence == inspector_export_selected_sequence_;
      });
  const bool selected_available = selected != source_events.end();
  if (!selected_available)
    inspector_export_selected_scope_ = false;

  ImGui::Text(
      "Frozen at sequence %llu | displayed %zu | matched %zu%s",
      static_cast<unsigned long long>(frozen_result->query.high_water_sequence),
      source_events.size(), frozen_result->query.matched_count,
      frozen_result->query.truncated ? " | source truncated" : "");
  const std::string visible_filter = frozen_result->effective_filter.empty()
                                         ? "(none)"
                                         : frozen_result->effective_filter;
  ImGui::TextWrapped("Filter: %s", visible_filter.c_str());

  ImGui::SeparatorText("Scope and format");
  int scope = inspector_export_selected_scope_ ? 1 : 0;
  if (ImGui::RadioButton("Filtered rows", &scope, 0)) {
    inspector_export_selected_scope_ = false;
  }
  ImGui::SameLine();
  ImGui::BeginDisabled(!selected_available);
  if (ImGui::RadioButton("Selected row", &scope, 1)) {
    inspector_export_selected_scope_ = true;
  }
  ShowHelpTooltip(selected_available
                      ? "Export only the row selected when this dialog opened."
                      : "Select a runtime-log row before opening Export.",
                  ImGuiHoveredFlags_DelayNormal |
                      ImGuiHoveredFlags_AllowWhenDisabled);
  ImGui::EndDisabled();

  int format =
      inspector_export_format_ == cyxwiz::RuntimeLogExportFormat::JsonLines ? 0
                                                                            : 1;
  if (ImGui::RadioButton("JSON Lines", &format, 0)) {
    inspector_export_format_ = cyxwiz::RuntimeLogExportFormat::JsonLines;
  }
  ImGui::SameLine();
  if (ImGui::RadioButton("Readable text", &format, 1)) {
    inspector_export_format_ = cyxwiz::RuntimeLogExportFormat::ReadableText;
  }

  ImGui::SeparatorText("Redaction");
  if (cyxwiz::ui::SecondaryButton("Shareable preset")) {
    inspector_export_redaction_ = {};
  }
  ShowHelpTooltip("Enable every supported sensitive-data redaction.");
  ImGui::SameLine();
  if (cyxwiz::ui::SecondaryButton("Raw preset")) {
    inspector_export_redaction_ = {false, false, false, false, false};
  }
  ShowHelpTooltip("Disable all redaction. Review the preview before export.");

  ImGui::Checkbox("Secrets", &inspector_export_redaction_.secrets);
  ImGui::SameLine();
  ImGui::Checkbox("Paths", &inspector_export_redaction_.paths);
  ImGui::SameLine();
  ImGui::Checkbox("Dataset names", &inspector_export_redaction_.dataset_names);
  ImGui::SameLine();
  ImGui::Checkbox("Query text", &inspector_export_redaction_.query_text);
  ImGui::SameLine();
  ImGui::Checkbox("Python/package output",
                  &inspector_export_redaction_.python_output);

  const bool any_redaction = inspector_export_redaction_.secrets ||
                             inspector_export_redaction_.paths ||
                             inspector_export_redaction_.dataset_names ||
                             inspector_export_redaction_.query_text ||
                             inspector_export_redaction_.python_output;
  if (!any_redaction) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.65f, 0.2f, 1.0f));
    ImGui::TextWrapped(
        "Raw export may contain credentials, local paths, dataset names, "
        "queries, or Python/package output.");
    ImGui::PopStyleColor();
  }

  ImGui::SeparatorText("Preview");
  const cyxwiz::RuntimeLogEvent *preview_event = nullptr;
  if (inspector_export_selected_scope_ && selected_available) {
    preview_event = &*selected;
  } else if (!source_events.empty()) {
    preview_event = &source_events.front();
  }
  if (preview_event && ImGui::BeginTable("RuntimeLogExportPreview", 2,
                                         ImGuiTableFlags_BordersInnerV |
                                             ImGuiTableFlags_Resizable)) {
    ImGui::TableSetupColumn("Original", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Exported", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableHeadersRow();
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    const auto original = FormatRuntimeLogDetails(*preview_event);
    ImGui::BeginChild("RuntimeLogOriginalPreview", ImVec2(0, 150.0f));
    ImGui::TextWrapped("%s", original.c_str());
    ImGui::EndChild();
    ImGui::TableSetColumnIndex(1);
    const auto exported = cyxwiz::RuntimeLogExportService::FormatEventText(
        *preview_event, inspector_export_redaction_);
    ImGui::BeginChild("RuntimeLogRedactedPreview", ImVec2(0, 150.0f));
    ImGui::TextWrapped("%s", exported.c_str());
    ImGui::EndChild();
    ImGui::EndTable();
  } else {
    ImGui::TextDisabled("The frozen filtered slice contains no rows.");
  }

  bool export_running = false;
  {
    std::lock_guard<std::mutex> lock(inspector_export_task_state_->mutex);
    export_running = inspector_export_task_state_->running;
  }
  if (cyxwiz::ui::PrimaryButton(
          "Save export...", !export_running && preview_event != nullptr,
          export_running ? "An export is already running."
                         : "The frozen filtered slice contains no rows.")) {
    const bool json_lines =
        inspector_export_format_ == cyxwiz::RuntimeLogExportFormat::JsonLines;
    const auto filters =
        json_lines ? cyxwiz::FileDialogs::FilterList{{"JSON Lines", "jsonl"},
                                                     {"All Files", "*"}}
                   : cyxwiz::FileDialogs::FilterList{{"Text Files", "txt"},
                                                     {"All Files", "*"}};
    const auto &project_manager = cyxwiz::ProjectManager::Instance();
    const std::string default_path = project_manager.HasActiveProject()
                                         ? project_manager.GetExportsPath()
                                         : std::string{};
    const std::string default_name =
        "runtime_logs_" +
        std::to_string(frozen_result->query.high_water_sequence) +
        (json_lines ? ".jsonl" : ".txt");
    const auto destination = cyxwiz::FileDialogs::SaveFile(
        "Export Runtime Logs", filters,
        default_path.empty() ? nullptr : default_path.c_str(),
        default_name.c_str());
    if (destination) {
      QueueRuntimeLogExport(*destination);
      ImGui::CloseCurrentPopup();
    }
  }
  ImGui::SameLine();
  if (cyxwiz::ui::SecondaryButton("Cancel"))
    ImGui::CloseCurrentPopup();
  ImGui::EndPopup();
}

void Console::AddLog(const std::string &message, LogLevel level) {
  std::lock_guard<std::mutex> lock(log_mutex_);
  LogEntry entry;
  entry.message = message;
  entry.level = level;
  entry.timestamp = std::chrono::system_clock::now();
  entry.sequence = ++next_command_sequence_;
  entry.block = current_command_block_;
  items_.push_back(entry);
  scroll_to_bottom_.store(true, std::memory_order_relaxed);

  // Command output is a small view transcript, separate from runtime truth.
  if (items_.size() > 1000) {
    items_.pop_front();
  }
}

void Console::AddInfo(const std::string &message) {
  AddLog(message, LogLevel::Info);
}

void Console::AddWarning(const std::string &message) {
  AddLog(message, LogLevel::Warning);
}

void Console::AddError(const std::string &message) {
  AddLog(message, LogLevel::Error);
}

void Console::AddSuccess(const std::string &message) {
  AddLog(message, LogLevel::Success);
}

void Console::Clear() {
  ClearCommandTranscript();
  ClearLogView();
}

void Console::ClearLogView() {
  const auto newest_sequence =
      cyxwiz::RuntimeLogStore::Instance().GetStats().newest_sequence;
  inspector_after_sequence_ = newest_sequence;
  inspector_selected_sequence_ = 0;
  RequestInspectorQuery(true);
}

void Console::ClearCommandTranscript() {
  {
    std::lock_guard<std::mutex> lock(log_mutex_);
    items_.clear();
  }
  selected_command_sequence_ = 0;
  selected_command_block_ = 0;
  // Keep only running blocks (pip): their later output shows as plain lines.
  for (auto it = command_blocks_.begin(); it != command_blocks_.end();) {
    if (it->second.running)
      ++it;
    else
      it = command_blocks_.erase(it);
  }
}

std::vector<Console::LogEntry> Console::SnapshotEntries() const {
  std::lock_guard<std::mutex> lock(log_mutex_);
  return {items_.begin(), items_.end()};
}

void Console::StartInspectorWorker() {
  inspector_worker_ = std::thread([this]() {
    for (;;) {
      cyxwiz::RuntimeLogInspectorRequest request;
      uint64_t generation = 0;
      {
        std::unique_lock<std::mutex> lock(inspector_mutex_);
        inspector_cv_.wait(lock, [this]() {
          return inspector_stop_ || inspector_request_pending_;
        });
        if (inspector_stop_)
          return;
        request = inspector_pending_request_;
        generation = inspector_request_generation_;
        inspector_request_pending_ = false;
      }

      auto result = std::make_shared<cyxwiz::RuntimeLogInspectorResult>(
          cyxwiz::QueryRuntimeLogInspector(cyxwiz::RuntimeLogStore::Instance(),
                                           request));
      inspector_query_executions_.fetch_add(1, std::memory_order_relaxed);
      {
        std::lock_guard<std::mutex> lock(inspector_mutex_);
        if (generation == inspector_request_generation_) {
          inspector_result_ = std::move(result);
        } else {
          inspector_query_stale_.fetch_add(1, std::memory_order_relaxed);
        }
      }
    }
  });
  RequestInspectorQuery(true);
}

void Console::StopInspectorWorker() {
  {
    std::lock_guard<std::mutex> lock(inspector_mutex_);
    inspector_stop_ = true;
  }
  inspector_cv_.notify_one();
  if (inspector_worker_.joinable())
    inspector_worker_.join();
  spdlog::debug(
      "Runtime log inspector worker summary: requested={}, executed={}, "
      "coalesced={}, stale_results={}",
      inspector_query_requests_.load(std::memory_order_relaxed),
      inspector_query_executions_.load(std::memory_order_relaxed),
      inspector_query_coalesced_.load(std::memory_order_relaxed),
      inspector_query_stale_.load(std::memory_order_relaxed));
}

void Console::RequestInspectorQuery(bool force) {
  const auto newest =
      cyxwiz::RuntimeLogStore::Instance().GetStats().newest_sequence;
  const uint64_t through =
      inspector_paused_ ? inspector_frozen_sequence_ : newest;
  if (!force && inspector_has_submitted_request_ &&
      inspector_last_requested_high_water_ == through &&
      inspector_last_requested_after_sequence_ == inspector_after_sequence_ &&
      inspector_last_requested_criteria_ == inspector_criteria_) {
    return;
  }

  cyxwiz::RuntimeLogInspectorRequest request;
  request.criteria = inspector_criteria_;
  request.after_sequence = inspector_after_sequence_;
  request.through_sequence = through;
  request.display_limit = 1000;
  {
    std::lock_guard<std::mutex> lock(inspector_mutex_);
    if (inspector_request_pending_) {
      inspector_query_coalesced_.fetch_add(1, std::memory_order_relaxed);
    }
    inspector_pending_request_ = std::move(request);
    inspector_request_pending_ = true;
    ++inspector_request_generation_;
    inspector_query_requests_.fetch_add(1, std::memory_order_relaxed);
  }
  inspector_last_requested_high_water_ = through;
  inspector_last_requested_after_sequence_ = inspector_after_sequence_;
  inspector_last_requested_criteria_ = inspector_criteria_;
  inspector_has_submitted_request_ = true;
  inspector_cv_.notify_one();
}

std::shared_ptr<const cyxwiz::RuntimeLogInspectorResult>
Console::SnapshotInspectorResult() const {
  std::lock_guard<std::mutex> lock(inspector_mutex_);
  return inspector_result_;
}

void Console::LoadSavedInspectorFilters() {
  inspector_saved_filters_.clear();
  for (const auto &stored :
       cyxwiz::core::EngineConfig::Instance().GetRuntimeLogSavedFilters()) {
    SavedInspectorFilter saved;
    saved.name = stored.name;
    saved.expression = stored.expression;
    const bool valid_name = !saved.name.empty() && saved.name.size() <= 63 &&
                            std::all_of(saved.name.begin(), saved.name.end(),
                                        [](unsigned char value) {
                                          return std::isalnum(value) != 0 ||
                                                 value == '_' || value == '-';
                                        });
    if (!valid_name) {
      saved.validation_error = "invalid name";
    } else {
      const auto parsed = cyxwiz::ParseRuntimeLogFilter(saved.expression);
      if (!parsed.Ok()) {
        saved.validation_error =
            parsed.error
                ? "position " + std::to_string(parsed.error->position) + ": " +
                      parsed.error->message
                : "invalid expression";
      }
    }
    inspector_saved_filters_.push_back(std::move(saved));
  }
}

bool Console::PersistSavedInspectorFilters() {
  std::vector<cyxwiz::core::RuntimeLogSavedFilterConfig> stored;
  stored.reserve(inspector_saved_filters_.size());
  for (const auto &saved : inspector_saved_filters_) {
    stored.push_back({saved.name, saved.expression});
  }
  auto &config = cyxwiz::core::EngineConfig::Instance();
  config.SetRuntimeLogSavedFilters(stored);
  if (!config.Save()) {
    inspector_filter_status_ = "Failed to persist saved views.";
    inspector_filter_status_error_ = true;
    return false;
  }
  return true;
}

void Console::ExecutePipCommand(const std::vector<std::string> &pip_arguments) {
  auto &pm = cyxwiz::ProjectManager::Instance();

  if (!pm.HasActiveProject()) {
    AddError("No active project - pip commands require an open project");
    return;
  }

  // Get the project's venv pip path
  std::filesystem::path project_root(pm.GetProjectRoot());
  std::filesystem::path venv_pip;

#ifdef _WIN32
  venv_pip = project_root / "python" / "Scripts" / "pip.exe";
#else
  venv_pip = project_root / "python" / "bin" / "pip";
#endif

  if (!std::filesystem::exists(venv_pip)) {
    AddError("Project virtual environment not found");
    AddInfo("Please wait for venv creation to complete or create it manually");
    return;
  }

  AddInfo("Project environment " + project_root.filename().string() +
          "/python; runs in the background, the Engine stays responsive");
  const uint64_t block = current_command_block_;
  spdlog::info("Console executing a project pip command asynchronously");

  // Run command asynchronously using AsyncTaskManager
  auto &task_mgr = cyxwiz::AsyncTaskManager::Instance();

  // The worker only posts to the UI thread; 'this' is dereferenced there,
  // and only while the owner token (a member) is alive.
  Console *console_ptr = this;
  const std::weak_ptr<const void> owner = task_owner_token_;

  const uint64_t task_id = task_mgr.RunAsync(
      "pip command",
      [console_ptr, owner, venv_pip, pip_arguments, block](cyxwiz::LambdaTask &task) {
        // Output is marshalled to the UI thread and dropped once the
        // console is gone; the worker never dereferences it.
        const auto emit = [&owner, console_ptr, block](LogLevel level,
                                                       std::string text) {
          cyxwiz::AsyncTaskManager::Instance().PostToMainThread(
              owner, [console_ptr, level, block, text = std::move(text)] {
                console_ptr->AddBlockLine(block, text, level);
              });
        };
        task.ReportProgress(0.1f, "Starting pip command...");

#ifdef _WIN32
        std::string full_command = QuoteWindowsArgument(venv_pip.string());
        for (const auto &argument : pip_arguments) {
          full_command += " " + QuoteWindowsArgument(argument);
        }

        // Windows: Use CreateProcess with pipes
        SECURITY_ATTRIBUTES sa;
        sa.nLength = sizeof(SECURITY_ATTRIBUTES);
        sa.bInheritHandle = TRUE;
        sa.lpSecurityDescriptor = NULL;

        HANDLE hStdoutRead, hStdoutWrite;
        if (!CreatePipe(&hStdoutRead, &hStdoutWrite, &sa, 0)) {
          emit(LogLevel::Error, "Failed to create pipe for command output");
          task.MarkFailed("Failed to create pipe");
          return;
        }

        SetHandleInformation(hStdoutRead, HANDLE_FLAG_INHERIT, 0);

        STARTUPINFOA si;
        PROCESS_INFORMATION pi;
        ZeroMemory(&si, sizeof(si));
        si.cb = sizeof(si);
        si.hStdError = hStdoutWrite;
        si.hStdOutput = hStdoutWrite;
        si.dwFlags |= STARTF_USESTDHANDLES;
        ZeroMemory(&pi, sizeof(pi));

        std::string cmd_copy =
            full_command; // CreateProcessA modifies the string
        if (!CreateProcessA(NULL, const_cast<char *>(cmd_copy.c_str()), NULL,
                            NULL, TRUE, CREATE_NO_WINDOW, NULL, NULL, &si,
                            &pi)) {
          emit(LogLevel::Error, "Failed to execute pip command");
          CloseHandle(hStdoutRead);
          CloseHandle(hStdoutWrite);
          task.MarkFailed("Failed to create process");
          return;
        }

        CloseHandle(hStdoutWrite);

        task.ReportProgress(0.3f, "Reading pip output...");

        // Read output in real-time
        char buffer[4096];
        DWORD bytes_read;
        std::string line_buffer;

        while (ReadFile(hStdoutRead, buffer, sizeof(buffer) - 1, &bytes_read,
                        NULL) &&
               bytes_read > 0) {
          buffer[bytes_read] = '\0';
          line_buffer += buffer;

          // Process complete lines
          size_t pos;
          while ((pos = line_buffer.find('\n')) != std::string::npos) {
            std::string line = line_buffer.substr(0, pos);
            if (!line.empty() && line.back() == '\r') {
              line.pop_back();
            }
            if (!line.empty()) {
              emit(LogLevel::Info, line);
            }
            line_buffer = line_buffer.substr(pos + 1);
          }

          // Check for cancellation
          if (task.IsCancelRequested()) {
            TerminateProcess(pi.hProcess, 1);
            emit(LogLevel::Warning, "Command cancelled by user");
            break;
          }
        }

        // Print remaining buffer
        if (!line_buffer.empty()) {
          emit(LogLevel::Info, line_buffer);
        }

        WaitForSingleObject(pi.hProcess, INFINITE);

        DWORD exit_code;
        GetExitCodeProcess(pi.hProcess, &exit_code);

        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
        CloseHandle(hStdoutRead);

        task.ReportProgress(1.0f, "Command finished");

        if (exit_code == 0) {
          emit(LogLevel::Success, "Command completed successfully");
        } else {
          emit(LogLevel::Error, "Command failed with exit code: " +
                                std::to_string(exit_code));
          task.MarkFailed("Exit code: " + std::to_string(exit_code));
        }
#else
        int output_pipe[2];
        if (pipe(output_pipe) != 0) {
          emit(LogLevel::Error, "Failed to create pipe for command output");
          task.MarkFailed("Failed to create pipe");
          return;
        }

        const pid_t child = fork();
        if (child < 0) {
          close(output_pipe[0]);
          close(output_pipe[1]);
          emit(LogLevel::Error, "Failed to execute pip command");
          task.MarkFailed("Failed to fork process");
          return;
        }
        if (child == 0) {
          dup2(output_pipe[1], STDOUT_FILENO);
          dup2(output_pipe[1], STDERR_FILENO);
          close(output_pipe[0]);
          close(output_pipe[1]);

          const std::string executable = venv_pip.string();
          std::vector<char *> argv;
          argv.reserve(pip_arguments.size() + 2);
          argv.push_back(const_cast<char *>(executable.c_str()));
          for (const auto &argument : pip_arguments) {
            argv.push_back(const_cast<char *>(argument.c_str()));
          }
          argv.push_back(nullptr);
          execv(executable.c_str(), argv.data());
          _exit(127);
        }

        close(output_pipe[1]);
        FILE *output = fdopen(output_pipe[0], "r");
        if (!output) {
          close(output_pipe[0]);
          kill(child, SIGTERM);
          waitpid(child, nullptr, 0);
          emit(LogLevel::Error, "Failed to read pip command output");
          task.MarkFailed("Failed to open command output");
          return;
        }

        task.ReportProgress(0.3f, "Reading pip output...");

        char buffer[4096];
        bool cancelled = false;
        while (fgets(buffer, sizeof(buffer), output) != nullptr) {
          std::string line(buffer);
          // Remove trailing newline
          if (!line.empty() && line.back() == '\n') {
            line.pop_back();
          }
          if (!line.empty()) {
            emit(LogLevel::Info, line);
          }

          // Check for cancellation
          if (task.IsCancelRequested()) {
            kill(child, SIGTERM);
            emit(LogLevel::Warning, "Command cancelled by user");
            cancelled = true;
            break;
          }
        }

        fclose(output);
        int status = 0;
        waitpid(child, &status, 0);
        if (cancelled)
          return;

        task.ReportProgress(1.0f, "Command finished");

        const int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
        if (exit_code == 0) {
          emit(LogLevel::Success, "Command completed successfully");
        } else {
          emit(LogLevel::Error, "Command failed with exit code: " +
                                std::to_string(exit_code));
          task.MarkFailed("Exit code: " + std::to_string(exit_code));
        }
#endif
      },
      nullptr, // progress callback
      [console_ptr, owner, block](bool success, const std::string &error) {
        if (!success && !error.empty()) {
          spdlog::error("Pip command task failed: {}", error);
        }
        // Completion callbacks run on the UI thread.
        if (owner.lock())
          console_ptr->FinishPipBlock(block, success);
      },
      owner);
  if (block != 0) {
    auto &state = command_blocks_[block];
    state.running = true;
    state.task_id = task_id;
  }
}

void Console::AppendCommandResult(
    const cyxwiz::RuntimeConsoleCommandResult &result) {
  for (const auto &line : result.lines) {
    switch (line.level) {
    case cyxwiz::RuntimeConsoleOutputLevel::Info:
      AddInfo(line.text);
      break;
    case cyxwiz::RuntimeConsoleOutputLevel::Warning:
      AddWarning(line.text);
      break;
    case cyxwiz::RuntimeConsoleOutputLevel::Error:
      AddError(line.text);
      break;
    case cyxwiz::RuntimeConsoleOutputLevel::Success:
      AddSuccess(line.text);
      break;
    case cyxwiz::RuntimeConsoleOutputLevel::Debug:
      AddLog(line.text, LogLevel::Debug);
      break;
    }
  }
}

const char *Console::GetLevelPrefix(LogLevel level) const {
  switch (level) {
  case LogLevel::Info:
    return "[INFO]";
  case LogLevel::Warning:
    return "[WARN]";
  case LogLevel::Error:
    return "[ERROR]";
  case LogLevel::Success:
    return "[OK]";
  case LogLevel::Debug:
    return "[DEBUG]";
  default:
    return "[???]";
  }
}

ImVec4 Console::GetLevelColor(LogLevel level) const {
  switch (level) {
  case LogLevel::Info:
    return ImVec4(0.8f, 0.8f, 0.8f, 1.0f); // Gray
  case LogLevel::Warning:
    return ImVec4(1.0f, 0.8f, 0.0f, 1.0f); // Yellow
  case LogLevel::Error:
    return ImVec4(1.0f, 0.3f, 0.3f, 1.0f); // Red
  case LogLevel::Success:
    return ImVec4(0.3f, 1.0f, 0.3f, 1.0f); // Green
  case LogLevel::Debug:
    return ImVec4(0.6f, 0.6f, 1.0f, 1.0f); // Blue
  default:
    return ImVec4(1.0f, 1.0f, 1.0f, 1.0f); // White
  }
}

void Console::OpenRuntimeLogExportDialog() {
  const auto result = SnapshotInspectorResult();
  if (!result)
    return;

  inspector_export_result_ = result;
  inspector_export_after_sequence_ = inspector_after_sequence_;
  inspector_export_selected_sequence_ = inspector_selected_sequence_;
  inspector_export_selected_scope_ = false;
  inspector_export_format_ = cyxwiz::RuntimeLogExportFormat::JsonLines;
  inspector_export_redaction_ = {};
  inspector_export_popup_requested_ = true;
}

void Console::QueueRuntimeLogExport(const std::filesystem::path &destination) {
  const auto frozen_result = inspector_export_result_;
  if (!frozen_result || destination.empty())
    return;

  auto output_path = destination;
  if (!output_path.has_extension()) {
    output_path +=
        inspector_export_format_ == cyxwiz::RuntimeLogExportFormat::JsonLines
            ? ".jsonl"
            : ".txt";
  }

  const auto selected_sequence =
      inspector_export_selected_scope_
          ? std::optional<uint64_t>(inspector_export_selected_sequence_)
          : std::nullopt;
  const auto after_sequence = inspector_export_after_sequence_;
  const auto format = inspector_export_format_;
  const auto redaction = inspector_export_redaction_;
  const auto task_state = inspector_export_task_state_;
  {
    std::lock_guard<std::mutex> lock(task_state->mutex);
    if (task_state->running)
      return;
    task_state->running = true;
    task_state->success = false;
    task_state->message.clear();
  }

  cyxwiz::AsyncTaskManager::Instance().RunAsync(
      "Export runtime logs",
      [frozen_result, selected_sequence, after_sequence, format, redaction,
       output_path, task_state](cyxwiz::LambdaTask &task) {
        try {
          task.ReportProgress(0.1f, "Freezing runtime-log slice...");
          const auto snapshot = cyxwiz::RuntimeLogExportService::Freeze(
              *frozen_result, after_sequence, selected_sequence);
          if (snapshot.events.empty()) {
            throw std::runtime_error(
                "The frozen runtime-log slice contains no events");
          }

          task.ReportProgress(0.45f, "Redacting and writing export...");
          cyxwiz::RuntimeLogExportRequest request;
          request.destination = output_path;
          request.format = format;
          request.redaction = redaction;
          const auto result =
              cyxwiz::RuntimeLogExportService::Write(snapshot, request);
          if (!result.success)
            throw std::runtime_error(result.error);

          {
            std::lock_guard<std::mutex> lock(task_state->mutex);
            task_state->running = false;
            task_state->success = true;
            task_state->message =
                "Exported " + std::to_string(result.events_written) +
                " runtime-log event(s) to " + result.destination.string();
            task_state->destination = result.destination;
          }
          task.ReportProgress(1.0f, "Runtime-log export complete");
        } catch (const std::exception &error) {
          {
            std::lock_guard<std::mutex> lock(task_state->mutex);
            task_state->running = false;
            task_state->success = false;
            task_state->message =
                std::string("Runtime-log export failed: ") + error.what();
          }
          throw;
        }
      });
}

void Console::ShowCopyNotification() {
  show_copy_notification_ = true;
  copy_notification_time_ = static_cast<float>(ImGui::GetTime());
}

} // namespace gui
