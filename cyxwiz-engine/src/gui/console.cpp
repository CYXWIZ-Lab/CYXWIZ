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
  output << std::put_time(&utc, "%H:%M:%S") << '.' << std::setfill('0')
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

template <size_t Size>
void CopyToBuffer(char (&buffer)[Size], const std::string &value) {
  static_assert(Size > 0);
  std::strncpy(buffer, value.c_str(), Size - 1);
  buffer[Size - 1] = '\0';
}

bool IsValidSavedViewName(const std::string &name) {
  return !name.empty() && name.size() <= 63 &&
         std::all_of(name.begin(), name.end(), [](unsigned char value) {
           return std::isalnum(value) != 0 || value == '_' || value == '-';
         });
}

std::string
BuildSavedViewExpression(const cyxwiz::RuntimeLogInspectorCriteria &criteria) {
  auto controls_only = criteria;
  controls_only.structured_filter.clear();
  if (cyxwiz::BuildRuntimeLogInspectorFilter(controls_only).empty()) {
    return criteria.structured_filter;
  }
  return cyxwiz::BuildRuntimeLogInspectorFilter(criteria);
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
};

Console::Console()
    : scroll_to_bottom_(false), show_window_(true), auto_scroll_(true),
      inspector_export_task_state_(
          std::make_shared<RuntimeLogExportTaskState>()),
      truth_provider_(cyxwiz::CreateEngineRuntimeTruthProvider()),
      command_service_(std::make_unique<cyxwiz::RuntimeConsoleCommandService>(
          cyxwiz::RuntimeLogStore::Instance(), truth_provider_.get())),
      agent_llm_(std::make_unique<cyxwiz::AgentLlmSession>()),
      python_repl_(std::make_unique<cyxwiz::PythonReplSession>()),
      show_copy_notification_(false), copy_notification_time_(0.0f) {
  memset(input_buf_, 0, sizeof(input_buf_));
  LoadSavedInspectorFilters();
  StartInspectorWorker();
}

Console::~Console() {
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

void Console::AppendScriptOutput(const std::string &source,
                                 const std::string &text, bool is_error) {
  python_repl_->AppendScriptOutput(source, text, is_error);
  const auto session = workbench_.EnsureSession(ConsoleSessionKind::PythonRepl);
  if (session) {
    workbench_.MarkUnread(*session.session_id);
  }
}

void Console::Render() {
  if (!show_window_)
    return;

  if (ImGui::Begin("Console", &show_window_)) {
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
    ImGui::TextDisabled("Project: %.*s",
                        static_cast<int>(workbench_.ActiveProjectRoot().size()),
                        workbench_.ActiveProjectRoot().data());
    ImGui::Separator();
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

void Console::RenderLogsSession(bool request_focus) {
  const double now = ImGui::GetTime();
  if (now >= inspector_next_refresh_time_) {
    RequestInspectorQuery(false);
    inspector_next_refresh_time_ = now + 0.1;
  }
  RenderLogsToolbar();
  ImGui::Separator();
  RenderInspectorTab(request_focus);
}

void Console::RenderCommandsSession(bool request_focus) {
  RenderCommandsToolbar();
  ImGui::Separator();
  RenderAllTab();
  ImGui::Separator();
  RenderCommandInput(request_focus);
}

void Console::RenderLogsToolbar() {
  if (ImGui::Button("Clear Log View")) {
    ClearLogView();
  }
  ShowHelpTooltip("Hide retained events through the current high-water mark. "
                  "Canonical runtime evidence is not deleted.");
  ImGui::SameLine();
  ImGui::BeginDisabled(inspector_selected_sequence_ == 0);
  if (ImGui::Button("Copy Selected")) {
    CopySelectedRuntimeLog();
  }
  ShowHelpTooltip("Copy the selected runtime row. You can also press Ctrl+C.",
                  ImGuiHoveredFlags_DelayNormal |
                      ImGuiHoveredFlags_AllowWhenDisabled);
  ImGui::EndDisabled();
  ImGui::SameLine();
  if (ImGui::Button("Copy Filtered Logs")) {
    CopyFilteredRuntimeLogs();
    ShowCopyNotification();
  }
  ShowHelpTooltip(
      "Copy the currently displayed filtered runtime rows to the clipboard "
      "(up to the 1,000-row display limit).");
  ImGui::SameLine();
  bool export_running = false;
  {
    std::lock_guard<std::mutex> lock(inspector_export_task_state_->mutex);
    export_running = inspector_export_task_state_->running;
  }
  ImGui::BeginDisabled(export_running);
  if (ImGui::Button("Export...")) {
    OpenRuntimeLogExportDialog();
  }
  ShowHelpTooltip(
      "Export the current frozen filtered rows or selected row as JSON "
      "Lines or readable text, with an explicit redaction preview.",
      ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_AllowWhenDisabled);
  ImGui::EndDisabled();
  ImGui::SameLine();
  ImGui::Checkbox("Auto-scroll", &auto_scroll_);
  ShowHelpTooltip("Follow newly received runtime events.");
  ImGui::SameLine();
  RenderCopyStatus();

  if (inspector_selected_sequence_ != 0 &&
      ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) &&
      ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_C) &&
      !ImGui::GetIO().WantTextInput) {
    CopySelectedRuntimeLog();
  }
}

void Console::RenderCommandsToolbar() {
  if (ImGui::Button("Clear Commands")) {
    ClearCommandTranscript();
  }
  ShowHelpTooltip(
      "Clear only the command transcript. Runtime logs are unchanged.");
  ImGui::SameLine();
  ImGui::BeginDisabled(selected_command_sequence_ == 0);
  if (ImGui::Button("Copy Selected")) {
    CopySelectedCommand();
  }
  ShowHelpTooltip(
      "Copy the selected command line or response. You can also press Ctrl+C.",
      ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_AllowWhenDisabled);
  ImGui::EndDisabled();
  ImGui::SameLine();
  if (ImGui::Button("Copy All Commands")) {
    CopyCommandTranscript();
    ShowCopyNotification();
  }
  ShowHelpTooltip(
      "Copy entered commands and their retained responses to the clipboard.");
  ImGui::SameLine();
  ImGui::Checkbox("Auto-scroll", &auto_scroll_);
  ShowHelpTooltip("Follow new command responses.");
  ImGui::SameLine();
  RenderCopyStatus();

  if (selected_command_sequence_ != 0 &&
      ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) &&
      ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_C) &&
      !ImGui::GetIO().WantTextInput) {
    CopySelectedCommand();
  }
}

void Console::RenderCommandInput(bool request_focus) {
  if (request_focus)
    command_input_focus_pending_ = true;
  bool reclaim_focus = false;
  const ImGuiInputTextFlags input_text_flags =
      ImGuiInputTextFlags_EnterReturnsTrue |
      ImGuiInputTextFlags_CallbackHistory;
  ImGui::PushItemWidth(-1.0f);
  if (command_input_focus_pending_)
    ImGui::SetKeyboardFocusHere();
  const bool submitted = ImGui::InputTextWithHint(
      "##input", "Enter command...", input_buf_, IM_ARRAYSIZE(input_buf_),
      input_text_flags, &Console::InputTextCallback, this);
  if (command_input_focus_pending_ && ImGui::IsItemActive())
    command_input_focus_pending_ = false;
  if (submitted) {
    if (input_buf_[0]) {
      ExecCommand(input_buf_);
    }
    input_buf_[0] = '\0';
    reclaim_focus = true;
  }
  ImGui::PopItemWidth();
  ShowHelpTooltip(
      "Execute an interactive Console command. Use Up and Down to navigate "
      "command history.");

  ImGui::SetItemDefaultFocus();
  if (reclaim_focus) {
    ImGui::SetKeyboardFocusHere(-1);
  }
}

void Console::RenderCopyStatus() {
  if (show_copy_notification_) {
    const float elapsed =
        static_cast<float>(ImGui::GetTime()) - copy_notification_time_;
    if (elapsed < 2.0f) {
      ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 1.0f, 0.3f, 1.0f));
      ImGui::Text("Copied!");
      ImGui::PopStyleColor();
      return;
    }
    show_copy_notification_ = false;
  }
  ImGui::TextDisabled("Ready");
}

void Console::RenderRuntimeLogExportStatus() {
  bool running = false;
  bool success = false;
  std::string message;
  {
    std::lock_guard<std::mutex> lock(inspector_export_task_state_->mutex);
    running = inspector_export_task_state_->running;
    success = inspector_export_task_state_->success;
    message = inspector_export_task_state_->message;
  }
  if (running) {
    ImGui::TextDisabled("Exporting frozen runtime-log slice...");
  } else if (!message.empty()) {
    ImGui::PushStyleColor(ImGuiCol_Text, success
                                             ? ImVec4(0.3f, 0.85f, 0.4f, 1.0f)
                                             : ImVec4(1.0f, 0.4f, 0.35f, 1.0f));
    ImGui::TextWrapped("%s", message.c_str());
    ImGui::PopStyleColor();
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
    if (ImGui::Button("Close"))
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
  if (ImGui::Button("Shareable preset")) {
    inspector_export_redaction_ = {};
  }
  ShowHelpTooltip("Enable every supported sensitive-data redaction.");
  ImGui::SameLine();
  if (ImGui::Button("Raw preset")) {
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
  ImGui::BeginDisabled(export_running || preview_event == nullptr);
  if (ImGui::Button("Save export...")) {
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
  ImGui::EndDisabled();
  ImGui::SameLine();
  if (ImGui::Button("Cancel"))
    ImGui::CloseCurrentPopup();
  ImGui::EndPopup();
}

void Console::RenderInspectorTab(bool request_focus) {
  const auto result = SnapshotInspectorResult();
  RenderInspectorFilters(result.get(), request_focus);
  RenderRuntimeLogExportStatus();
  RenderRuntimeLogExportDialog();

  if (!result) {
    ImGui::TextDisabled("Loading runtime events...");
    return;
  }
  if (result->filter_error) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.35f, 1.0f));
    ImGui::TextWrapped("Filter error at %zu: %s",
                       result->filter_error->position,
                       result->filter_error->message.c_str());
    ImGui::PopStyleColor();
  }
  if (!result->effective_filter.empty()) {
    constexpr size_t max_visible_filter = 140;
    const std::string visible_filter =
        result->effective_filter.size() <= max_visible_filter
            ? result->effective_filter
            : result->effective_filter.substr(0, max_visible_filter - 3) +
                  "...";
    ImGui::TextDisabled("Active: %s", visible_filter.c_str());
    ShowHelpTooltip(result->effective_filter.c_str());
  }

  const auto &query = result->query;
  ImGui::TextDisabled(
      "Showing %zu / matched %zu | retained %zu / %zu | evicted %llu | "
      "high-water %llu%s",
      query.events.size(), query.matched_count, query.store_stats.size,
      query.store_stats.capacity,
      static_cast<unsigned long long>(query.store_stats.evicted_count),
      static_cast<unsigned long long>(query.high_water_sequence),
      inspector_paused_ ? " | paused" : "");
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
    ImGui::SetTooltip(
        "Query worker: requested %llu, executed %llu, coalesced %llu, "
        "stale results discarded %llu. The queue holds at most one "
        "pending request.",
        static_cast<unsigned long long>(
            inspector_query_requests_.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(
            inspector_query_executions_.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(
            inspector_query_coalesced_.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(
            inspector_query_stale_.load(std::memory_order_relaxed)));
  }

  ImGui::BeginChild("RuntimeLogInspectorBody", ImVec2(0, 0), false);
  const bool has_selection = std::any_of(
      query.events.begin(), query.events.end(), [this](const auto &event) {
        return event.sequence == inspector_selected_sequence_;
      });
  if (has_selection) {
    const float available_width = ImGui::GetContentRegionAvail().x;
    const float details_width =
        std::clamp(available_width * 0.32f, 220.0f, 320.0f);
    if (ImGui::BeginTable("RuntimeLogInspectorSplit", 2,
                          ImGuiTableFlags_Resizable |
                              ImGuiTableFlags_BordersInnerV)) {
      ImGui::TableSetupColumn("Rows", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Details", ImGuiTableColumnFlags_WidthFixed,
                              details_width);
      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0);
      RenderInspectorTable(result.get());
      ImGui::TableSetColumnIndex(1);
      RenderInspectorDetails(result.get());
      ImGui::EndTable();
    }
  } else {
    RenderInspectorTable(result.get());
  }
  ImGui::EndChild();
}

void Console::RenderInspectorFilters(
    const cyxwiz::RuntimeLogInspectorResult *result, bool request_focus) {
  if (request_focus)
    log_search_focus_pending_ = true;
  bool changed = false;
  static constexpr std::array<const char *, 6> level_labels = {
      "Trace", "Debug", "Info", "Warn", "Error", "Critical"};
  static constexpr std::array<const char *, 6> level_help = {
      "Most detailed severity. Supported by the engine, but normally "
      "filtered because Release starts at Info and Debug starts at Debug.",
      "Developer diagnostic severity. Enabled by the default Debug build.",
      "Normal runtime progress and lifecycle information.",
      "A recoverable problem or condition requiring attention.",
      "An operation failed but the process may continue.",
      "Highest severity for fatal or process-ending failures."};
  for (size_t index = 0; index < inspector_criteria_.levels.size(); ++index) {
    if (index != 0)
      ImGui::SameLine();
    changed |= ImGui::Checkbox(level_labels[index],
                               &inspector_criteria_.levels[index]);
    ShowHelpTooltip(level_help[index]);
  }
  ImGui::SameLine();
  if (ImGui::Button(inspector_paused_ ? "Resume" : "Pause")) {
    inspector_paused_ = !inspector_paused_;
    if (inspector_paused_) {
      inspector_frozen_sequence_ =
          cyxwiz::RuntimeLogStore::Instance().GetStats().newest_sequence;
    }
    changed = true;
  }
  ShowHelpTooltip(
      inspector_paused_
          ? "Resume the live log tail from the current runtime high-water mark."
          : "Freeze this view at its current high-water mark while ingestion "
            "continues.");
  ImGui::SameLine();
  if (ImGui::Button("Reset")) {
    inspector_criteria_ = {};
    inspector_text_[0] = '\0';
    inspector_filter_[0] = '\0';
    inspector_filter_name_[0] = '\0';
    inspector_selected_saved_filter_ = -1;
    inspector_selected_sequence_ = 0;
    inspector_filter_status_.clear();
    changed = true;
  }
  ShowHelpTooltip("Reset all runtime-log filters and row selection.");
  ImGui::SameLine();
  if (ImGui::Button("Show retained")) {
    inspector_after_sequence_ = 0;
    changed = true;
  }
  ShowHelpTooltip("Show retained events hidden by Clear Log View. Store "
                  "eviction still applies.");

  ImGui::SetNextItemWidth(220.0f);
  if (log_search_focus_pending_)
    ImGui::SetKeyboardFocusHere();
  const bool search_changed = ImGui::InputTextWithHint(
      "##runtime_text", "Search messages (case-insensitive)...",
      inspector_text_, IM_ARRAYSIZE(inspector_text_));
  if (log_search_focus_pending_ && ImGui::IsItemActive())
    log_search_focus_pending_ = false;
  if (search_changed) {
    inspector_criteria_.text = inspector_text_;
    changed = true;
  }
  ImGui::SameLine();
  const size_t field_filter_count =
      static_cast<size_t>(!inspector_criteria_.category.empty()) +
      static_cast<size_t>(!inspector_criteria_.source.empty()) +
      static_cast<size_t>(!inspector_criteria_.code.empty()) +
      static_cast<size_t>(!inspector_criteria_.run_id.empty()) +
      static_cast<size_t>(!inspector_criteria_.backend.empty()) +
      static_cast<size_t>(inspector_criteria_.task_id.has_value()) +
      static_cast<size_t>(inspector_criteria_.device_id.has_value());
  const std::string filters_label =
      field_filter_count == 0
          ? "Filters"
          : "Filters (" + std::to_string(field_filter_count) + ')';
  const float trailing_buttons_width =
      ImGui::CalcTextSize("?").x +
      ImGui::CalcTextSize(filters_label.c_str()).x +
      ImGui::GetStyle().FramePadding.x * 4.0f +
      ImGui::GetStyle().ItemSpacing.x * 2.0f;
  ImGui::SetNextItemWidth(std::max(120.0f, ImGui::GetContentRegionAvail().x -
                                               trailing_buttons_width));
  if (ImGui::InputTextWithHint("##runtime_filter", "Structured filter...",
                               inspector_filter_,
                               IM_ARRAYSIZE(inspector_filter_))) {
    inspector_criteria_.structured_filter = inspector_filter_;
    changed = true;
  }
  ImGui::SameLine();
  if (ImGui::SmallButton("?##StructuredFilterHelp")) {
    ImGui::OpenPopup("StructuredFilterHelp");
  }
  ShowHelpTooltip("Structured filter syntax and examples.");
  ImGui::SameLine();
  if (ImGui::Button(filters_label.c_str())) {
    ImGui::OpenPopup("RuntimeLogAdvancedFilters");
  }
  ShowHelpTooltip("Open category, source, code, run, backend, task, device, "
                  "and saved-filter controls.");

  ImGui::SetNextWindowSize(ImVec2(680.0f, 560.0f), ImGuiCond_Appearing);
  if (ImGui::BeginPopup("StructuredFilterHelp")) {
    ImGui::TextUnformatted("Structured filter help");
    ImGui::Separator();
    ImGui::BeginChild("StructuredFilterHelpContent", ImVec2(0, 0), false);
    const auto help = cyxwiz::RuntimeLogFilterHelpText();
    ImGui::TextUnformatted(help.data(), help.data() + help.size());
    ImGui::EndChild();
    ImGui::EndPopup();
  }

  ImGui::SetNextWindowSize(ImVec2(520.0f, 0.0f), ImGuiCond_Appearing);
  if (ImGui::BeginPopup("RuntimeLogAdvancedFilters")) {
    ImGui::TextUnformatted("Field filters");
    ImGui::TextDisabled("Each selection is combined with AND.");
    if (result && ImGui::BeginTable("RuntimeLogFieldFilters", 2,
                                    ImGuiTableFlags_SizingStretchSame)) {
      const auto string_combo =
          [&changed](const char *label, const char *id, std::string &selected,
                     const std::vector<std::string> &values) {
            ImGui::TableNextColumn();
            ImGui::TextDisabled("%s", label);
            ImGui::SetNextItemWidth(-1.0f);
            if (!ImGui::BeginCombo(id, selected.empty() ? "Any"
                                                        : selected.c_str())) {
              return;
            }
            if (ImGui::Selectable("Any", selected.empty())) {
              selected.clear();
              changed = true;
            }
            for (const auto &value : values) {
              if (ImGui::Selectable(value.c_str(), selected == value)) {
                selected = value;
                changed = true;
              }
            }
            ImGui::EndCombo();
          };

      const auto &facets = result->query.facets;
      string_combo("Category", "##FilterCategory", inspector_criteria_.category,
                   facets.categories);
      string_combo("Source", "##FilterSource", inspector_criteria_.source,
                   facets.sources);
      string_combo("Code", "##FilterCode", inspector_criteria_.code,
                   facets.codes);
      string_combo("Run", "##FilterRun", inspector_criteria_.run_id,
                   facets.run_ids);
      string_combo("Backend", "##FilterBackend", inspector_criteria_.backend,
                   facets.backends);

      ImGui::TableNextColumn();
      ImGui::TextDisabled("Task");
      ImGui::SetNextItemWidth(-1.0f);
      const std::string task_preview =
          inspector_criteria_.task_id
              ? std::to_string(*inspector_criteria_.task_id)
              : "Any";
      if (ImGui::BeginCombo("##FilterTask", task_preview.c_str())) {
        if (ImGui::Selectable("Any", !inspector_criteria_.task_id)) {
          inspector_criteria_.task_id.reset();
          changed = true;
        }
        for (const auto value : facets.task_ids) {
          const auto label = std::to_string(value);
          if (ImGui::Selectable(label.c_str(),
                                inspector_criteria_.task_id == value)) {
            inspector_criteria_.task_id = value;
            changed = true;
          }
        }
        ImGui::EndCombo();
      }

      ImGui::TableNextColumn();
      ImGui::TextDisabled("Device");
      ImGui::SetNextItemWidth(-1.0f);
      const std::string device_preview =
          inspector_criteria_.device_id
              ? std::to_string(*inspector_criteria_.device_id)
              : "Any";
      if (ImGui::BeginCombo("##FilterDevice", device_preview.c_str())) {
        if (ImGui::Selectable("Any", !inspector_criteria_.device_id)) {
          inspector_criteria_.device_id.reset();
          changed = true;
        }
        for (const auto value : facets.device_ids) {
          const auto label = std::to_string(value);
          if (ImGui::Selectable(label.c_str(),
                                inspector_criteria_.device_id == value)) {
            inspector_criteria_.device_id = value;
            changed = true;
          }
        }
        ImGui::EndCombo();
      }
      ImGui::EndTable();
    } else if (!result) {
      ImGui::TextDisabled("Log fields are still loading.");
    }

    ImGui::Spacing();
    ImGui::SeparatorText("Saved views");
    ImGui::TextDisabled("A saved view captures search, severity, fields, and "
                        "structured filter.");
    const bool has_saved_selection =
        inspector_selected_saved_filter_ >= 0 &&
        inspector_selected_saved_filter_ <
            static_cast<int>(inspector_saved_filters_.size());
    std::string saved_preview = "Custom";
    if (has_saved_selection) {
      const auto &saved =
          inspector_saved_filters_[inspector_selected_saved_filter_];
      saved_preview = saved.name;
      if (BuildSavedViewExpression(inspector_criteria_) != saved.expression) {
        saved_preview += " (modified)";
      }
    }
    ImGui::SetNextItemWidth(-1.0f);
    if (ImGui::BeginCombo("##SavedRuntimeLogView", saved_preview.c_str())) {
      if (ImGui::Selectable("Custom", !has_saved_selection)) {
        inspector_selected_saved_filter_ = -1;
        inspector_filter_name_[0] = '\0';
        inspector_filter_status_.clear();
      }
      for (size_t index = 0; index < inspector_saved_filters_.size(); ++index) {
        const auto &saved = inspector_saved_filters_[index];
        const std::string label = saved.validation_error.empty()
                                      ? saved.name
                                      : saved.name + " (invalid)";
        if (ImGui::Selectable(label.c_str(), inspector_selected_saved_filter_ ==
                                                 static_cast<int>(index))) {
          inspector_selected_saved_filter_ = static_cast<int>(index);
          inspector_criteria_ = {};
          inspector_text_[0] = '\0';
          inspector_criteria_.structured_filter = saved.expression;
          CopyToBuffer(inspector_filter_, saved.expression);
          inspector_filter_name_[0] = '\0';
          inspector_filter_status_ = "Applied saved view '" + saved.name + "'";
          inspector_filter_status_error_ = false;
          changed = true;
        }
      }
      ImGui::EndCombo();
    }
    ShowHelpTooltip(
        "Select a saved view to apply its complete filter immediately.");

    ImGui::TextDisabled("New view name");
    ImGui::SetNextItemWidth(-1.0f);
    ImGui::InputTextWithHint(
        "##SavedRuntimeLogViewName", "View name (letters, digits, _ or -)",
        inspector_filter_name_, IM_ARRAYSIZE(inspector_filter_name_));

    const auto set_status = [this](std::string message, bool error) {
      inspector_filter_status_ = std::move(message);
      inspector_filter_status_error_ = error;
    };
    const auto validate_current = [this, &set_status](std::string &expression) {
      expression = BuildSavedViewExpression(inspector_criteria_);
      if (expression.empty()) {
        set_status("Set at least one filter before saving a view.", true);
        return false;
      }
      const auto parsed = cyxwiz::ParseRuntimeLogFilter(expression);
      if (!parsed.Ok()) {
        set_status("Fix the active filter before saving this view.", true);
        return false;
      }
      return true;
    };

    if (ImGui::Button("Save As")) {
      const std::string name = inspector_filter_name_;
      std::string expression;
      const auto existing = std::find_if(
          inspector_saved_filters_.begin(), inspector_saved_filters_.end(),
          [&name](const auto &saved) { return saved.name == name; });
      if (!IsValidSavedViewName(name)) {
        set_status("View name must use letters, digits, '_' or '-' (max 63).",
                   true);
      } else if (existing != inspector_saved_filters_.end()) {
        set_status("That view already exists. Select it and use Update.", true);
      } else if (inspector_saved_filters_.size() >= 32) {
        set_status("At most 32 saved views are allowed.", true);
      } else if (validate_current(expression)) {
        inspector_saved_filters_.push_back({name, expression, {}});
        inspector_selected_saved_filter_ =
            static_cast<int>(inspector_saved_filters_.size() - 1);
        if (PersistSavedInspectorFilters()) {
          set_status("Saved view '" + name + "'", false);
        }
      }
    }
    ShowHelpTooltip("Create a new view from every currently active filter.");
    ImGui::SameLine();

    ImGui::BeginDisabled(!has_saved_selection);
    if (ImGui::Button("Update") && has_saved_selection) {
      std::string expression;
      if (validate_current(expression)) {
        auto &saved =
            inspector_saved_filters_[inspector_selected_saved_filter_];
        saved.expression = std::move(expression);
        saved.validation_error.clear();
        if (PersistSavedInspectorFilters()) {
          set_status("Updated view '" + saved.name + "'", false);
        }
      }
    }
    ShowHelpTooltip(
        "Replace the selected view with every currently active filter.",
        ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_AllowWhenDisabled);
    ImGui::SameLine();
    if (ImGui::Button("Delete") && has_saved_selection) {
      const std::string name =
          inspector_saved_filters_[inspector_selected_saved_filter_].name;
      inspector_saved_filters_.erase(inspector_saved_filters_.begin() +
                                     inspector_selected_saved_filter_);
      inspector_selected_saved_filter_ = -1;
      inspector_filter_name_[0] = '\0';
      if (PersistSavedInspectorFilters()) {
        set_status("Deleted view '" + name + "'", false);
      }
    }
    ShowHelpTooltip(
        "Delete the selected saved view. The active filter remains applied.",
        ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_AllowWhenDisabled);
    ImGui::EndDisabled();

    if (!inspector_filter_status_.empty()) {
      const ImVec4 color = inspector_filter_status_error_
                               ? ImVec4(1.0f, 0.4f, 0.35f, 1.0f)
                               : ImVec4(0.35f, 0.85f, 0.45f, 1.0f);
      ImGui::PushStyleColor(ImGuiCol_Text, color);
      ImGui::TextWrapped("%s", inspector_filter_status_.c_str());
      ImGui::PopStyleColor();
    }
    const bool still_has_saved_selection =
        inspector_selected_saved_filter_ >= 0 &&
        inspector_selected_saved_filter_ <
            static_cast<int>(inspector_saved_filters_.size());
    if (still_has_saved_selection &&
        !inspector_saved_filters_[inspector_selected_saved_filter_]
             .validation_error.empty()) {
      ImGui::TextWrapped(
          "Saved view error: %s",
          inspector_saved_filters_[inspector_selected_saved_filter_]
              .validation_error.c_str());
    }
    ImGui::EndPopup();
  }

  if (changed)
    RequestInspectorQuery(true);
}

void Console::RenderInspectorTable(
    const cyxwiz::RuntimeLogInspectorResult *result) {
  ImGui::BeginChild("RuntimeLogRows", ImVec2(0, 0), false);
  const ImGuiTableFlags flags =
      ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV |
      ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY |
      ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_NoSavedSettings;
  if (ImGui::BeginTable("RuntimeLogTable", 8, flags)) {
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableSetupColumn("Time", ImGuiTableColumnFlags_WidthFixed, 92.0f);
    ImGui::TableSetupColumn("Level", ImGuiTableColumnFlags_WidthFixed, 58.0f);
    ImGui::TableSetupColumn("Category", ImGuiTableColumnFlags_WidthFixed,
                            82.0f);
    ImGui::TableSetupColumn("Source", ImGuiTableColumnFlags_WidthFixed, 110.0f);
    ImGui::TableSetupColumn("Code", ImGuiTableColumnFlags_WidthFixed, 82.0f);
    ImGui::TableSetupColumn("Run", ImGuiTableColumnFlags_WidthFixed, 126.0f);
    ImGui::TableSetupColumn("Device", ImGuiTableColumnFlags_WidthFixed, 92.0f);
    ImGui::TableSetupColumn("Message", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableHeadersRow();

    ImGuiListClipper clipper;
    clipper.Begin(static_cast<int>(result->query.events.size()));
    while (clipper.Step()) {
      for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {
        const auto &event = result->query.events[row];
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        const auto timestamp = FormatLogTimestamp(event.timestamp_utc);
        const bool selected = inspector_selected_sequence_ == event.sequence;
        ImGui::PushID(static_cast<int>(event.sequence));
        if (ImGui::Selectable(timestamp.c_str(), selected,
                              ImGuiSelectableFlags_SpanAllColumns |
                                  ImGuiSelectableFlags_AllowDoubleClick)) {
          inspector_selected_sequence_ = event.sequence;
          if (ImGui::IsMouseDoubleClicked(0)) {
            const auto line = FormatRuntimeLogRow(event);
            ImGui::SetClipboardText(line.c_str());
            ShowCopyNotification();
          }
        }
        if (ImGui::BeginPopupContextItem("RuntimeLogContext")) {
          inspector_selected_sequence_ = event.sequence;
          if (ImGui::MenuItem("Copy message")) {
            ImGui::SetClipboardText(event.message.c_str());
            ShowCopyNotification();
          }
          if (ImGui::MenuItem("Copy row")) {
            const auto line = FormatRuntimeLogRow(event);
            ImGui::SetClipboardText(line.c_str());
            ShowCopyNotification();
          }
          ImGui::EndPopup();
        }
        ImGui::PopID();

        ImGui::TableSetColumnIndex(1);
        const auto level_name = cyxwiz::RuntimeLogLevelName(event.level);
        ImGui::TextUnformatted(level_name.data(),
                               level_name.data() + level_name.size());
        ImGui::TableSetColumnIndex(2);
        ImGui::TextUnformatted(event.category.c_str());
        ImGui::TableSetColumnIndex(3);
        ImGui::TextUnformatted(event.source.c_str());
        ImGui::TableSetColumnIndex(4);
        ImGui::TextUnformatted(event.primary_error_code.c_str());
        ImGui::TableSetColumnIndex(5);
        ImGui::TextUnformatted(event.run_id.c_str());
        ImGui::TableSetColumnIndex(6);
        if (!event.backend.empty() || event.device_id >= 0) {
          ImGui::Text("%s:%d", event.backend.c_str(), event.device_id);
        }
        ImGui::TableSetColumnIndex(7);
        ImGui::TextUnformatted(event.message.c_str());
      }
    }
    if (auto_scroll_ && !inspector_paused_ &&
        inspector_last_rendered_high_water_ !=
            result->query.high_water_sequence) {
      ImGui::SetScrollHereY(1.0f);
    }
    inspector_last_rendered_high_water_ = result->query.high_water_sequence;
    ImGui::EndTable();
  }
  ImGui::EndChild();
}

void Console::RenderInspectorDetails(
    const cyxwiz::RuntimeLogInspectorResult *result) {
  const auto selected =
      std::find_if(result->query.events.begin(), result->query.events.end(),
                   [this](const auto &event) {
                     return event.sequence == inspector_selected_sequence_;
                   });
  ImGui::BeginChild("RuntimeLogDetails", ImVec2(0, 0), false);
  if (selected == result->query.events.end()) {
    ImGui::TextDisabled("Select a row to inspect structured fields");
    ImGui::EndChild();
    return;
  }
  const auto &event = *selected;
  auto details = FormatRuntimeLogDetails(event);
  ImGui::InputTextMultiline("##RuntimeLogSelectedDetails", details.data(),
                            details.size() + 1, ImVec2(-1.0f, -1.0f),
                            ImGuiInputTextFlags_ReadOnly |
                                ImGuiInputTextFlags_NoHorizontalScroll);
  ShowHelpTooltip(
      "Select any text to copy it, or use Copy Selected for the complete row.");
  ImGui::EndChild();
}

void Console::RenderAllTab() {
  const float footer_height =
      ImGui::GetStyle().ItemSpacing.y + ImGui::GetFrameHeightWithSpacing();
  ImGui::BeginChild("AllLogsRegion", ImVec2(0, -footer_height), false,
                    ImGuiWindowFlags_HorizontalScrollbar);

  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 1));

  const auto entries = SnapshotEntries();
  for (const auto &entry : entries) {
    ImVec4 color = GetLevelColor(entry.level);
    ImGui::PushStyleColor(ImGuiCol_Text, color);

    // Use Selectable for right-click support
    char buf[1024];
    const auto timestamp = FormatLogTimestamp(entry.timestamp);
    snprintf(buf, sizeof(buf), "[%s] %s %s", timestamp.c_str(),
             GetLevelPrefix(entry.level), entry.message.c_str());

    ImGui::PushID(static_cast<int>(entry.sequence));
    const bool selected = selected_command_sequence_ == entry.sequence;
    if (ImGui::Selectable(buf, selected,
                          ImGuiSelectableFlags_AllowDoubleClick)) {
      selected_command_sequence_ = entry.sequence;
      if (ImGui::IsMouseDoubleClicked(0)) {
        ImGui::SetClipboardText(buf);
        ShowCopyNotification();
      }
    }

    if (ImGui::BeginPopupContextItem("LogContextMenu")) {
      selected_command_sequence_ = entry.sequence;
      if (ImGui::MenuItem("Copy Message")) {
        ImGui::SetClipboardText(entry.message.c_str());
        ShowCopyNotification();
      }
      if (ImGui::MenuItem("Copy Full Line")) {
        ImGui::SetClipboardText(buf);
        ShowCopyNotification();
      }
      ImGui::EndPopup();
    }
    ImGui::PopID();

    ImGui::PopStyleColor();
  }

  if (auto_scroll_ && (scroll_to_bottom_.load(std::memory_order_relaxed) ||
                       ImGui::GetScrollY() >= ImGui::GetScrollMaxY())) {
    ImGui::SetScrollHereY(1.0f);
  }

  ImGui::PopStyleVar();
  ImGui::EndChild();
}

void Console::AddLog(const std::string &message, LogLevel level) {
  std::lock_guard<std::mutex> lock(log_mutex_);
  LogEntry entry;
  entry.message = message;
  entry.level = level;
  entry.timestamp = std::chrono::system_clock::now();
  entry.sequence = ++next_command_sequence_;
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

  AddInfo("Executing project pip command");
  AddInfo("Running in background (UI remains responsive)...");
  spdlog::info("Console executing a project pip command asynchronously");

  // Run command asynchronously using AsyncTaskManager
  auto &task_mgr = cyxwiz::AsyncTaskManager::Instance();

  // Capture 'this' pointer for thread-safe logging
  Console *console_ptr = this;

  task_mgr.RunAsync(
      "pip command",
      [console_ptr, venv_pip, pip_arguments](cyxwiz::LambdaTask &task) {
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
          console_ptr->AddError("Failed to create pipe for command output");
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
          console_ptr->AddError("Failed to execute pip command");
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
              console_ptr->AddInfo(line);
            }
            line_buffer = line_buffer.substr(pos + 1);
          }

          // Check for cancellation
          if (task.IsCancelRequested()) {
            TerminateProcess(pi.hProcess, 1);
            console_ptr->AddWarning("Command cancelled by user");
            break;
          }
        }

        // Print remaining buffer
        if (!line_buffer.empty()) {
          console_ptr->AddInfo(line_buffer);
        }

        WaitForSingleObject(pi.hProcess, INFINITE);

        DWORD exit_code;
        GetExitCodeProcess(pi.hProcess, &exit_code);

        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
        CloseHandle(hStdoutRead);

        task.ReportProgress(1.0f, "Command finished");

        if (exit_code == 0) {
          console_ptr->AddSuccess("Command completed successfully");
        } else {
          console_ptr->AddError("Command failed with exit code: " +
                                std::to_string(exit_code));
          task.MarkFailed("Exit code: " + std::to_string(exit_code));
        }
#else
        int output_pipe[2];
        if (pipe(output_pipe) != 0) {
          console_ptr->AddError("Failed to create pipe for command output");
          task.MarkFailed("Failed to create pipe");
          return;
        }

        const pid_t child = fork();
        if (child < 0) {
          close(output_pipe[0]);
          close(output_pipe[1]);
          console_ptr->AddError("Failed to execute pip command");
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
          console_ptr->AddError("Failed to read pip command output");
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
            console_ptr->AddInfo(line);
          }

          // Check for cancellation
          if (task.IsCancelRequested()) {
            kill(child, SIGTERM);
            console_ptr->AddWarning("Command cancelled by user");
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
          console_ptr->AddSuccess("Command completed successfully");
        } else {
          console_ptr->AddError("Command failed with exit code: " +
                                std::to_string(exit_code));
          task.MarkFailed("Exit code: " + std::to_string(exit_code));
        }
#endif
      },
      nullptr, // progress callback
      [console_ptr](bool success, const std::string &error) {
        if (!success && !error.empty()) {
          spdlog::error("Pip command task failed: {}", error);
        }
      });
}

void Console::ExecCommand(const char *command) {
  AddLog(std::string("> ") + command, LogLevel::Info);
  const auto result = command_service_->Execute(command);
  switch (result.action) {
  case cyxwiz::RuntimeConsoleAction::Clear:
    ClearCommandTranscript();
    break;
  case cyxwiz::RuntimeConsoleAction::ExecutePip:
    ExecutePipCommand(result.action_arguments);
    break;
  case cyxwiz::RuntimeConsoleAction::None:
    break;
  }
  AppendCommandResult(result);
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

int Console::InputTextCallback(ImGuiInputTextCallbackData *data) {
  return static_cast<Console *>(data->UserData)->HandleInputTextCallback(data);
}

int Console::HandleInputTextCallback(ImGuiInputTextCallbackData *data) {
  if (data->EventFlag != ImGuiInputTextFlags_CallbackHistory)
    return 0;
  const auto command = data->EventKey == ImGuiKey_UpArrow
                           ? command_service_->PreviousCommand()
                           : command_service_->NextCommand();
  if (!command)
    return 0;
  data->DeleteChars(0, data->BufTextLen);
  data->InsertChars(0, command->c_str());
  return 0;
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

void Console::CopyCommandTranscript() {
  std::ostringstream ss;
  for (const auto &entry : SnapshotEntries()) {
    ss << "[" << FormatLogTimestamp(entry.timestamp) << "] "
       << GetLevelPrefix(entry.level) << " " << entry.message << "\n";
  }
  ImGui::SetClipboardText(ss.str().c_str());
}

void Console::CopySelectedCommand() {
  const auto entries = SnapshotEntries();
  const auto selected =
      std::find_if(entries.begin(), entries.end(), [this](const auto &entry) {
        return entry.sequence == selected_command_sequence_;
      });
  if (selected == entries.end()) {
    selected_command_sequence_ = 0;
    return;
  }
  std::ostringstream output;
  output << '[' << FormatLogTimestamp(selected->timestamp) << "] "
         << GetLevelPrefix(selected->level) << ' ' << selected->message;
  ImGui::SetClipboardText(output.str().c_str());
  ShowCopyNotification();
}

void Console::CopyFilteredRuntimeLogs() {
  const auto result = SnapshotInspectorResult();
  if (!result)
    return;
  std::ostringstream output;
  for (const auto &event : result->query.events) {
    output << FormatRuntimeLogRow(event) << '\n';
  }
  ImGui::SetClipboardText(output.str().c_str());
}

void Console::CopySelectedRuntimeLog() {
  const auto result = SnapshotInspectorResult();
  if (!result)
    return;
  const auto selected =
      std::find_if(result->query.events.begin(), result->query.events.end(),
                   [this](const auto &event) {
                     return event.sequence == inspector_selected_sequence_;
                   });
  if (selected == result->query.events.end()) {
    inspector_selected_sequence_ = 0;
    return;
  }
  const auto output = FormatRuntimeLogRow(*selected);
  ImGui::SetClipboardText(output.c_str());
  ShowCopyNotification();
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
