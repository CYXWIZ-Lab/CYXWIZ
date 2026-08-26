#pragma once

#include "../core/runtime_log_export.h"
#include "../core/runtime_log_inspector.h"
#include "../plugin/interfaces/i_assistant_provider.h"
#include "../scripting/script_output_sink.h"
#include "console_workbench.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

struct ImVec4;
struct ImGuiInputTextCallbackData;

namespace scripting {
class ScriptingEngine;
}

namespace cyxwiz {
enum class RuntimeLogLevel : uint8_t;
class AgentLlmSession;
class LocalShellSession;
class PythonReplSession;
struct RuntimeConsoleCommandResult;
class RuntimeConsoleCommandService;
class RuntimeTruthQueryProvider;
} // namespace cyxwiz

namespace gui {

class Console : public scripting::IScriptOutputSink {
public:
  enum class LogLevel { Info, Warning, Error, Success, Debug };

  struct LogEntry {
    std::string message;
    LogLevel level;
    std::chrono::system_clock::time_point timestamp;
    uint64_t sequence = 0;
  };

  Console();
  ~Console();

  void Render();
  void SetScriptingEngine(
      std::shared_ptr<scripting::ScriptingEngine> scripting_engine);
  void SetAssistantCommandHandler(
      std::function<cyxwiz::plugin::AssistantCommandResponse(
          const cyxwiz::plugin::AssistantCommandRequest &)>
          handler);
  void SetProjectRoot(std::string project_root);
  void CloseProject(std::string_view project_root);
  bool ActivatePythonRepl();
  void AppendScriptOutput(const std::string &source, const std::string &text,
                          bool is_error = false) override;
  void AddLog(const std::string &message, LogLevel level = LogLevel::Info);
  void AddInfo(const std::string &message);
  void AddWarning(const std::string &message);
  void AddError(const std::string &message);
  void AddSuccess(const std::string &message);
  void Clear();

  // Visibility control for sidebar integration
  bool *GetVisiblePtr() { return &show_window_; }

private:
  struct RuntimeLogExportTaskState;

  struct SavedInspectorFilter {
    std::string name;
    std::string expression;
    std::string validation_error;
  };

  void RenderInspectorTab(bool request_focus);
  void RenderActiveSession(bool request_focus);
  void RenderLocalShellSession(bool request_focus);
  void PruneLocalShellSessions();
  void RenderLogsSession(bool request_focus);
  void RenderCommandsSession(bool request_focus);
  void RenderInspectorFilters(const cyxwiz::RuntimeLogInspectorResult *result,
                              bool request_focus);
  void RenderInspectorTable(const cyxwiz::RuntimeLogInspectorResult *result);
  void RenderInspectorDetails(const cyxwiz::RuntimeLogInspectorResult *result);
  void RenderLogsToolbar();
  void RenderCommandsToolbar();
  void RenderCommandInput(bool request_focus);
  void RenderCopyStatus();
  void RenderRuntimeLogExportDialog();
  void RenderRuntimeLogExportStatus();
  void RenderAllTab();
  void ExecCommand(const char *command);
  void AppendCommandResult(const cyxwiz::RuntimeConsoleCommandResult &result);
  static int InputTextCallback(ImGuiInputTextCallbackData *data);
  int HandleInputTextCallback(ImGuiInputTextCallbackData *data);
  void ExecutePipCommand(const std::vector<std::string> &pip_arguments);
  void CopyCommandTranscript();
  void CopySelectedCommand();
  void CopyFilteredRuntimeLogs();
  void CopySelectedRuntimeLog();
  void OpenRuntimeLogExportDialog();
  void QueueRuntimeLogExport(const std::filesystem::path &destination);
  const char *GetLevelPrefix(LogLevel level) const;
  ImVec4 GetLevelColor(LogLevel level) const;
  std::vector<LogEntry> SnapshotEntries() const;
  void StartInspectorWorker();
  void StopInspectorWorker();
  void RequestInspectorQuery(bool force = false);
  std::shared_ptr<const cyxwiz::RuntimeLogInspectorResult>
  SnapshotInspectorResult() const;
  void LoadSavedInspectorFilters();
  bool PersistSavedInspectorFilters();
  void ClearLogView();
  void ClearCommandTranscript();

  std::deque<LogEntry> items_;
  ConsoleWorkbench workbench_;
  std::unique_ptr<cyxwiz::AgentLlmSession> agent_llm_;
  std::unique_ptr<cyxwiz::PythonReplSession> python_repl_;
  std::unordered_map<std::uint64_t, std::unique_ptr<cyxwiz::LocalShellSession>>
      local_shell_sessions_;
  uint64_t next_command_sequence_ = 0;
  uint64_t selected_command_sequence_ = 0;
  bool command_input_focus_pending_ = false;
  bool log_search_focus_pending_ = false;
  char input_buf_[256];
  std::atomic<bool> scroll_to_bottom_;
  bool show_window_;
  bool auto_scroll_;
  cyxwiz::RuntimeLogInspectorCriteria inspector_criteria_;
  char inspector_text_[128]{};
  char inspector_filter_[1024]{};
  char inspector_filter_name_[64]{};
  std::vector<SavedInspectorFilter> inspector_saved_filters_;
  int inspector_selected_saved_filter_ = -1;
  std::string inspector_filter_status_;
  bool inspector_filter_status_error_ = false;
  bool inspector_paused_ = false;
  uint64_t inspector_frozen_sequence_ = 0;
  uint64_t inspector_after_sequence_ = 0;
  uint64_t inspector_selected_sequence_ = 0;
  uint64_t inspector_last_requested_high_water_ = 0;
  uint64_t inspector_last_requested_after_sequence_ = 0;
  uint64_t inspector_last_rendered_high_water_ = 0;
  cyxwiz::RuntimeLogInspectorCriteria inspector_last_requested_criteria_;
  bool inspector_has_submitted_request_ = false;
  double inspector_next_refresh_time_ = 0.0;
  bool inspector_export_popup_requested_ = false;
  bool inspector_export_selected_scope_ = false;
  uint64_t inspector_export_selected_sequence_ = 0;
  uint64_t inspector_export_after_sequence_ = 0;
  cyxwiz::RuntimeLogExportFormat inspector_export_format_ =
      cyxwiz::RuntimeLogExportFormat::JsonLines;
  cyxwiz::RuntimeLogRedactionOptions inspector_export_redaction_;
  std::shared_ptr<const cyxwiz::RuntimeLogInspectorResult>
      inspector_export_result_;
  std::shared_ptr<RuntimeLogExportTaskState> inspector_export_task_state_;

  mutable std::mutex inspector_mutex_;
  std::condition_variable inspector_cv_;
  std::thread inspector_worker_;
  bool inspector_stop_ = false;
  bool inspector_request_pending_ = false;
  uint64_t inspector_request_generation_ = 0;
  std::atomic<uint64_t> inspector_query_requests_{0};
  std::atomic<uint64_t> inspector_query_executions_{0};
  std::atomic<uint64_t> inspector_query_coalesced_{0};
  std::atomic<uint64_t> inspector_query_stale_{0};
  cyxwiz::RuntimeLogInspectorRequest inspector_pending_request_;
  std::shared_ptr<const cyxwiz::RuntimeLogInspectorResult> inspector_result_;
  std::unique_ptr<cyxwiz::RuntimeTruthQueryProvider> truth_provider_;
  std::unique_ptr<cyxwiz::RuntimeConsoleCommandService> command_service_;

  mutable std::mutex log_mutex_; // Thread-safe logging

  // Copy notification state
  bool show_copy_notification_;
  float copy_notification_time_;
  void ShowCopyNotification();
};

} // namespace gui
