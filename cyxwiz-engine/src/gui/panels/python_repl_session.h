#pragma once

#include "../../core/python_repl_presentation.h"
#include "../../scripting/script_output_sink.h"
#include "../../scripting/scripting_engine.h"
#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

struct ImGuiInputTextCallbackData;

namespace cyxwiz {

/**
 * Embedded Python REPL session for the unified Console workbench.
 *
 * VS Code-style layout (tofix121): a status header (interpreter, state,
 * Restart / Copy all / Clear, Interrupt while running), a monospace
 * transcript of entries (code with >>> / ... prompts and syntax colours,
 * output, warnings, tracebacks, script runs), and a multi-line input pinned
 * to the bottom. Wording and classification live in
 * core/python_repl_presentation.
 */
class PythonReplSession : public scripting::IScriptOutputSink {
public:
  PythonReplSession();
  ~PythonReplSession() override;

  void RenderContent();
  void RequestInputFocus() { focus_input_ = true; }

  void SetScriptingEngine(std::shared_ptr<scripting::ScriptingEngine> engine);
  void ResetProjectState();

  void AppendScriptOutput(const std::string &source, const std::string &text,
                          bool is_error = false) override;
  void EndScriptOutput(const std::string &source, bool success, bool cancelled,
                       double seconds) override;

private:
  struct Entry {
    enum class Kind { System, Command, Script };
    Kind kind = Kind::System;
    std::string code;       // Command: the submitted source
    std::vector<std::vector<repl::Token>> code_lines; // highlighted code
    std::string output;     // stdout and printed results
    std::string warnings;   // stderr on success
    std::string error;      // one-line error (exception or engine message)
    std::string traceback;  // formatted traceback (errors)
    std::string hint;       // e.g. "pip install pandas"
    std::string note;       // muted line, e.g. where a paused debug run evaluated it
    std::string source;     // Script: display name
    bool running = false;   // still executing / producing output
    bool failed = false;
    bool cancelled = false;
    double seconds = 0.0;
  };

  // Rendering
  void RenderHeader();
  void RenderTranscript();
  void RenderEntry(int index, const Entry &entry);
  void RenderInput();
  void RenderCompletionPopup();
  void HandleShortcuts();

  // Actions
  void Submit();
  void ExecuteCommand(const std::string &command);
  void ClearTranscript();
  void CopyAll();
  void ResetSession();
  void AddSystem(const std::string &text);
  static std::string EntryText(const Entry &entry);
  Entry &AddEntry(Entry entry);
  void SetInputText(const std::string &text);
  repl::StatusView BuildStatus() const;

  // History
  void AddToHistory(const std::string &command);
  bool RecallHistory(int direction, const std::string &current,
                     std::string &out);

  // Input callback
  static int InputTextCallback(ImGuiInputTextCallbackData *data);
  int HandleInputTextCallback(ImGuiInputTextCallbackData *data);
  void ApplyCompletion(ImGuiInputTextCallbackData *data,
                       const std::string &completion);

  // Async execution
  void StartAsyncCommand(const std::string &command);
  void CheckAsyncCompletion();
  void StopAsyncCommand();
  void RefreshInterpreterInfo(bool force);

  std::shared_ptr<scripting::ScriptingEngine> scripting_engine_;
  std::vector<Entry> entries_;
  std::vector<std::string> command_history_;
  int history_position_ = -1;
  std::string history_draft_;

  // Input
  std::vector<char> input_buffer_;
  bool focus_input_ = true;
  bool pending_enter_ = false;       // plain Enter pressed this frame
  bool pending_shift_enter_ = false; // Shift+Enter pressed this frame
  bool pending_history_up_ = false;
  bool pending_history_down_ = false;
  bool swallow_enter_ = false;       // remove the newline Enter inserted
  bool restore_cursor_ = false;      // undo caret move from list navigation
  bool submit_requested_ = false;
  bool has_submission_ = false;
  std::string submitted_text_;
  int pending_cursor_ = -1;          // cursor to set on the next callback
  int cursor_pos_ = 0;
  float input_extra_height_ = 0.0f;  // added by the drag handle
  float input_height_ = 0.0f;        // this frame's input height
  int hint_rows_ = 1;                // rows the hints wrapped to
  bool input_active_ = false;
  unsigned int input_id_ = 0;        // ImGuiID of the input widget

  // Completion
  std::vector<std::string> completion_all_;   // matches for the Tab word
  std::vector<std::string> completion_items_; // filtered by the current word
  int completion_selected_ = 0;
  bool completion_open_ = false;
  bool completion_accept_ = false;
  bool completion_scroll_ = false;
  std::string completion_word_;
  float completion_anchor_x_ = 0.0f;
  float completion_anchor_y_ = 0.0f;

  // Transcript
  int selected_entry_ = -1;
  bool auto_scroll_ = true;
  bool scroll_to_bottom_ = false;
  float notice_timer_ = 0.0f;
  std::string notice_;  // short feedback in the footer ("Copied", errors)

  // Running command
  std::atomic<bool> command_executing_{false};
  // While a debug run is paused, input runs in its innermost frame (P6).
  bool debug_console_ = false;
  std::chrono::steady_clock::time_point command_started_{};
  int running_entry_ = -1;

  // Input colouring cache
  std::string highlight_text_;
  std::vector<std::vector<repl::Token>> highlight_lines_;

  repl::StatusView status_{};

  // Interpreter information shown in the header
  scripting::ScriptingEngine::InterpreterInfo interpreter_{};
  std::chrono::steady_clock::time_point interpreter_refreshed_{};
  bool bundled_runtime_ = false;
  std::string last_init_error_;
};

} // namespace cyxwiz
