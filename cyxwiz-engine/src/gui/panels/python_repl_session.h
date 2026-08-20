#pragma once

#include "../../scripting/script_output_sink.h"
#include "../../scripting/scripting_engine.h"
#include <atomic>
#include <memory>
#include <string>
#include <vector>

struct ImGuiInputTextCallbackData;

namespace cyxwiz {

/**
 * Embedded Python REPL session for the unified Console workbench.
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

private:
  // Output entry (command or result)
  struct OutputEntry {
    enum class Type {
      Command, // User input (fx:>> ...)
      Result,  // Execution result
      Error    // Error message
    };

    Type type;
    std::string text;
  };

  // Rendering functions
  void RenderOutputArea();
  void RenderInputArea();
  std::string BuildOutputTranscript() const;

  // Command execution
  void ExecuteCommand(const std::string &command);
  void ClearOutput();

  // Command history navigation
  void NavigateHistory(int direction); // -1 = up, +1 = down
  void AddToHistory(const std::string &command);

  // Auto-completion
  void GetCompletions(const std::string &partial,
                      std::vector<std::string> &suggestions);
  void ApplyCompletion(const std::string &completion);
  void RenderCompletionPopup();
  static int InputTextCallback(ImGuiInputTextCallbackData *data);
  int HandleInputTextCallback(ImGuiInputTextCallbackData *data);

  // Data
  std::shared_ptr<scripting::ScriptingEngine> scripting_engine_;
  std::vector<OutputEntry> output_;
  std::vector<std::string> command_history_;
  int history_position_;

  // Auto-completion state
  std::vector<std::string> completion_suggestions_;
  int completion_selected_;
  bool show_completion_popup_;
  std::string completion_prefix_;

  // UI state
  char input_buffer_[4096]; // Larger buffer for multi-line input
  bool scroll_to_bottom_;
  bool focus_input_;
  int selected_output_index_ = -1;

  // Async command execution
  void StartAsyncCommand(const std::string &command);
  void CheckAsyncCompletion();
  void StopAsyncCommand();

  std::atomic<bool> command_executing_{false};
  std::string executing_command_; // Command being executed (for display)
};

} // namespace cyxwiz
