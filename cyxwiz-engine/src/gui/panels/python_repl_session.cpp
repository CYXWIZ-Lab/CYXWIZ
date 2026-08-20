#include "python_repl_session.h"
#include "../../scripting/scripting_engine.h"
#include <algorithm>
#include <cstring>
#include <imgui.h>
#include <sstream>

namespace cyxwiz {

namespace {

std::string Trim(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return "";
  }
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

} // namespace

PythonReplSession::PythonReplSession()
    : history_position_(-1), completion_selected_(0),
      show_completion_popup_(false), scroll_to_bottom_(false),
      focus_input_(true) {
  std::memset(input_buffer_, 0, sizeof(input_buffer_));

  // Welcome message
  OutputEntry welcome;
  welcome.type = OutputEntry::Type::Result;
  welcome.text =
      "CyxWiz Python REPL\n"
      "Type 'help()' for help, 'clear' to clear output\n"
      "Enter: execute | Shift/Ctrl+Enter: new line | Tab: autocomplete\n";
  output_.push_back(welcome);
}

PythonReplSession::~PythonReplSession() {
  if (command_executing_) {
    StopAsyncCommand();
  }
}

void PythonReplSession::SetScriptingEngine(
    std::shared_ptr<scripting::ScriptingEngine> engine) {
  scripting_engine_ = engine;
}

void PythonReplSession::ResetProjectState() {
  if (command_executing_)
    StopAsyncCommand();
  command_executing_ = false;
  executing_command_.clear();
  output_.clear();
  command_history_.clear();
  completion_suggestions_.clear();
  history_position_ = -1;
  completion_selected_ = 0;
  show_completion_popup_ = false;
  selected_output_index_ = -1;
  std::memset(input_buffer_, 0, sizeof(input_buffer_));
  scroll_to_bottom_ = true;
  focus_input_ = true;

  OutputEntry ready;
  ready.type = OutputEntry::Type::Result;
  ready.text = "Python REPL project state reset.\n";
  output_.push_back(std::move(ready));
}

void PythonReplSession::RenderContent() {
  CheckAsyncCompletion();

  // Output area (scrollable)
  RenderOutputArea();

  ImGui::Separator();

  // Show "Running..." indicator and Stop button if Python command is executing
  if (command_executing_) {
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "Running...");
    ImGui::SameLine();
    if (ImGui::Button("Stop")) {
      StopAsyncCommand();
    }
  } else {
    RenderInputArea();
  }

  // Render completion popup if active
  RenderCompletionPopup();
}

void PythonReplSession::RenderOutputArea() {
  // Child window for scrollable output
  // Reserve space for: separator + prompt + 3-line input + hint line
  float line_height = ImGui::GetTextLineHeight();
  const float footer_height =
      ImGui::GetStyle().ItemSpacing.y * 3 + line_height // prompt line
      + 3.0f * line_height +
      ImGui::GetStyle().FramePadding.y * 2 // multiline input (min 3 lines)
      + line_height;                       // hint line
  const float available_height = ImGui::GetContentRegionAvail().y;
  const float min_output_height = std::max(48.0f, line_height * 3.0f);
  const float output_height =
      std::max(min_output_height, available_height - footer_height);
  ImGui::BeginChild("OutputRegion", ImVec2(0.0f, output_height), false,
                    ImGuiWindowFlags_HorizontalScrollbar);

  if (ImGui::Button("Copy output")) {
    const std::string transcript = BuildOutputTranscript();
    ImGui::SetClipboardText(transcript.c_str());
  }
  ImGui::SameLine();
  if (ImGui::Button("Copy selected") && selected_output_index_ >= 0 &&
      selected_output_index_ < static_cast<int>(output_.size())) {
    ImGui::SetClipboardText(output_[selected_output_index_].text.c_str());
  }
  ImGui::SameLine();
  if (ImGui::Button("Clear")) {
    ClearOutput();
    ImGui::EndChild();
    return;
  }
  ImGui::Separator();

  // Render each output entry. ImGui selectable labels must be single-line in
  // Debug builds, so multi-line command output is split into selectable rows
  // that still map back to the original output entry for copy/select actions.
  for (int i = 0; i < static_cast<int>(output_.size()); ++i) {
    const auto &entry = output_[i];
    ImGui::PushID(i);
    const bool selected = (selected_output_index_ == i);

    const bool command_color = entry.type == OutputEntry::Type::Command;
    const bool error_color = entry.type == OutputEntry::Type::Error;
    if (command_color) {
      ImGui::PushStyleColor(ImGuiCol_Text,
                            ImVec4(0.5f, 1.0f, 0.5f, 1.0f)); // Green
    } else if (error_color) {
      ImGui::PushStyleColor(ImGuiCol_Text,
                            ImVec4(1.0f, 0.4f, 0.4f, 1.0f)); // Red
    }

    std::istringstream line_stream(entry.text);
    std::string line;
    int line_index = 0;
    bool rendered_line = false;
    while (std::getline(line_stream, line)) {
      if (!line.empty() && line.back() == '\r') {
        line.pop_back();
      }

      ImGui::PushID(line_index++);
      const char *label = line.empty() ? " " : line.c_str();
      if (ImGui::Selectable(label, selected,
                            ImGuiSelectableFlags_AllowDoubleClick)) {
        selected_output_index_ = i;
      }
      if (ImGui::BeginPopupContextItem()) {
        if (ImGui::MenuItem("Copy entry")) {
          ImGui::SetClipboardText(entry.text.c_str());
        }
        if (ImGui::MenuItem("Select entry")) {
          selected_output_index_ = i;
        }
        ImGui::EndPopup();
      }
      ImGui::PopID();
      rendered_line = true;
    }
    if (!rendered_line) {
      if (ImGui::Selectable(" ", selected,
                            ImGuiSelectableFlags_AllowDoubleClick)) {
        selected_output_index_ = i;
      }
    }

    if (command_color || error_color) {
      ImGui::PopStyleColor();
    }
    ImGui::PopID();
  }

  // Auto-scroll to bottom when new output is added
  if (scroll_to_bottom_) {
    ImGui::SetScrollHereY(1.0f);
    scroll_to_bottom_ = false;
  }

  ImGui::EndChild();
}

void PythonReplSession::RenderInputArea() {
  // Prompt label (italic-style light blue)
  ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.7f, 0.9f, 1.0f, 1.0f));
  ImGui::TextUnformatted("fx:>>");
  ImGui::PopStyleColor();
  ImGui::SameLine();

  // Stable single-line REPL input. The previous multiline input used history,
  // completion, callback-always, EnterReturnsTrue, and CtrlEnterForNewLine
  // together; Debug ImGui/CRT can abort on that combination. Keep the command
  // window reliable first, then layer richer editing back in deliberately.
  const ImGuiInputTextFlags flags = ImGuiInputTextFlags_EnterReturnsTrue |
                                    ImGuiInputTextFlags_CallbackHistory |
                                    ImGuiInputTextFlags_CallbackCompletion;

  const bool focus_requested = focus_input_;
  if (focus_requested) {
    ImGui::SetKeyboardFocusHere();
  }

  const bool enter_pressed =
      ImGui::InputText("##input", input_buffer_, sizeof(input_buffer_), flags,
                       &PythonReplSession::InputTextCallback, this);
  if (focus_requested && ImGui::IsItemActive()) {
    focus_input_ = false;
  }

  if (enter_pressed) {
    std::string command(input_buffer_);

    while (!command.empty() &&
           (command.back() == '\n' || command.back() == '\r' ||
            command.back() == ' ')) {
      command.pop_back();
    }

    if (!command.empty()) {
      ExecuteCommand(command);
      std::memset(input_buffer_, 0, sizeof(input_buffer_));
      focus_input_ = true;
    }
    show_completion_popup_ = false;
  }

  ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
  ImGui::TextUnformatted("Enter: execute");
  ImGui::PopStyleColor();
}

void PythonReplSession::ExecuteCommand(const std::string &command) {
  const std::string trimmed_command = Trim(command);

  if (trimmed_command.empty()) {
    return;
  }

  // Add command to output
  OutputEntry cmd_entry;
  cmd_entry.type = OutputEntry::Type::Command;
  cmd_entry.text = "fx:>> " + trimmed_command;
  output_.push_back(cmd_entry);

  // Add to history
  AddToHistory(trimmed_command);

  // Handle special commands
  if (trimmed_command == "clear") {
    ClearOutput();
    scroll_to_bottom_ = true;
    return;
  }

  if (trimmed_command == "help" || trimmed_command == "help()") {
    OutputEntry help;
    help.type = OutputEntry::Type::Result;
    help.text = R"(CyxWiz Python REPL Help
=======================

COMMANDS:
  clear       - Clear output window
  help()      - Show this help message

DUCKDB (SQL Analytics):
  sql(query)       - Run SQL query on in-memory database
  read_csv(path)   - Load CSV file
  read_parquet(p)  - Load Parquet file
  read_json(path)  - Load JSON file
  db               - DuckDB connection object

  Examples:
    sql("SELECT 1 + 1 AS result")
    sql("SELECT * FROM 'data.csv' LIMIT 10")
    read_csv('data.csv').filter('age > 30')

POLARS (Fast DataFrames):
  pl               - Polars module
  df(data)         - Create DataFrame
  col('name')      - Column expression
  pl_csv(path)     - Read CSV file
  pl_parquet(p)    - Read Parquet file
  scan_csv(path)   - Lazy CSV reader
  scan_parquet(p)  - Lazy Parquet reader

  Examples:
    data = df({'a': [1, 2, 3], 'b': [4, 5, 6]})
    data.filter(col('a') > 1)
    pl_csv('data.csv').head(10)

MATLAB-STYLE FUNCTIONS:
  Linear Algebra:  eye, zeros, ones, svd, eig, qr, chol, lu, det,
                   rank, trace, norm, cond, inv, transpose, solve
  Signal:          fft, ifft, conv, spectrogram, lowpass, highpass
  Statistics:      kmeans, dbscan, gmm, pca, tsne
  Time Series:     acf, pacf, decompose, stationarity, arima

  Examples:
    I = eye(3)              # 3x3 identity matrix
    pm(I)                   # Print matrix nicely
    U, S, V = svd([[1,2],[3,4]])

Type any Python code to execute.
)";
    output_.push_back(help);
    scroll_to_bottom_ = true;
    return;
  }

  if (trimmed_command.front() == '/') {
    OutputEntry error;
    error.type = OutputEntry::Type::Error;
    error.text = "Slash commands are not Python. Use the Agent LLM Console "
                 "session for assistant requests.";
    output_.push_back(error);
    scroll_to_bottom_ = true;
    return;
  }

  // Execute Python command asynchronously
  if (scripting_engine_) {
    StartAsyncCommand(trimmed_command);
  } else {
    OutputEntry error;
    error.type = OutputEntry::Type::Error;
    error.text = "Error: Scripting engine not initialized";
    output_.push_back(error);
  }

  scroll_to_bottom_ = true;
}

void PythonReplSession::ClearOutput() {
  output_.clear();
  selected_output_index_ = -1;

  // Re-add welcome message
  OutputEntry welcome;
  welcome.type = OutputEntry::Type::Result;
  welcome.text = "Output cleared.\n";
  output_.push_back(welcome);
}

std::string PythonReplSession::BuildOutputTranscript() const {
  std::string text;
  for (const auto &entry : output_) {
    text += entry.text;
    text += '\n';
  }
  return text;
}

void PythonReplSession::AppendScriptOutput(const std::string &source,
                                           const std::string &text,
                                           bool is_error) {
  // Add script name as command entry
  OutputEntry script_entry;
  script_entry.type = OutputEntry::Type::Command;
  script_entry.text = "Running script: " + source;
  output_.push_back(script_entry);

  // Add output or error
  if (!text.empty()) {
    OutputEntry result_entry;
    result_entry.type =
        is_error ? OutputEntry::Type::Error : OutputEntry::Type::Result;
    result_entry.text = text;
    output_.push_back(result_entry);
  }

  scroll_to_bottom_ = true;
}

void PythonReplSession::NavigateHistory(int direction) {
  if (command_history_.empty())
    return;

  if (direction < 0) {
    // Up arrow - go back in history
    if (history_position_ < static_cast<int>(command_history_.size()) - 1) {
      history_position_++;
    }
  } else {
    // Down arrow - go forward in history
    if (history_position_ > -1) {
      history_position_--;
    }
  }

  if (history_position_ >= 0 &&
      history_position_ < static_cast<int>(command_history_.size())) {
    // Copy command from history to input buffer
    const std::string &cmd =
        command_history_[command_history_.size() - 1 - history_position_];
    std::strncpy(input_buffer_, cmd.c_str(), sizeof(input_buffer_) - 1);
    input_buffer_[sizeof(input_buffer_) - 1] = '\0';
  } else {
    // Clear input if at the end of history
    std::memset(input_buffer_, 0, sizeof(input_buffer_));
  }
}

void PythonReplSession::AddToHistory(const std::string &command) {
  // Don't add empty commands or duplicates of the last command
  if (command.empty())
    return;
  if (!command_history_.empty() && command_history_.back() == command)
    return;

  command_history_.push_back(command);

  // Limit history size to 100 entries
  if (command_history_.size() > 100) {
    command_history_.erase(command_history_.begin());
  }

  history_position_ = -1; // Reset history navigation
}

void PythonReplSession::GetCompletions(const std::string &partial,
                                       std::vector<std::string> &suggestions) {
  if (!scripting_engine_ || scripting_engine_->IsCommandRunning() ||
      scripting_engine_->IsScriptRunning()) {
    return;
  }

  // Extract the last word/identifier from the partial input
  std::string word = partial;
  size_t last_space = partial.find_last_of(" \t\n()[]{}+-*/=,<>!&|");
  if (last_space != std::string::npos) {
    word = partial.substr(last_space + 1);
  }

  if (word.empty())
    return;

  // Check if it's a dotted attribute access (e.g., "math.sq")
  size_t dot_pos = word.find_last_of('.');
  if (dot_pos != std::string::npos) {
    // Attribute completion: "module.attr"
    std::string module = word.substr(0, dot_pos);
    std::string attr_prefix = word.substr(dot_pos + 1);

    // Use Python dir() to get module attributes
    std::string introspect_code = "try:\n"
                                  "    import builtins\n"
                                  "    obj = " +
                                  module +
                                  "\n"
                                  "    attrs = [a for a in dir(obj) if not "
                                  "a.startswith('_') and a.startswith('" +
                                  attr_prefix +
                                  "')]\n"
                                  "    print('\\n'.join(attrs))\n"
                                  "except: pass\n";

    auto result = scripting_engine_->ExecuteCommand(introspect_code);
    if (result.success && !result.output.empty()) {
      // Parse newline-separated attributes
      std::string attr;
      std::istringstream stream(result.output);
      while (std::getline(stream, attr)) {
        if (!attr.empty()) {
          suggestions.push_back(module + "." + attr);
        }
      }
    }
  } else {
    // Simple identifier completion
    // Get globals/builtins that match the prefix
    std::string introspect_code =
        "try:\n"
        "    import builtins\n"
        "    matches = []\n"
        "    # Check globals\n"
        "    for name in dir():\n"
        "        if not name.startswith('_') and name.startswith('" +
        word +
        "'):\n"
        "            matches.append(name)\n"
        "    # Check builtins\n"
        "    for name in dir(builtins):\n"
        "        if not name.startswith('_') and name.startswith('" +
        word +
        "'):\n"
        "            matches.append(name)\n"
        "    # Remove duplicates and sort\n"
        "    matches = sorted(set(matches))\n"
        "    print('\\n'.join(matches))\n"
        "except: pass\n";

    auto result = scripting_engine_->ExecuteCommand(introspect_code);
    if (result.success && !result.output.empty()) {
      // Parse newline-separated identifiers
      std::string identifier;
      std::istringstream stream(result.output);
      while (std::getline(stream, identifier)) {
        if (!identifier.empty()) {
          suggestions.push_back(identifier);
        }
      }
    }

    // Add common CyxWiz-specific completions
    std::vector<std::string> cyxwiz_keywords = {"import", "pycyxwiz", "math",
                                                "random", "json",     "numpy",
                                                "help",   "clear"};

    for (const auto &kw : cyxwiz_keywords) {
      if (kw.find(word) == 0) { // Starts with word
        // Check if not already in suggestions
        if (std::find(suggestions.begin(), suggestions.end(), kw) ==
            suggestions.end()) {
          suggestions.push_back(kw);
        }
      }
    }

    // Sort suggestions alphabetically
    std::sort(suggestions.begin(), suggestions.end());
  }

  // Limit to 20 suggestions
  if (suggestions.size() > 20) {
    suggestions.resize(20);
  }
}

void PythonReplSession::ApplyCompletion(const std::string &completion) {
  // Replace the last word in input buffer with the completion
  std::string current(input_buffer_);
  size_t last_space = current.find_last_of(" \t\n()[]{}+-*/=,<>!&|");

  if (last_space != std::string::npos) {
    // Replace the word after the last separator
    std::string prefix = current.substr(0, last_space + 1);
    std::string result = prefix + completion;
    std::strncpy(input_buffer_, result.c_str(), sizeof(input_buffer_) - 1);
    input_buffer_[sizeof(input_buffer_) - 1] = '\0';
  } else {
    // Replace entire buffer
    std::strncpy(input_buffer_, completion.c_str(), sizeof(input_buffer_) - 1);
    input_buffer_[sizeof(input_buffer_) - 1] = '\0';
  }
}

void PythonReplSession::RenderCompletionPopup() {
  if (!show_completion_popup_ || completion_suggestions_.empty())
    return;

  // Calculate popup position (below the input field)
  ImVec2 input_pos = ImGui::GetItemRectMin();
  ImVec2 input_size = ImGui::GetItemRectSize();
  ImVec2 popup_pos(input_pos.x, input_pos.y + input_size.y);

  ImGui::SetNextWindowPos(popup_pos);
  ImGui::SetNextWindowSize(ImVec2(400, 0)); // Auto height

  if (ImGui::Begin("##CompletionPopup", nullptr,
                   ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                       ImGuiWindowFlags_NoMove |
                       ImGuiWindowFlags_AlwaysAutoResize |
                       ImGuiWindowFlags_NoFocusOnAppearing)) {

    ImGui::Text("Suggestions (use Tab to apply, Up/Down to navigate):");
    ImGui::Separator();

    // Render each suggestion
    for (int i = 0; i < static_cast<int>(completion_suggestions_.size()); i++) {
      bool is_selected = (i == completion_selected_);

      if (is_selected) {
        ImGui::PushStyleColor(
            ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f)); // Yellow highlight
      }

      ImGui::Text("  %s", completion_suggestions_[i].c_str());

      if (is_selected) {
        ImGui::PopStyleColor();
      }
    }
  }
  ImGui::End();
}

// ========== Async Command Execution ==========

void PythonReplSession::StartAsyncCommand(const std::string &command) {
  if (command_executing_ || !scripting_engine_) {
    return;
  }

  executing_command_ = command;
  command_executing_ = true;
  // Use the scripting engine's async command execution
  scripting_engine_->ExecuteCommandAsync(command);
}

int PythonReplSession::InputTextCallback(ImGuiInputTextCallbackData *data) {
  return static_cast<PythonReplSession *>(data->UserData)
      ->HandleInputTextCallback(data);
}

int PythonReplSession::HandleInputTextCallback(
    ImGuiInputTextCallbackData *data) {
  if (data->EventFlag == ImGuiInputTextFlags_CallbackHistory) {
    NavigateHistory(data->EventKey == ImGuiKey_UpArrow ? -1 : 1);
    data->DeleteChars(0, data->BufTextLen);
    data->InsertChars(0, input_buffer_);
    return 0;
  }

  if (data->EventFlag == ImGuiInputTextFlags_CallbackCompletion) {
    completion_suggestions_.clear();
    GetCompletions(data->Buf, completion_suggestions_);
    completion_selected_ = 0;
    show_completion_popup_ = !completion_suggestions_.empty();
    if (!completion_suggestions_.empty()) {
      std::strncpy(input_buffer_, data->Buf, sizeof(input_buffer_) - 1);
      input_buffer_[sizeof(input_buffer_) - 1] = '\0';
      ApplyCompletion(completion_suggestions_.front());
      data->DeleteChars(0, data->BufTextLen);
      data->InsertChars(0, input_buffer_);
    }
  }
  return 0;
}

void PythonReplSession::CheckAsyncCompletion() {
  if (!command_executing_ || !scripting_engine_) {
    return;
  }

  // Check if command has finished
  if (!scripting_engine_->IsCommandRunning()) {
    // Get the result
    auto result_opt = scripting_engine_->GetCommandResult();
    if (result_opt) {
      auto &result = *result_opt;

      OutputEntry result_entry;
      if (result.success) {
        result_entry.type = OutputEntry::Type::Result;
        result_entry.text = result.output.empty() ? "" : result.output;
      } else {
        result_entry.type = OutputEntry::Type::Error;
        if (result.timeout_exceeded) {
          result_entry.text = "Command interrupted (timeout)";
        } else if (result.was_cancelled) {
          result_entry.text = "Command cancelled";
        } else {
          result_entry.text = "Error: " + result.error_message;
        }
      }

      if (!result_entry.text.empty()) {
        output_.push_back(result_entry);
      }

      scroll_to_bottom_ = true;
    }

    command_executing_ = false;
    executing_command_.clear();
    focus_input_ = true;
  }
}

void PythonReplSession::StopAsyncCommand() {
  if (!command_executing_ || !scripting_engine_) {
    return;
  }

  scripting_engine_->StopCommand();
}

} // namespace cyxwiz
