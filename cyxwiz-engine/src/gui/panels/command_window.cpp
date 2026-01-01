#include "command_window.h"
#include "../../scripting/scripting_engine.h"
#include <imgui.h>
#include <cstring>
#include <sstream>
#include <algorithm>

namespace cyxwiz {

CommandWindowPanel::CommandWindowPanel()
    : Panel("Command Window", true)
    , history_position_(-1)
    , completion_selected_(0)
    , show_completion_popup_(false)
    , scroll_to_bottom_(false)
    , focus_input_(true)
{
    std::memset(input_buffer_, 0, sizeof(input_buffer_));

    // Welcome message
    OutputEntry welcome;
    welcome.type = OutputEntry::Type::Result;
    welcome.text = "CyxWiz Python Command Window\n"
                   "Type 'help()' for help, 'clear' to clear output\n"
                   "Enter: execute | Shift/Ctrl+Enter: new line | Tab: autocomplete\n";
    output_.push_back(welcome);
}

CommandWindowPanel::~CommandWindowPanel() {
    // Stop any running command
    if (command_executing_) {
        StopAsyncCommand();
    }
    // Wait for thread to finish
    if (command_thread_ && command_thread_->joinable()) {
        command_thread_->join();
    }
}

void CommandWindowPanel::SetScriptingEngine(std::shared_ptr<scripting::ScriptingEngine> engine) {
    scripting_engine_ = engine;
}

void CommandWindowPanel::Render() {
    if (!visible_) return;

    // Check for async command completion
    CheckAsyncCompletion();

    ImGui::Begin(GetName(), &visible_);

    // Output area (scrollable)
    RenderOutputArea();

    ImGui::Separator();

    // Show "Running..." indicator and Stop button if command is executing
    if (command_executing_) {
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "Running...");
        ImGui::SameLine();
        if (ImGui::Button("Stop")) {
            StopAsyncCommand();
        }
    } else {
        // Input area (bottom) - only show when not executing
        RenderInputArea();
    }

    // Render completion popup if active
    RenderCompletionPopup();

    ImGui::End();
}

void CommandWindowPanel::RenderOutputArea() {
    // Child window for scrollable output
    // Reserve space for: separator + prompt + 3-line input + hint line
    float line_height = ImGui::GetTextLineHeight();
    const float footer_height = ImGui::GetStyle().ItemSpacing.y * 3
                              + line_height  // prompt line
                              + 3.0f * line_height + ImGui::GetStyle().FramePadding.y * 2  // multiline input (min 3 lines)
                              + line_height; // hint line
    ImGui::BeginChild("OutputRegion", ImVec2(0, -footer_height), false, ImGuiWindowFlags_HorizontalScrollbar);

    // Render each output entry
    for (const auto& entry : output_) {
        switch (entry.type) {
            case OutputEntry::Type::Command:
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 1.0f, 0.5f, 1.0f)); // Green
                ImGui::TextUnformatted(entry.text.c_str());
                ImGui::PopStyleColor();
                break;

            case OutputEntry::Type::Result:
                ImGui::TextUnformatted(entry.text.c_str());
                break;

            case OutputEntry::Type::Error:
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f)); // Red
                ImGui::TextUnformatted(entry.text.c_str());
                ImGui::PopStyleColor();
                break;
        }
    }

    // Auto-scroll to bottom when new output is added
    if (scroll_to_bottom_) {
        ImGui::SetScrollHereY(1.0f);
        scroll_to_bottom_ = false;
    }

    ImGui::EndChild();
}

void CommandWindowPanel::RenderInputArea() {
    // Prompt label (italic-style light blue)
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.7f, 0.9f, 1.0f, 1.0f));
    ImGui::TextUnformatted("fx:>>");
    ImGui::PopStyleColor();
    ImGui::SameLine();

    // Multi-line input field
    // CtrlEnterForNewLine: Enter submits, Ctrl+Enter inserts newline
    // EnterReturnsTrue: Return true when Enter is pressed
    // We also handle Shift+Enter to insert newline via callback
    ImGuiInputTextFlags flags = ImGuiInputTextFlags_CallbackHistory
                              | ImGuiInputTextFlags_CallbackCompletion
                              | ImGuiInputTextFlags_CallbackAlways
                              | ImGuiInputTextFlags_EnterReturnsTrue
                              | ImGuiInputTextFlags_CtrlEnterForNewLine;

    // Track if we should insert newline (Shift+Enter)
    static bool insert_newline = false;

    // Handle history navigation, auto-completion, and Shift+Enter for newline
    auto callback = [](ImGuiInputTextCallbackData* data) -> int {
        CommandWindowPanel* panel = (CommandWindowPanel*)data->UserData;

        if (data->EventFlag == ImGuiInputTextFlags_CallbackAlways) {
            // Handle Shift+Enter to insert newline
            ImGuiIO& io = ImGui::GetIO();
            bool enter_pressed = ImGui::IsKeyPressed(ImGuiKey_Enter) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter);
            if (enter_pressed && io.KeyShift) {
                data->InsertChars(data->CursorPos, "\n");
                insert_newline = true;  // Signal that we handled it
            }
        } else if (data->EventFlag == ImGuiInputTextFlags_CallbackHistory) {
            if (data->EventKey == ImGuiKey_UpArrow) {
                if (panel->show_completion_popup_) {
                    // Navigate up in completion list
                    if (panel->completion_selected_ > 0) {
                        panel->completion_selected_--;
                    }
                    return 0;
                }
                panel->NavigateHistory(-1);
                data->DeleteChars(0, data->BufTextLen);
                data->InsertChars(0, panel->input_buffer_);
                data->SelectAll();
            } else if (data->EventKey == ImGuiKey_DownArrow) {
                if (panel->show_completion_popup_) {
                    // Navigate down in completion list
                    if (panel->completion_selected_ < static_cast<int>(panel->completion_suggestions_.size()) - 1) {
                        panel->completion_selected_++;
                    }
                    return 0;
                }
                panel->NavigateHistory(1);
                data->DeleteChars(0, data->BufTextLen);
                data->InsertChars(0, panel->input_buffer_);
                data->SelectAll();
            }
        } else if (data->EventFlag == ImGuiInputTextFlags_CallbackCompletion) {
            // Tab key pressed - show completions or apply selected
            if (panel->show_completion_popup_ && !panel->completion_suggestions_.empty()) {
                // Apply selected completion
                panel->ApplyCompletion(panel->completion_suggestions_[panel->completion_selected_]);
                panel->show_completion_popup_ = false;
                data->DeleteChars(0, data->BufTextLen);
                data->InsertChars(0, panel->input_buffer_);
            } else {
                // Get completions for current input
                std::string partial(data->Buf, data->BufTextLen);
                panel->completion_suggestions_.clear();
                panel->GetCompletions(partial, panel->completion_suggestions_);

                if (!panel->completion_suggestions_.empty()) {
                    panel->completion_prefix_ = partial;
                    panel->completion_selected_ = 0;
                    panel->show_completion_popup_ = true;
                }
            }
        }
        return 0;
    };

    // Always keep focus on input field when command window is visible
    // This fixes the issue where Ctrl+Enter loses focus
    ImGui::SetKeyboardFocusHere();

    // Calculate input height (3 lines minimum, grows with content)
    float line_height = ImGui::GetTextLineHeight();
    int num_lines = 1;
    for (const char* p = input_buffer_; *p; p++) {
        if (*p == '\n') num_lines++;
    }
    float input_height = std::max(3.0f, static_cast<float>(num_lines + 1)) * line_height + ImGui::GetStyle().FramePadding.y * 2;
    input_height = std::min(input_height, 8.0f * line_height); // Max 8 lines

    // Multi-line input text
    // EnterReturnsTrue makes it return true when Enter is pressed
    bool enter_pressed = ImGui::InputTextMultiline("##input", input_buffer_, sizeof(input_buffer_),
                                                    ImVec2(-1.0f, input_height), flags, callback, this);

    // Execute on Enter (but not if Shift+Enter was used to insert newline)
    if (enter_pressed && !insert_newline) {
        std::string command(input_buffer_);

        // Remove trailing newline/whitespace
        while (!command.empty() && (command.back() == '\n' || command.back() == '\r' || command.back() == ' ')) {
            command.pop_back();
        }

        if (!command.empty()) {
            ExecuteCommand(command);
            std::memset(input_buffer_, 0, sizeof(input_buffer_));
            focus_input_ = true; // Refocus after execution
        }
        show_completion_popup_ = false; // Hide completion on Enter
    }

    // Reset the newline flag
    insert_newline = false;

    // Show hint below input
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
    ImGui::TextUnformatted("Enter: execute | Shift/Ctrl+Enter: new line | Tab: autocomplete");
    ImGui::PopStyleColor();

    // Hide completion popup if user types (changes input)
    std::string current_input(input_buffer_);
    if (show_completion_popup_ && current_input != completion_prefix_) {
        show_completion_popup_ = false;
    }
}

void CommandWindowPanel::ExecuteCommand(const std::string& command) {
    // Add command to output
    OutputEntry cmd_entry;
    cmd_entry.type = OutputEntry::Type::Command;
    cmd_entry.text = "fx:>> " + command;
    output_.push_back(cmd_entry);

    // Add to history
    AddToHistory(command);

    // Handle special commands
    if (command == "clear") {
        ClearOutput();
        scroll_to_bottom_ = true;
        return;
    }

    if (command == "help" || command == "help()") {
        OutputEntry help;
        help.type = OutputEntry::Type::Result;
        help.text = R"(CyxWiz Command Window Help
==========================

COMMANDS:
  clear       - Clear output window
  help()      - Show this help message

MATLAB-STYLE FUNCTIONS (auto-loaded):
  Linear Algebra:  eye, zeros, ones, svd, eig, qr, chol, lu, det,
                   rank, trace, norm, cond, inv, transpose, solve, lstsq, matmul
  Signal:          fft, ifft, conv, conv2, spectrogram, lowpass, highpass,
                   bandpass, filter, findpeaks, sine, square, noise
  Statistics:      kmeans, dbscan, gmm, pca, tsne, silhouette,
                   confusion_matrix, roc
  Time Series:     acf, pacf, decompose, stationarity, arima, diff,
                   rolling_mean, rolling_std

UTILITY FUNCTIONS:
  printmat(A)      - Print matrix in aligned format (alias: pm)
  pm(A, precision=4) - Print with custom decimal precision

EXAMPLES:
  I = eye(3)              # Create 3x3 identity matrix
  pm(I)                   # Print matrix nicely
  A = [[1,2],[3,4]]
  U, S, V = svd(A)        # Singular value decomposition
  spectrum = fft([1,2,3,4])  # FFT of signal

GROUPED NAMESPACE (alternative):
  cyx.linalg.svd(A)       # Same as svd(A)
  cyx.signal.fft(x)       # Same as fft(x)
  cyx.stats.kmeans(data, k=3)
  cyx.timeseries.arima(data, horizon=5)

Type any Python code to execute.
)";
        output_.push_back(help);
        scroll_to_bottom_ = true;
        return;
    }

    // Execute Python command asynchronously
    if (scripting_engine_) {
        StartAsyncCommand(command);
    } else {
        OutputEntry error;
        error.type = OutputEntry::Type::Error;
        error.text = "Error: Scripting engine not initialized";
        output_.push_back(error);
    }

    scroll_to_bottom_ = true;
}

void CommandWindowPanel::ClearOutput() {
    output_.clear();

    // Re-add welcome message
    OutputEntry welcome;
    welcome.type = OutputEntry::Type::Result;
    welcome.text = "Output cleared.\n";
    output_.push_back(welcome);
}

void CommandWindowPanel::DisplayScriptOutput(const std::string& script_name, const std::string& output, bool is_error) {
    // Add script name as command entry
    OutputEntry script_entry;
    script_entry.type = OutputEntry::Type::Command;
    script_entry.text = "Running script: " + script_name;
    output_.push_back(script_entry);

    // Add output or error
    if (!output.empty()) {
        OutputEntry result_entry;
        result_entry.type = is_error ? OutputEntry::Type::Error : OutputEntry::Type::Result;
        result_entry.text = output;
        output_.push_back(result_entry);
    }

    scroll_to_bottom_ = true;
}

void CommandWindowPanel::NavigateHistory(int direction) {
    if (command_history_.empty()) return;

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

    if (history_position_ >= 0 && history_position_ < static_cast<int>(command_history_.size())) {
        // Copy command from history to input buffer
        const std::string& cmd = command_history_[command_history_.size() - 1 - history_position_];
        std::strncpy(input_buffer_, cmd.c_str(), sizeof(input_buffer_) - 1);
    } else {
        // Clear input if at the end of history
        std::memset(input_buffer_, 0, sizeof(input_buffer_));
    }
}

void CommandWindowPanel::AddToHistory(const std::string& command) {
    // Don't add empty commands or duplicates of the last command
    if (command.empty()) return;
    if (!command_history_.empty() && command_history_.back() == command) return;

    command_history_.push_back(command);

    // Limit history size to 100 entries
    if (command_history_.size() > 100) {
        command_history_.erase(command_history_.begin());
    }

    history_position_ = -1; // Reset history navigation
}

void CommandWindowPanel::GetCompletions(const std::string& partial, std::vector<std::string>& suggestions) {
    if (!scripting_engine_) return;

    // Extract the last word/identifier from the partial input
    std::string word = partial;
    size_t last_space = partial.find_last_of(" \t\n()[]{}+-*/=,<>!&|");
    if (last_space != std::string::npos) {
        word = partial.substr(last_space + 1);
    }

    if (word.empty()) return;

    // Check if it's a dotted attribute access (e.g., "math.sq")
    size_t dot_pos = word.find_last_of('.');
    if (dot_pos != std::string::npos) {
        // Attribute completion: "module.attr"
        std::string module = word.substr(0, dot_pos);
        std::string attr_prefix = word.substr(dot_pos + 1);

        // Use Python dir() to get module attributes
        std::string introspect_code =
            "try:\n"
            "    import builtins\n"
            "    obj = " + module + "\n"
            "    attrs = [a for a in dir(obj) if not a.startswith('_') and a.startswith('" + attr_prefix + "')]\n"
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
            "        if not name.startswith('_') and name.startswith('" + word + "'):\n"
            "            matches.append(name)\n"
            "    # Check builtins\n"
            "    for name in dir(builtins):\n"
            "        if not name.startswith('_') and name.startswith('" + word + "'):\n"
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
        std::vector<std::string> cyxwiz_keywords = {
            "import", "pycyxwiz", "math", "random", "json", "numpy", "help", "clear"
        };

        for (const auto& kw : cyxwiz_keywords) {
            if (kw.find(word) == 0) { // Starts with word
                // Check if not already in suggestions
                if (std::find(suggestions.begin(), suggestions.end(), kw) == suggestions.end()) {
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

void CommandWindowPanel::ApplyCompletion(const std::string& completion) {
    // Replace the last word in input buffer with the completion
    std::string current(input_buffer_);
    size_t last_space = current.find_last_of(" \t\n()[]{}+-*/=,<>!&|");

    if (last_space != std::string::npos) {
        // Replace the word after the last separator
        std::string prefix = current.substr(0, last_space + 1);
        std::string result = prefix + completion;
        std::strncpy(input_buffer_, result.c_str(), sizeof(input_buffer_) - 1);
    } else {
        // Replace entire buffer
        std::strncpy(input_buffer_, completion.c_str(), sizeof(input_buffer_) - 1);
    }
}

void CommandWindowPanel::RenderCompletionPopup() {
    if (!show_completion_popup_ || completion_suggestions_.empty()) return;

    // Calculate popup position (below the input field)
    ImVec2 input_pos = ImGui::GetItemRectMin();
    ImVec2 input_size = ImGui::GetItemRectSize();
    ImVec2 popup_pos(input_pos.x, input_pos.y + input_size.y);

    ImGui::SetNextWindowPos(popup_pos);
    ImGui::SetNextWindowSize(ImVec2(400, 0)); // Auto height

    if (ImGui::Begin("##CompletionPopup", nullptr,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoFocusOnAppearing)) {

        ImGui::Text("Suggestions (use Tab to apply, Up/Down to navigate):");
        ImGui::Separator();

        // Render each suggestion
        for (int i = 0; i < static_cast<int>(completion_suggestions_.size()); i++) {
            bool is_selected = (i == completion_selected_);

            if (is_selected) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f)); // Yellow highlight
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

void CommandWindowPanel::StartAsyncCommand(const std::string& command) {
    if (command_executing_ || !scripting_engine_) {
        return;
    }

    executing_command_ = command;
    command_executing_ = true;
    command_cancel_requested_ = false;

    // Use the scripting engine's async command execution
    scripting_engine_->ExecuteCommandAsync(command);
}

void CommandWindowPanel::CheckAsyncCompletion() {
    if (!command_executing_ || !scripting_engine_) {
        return;
    }

    // Check if command has finished
    if (!scripting_engine_->IsCommandRunning()) {
        // Get the result
        auto result_opt = scripting_engine_->GetCommandResult();
        if (result_opt) {
            auto& result = *result_opt;

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

void CommandWindowPanel::StopAsyncCommand() {
    if (!command_executing_ || !scripting_engine_) {
        return;
    }

    command_cancel_requested_ = true;
    scripting_engine_->StopCommand();
}

} // namespace cyxwiz
