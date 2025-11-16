#include "command_window.h"
#include "../../scripting/scripting_engine.h"
#include <imgui.h>
#include <cstring>

namespace cyxwiz {

CommandWindowPanel::CommandWindowPanel()
    : Panel("Command Window", true)
    , history_position_(-1)
    , scroll_to_bottom_(false)
    , focus_input_(true)
{
    std::memset(input_buffer_, 0, sizeof(input_buffer_));

    // Welcome message
    OutputEntry welcome;
    welcome.type = OutputEntry::Type::Result;
    welcome.text = "CyxWiz Python Command Window\nType 'help()' for help, 'clear' to clear output\n";
    output_.push_back(welcome);
}

void CommandWindowPanel::SetScriptingEngine(std::shared_ptr<scripting::ScriptingEngine> engine) {
    scripting_engine_ = engine;
}

void CommandWindowPanel::Render() {
    if (!visible_) return;

    ImGui::Begin(GetName(), &visible_);

    // Output area (scrollable)
    RenderOutputArea();

    ImGui::Separator();

    // Input area (bottom)
    RenderInputArea();

    ImGui::End();
}

void CommandWindowPanel::RenderOutputArea() {
    // Child window for scrollable output
    const float footer_height = ImGui::GetStyle().ItemSpacing.y + ImGui::GetFrameHeightWithSpacing();
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
    // Prompt label
    ImGui::Text("f:>");
    ImGui::SameLine();

    // Input field
    ImGui::PushItemWidth(-1.0f);

    ImGuiInputTextFlags flags = ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CallbackHistory;

    // Handle history navigation
    auto callback = [](ImGuiInputTextCallbackData* data) -> int {
        CommandWindowPanel* panel = (CommandWindowPanel*)data->UserData;

        if (data->EventFlag == ImGuiInputTextFlags_CallbackHistory) {
            if (data->EventKey == ImGuiKey_UpArrow) {
                panel->NavigateHistory(-1);
                data->DeleteChars(0, data->BufTextLen);
                data->InsertChars(0, panel->input_buffer_);
                data->SelectAll();
            } else if (data->EventKey == ImGuiKey_DownArrow) {
                panel->NavigateHistory(1);
                data->DeleteChars(0, data->BufTextLen);
                data->InsertChars(0, panel->input_buffer_);
                data->SelectAll();
            }
        }
        return 0;
    };

    // Auto-focus on input field
    if (focus_input_) {
        ImGui::SetKeyboardFocusHere();
        focus_input_ = false;
    }

    if (ImGui::InputText("##input", input_buffer_, sizeof(input_buffer_), flags, callback, this)) {
        std::string command(input_buffer_);

        if (!command.empty()) {
            ExecuteCommand(command);
            std::memset(input_buffer_, 0, sizeof(input_buffer_));
            focus_input_ = true; // Refocus after execution
        }
    }

    ImGui::PopItemWidth();
}

void CommandWindowPanel::ExecuteCommand(const std::string& command) {
    // Add command to output
    OutputEntry cmd_entry;
    cmd_entry.type = OutputEntry::Type::Command;
    cmd_entry.text = "f:> " + command;
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
        help.text = "Available commands:\n"
                   "  clear       - Clear output window\n"
                   "  help()      - Show this help message\n"
                   "  import pycyxwiz - Import CyxWiz Python module\n"
                   "\nYou can execute any Python code here.\n";
        output_.push_back(help);
        scroll_to_bottom_ = true;
        return;
    }

    // Execute Python command
    if (scripting_engine_) {
        auto result = scripting_engine_->ExecuteCommand(command);

        OutputEntry result_entry;
        if (result.success) {
            result_entry.type = OutputEntry::Type::Result;
            result_entry.text = result.output.empty() ? "" : result.output;
        } else {
            result_entry.type = OutputEntry::Type::Error;
            result_entry.text = "Error: " + result.error_message;
        }

        if (!result_entry.text.empty()) {
            output_.push_back(result_entry);
        }
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

} // namespace cyxwiz
