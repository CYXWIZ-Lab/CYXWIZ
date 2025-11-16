#pragma once

#include "../panel.h"
#include <string>
#include <vector>
#include <memory>

namespace scripting {
    class ScriptingEngine;
}

namespace cyxwiz {

/**
 * Command Window Panel (MATLAB-style REPL)
 * Interactive Python command interface with output display and command history
 */
class CommandWindowPanel : public Panel {
public:
    CommandWindowPanel();
    ~CommandWindowPanel() override = default;

    void Render() override;

    // Set scripting engine (shared with other panels)
    void SetScriptingEngine(std::shared_ptr<scripting::ScriptingEngine> engine);

private:
    // Output entry (command or result)
    struct OutputEntry {
        enum class Type {
            Command,    // User input (f:> ...)
            Result,     // Execution result
            Error       // Error message
        };

        Type type;
        std::string text;
    };

    // Rendering functions
    void RenderOutputArea();
    void RenderInputArea();

    // Command execution
    void ExecuteCommand(const std::string& command);
    void ClearOutput();

    // Command history navigation
    void NavigateHistory(int direction); // -1 = up, +1 = down
    void AddToHistory(const std::string& command);

    // Data
    std::shared_ptr<scripting::ScriptingEngine> scripting_engine_;
    std::vector<OutputEntry> output_;
    std::vector<std::string> command_history_;
    int history_position_;

    // UI state
    char input_buffer_[1024];
    bool scroll_to_bottom_;
    bool focus_input_;
};

} // namespace cyxwiz
