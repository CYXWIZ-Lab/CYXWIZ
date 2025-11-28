#pragma once

#include <vector>
#include <string>

struct ImVec4;

namespace gui {

class Console {
public:
    enum class LogLevel {
        Info,
        Warning,
        Error,
        Success,
        Debug
    };

    struct LogEntry {
        std::string message;
        LogLevel level;
        float timestamp;
    };

    Console();
    ~Console();

    void Render();
    void AddLog(const std::string& message, LogLevel level = LogLevel::Info);
    void AddInfo(const std::string& message);
    void AddWarning(const std::string& message);
    void AddError(const std::string& message);
    void AddSuccess(const std::string& message);
    void Clear();

    // Visibility control for sidebar integration
    bool* GetVisiblePtr() { return &show_window_; }

private:
    void RenderLogTab(const char* name, LogLevel filter);
    void RenderAllTab();
    void ExecCommand(const char* command);
    const char* GetLevelPrefix(LogLevel level) const;
    ImVec4 GetLevelColor(LogLevel level) const;

    std::vector<LogEntry> items_;
    char input_buf_[256];
    bool scroll_to_bottom_;
    bool show_window_;
    bool auto_scroll_;
    int selected_tab_;
};

} // namespace gui
