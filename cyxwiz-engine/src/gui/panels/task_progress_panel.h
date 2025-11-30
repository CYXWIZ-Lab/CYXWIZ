#pragma once

#include "gui/panel.h"
#include "core/async_task_manager.h"
#include <imgui.h>
#include <vector>
#include <string>

namespace cyxwiz {

// Panel that displays active and recent tasks with progress
class TaskProgressPanel : public Panel {
public:
    TaskProgressPanel();
    ~TaskProgressPanel() override = default;

    void Render() override;
    const char* GetName() const override { return "Tasks"; }

    // Get status for status bar display
    bool HasActiveTasks() const;
    size_t GetActiveTaskCount() const;
    std::string GetStatusSummary() const;

private:
    void RenderTaskItem(const TaskInfo& info);
    void RenderProgressBar(float progress, const ImVec2& size);
    std::string FormatDuration(std::chrono::steady_clock::time_point start,
                               std::chrono::steady_clock::time_point end) const;
    std::string GetStateString(TaskState state) const;
    ImVec4 GetStateColor(TaskState state) const;

    bool show_completed_ = true;
    size_t max_recent_tasks_ = 10;
};

// Inline progress indicator for status bar
class TaskStatusIndicator {
public:
    // Render a small status indicator (for toolbar/status bar)
    // Returns true if clicked (to open TaskProgressPanel)
    static bool Render();

    // Render a spinner animation
    static void RenderSpinner(float radius = 8.0f);
};

// Modal progress dialog for blocking operations (optional use)
class ProgressDialog {
public:
    ProgressDialog(const std::string& title);
    ~ProgressDialog();

    // Show/update the dialog
    void Show(const std::string& message, float progress = -1.0f);
    void SetMessage(const std::string& message);
    void SetProgress(float progress);
    void Close();

    // Check if user clicked cancel
    bool WasCancelled() const { return cancelled_; }

    // Render (call each frame while visible)
    void Render();

    bool IsOpen() const { return is_open_; }

private:
    std::string title_;
    std::string message_;
    float progress_ = -1.0f;  // -1 = indeterminate
    bool is_open_ = false;
    bool cancelled_ = false;
    bool show_cancel_ = true;
};

} // namespace cyxwiz
