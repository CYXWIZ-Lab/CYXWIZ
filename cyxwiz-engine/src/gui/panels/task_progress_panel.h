#pragma once

#include "gui/panel.h"
#include "core/task_list_presentation.h"
#include "core/training_trace_collector.h"
#include <chrono>
#include <functional>

namespace cyxwiz {

// Background Tasks: the running tasks, the finished ones and the last
// training run in one line. Worded by core/task_list_presentation.
class TaskProgressPanel : public Panel {
public:
    TaskProgressPanel();
    ~TaskProgressPanel() override = default;

    void Render() override;
    const char* GetName() const override { return "Background Tasks"; }

    // "Open in Training Dashboard" on the last-run line.
    void SetOpenTrainingDashboard(std::function<void()> open) { open_training_dashboard_ = std::move(open); }

private:
    task_list::Input ReadTasks();
    void RenderRunLine(const task_list::RunLine& line);
    void RenderRows(const char* id, const std::vector<task_list::Row>& rows, bool running);

    bool show_finished_ = true;
    std::function<void()> open_training_dashboard_;
    // LatestTrace copies the whole trace and may read it from disk: at most every 500 ms
    TrainingTraceSummary trace_;
    std::chrono::steady_clock::time_point trace_read_at_{};
};

// The status bar's task indicator: a spinner and "N tasks running" while
// tasks run, the task list on hover. Returns true when clicked.
class TaskStatusIndicator {
public:
    static bool Render();
    // Its width, to place it at the right of the status bar (0: nothing runs).
    static float Width();
};

}  // namespace cyxwiz
