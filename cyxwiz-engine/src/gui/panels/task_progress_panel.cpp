#include "task_progress_panel.h"
#include "core/async_task_manager.h"
#include "core/engine_config.h"
#include "core/training_manager.h"
#include "gui/icons.h"
#include "gui/ui_buttons.h"
#include "gui/ui_tokens.h"
#include "gui/ui_widgets.h"
#include <imgui.h>
#include <algorithm>
#include <cstdint>
#include <string>

namespace cyxwiz {
namespace {

task_list::State StateOf(TaskState s) {
    switch (s) {
        case TaskState::Pending: return task_list::State::Pending;
        case TaskState::Running: return task_list::State::Running;
        case TaskState::Completed: return task_list::State::Completed;
        case TaskState::Failed: return task_list::State::Failed;
        case TaskState::Cancelled: return task_list::State::Cancelled;
    }
    return task_list::State::Pending;
}

const char* IconOf(task_list::Mark m) {
    switch (m) {
        case task_list::Mark::Waiting: return ICON_FA_CLOCK;
        case task_list::Mark::Running: return ICON_FA_SPINNER;
        case task_list::Mark::Done: return ICON_FA_CIRCLE_CHECK;
        case task_list::Mark::Failed: return ICON_FA_TRIANGLE_EXCLAMATION;
        case task_list::Mark::Cancelled: return ICON_FA_CIRCLE_XMARK;
    }
    return ICON_FA_CLOCK;
}

ImVec4 ColourOf(task_list::Mark m) {
    const ui::Tokens& t = ui::CurrentTokens();
    switch (m) {
        case task_list::Mark::Waiting: return t.pending;
        case task_list::Mark::Running: return t.running;
        case task_list::Mark::Done: return t.success;
        case task_list::Mark::Failed: return t.caution;
        case task_list::Mark::Cancelled: return t.warning;
    }
    return t.text;
}

task_list::TaskFacts FactsOf(const TaskInfo& info, std::chrono::steady_clock::time_point steady_now,
                             std::chrono::system_clock::time_point wall_now) {
    task_list::TaskFacts f;
    f.id = info.id;
    f.name = info.name;
    f.message = info.status_message;
    f.error = info.error_message;
    f.state = StateOf(info.state);
    f.progress = info.progress;
    f.cancellable = info.cancellable;
    using std::chrono::duration_cast;
    using std::chrono::milliseconds;
    const bool finished = info.state == TaskState::Completed || info.state == TaskState::Failed ||
                          info.state == TaskState::Cancelled;
    if (info.state == TaskState::Running) {
        f.elapsed_ms = duration_cast<milliseconds>(steady_now - info.start_time).count();
    } else if (finished) {
        f.elapsed_ms = duration_cast<milliseconds>(info.end_time - info.start_time).count();
        const auto ago = duration_cast<std::chrono::system_clock::duration>(steady_now - info.end_time);
        f.ended_at = std::chrono::system_clock::to_time_t(wall_now - ago);
    }
    return f;
}

void Heading(const char* text) {
    ImGui::Spacing();
    ImGui::TextColored(ui::CurrentTokens().text_dim, "%s", text);
}

}  // namespace

TaskProgressPanel::TaskProgressPanel()
    : Panel("Background Tasks", false),
      show_finished_(core::EngineConfig::Instance().GetMonitor().show_finished_tasks) {}

task_list::Input TaskProgressPanel::ReadTasks() {
    auto& manager = AsyncTaskManager::Instance();
    const auto steady_now = std::chrono::steady_clock::now();
    const auto wall_now = std::chrono::system_clock::now();
    task_list::Input in;
    in.now = std::chrono::system_clock::to_time_t(wall_now);
    in.show_finished = show_finished_;
    for (const TaskInfo& info : manager.GetActiveTasks()) in.running.push_back(FactsOf(info, steady_now, wall_now));
    in.finished_total = manager.GetFinishedTaskCount();
    if (show_finished_)
        for (const TaskInfo& info : manager.GetFinishedTasks(task_list::kFinishedShown))
            in.finished.push_back(FactsOf(info, steady_now, wall_now));

    if (steady_now - trace_read_at_ > std::chrono::milliseconds(500)) {
        trace_ = TrainingTraceCollector::LatestTrace();
        trace_read_at_ = steady_now;
    }
    if (trace_.available)
        in.last_run = task_list::RunFacts{trace_.status, trace_.effective_backend, trace_.effective_device_name,
                                          trace_.latest_timestamp, TrainingManager::Instance().IsTrainingActive()};
    return in;
}

void TaskProgressPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(520, 360), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin(GetName(), &visible_)) {
        ImGui::End();
        return;
    }
    const ui::Tokens& t = ui::CurrentTokens();
    const task_list::View view = task_list::Build(ReadTasks());

    // Summary, then Show finished and Cancel all at the right
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_bright, "%s", view.summary.c_str());
    const char* kShow = "Show finished";
    const float box = ImGui::GetFrameHeight() + ImGui::GetStyle().ItemInnerSpacing.x + ImGui::CalcTextSize(kShow).x;
    const float cancel = ui::ButtonWidth("Cancel all", ui::ButtonSize::Small);
    ui::SameLineRight(box + ImGui::GetStyle().ItemSpacing.x * 2.0f + cancel);
    ImGui::Checkbox(kShow, &show_finished_);
    ui::Tooltip("Your default is in Preferences > Tools > Background Tasks");
    ImGui::SameLine(0.0f, ImGui::GetStyle().ItemSpacing.x * 2.0f);
    if (ui::DangerButton("Cancel all", view.cancel_all_enabled, view.cancel_all_tooltip.c_str()))
        AsyncTaskManager::Instance().CancelAll();
    if (view.cancel_all_enabled) ui::Tooltip(view.cancel_all_tooltip.c_str());

    if (view.last_run.shown) {
        ImGui::Spacing();
        RenderRunLine(view.last_run);
    }

    Heading("Running");
    if (!view.running_empty.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_dim, "%s", view.running_empty.c_str());
        ImGui::PopTextWrapPos();
    } else {
        RenderRows("##running", view.running, true);
    }

    if (show_finished_) {
        Heading(view.finished_heading.c_str());
        if (!view.finished_empty.empty())
            ImGui::TextColored(t.text_dim, "%s", view.finished_empty.c_str());
        else
            RenderRows("##finished", view.finished, false);
    } else if (!view.hidden_note.empty()) {
        ImGui::Spacing();
        ImGui::TextColored(t.text_dim, "%s", view.hidden_note.c_str());
    }
    ImGui::End();
}

void TaskProgressPanel::RenderRunLine(const task_list::RunLine& line) {
    const ui::Tokens& t = ui::CurrentTokens();
    ui::BeginCard("##last_run");
    ImGui::TextColored(ColourOf(line.mark), "%s", IconOf(line.mark));
    ImGui::SameLine();
    ImGui::TextColored(t.text_bright, "%s", line.text.c_str());
    if (!line.detail.empty()) {
        ImGui::SameLine();
        ImGui::TextColored(t.text_dim, "%s", line.detail.c_str());
    }
    if (open_training_dashboard_) {
        const char* kOpen = "Open in Training Dashboard";
        ui::SameLineRight(ui::ButtonWidth(kOpen, ui::ButtonSize::Small));
        if (ui::LinkButton(kOpen)) open_training_dashboard_();
    }
    ui::EndCard();
}

void TaskProgressPanel::RenderRows(const char* id, const std::vector<task_list::Row>& rows, bool running) {
    const ui::Tokens& t = ui::CurrentTokens();
    const float icon_w = ImGui::GetFrameHeight();
    const float right_w = running ? 260.0f : ImGui::CalcTextSize("cancelled after 59 min 59 s").x;
    const float end_w = running ? ImGui::GetFrameHeight() : ImGui::CalcTextSize("00 Nov 00:00").x;
    if (!ImGui::BeginTable(id, 4, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_PadOuterX)) return;
    ImGui::TableSetupColumn("icon", ImGuiTableColumnFlags_WidthFixed, icon_w);
    ImGui::TableSetupColumn("task", ImGuiTableColumnFlags_WidthStretch, 1.0f);
    ImGui::TableSetupColumn("time", ImGuiTableColumnFlags_WidthFixed, right_w);
    ImGui::TableSetupColumn("end", ImGuiTableColumnFlags_WidthFixed, end_w);
    for (const task_list::Row& row : rows) {
        ImGui::PushID(reinterpret_cast<const void*>(static_cast<uintptr_t>(row.id)));
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextColored(ColourOf(row.mark), "%s", IconOf(row.mark));
        ImGui::TableNextColumn();
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text, "%s", row.name.c_str());
        if (!row.message.empty()) ImGui::TextColored(t.text_dim, "%s", row.message.c_str());
        if (!row.error.empty()) ImGui::TextColored(t.caution, "%s", row.error.c_str());
        ImGui::PopTextWrapPos();
        ImGui::TableNextColumn();
        if (row.show_progress) {
            const float text_w = ImGui::CalcTextSize("100% \xC2\xB7 59 min 59 s").x;
            ui::ProgressLine(row.progress, t.accent, std::max(40.0f, right_w - text_w - 8.0f));
            ImGui::SameLine();
            ImGui::TextColored(t.text_dim, "%s", row.progress_text.c_str());
        } else {
            ImGui::TextColored(t.text_dim, "%s", row.took.c_str());
        }
        ImGui::TableNextColumn();
        if (running) {
            if (row.can_cancel &&
                ui::GhostButton(ICON_FA_XMARK, true, nullptr, false, ui::ButtonSize::Small, &t.error))
                AsyncTaskManager::Instance().Cancel(row.id);
            if (row.can_cancel) ui::Tooltip(("Cancel " + row.name).c_str());
        } else {
            ImGui::TextColored(t.text_dim, "%s", row.clock.c_str());
        }
        ImGui::PopID();
    }
    ImGui::EndTable();
}

// ============================================================================
// TaskStatusIndicator
// ============================================================================

namespace {
std::string RunningSummary(size_t count) {
    return count == 1 ? std::string("1 task running") : std::to_string(count) + " tasks running";
}
}  // namespace

float TaskStatusIndicator::Width() {
    const size_t count = AsyncTaskManager::Instance().GetActiveTaskCount();
    if (count == 0) return 0.0f;
    return ImGui::CalcTextSize(ICON_FA_SPINNER).x + ImGui::GetStyle().ItemSpacing.x +
           ImGui::CalcTextSize(RunningSummary(count).c_str()).x;
}

bool TaskStatusIndicator::Render() {
    const auto tasks = AsyncTaskManager::Instance().GetActiveTasks();
    if (tasks.empty()) return false;
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::BeginGroup();
    ImGui::TextColored(t.running, "%s", ICON_FA_SPINNER);
    ImGui::SameLine();
    ImGui::TextColored(t.running, "%s", RunningSummary(tasks.size()).c_str());
    ImGui::EndGroup();
    bool clicked = false;
    if (ImGui::IsItemHovered()) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        ImGui::BeginTooltip();
        for (const TaskInfo& task : tasks)
            ImGui::Text("%s: %.0f%%", task.name.c_str(), task.progress * 100.0f);
        ImGui::TextColored(t.text_dim, "Click to open Background Tasks");
        ImGui::EndTooltip();
        clicked = ImGui::IsMouseClicked(ImGuiMouseButton_Left);
    }
    return clicked;
}

}  // namespace cyxwiz
