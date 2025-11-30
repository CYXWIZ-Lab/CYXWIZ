#include "task_progress_panel.h"
#include "gui/icons.h"
#include <imgui.h>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace cyxwiz {

// ============================================================================
// TaskProgressPanel Implementation
// ============================================================================

TaskProgressPanel::TaskProgressPanel()
    : Panel("Tasks", false) {
}

void TaskProgressPanel::Render() {
    if (!visible_) {
        return;
    }

    ImGui::SetNextWindowSize(ImVec2(400, 300), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Background Tasks", &visible_, ImGuiWindowFlags_None)) {
        auto& manager = AsyncTaskManager::Instance();

        // Toolbar
        if (ImGui::Button(ICON_FA_STOP " Cancel All")) {
            manager.CancelAll();
        }
        ImGui::SameLine();
        ImGui::Checkbox("Show Completed", &show_completed_);

        ImGui::Separator();

        // Get tasks
        auto tasks = manager.GetRecentTasks(max_recent_tasks_);

        if (tasks.empty()) {
            ImGui::TextDisabled("No active tasks");
        } else {
            // Active tasks first
            bool has_active = false;
            for (const auto& task : tasks) {
                if (task.state == TaskState::Running || task.state == TaskState::Pending) {
                    if (!has_active) {
                        ImGui::Text("Active Tasks:");
                        ImGui::Separator();
                        has_active = true;
                    }
                    RenderTaskItem(task);
                }
            }

            // Completed tasks
            if (show_completed_) {
                bool has_completed = false;
                for (const auto& task : tasks) {
                    if (task.state == TaskState::Completed ||
                        task.state == TaskState::Failed ||
                        task.state == TaskState::Cancelled) {
                        if (!has_completed) {
                            if (has_active) {
                                ImGui::Spacing();
                            }
                            ImGui::Text("Recent Tasks:");
                            ImGui::Separator();
                            has_completed = true;
                        }
                        RenderTaskItem(task);
                    }
                }
            }
        }
    }
    ImGui::End();
}

void TaskProgressPanel::RenderTaskItem(const TaskInfo& info) {
    ImGui::PushID(static_cast<int>(info.id));

    // State icon and color
    ImVec4 color = GetStateColor(info.state);
    ImGui::PushStyleColor(ImGuiCol_Text, color);

    const char* icon = ICON_FA_CIRCLE;
    switch (info.state) {
        case TaskState::Pending: icon = ICON_FA_CLOCK; break;
        case TaskState::Running: icon = ICON_FA_SPINNER; break;
        case TaskState::Completed: icon = ICON_FA_CIRCLE_CHECK; break;
        case TaskState::Failed: icon = ICON_FA_CIRCLE_XMARK; break;
        case TaskState::Cancelled: icon = ICON_FA_CIRCLE_XMARK; break;
    }

    ImGui::Text("%s", icon);
    ImGui::PopStyleColor();

    ImGui::SameLine();

    // Task name and status
    ImGui::BeginGroup();
    ImGui::Text("%s", info.name.c_str());

    if (!info.status_message.empty()) {
        ImGui::TextDisabled("%s", info.status_message.c_str());
    }

    // Progress bar for running tasks
    if (info.state == TaskState::Running) {
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 80);
        RenderProgressBar(info.progress, ImVec2(-80, 0));

        ImGui::SameLine();
        ImGui::Text("%.0f%%", info.progress * 100.0f);
    }

    // Duration
    if (info.state != TaskState::Pending) {
        auto end = info.state == TaskState::Running ?
                   std::chrono::steady_clock::now() : info.end_time;
        std::string duration = FormatDuration(info.start_time, end);
        ImGui::TextDisabled("Duration: %s", duration.c_str());
    }

    ImGui::EndGroup();

    // Cancel button for running/pending tasks
    if ((info.state == TaskState::Running || info.state == TaskState::Pending) &&
        info.cancellable) {
        ImGui::SameLine(ImGui::GetWindowWidth() - 40);
        if (ImGui::Button(ICON_FA_XMARK)) {
            AsyncTaskManager::Instance().Cancel(info.id);
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Cancel task");
        }
    }

    // Error message
    if (info.state == TaskState::Failed && !info.error_message.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
        ImGui::TextWrapped("Error: %s", info.error_message.c_str());
        ImGui::PopStyleColor();
    }

    ImGui::Separator();
    ImGui::PopID();
}

void TaskProgressPanel::RenderProgressBar(float progress, const ImVec2& size) {
    ImGui::ProgressBar(progress, size, "");
}

std::string TaskProgressPanel::FormatDuration(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point end) const {

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    auto ms = duration.count();

    std::ostringstream oss;

    if (ms < 1000) {
        oss << ms << "ms";
    } else if (ms < 60000) {
        oss << std::fixed << std::setprecision(1) << (ms / 1000.0) << "s";
    } else {
        auto minutes = ms / 60000;
        auto seconds = (ms % 60000) / 1000;
        oss << minutes << "m " << seconds << "s";
    }

    return oss.str();
}

std::string TaskProgressPanel::GetStateString(TaskState state) const {
    switch (state) {
        case TaskState::Pending: return "Pending";
        case TaskState::Running: return "Running";
        case TaskState::Completed: return "Completed";
        case TaskState::Failed: return "Failed";
        case TaskState::Cancelled: return "Cancelled";
        default: return "Unknown";
    }
}

ImVec4 TaskProgressPanel::GetStateColor(TaskState state) const {
    switch (state) {
        case TaskState::Pending: return ImVec4(0.7f, 0.7f, 0.7f, 1.0f);
        case TaskState::Running: return ImVec4(0.3f, 0.7f, 1.0f, 1.0f);
        case TaskState::Completed: return ImVec4(0.3f, 0.9f, 0.3f, 1.0f);
        case TaskState::Failed: return ImVec4(1.0f, 0.3f, 0.3f, 1.0f);
        case TaskState::Cancelled: return ImVec4(1.0f, 0.7f, 0.3f, 1.0f);
        default: return ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
    }
}

bool TaskProgressPanel::HasActiveTasks() const {
    return AsyncTaskManager::Instance().HasActiveTasks();
}

size_t TaskProgressPanel::GetActiveTaskCount() const {
    return AsyncTaskManager::Instance().GetActiveTaskCount();
}

std::string TaskProgressPanel::GetStatusSummary() const {
    auto count = GetActiveTaskCount();
    if (count == 0) {
        return "";
    } else if (count == 1) {
        return "1 task running";
    } else {
        return std::to_string(count) + " tasks running";
    }
}

// ============================================================================
// TaskStatusIndicator Implementation
// ============================================================================

bool TaskStatusIndicator::Render() {
    auto& manager = AsyncTaskManager::Instance();
    auto count = manager.GetActiveTaskCount();

    if (count == 0) {
        return false;
    }

    bool clicked = false;

    // Get active tasks for tooltip
    auto tasks = manager.GetActiveTasks();

    // Spinner + count
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 0.7f, 1.0f, 1.0f));

    // Animated spinner
    float time = static_cast<float>(ImGui::GetTime());
    int frame = static_cast<int>(time * 10) % 4;
    const char* spinner_frames[] = { ICON_FA_SPINNER, ICON_FA_SPINNER, ICON_FA_SPINNER, ICON_FA_SPINNER };
    (void)spinner_frames;

    // Use rotation for spinner effect
    ImGui::Text(ICON_FA_SPINNER);
    ImGui::PopStyleColor();

    ImGui::SameLine();
    ImGui::Text("%zu", count);

    // Make it clickable
    ImVec2 min = ImGui::GetItemRectMin();
    ImVec2 max = ImGui::GetItemRectMax();
    min.x -= 20;  // Include spinner

    if (ImGui::IsMouseHoveringRect(min, max)) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);

        // Tooltip with task list
        ImGui::BeginTooltip();
        ImGui::Text("Active Tasks:");
        ImGui::Separator();
        for (const auto& task : tasks) {
            ImGui::Text("%s: %.0f%%", task.name.c_str(), task.progress * 100.0f);
        }
        ImGui::Separator();
        ImGui::TextDisabled("Click to view all tasks");
        ImGui::EndTooltip();

        if (ImGui::IsMouseClicked(0)) {
            clicked = true;
        }
    }

    return clicked;
}

void TaskStatusIndicator::RenderSpinner(float radius) {
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImVec2 center = ImVec2(pos.x + radius, pos.y + radius);

    float time = static_cast<float>(ImGui::GetTime());
    int num_segments = 12;
    float start_angle = time * 4.0f;

    ImU32 color = ImGui::GetColorU32(ImGuiCol_Text);
    ImU32 color_fade = ImGui::GetColorU32(ImVec4(0.5f, 0.5f, 0.5f, 0.3f));

    for (int i = 0; i < num_segments; ++i) {
        float angle = start_angle + (i * 2.0f * 3.14159f / num_segments);
        float alpha = static_cast<float>(i) / num_segments;

        ImVec2 p1 = ImVec2(
            center.x + cosf(angle) * (radius * 0.5f),
            center.y + sinf(angle) * (radius * 0.5f)
        );
        ImVec2 p2 = ImVec2(
            center.x + cosf(angle) * radius,
            center.y + sinf(angle) * radius
        );

        ImU32 line_color = ImGui::GetColorU32(ImVec4(0.3f, 0.7f, 1.0f, alpha));
        draw_list->AddLine(p1, p2, line_color, 2.0f);
    }

    ImGui::Dummy(ImVec2(radius * 2, radius * 2));
}

// ============================================================================
// ProgressDialog Implementation
// ============================================================================

ProgressDialog::ProgressDialog(const std::string& title)
    : title_(title) {
}

ProgressDialog::~ProgressDialog() {
    Close();
}

void ProgressDialog::Show(const std::string& message, float progress) {
    message_ = message;
    progress_ = progress;
    is_open_ = true;
    cancelled_ = false;
}

void ProgressDialog::SetMessage(const std::string& message) {
    message_ = message;
}

void ProgressDialog::SetProgress(float progress) {
    progress_ = progress;
}

void ProgressDialog::Close() {
    is_open_ = false;
}

void ProgressDialog::Render() {
    if (!is_open_) {
        return;
    }

    ImGui::OpenPopup(title_.c_str());

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(400, 0), ImGuiCond_Always);

    if (ImGui::BeginPopupModal(title_.c_str(), nullptr,
                                ImGuiWindowFlags_AlwaysAutoResize |
                                ImGuiWindowFlags_NoMove)) {

        // Message
        ImGui::TextWrapped("%s", message_.c_str());
        ImGui::Spacing();

        // Progress bar
        if (progress_ >= 0.0f) {
            // Determinate progress
            ImGui::ProgressBar(progress_, ImVec2(-1, 0), "");
            ImGui::Text("%.0f%% complete", progress_ * 100.0f);
        } else {
            // Indeterminate progress (animated)
            float time = static_cast<float>(ImGui::GetTime());
            float fake_progress = (sinf(time * 2.0f) + 1.0f) * 0.5f;
            ImGui::ProgressBar(fake_progress, ImVec2(-1, 0), "");
            ImGui::TextDisabled("Please wait...");
        }

        ImGui::Spacing();

        // Cancel button
        if (show_cancel_) {
            if (ImGui::Button("Cancel", ImVec2(120, 0))) {
                cancelled_ = true;
                is_open_ = false;
                ImGui::CloseCurrentPopup();
            }
        }

        ImGui::EndPopup();
    }
}

} // namespace cyxwiz
