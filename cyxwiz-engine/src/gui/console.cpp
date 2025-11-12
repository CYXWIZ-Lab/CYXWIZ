#include "console.h"
#include <imgui.h>
#include <cstring>
#include <algorithm>

namespace gui {

Console::Console()
    : scroll_to_bottom_(false)
    , show_window_(true)
    , auto_scroll_(true)
    , selected_tab_(0)
{
    memset(input_buf_, 0, sizeof(input_buf_));
    // Note: Cannot call AddInfo() here as ImGui::GetTime() requires an active ImGui frame
    // Initial messages will be added in the first Render() call
}

Console::~Console() = default;

void Console::Render() {
    if (!show_window_) return;

    // Add initial messages on first render when ImGui context is active
    static bool first_render = true;
    if (first_render) {
        AddInfo("CyxWiz Console initialized");
        AddInfo("Type 'help' for available commands");
        AddInfo("Ready");
        first_render = false;
    }

    if (ImGui::Begin("Console", &show_window_)) {
        // Toolbar
        if (ImGui::Button("Clear")) {
            Clear();
        }
        ImGui::SameLine();
        if (ImGui::Button("Copy")) {
            ImGui::LogToClipboard();
        }
        ImGui::SameLine();
        ImGui::Checkbox("Auto-scroll", &auto_scroll_);

        ImGui::Separator();

        // Tabs
        if (ImGui::BeginTabBar("ConsoleTabs", ImGuiTabBarFlags_None)) {
            if (ImGui::BeginTabItem("All")) {
                RenderAllTab();
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("Info")) {
                RenderLogTab("Info", LogLevel::Info);
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("Warnings")) {
                RenderLogTab("Warnings", LogLevel::Warning);
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("Errors")) {
                RenderLogTab("Errors", LogLevel::Error);
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("Success")) {
                RenderLogTab("Success", LogLevel::Success);
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }

        ImGui::Separator();

        // Command input at bottom
        bool reclaim_focus = false;
        ImGuiInputTextFlags input_text_flags = ImGuiInputTextFlags_EnterReturnsTrue;
        ImGui::PushItemWidth(-1.0f);
        if (ImGui::InputTextWithHint("##input", "Enter command...", input_buf_, IM_ARRAYSIZE(input_buf_), input_text_flags)) {
            char* s = input_buf_;
            if (s[0]) {
                ExecCommand(s);
            }
            strcpy(s, "");
            reclaim_focus = true;
        }
        ImGui::PopItemWidth();

        ImGui::SetItemDefaultFocus();
        if (reclaim_focus) {
            ImGui::SetKeyboardFocusHere(-1);
        }
    }
    ImGui::End();
}

void Console::RenderAllTab() {
    const float footer_height = ImGui::GetStyle().ItemSpacing.y + ImGui::GetFrameHeightWithSpacing();
    ImGui::BeginChild("AllLogsRegion", ImVec2(0, -footer_height), false, ImGuiWindowFlags_HorizontalScrollbar);

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 1));

    for (const auto& entry : items_) {
        ImVec4 color = GetLevelColor(entry.level);
        ImGui::PushStyleColor(ImGuiCol_Text, color);

        ImGui::Text("[%.2fs] %s %s",
            entry.timestamp,
            GetLevelPrefix(entry.level),
            entry.message.c_str());

        ImGui::PopStyleColor();
    }

    if (auto_scroll_ && (scroll_to_bottom_ || ImGui::GetScrollY() >= ImGui::GetScrollMaxY())) {
        ImGui::SetScrollHereY(1.0f);
    }

    scroll_to_bottom_ = false;
    ImGui::PopStyleVar();
    ImGui::EndChild();
}

void Console::RenderLogTab(const char* name, LogLevel filter) {
    const float footer_height = ImGui::GetStyle().ItemSpacing.y + ImGui::GetFrameHeightWithSpacing();
    ImGui::BeginChild((std::string(name) + "Region").c_str(), ImVec2(0, -footer_height), false, ImGuiWindowFlags_HorizontalScrollbar);

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 1));

    int count = 0;
    for (const auto& entry : items_) {
        if (entry.level == filter) {
            ImVec4 color = GetLevelColor(entry.level);
            ImGui::PushStyleColor(ImGuiCol_Text, color);

            ImGui::Text("[%.2fs] %s",
                entry.timestamp,
                entry.message.c_str());

            ImGui::PopStyleColor();
            count++;
        }
    }

    if (count == 0) {
        ImGui::TextDisabled("No %s messages", name);
    }

    if (auto_scroll_ && (scroll_to_bottom_ || ImGui::GetScrollY() >= ImGui::GetScrollMaxY())) {
        ImGui::SetScrollHereY(1.0f);
    }

    scroll_to_bottom_ = false;
    ImGui::PopStyleVar();
    ImGui::EndChild();
}

void Console::AddLog(const std::string& message, LogLevel level) {
    LogEntry entry;
    entry.message = message;
    entry.level = level;
    entry.timestamp = ImGui::GetTime();
    items_.push_back(entry);
    scroll_to_bottom_ = true;

    // Keep history bounded (e.g., 1000 entries)
    if (items_.size() > 1000) {
        items_.erase(items_.begin());
    }
}

void Console::AddInfo(const std::string& message) {
    AddLog(message, LogLevel::Info);
}

void Console::AddWarning(const std::string& message) {
    AddLog(message, LogLevel::Warning);
}

void Console::AddError(const std::string& message) {
    AddLog(message, LogLevel::Error);
}

void Console::AddSuccess(const std::string& message) {
    AddLog(message, LogLevel::Success);
}

void Console::Clear() {
    items_.clear();
    AddInfo("Console cleared");
}

void Console::ExecCommand(const char* command) {
    AddLog(std::string("> ") + command, LogLevel::Info);

    // TODO: Integrate with Python engine or command processor
    if (strcmp(command, "clear") == 0) {
        Clear();
    } else if (strcmp(command, "help") == 0) {
        AddInfo("Available commands:");
        AddInfo("  clear - Clear console");
        AddInfo("  help - Show this message");
        AddInfo("  test - Test log levels");
    } else if (strcmp(command, "test") == 0) {
        AddInfo("This is an info message");
        AddWarning("This is a warning message");
        AddError("This is an error message");
        AddSuccess("This is a success message");
    } else {
        AddError(std::string("Unknown command: ") + command);
        AddInfo("Type 'help' for available commands");
    }
}

const char* Console::GetLevelPrefix(LogLevel level) const {
    switch (level) {
        case LogLevel::Info:    return "[INFO]";
        case LogLevel::Warning: return "[WARN]";
        case LogLevel::Error:   return "[ERROR]";
        case LogLevel::Success: return "[OK]";
        case LogLevel::Debug:   return "[DEBUG]";
        default:                return "[???]";
    }
}

ImVec4 Console::GetLevelColor(LogLevel level) const {
    switch (level) {
        case LogLevel::Info:    return ImVec4(0.8f, 0.8f, 0.8f, 1.0f); // Gray
        case LogLevel::Warning: return ImVec4(1.0f, 0.8f, 0.0f, 1.0f); // Yellow
        case LogLevel::Error:   return ImVec4(1.0f, 0.3f, 0.3f, 1.0f); // Red
        case LogLevel::Success: return ImVec4(0.3f, 1.0f, 0.3f, 1.0f); // Green
        case LogLevel::Debug:   return ImVec4(0.6f, 0.6f, 1.0f, 1.0f); // Blue
        default:                return ImVec4(1.0f, 1.0f, 1.0f, 1.0f); // White
    }
}

} // namespace gui
