// logs_panel.cpp - Log viewer with daemon integration
#include "gui/panels/logs_panel.h"
#include "gui/icons.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace cyxwiz::servernode::gui {

void LogsPanel::Render() {
    ImGui::PushFont(GetSafeFont(FONT_LARGE));
    ImGui::Text("%s Logs", ICON_FA_SCROLL);
    ImGui::PopFont();
    ImGui::Separator();

    // Connection status
    if (IsDaemonConnected()) {
        ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "%s Daemon Connected", ICON_FA_LINK);
    } else {
        ImGui::TextColored(ImVec4(0.8f, 0.3f, 0.3f, 1.0f), "%s Daemon Disconnected", ICON_FA_LINK_SLASH);
        ImGui::TextDisabled("Connect to daemon to view logs.");
        return;
    }

    ImGui::Spacing();

    // Controls row
    if (ImGui::Button(ICON_FA_ROTATE " Refresh")) {
        RefreshLogs();
    }
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_TRASH " Clear")) {
        logs_.clear();
        debug_count_ = info_count_ = warn_count_ = error_count_ = 0;
    }

    ImGui::SameLine();
    ImGui::Checkbox("Auto-scroll", &auto_scroll_);

    ImGui::Spacing();

    // Filters row
    ImGui::SetNextItemWidth(120);
    const char* levels[] = { "All", "Debug", "Info", "Warning", "Error" };
    ImGui::Combo("##Level", &log_level_filter_, levels, IM_ARRAYSIZE(levels));

    ImGui::SameLine();
    ImGui::SetNextItemWidth(250);
    ImGui::InputTextWithHint("##Search", "Filter logs...", search_filter_, sizeof(search_filter_));

    ImGui::SameLine();
    ImGui::TextDisabled("| %d debug, %d info, %d warn, %d error",
                        debug_count_, info_count_, warn_count_, error_count_);

    ImGui::Spacing();

    // Load logs if not loaded
    if (!logs_loaded_) {
        RefreshLogs();
        logs_loaded_ = true;
    }

    // Log view
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.08f, 0.08f, 0.08f, 1.0f));
    ImGui::BeginChild("LogView", ImVec2(0, 0), true);

    ImGui::PushFont(GetSafeFont(FONT_MONO));

    if (logs_.empty()) {
        ImGui::TextDisabled("No logs to display.");
        ImGui::TextDisabled("Logs from the daemon will appear here.");
    } else {
        // Use clipper for performance with many log entries
        ImGuiListClipper clipper;

        // First, count visible logs for the clipper
        std::vector<size_t> visible_indices;
        for (size_t i = 0; i < logs_.size(); i++) {
            if (PassesFilter(logs_[i])) {
                visible_indices.push_back(i);
            }
        }

        clipper.Begin((int)visible_indices.size());
        while (clipper.Step()) {
            for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
                RenderLogEntry(logs_[visible_indices[row]]);
            }
        }
        clipper.End();
    }

    ImGui::PopFont();

    // Auto-scroll
    if (auto_scroll_ && ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 10) {
        ImGui::SetScrollHereY(1.0f);
    }

    ImGui::EndChild();
    ImGui::PopStyleColor();
}

void LogsPanel::RenderLogEntry(const ipc::LogEntry& entry) {
    // Timestamp
    ImGui::TextDisabled("%s", FormatTimestamp(entry.timestamp).c_str());
    ImGui::SameLine();

    // Level with color
    ImVec4 color = GetLevelColor(entry.level);
    ImGui::TextColored(color, "[%s]", entry.level.c_str());
    ImGui::SameLine();

    // Source (if available)
    if (!entry.source.empty()) {
        ImGui::TextDisabled("[%s]", entry.source.c_str());
        ImGui::SameLine();
    }

    // Message
    ImGui::TextWrapped("%s", entry.message.c_str());
}

void LogsPanel::RefreshLogs() {
    std::vector<ipc::LogEntry> entries;
    auto* client = GetDaemonClient();
    if (client && client->IsConnected()) {
        // Get the level filter string
        std::string level_filter;
        switch (log_level_filter_) {
            case 1: level_filter = "debug"; break;
            case 2: level_filter = "info"; break;
            case 3: level_filter = "warn"; break;
            case 4: level_filter = "error"; break;
            default: level_filter = ""; break;
        }

        if (client->GetLogs(entries, 500, level_filter)) {
            logs_.clear();
            debug_count_ = info_count_ = warn_count_ = error_count_ = 0;

            for (auto& entry : entries) {
                logs_.push_back(std::move(entry));

                // Count by level
                if (logs_.back().level == "debug") debug_count_++;
                else if (logs_.back().level == "info") info_count_++;
                else if (logs_.back().level == "warn" || logs_.back().level == "warning") warn_count_++;
                else if (logs_.back().level == "error") error_count_++;

                // Limit buffer size
                if (logs_.size() > MAX_LOGS) {
                    logs_.pop_front();
                }
            }

            spdlog::debug("Loaded {} log entries", logs_.size());
        }
    }
}

std::string LogsPanel::FormatTimestamp(int64_t timestamp) {
    if (timestamp <= 0) return "??:??:??";

    std::time_t time = static_cast<std::time_t>(timestamp);
    std::tm* tm = std::localtime(&time);
    if (!tm) return "??:??:??";

    std::ostringstream oss;
    oss << std::put_time(tm, "%H:%M:%S");
    return oss.str();
}

ImVec4 LogsPanel::GetLevelColor(const std::string& level) {
    if (level == "debug" || level == "trace") {
        return ImVec4(0.5f, 0.5f, 0.5f, 1.0f);  // Gray
    } else if (level == "info") {
        return ImVec4(0.3f, 0.7f, 0.9f, 1.0f);  // Blue
    } else if (level == "warn" || level == "warning") {
        return ImVec4(1.0f, 0.8f, 0.3f, 1.0f);  // Yellow
    } else if (level == "error" || level == "critical") {
        return ImVec4(1.0f, 0.3f, 0.3f, 1.0f);  // Red
    }
    return ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
}

bool LogsPanel::PassesFilter(const ipc::LogEntry& entry) {
    // Level filter
    if (log_level_filter_ > 0) {
        int entry_level = 0;
        if (entry.level == "debug" || entry.level == "trace") entry_level = 1;
        else if (entry.level == "info") entry_level = 2;
        else if (entry.level == "warn" || entry.level == "warning") entry_level = 3;
        else if (entry.level == "error" || entry.level == "critical") entry_level = 4;

        if (entry_level < log_level_filter_) {
            return false;
        }
    }

    // Text search filter
    if (strlen(search_filter_) > 0) {
        std::string filter_lower = search_filter_;
        std::transform(filter_lower.begin(), filter_lower.end(), filter_lower.begin(), ::tolower);

        std::string message_lower = entry.message;
        std::transform(message_lower.begin(), message_lower.end(), message_lower.begin(), ::tolower);

        std::string source_lower = entry.source;
        std::transform(source_lower.begin(), source_lower.end(), source_lower.begin(), ::tolower);

        if (message_lower.find(filter_lower) == std::string::npos &&
            source_lower.find(filter_lower) == std::string::npos) {
            return false;
        }
    }

    return true;
}

} // namespace cyxwiz::servernode::gui
