// logs_panel.h - Log viewer with daemon integration
#pragma once
#include "gui/server_panel.h"
#include "ipc/daemon_client.h"
#include <vector>
#include <string>
#include <deque>

namespace cyxwiz::servernode::gui {

class LogsPanel : public ServerPanel {
public:
    LogsPanel() : ServerPanel("Logs") {}
    void Render() override;

private:
    void RenderLogEntry(const ipc::LogEntry& entry);
    void RefreshLogs();
    std::string FormatTimestamp(int64_t timestamp);
    ImVec4 GetLevelColor(const std::string& level);
    bool PassesFilter(const ipc::LogEntry& entry);

    // Logs buffer (circular, max entries)
    std::deque<ipc::LogEntry> logs_;
    static constexpr size_t MAX_LOGS = 1000;
    bool logs_loaded_ = false;

    // Filters
    int log_level_filter_ = 0;  // 0=All, 1=Debug, 2=Info, 3=Warn, 4=Error
    char search_filter_[256] = "";
    bool auto_scroll_ = true;

    // Stats
    int debug_count_ = 0;
    int info_count_ = 0;
    int warn_count_ = 0;
    int error_count_ = 0;
};

} // namespace cyxwiz::servernode::gui
