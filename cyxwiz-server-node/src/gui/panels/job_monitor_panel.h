// job_monitor_panel.h - Active jobs monitoring with daemon integration
#pragma once
#include "gui/server_panel.h"
#include "ipc/daemon_client.h"
#include <vector>
#include <string>
#include <map>

namespace cyxwiz::servernode::gui {

class JobMonitorPanel : public ServerPanel {
public:
    JobMonitorPanel() : ServerPanel("Job Monitor") {}
    void Render() override;

private:
    void RenderJobCard(const ipc::JobInfo& job);
    void RenderCancelDialog();
    void RefreshJobs();
    const char* GetStatusText(int status);
    ImVec4 GetStatusColor(int status);
    std::string FormatDuration(int64_t seconds);

    // Jobs list
    std::vector<ipc::JobInfo> jobs_;
    bool jobs_loaded_ = false;

    // Loss history for plotting (per job)
    std::map<std::string, std::vector<float>> loss_history_;

    // Cancel confirmation dialog
    bool show_cancel_dialog_ = false;
    std::string pending_cancel_id_;
    std::string pending_cancel_name_;
    std::string cancel_error_;

    // Filter
    bool show_completed_ = false;
    bool show_failed_ = false;
};

} // namespace cyxwiz::servernode::gui
