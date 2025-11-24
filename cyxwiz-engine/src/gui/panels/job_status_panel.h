#pragma once

#include "../panel.h"
#include "../../network/job_manager.h"
#include <imgui.h>
#include <memory>
#include <string>
#include <chrono>

namespace cyxwiz {

/**
 * JobStatusPanel - Displays job submission, assignment, and P2P auto-connection status
 *
 * This panel visualizes the orchestration workflow:
 * 1. Job submission to Central Server
 * 2. Waiting for node assignment
 * 3. Receiving NodeAssignment (node_id, endpoint, JWT token)
 * 4. Auto-triggering P2P connection
 * 5. Connection established and training started
 *
 * This is separate from P2PTrainingPanel which monitors the training itself.
 * This panel focuses on the job lifecycle and orchestration.
 */
class JobStatusPanel : public Panel {
public:
    JobStatusPanel();
    ~JobStatusPanel() override;

    void Render() override;

    // Set JobManager reference (called by MainWindow after construction)
    void SetJobManager(network::JobManager* job_manager);

    // Manual refresh trigger
    void Refresh();

private:
    // Render sub-components
    void RenderJobList();
    void RenderSelectedJobDetails();
    void RenderNodeAssignment();
    void RenderP2PConnectionStatus();
    void RenderJobControls();

    // Helper methods
    void SelectJob(const std::string& job_id);
    std::string GetStatusString(int status_code) const;
    ImVec4 GetStatusColor(int status_code) const;
    std::string FormatTimestamp(int64_t unix_timestamp) const;
    std::string GetTimeUntilExpiration(int64_t expiration_timestamp) const;

    // JobManager reference (not owned)
    network::JobManager* job_manager_;

    // UI state
    std::string selected_job_id_;
    bool auto_refresh_;
    float refresh_interval_;  // seconds
    std::chrono::steady_clock::time_point last_refresh_;

    // Display settings
    bool show_all_jobs_;  // false = only show P2P jobs
    bool show_completed_jobs_;
    int max_displayed_jobs_;

    // Job submission UI
    char model_definition_input_[256];
    char dataset_uri_input_[256];
    bool submit_job_dialog_open_;
};

} // namespace cyxwiz
