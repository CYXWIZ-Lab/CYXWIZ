#pragma once
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <map>
#include <atomic>
#include "job.pb.h"

// Forward declare DatasetHandle from data_registry
namespace cyxwiz {
class DatasetHandle;
class JobStatusPanel;
}

namespace network {

class GRPCClient;
class P2PClient;

struct ActiveJob {
    std::string job_id;
    cyxwiz::protocol::JobStatus status;
    std::chrono::steady_clock::time_point last_update;

    // P2P-specific fields
    bool is_p2p_job;
    std::string assigned_node_address;
    std::string p2p_auth_token;
    std::shared_ptr<P2PClient> p2p_client;

    // Store original job config for P2P transmission
    cyxwiz::protocol::JobConfig original_config;
    std::string initial_dataset_bytes;  // For inline dataset transfer

    // Dataset handle for remote streaming (lazy loading)
    std::shared_ptr<cyxwiz::DatasetHandle> remote_dataset;

    // Cancel safety flag - prevents callbacks from accessing job after cancel
    std::shared_ptr<std::atomic<bool>> is_cancelled = std::make_shared<std::atomic<bool>>(false);
};

class JobManager {
public:
    explicit JobManager(GRPCClient* client);
    ~JobManager();

    // Update all active jobs (call this in main loop)
    void Update();

    // Fetch all jobs from server and merge into active_jobs_ list
    void RefreshJobList();

    // Job submission with full configuration (to Central Server)
    bool SubmitJob(const cyxwiz::protocol::JobConfig& config, std::string& out_job_id);

    // Simplified job submission for testing
    bool SubmitSimpleJob(const std::string& model_definition,
                         const std::string& dataset_uri,
                         std::string& out_job_id);

    // P2P workflow: Submit job and get node assignment from Central Server
    bool SubmitJobWithP2P(const cyxwiz::protocol::JobConfig& config,
                         std::string& out_job_id);

    // P2P workflow: Connect to assigned node and start P2P job execution
    bool StartP2PExecution(const std::string& job_id);

    // Set dataset for remote streaming (lazy loading from Engine)
    // Call this before StartP2PExecution to enable dataset streaming
    void SetRemoteDataset(const std::string& job_id, std::shared_ptr<cyxwiz::DatasetHandle> dataset);

    // Get P2P client for a specific job (for UI integration)
    std::shared_ptr<P2PClient> GetP2PClient(const std::string& job_id);

    // Set JobStatusPanel for progress forwarding
    void SetJobStatusPanel(cyxwiz::JobStatusPanel* panel) { job_status_panel_ = panel; }

    // Job control
    void CancelJob(const std::string& job_id);
    bool DeleteJob(const std::string& job_id);  // Deletes from Central Server
    void RemoveLocalJob(const std::string& job_id);  // Removes from local list only
    void ClearAllLocalJobs();  // Clears entire local job list (for historical view)

    // Query specific job status
    bool GetJobStatus(const std::string& job_id, cyxwiz::protocol::JobStatus& out_status);

    // Get all active jobs
    const std::vector<ActiveJob>& GetActiveJobs() const { return active_jobs_; }

    // Check if connected to server
    bool IsConnected() const;

    // Handle server disconnection - close all P2P connections and stop polling
    void OnServerDisconnected();

    // Close all active P2P connections
    void CloseAllP2PConnections();

private:
    GRPCClient* client_;
    std::vector<ActiveJob> active_jobs_;
    std::chrono::seconds status_poll_interval_{30}; // Poll every 30 seconds (P2P stream provides real-time updates)

    // P2P clients map (job_id -> P2PClient)
    std::map<std::string, std::shared_ptr<P2PClient>> p2p_clients_;

    // UI panel for progress forwarding (not owned)
    cyxwiz::JobStatusPanel* job_status_panel_ = nullptr;

    // Track consecutive failures to detect server disconnection
    int consecutive_failures_ = 0;
    static constexpr int MAX_CONSECUTIVE_FAILURES = 3;

    // Helper: Find active job by ID
    ActiveJob* FindJob(const std::string& job_id);
};

} // namespace network
