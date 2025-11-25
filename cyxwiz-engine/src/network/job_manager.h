#pragma once
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <map>
#include "job.pb.h"

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

    // Get P2P client for a specific job (for UI integration)
    std::shared_ptr<P2PClient> GetP2PClient(const std::string& job_id);

    // Job control
    void CancelJob(const std::string& job_id);

    // Query specific job status
    bool GetJobStatus(const std::string& job_id, cyxwiz::protocol::JobStatus& out_status);

    // Get all active jobs
    const std::vector<ActiveJob>& GetActiveJobs() const { return active_jobs_; }

    // Check if connected to server
    bool IsConnected() const;

private:
    GRPCClient* client_;
    std::vector<ActiveJob> active_jobs_;
    std::chrono::seconds status_poll_interval_{5}; // Poll every 5 seconds

    // P2P clients map (job_id -> P2PClient)
    std::map<std::string, std::shared_ptr<P2PClient>> p2p_clients_;

    // Helper: Find active job by ID
    ActiveJob* FindJob(const std::string& job_id);
};

} // namespace network
