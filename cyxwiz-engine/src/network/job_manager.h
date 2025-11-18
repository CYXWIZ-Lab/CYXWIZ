#pragma once
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include "job.pb.h"

namespace network {

class GRPCClient;

struct ActiveJob {
    std::string job_id;
    cyxwiz::protocol::JobStatus status;
    std::chrono::steady_clock::time_point last_update;
};

class JobManager {
public:
    explicit JobManager(GRPCClient* client);
    ~JobManager();

    // Update all active jobs (call this in main loop)
    void Update();

    // Job submission with full configuration
    bool SubmitJob(const cyxwiz::protocol::JobConfig& config, std::string& out_job_id);

    // Simplified job submission for testing
    bool SubmitSimpleJob(const std::string& model_definition,
                         const std::string& dataset_uri,
                         std::string& out_job_id);

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
};

} // namespace network
