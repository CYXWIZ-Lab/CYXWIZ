#include "job_manager.h"
#include "grpc_client.h"
#include "p2p_client.h"
#include "common.pb.h"
#include <spdlog/spdlog.h>
#include <algorithm>

namespace network {

JobManager::JobManager(GRPCClient* client) : client_(client) {
}

JobManager::~JobManager() = default;

void JobManager::RefreshJobList() {
    if (!IsConnected()) {
        return;
    }

    cyxwiz::protocol::ListJobsResponse response;
    if (!client_->ListJobs(response)) {
        spdlog::error("Failed to fetch job list from server");
        return;
    }

    spdlog::info("Fetched {} jobs from server", response.jobs_size());

    // Merge server jobs into our active_jobs_ list
    for (const auto& job_status : response.jobs()) {
        std::string job_id = job_status.job_id();

        // Check if we already have this job
        ActiveJob* existing_job = FindJob(job_id);

        if (existing_job) {
            // Update existing job
            existing_job->status = job_status;
            existing_job->last_update = std::chrono::steady_clock::now();
        } else {
            // Add new job to our list
            ActiveJob new_job;
            new_job.job_id = job_id;
            new_job.status = job_status;
            new_job.last_update = std::chrono::steady_clock::now();
            new_job.is_p2p_job = true;  // Assume P2P for jobs from server
            new_job.assigned_node_address = "";
            new_job.p2p_auth_token = "";
            new_job.p2p_client = nullptr;

            active_jobs_.push_back(new_job);
            spdlog::info("Discovered new job from server: {}", job_id);
        }
    }
}

void JobManager::Update() {
    if (!IsConnected()) {
        return;
    }

    // Periodically refresh the job list from server
    static auto last_list_refresh = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    auto list_refresh_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_list_refresh);

    if (list_refresh_elapsed >= std::chrono::seconds(10)) {
        RefreshJobList();
        last_list_refresh = now;
    }

    // Poll status for all active jobs

    for (auto& job : active_jobs_) {
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - job.last_update);

        if (elapsed >= status_poll_interval_) {
            cyxwiz::protocol::GetJobStatusResponse response;
            if (client_->GetJobStatus(job.job_id, response)) {
                job.status = response.status();
                job.last_update = now;

                // Log status updates
                spdlog::debug("Job {} status: {}, progress: {:.1f}%",
                             job.job_id,
                             cyxwiz::protocol::StatusCode_Name(job.status.status()),
                             job.status.progress() * 100.0);

                // Phase 5: Check for NodeAssignment and auto-trigger P2P connection
                if (job.is_p2p_job && response.has_node_assignment() && !job.p2p_client) {
                    const auto& assignment = response.node_assignment();

                    spdlog::info("🎯 Job {} assigned to node!", job.job_id);
                    spdlog::info("   Node ID: {}", assignment.node_id());
                    spdlog::info("   Endpoint: {}", assignment.node_endpoint());
                    spdlog::info("   Token expires: {} (Unix timestamp)", assignment.token_expires_at());
                    spdlog::info("   JWT token: {}...", assignment.auth_token().substr(0, 30));

                    // Store node assignment details
                    job.assigned_node_address = assignment.node_endpoint();
                    job.p2p_auth_token = assignment.auth_token();

                    // Auto-trigger P2P connection
                    spdlog::info("🚀 Auto-triggering P2P connection for job {}", job.job_id);
                    if (!StartP2PExecution(job.job_id)) {
                        spdlog::error("Failed to start P2P execution for job {}", job.job_id);
                        // Don't fail completely - will retry on next poll
                    }
                }

                // Remove completed/failed jobs after a while
                if (job.status.status() == cyxwiz::protocol::STATUS_SUCCESS ||
                    job.status.status() == cyxwiz::protocol::STATUS_FAILED ||
                    job.status.status() == cyxwiz::protocol::STATUS_CANCELLED) {
                    spdlog::info("Job {} finished with status: {}",
                                job.job_id,
                                cyxwiz::protocol::StatusCode_Name(job.status.status()));
                }
            }
        }
    }

    // Clean up finished jobs older than 1 minute
    active_jobs_.erase(
        std::remove_if(active_jobs_.begin(), active_jobs_.end(),
            [&now](const ActiveJob& job) {
                bool is_finished = (job.status.status() == cyxwiz::protocol::STATUS_SUCCESS ||
                                   job.status.status() == cyxwiz::protocol::STATUS_FAILED ||
                                   job.status.status() == cyxwiz::protocol::STATUS_CANCELLED);
                auto age = std::chrono::duration_cast<std::chrono::seconds>(now - job.last_update);
                return is_finished && age > std::chrono::seconds(60);
            }),
        active_jobs_.end()
    );
}

bool JobManager::SubmitJob(const cyxwiz::protocol::JobConfig& config, std::string& out_job_id) {
    if (!IsConnected()) {
        spdlog::error("Not connected to server");
        return false;
    }

    // Create submit job request
    cyxwiz::protocol::SubmitJobRequest request;
    *request.mutable_config() = config;

    // Submit via gRPC
    cyxwiz::protocol::SubmitJobResponse response;
    if (!client_->SubmitJob(request, response)) {
        spdlog::error("Failed to submit job: {}", client_->GetLastError());
        return false;
    }

    if (response.status() != cyxwiz::protocol::STATUS_SUCCESS) {
        spdlog::error("Job submission rejected: {}",
                     response.has_error() ? response.error().message() : "Unknown error");
        return false;
    }

    out_job_id = response.job_id();
    spdlog::info("Job submitted successfully. Job ID: {}", out_job_id);

    // Add to active jobs list
    ActiveJob active_job;
    active_job.job_id = out_job_id;
    active_job.status.set_job_id(out_job_id);
    active_job.status.set_status((cyxwiz::protocol::StatusCode)3); // STATUS_PENDING (avoid Windows macro conflict)
    active_job.last_update = std::chrono::steady_clock::now();
    active_job.is_p2p_job = false;  // Traditional job, not P2P
    active_jobs_.push_back(active_job);

    return true;
}

bool JobManager::SubmitSimpleJob(const std::string& model_definition,
                                   const std::string& dataset_uri,
                                   std::string& out_job_id) {
    // Create a simple job config for testing
    cyxwiz::protocol::JobConfig config;
    config.set_job_type(cyxwiz::protocol::JOB_TYPE_TRAINING);
    config.set_priority(cyxwiz::protocol::PRIORITY_NORMAL);
    config.set_model_definition(model_definition);
    config.set_dataset_uri(dataset_uri);
    config.set_batch_size(32);
    config.set_epochs(10);
    config.set_required_device(cyxwiz::protocol::DEVICE_CUDA); // Request CUDA GPU
    config.set_estimated_memory(1024 * 1024 * 1024);  // 1 GB
    config.set_estimated_duration(600);                // 10 minutes
    config.set_payment_amount(1.0);                    // 1 CYXWIZ token

    return SubmitJob(config, out_job_id);
}

void JobManager::CancelJob(const std::string& job_id) {
    if (!IsConnected()) {
        spdlog::error("Not connected to server");
        return;
    }

    spdlog::info("Cancelling job: {}", job_id);

    cyxwiz::protocol::CancelJobResponse response;
    if (client_->CancelJob(job_id, response)) {
        if (response.status() == cyxwiz::protocol::STATUS_SUCCESS) {
            spdlog::info("Job {} cancelled successfully", job_id);

            // Update status in active jobs
            for (auto& job : active_jobs_) {
                if (job.job_id == job_id) {
                    job.status.set_status(cyxwiz::protocol::STATUS_CANCELLED);
                    job.last_update = std::chrono::steady_clock::now();
                    break;
                }
            }
        } else {
            spdlog::warn("Job cancellation status: {}",
                        cyxwiz::protocol::StatusCode_Name(response.status()));
        }
    } else {
        spdlog::error("Failed to cancel job: {}", client_->GetLastError());
    }
}

bool JobManager::GetJobStatus(const std::string& job_id, cyxwiz::protocol::JobStatus& out_status) {
    if (!IsConnected()) {
        spdlog::error("Not connected to server");
        return false;
    }

    cyxwiz::protocol::GetJobStatusResponse response;
    if (!client_->GetJobStatus(job_id, response)) {
        spdlog::error("Failed to get job status: {}", client_->GetLastError());
        return false;
    }

    out_status = response.status();
    return true;
}

bool JobManager::IsConnected() const {
    return client_ && client_->IsConnected();
}

// P2P workflow: Submit job to Central Server and get node assignment
bool JobManager::SubmitJobWithP2P(const cyxwiz::protocol::JobConfig& config, std::string& out_job_id) {
    if (!IsConnected()) {
        spdlog::error("Not connected to Central Server");
        return false;
    }

    // Submit job to Central Server (same as regular submission)
    if (!SubmitJob(config, out_job_id)) {
        return false;
    }

    // Mark this job as P2P
    ActiveJob* job = FindJob(out_job_id);
    if (job) {
        job->is_p2p_job = true;
        spdlog::info("Job {} marked as P2P job. Waiting for node assignment...", out_job_id);
    }

    // Note: The Central Server will assign a node and provide:
    // 1. Node address (from NodeAssignment in job.proto)
    // 2. JWT authentication token for P2P connection
    //
    // These will be retrieved via GetJobStatus() polling in Update() method
    // Once we detect node assignment, StartP2PExecution() should be called

    return true;
}

// P2P workflow: Connect to assigned node and start job execution
bool JobManager::StartP2PExecution(const std::string& job_id) {
    ActiveJob* job = FindJob(job_id);
    if (!job) {
        spdlog::error("Job {} not found in active jobs", job_id);
        return false;
    }

    if (!job->is_p2p_job) {
        spdlog::error("Job {} is not a P2P job", job_id);
        return false;
    }

    if (job->assigned_node_address.empty()) {
        spdlog::error("Job {} has no assigned node yet", job_id);
        return false;
    }

    if (job->p2p_auth_token.empty()) {
        spdlog::warn("Job {} has no P2P auth token, using empty token", job_id);
    }

    spdlog::info("Starting P2P execution for job {} on node {}", job_id, job->assigned_node_address);

    // Create P2P client
    auto p2p_client = std::make_shared<P2PClient>();

    // Connect to assigned node
    if (!p2p_client->ConnectToNode(job->assigned_node_address, job_id, job->p2p_auth_token)) {
        spdlog::error("Failed to connect to node: {}", p2p_client->GetLastError());
        return false;
    }

    // Send job configuration to node
    cyxwiz::protocol::JobConfig job_config;
    // TODO: Load job config from original submission (currently we don't store it)
    // For now, we'll need to pass it as parameter or store it in ActiveJob

    // Start training stream
    if (!p2p_client->StartTrainingStream(job_id)) {
        spdlog::error("Failed to start training stream: {}", p2p_client->GetLastError());
        return false;
    }

    // Store P2P client
    job->p2p_client = p2p_client;
    p2p_clients_[job_id] = p2p_client;

    spdlog::info("P2P execution started for job {}", job_id);
    return true;
}

// Get P2P client for a specific job (for UI integration)
std::shared_ptr<P2PClient> JobManager::GetP2PClient(const std::string& job_id) {
    auto it = p2p_clients_.find(job_id);
    if (it != p2p_clients_.end()) {
        return it->second;
    }
    return nullptr;
}

// Helper: Find active job by ID
ActiveJob* JobManager::FindJob(const std::string& job_id) {
    for (auto& job : active_jobs_) {
        if (job.job_id == job_id) {
            return &job;
        }
    }
    return nullptr;
}

} // namespace network
