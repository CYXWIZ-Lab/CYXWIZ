#include "job_manager.h"
#include "grpc_client.h"
#include "common.pb.h"
#include <spdlog/spdlog.h>
#include <algorithm>

namespace network {

JobManager::JobManager(GRPCClient* client) : client_(client) {
}

JobManager::~JobManager() = default;

void JobManager::Update() {
    if (!IsConnected()) {
        return;
    }

    // Poll status for all active jobs
    auto now = std::chrono::steady_clock::now();

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

} // namespace network
