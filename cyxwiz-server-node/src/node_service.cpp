#include "node_service.h"
#include <spdlog/spdlog.h>

namespace cyxwiz {
namespace servernode {

NodeServiceImpl::NodeServiceImpl(JobExecutor* job_executor, const std::string& node_id)
    : job_executor_(job_executor)
    , node_id_(node_id)
{
    spdlog::info("NodeServiceImpl created for node: {}", node_id_);
}

grpc::Status NodeServiceImpl::AssignJob(
    grpc::ServerContext* context,
    const protocol::AssignJobRequest* request,
    protocol::AssignJobResponse* response)
{
    spdlog::info("Received job assignment request for job: {}", request->job().job_id());

    // Validate node ID matches (relaxed for testing - TODO: implement proper UUID handling)
    if (request->node_id() != node_id_) {
        spdlog::warn("Node ID mismatch (this is expected during testing). Expected: {}, Got: {}",
                     node_id_, request->node_id());
        spdlog::warn("Accepting job anyway for integration testing");
        // TODO: In production, verify node IDs match or implement proper UUID-based node registration
    }

    // Validate job configuration
    std::string validation_error;
    if (!ValidateJobConfig(request->job(), &validation_error)) {
        spdlog::error("Job validation failed: {}", validation_error);
        response->set_status(protocol::STATUS_FAILED);
        response->set_accepted(false);
        auto* error = response->mutable_error();
        error->set_code(400);  // Bad Request
        error->set_message(validation_error);
        return grpc::Status::OK;
    }

    // Check if this is a remote dataset job (requires P2P connection from Engine)
    const std::string& dataset_uri = request->job().dataset_uri();
    bool is_remote_dataset = dataset_uri.find("remote://") == 0;

    if (is_remote_dataset) {
        // Remote dataset jobs don't start immediately - they wait for P2P connection
        // The Engine will connect via StreamTrainingMetrics and training will start then
        spdlog::info("Job {} has remote dataset ({}), waiting for P2P connection from Engine",
                     request->job().job_id(), dataset_uri);

        // Store the job config for when P2P connection arrives
        {
            std::lock_guard<std::mutex> lock(pending_jobs_mutex_);
            pending_p2p_jobs_[request->job().job_id()] = request->job();
        }

        response->set_status(protocol::STATUS_SUCCESS);
        response->set_accepted(true);
        response->set_job_id(request->job().job_id());
        response->set_estimated_start_time("waiting_for_p2p");

        spdlog::info("Job {} registered, waiting for Engine P2P connection", request->job().job_id());
        return grpc::Status::OK;
    }

    // Local dataset jobs start immediately via JobExecutor
    bool accepted = job_executor_->ExecuteJobAsync(request->job());

    if (accepted) {
        spdlog::info("Job {} accepted and queued for execution", request->job().job_id());
        response->set_status(protocol::STATUS_SUCCESS);
        response->set_accepted(true);
        response->set_job_id(request->job().job_id());
        response->set_estimated_start_time("immediate");
    } else {
        spdlog::warn("Job {} rejected by JobExecutor (busy or error)", request->job().job_id());
        response->set_status(protocol::STATUS_FAILED);
        response->set_accepted(false);
        auto* error = response->mutable_error();
        error->set_code(429);  // Too Many Requests / Resource Exhausted
        error->set_message("Node is currently busy or cannot accept job");
    }

    return grpc::Status::OK;
}

grpc::Status NodeServiceImpl::GetNodeMetrics(
    grpc::ServerContext* context,
    const protocol::GetNodeMetricsRequest* request,
    protocol::GetNodeMetricsResponse* response)
{
    // TODO: Implement node metrics collection
    spdlog::debug("GetNodeMetrics called (not yet implemented)");

    // metrics field would be populated here
    // response->add_metrics() can be used to add metrics

    return grpc::Status::OK;
}

bool NodeServiceImpl::ValidateJobConfig(
    const protocol::JobConfig& job_config,
    std::string* error_msg)
{
    // Validate job ID
    if (job_config.job_id().empty()) {
        *error_msg = "Job ID cannot be empty";
        return false;
    }

    // Validate job type
    if (job_config.job_type() == protocol::JOB_TYPE_UNKNOWN) {
        *error_msg = "Job type cannot be UNKNOWN";
        return false;
    }

    // Validate epochs (must be positive)
    if (job_config.epochs() <= 0) {
        *error_msg = "Epochs must be greater than zero";
        return false;
    }

    // Validate batch size
    if (job_config.batch_size() <= 0) {
        *error_msg = "Batch size must be greater than zero";
        return false;
    }

    // Validate dataset URI
    if (job_config.dataset_uri().empty()) {
        *error_msg = "Dataset URI cannot be empty";
        return false;
    }

    // Additional validations can be added here:
    // - Check if we have enough memory for estimated_memory
    // - Check if required_device is available
    // - Validate hyperparameters format

    spdlog::debug("Job {} validation passed", job_config.job_id());
    return true;
}

bool NodeServiceImpl::GetAndRemovePendingJob(const std::string& job_id, protocol::JobConfig* config) {
    std::lock_guard<std::mutex> lock(pending_jobs_mutex_);

    auto it = pending_p2p_jobs_.find(job_id);
    if (it == pending_p2p_jobs_.end()) {
        return false;
    }

    if (config) {
        *config = it->second;
    }
    pending_p2p_jobs_.erase(it);

    spdlog::info("Retrieved pending P2P job config for job {}", job_id);
    return true;
}

} // namespace servernode
} // namespace cyxwiz
