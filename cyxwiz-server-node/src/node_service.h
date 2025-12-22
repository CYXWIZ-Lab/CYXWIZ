#pragma once

#include <grpcpp/grpcpp.h>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <chrono>
#include "node.grpc.pb.h"
#include "job_executor.h"

namespace cyxwiz {
namespace servernode {

/**
 * @brief NodeService implementation for Server Node
 *
 * Handles incoming RPCs from Central Server:
 * - AssignJob: Receive job assignments
 * - (Future) Other node management RPCs
 *
 * This service runs on the Server Node and is called BY the Central Server.
 */
class NodeServiceImpl final : public protocol::NodeService::Service {
public:
    /**
     * @brief Construct a new Node Service Impl object
     *
     * @param job_executor Pointer to JobExecutor for executing assigned jobs
     * @param node_id This node's unique identifier
     */
    explicit NodeServiceImpl(JobExecutor* job_executor, const std::string& node_id);

    /**
     * @brief Handle job assignment from Central Server
     *
     * This RPC is called when the Central Server assigns a job to this node.
     * The implementation validates the job, passes it to JobExecutor, and
     * returns whether the job was accepted.
     *
     * @param context gRPC server context
     * @param request Job assignment request containing JobConfig
     * @param response Response indicating acceptance/rejection
     * @return grpc::Status OK if RPC succeeded, error status otherwise
     */
    grpc::Status AssignJob(
        grpc::ServerContext* context,
        const protocol::AssignJobRequest* request,
        protocol::AssignJobResponse* response) override;

    /**
     * @brief Get current node metrics
     *
     * Future implementation for reporting node health and status.
     *
     * @param context gRPC server context
     * @param request Metrics request
     * @param response Metrics data
     * @return grpc::Status
     */
    grpc::Status GetNodeMetrics(
        grpc::ServerContext* context,
        const protocol::GetNodeMetricsRequest* request,
        protocol::GetNodeMetricsResponse* response) override;

    /**
     * @brief Get pending P2P job config (for remote dataset jobs)
     *
     * When a job with remote:// dataset URI is assigned, it's stored here
     * until the Engine connects via P2P. The job_execution_service can
     * retrieve and remove the config when P2P connection arrives.
     *
     * @param job_id Job ID to look up
     * @param config Output parameter for job config
     * @return true if job was found and removed from pending
     */
    bool GetAndRemovePendingJob(const std::string& job_id, protocol::JobConfig* config);

    /**
     * @brief Clear all pending P2P jobs
     *
     * Called when Central Server connection is lost to cleanup stale jobs.
     */
    void ClearAllPendingJobs();

private:
    /**
     * @brief Validate job configuration before acceptance
     *
     * Checks if:
     * - Job ID is valid
     * - Hyperparameters are parseable
     * - Resource requirements can be met
     *
     * @param job_config Job configuration to validate
     * @param error_msg Output parameter for error details
     * @return true if job is valid and can be executed
     */
    bool ValidateJobConfig(const protocol::JobConfig& job_config, std::string* error_msg);

    /// Clean up pending jobs that have exceeded the timeout
    void CleanupExpiredPendingJobs();

    JobExecutor* job_executor_;  ///< Non-owning pointer to JobExecutor
    std::string node_id_;         ///< This node's unique ID

    /// Pending job with timestamp for timeout tracking
    struct PendingJob {
        protocol::JobConfig config;
        std::chrono::steady_clock::time_point registered_at;
    };

    /// Jobs with remote datasets waiting for P2P connection (timeout: 10 seconds)
    std::map<std::string, PendingJob> pending_p2p_jobs_;
    mutable std::mutex pending_jobs_mutex_;
    static constexpr int PENDING_JOB_TIMEOUT_SECONDS = 10;
};

} // namespace servernode
} // namespace cyxwiz
