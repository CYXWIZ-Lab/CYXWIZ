#pragma once

#include <grpcpp/grpcpp.h>
#include <memory>
#include <string>
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

    JobExecutor* job_executor_;  ///< Non-owning pointer to JobExecutor
    std::string node_id_;         ///< This node's unique ID
};

} // namespace servernode
} // namespace cyxwiz
