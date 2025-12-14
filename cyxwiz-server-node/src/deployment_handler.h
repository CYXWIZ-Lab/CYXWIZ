#pragma once

#include <memory>
#include <string>
#include <grpcpp/grpcpp.h>
#include "deployment.pb.h"
#include "deployment.grpc.pb.h"
#include "deployment_manager.h"

namespace cyxwiz {
namespace servernode {

// gRPC service implementation for receiving deployment assignments from Central Server
class DeploymentServiceImpl final : public protocol::DeploymentService::Service {
public:
    explicit DeploymentServiceImpl(std::shared_ptr<DeploymentManager> manager);
    ~DeploymentServiceImpl() override = default;

    // Called by Central Server to create a deployment on this node
    grpc::Status CreateDeployment(
        grpc::ServerContext* context,
        const protocol::CreateDeploymentRequest* request,
        protocol::CreateDeploymentResponse* response) override;

    // Called by Central Server to stop a running deployment
    grpc::Status StopDeployment(
        grpc::ServerContext* context,
        const protocol::StopDeploymentRequest* request,
        protocol::StopDeploymentResponse* response) override;

    // Called by Central Server to query deployment status
    grpc::Status GetDeployment(
        grpc::ServerContext* context,
        const protocol::GetDeploymentRequest* request,
        protocol::GetDeploymentResponse* response) override;

    // Called by Central Server to get deployment metrics
    grpc::Status GetDeploymentMetrics(
        grpc::ServerContext* context,
        const protocol::GetDeploymentMetricsRequest* request,
        protocol::GetDeploymentMetricsResponse* response) override;

    // Called by Engine/Central Server to list all deployments on this node
    grpc::Status ListDeployments(
        grpc::ServerContext* context,
        const protocol::ListDeploymentsRequest* request,
        protocol::ListDeploymentsResponse* response) override;

private:
    std::shared_ptr<DeploymentManager> manager_;
};

// Main handler class that manages the gRPC server lifecycle
class DeploymentHandler {
public:
    DeploymentHandler(const std::string& listen_address,
                     std::shared_ptr<DeploymentManager> manager);
    ~DeploymentHandler();

    // Start the gRPC server (blocking call)
    bool Start();

    // Stop the gRPC server gracefully
    void Stop();

    // Check if server is running
    bool IsRunning() const { return running_; }

private:
    std::string listen_address_;
    std::shared_ptr<DeploymentManager> manager_;
    std::unique_ptr<grpc::Server> server_;
    std::unique_ptr<DeploymentServiceImpl> service_;
    bool running_;
};

} // namespace servernode
} // namespace cyxwiz
