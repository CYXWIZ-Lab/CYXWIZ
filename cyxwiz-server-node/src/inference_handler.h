// inference_handler.h - gRPC InferenceService implementation
#pragma once

#include <grpcpp/grpcpp.h>
#include "inference.grpc.pb.h"
#include <memory>
#include <string>
#include <atomic>

namespace cyxwiz::servernode {

class DeploymentManager;

// gRPC InferenceService implementation
class InferenceServiceImpl final : public protocol::InferenceService::Service {
public:
    explicit InferenceServiceImpl(DeploymentManager* manager);
    ~InferenceServiceImpl() override = default;

    // Run inference on a deployed model
    grpc::Status Infer(
        grpc::ServerContext* context,
        const protocol::InferRequest* request,
        protocol::InferResponse* response) override;

    // Get model input/output specifications
    grpc::Status GetModelInfo(
        grpc::ServerContext* context,
        const protocol::GetModelInfoRequest* request,
        protocol::GetModelInfoResponse* response) override;

private:
    DeploymentManager* manager_;
};

// Server wrapper for easy lifecycle management
class InferenceServer {
public:
    InferenceServer(const std::string& address, DeploymentManager* manager);
    ~InferenceServer();

    // Start the gRPC server (non-blocking)
    bool Start();

    // Stop the server
    void Stop();

    // Check if running
    bool IsRunning() const { return running_; }

    // Get address
    const std::string& GetAddress() const { return address_; }

private:
    std::string address_;
    DeploymentManager* manager_;
    std::unique_ptr<InferenceServiceImpl> service_;
    std::unique_ptr<grpc::Server> server_;
    std::atomic<bool> running_{false};
};

} // namespace cyxwiz::servernode
