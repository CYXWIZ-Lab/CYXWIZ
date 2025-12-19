#pragma once
#include <string>
#include <memory>
#include <grpcpp/grpcpp.h>
#include "job.grpc.pb.h"
#include "common.pb.h"

namespace network {

class GRPCClient {
public:
    GRPCClient();
    ~GRPCClient();

    // Connection management
    bool Connect(const std::string& server_address);
    void Disconnect();
    bool IsConnected() const { return connected_; }
    std::string GetServerAddress() const { return server_address_; }

    // Authentication
    void SetAuthToken(const std::string& token) { auth_token_ = token; }
    void ClearAuthToken() { auth_token_.clear(); }
    bool HasAuthToken() const { return !auth_token_.empty(); }

    // Job operations
    bool SubmitJob(const cyxwiz::protocol::SubmitJobRequest& request,
                   cyxwiz::protocol::SubmitJobResponse& response);

    bool GetJobStatus(const std::string& job_id,
                      cyxwiz::protocol::GetJobStatusResponse& response);

    bool CancelJob(const std::string& job_id,
                    cyxwiz::protocol::CancelJobResponse& response);

    bool ListJobs(cyxwiz::protocol::ListJobsResponse& response,
                  const std::string& user_id = "",
                  int page_size = 100);

    // Get last error message
    std::string GetLastError() const { return last_error_; }

private:
    // Add authorization header to gRPC context
    void AddAuthMetadata(grpc::ClientContext& context);

    bool connected_;
    std::string server_address_;
    std::string last_error_;
    std::string auth_token_;  // JWT token for authentication

    std::shared_ptr<grpc::Channel> channel_;
    std::unique_ptr<cyxwiz::protocol::JobService::Stub> job_stub_;
};

} // namespace network
