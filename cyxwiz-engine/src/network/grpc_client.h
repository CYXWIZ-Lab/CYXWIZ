#pragma once
#include <string>
#include <memory>
#include <vector>
#include <grpcpp/grpcpp.h>
#include "job.grpc.pb.h"
#include "node.grpc.pb.h"
#include "common.pb.h"

namespace network {

// Simplified NodeInfo for UI display
struct NodeDisplayInfo {
    std::string node_id;
    std::string name;
    std::string region;
    bool is_online = false;

    // Network endpoint
    std::string ip_address;
    int port = 50052;  // Default P2P port

    // Hardware
    std::string device_type;  // "CUDA", "OpenCL", "CPU"
    int64_t vram_bytes = 0;
    int cpu_cores = 0;

    // Performance
    double compute_score = 0.0;
    double reputation_score = 0.0;
    int64_t total_jobs_completed = 0;

    // Pricing
    double price_per_hour = 0.0;
    double price_usd_equivalent = 0.0;
    std::string billing_model;  // "Hourly", "Per Epoch", etc.
    bool free_tier_available = false;

    // Staking
    double staked_amount = 0.0;
};

// Search criteria for FindNodes
struct NodeSearchCriteria {
    // Hardware requirements
    std::string required_device;  // "", "CUDA", "OpenCL", "CPU"
    int64_t min_vram = 0;
    double min_compute_score = 0.0;

    // Trust requirements
    double min_reputation = 0.0;
    double min_stake = 0.0;

    // Pricing
    double max_price_per_hour = 0.0;  // 0 = no limit
    bool require_free_tier = false;

    // Location
    std::string preferred_region;

    // Results
    int max_results = 50;

    // Sorting: 0=price, 1=performance, 2=reputation, 3=availability
    int sort_by = 0;
};

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

    bool DeleteJob(const std::string& job_id,
                    cyxwiz::protocol::DeleteJobResponse& response);

    bool ListJobs(cyxwiz::protocol::ListJobsResponse& response,
                  const std::string& user_id = "",
                  int page_size = 100);

    // Node discovery operations
    bool ListNodes(std::vector<NodeDisplayInfo>& out_nodes,
                   bool online_only = true,
                   int page_size = 50);

    bool FindNodes(const NodeSearchCriteria& criteria,
                   std::vector<NodeDisplayInfo>& out_nodes);

    bool GetNodeInfo(const std::string& node_id,
                     NodeDisplayInfo& out_node);

    // Get last error message
    std::string GetLastError() const { return last_error_; }

private:
    // Helper to convert proto NodeInfo to display struct
    static NodeDisplayInfo ConvertNodeInfo(const cyxwiz::protocol::NodeInfo& proto);
    // Add authorization header to gRPC context
    void AddAuthMetadata(grpc::ClientContext& context);

    bool connected_;
    std::string server_address_;
    std::string last_error_;
    std::string auth_token_;  // JWT token for authentication

    std::shared_ptr<grpc::Channel> channel_;
    std::unique_ptr<cyxwiz::protocol::JobService::Stub> job_stub_;
    std::unique_ptr<cyxwiz::protocol::NodeDiscoveryService::Stub> node_discovery_stub_;
};

} // namespace network
