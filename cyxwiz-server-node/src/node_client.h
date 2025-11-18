#pragma once

#include <memory>
#include <string>
#include <thread>
#include <atomic>
#include <grpcpp/grpcpp.h>
#include "node.pb.h"
#include "node.grpc.pb.h"
#include "common.pb.h"

namespace cyxwiz {
namespace servernode {

// Hardware information collector
class HardwareDetector {
public:
    static protocol::NodeInfo DetectHardwareInfo(const std::string& node_id);
    static int64_t GetAvailableRAM();  // Public - needed for heartbeat updates

private:
    static int GetCPUCores();
    static int64_t GetTotalRAM();
    static void DetectDevices(protocol::NodeInfo* node_info);
    static std::string GetLocalIPAddress();
};

// gRPC client for communicating with Central Server
class NodeClient {
public:
    NodeClient(const std::string& central_server_address, const std::string& node_id);
    ~NodeClient();

    // Register this node with Central Server
    bool Register();

    // Start heartbeat loop (background thread)
    bool StartHeartbeat(int interval_seconds = 10);

    // Stop heartbeat loop
    void StopHeartbeat();

    // Send single heartbeat
    bool SendHeartbeat();

    // Update active jobs list
    void SetActiveJobs(const std::vector<std::string>& job_ids);

    // Get assigned node ID from server
    std::string GetNodeId() const { return node_id_; }

    // Check if successfully registered
    bool IsRegistered() const { return is_registered_; }

    // ========================================================================
    // Job Status Reporting (Server Node → Central Server)
    // ========================================================================

    // Update job progress during training
    bool UpdateJobStatus(
        const std::string& job_id,
        protocol::StatusCode status,
        double progress,                              // 0.0 to 1.0
        const std::map<std::string, double>& metrics, // loss, accuracy, etc.
        int32_t current_epoch,
        const std::string& log_message = ""
    );

    // Report final job result (completion or failure)
    bool ReportJobResult(
        const std::string& job_id,
        protocol::StatusCode final_status,            // STATUS_SUCCESS or STATUS_FAILED
        const std::map<std::string, double>& final_metrics,
        const std::string& model_weights_uri = "",
        const std::string& model_weights_hash = "",
        int64_t model_size = 0,
        int64_t total_compute_time_ms = 0,
        const std::string& error_message = ""
    );

private:
    void HeartbeatLoop();

    std::string central_server_address_;
    std::string node_id_;
    std::string session_token_;
    bool is_registered_;

    std::unique_ptr<protocol::NodeService::Stub> stub_;
    std::unique_ptr<protocol::JobStatusService::Stub> job_status_stub_;
    std::shared_ptr<grpc::Channel> channel_;

    // Heartbeat thread
    std::thread heartbeat_thread_;
    std::atomic<bool> should_stop_heartbeat_{false};
    int heartbeat_interval_seconds_;

    // Current status
    std::vector<std::string> active_jobs_;
    std::mutex jobs_mutex_;
};

} // namespace servernode
} // namespace cyxwiz
