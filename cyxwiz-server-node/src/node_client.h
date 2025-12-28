#pragma once

#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <functional>
#include <grpcpp/grpcpp.h>
#include "node.pb.h"
#include "node.grpc.pb.h"
#include "common.pb.h"
#include "reservation.pb.h"
#include "reservation.grpc.pb.h"

namespace cyxwiz {
namespace servernode {

// Device allocation information from GUI
// Used to specify which devices the user wants to share and their resource limits
struct DeviceAllocation {
    int device_type;           // 0=CPU, 1=CUDA, 2=OpenCL (matches protocol::DeviceType)
    int device_id;             // Device index (0, 1, 2, etc.)
    std::string device_name;   // Human-readable name (e.g., "NVIDIA RTX 3080")
    bool is_enabled;           // Whether user enabled this device for sharing
    uint64_t vram_total_mb;    // Total VRAM in MB (GPU only)
    uint64_t vram_allocated_mb; // VRAM allocated for sharing in MB
    int cores_allocated;       // CPU cores allocated (CPU only)
    uint64_t memory_total;     // Total memory in bytes
    int compute_units;         // Compute units (GPU cores)
};

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

    // Register this node with Central Server (auto-detects hardware)
    bool Register();

    // Register this node with specific device allocations from GUI
    // Only enabled devices from allocations will be sent to Central Server
    bool RegisterWithAllocations(const std::vector<DeviceAllocation>& allocations);

    // Start heartbeat loop (background thread)
    bool StartHeartbeat(int interval_seconds = 10);

    // Stop heartbeat loop
    void StopHeartbeat();

    // Disconnect from Central Server (stops heartbeat and clears registration)
    void Disconnect();

    // Send single heartbeat
    bool SendHeartbeat();

    // Update active jobs list
    void SetActiveJobs(const std::vector<std::string>& job_ids);

    // Get assigned node ID from server
    std::string GetNodeId() const { return node_id_; }

    // Check if successfully registered
    bool IsRegistered() const { return is_registered_; }

    // Set authentication token (JWT from user login)
    void SetAuthToken(const std::string& token) { auth_token_ = token; }
    void ClearAuthToken() { auth_token_.clear(); }
    bool HasAuthToken() const { return !auth_token_.empty(); }

    // Callback for when Central Server connection is lost
    using ConnectionLostCallback = std::function<void()>;
    void SetConnectionLostCallback(ConnectionLostCallback callback) { connection_lost_callback_ = callback; }

    // Callback for authentication failures (token invalid/revoked)
    // This is called when heartbeat fails due to auth error - user needs to re-login
    using AuthFailedCallback = std::function<void(const std::string& reason)>;
    void SetAuthFailedCallback(AuthFailedCallback callback) { auth_failed_callback_ = callback; }

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

    // ========================================================================
    // Reservation-Based Payment System (Server Node → Central Server)
    // ========================================================================

    // Report job completion WITHIN reservation (reputation only, no payment)
    // Called when a training job finishes but reservation timer is still running
    bool ReportJobCompleteFromNode(
        const std::string& reservation_id,
        const std::string& job_id,
        bool success,
        const std::string& model_hash = "",
        const std::map<std::string, double>& final_metrics = {},
        int64_t training_time_seconds = 0,
        int32_t epochs_completed = 0
    );

    // Report reservation timer expired (triggers payment release)
    // Called when the countdown timer reaches 0
    bool ReportReservationEndFromNode(
        const std::string& reservation_id,
        int32_t jobs_completed,
        int64_t total_compute_time = 0,
        bool node_available = true
    );

private:
    void HeartbeatLoop();
    void AddAuthMetadata(grpc::ClientContext& context);

    std::string central_server_address_;
    std::string node_id_;
    std::string session_token_;
    std::string auth_token_;  // JWT token for authentication
    bool is_registered_;

    std::unique_ptr<protocol::NodeService::Stub> stub_;
    std::unique_ptr<protocol::JobStatusService::Stub> job_status_stub_;
    std::unique_ptr<protocol::JobReservationService::Stub> reservation_stub_;
    std::shared_ptr<grpc::Channel> channel_;

    // Heartbeat thread
    std::thread heartbeat_thread_;
    std::atomic<bool> should_stop_heartbeat_{false};
    int heartbeat_interval_seconds_;

    // Current status
    std::vector<std::string> active_jobs_;
    std::mutex jobs_mutex_;

    // Callback for connection lost events
    ConnectionLostCallback connection_lost_callback_;

    // Callback for authentication failures
    AuthFailedCallback auth_failed_callback_;
    std::atomic<bool> auth_failed_{false};  // Prevent repeated callbacks
};

} // namespace servernode
} // namespace cyxwiz
