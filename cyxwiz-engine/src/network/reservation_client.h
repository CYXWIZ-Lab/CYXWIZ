#pragma once

#include <string>
#include <memory>
#include <functional>
#include <thread>
#include <atomic>
#include <chrono>
#include <grpcpp/grpcpp.h>
#include "reservation.grpc.pb.h"
#include "job.pb.h"

namespace network {

// Reservation information returned from ReserveNode
struct ReservationInfo {
    std::string reservation_id;
    std::string job_id;
    std::string node_endpoint;     // IP:port for P2P connection
    std::string p2p_auth_token;    // JWT for P2P authentication
    int64_t p2p_token_expires;     // Unix timestamp
    std::string escrow_account;    // Solana escrow PDA
    int64_t escrow_amount;         // Amount in lamports
    int64_t start_time;            // Unix timestamp
    int64_t end_time;              // Unix timestamp
    double price_per_hour;         // CYXWIZ tokens
};

// Callback types
using ReservationCallback = std::function<void(bool success, const ReservationInfo& info, const std::string& error)>;
using HeartbeatCallback = std::function<void(int64_t time_remaining, bool should_extend)>;
using ReleaseCallback = std::function<void(bool success, int64_t payment_released, int64_t refund_amount)>;

/**
 * ReservationClient - Manages node reservations with Central Server
 *
 * This client handles the reservation workflow:
 * 1. ReserveNode - Reserve a node for a duration, creates escrow
 * 2. EngineHeartbeat - Send periodic heartbeats during reservation
 * 3. ConfirmJobComplete - Confirm job completion to release payment
 * 4. ReleaseReservation - Release early with proportional refund
 *
 * Usage:
 *   ReservationClient client;
 *   client.Connect("localhost:50051");
 *   auto info = client.ReserveNode(node_id, wallet, duration_minutes, job_config);
 *   if (info.has_value()) {
 *       // Connect to node via P2P
 *       client.StartHeartbeat(info->reservation_id);
 *       // ... training ...
 *       client.ConfirmJobComplete(info->reservation_id, model_hash, metrics);
 *   }
 */
class ReservationClient {
public:
    ReservationClient();
    ~ReservationClient();

    // Connection management
    bool Connect(const std::string& server_address);
    void Disconnect();
    bool IsConnected() const { return connected_; }

    // Authentication
    void SetAuthToken(const std::string& token) { auth_token_ = token; }
    void ClearAuthToken() { auth_token_.clear(); }
    bool HasAuthToken() const { return !auth_token_.empty(); }

    // Reserve a node for training
    // Returns nullopt on failure, reservation info on success
    std::optional<ReservationInfo> ReserveNode(
        const std::string& node_id,
        const std::string& user_wallet,
        int32_t duration_minutes,
        const cyxwiz::protocol::JobConfig& job_config);

    // Async version with callback
    void ReserveNodeAsync(
        const std::string& node_id,
        const std::string& user_wallet,
        int32_t duration_minutes,
        const cyxwiz::protocol::JobConfig& job_config,
        ReservationCallback callback);

    // Confirm job completion (releases payment to node)
    bool ConfirmJobComplete(
        const std::string& reservation_id,
        const std::string& job_id,
        const std::string& model_hash,
        const std::map<std::string, double>& final_metrics,
        bool success = true);

    // Extend reservation time
    bool ExtendReservation(
        const std::string& reservation_id,
        int32_t additional_minutes,
        int64_t& new_expires,
        int64_t& additional_escrow);

    // Release reservation early (proportional refund)
    bool ReleaseReservation(
        const std::string& reservation_id,
        const std::string& reason,
        int64_t& time_used,
        int64_t& payment_released,
        int64_t& refund_amount);

    // Get reservation details
    bool GetReservation(
        const std::string& reservation_id,
        cyxwiz::protocol::ReservationInfo& out_info);

    // Heartbeat management
    void StartHeartbeat(
        const std::string& reservation_id,
        const std::string& job_id = "",
        int interval_seconds = 30);

    void StopHeartbeat();
    bool IsHeartbeatActive() const { return heartbeat_active_; }

    // Update heartbeat with current job state
    void UpdateHeartbeatState(
        const std::string& job_id,
        bool training_active,
        double progress,
        const std::map<std::string, double>& metrics);

    // Set callback for heartbeat responses
    void SetHeartbeatCallback(HeartbeatCallback callback) { heartbeat_callback_ = callback; }

    // Get last error
    std::string GetLastError() const { return last_error_; }

    // Get current reservation
    const ReservationInfo& GetCurrentReservation() const { return current_reservation_; }
    bool HasActiveReservation() const { return !current_reservation_.reservation_id.empty(); }

private:
    // Add authorization header to gRPC context
    void AddAuthMetadata(grpc::ClientContext& context);

    // Heartbeat thread function
    void HeartbeatThreadFunc();

    // Send a single heartbeat
    bool SendHeartbeat();

    // Connection state
    bool connected_;
    std::string server_address_;
    std::string last_error_;
    std::string auth_token_;

    // gRPC
    std::shared_ptr<grpc::Channel> channel_;
    std::unique_ptr<cyxwiz::protocol::JobReservationService::Stub> stub_;

    // Current reservation
    ReservationInfo current_reservation_;

    // Heartbeat state
    std::atomic<bool> heartbeat_active_;
    std::thread heartbeat_thread_;
    std::string heartbeat_reservation_id_;
    std::string heartbeat_job_id_;
    std::atomic<bool> heartbeat_training_active_;
    std::atomic<double> heartbeat_progress_;
    std::map<std::string, double> heartbeat_metrics_;
    std::mutex heartbeat_metrics_mutex_;
    int heartbeat_interval_seconds_;
    HeartbeatCallback heartbeat_callback_;
};

} // namespace network
