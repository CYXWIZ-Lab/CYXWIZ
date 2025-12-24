#include "reservation_client.h"
#include "../auth/auth_client.h"
#include <spdlog/spdlog.h>

namespace network {

ReservationClient::ReservationClient()
    : connected_(false)
    , heartbeat_active_(false)
    , heartbeat_training_active_(false)
    , heartbeat_progress_(0.0)
    , heartbeat_interval_seconds_(30)
{
}

ReservationClient::~ReservationClient() {
    StopHeartbeat();
    Disconnect();
}

bool ReservationClient::Connect(const std::string& server_address) {
    if (connected_) {
        Disconnect();
    }

    try {
        channel_ = grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
        stub_ = cyxwiz::protocol::JobReservationService::NewStub(channel_);

        // Test connection with a short deadline
        auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(5);
        if (!channel_->WaitForConnected(deadline)) {
            last_error_ = "Connection timeout";
            return false;
        }

        connected_ = true;
        server_address_ = server_address;
        spdlog::info("ReservationClient connected to {}", server_address);
        return true;
    } catch (const std::exception& e) {
        last_error_ = e.what();
        spdlog::error("ReservationClient connection failed: {}", e.what());
        return false;
    }
}

void ReservationClient::Disconnect() {
    StopHeartbeat();
    connected_ = false;
    stub_.reset();
    channel_.reset();
    current_reservation_ = ReservationInfo{};
}

void ReservationClient::AddAuthMetadata(grpc::ClientContext& context) {
    // Get fresh token from AuthClient (in case user logged in after startup)
    auto& auth = cyxwiz::auth::AuthClient::Instance();
    std::string token = auth.IsAuthenticated() ? auth.GetJwtToken() : auth_token_;

    if (!token.empty()) {
        context.AddMetadata("authorization", "Bearer " + token);
        spdlog::debug("Added auth token to ReservationClient request");
    } else {
        spdlog::warn("No auth token available for ReservationClient request");
    }
}

std::optional<ReservationInfo> ReservationClient::ReserveNode(
    const std::string& node_id,
    const std::string& user_wallet,
    int32_t duration_minutes,
    const cyxwiz::protocol::JobConfig& job_config) {

    if (!connected_ || !stub_) {
        last_error_ = "Not connected";
        return std::nullopt;
    }

    spdlog::info("Reserving node {} for {} minutes...", node_id, duration_minutes);

    grpc::ClientContext context;
    AddAuthMetadata(context);
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(30));

    cyxwiz::protocol::ReserveNodeRequest request;
    request.set_node_id(node_id);
    request.set_user_wallet(user_wallet);
    request.set_duration_minutes(duration_minutes);
    *request.mutable_job_config() = job_config;

    cyxwiz::protocol::ReserveNodeResponse response;
    grpc::Status status = stub_->ReserveNode(&context, request, &response);

    if (!status.ok()) {
        last_error_ = "gRPC error: " + status.error_message();
        spdlog::error("ReserveNode failed: {}", last_error_);
        return std::nullopt;
    }

    if (response.status() != cyxwiz::protocol::STATUS_SUCCESS) {
        last_error_ = response.error().message();
        spdlog::error("ReserveNode rejected: {}", last_error_);
        return std::nullopt;
    }

    // Build reservation info
    ReservationInfo info;
    info.reservation_id = response.reservation_id();
    info.job_id = response.job_id();
    info.node_endpoint = response.node_endpoint();
    info.p2p_auth_token = response.p2p_auth_token();
    info.p2p_token_expires = response.p2p_token_expires();
    info.escrow_account = response.escrow_account();
    info.escrow_amount = response.escrow_amount();
    info.start_time = response.reservation_start();
    info.end_time = response.reservation_expires();
    info.price_per_hour = response.price_per_hour();

    current_reservation_ = info;

    spdlog::info("Node reserved successfully!");
    spdlog::info("  Reservation ID: {}", info.reservation_id);
    spdlog::info("  Node endpoint: {}", info.node_endpoint);
    spdlog::info("  Duration: {} minutes", duration_minutes);
    spdlog::info("  Escrow: {} lamports", info.escrow_amount);

    return info;
}

void ReservationClient::ReserveNodeAsync(
    const std::string& node_id,
    const std::string& user_wallet,
    int32_t duration_minutes,
    const cyxwiz::protocol::JobConfig& job_config,
    ReservationCallback callback) {

    std::thread([this, node_id, user_wallet, duration_minutes, job_config, callback]() {
        auto result = ReserveNode(node_id, user_wallet, duration_minutes, job_config);
        if (result.has_value()) {
            callback(true, result.value(), "");
        } else {
            callback(false, ReservationInfo{}, last_error_);
        }
    }).detach();
}

bool ReservationClient::ConfirmJobComplete(
    const std::string& reservation_id,
    const std::string& job_id,
    const std::string& model_hash,
    const std::map<std::string, double>& final_metrics,
    bool success) {

    if (!connected_ || !stub_) {
        last_error_ = "Not connected";
        return false;
    }

    spdlog::info("Confirming job completion for reservation {}...", reservation_id);

    grpc::ClientContext context;
    AddAuthMetadata(context);
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(30));

    cyxwiz::protocol::ConfirmJobCompleteRequest request;
    request.set_reservation_id(reservation_id);
    request.set_job_id(job_id);
    request.set_model_hash(model_hash);
    request.set_success(success);
    for (const auto& [key, value] : final_metrics) {
        (*request.mutable_final_metrics())[key] = value;
    }

    cyxwiz::protocol::ConfirmJobCompleteResponse response;
    grpc::Status status = stub_->ConfirmJobComplete(&context, request, &response);

    if (!status.ok()) {
        last_error_ = "gRPC error: " + status.error_message();
        spdlog::error("ConfirmJobComplete failed: {}", last_error_);
        return false;
    }

    if (response.status() != cyxwiz::protocol::STATUS_SUCCESS) {
        last_error_ = response.error().message();
        spdlog::error("ConfirmJobComplete rejected: {}", last_error_);
        return false;
    }

    spdlog::info("Job completion confirmed! Payment released: {}", response.payment_released());
    if (!response.payment_tx_hash().empty()) {
        spdlog::info("  Transaction: {}", response.payment_tx_hash());
    }

    // Clear current reservation
    current_reservation_ = ReservationInfo{};
    StopHeartbeat();

    return true;
}

bool ReservationClient::ExtendReservation(
    const std::string& reservation_id,
    int32_t additional_minutes,
    int64_t& new_expires,
    int64_t& additional_escrow) {

    if (!connected_ || !stub_) {
        last_error_ = "Not connected";
        return false;
    }

    spdlog::info("Extending reservation {} by {} minutes...", reservation_id, additional_minutes);

    grpc::ClientContext context;
    AddAuthMetadata(context);
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(30));

    cyxwiz::protocol::ExtendReservationRequest request;
    request.set_reservation_id(reservation_id);
    request.set_additional_minutes(additional_minutes);

    cyxwiz::protocol::ExtendReservationResponse response;
    grpc::Status status = stub_->ExtendReservation(&context, request, &response);

    if (!status.ok()) {
        last_error_ = "gRPC error: " + status.error_message();
        spdlog::error("ExtendReservation failed: {}", last_error_);
        return false;
    }

    if (response.status() != cyxwiz::protocol::STATUS_SUCCESS) {
        last_error_ = response.error().message();
        spdlog::error("ExtendReservation rejected: {}", last_error_);
        return false;
    }

    new_expires = response.new_expires();
    additional_escrow = response.additional_escrow();

    // Update current reservation
    current_reservation_.end_time = new_expires;
    current_reservation_.escrow_amount += additional_escrow;

    spdlog::info("Reservation extended! New end time: {}", new_expires);
    return true;
}

bool ReservationClient::ReleaseReservation(
    const std::string& reservation_id,
    const std::string& reason,
    int64_t& time_used,
    int64_t& payment_released,
    int64_t& refund_amount) {

    if (!connected_ || !stub_) {
        last_error_ = "Not connected";
        return false;
    }

    spdlog::info("Releasing reservation {} early...", reservation_id);

    grpc::ClientContext context;
    AddAuthMetadata(context);
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(30));

    cyxwiz::protocol::ReleaseReservationRequest request;
    request.set_reservation_id(reservation_id);
    request.set_reason(reason);

    cyxwiz::protocol::ReleaseReservationResponse response;
    grpc::Status status = stub_->ReleaseReservation(&context, request, &response);

    if (!status.ok()) {
        last_error_ = "gRPC error: " + status.error_message();
        spdlog::error("ReleaseReservation failed: {}", last_error_);
        return false;
    }

    if (response.status() != cyxwiz::protocol::STATUS_SUCCESS) {
        last_error_ = response.error().message();
        spdlog::error("ReleaseReservation rejected: {}", last_error_);
        return false;
    }

    time_used = response.time_used_seconds();
    payment_released = response.payment_released();
    refund_amount = response.refund_amount();

    spdlog::info("Reservation released!");
    spdlog::info("  Time used: {} seconds", time_used);
    spdlog::info("  Payment to node: {} lamports", payment_released);
    spdlog::info("  Refund: {} lamports", refund_amount);

    // Clear current reservation
    current_reservation_ = ReservationInfo{};
    StopHeartbeat();

    return true;
}

bool ReservationClient::GetReservation(
    const std::string& reservation_id,
    cyxwiz::protocol::ReservationInfo& out_info) {

    if (!connected_ || !stub_) {
        last_error_ = "Not connected";
        return false;
    }

    grpc::ClientContext context;
    AddAuthMetadata(context);
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));

    cyxwiz::protocol::GetReservationRequest request;
    request.set_reservation_id(reservation_id);

    cyxwiz::protocol::GetReservationResponse response;
    grpc::Status status = stub_->GetReservation(&context, request, &response);

    if (!status.ok()) {
        last_error_ = "gRPC error: " + status.error_message();
        return false;
    }

    if (response.status() != cyxwiz::protocol::STATUS_SUCCESS) {
        last_error_ = response.error().message();
        return false;
    }

    out_info = response.reservation();
    return true;
}

bool ReservationClient::GetActiveReservations(
    const std::string& user_wallet,
    std::vector<cyxwiz::protocol::ActiveReservationInfo>& out_reservations) {

    if (!connected_ || !stub_) {
        last_error_ = "Not connected";
        return false;
    }

    spdlog::info("Checking for active reservations for wallet: {}", user_wallet);

    grpc::ClientContext context;
    AddAuthMetadata(context);
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));

    cyxwiz::protocol::GetActiveReservationsRequest request;
    request.set_user_wallet(user_wallet);

    cyxwiz::protocol::GetActiveReservationsResponse response;
    grpc::Status status = stub_->GetActiveReservations(&context, request, &response);

    if (!status.ok()) {
        last_error_ = "gRPC error: " + status.error_message();
        spdlog::error("GetActiveReservations failed: {}", last_error_);
        return false;
    }

    if (response.status() != cyxwiz::protocol::STATUS_SUCCESS) {
        last_error_ = response.error().message();
        spdlog::warn("GetActiveReservations: {}", last_error_);
        return false;
    }

    out_reservations.clear();
    for (const auto& res : response.reservations()) {
        out_reservations.push_back(res);
        spdlog::info("Found active reservation: {} (node: {}, time left: {}s)",
                     res.reservation_id(), res.node_id(), res.time_remaining_seconds());
    }

    spdlog::info("Found {} active reservation(s)", out_reservations.size());
    return true;
}

bool ReservationClient::GetReconnectionToken(
    const std::string& reservation_id,
    const std::string& user_wallet,
    std::string& out_p2p_token,
    int64_t& out_token_expires,
    std::string& out_node_endpoint,
    int64_t& out_time_remaining) {

    if (!connected_ || !stub_) {
        last_error_ = "Not connected";
        return false;
    }

    spdlog::info("Getting reconnection token for reservation: {}", reservation_id);

    grpc::ClientContext context;
    AddAuthMetadata(context);
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));

    cyxwiz::protocol::GetReconnectionTokenRequest request;
    request.set_reservation_id(reservation_id);
    request.set_user_wallet(user_wallet);

    cyxwiz::protocol::GetReconnectionTokenResponse response;
    grpc::Status status = stub_->GetReconnectionToken(&context, request, &response);

    if (!status.ok()) {
        last_error_ = "gRPC error: " + status.error_message();
        spdlog::error("GetReconnectionToken failed: {}", last_error_);
        return false;
    }

    if (response.status() != cyxwiz::protocol::STATUS_SUCCESS) {
        last_error_ = response.error().message();
        spdlog::error("GetReconnectionToken rejected: {}", last_error_);
        return false;
    }

    out_p2p_token = response.p2p_auth_token();
    out_token_expires = response.p2p_token_expires();
    out_node_endpoint = response.node_endpoint();
    out_time_remaining = response.time_remaining_seconds();

    // Update current reservation with reconnection info
    current_reservation_.reservation_id = reservation_id;
    current_reservation_.node_endpoint = out_node_endpoint;
    current_reservation_.p2p_auth_token = out_p2p_token;
    current_reservation_.p2p_token_expires = out_token_expires;

    spdlog::info("Reconnection token obtained! Endpoint: {}, Time remaining: {}s",
                 out_node_endpoint, out_time_remaining);

    return true;
}

void ReservationClient::StartHeartbeat(
    const std::string& reservation_id,
    const std::string& job_id,
    int interval_seconds) {

    if (heartbeat_active_) {
        StopHeartbeat();
    }

    heartbeat_reservation_id_ = reservation_id;
    heartbeat_job_id_ = job_id;
    heartbeat_interval_seconds_ = interval_seconds;
    heartbeat_active_ = true;
    heartbeat_training_active_ = false;
    heartbeat_progress_ = 0.0;

    heartbeat_thread_ = std::thread(&ReservationClient::HeartbeatThreadFunc, this);

    spdlog::info("Heartbeat started for reservation {} (interval: {}s)",
                 reservation_id, interval_seconds);
}

void ReservationClient::StopHeartbeat() {
    heartbeat_active_ = false;
    if (heartbeat_thread_.joinable()) {
        heartbeat_thread_.join();
    }
}

void ReservationClient::UpdateHeartbeatState(
    const std::string& job_id,
    bool training_active,
    double progress,
    const std::map<std::string, double>& metrics) {

    heartbeat_job_id_ = job_id;
    heartbeat_training_active_ = training_active;
    heartbeat_progress_ = progress;

    {
        std::lock_guard<std::mutex> lock(heartbeat_metrics_mutex_);
        heartbeat_metrics_ = metrics;
    }
}

void ReservationClient::HeartbeatThreadFunc() {
    while (heartbeat_active_) {
        if (!SendHeartbeat()) {
            spdlog::warn("Heartbeat failed, will retry...");
        }

        // Sleep in small intervals to allow quick shutdown
        for (int i = 0; i < heartbeat_interval_seconds_ * 10 && heartbeat_active_; i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    spdlog::info("Heartbeat stopped for reservation {}", heartbeat_reservation_id_);
}

bool ReservationClient::SendHeartbeat() {
    if (!connected_ || !stub_) {
        return false;
    }

    grpc::ClientContext context;
    AddAuthMetadata(context);
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));

    cyxwiz::protocol::EngineHeartbeatRequest request;
    request.set_reservation_id(heartbeat_reservation_id_);
    request.set_job_id(heartbeat_job_id_);
    request.set_training_active(heartbeat_training_active_);
    request.set_job_progress(heartbeat_progress_);

    {
        std::lock_guard<std::mutex> lock(heartbeat_metrics_mutex_);
        for (const auto& [key, value] : heartbeat_metrics_) {
            (*request.mutable_current_metrics())[key] = value;
        }
    }

    cyxwiz::protocol::EngineHeartbeatResponse response;
    grpc::Status status = stub_->EngineHeartbeat(&context, request, &response);

    if (!status.ok()) {
        spdlog::warn("Heartbeat RPC failed: {}", status.error_message());
        return false;
    }

    if (response.status() != cyxwiz::protocol::STATUS_SUCCESS) {
        spdlog::warn("Heartbeat rejected: {}", response.message());
        return false;
    }

    // Notify callback
    if (heartbeat_callback_) {
        heartbeat_callback_(response.time_remaining_seconds(), response.should_extend());
    }

    // Log if time is running low
    if (response.should_extend()) {
        spdlog::warn("Reservation time running low! {} seconds remaining",
                     response.time_remaining_seconds());
    }

    return true;
}

} // namespace network
