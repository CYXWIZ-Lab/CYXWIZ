#include "grpc_client.h"
#include <spdlog/spdlog.h>

namespace network {

GRPCClient::GRPCClient() : connected_(false) {
}

GRPCClient::~GRPCClient() {
    Disconnect();
}

bool GRPCClient::Connect(const std::string& server_address) {
    spdlog::info("Connecting to Central Server: {}", server_address);

    try {
        // Create gRPC channel
        channel_ = grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());

        // Create stub for JobService
        job_stub_ = cyxwiz::protocol::JobService::NewStub(channel_);

        // Test connection by checking channel state
        auto state = channel_->GetState(true);
        if (state == GRPC_CHANNEL_SHUTDOWN || state == GRPC_CHANNEL_TRANSIENT_FAILURE) {
            last_error_ = "Failed to connect to server: Channel in bad state";
            spdlog::error(last_error_);
            return false;
        }

        server_address_ = server_address;
        connected_ = true;
        spdlog::info("Successfully connected to Central Server");
        return true;

    } catch (const std::exception& e) {
        last_error_ = std::string("Connection error: ") + e.what();
        spdlog::error(last_error_);
        connected_ = false;
        return false;
    }
}

void GRPCClient::Disconnect() {
    if (connected_) {
        spdlog::info("Disconnecting from Central Server");
        job_stub_.reset();
        channel_.reset();
        connected_ = false;
        server_address_.clear();
    }
}

bool GRPCClient::SubmitJob(const cyxwiz::protocol::SubmitJobRequest& request,
                            cyxwiz::protocol::SubmitJobResponse& response) {
    if (!connected_ || !job_stub_) {
        last_error_ = "Not connected to server";
        spdlog::error(last_error_);
        return false;
    }

    spdlog::info("Submitting job to Central Server...");

    try {
        grpc::ClientContext context;

        // Set timeout for the RPC call (30 seconds)
        std::chrono::system_clock::time_point deadline =
            std::chrono::system_clock::now() + std::chrono::seconds(30);
        context.set_deadline(deadline);

        // Make the RPC call
        grpc::Status status = job_stub_->SubmitJob(&context, request, &response);

        if (status.ok()) {
            spdlog::info("Job submitted successfully. Job ID: {}", response.job_id());
            return true;
        } else {
            last_error_ = "Job submission failed: " + status.error_message();
            spdlog::error("{} (code: {})", last_error_, static_cast<int>(status.error_code()));
            return false;
        }

    } catch (const std::exception& e) {
        last_error_ = std::string("Job submission error: ") + e.what();
        spdlog::error(last_error_);
        return false;
    }
}

bool GRPCClient::GetJobStatus(const std::string& job_id,
                                cyxwiz::protocol::GetJobStatusResponse& response) {
    if (!connected_ || !job_stub_) {
        last_error_ = "Not connected to server";
        spdlog::error(last_error_);
        return false;
    }

    try {
        cyxwiz::protocol::GetJobStatusRequest request;
        request.set_job_id(job_id);

        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));

        grpc::Status status = job_stub_->GetJobStatus(&context, request, &response);

        if (status.ok()) {
            return true;
        } else {
            last_error_ = "Get job status failed: " + status.error_message();
            spdlog::error(last_error_);
            return false;
        }

    } catch (const std::exception& e) {
        last_error_ = std::string("Get job status error: ") + e.what();
        spdlog::error(last_error_);
        return false;
    }
}

bool GRPCClient::CancelJob(const std::string& job_id,
                             cyxwiz::protocol::CancelJobResponse& response) {
    if (!connected_ || !job_stub_) {
        last_error_ = "Not connected to server";
        spdlog::error(last_error_);
        return false;
    }

    try {
        cyxwiz::protocol::CancelJobRequest request;
        request.set_job_id(job_id);

        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));

        grpc::Status status = job_stub_->CancelJob(&context, request, &response);

        if (status.ok()) {
            spdlog::info("Job cancelled successfully: {}", job_id);
            return true;
        } else {
            last_error_ = "Cancel job failed: " + status.error_message();
            spdlog::error(last_error_);
            return false;
        }

    } catch (const std::exception& e) {
        last_error_ = std::string("Cancel job error: ") + e.what();
        spdlog::error(last_error_);
        return false;
    }
}

} // namespace network
