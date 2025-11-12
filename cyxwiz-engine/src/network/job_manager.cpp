#include "job_manager.h"
#include "grpc_client.h"
#include <spdlog/spdlog.h>

namespace network {

JobManager::JobManager(GRPCClient* client) : client_(client) {
}

JobManager::~JobManager() = default;

void JobManager::Update() {
    // TODO: Poll for job updates
}

bool JobManager::SubmitJob(const std::string& job_config) {
    if (!client_ || !client_->IsConnected()) {
        spdlog::error("Not connected to server");
        return false;
    }

    // TODO: Submit job via gRPC
    spdlog::info("Submitting job...");
    return true;
}

void JobManager::CancelJob(const std::string& job_id) {
    // TODO: Cancel job via gRPC
    spdlog::info("Cancelling job: {}", job_id);
}

} // namespace network
