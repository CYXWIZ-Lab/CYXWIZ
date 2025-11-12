#include "grpc_client.h"
#include <spdlog/spdlog.h>

namespace network {

GRPCClient::GRPCClient() : connected_(false) {
}

GRPCClient::~GRPCClient() {
    Disconnect();
}

bool GRPCClient::Connect(const std::string& server_address) {
    spdlog::info("Connecting to server: {}", server_address);
    server_address_ = server_address;
    // TODO: Implement gRPC connection
    connected_ = true;
    return true;
}

void GRPCClient::Disconnect() {
    if (connected_) {
        spdlog::info("Disconnecting from server");
        // TODO: Close gRPC channel
        connected_ = false;
    }
}

} // namespace network
