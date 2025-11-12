#pragma once
#include <string>
#include <memory>

namespace network {

class GRPCClient {
public:
    GRPCClient();
    ~GRPCClient();

    bool Connect(const std::string& server_address);
    void Disconnect();
    bool IsConnected() const { return connected_; }

private:
    bool connected_;
    std::string server_address_;
};

} // namespace network
