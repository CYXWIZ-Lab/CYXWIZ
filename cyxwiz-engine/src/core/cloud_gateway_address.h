#pragma once

// Where the CyxCloud gateway's gRPC endpoint is when the config names none:
// the same host as the central server, port 50060 (deploy/local GATEWAY_GRPC).
// The Server Node's own ports are 50052-50057, so the gateway never shares
// them (TOFIX164 slice 1d).

#include <string>

namespace cyxwiz {
namespace core {

inline constexpr const char* kDefaultCloudGatewayPort = "50060";

// `configured` wins when set; otherwise the central address with its port
// replaced. A central address without a port keeps its host as is.
inline std::string ResolveCloudGatewayAddress(const std::string& configured, const std::string& central) {
    if (!configured.empty()) return configured;
    if (central.empty()) return std::string("localhost:") + kDefaultCloudGatewayPort;
    // IPv6 literals ([::1]:50051) keep their brackets: the port is after the
    // last ':' only when that ':' comes after the closing bracket.
    const auto close_bracket = central.rfind(']');
    const auto colon = central.rfind(':');
    const bool has_port =
        colon != std::string::npos && (close_bracket == std::string::npos || colon > close_bracket);
    const std::string host = has_port ? central.substr(0, colon) : central;
    return host + ":" + kDefaultCloudGatewayPort;
}

}  // namespace core
}  // namespace cyxwiz
