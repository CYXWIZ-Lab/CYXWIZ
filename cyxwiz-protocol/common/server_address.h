#pragma once

// One address for the CyxWiz server side (TOFIX168 step 3), with the same
// rules as the cyxcloud CLI (cyx_cloud docs/usage/cli.md):
//
//   https://HOST[:PORT]   a front door with TLS: central, the CyxCloud gateway
//                         and the CyxWiz API all at that one origin
//   http://HOST:PORT      a front door without TLS (local, LAN)
//   HOST, http://HOST     no front door: the three services on their own
//                         ports (central 50051, gateway 50060, API 3002)
//
// gRPC addresses may carry a scheme from then on ("https://host:443" means
// TLS); every channel to the server side is made by CreateServerChannel.

#include <memory>
#include <string>

namespace grpc {
class Channel;
class ChannelArguments;
}

namespace cyxwiz::protocol {

inline constexpr int kCentralGrpcPort = 50051;
inline constexpr int kGatewayGrpcPort = 50060;
inline constexpr int kApiPort = 3002;

// Where a client reaches each service.
struct ServerEndpoints {
    bool front_door = false;
    bool tls = false;
    std::string central;       // gRPC address: "host:port" or "https://host:port"
    std::string gateway;       // gRPC address, same form
    std::string api_base_url;  // "http(s)://host[:port]/api"
    std::string error;         // set when the address is not usable
};

// The endpoints for what the user typed as the server address.
ServerEndpoints ResolveServerAddress(const std::string& address);

// A gRPC address as a channel target: "https://h[:p]" -> "h:p" (443 by
// default) with TLS; "http://h:p" and "h:p" -> plain.
struct GrpcTarget {
    std::string target;
    bool tls = false;
};
GrpcTarget ParseGrpcAddress(const std::string& address);

// A channel to a server-side service: SSL credentials (the system's trusted
// roots) when the address says https://, plain otherwise.
std::shared_ptr<grpc::Channel> CreateServerChannel(const std::string& address);
std::shared_ptr<grpc::Channel> CreateServerChannel(const std::string& address,
                                                   const grpc::ChannelArguments& args);

}  // namespace cyxwiz::protocol
