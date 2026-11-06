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
//
// Plain text (http://, or a bare host) is accepted only to this machine or a
// private network, the rule the cyxcloud CLI and node apply
// (cyxcloud-core plain_http, TOFIX192 D3): a sign-in token sent in clear
// across the internet can be read on the way.

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

// Whether plain http:// may be used to reach `host` (a name, an IP, an IPv6
// literal with or without brackets): loopback, private and link-local
// addresses (10/8, 172.16/12, 192.168/16, 169.254/16, 100.64/10, fc00::/7,
// fe80::/10), and LAN names (one label, or .local .lan .internal .home.arpa).
bool PlainHttpAllowed(const std::string& host);

// Whether `host` is this machine.
bool IsLoopbackHost(const std::string& host);

// The server's private CA, a PEM file (a test server's ca.crt, cyx_cloud
// docs/usage/tls.md), for a server whose certificate no public authority
// signed. Process-wide: the Engine and the server node set it from their
// settings. Empty means the system's trusted roots; when nothing was set,
// the environment variable CYXWIZ_CA_FILE is used.
void SetServerCaFile(const std::string& path);
std::string ServerCaFile();

// Why the CA file could not be used ("" when it can, or none is set).
std::string ServerCaError();

// A gRPC address as a channel target: "https://h[:p]" -> "h:p" (443 by
// default) with TLS; "http://h:p" and "h:p" -> plain.
struct GrpcTarget {
    std::string target;
    bool tls = false;
};
GrpcTarget ParseGrpcAddress(const std::string& address);

// A channel to a server-side service: TLS when the address says https://,
// verified against the server's CA when one is set (and only it), else the
// system's roots; plain otherwise. A CA file that cannot be read fails
// closed: no certificate verifies.
std::shared_ptr<grpc::Channel> CreateServerChannel(const std::string& address);
std::shared_ptr<grpc::Channel> CreateServerChannel(const std::string& address,
                                                   const grpc::ChannelArguments& args);

}  // namespace cyxwiz::protocol
