#include "server_address.h"

#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <grpcpp/support/channel_arguments.h>

#include <algorithm>
#include <cctype>

namespace cyxwiz::protocol {
namespace {

std::string Trim(const std::string& s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

bool StartsWith(const std::string& s, const char* prefix) {
    return s.rfind(prefix, 0) == 0;
}

// "host:port", "[::1]:port" -> true and the parts; "host", "[::1]" -> false.
bool SplitPort(const std::string& host_port, std::string* host, std::string* port) {
    const auto close_bracket = host_port.rfind(']');
    const auto colon = host_port.rfind(':');
    if (colon == std::string::npos ||
        (close_bracket != std::string::npos && colon < close_bracket)) {
        *host = host_port;
        port->clear();
        return false;
    }
    *host = host_port.substr(0, colon);
    *port = host_port.substr(colon + 1);
    return true;
}

bool ValidPort(const std::string& port) {
    if (port.empty() || port.size() > 5 ||
        !std::all_of(port.begin(), port.end(), [](unsigned char c) { return std::isdigit(c); })) {
        return false;
    }
    const int n = std::stoi(port);
    return n > 0 && n <= 65535;
}

ServerEndpoints Error(std::string message) {
    ServerEndpoints e;
    e.error = std::move(message);
    return e;
}

}  // namespace

ServerEndpoints ResolveServerAddress(const std::string& typed) {
    std::string address = Trim(typed);
    while (!address.empty() && address.back() == '/') {
        address.pop_back();
    }
    if (address.empty()) {
        return Error("the server address is empty");
    }

    const bool https = StartsWith(address, "https://");
    const bool http = StartsWith(address, "http://");
    const std::string rest = https ? address.substr(8) : http ? address.substr(7) : address;
    if (rest.empty() || rest.find('/') != std::string::npos) {
        return Error("'" + typed + "' is not a server address (no path: https://HOST, http://HOST:PORT or HOST)");
    }

    std::string host, port;
    const bool has_port = SplitPort(rest, &host, &port);
    if (host.empty() || (has_port && !ValidPort(port))) {
        return Error("'" + typed + "' has no usable host or port");
    }

    ServerEndpoints e;
    if (https || (http && has_port)) {
        // A front door: everything at one origin
        e.front_door = true;
        e.tls = https;
        const std::string grpc_port = has_port ? port : (https ? "443" : "80");
        const std::string target = host + ":" + grpc_port;
        e.central = https ? "https://" + target : target;
        e.gateway = e.central;
        e.api_base_url = (https ? "https://" : "http://") + rest + "/api";
        return e;
    }
    if (has_port) {
        return Error("'" + typed +
                     "': give the server's host name or IP, or a front door as http://HOST:PORT or https://HOST");
    }
    // No front door: the three services on their own ports
    e.central = host + ":" + std::to_string(kCentralGrpcPort);
    e.gateway = host + ":" + std::to_string(kGatewayGrpcPort);
    e.api_base_url = "http://" + host + ":" + std::to_string(kApiPort) + "/api";
    return e;
}

GrpcTarget ParseGrpcAddress(const std::string& typed) {
    std::string address = Trim(typed);
    while (!address.empty() && address.back() == '/') {
        address.pop_back();
    }
    GrpcTarget t;
    if (StartsWith(address, "https://")) {
        t.tls = true;
        address = address.substr(8);
        std::string host, port;
        if (!SplitPort(address, &host, &port)) {
            address += ":443";
        }
    } else if (StartsWith(address, "http://")) {
        address = address.substr(7);
        std::string host, port;
        if (!SplitPort(address, &host, &port)) {
            address += ":80";
        }
    }
    t.target = address;
    return t;
}

std::shared_ptr<grpc::Channel> CreateServerChannel(const std::string& address) {
    return CreateServerChannel(address, grpc::ChannelArguments());
}

std::shared_ptr<grpc::Channel> CreateServerChannel(const std::string& address,
                                                   const grpc::ChannelArguments& args) {
    const GrpcTarget t = ParseGrpcAddress(address);
    auto credentials = t.tls ? grpc::SslCredentials(grpc::SslCredentialsOptions())
                             : grpc::InsecureChannelCredentials();
    return grpc::CreateCustomChannel(t.target, credentials, args);
}

}  // namespace cyxwiz::protocol
