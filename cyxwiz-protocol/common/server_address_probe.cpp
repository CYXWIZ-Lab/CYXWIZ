#include "server_address_probe.h"

#include <grpcpp/channel.h>

namespace cyxwiz::protocol {
namespace {

ServiceReach GrpcReach(const std::string& address, std::chrono::milliseconds timeout) {
    auto channel = CreateServerChannel(address);
    const bool up = channel->WaitForConnected(std::chrono::system_clock::now() + timeout);
    return up ? ServiceReach::Reachable : ServiceReach::NotReachable;
}

// "http://h:3002/api" -> "http://h:3002"
std::string OriginOf(const std::string& api_base_url) {
    const std::string suffix = "/api";
    if (api_base_url.size() >= suffix.size() &&
        api_base_url.compare(api_base_url.size() - suffix.size(), suffix.size(), suffix) == 0) {
        return api_base_url.substr(0, api_base_url.size() - suffix.size());
    }
    return api_base_url;
}

}  // namespace

ServerReach ProbeServer(const std::string& typed, ServiceSet set, const HttpGetFn& http_get,
                        std::chrono::milliseconds timeout) {
    const ServerEndpoints e = ResolveServerAddress(typed);
    if (!e.error.empty()) {
        ServerReach r;
        r.central = r.gateway = r.api = ServiceReach::NotReachable;
        return r;
    }
    return ProbeEndpoints(e, set, http_get, timeout);
}

ServerReach ProbeEndpoints(const ServerEndpoints& e, ServiceSet set, const HttpGetFn& http_get,
                           std::chrono::milliseconds timeout) {
    ServerReach r;
    r.central = r.gateway = r.api = ServiceReach::NotReachable;
    const std::string origin = OriginOf(e.api_base_url);
    std::string body;
    if (e.front_door) {
        const int status = http_get(origin + "/health", &body);
        return ReachFromFrontDoorHealth(status, body);
    }
    r.central = GrpcReach(e.central, timeout);
    if (set == ServiceSet::EngineServices && !e.gateway.empty()) {
        r.gateway = GrpcReach(e.gateway, timeout);
    }
    r.api = http_get(origin + "/health", &body) == 200 ? ServiceReach::Reachable : ServiceReach::NotReachable;
    return r;
}

}  // namespace cyxwiz::protocol
