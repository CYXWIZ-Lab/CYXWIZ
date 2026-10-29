// One server address for the Engine and the server node (TOFIX168 step 3):
// the same rules as the cyxcloud CLI.
#include "common/server_address.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {
int failures = 0;
void Check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAILED: %s\n", what);
        ++failures;
    }
}
}  // namespace

int main() {
    using cyxwiz::protocol::ParseGrpcAddress;
    using cyxwiz::protocol::ResolveServerAddress;

    // A front door with TLS: everything at one origin
    auto e = ResolveServerAddress("https://cloud.cyxwiz.ai");
    Check(e.error.empty() && e.front_door && e.tls, "https is a front door with TLS");
    Check(e.central == "https://cloud.cyxwiz.ai:443", "central over TLS on 443");
    Check(e.gateway == e.central, "the gateway at the same door");
    Check(e.api_base_url == "https://cloud.cyxwiz.ai/api", "sign-in at the door's /api");
    e = ResolveServerAddress("https://cloud.cyxwiz.ai:8443/");
    Check(e.central == "https://cloud.cyxwiz.ai:8443" && e.api_base_url == "https://cloud.cyxwiz.ai:8443/api",
          "an https port is kept, a trailing slash dropped");

    // A front door without TLS (local, LAN)
    e = ResolveServerAddress("http://192.168.1.216:8000");
    Check(e.error.empty() && e.front_door && !e.tls, "http with a port is a front door without TLS");
    Check(e.central == "192.168.1.216:8000" && e.gateway == "192.168.1.216:8000", "plain gRPC at the door");
    Check(e.api_base_url == "http://192.168.1.216:8000/api", "sign-in at the door");

    // No front door: the three services on their own ports
    e = ResolveServerAddress("192.168.1.216");
    Check(e.error.empty() && !e.front_door && !e.tls, "a bare host is three services");
    Check(e.central == "192.168.1.216:50051", "central port");
    Check(e.gateway == "192.168.1.216:50060", "gateway port");
    Check(e.api_base_url == "http://192.168.1.216:3002/api", "API port");
    Check(ResolveServerAddress("http://localhost").central == "localhost:50051", "http without a port too");

    // Unclear addresses are refused with the reason
    Check(!ResolveServerAddress("").error.empty(), "empty");
    Check(!ResolveServerAddress("host:9999").error.empty(), "bare host:port is ambiguous");
    Check(!ResolveServerAddress("https://x/y").error.empty(), "a path");
    Check(!ResolveServerAddress("http://x:99999").error.empty(), "a port out of range");

    // gRPC addresses as channel targets
    auto t = ParseGrpcAddress("https://cloud.cyxwiz.ai:443");
    Check(t.tls && t.target == "cloud.cyxwiz.ai:443", "https target");
    t = ParseGrpcAddress("https://cloud.cyxwiz.ai");
    Check(t.tls && t.target == "cloud.cyxwiz.ai:443", "https default port");
    t = ParseGrpcAddress("localhost:50051");
    Check(!t.tls && t.target == "localhost:50051", "host:port stays plain");
    t = ParseGrpcAddress("http://10.0.0.5:8000");
    Check(!t.tls && t.target == "10.0.0.5:8000", "http target");
    t = ParseGrpcAddress("[::1]:50060");
    Check(!t.tls && t.target == "[::1]:50060", "IPv6 literal");

    if (failures == 0) std::printf("test_server_address: all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
