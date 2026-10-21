// Where the Engine looks for the CyxCloud gateway (TOFIX164 slice 1d).
#include "core/cloud_gateway_address.h"

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
    using cyxwiz::core::ResolveCloudGatewayAddress;
    Check(ResolveCloudGatewayAddress("", "localhost:50051") == "localhost:50060", "central host, gateway port");
    Check(ResolveCloudGatewayAddress("", "192.168.1.116:50051") == "192.168.1.116:50060", "LAN central -> LAN gateway");
    Check(ResolveCloudGatewayAddress("", "cloud.cyxwiz.io") == "cloud.cyxwiz.io:50060", "host without port");
    Check(ResolveCloudGatewayAddress("", "[::1]:50051") == "[::1]:50060", "IPv6 literal keeps brackets");
    Check(ResolveCloudGatewayAddress("", "") == "localhost:50060", "nothing configured at all");
    Check(ResolveCloudGatewayAddress("gw.local:7000", "localhost:50051") == "gw.local:7000", "configured wins");
    if (failures == 0) std::printf("test_cloud_gateway_address: all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
