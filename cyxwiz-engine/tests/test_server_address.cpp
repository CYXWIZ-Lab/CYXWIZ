// One server address for the Engine and the server node (TOFIX168 step 3):
// the same rules as the cyxcloud CLI.
#include "common/server_address.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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

    // Plain text only to this machine or a private network (the cyxcloud
    // CLI's rule): a public server must be https
    using cyxwiz::protocol::PlainHttpAllowed;
    for (const char* host : {"localhost", "127.0.0.1", "::1", "[::1]", "10.1.2.3", "172.16.0.9", "172.31.255.1",
                             "192.168.1.180", "169.254.10.1", "100.100.1.2", "fd12:3456::1", "fe80::1",
                             "::ffff:192.168.1.5", "hp-linux", "gateway", "nas.local", "box.lan", "svc.internal",
                             "printer.home.arpa"}) {
        Check(PlainHttpAllowed(host), host);
    }
    for (const char* host : {"8.8.8.8", "203.0.113.9", "172.32.0.1", "100.128.0.1", "2001:db8::1", "::ffff:8.8.8.8",
                             "cloud.cyxwiz.ai", "example.com", "",
                             // look private or one-label, but the host dialled is public
                             "134744072", "0x8080808", "010.0.0.1", "fd00::1]evil.com", "evil.com#.local",
                             "user@x.local", "evil.com\\@192.168.1.1"}) {
        Check(!PlainHttpAllowed(host), host);
    }
    e = ResolveServerAddress("http://cloud.cyxwiz.ai:8000");
    Check(!e.error.empty() && e.error.find("https://") != std::string::npos,
          "plain to a public front door is refused, with what to use");
    Check(!ResolveServerAddress("203.0.113.9").error.empty(), "a bare public host is plain too");
    Check(ResolveServerAddress("http://192.168.1.180:8000").error.empty(), "a LAN front door may be plain");
    Check(ResolveServerAddress("https://203.0.113.9").error.empty(), "https to anywhere");

    // The server's CA: set by the app, else CYXWIZ_CA_FILE; a bad file is named
    using cyxwiz::protocol::ServerCaError;
    using cyxwiz::protocol::ServerCaFile;
    using cyxwiz::protocol::SetServerCaFile;
    const auto dir = std::filesystem::temp_directory_path() / "cyxwiz_test_server_ca";
    std::filesystem::create_directories(dir);
    const auto good = (dir / "ca.crt").string();
    const auto bad = (dir / "notes.txt").string();
    std::ofstream(good) << "-----BEGIN CERTIFICATE-----\nMIIB\n-----END CERTIFICATE-----\n";
    std::ofstream(bad) << "not a certificate";
    SetServerCaFile("");
#ifdef _WIN32
    _putenv_s("CYXWIZ_CA_FILE", good.c_str());
#else
    setenv("CYXWIZ_CA_FILE", good.c_str(), 1);
#endif
    Check(ServerCaFile() == good, "no setting: the environment's CA");
    SetServerCaFile(bad);
    Check(ServerCaFile() == bad, "the app's setting wins");
    Check(!ServerCaError().empty(), "a file with no certificate is named");
    SetServerCaFile((dir / "missing.crt").string());
    Check(!ServerCaError().empty(), "a missing file is named");
    SetServerCaFile(good);
    Check(ServerCaError().empty(), "a PEM file is fine");
#ifdef _WIN32
    _putenv_s("CYXWIZ_CA_FILE", "");
#else
    unsetenv("CYXWIZ_CA_FILE");
#endif
    SetServerCaFile("");
    Check(ServerCaFile().empty() && ServerCaError().empty(), "none: the system's roots");
    std::filesystem::remove_all(dir);

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
