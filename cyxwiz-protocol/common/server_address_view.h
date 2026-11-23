#pragma once

// What a screen shows about the one server address (TOFIX168 step 3b): the
// Engine's Preferences > Servers tab and the server node's Settings >
// Network and sign-in dialog render this, so they word it the same way.
// Pure data: no ImGui, no network.

#include "server_address.h"

#include <string>
#include <vector>

namespace cyxwiz::protocol {

enum class ServiceReach { NotChecked, Checking, Reachable, NotReachable };

struct ServiceRow {
    std::string name;     // "Central server", "CyxCloud gateway", "Sign-in"
    std::string address;  // as shown: "localhost:8000", "cloud.cyxwiz.ai:443 (TLS)"
    ServiceReach reach = ServiceReach::NotChecked;
};

struct ServerAddressView {
    bool usable = false;
    bool blank = false;     // no address: the separate addresses apply
    std::string problem;    // why the typed address can't be used
    std::string kind_line;  // what kind of address it is, in one sentence
    std::vector<ServiceRow> rows;
};

// Which services a screen lists: the Engine uses all three; a server node
// gets its gateway address in each job, so it lists central and sign-in.
enum class ServiceSet { EngineServices, NodeServices };

ServerAddressView DescribeServerAddress(const std::string& typed, ServiceSet set);

// The rows when the server address is empty and the separate per-service
// addresses (development) apply. An empty gateway is left out.
ServerAddressView DescribeSeparateAddresses(const std::string& central, const std::string& gateway,
                                            const std::string& api_base_url);

// The probe result for each service, in row order (central, gateway, sign-in).
struct ServerReach {
    ServiceReach central = ServiceReach::NotChecked;
    ServiceReach gateway = ServiceReach::NotChecked;
    ServiceReach api = ServiceReach::NotChecked;
};
void ApplyReach(ServerAddressView& view, const ServerReach& reach);

// A front door's GET /health reply: {"front":"ok","services":{"api":"up",...}}.
// No reply (status 0) or one the door did not write: every service not reachable.
ServerReach ReachFromFrontDoorHealth(int http_status, const std::string& body);

const char* ReachWord(ServiceReach reach);

// A gRPC address for people: "https://h:443" -> "h:443 (TLS)".
std::string DisplayGrpcAddress(const std::string& grpc_address);

struct ServerAddressExample {
    const char* label;
    const char* value;
};
// Clickable examples under the address field.
const std::vector<ServerAddressExample>& ServerAddressExamples();

}  // namespace cyxwiz::protocol
