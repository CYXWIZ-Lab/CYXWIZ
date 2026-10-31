#include "server_address_view.h"

namespace cyxwiz::protocol {
namespace {

bool IsBlank(const std::string& s) {
    return s.find_first_not_of(" \t\r\n") == std::string::npos;
}

// "services":{...,"central":"up",...} -> the word for one service
ServiceReach ServiceState(const std::string& body, const std::string& name) {
    const auto key = body.find("\"" + name + "\"");
    if (key == std::string::npos) {
        return ServiceReach::NotReachable;
    }
    const auto open = body.find('"', body.find(':', key) + 1);
    if (open == std::string::npos) {
        return ServiceReach::NotReachable;
    }
    return body.compare(open, 4, "\"up\"") == 0 ? ServiceReach::Reachable : ServiceReach::NotReachable;
}

}  // namespace

ServerAddressView DescribeServerAddress(const std::string& typed, ServiceSet set) {
    ServerAddressView view;
    if (IsBlank(typed)) {
        view.blank = true;
        view.kind_line = "No server address: the separate addresses (development) apply.";
        return view;
    }
    const ServerEndpoints e = ResolveServerAddress(typed);
    if (!e.error.empty()) {
        view.problem = "This address can't be used: " + e.error + ".";
        return view;
    }
    view.usable = true;
    if (e.front_door && e.tls) {
        view.kind_line = "A front door with TLS: everything at this one address, encrypted.";
    } else if (e.front_door) {
        view.kind_line = "A front door without TLS: everything at this one port. For this PC or the local network.";
    } else {
        view.kind_line = "No front door: central, the gateway and sign-in on their own ports (" +
                         std::to_string(kCentralGrpcPort) + ", " + std::to_string(kGatewayGrpcPort) + ", " +
                         std::to_string(kApiPort) + ").";
    }
    view.rows.push_back({"Central server", DisplayGrpcAddress(e.central), ServiceReach::NotChecked});
    if (set == ServiceSet::EngineServices) {
        view.rows.push_back({"CyxCloud gateway", DisplayGrpcAddress(e.gateway), ServiceReach::NotChecked});
    }
    view.rows.push_back({"Sign-in", e.api_base_url, ServiceReach::NotChecked});
    return view;
}

ServerAddressView DescribeSeparateAddresses(const std::string& central, const std::string& gateway,
                                            const std::string& api_base_url) {
    ServerAddressView view;
    view.blank = true;
    view.kind_line = "No server address: the separate addresses (development) apply.";
    view.rows.push_back({"Central server", DisplayGrpcAddress(central), ServiceReach::NotChecked});
    if (!gateway.empty()) {
        view.rows.push_back({"CyxCloud gateway", DisplayGrpcAddress(gateway), ServiceReach::NotChecked});
    }
    view.rows.push_back({"Sign-in", api_base_url, ServiceReach::NotChecked});
    return view;
}

void ApplyReach(ServerAddressView& view, const ServerReach& reach) {
    for (auto& row : view.rows) {
        if (row.name == "Central server") {
            row.reach = reach.central;
        } else if (row.name == "CyxCloud gateway") {
            row.reach = reach.gateway;
        } else {
            row.reach = reach.api;
        }
    }
}

ServerReach ReachFromFrontDoorHealth(int http_status, const std::string& body) {
    ServerReach r;
    r.central = r.gateway = r.api = ServiceReach::NotReachable;
    // The door answers 503 when a service is down; the body still says which.
    if ((http_status != 200 && http_status != 503) || body.find("\"services\"") == std::string::npos) {
        return r;
    }
    r.central = ServiceState(body, "central");
    r.gateway = ServiceState(body, "gateway");
    r.api = ServiceState(body, "api");
    return r;
}

const char* ReachWord(ServiceReach reach) {
    switch (reach) {
        case ServiceReach::Checking: return "Checking...";
        case ServiceReach::Reachable: return "Reachable";
        case ServiceReach::NotReachable: return "Not reachable";
        case ServiceReach::NotChecked: break;
    }
    return "Not checked yet";
}

std::string DisplayGrpcAddress(const std::string& grpc_address) {
    const GrpcTarget t = ParseGrpcAddress(grpc_address);
    return t.tls ? t.target + " (TLS)" : t.target;
}

const std::vector<ServerAddressExample>& ServerAddressExamples() {
    static const std::vector<ServerAddressExample> examples = {
        {"This PC", "http://localhost:8000"},
        {"Another PC", "http://192.168.1.216:8000"},
        {"No front door", "192.168.1.216"},
        {"CyxWiz", "https://cloud.cyxwiz.ai"},
    };
    return examples;
}

}  // namespace cyxwiz::protocol
