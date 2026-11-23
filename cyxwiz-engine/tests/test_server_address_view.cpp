// What the Servers screens show for the one server address (TOFIX168 step
// 3b): the rows, the kind of address, the problem line and the front door's
// /health reply. The Engine and the server node render this same model.
#include "common/server_address_view.h"

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
    using namespace cyxwiz::protocol;

    // This PC's front door (deploy/local runs one on 8000)
    auto v = DescribeServerAddress("http://localhost:8000", ServiceSet::EngineServices);
    Check(v.usable && v.problem.empty(), "the local front door is usable");
    Check(v.kind_line.find("front door without TLS") != std::string::npos, "says front door without TLS");
    Check(v.rows.size() == 3, "the Engine lists three services");
    Check(v.rows[0].name == "Central server" && v.rows[0].address == "localhost:8000", "central row");
    Check(v.rows[1].name == "CyxCloud gateway" && v.rows[1].address == "localhost:8000", "gateway row");
    Check(v.rows[2].name == "Sign-in" && v.rows[2].address == "http://localhost:8000/api", "sign-in row");
    Check(v.rows[0].reach == ServiceReach::NotChecked, "not checked before a probe");

    // A server node lists central and sign-in only
    v = DescribeServerAddress("http://localhost:8000", ServiceSet::NodeServices);
    Check(v.rows.size() == 2 && v.rows[1].name == "Sign-in", "a node lists two services");

    // TLS front door: the gRPC rows say TLS
    v = DescribeServerAddress("https://cloud.cyxwiz.ai", ServiceSet::EngineServices);
    Check(v.usable && v.kind_line.find("with TLS") != std::string::npos, "TLS kind");
    Check(v.rows[0].address == "cloud.cyxwiz.ai:443 (TLS)", "central shown with TLS");
    Check(v.rows[2].address == "https://cloud.cyxwiz.ai/api", "sign-in over https");

    // No front door: three ports
    v = DescribeServerAddress("192.168.1.216", ServiceSet::EngineServices);
    Check(v.usable && v.kind_line.find("No front door") != std::string::npos, "three services kind");
    Check(v.rows[0].address == "192.168.1.216:50051" && v.rows[1].address == "192.168.1.216:50060",
          "central and gateway ports");
    Check(v.rows[2].address == "http://192.168.1.216:3002/api", "API port");

    // Problems
    v = DescribeServerAddress("  ", ServiceSet::EngineServices);
    Check(!v.usable && v.blank && v.problem.empty() && v.rows.empty(), "empty: the separate addresses apply");
    v = DescribeSeparateAddresses("localhost:50051", "", "http://127.0.0.1:3002/api");
    Check(v.blank && v.rows.size() == 2 && v.rows[0].address == "localhost:50051", "separate addresses, no gateway");
    v = DescribeServerAddress("192.168.1.216:8000", ServiceSet::EngineServices);
    Check(!v.usable && v.problem.find("can't be used") != std::string::npos, "bare host:port refused");

    // The door's /health reply as it is on this PC, and with the gateway down
    auto r = ReachFromFrontDoorHealth(200, R"({"front":"ok","services":{"api":"up","central":"up","gateway":"up"}})");
    Check(r.central == ServiceReach::Reachable && r.gateway == ServiceReach::Reachable &&
              r.api == ServiceReach::Reachable,
          "all up");
    r = ReachFromFrontDoorHealth(503, R"({"front":"ok","services":{"api":"up","central":"up","gateway":"down"}})");
    Check(r.central == ServiceReach::Reachable && r.gateway == ServiceReach::NotReachable &&
              r.api == ServiceReach::Reachable,
          "503 still says which service is down");
    r = ReachFromFrontDoorHealth(0, "");
    Check(r.central == ServiceReach::NotReachable && r.api == ServiceReach::NotReachable, "no answer");
    r = ReachFromFrontDoorHealth(200, "<html>a website</html>");
    Check(r.central == ServiceReach::NotReachable, "something that is not a front door");

    v = DescribeServerAddress("http://localhost:8000", ServiceSet::EngineServices);
    ServerReach mixed;
    mixed.central = ServiceReach::Reachable;
    mixed.gateway = ServiceReach::NotReachable;
    mixed.api = ServiceReach::Checking;
    ApplyReach(v, mixed);
    Check(v.rows[0].reach == ServiceReach::Reachable && v.rows[1].reach == ServiceReach::NotReachable &&
              v.rows[2].reach == ServiceReach::Checking,
          "reach lands on the right rows");
    Check(std::string(ReachWord(ServiceReach::NotReachable)) == "Not reachable", "reach words");
    Check(ServerAddressExamples().size() == 4, "four examples");

    if (failures == 0) std::printf("test_server_address_view: all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
