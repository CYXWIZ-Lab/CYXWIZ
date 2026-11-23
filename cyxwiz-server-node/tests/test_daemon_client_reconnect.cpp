// The GUI's daemon client finds its daemon again after the daemon restarts,
// and finds a daemon that starts after the GUI (found 2026-11-10: the Mac GUI
// stayed disconnected after its daemon was replaced).
#include "ipc/daemon_client.h"

#include <catch2/catch_test_macros.hpp>

#include <daemon.grpc.pb.h>
#include <grpcpp/grpcpp.h>

#include <chrono>
#include <memory>
#include <string>
#include <thread>

using cyxwiz::servernode::ipc::DaemonClient;
using cyxwiz::servernode::ipc::DaemonStatus;

namespace {

// A daemon that only answers GetStatus.
class FakeDaemon final : public cyxwiz::daemon::DaemonService::Service {
public:
    grpc::Status GetStatus(grpc::ServerContext*, const cyxwiz::daemon::GetStatusRequest*,
                           cyxwiz::daemon::GetStatusResponse* response) override {
        response->set_node_id("fake-node");
        return grpc::Status::OK;
    }
};

struct RunningDaemon {
    FakeDaemon service;
    std::unique_ptr<grpc::Server> server;

    explicit RunningDaemon(int& port) {
        grpc::ServerBuilder builder;
        const std::string address = port == 0 ? "127.0.0.1:0" : "127.0.0.1:" + std::to_string(port);
        builder.AddListeningPort(address, grpc::InsecureServerCredentials(), &port);
        builder.RegisterService(&service);
        server = builder.BuildAndStart();
    }
    ~RunningDaemon() {
        if (server) server->Shutdown(std::chrono::system_clock::now());
    }
};

bool ConnectedWithin(DaemonClient& client, std::chrono::seconds limit) {
    const auto until = std::chrono::steady_clock::now() + limit;
    while (std::chrono::steady_clock::now() < until) {
        if (client.IsConnected()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return client.IsConnected();
}

}  // namespace

TEST_CASE("the client reconnects after its daemon restarts", "[daemon_client]") {
    int port = 0;
    auto daemon = std::make_unique<RunningDaemon>(port);
    REQUIRE(daemon->server != nullptr);

    DaemonClient client;
    REQUIRE(client.Connect("127.0.0.1:" + std::to_string(port)));
    DaemonStatus status;
    CHECK(client.GetStatus(status));
    CHECK(status.node_id == "fake-node");

    daemon.reset();  // the daemon stops
    CHECK_FALSE(client.GetStatus(status));
    CHECK_FALSE(client.IsConnected());

    daemon = std::make_unique<RunningDaemon>(port);  // and starts again on its port
    REQUIRE(daemon->server != nullptr);
    CHECK(ConnectedWithin(client, std::chrono::seconds(10)));
    CHECK(client.GetStatus(status));
}

TEST_CASE("a daemon started after the client is found", "[daemon_client]") {
    int port = 0;
    {
        RunningDaemon probe(port);  // only to pick a free port
        REQUIRE(probe.server != nullptr);
    }

    DaemonClient client;
    CHECK_FALSE(client.Connect("127.0.0.1:" + std::to_string(port)));
    CHECK_FALSE(client.IsConnected());

    RunningDaemon daemon(port);
    REQUIRE(daemon.server != nullptr);
    CHECK(ConnectedWithin(client, std::chrono::seconds(10)));
    DaemonStatus status;
    CHECK(client.GetStatus(status));
}
