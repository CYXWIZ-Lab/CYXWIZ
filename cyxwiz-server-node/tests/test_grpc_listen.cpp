// A daemon's gRPC port cannot be taken by a second daemon (found on the Mac,
// 2026-11-10: three daemons served the same ports at once).
#include "core/grpc_listen.h"

#include <catch2/catch_test_macros.hpp>

#include <daemon.grpc.pb.h>
#include <grpcpp/server.h>

using cyxwiz::servernode::core::ListenExclusively;

// gRPC will not start a server with no service; every method of this one
// answers UNIMPLEMENTED, which is all a port test needs. A service belongs to
// one server, so each server gets its own.
using UnimplementedService = cyxwiz::daemon::DaemonService::Service;

TEST_CASE("a second server cannot listen on a port another one owns", "[grpc_listen]") {
    int port = 0;
    grpc::ServerBuilder first;
    first.AddChannelArgument(GRPC_ARG_ALLOW_REUSEPORT, 0);
    first.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    UnimplementedService first_service;
    first.RegisterService(&first_service);
    auto first_server = first.BuildAndStart();
    REQUIRE(first_server != nullptr);
    REQUIRE(port > 0);

    const std::string address = "127.0.0.1:" + std::to_string(port);
    grpc::ServerBuilder second;
    ListenExclusively(second, address, grpc::InsecureServerCredentials());
    UnimplementedService second_service;
    second.RegisterService(&second_service);
    auto second_server = second.BuildAndStart();
    CHECK(second_server == nullptr);

    first_server->Shutdown();
}

TEST_CASE("the port is free again once its server stops", "[grpc_listen]") {
    int port = 0;
    grpc::ServerBuilder first;
    first.AddChannelArgument(GRPC_ARG_ALLOW_REUSEPORT, 0);
    first.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    UnimplementedService first_service;
    first.RegisterService(&first_service);
    auto first_server = first.BuildAndStart();
    REQUIRE(first_server != nullptr);
    first_server->Shutdown();
    first_server.reset();

    grpc::ServerBuilder again;
    ListenExclusively(again, "127.0.0.1:" + std::to_string(port), grpc::InsecureServerCredentials());
    UnimplementedService again_service;
    again.RegisterService(&again_service);
    auto again_server = again.BuildAndStart();
    CHECK(again_server != nullptr);
    if (again_server) again_server->Shutdown();
}
