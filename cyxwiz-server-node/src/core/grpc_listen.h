// grpc_listen.h - a gRPC listening port the daemon owns alone
#pragma once

#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server_builder.h>

#include <memory>
#include <string>

namespace cyxwiz::servernode::core {

// Adds `address` to `builder` with port sharing off. On Linux and macOS gRPC
// turns SO_REUSEPORT on by default, so a second daemon started by mistake
// bound the same ports and the system split connections between the two (on
// the Mac the GUI's Apply reached an old daemon whose sign-in had expired).
// Without it the second bind fails, BuildAndStart returns null and the second
// daemon stops at start. Windows never shared the port.
//
// `selected_port`, when given, receives the port gRPC bound once the server
// starts: the way to learn the port of an address ending in ":0" (any free
// port), which tests use so they never collide with a running node.
inline void ListenExclusively(grpc::ServerBuilder& builder, const std::string& address,
                              std::shared_ptr<grpc::ServerCredentials> credentials,
                              int* selected_port = nullptr) {
    builder.AddChannelArgument(GRPC_ARG_ALLOW_REUSEPORT, 0);
    builder.AddListeningPort(address, std::move(credentials), selected_port);
}

}  // namespace cyxwiz::servernode::core
