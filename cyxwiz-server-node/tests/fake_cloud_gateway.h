#pragma once

// A CyxCloud gateway DataService in this process, for node tests (TOFIX164):
// serves one file in chunks the way the real one does after slice 2 (token
// required, DATA_LOSS instead of a skip, SHA-256 trailer).

#include "../../cyxwiz-engine/src/core/sha256_digest.h"

#include <catch2/catch_test_macros.hpp>
#include <cyxcloud_data.grpc.pb.h>
#include <grpcpp/grpcpp.h>

#include <chrono>
#include <memory>
#include <string>
#include <thread>

namespace cyxwiz {
namespace server_node {
namespace test {

enum class FakeGatewayMode { Ok, DataLossAtChunk, WrongHash, NoLastChunk, Slow };

// Serves `content` in `chunk_size` pieces; `token` is the only accepted token.
class FakeDataService final : public cyxcloud::data::DataService::Service {
public:
    std::string content;
    std::string token = "good-token";
    size_t chunk_size = 1000;
    FakeGatewayMode mode = FakeGatewayMode::Ok;
    int fail_at = 1;  // chunk index that fails in DataLossAtChunk
    int requests = 0;

    grpc::Status StreamData(grpc::ServerContext*, const cyxcloud::data::StreamDataRequest* request,
                            grpc::ServerWriter<cyxcloud::data::DataChunk>* writer) override {
        ++requests;
        if (request->access_token().empty()) {
            return grpc::Status(grpc::StatusCode::UNAUTHENTICATED, "Dataset access token required");
        }
        if (request->access_token() != token) {
            return grpc::Status(grpc::StatusCode::UNAUTHENTICATED, "Dataset access token is invalid, expired or revoked");
        }
        if (request->shuffle()) return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "shuffle is not supported");

        cyxwiz::Sha256Hasher hasher;
        std::string error;
        const size_t count = (content.size() + chunk_size - 1) / chunk_size;
        for (size_t i = 0; i < count; ++i) {
            if (mode == FakeGatewayMode::DataLossAtChunk && static_cast<int>(i) == fail_at) {
                return grpc::Status(grpc::StatusCode::DATA_LOSS,
                                    "chunk " + std::to_string(i + 1) + " of " + std::to_string(count) +
                                        " could not be served: no storage node holds it");
            }
            if (mode == FakeGatewayMode::Slow) std::this_thread::sleep_for(std::chrono::milliseconds(1500));
            const std::string piece = content.substr(i * chunk_size, chunk_size);
            hasher.Update(piece, error);
            cyxcloud::data::DataChunk chunk;
            chunk.set_batch_index(i);
            chunk.set_data(piece);
            const bool last = i + 1 == count;
            if (last && mode != FakeGatewayMode::NoLastChunk) {
                std::string digest;
                hasher.Finish(digest, error);
                if (mode == FakeGatewayMode::WrongHash) digest[0] = digest[0] == 'a' ? 'b' : 'a';
                chunk.set_is_last(true);
                chunk.set_stream_sha256(digest);
                chunk.set_total_bytes(content.size());
            }
            if (!writer->Write(chunk)) break;
        }
        return grpc::Status::OK;
    }
};

struct FakeGateway {
    FakeDataService service;
    std::unique_ptr<grpc::Server> server;
    int port = 0;

    FakeGateway() {
        grpc::ServerBuilder builder;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
        builder.RegisterService(&service);
        server = builder.BuildAndStart();
        REQUIRE(server);
        REQUIRE(port > 0);
    }
    ~FakeGateway() {
        if (server) server->Shutdown();
    }
    std::string address() const { return "127.0.0.1:" + std::to_string(port); }
};

}  // namespace test
}  // namespace server_node
}  // namespace cyxwiz
