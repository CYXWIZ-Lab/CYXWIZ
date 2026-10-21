// A node pulls a dataset file from CyxCloud (TOFIX164 slice 3): a fake
// gateway DataService in this process serves a file in chunks the way the
// real one does after slice 2 (token required, DATA_LOSS instead of a skip,
// SHA-256 trailer), and the fetcher must accept only what verifies.
#include "cloud_dataset_fetcher.h"

#include "../../cyxwiz-engine/src/core/sha256_digest.h"

#include <catch2/catch_test_macros.hpp>
#include <cyxcloud_data.grpc.pb.h>
#include <grpcpp/grpcpp.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>

namespace fs = std::filesystem;
using cyxwiz::server_node::CloudDatasetFetcher;
using cyxwiz::server_node::SafeCloudFileName;

namespace {

enum class Mode { Ok, DataLossAtChunk, WrongHash, NoLastChunk, Slow };

// Serves `content` in `chunk_size` pieces; `token` is the only accepted token.
class FakeDataService final : public cyxcloud::data::DataService::Service {
public:
    std::string content;
    std::string token = "good-token";
    size_t chunk_size = 1000;
    Mode mode = Mode::Ok;
    int fail_at = 1;  // chunk index that fails in DataLossAtChunk
    int requests = 0;

    grpc::Status StreamData(grpc::ServerContext*, const cyxcloud::data::StreamDataRequest* request,
                            grpc::ServerWriter<cyxcloud::data::DataChunk>* writer) override {
        ++requests;
        if (request->access_token().empty()) return grpc::Status(grpc::StatusCode::UNAUTHENTICATED, "Dataset access token required");
        if (request->access_token() != token) {
            return grpc::Status(grpc::StatusCode::UNAUTHENTICATED, "Dataset access token is invalid, expired or revoked");
        }
        if (request->shuffle()) return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "shuffle is not supported");

        cyxwiz::Sha256Hasher hasher;
        std::string error;
        const size_t count = (content.size() + chunk_size - 1) / chunk_size;
        for (size_t i = 0; i < count; ++i) {
            if (mode == Mode::DataLossAtChunk && static_cast<int>(i) == fail_at) {
                return grpc::Status(grpc::StatusCode::DATA_LOSS,
                                    "chunk " + std::to_string(i + 1) + " of " + std::to_string(count) +
                                        " could not be served: no storage node holds it");
            }
            if (mode == Mode::Slow) std::this_thread::sleep_for(std::chrono::milliseconds(1500));
            const std::string piece = content.substr(i * chunk_size, chunk_size);
            hasher.Update(piece, error);
            cyxcloud::data::DataChunk chunk;
            chunk.set_batch_index(i);
            chunk.set_data(piece);
            const bool last = i + 1 == count;
            if (last && mode != Mode::NoLastChunk) {
                std::string digest;
                hasher.Finish(digest, error);
                if (mode == Mode::WrongHash) digest[0] = digest[0] == 'a' ? 'b' : 'a';
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

cyxwiz::protocol::CloudDatasetSource Source(const FakeGateway& gateway, const std::string& token = "good-token") {
    cyxwiz::protocol::CloudDatasetSource source;
    source.set_dataset_name("tiny causal lm tokens");
    source.set_gateway_address(gateway.address());
    source.set_file_id("2f1c7f8e-7d3a-4b4e-9c2d-1a2b3c4d5e6f");
    source.set_access_token(token);
    return source;
}

std::string ReadAll(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::string Content(size_t bytes) {
    std::string s;
    s.reserve(bytes);
    for (size_t i = 0; i < bytes; ++i) s.push_back(static_cast<char>('A' + (i * 7 + i / 13) % 26));
    return s;
}

}  // namespace

TEST_CASE("cloud dataset fetch: the file arrives whole and verified", "[cloud]") {
    const fs::path work = fs::temp_directory_path() / "cyxwiz_cloud_fetch_test";
    std::error_code ec;
    fs::remove_all(work, ec);
    FakeGateway gateway;
    gateway.service.content = Content(4321);
    gateway.service.chunk_size = 1000;  // 5 chunks, the last short
    CloudDatasetFetcher fetcher(work);

    fs::path file;
    std::string error;
    REQUIRE(fetcher.Fetch(Source(gateway), file, error));
    CHECK(error.empty());
    CHECK(fs::exists(file));
    CHECK(file.filename().string() == "tiny_causal_lm_tokens.download");
    CHECK(ReadAll(file) == gateway.service.content);
    CHECK(fetcher.bytes_received() == 4321);
    CHECK(!fs::exists(fs::path(file.string() + ".part")));
    fs::remove_all(work, ec);
}

TEST_CASE("cloud dataset fetch: what the gateway refuses or loses never binds", "[cloud]") {
    const fs::path work = fs::temp_directory_path() / "cyxwiz_cloud_fetch_test_fail";
    std::error_code ec;
    fs::remove_all(work, ec);
    FakeGateway gateway;
    gateway.service.content = Content(3000);
    gateway.service.chunk_size = 1000;
    CloudDatasetFetcher fetcher(work);
    fs::path file;
    std::string error;

    SECTION("a chunk the gateway cannot serve") {
        gateway.service.mode = Mode::DataLossAtChunk;
        gateway.service.fail_at = 1;
        REQUIRE_FALSE(fetcher.Fetch(Source(gateway), file, error));
        CHECK(error.find("DATA_LOSS") != std::string::npos);
        CHECK(error.find("chunk 2 of 3") != std::string::npos);
        CHECK(error.find("tiny causal lm tokens") != std::string::npos);
    }
    SECTION("a trailer whose hash does not match") {
        gateway.service.mode = Mode::WrongHash;
        REQUIRE_FALSE(fetcher.Fetch(Source(gateway), file, error));
        CHECK(error.find("SHA-256") != std::string::npos);
    }
    SECTION("a stream that ends without a last chunk") {
        gateway.service.mode = Mode::NoLastChunk;
        REQUIRE_FALSE(fetcher.Fetch(Source(gateway), file, error));
        CHECK(error.find("without a last chunk") != std::string::npos);
    }
    SECTION("a wrong token") {
        REQUIRE_FALSE(fetcher.Fetch(Source(gateway, "stale-token"), file, error));
        CHECK(error.find("UNAUTHENTICATED") != std::string::npos);
    }
    SECTION("a job without a token never calls the gateway") {
        REQUIRE_FALSE(fetcher.Fetch(Source(gateway, ""), file, error));
        CHECK(error.find("no access token") != std::string::npos);
        CHECK(gateway.service.requests == 0);
    }
    SECTION("a gateway that stops sending") {
        gateway.service.mode = Mode::Slow;
        CloudDatasetFetcher impatient(work, 500);
        REQUIRE_FALSE(impatient.Fetch(Source(gateway), file, error));
        CHECK(error.find("no data for") != std::string::npos);
    }
    // Nothing half-written stays behind.
    CHECK(!fs::exists(work / "tiny_causal_lm_tokens.download"));
    CHECK(!fs::exists(work / "tiny_causal_lm_tokens.download.part"));
    fs::remove_all(work, ec);
}

TEST_CASE("cloud dataset file names are safe", "[cloud]") {
    CHECK(SafeCloudFileName("tiny causal lm tokens") == "tiny_causal_lm_tokens");
    CHECK(SafeCloudFileName("a/b\\c:d") == "a_b_c_d");
    CHECK(SafeCloudFileName("") == "dataset");
}
