// A node pulls a dataset file from CyxCloud (TOFIX164 slice 3): a fake
// gateway DataService in this process serves a file in chunks the way the
// real one does after slice 2 (token required, DATA_LOSS instead of a skip,
// SHA-256 trailer), and the fetcher must accept only what verifies.
#include "cloud_dataset_fetcher.h"
#include "fake_cloud_gateway.h"

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;
using cyxwiz::server_node::CloudDatasetFetcher;
using cyxwiz::server_node::CloudFileExtension;
using cyxwiz::server_node::SafeCloudFileName;

namespace {

using cyxwiz::server_node::test::FakeGateway;
using Mode = cyxwiz::server_node::test::FakeGatewayMode;

cyxwiz::protocol::CloudDatasetSource Source(const FakeGateway& gateway, const std::string& token = "good-token") {
    cyxwiz::protocol::CloudDatasetSource source;
    source.set_dataset_name("tiny causal lm tokens");
    source.set_gateway_address(gateway.address());
    source.set_file_id("2f1c7f8e-7d3a-4b4e-9c2d-1a2b3c4d5e6f");
    source.set_access_token(token);
    source.set_file_name("tokens.arrow");
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
    CHECK(file.filename().string() == "tiny_causal_lm_tokens.arrow");
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
    CHECK(!fs::exists(work / "tiny_causal_lm_tokens.arrow"));
    CHECK(!fs::exists(work / "tiny_causal_lm_tokens.arrow.part"));
    fs::remove_all(work, ec);
}

TEST_CASE("cloud dataset file names are safe and keep the format", "[cloud]") {
    CHECK(SafeCloudFileName("tiny causal lm tokens") == "tiny_causal_lm_tokens");
    CHECK(SafeCloudFileName("a/b\\c:d") == "a_b_c_d");
    CHECK(SafeCloudFileName("") == "dataset");
    CHECK(CloudFileExtension("tokens.parquet") == ".parquet");
    CHECK(CloudFileExtension("TOKENS.PARQUET") == ".parquet");
    CHECK(CloudFileExtension("tokens.arrow") == ".arrow");
    CHECK(CloudFileExtension("tokens.feather") == ".arrow");
    CHECK(CloudFileExtension("") == ".arrow");
}
