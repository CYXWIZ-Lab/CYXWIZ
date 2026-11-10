// The P2P secret comes from a file kept to its owner, never the command line
// (found 2026-11-10: the secret was readable in the process list).
#include "core/secret_file.h"

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <random>

using cyxwiz::servernode::core::ReadSecretFile;
namespace fs = std::filesystem;

namespace {

// A fresh folder for one test, removed afterwards.
struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() / ("cyxwiz-secret-test-" + std::to_string(std::random_device{}()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

fs::path WriteFile(const fs::path& path, const std::string& text) {
    std::ofstream(path, std::ios::binary) << text;
#ifndef _WIN32
    fs::permissions(path, fs::perms::owner_read | fs::perms::owner_write, fs::perm_options::replace);
#endif
    return path;
}

}  // namespace

TEST_CASE("the secret is read without its line end or surrounding spaces", "[secret_file]") {
    TempDir dir;
    const auto result = ReadSecretFile(WriteFile(dir.path / "p2p.secret", "  abc123\r\n"));
    CHECK(result.error.empty());
    CHECK(result.secret == "abc123");
}

TEST_CASE("a missing or empty secret file is refused", "[secret_file]") {
    TempDir dir;
    CHECK_FALSE(ReadSecretFile(dir.path / "missing.secret").error.empty());
    CHECK_FALSE(ReadSecretFile(WriteFile(dir.path / "empty.secret", " \n")).error.empty());
    CHECK_FALSE(ReadSecretFile(dir.path).error.empty());  // a folder, not a file
}

#ifndef _WIN32
TEST_CASE("a secret file other users can read is refused", "[secret_file]") {
    TempDir dir;
    const auto path = WriteFile(dir.path / "p2p.secret", "abc123\n");
    fs::permissions(path, fs::perms::group_read | fs::perms::others_read, fs::perm_options::add);
    const auto result = ReadSecretFile(path);
    CHECK(result.secret.empty());
    CHECK(result.error.find("chmod 600") != std::string::npos);
}
#endif
