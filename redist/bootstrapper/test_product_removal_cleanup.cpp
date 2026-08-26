#include "backend_pack_platform.h"
#include "product_removal_cleanup.h"
#include "product_removal_request.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        path_ = std::filesystem::temp_directory_path() /
            ("cyxwiz-removal-cleanup-test-" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(path_);
        path_ = std::filesystem::canonical(path_);
    }
    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }
    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

void Touch(const std::filesystem::path& path) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream << "fixture\n";
    Check(static_cast<bool>(stream), "Fixture file creation must succeed");
}

struct QuarantineFixture {
    explicit QuarantineFixture(const std::filesystem::path& parent)
        : root(parent / "CyxWiz"), runtime_root(root / "runtime") {
        const auto base = runtime_root / "base" / "base-v1";
        std::filesystem::create_directories(base);
        Touch(root / std::string(
            cyxwiz::runtime::CurrentRuntimeBootstrapperExecutableName()));
        Touch(base / std::string(
            cyxwiz::runtime::CurrentEngineExecutableName()));
        Touch(base / "resources" / "nested" / "asset.bin");
        cyxwiz::runtime::ActiveRuntimeState active;
        active.runtime_set_id = "set-v1";
        active.generation = 11;
        active.base_pack_id = "base-v1";
        std::string error;
        Check(cyxwiz::runtime::SaveActiveRuntimeStateAtomic(
                  runtime_root / "active-runtime.json", active, error),
              "Active state fixture must publish: " + error);
        cyxwiz::runtime::ProductInstallationReceipt receipt;
        Check(cyxwiz::runtime::PublishProductInstallationReceipt(
                  root, cyxwiz::runtime::ProductInstallScope::CurrentUser,
                  receipt, error),
              "Receipt fixture must publish: " + error);
        cyxwiz::runtime::ProductRemovalAuthorization authorization;
        Check(cyxwiz::runtime::QueueProductRemovalRequest(
                  root, cyxwiz::runtime::ProductInstallScope::CurrentUser,
                  authorization, error),
              "Removal request fixture must queue: " + error);
        Check(cyxwiz::runtime::QuarantineProductInstallation(
                  authorization, quarantined, error),
              "Fixture quarantine must succeed: " + error);
    }

    std::filesystem::path root;
    std::filesystem::path runtime_root;
    cyxwiz::runtime::QuarantinedProductInstallation quarantined;
};

void TestRemovesPayloadWithoutFollowingDirectoryLink() {
    TemporaryDirectory temporary;
    QuarantineFixture product(temporary.path());
    const auto outside = temporary.path() / "outside";
    Touch(outside / "keep.txt");
    std::error_code link_error;
    std::filesystem::create_directory_symlink(
        outside, product.quarantined.quarantine_root / "outside-link",
        link_error);
    cyxwiz::runtime::ProductRemovalCleanupResult result;
    std::string error;
    const bool cleaned =
        cyxwiz::runtime::CleanupQuarantinedProductInstallation(
            product.quarantined, result, error);
    const bool quarantine_absent =
        !std::filesystem::exists(product.quarantined.quarantine_root);
    const bool target_preserved =
        std::filesystem::is_regular_file(outside / "keep.txt");
    Check(cleaned && result.complete && quarantine_absent && target_preserved,
          "Cleanup must remove only the quarantine and preserve link targets: " +
              error + " cleaned=" + std::to_string(cleaned) +
              " complete=" + std::to_string(result.complete) +
              " quarantine_absent=" + std::to_string(quarantine_absent) +
              " target_preserved=" + std::to_string(target_preserved));
    if (!link_error) {
        Check(result.removed_entries >= 9,
              "Cleanup must count the removed link and owned payload");
    }
}

void TestRejectsChangedQuarantineIdentity() {
    TemporaryDirectory temporary;
    QuarantineFixture product(temporary.path());
    auto changed = product.quarantined;
    changed.install_id.front() =
        changed.install_id.front() == '0' ? '1' : '0';
    cyxwiz::runtime::ProductRemovalCleanupResult result;
    std::string error;
    Check(!cyxwiz::runtime::CleanupQuarantinedProductInstallation(
              changed, result, error) && !result.complete &&
              std::filesystem::is_regular_file(
                  product.quarantined.quarantine_root /
                  ".cyxwiz-installation.json"),
          "Changed quarantine identity must preserve all recovery evidence");
}

void TestCompletesRecoveryAfterRequestWasAlreadyRemoved() {
    TemporaryDirectory temporary;
    QuarantineFixture product(temporary.path());
    std::filesystem::remove(
        product.quarantined.quarantine_root /
        ".cyxwiz-removal-request.json");
    cyxwiz::runtime::ProductRemovalCleanupResult result;
    std::string error;
    Check(cyxwiz::runtime::CleanupQuarantinedProductInstallation(
              product.quarantined, result, error) && result.complete &&
              !std::filesystem::exists(product.quarantined.quarantine_root),
          "Cleanup retry must finish after request evidence was removed: " +
              error);
}

void TestPayloadFailurePreservesRecoveryEvidence() {
    TemporaryDirectory temporary;
    QuarantineFixture product(temporary.path());
    const auto blocked =
        product.quarantined.quarantine_root / "blocked.bin";
    Touch(blocked);
#ifdef _WIN32
    const HANDLE held = ::CreateFileW(
        blocked.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    Check(held != INVALID_HANDLE_VALUE, "Blocked cleanup fixture must open");
#else
    std::error_code permission_error;
    std::filesystem::permissions(
        product.quarantined.quarantine_root,
        std::filesystem::perms::owner_read |
            std::filesystem::perms::owner_exec,
        std::filesystem::perm_options::replace, permission_error);
    Check(!permission_error, "Blocked cleanup permissions must publish");
#endif
    cyxwiz::runtime::ProductRemovalCleanupResult result;
    std::string error;
    Check(!cyxwiz::runtime::CleanupQuarantinedProductInstallation(
              product.quarantined, result, error) && !result.complete &&
              std::filesystem::is_regular_file(
                  product.quarantined.quarantine_root /
                  ".cyxwiz-installation.json") &&
              std::filesystem::is_regular_file(
                  product.quarantined.quarantine_root /
                  ".cyxwiz-removal-request.json"),
          "A payload cleanup failure must preserve both recovery documents");
#ifdef _WIN32
    ::CloseHandle(held);
#else
    std::filesystem::permissions(
        product.quarantined.quarantine_root,
        std::filesystem::perms::owner_all,
        std::filesystem::perm_options::add);
#endif
}

}  // namespace

int main() {
    TestRemovesPayloadWithoutFollowingDirectoryLink();
    TestRejectsChangedQuarantineIdentity();
    TestCompletesRecoveryAfterRequestWasAlreadyRemoved();
    TestPayloadFailurePreservesRecoveryEvidence();
    std::cout << "Product removal cleanup contracts passed\n";
    return 0;
}
