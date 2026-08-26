#include "backend_pack_platform.h"
#include "product_removal_handoff.h"
#include "product_removal_request.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

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
            ("cyxwiz-removal-handoff-test-" + std::to_string(
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
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream << "fixture\n";
    Check(static_cast<bool>(stream), "Fixture file creation must succeed");
}

struct ProductFixture {
    ProductFixture(
        const std::filesystem::path& parent,
        const std::filesystem::path& built_finalizer)
        : root(parent / "CyxWiz"), runtime_root(root / "runtime") {
        const auto base = runtime_root / "base" / "base-v1";
        std::filesystem::create_directories(base);
        Touch(root / std::string(
            cyxwiz::runtime::CurrentRuntimeBootstrapperExecutableName()));
        Touch(base / std::string(
            cyxwiz::runtime::CurrentEngineExecutableName()));
        std::filesystem::copy_file(
            built_finalizer,
            root / std::string(
                cyxwiz::runtime::CurrentProductRemovalFinalizerExecutableName()));
#ifndef _WIN32
        std::filesystem::permissions(
            root / std::string(
                cyxwiz::runtime::CurrentProductRemovalFinalizerExecutableName()),
            std::filesystem::perms::owner_exec,
            std::filesystem::perm_options::add);
#endif
        active.runtime_set_id = "set-v1";
        active.generation = 9;
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
    }

    std::filesystem::path root;
    std::filesystem::path runtime_root;
    cyxwiz::runtime::ActiveRuntimeState active;
};

std::string AwaitResult(cyxwiz::runtime::ProductRemovalHandoff& handoff) {
    handoff.parent_lifetime.Close();
    for (int attempt = 0; attempt < 500; ++attempt) {
        std::error_code error;
        if (std::filesystem::is_regular_file(handoff.result_path, error) &&
            !error) {
            std::ifstream stream(handoff.result_path, std::ios::binary);
            return std::string(
                std::istreambuf_iterator<char>(stream),
                std::istreambuf_iterator<char>());
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return {};
}

void RemoveStaging(const cyxwiz::runtime::ProductRemovalHandoff& handoff) {
    std::error_code ignored;
    std::filesystem::remove(handoff.result_path, ignored);
    std::filesystem::remove(handoff.staged_finalizer, ignored);
    std::filesystem::remove(handoff.staged_finalizer.parent_path(), ignored);
}

void TestNoRequestDoesNotLaunch(const std::filesystem::path& temporary) {
    std::string error;
    auto handoff = cyxwiz::runtime::SchedulePendingProductRemoval(
        temporary / "missing-product", error);
    Check(handoff.status ==
              cyxwiz::runtime::ProductRemovalHandoffStatus::NoRequest &&
              !handoff.parent_lifetime.valid() && error.empty(),
          "An absent request must be an inert no-op");
}

void TestLaunchWaitsForExplicitParentClose(
    const std::filesystem::path& temporary,
    const std::filesystem::path& finalizer) {
    ProductFixture product(temporary, finalizer);
    std::string error;
    auto handoff = cyxwiz::runtime::SchedulePendingProductRemoval(
        product.root, error);
    Check(handoff.status ==
              cyxwiz::runtime::ProductRemovalHandoffStatus::Scheduled &&
              handoff.parent_lifetime.valid() &&
              std::filesystem::is_regular_file(handoff.staged_finalizer),
          "A valid request must stage and launch its detached finalizer: " +
              error);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    Check(!std::filesystem::exists(handoff.result_path),
          "Detached finalizer must not validate before parent EOF");
    Check(AwaitResult(handoff) == "authorized\n",
          "An unchanged request must authorize after parent EOF");
    RemoveStaging(handoff);
}

void TestDetachedFinalizerRejectsStateChangedDuringWait(
    const std::filesystem::path& temporary,
    const std::filesystem::path& finalizer) {
    ProductFixture product(temporary, finalizer);
    std::string error;
    auto handoff = cyxwiz::runtime::SchedulePendingProductRemoval(
        product.root, error);
    Check(handoff.status ==
              cyxwiz::runtime::ProductRemovalHandoffStatus::Scheduled,
          "Stale-state fixture must launch: " + error);
    auto changed = product.active;
    ++changed.generation;
    Check(cyxwiz::runtime::SaveActiveRuntimeStateAtomic(
              product.runtime_root / "active-runtime.json", changed, error),
          "Changed runtime fixture must publish: " + error);
    Check(AwaitResult(handoff) == "rejected\n",
          "Detached finalizer must reject state changed before parent EOF");
    RemoveStaging(handoff);
}

void TestRejectsOversizedFinalizerBeforeLaunch(
    const std::filesystem::path& temporary,
    const std::filesystem::path& finalizer) {
    ProductFixture product(temporary, finalizer);
    const auto installed_finalizer = product.root / std::string(
        cyxwiz::runtime::CurrentProductRemovalFinalizerExecutableName());
    std::error_code filesystem_error;
    std::filesystem::resize_file(
        installed_finalizer, 16 * 1024 * 1024 + 1, filesystem_error);
    Check(!filesystem_error, "Oversized finalizer fixture must publish");
    std::string error;
    auto handoff = cyxwiz::runtime::SchedulePendingProductRemoval(
        product.root, error);
    Check(handoff.status ==
              cyxwiz::runtime::ProductRemovalHandoffStatus::Rejected &&
              !handoff.parent_lifetime.valid(),
          "An oversized finalizer must fail before process launch");
}

}  // namespace

int main(int argc, char** argv) {
    Check(argc > 0, "The test executable path is required");
    std::error_code error;
    const auto binary_directory = std::filesystem::canonical(
        std::filesystem::absolute(argv[0]), error).parent_path();
    Check(!error, "The test executable directory must resolve");
    const auto finalizer = binary_directory /
        std::string(
            cyxwiz::runtime::CurrentProductRemovalFinalizerExecutableName());
    Check(std::filesystem::is_regular_file(finalizer),
          "The built removal finalizer is required");
    TemporaryDirectory temporary;
    TestNoRequestDoesNotLaunch(temporary.path());
    TestLaunchWaitsForExplicitParentClose(
        temporary.path() / "accepted", finalizer);
    TestDetachedFinalizerRejectsStateChangedDuringWait(
        temporary.path() / "rejected", finalizer);
    TestRejectsOversizedFinalizerBeforeLaunch(
        temporary.path() / "oversized", finalizer);
    std::cout << "Product removal handoff contracts passed\n";
    return 0;
}
