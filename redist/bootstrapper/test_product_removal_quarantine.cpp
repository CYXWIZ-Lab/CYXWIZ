#include "backend_pack_platform.h"
#include "product_removal_quarantine.h"
#include "product_removal_request.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

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
            ("cyxwiz-removal-quarantine-test-" + std::to_string(
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

void Touch(
    const std::filesystem::path& path,
    const std::string& content = "fixture\n") {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream << content;
    Check(static_cast<bool>(stream), "Fixture file creation must succeed");
}

struct ProductFixture {
    explicit ProductFixture(const std::filesystem::path& parent)
        : root(parent / "CyxWiz"), runtime_root(root / "runtime") {
        const auto base = runtime_root / "base" / "base-v1";
        std::filesystem::create_directories(base);
        Touch(root / std::string(
            cyxwiz::runtime::CurrentRuntimeBootstrapperExecutableName()));
        Touch(base / std::string(
            cyxwiz::runtime::CurrentEngineExecutableName()));
        Touch(
            base / "RUNTIME_VERSIONS.json",
            R"({"arrayfire":"3.10.0","cyxwiz":"0.2.0","python":"3.12.0"})");
        active.runtime_set_id = "set-v1";
        active.generation = 5;
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
    }

    cyxwiz::runtime::ProductRemovalAuthorization Queue() const {
        cyxwiz::runtime::ProductRemovalAuthorization authorization;
        std::string error;
        Check(cyxwiz::runtime::QueueProductRemovalRequest(
                  root, cyxwiz::runtime::ProductInstallScope::CurrentUser,
                  authorization, error),
              "Removal request fixture must queue: " + error);
        return authorization;
    }

    std::filesystem::path root;
    std::filesystem::path runtime_root;
    cyxwiz::runtime::ActiveRuntimeState active;
};

void TestAtomicallyQuarantinesExactInstallation() {
    TemporaryDirectory temporary;
    ProductFixture product(temporary.path());
    const auto authorization = product.Queue();
    const auto expected =
        cyxwiz::runtime::ProductRemovalQuarantinePath(authorization);
    cyxwiz::runtime::QuarantinedProductInstallation quarantined;
    std::string error;
    Check(cyxwiz::runtime::QuarantineProductInstallation(
              authorization, quarantined, error) &&
              quarantined.quarantine_root == expected &&
              !std::filesystem::exists(product.root) &&
              std::filesystem::is_directory(expected) &&
              std::filesystem::is_regular_file(
                  expected / ".cyxwiz-removal-request.json"),
          "An unchanged installation must move atomically into quarantine: " +
              error);
    Check(cyxwiz::runtime::ValidateQuarantinedProductInstallation(
              quarantined, error),
          "The exact relocated receipt must validate quarantine: " + error);
    quarantined.install_id.front() =
        quarantined.install_id.front() == '0' ? '1' : '0';
    Check(!cyxwiz::runtime::ValidateQuarantinedProductInstallation(
              quarantined, error),
          "A changed quarantine identity must fail closed");
}

void TestRejectsStaleAuthorizationWithoutMovingRoot() {
    TemporaryDirectory temporary;
    ProductFixture product(temporary.path());
    const auto authorization = product.Queue();
    auto changed = product.active;
    changed.generation = 6;
    std::string error;
    Check(cyxwiz::runtime::SaveActiveRuntimeStateAtomic(
              product.runtime_root / "active-runtime.json", changed, error),
          "Changed runtime fixture must publish: " + error);
    cyxwiz::runtime::QuarantinedProductInstallation quarantined;
    Check(!cyxwiz::runtime::QuarantineProductInstallation(
              authorization, quarantined, error) &&
              std::filesystem::is_directory(product.root),
          "Stale authorization must preserve the product root");
}

void TestRejectsExistingQuarantineWithoutOverwrite() {
    TemporaryDirectory temporary;
    ProductFixture product(temporary.path());
    const auto authorization = product.Queue();
    const auto quarantine =
        cyxwiz::runtime::ProductRemovalQuarantinePath(authorization);
    std::filesystem::create_directories(quarantine);
    Touch(quarantine / "unmanaged.txt");
    cyxwiz::runtime::QuarantinedProductInstallation output;
    std::string error;
    Check(!cyxwiz::runtime::QuarantineProductInstallation(
              authorization, output, error) &&
              std::filesystem::is_directory(product.root) &&
              std::filesystem::is_regular_file(
                  quarantine / "unmanaged.txt"),
          "An existing quarantine must never be replaced or merged");
}

}  // namespace

int main() {
    TestAtomicallyQuarantinesExactInstallation();
    TestRejectsStaleAuthorizationWithoutMovingRoot();
    TestRejectsExistingQuarantineWithoutOverwrite();
    std::cout << "Product removal quarantine contracts passed\n";
    return 0;
}
