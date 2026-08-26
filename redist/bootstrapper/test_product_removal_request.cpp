#include "backend_pack_platform.h"
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
            ("cyxwiz-removal-request-test-" + std::to_string(
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

void WriteText(const std::filesystem::path& path, const std::string& content) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream << content;
    Check(static_cast<bool>(stream), "Fixture write must succeed");
}

struct ProductFixture {
    explicit ProductFixture(
        const std::filesystem::path& parent,
        std::string name = "CyxWiz")
        : root(parent / std::move(name)), runtime_root(root / "runtime") {
        const auto base = runtime_root / "base" / "base-v1";
        std::filesystem::create_directories(base);
        WriteText(
            root / std::string(
                cyxwiz::runtime::CurrentRuntimeBootstrapperExecutableName()),
            "launcher\n");
        WriteText(
            base / std::string(
                cyxwiz::runtime::CurrentEngineExecutableName()),
            "engine\n");
        active.runtime_set_id = "set-v1";
        active.generation = 3;
        active.base_pack_id = "base-v1";
        std::string error;
        Check(cyxwiz::runtime::SaveActiveRuntimeStateAtomic(
                  runtime_root / "active-runtime.json", active, error),
              "Active runtime fixture must publish: " + error);
        cyxwiz::runtime::ProductInstallationReceipt receipt;
        Check(cyxwiz::runtime::PublishProductInstallationReceipt(
                  root, cyxwiz::runtime::ProductInstallScope::CurrentUser,
                  receipt, error),
              "Product receipt fixture must publish: " + error);
    }

    std::filesystem::path root;
    std::filesystem::path runtime_root;
    cyxwiz::runtime::ActiveRuntimeState active;
};

void TestQueuesAndLoadsExactAuthorization() {
    TemporaryDirectory temporary;
    ProductFixture product(temporary.path());
    cyxwiz::runtime::ProductRemovalAuthorization queued;
    std::string error;
    Check(cyxwiz::runtime::QueueProductRemovalRequest(
              product.root,
              cyxwiz::runtime::ProductInstallScope::CurrentUser,
              queued, error) &&
              queued.install_root == product.root &&
              queued.runtime.generation == 3,
          "An exact installation must queue removal: " + error);
    cyxwiz::runtime::ProductRemovalAuthorization loaded;
    Check(cyxwiz::runtime::LoadProductRemovalRequest(
              product.root, loaded, error) &&
              loaded.install_id == queued.install_id &&
              loaded.runtime.base_pack_id == "base-v1",
          "A fresh exact removal request must load: " + error);

    auto changed = product.active;
    changed.generation = 4;
    Check(cyxwiz::runtime::SaveActiveRuntimeStateAtomic(
              product.runtime_root / "active-runtime.json", changed, error),
          "Changed active state fixture must publish: " + error);
    Check(!cyxwiz::runtime::LoadProductRemovalRequest(
              product.root, loaded, error) &&
              error.find("stale or invalid") != std::string::npos,
          "A runtime mutation must invalidate a queued removal request");
}

void TestRejectsCopiedAndUnknownFieldRequests() {
    TemporaryDirectory temporary;
    ProductFixture first(temporary.path(), "First");
    ProductFixture second(temporary.path(), "Second");
    cyxwiz::runtime::ProductRemovalAuthorization queued;
    std::string error;
    Check(cyxwiz::runtime::QueueProductRemovalRequest(
              first.root,
              cyxwiz::runtime::ProductInstallScope::CurrentUser,
              queued, error),
          "Copied-request fixture must queue removal: " + error);
    std::filesystem::copy_file(
        cyxwiz::runtime::ProductRemovalRequestPath(first.root),
        cyxwiz::runtime::ProductRemovalRequestPath(second.root),
        std::filesystem::copy_options::overwrite_existing);
    cyxwiz::runtime::ProductRemovalAuthorization loaded;
    Check(!cyxwiz::runtime::LoadProductRemovalRequest(
              second.root, loaded, error) &&
              error.find("another root") != std::string::npos,
          "A copied removal request must not authorize another root");

    WriteText(
        cyxwiz::runtime::ProductRemovalRequestPath(first.root),
        R"({"schema_version":1,"kind":"cyxwiz-product-removal","install_root":"x","scope":"current_user","install_id":"x","runtime":{},"unknown":true})");
    Check(!cyxwiz::runtime::LoadProductRemovalRequest(
              first.root, loaded, error) &&
              error.find("schema") != std::string::npos,
          "Unknown request fields must fail closed");

    Check(cyxwiz::runtime::QueueProductRemovalRequest(
              first.root,
              cyxwiz::runtime::ProductInstallScope::CurrentUser,
              queued, error),
          "A fresh explicit removal choice may replace a corrupt stale request");
}

void TestRejectsWrongScopeAndUnsafeRequestFile() {
    TemporaryDirectory temporary;
    ProductFixture product(temporary.path());
    cyxwiz::runtime::ProductRemovalAuthorization queued;
    std::string error;
    Check(!cyxwiz::runtime::QueueProductRemovalRequest(
              product.root,
              cyxwiz::runtime::ProductInstallScope::AllUsers,
              queued, error) &&
              error.find("scope") != std::string::npos,
          "A scope mismatch must not queue product removal");
    std::filesystem::create_directory(
        cyxwiz::runtime::ProductRemovalRequestPath(product.root));
    Check(!cyxwiz::runtime::LoadProductRemovalRequest(
              product.root, queued, error) &&
              error.find("regular file") != std::string::npos,
          "A nonregular request path must fail closed");
}

}  // namespace

int main() {
    TestQueuesAndLoadsExactAuthorization();
    TestRejectsCopiedAndUnknownFieldRequests();
    TestRejectsWrongScopeAndUnsafeRequestFile();
    std::cout << "Product removal request contracts passed\n";
    return 0;
}
