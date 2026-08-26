#include "backend_pack_platform.h"
#include "product_removal_authorization.h"

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
            ("cyxwiz-removal-authorization-test-" + std::to_string(
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

void WriteFixture(
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
        WriteFixture(root / std::string(
            cyxwiz::runtime::CurrentRuntimeBootstrapperExecutableName()));
        WriteFixture(base / std::string(
            cyxwiz::runtime::CurrentEngineExecutableName()));
        WriteFixture(
            base / "RUNTIME_VERSIONS.json",
            R"({"arrayfire":"3.10.0","cyxwiz":"0.2.0","python":"3.12.0"})");
        active.runtime_set_id = "set-v1";
        active.generation = 7;
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

void TestCapturesAndRevalidatesExactIdentity() {
    TemporaryDirectory temporary;
    ProductFixture product(temporary.path());
    cyxwiz::runtime::ProductRemovalAuthorization authorization;
    std::string error;
    Check(cyxwiz::runtime::CaptureProductRemovalAuthorization(
              product.root,
              cyxwiz::runtime::ProductInstallScope::CurrentUser,
              authorization, error) &&
              authorization.install_id.size() == 32 &&
              authorization.product_version == "0.2.0" &&
              authorization.runtime.runtime_set_id == "set-v1" &&
              authorization.runtime.generation == 7,
          "A complete exact installation must produce removal authorization: " +
              error);
    Check(cyxwiz::runtime::ValidateProductRemovalAuthorization(
              authorization, error),
          "An unchanged installation must retain removal authorization: " +
              error);

    const auto versions = product.runtime_root / "base" / "base-v1" /
        "RUNTIME_VERSIONS.json";
    WriteFixture(
        versions,
        R"({"arrayfire":"3.10.0","cyxwiz":"0.3.0","python":"3.12.0"})");
    Check(!cyxwiz::runtime::ValidateProductRemovalAuthorization(
              authorization, error) &&
              error.find("release version changed") != std::string::npos,
          "A product release version change must invalidate removal");
    WriteFixture(
        versions,
        R"({"arrayfire":"3.10.0","cyxwiz":"0.2.0","python":"3.12.0"})");

    auto changed = product.active;
    changed.generation = 8;
    Check(cyxwiz::runtime::SaveActiveRuntimeStateAtomic(
              product.runtime_root / "active-runtime.json", changed, error),
          "Changed runtime fixture must publish: " + error);
    Check(!cyxwiz::runtime::ValidateProductRemovalAuthorization(
              authorization, error) &&
              error.find("active runtime changed") != std::string::npos,
          "A runtime generation change must invalidate removal");

    changed = product.active;
    changed.base_pack_id = "base-v2";
    const auto replacement_base =
        product.runtime_root / "base" / changed.base_pack_id;
    std::filesystem::create_directories(replacement_base);
    WriteFixture(replacement_base / std::string(
        cyxwiz::runtime::CurrentEngineExecutableName()));
    WriteFixture(
        replacement_base / "RUNTIME_VERSIONS.json",
        R"({"arrayfire":"3.10.0","cyxwiz":"0.2.0","python":"3.12.0"})");
    Check(cyxwiz::runtime::SaveActiveRuntimeStateAtomic(
              product.runtime_root / "active-runtime.json", changed, error),
          "Changed-base fixture must publish: " + error);
    Check(!cyxwiz::runtime::ValidateProductRemovalAuthorization(
              authorization, error) &&
              error.find("active runtime changed") != std::string::npos,
          "A same-generation base change must invalidate removal");
}

void TestRejectsChangedReceiptAndScope() {
    TemporaryDirectory temporary;
    ProductFixture product(temporary.path());
    cyxwiz::runtime::ProductRemovalAuthorization authorization;
    std::string error;
    Check(!cyxwiz::runtime::CaptureProductRemovalAuthorization(
              product.root, cyxwiz::runtime::ProductInstallScope::AllUsers,
              authorization, error) &&
              error.find("scope") != std::string::npos,
          "A mismatched install scope must not authorize removal");
    Check(cyxwiz::runtime::CaptureProductRemovalAuthorization(
              product.root,
              cyxwiz::runtime::ProductInstallScope::CurrentUser,
              authorization, error),
          "Receipt-change fixture must capture authorization");
    authorization.install_id.front() =
        authorization.install_id.front() == '0' ? '1' : '0';
    Check(!cyxwiz::runtime::ValidateProductRemovalAuthorization(
              authorization, error) &&
              error.find("installation identity changed") !=
                  std::string::npos,
          "A changed installation identity must invalidate removal");
}

void TestRejectsMissingStableProductBoundary() {
    TemporaryDirectory temporary;
    ProductFixture product(temporary.path());
    std::filesystem::remove(
        product.root / std::string(
            cyxwiz::runtime::CurrentRuntimeBootstrapperExecutableName()));
    cyxwiz::runtime::ProductRemovalAuthorization authorization;
    std::string error;
    Check(!cyxwiz::runtime::CaptureProductRemovalAuthorization(
              product.root,
              cyxwiz::runtime::ProductInstallScope::CurrentUser,
              authorization, error) &&
              error.find("launcher") != std::string::npos,
          "A product without its stable launcher must not authorize removal");
}

void TestRejectsMissingReleaseIdentity() {
    TemporaryDirectory temporary;
    ProductFixture product(temporary.path());
    std::filesystem::remove(
        product.runtime_root / "base" / "base-v1" /
        "RUNTIME_VERSIONS.json");
    cyxwiz::runtime::ProductRemovalAuthorization authorization;
    std::string error;
    Check(!cyxwiz::runtime::CaptureProductRemovalAuthorization(
              product.root,
              cyxwiz::runtime::ProductInstallScope::CurrentUser,
              authorization, error) &&
              error.find("release identity") != std::string::npos,
          "A product without signed release identity must not authorize removal");
}

}  // namespace

int main() {
    TestCapturesAndRevalidatesExactIdentity();
    TestRejectsChangedReceiptAndScope();
    TestRejectsMissingStableProductBoundary();
    TestRejectsMissingReleaseIdentity();
    std::cout << "Product removal authorization contracts passed\n";
    return 0;
}
