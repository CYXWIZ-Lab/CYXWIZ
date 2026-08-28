#include "backend_pack_platform.h"
#include "product_removal_finalizer.h"
#include "product_removal_request.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <string>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <unistd.h>
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
            ("cyxwiz-removal-finalizer-test-" + std::to_string(
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
        active.generation = 7;
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

struct LifetimePipe {
    std::uintptr_t read_token = 0;
#ifdef _WIN32
    HANDLE write_handle = nullptr;
#else
    int write_descriptor = -1;
#endif
};

LifetimePipe OpenLifetimePipe() {
    LifetimePipe output;
#ifdef _WIN32
    HANDLE read_handle = nullptr;
    Check(::CreatePipe(&read_handle, &output.write_handle, nullptr, 0) != FALSE,
          "Lifetime pipe creation must succeed");
    output.read_token = reinterpret_cast<std::uintptr_t>(read_handle);
#else
    int descriptors[2] = {-1, -1};
    Check(::pipe(descriptors) == 0, "Lifetime pipe creation must succeed");
    output.read_token = static_cast<std::uintptr_t>(descriptors[0]);
    output.write_descriptor = descriptors[1];
#endif
    return output;
}

void CloseLifetimeWriter(LifetimePipe& pipe) {
#ifdef _WIN32
    Check(::CloseHandle(pipe.write_handle) != FALSE,
          "Lifetime pipe writer must close");
    pipe.write_handle = nullptr;
#else
    Check(::close(pipe.write_descriptor) == 0,
          "Lifetime pipe writer must close");
    pipe.write_descriptor = -1;
#endif
}

struct AwaitResult {
    bool accepted = false;
    cyxwiz::runtime::ProductRemovalAuthorization authorization;
    std::string error;
};

std::future<AwaitResult> BeginAwait(
    const ProductFixture& product,
    std::uintptr_t read_token) {
    return std::async(std::launch::async, [&product, read_token] {
        AwaitResult result;
        result.accepted = cyxwiz::runtime::AwaitAuthorizedProductRemoval(
            product.root, read_token, result.authorization, result.error);
        return result;
    });
}

void TestWaitsForParentLifetimeBeforeValidation() {
    TemporaryDirectory temporary;
    ProductFixture product(temporary.path());
    auto pipe = OpenLifetimePipe();
    auto result = BeginAwait(product, pipe.read_token);
    Check(result.wait_for(std::chrono::milliseconds(50)) ==
              std::future_status::timeout,
          "Finalizer must remain blocked while the parent owns its pipe end");
    CloseLifetimeWriter(pipe);
    const auto completed = result.get();
    Check(completed.accepted &&
              completed.authorization.install_root == product.root,
          "Finalizer must accept the unchanged request after parent EOF: " +
              completed.error);
}

void TestRevalidatesAfterParentLifetimeEnds() {
    TemporaryDirectory temporary;
    ProductFixture product(temporary.path());
    auto pipe = OpenLifetimePipe();
    auto result = BeginAwait(product, pipe.read_token);
    Check(result.wait_for(std::chrono::milliseconds(50)) ==
              std::future_status::timeout,
          "Stale-request test must reach the lifetime wait boundary");
    auto changed = product.active;
    ++changed.generation;
    std::string error;
    Check(cyxwiz::runtime::SaveActiveRuntimeStateAtomic(
              product.runtime_root / "active-runtime.json", changed, error),
          "Changed runtime fixture must publish: " + error);
    CloseLifetimeWriter(pipe);
    const auto completed = result.get();
    Check(!completed.accepted && completed.authorization.install_root.empty(),
          "Finalizer must reject a request changed while its parent was alive");
}

}  // namespace

int main() {
    TestWaitsForParentLifetimeBeforeValidation();
    TestRevalidatesAfterParentLifetimeEnds();
    std::cout << "Product removal finalizer contracts passed\n";
    return 0;
}
