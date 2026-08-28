#include "installer/installer_product_removal.h"

#include "backend_pack_platform.h"
#include "product_installation_receipt.h"
#include "product_removal_request.h"
#include "runtime_layout.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

void Check(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

class TemporaryDirectory {
public:
  TemporaryDirectory() {
    path_ = std::filesystem::temp_directory_path() /
            ("cyxwiz-installer-removal-test-" + std::to_string(
                 std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(path_);
    path_ = std::filesystem::canonical(path_);
  }
  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }
  const std::filesystem::path &path() const { return path_; }

private:
  std::filesystem::path path_;
};

void Write(const std::filesystem::path &path, const std::string &content) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream << content;
  Check(static_cast<bool>(stream), "Fixture write must succeed");
}

struct ProductFixture {
  explicit ProductFixture(const std::filesystem::path &parent)
      : root(parent / "CyxWiz"), runtime_root(root / "runtime") {
    const auto base = runtime_root / "base" / "base-v1";
    std::filesystem::create_directories(base);
    Write(root / std::string(
                     cyxwiz::runtime::CurrentRuntimeBootstrapperExecutableName()),
          "launcher\n");
    Write(root / std::string(
                     cyxwiz::runtime::CurrentProductRemovalFinalizerExecutableName()),
          "finalizer\n");
    Write(base / std::string(cyxwiz::runtime::CurrentEngineExecutableName()),
          "engine\n");
    Write(base / "RUNTIME_VERSIONS.json",
          R"({"arrayfire":"3.10.0","cyxwiz":"0.2.0","python":"3.12.0"})");
    cyxwiz::runtime::ActiveRuntimeState active;
    active.runtime_set_id = "set-v1";
    active.generation = 1;
    active.base_pack_id = "base-v1";
    std::string error;
    Check(cyxwiz::runtime::SaveActiveRuntimeStateAtomic(
              runtime_root / "active-runtime.json", active, error),
          "Active runtime fixture must publish: " + error);
    cyxwiz::runtime::ProductInstallationReceipt receipt;
    Check(cyxwiz::runtime::PublishProductInstallationReceipt(
              root, cyxwiz::runtime::ProductInstallScope::CurrentUser,
              receipt, error),
          "Installation receipt fixture must publish: " + error);
  }

  std::filesystem::path root;
  std::filesystem::path runtime_root;
};

void TestRequiresStableBootstrapperHost() {
  TemporaryDirectory temporary;
  ProductFixture product(temporary.path());
  const auto direct = cyxwiz::installer::InspectInstallerProductRemoval(
      product.runtime_root, false);
  Check(direct.installed && !direct.available &&
            direct.message.find("through CyxWiz Installer") !=
                std::string::npos,
        "Direct base-GUI launch must not offer an orphaned removal request");

  const auto hosted = cyxwiz::installer::InspectInstallerProductRemoval(
      product.runtime_root, true);
  Check(hosted.installed && hosted.available &&
            hosted.install_root == product.root,
        "A complete stable-hosted installation must offer full removal");
}

void TestQueuesExactSchemaTwoRequest() {
  TemporaryDirectory temporary;
  ProductFixture product(temporary.path());
  const auto state = cyxwiz::installer::InspectInstallerProductRemoval(
      product.runtime_root, true);
  std::string message;
  Check(cyxwiz::installer::QueueInstallerProductRemoval(state, message),
        "Explicit GUI confirmation must queue product removal: " + message);
  cyxwiz::runtime::ProductRemovalAuthorization queued;
  std::string error;
  Check(cyxwiz::runtime::LoadProductRemovalRequest(
            product.root, queued, error) &&
            queued.install_root == product.root &&
            queued.product_version == "0.2.0",
        "The GUI must queue the exact current product identity: " + error);
}

void TestRejectsMissingFinalizer() {
  TemporaryDirectory temporary;
  ProductFixture product(temporary.path());
  std::filesystem::remove(
      product.root / std::string(
                         cyxwiz::runtime::CurrentProductRemovalFinalizerExecutableName()));
  const auto state = cyxwiz::installer::InspectInstallerProductRemoval(
      product.runtime_root, true);
  Check(state.installed && !state.available &&
            state.message.find("finalizer") != std::string::npos,
        "Removal must stay disabled without the verified finalizer");
}

} // namespace

int main() {
  TestRequiresStableBootstrapperHost();
  TestQueuesExactSchemaTwoRequest();
  TestRejectsMissingFinalizer();
  std::cout << "Installer product removal contracts passed\n";
  return 0;
}
