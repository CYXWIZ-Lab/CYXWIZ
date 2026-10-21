#include "installer/installer_external_session.h"
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
            ("cyxwiz-installer-removal-test-" +
             std::to_string(
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
    Write(root /
              std::string(
                  cyxwiz::runtime::CurrentRuntimeBootstrapperExecutableName()),
          "launcher\n");
    Write(root /
              std::string(cyxwiz::runtime::
                              CurrentProductRemovalFinalizerExecutableName()),
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
              root, cyxwiz::runtime::ProductInstallScope::CurrentUser, receipt,
              error),
          "Installation receipt fixture must publish: " + error);
  }

  std::filesystem::path root;
  std::filesystem::path runtime_root;
};

void TestRequiresExternalSession() {
  TemporaryDirectory temporary;
  ProductFixture product(temporary.path());
  const auto direct = cyxwiz::installer::InspectInstallerProductRemoval(
      product.runtime_root, false);
  Check(direct.installed && !direct.available && direct.requires_stable_host &&
            direct.message.find("external maintenance session") !=
                std::string::npos,
        "Direct base-GUI launch must not offer an orphaned removal request");

  const auto hosted = cyxwiz::installer::InspectInstallerProductRemoval(
      product.runtime_root, true);
  Check(hosted.installed && hosted.available && !hosted.requires_stable_host &&
            hosted.install_root == product.root,
        "A complete installation must offer removal from an external session");
}

void TestExternalRemovalRejectsInUseProduct() {
  TemporaryDirectory temporary;
  ProductFixture product(temporary.path());
  const auto state = cyxwiz::installer::InspectInstallerProductRemoval(
      product.runtime_root, true);
  const auto result = cyxwiz::installer::RemoveInstallerProduct(
      state, product.root / "runtime/base/base-v1");
  Check(
      !result.succeeded && std::filesystem::exists(product.root),
      "In-product manager must fail before any registration or file mutation");
  Check(!std::filesystem::exists(
            cyxwiz::runtime::ProductRemovalRequestPath(product.root)),
        "Rejected external removal must not queue an orphan request");
}

void TestExternalSessionStaging() {
  TemporaryDirectory temporary;
  const auto source = temporary.path() / "source";
  const auto metadata = temporary.path() / "metadata";
  const auto output = temporary.path() / "session";
  std::filesystem::create_directories(source);
  std::filesystem::create_directories(metadata / "catalogs/manifests");
  for (const auto name :
       {cyxwiz::runtime::CurrentInstallerManagerExecutableName(),
        cyxwiz::runtime::CurrentBackendPackInstallerExecutableName(),
        cyxwiz::runtime::CurrentRuntimeBootstrapperExecutableName(),
        cyxwiz::runtime::CurrentProductRemovalFinalizerExecutableName()})
    Write(source / name, "tool");
  Write(source / "dependency.bin", "loaded library");
  Write(source / "engine-data.bin", "must not copy");
  Write(metadata / "catalogs/manifests/cpu.json", "metadata");
  Write(metadata / "catalogs/manifests/large.zip", "must not copy");
  std::string error;
  Check(cyxwiz::installer::StageExternalInstallerSession(
            source, metadata, {source / "dependency.bin"}, output, error),
        "External session must stage a bounded dependency closure: " + error);
  Check(std::filesystem::exists(output / "dependency.bin") &&
            std::filesystem::exists(output /
                                    "runtime/catalogs/manifests/cpu.json") &&
            !std::filesystem::exists(output / "engine-data.bin") &&
            !std::filesystem::exists(output /
                                     "runtime/catalogs/manifests/large.zip"),
        "Session must exclude runtime payloads and archives");
  Check(!cyxwiz::installer::StageExternalInstallerSession(source, metadata, {},
                                                          output, error),
        "Session must not overwrite an existing directory");
  Check(!cyxwiz::installer::StageExternalInstallerSession(
            source, metadata, {}, source / "nested", error),
        "Session must not stage inside its source");
#ifdef _WIN32
  // The Windows loader reports import-table casing that differs from disk.
  Write(source / "vcruntime140.dll", "loaded runtime");
  const auto cased_output = temporary.path() / "session-cased";
  const bool cased_staged = cyxwiz::installer::StageExternalInstallerSession(
      source, metadata, {source / "VCRUNTIME140.dll"}, cased_output, error);
  Check(cased_staged, "Loader casing must not stage one file twice: " + error);
  Check(std::filesystem::exists(cased_output / "vcruntime140.dll"),
        "Case-folded module must still be staged");
#endif
  Check(cyxwiz::installer::InstallerPathWithin(source / "child", source) &&
            !cyxwiz::installer::InstallerPathWithin(
                temporary.path() / "source-other", source),
        "External-session containment must use path components, not string "
        "prefixes");
  const auto ready = cyxwiz::installer::PrepareExternalInstallerSession(
      output, source / "runtime", metadata, {"installer"}, error);
  Check(ready == cyxwiz::installer::ExternalInstallerSession::Ready,
        "External manager must not relocate or loop");
}

void TestArchiveRuntimeIsInSessionClosure() {
#ifdef _WIN32
  std::string error;
  const auto modules = cyxwiz::installer::LoadedInstallerModules(error);
  Check(error.empty(), "Loaded session modules must be inspectable");
  bool archive_found = false;
  for (const auto& module : modules) {
    if (module.filename() == "archive.dll") archive_found = true;
  }
  Check(archive_found,
        "The session must include the helper's archive runtime, not just GUI dependencies");
#endif
}

void TestRevalidatesBeforeRemoval() {
  TemporaryDirectory temporary;
  ProductFixture product(temporary.path());
  const auto state = cyxwiz::installer::InspectInstallerProductRemoval(
      product.runtime_root, true);
  Write(product.root / "runtime/base/base-v1/RUNTIME_VERSIONS.json", "invalid");
  const auto result = cyxwiz::installer::RemoveInstallerProduct(state, temporary.path());
  Check(!result.succeeded && std::filesystem::exists(product.root),
        "Authorization changed after review must fail before deletion");
}

void TestRejectsMissingFinalizer() {
  TemporaryDirectory temporary;
  ProductFixture product(temporary.path());
  std::filesystem::remove(
      product.root /
      std::string(
          cyxwiz::runtime::CurrentProductRemovalFinalizerExecutableName()));
  const auto state = cyxwiz::installer::InspectInstallerProductRemoval(
      product.runtime_root, true);
  Check(state.installed && !state.available &&
            state.message.find("finalizer") != std::string::npos,
        "Removal must stay disabled without the verified finalizer");
}

} // namespace

int main() {
  TestRequiresExternalSession();
  TestExternalRemovalRejectsInUseProduct();
  TestExternalSessionStaging();
  TestArchiveRuntimeIsInSessionClosure();
  TestRevalidatesBeforeRemoval();
  TestRejectsMissingFinalizer();
  std::cout << "Installer product removal contracts passed\n";
  return 0;
}
