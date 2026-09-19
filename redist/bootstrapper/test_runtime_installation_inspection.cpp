#include "backend_pack_platform.h"
#include "product_installation_receipt.h"
#include "runtime_installation_inspection.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
namespace fs = std::filesystem;
using namespace cyxwiz::runtime;

void Write(const fs::path &path, const std::string &value = "fixture") {
  fs::create_directories(path.parent_path());
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream << value;
  if (!stream)
    throw std::runtime_error("Cannot write inspection fixture");
}

struct Fixture {
  fs::path product =
      fs::temp_directory_path() /
      ("cyxwiz-inspection-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::path runtime = product / "runtime";
  ~Fixture() {
    std::error_code ignored;
    fs::remove_all(product, ignored);
  }

  void Activate() const {
    Write(runtime / "base/base-v1" / CurrentEngineExecutableName());
    ActiveRuntimeState state;
    state.runtime_set_id = "set-v1";
    state.base_pack_id = "base-v1";
    state.generation = 1;
    std::string error;
    if (!SaveActiveRuntimeStateAtomic(runtime / "active-runtime.json", state,
                                      error))
      throw std::runtime_error(error);
  }
};
} // namespace

int main() {
  int failures = 0;
  const auto check = [&](bool passed, const char *message) {
    if (!passed) {
      ++failures;
      std::cerr << "FAIL: " << message << '\n';
    }
  };
  using C = RuntimeInstallationCondition;
  check(InspectRuntimeInstallation("relative/runtime").condition ==
            C::RecoveryRequired,
        "Relative locations must fail closed");
  {
    Fixture f;
    check(InspectRuntimeInstallation(f.runtime).condition == C::Fresh &&
              !fs::exists(f.product),
          "A missing target is fresh and inspection must not create it");
    Write(f.runtime / "cache/artifacts/download.part");
    Write(f.runtime / "staging/session/operation.lock");
    Write(f.runtime / "catalogs/current.json");
    Write(f.product / "project.txt", "preserve");
    check(InspectRuntimeInstallation(f.runtime).condition == C::Fresh &&
              fs::file_size(f.product / "project.txt") == 8 &&
              fs::exists(f.runtime / "cache/artifacts/download.part"),
          "Download-only retries remain possible without modifying cached "
          "downloads or projects");
    f.Activate();
    const auto active = InspectRuntimeInstallation(f.runtime);
    check(active.condition == C::Active &&
              active.active.base_pack_id == "base-v1" &&
              active.active.generation == 1,
          "Healthy structural state enters maintenance");
    fs::remove(f.runtime / "base/base-v1" / CurrentEngineExecutableName());
    check(InspectRuntimeInstallation(f.runtime).condition ==
              C::RecoveryRequired,
          "Missing engine executable requires recovery despite valid "
          "activation JSON");
    f.Activate();
    Write(f.runtime / "active-runtime.json", "{}");
    check(InspectRuntimeInstallation(f.runtime).condition ==
                  C::RecoveryRequired &&
              fs::file_size(f.runtime / "active-runtime.json") == 2,
          "Invalid activation state must be preserved, not overwritten");
    fs::remove(f.runtime / "active-runtime.json");
    check(InspectRuntimeInstallation(f.runtime).condition ==
              C::RecoveryRequired,
          "Missing activation with an installed base is not a fresh install");
  }
  for (const auto *marker :
       {"base", "packs", "installed-metadata", "rollback"}) {
    Fixture f;
    fs::create_directories(f.runtime / marker);
    check(InspectRuntimeInstallation(f.runtime).condition == C::Fresh,
          "Empty managed containers alone do not imply an installation");
    Write(f.runtime / marker / "retained");
    check(InspectRuntimeInstallation(f.runtime).condition ==
              C::RecoveryRequired,
          "Retained installed-state evidence must not be treated as fresh");
  }
  for (const auto name :
       {".cyxwiz-installation.json", "launcher", "finalizer"}) {
    Fixture f;
    const auto marker =
        std::string(name) == "launcher"
            ? f.product / CurrentRuntimeBootstrapperExecutableName()
        : std::string(name) == "finalizer"
            ? f.product / CurrentProductRemovalFinalizerExecutableName()
            : ProductInstallationReceiptPath(f.product);
    Write(marker);
    check(InspectRuntimeInstallation(f.runtime).condition ==
              C::RecoveryRequired,
          "Missing whole runtime must not hide existing product markers");
  }
  {
    Fixture f;
    f.Activate();
    ActiveRuntimeState state;
    std::string error;
    if (!LoadActiveRuntimeState(f.runtime / "active-runtime.json", state,
                                error))
      throw std::runtime_error(error);
    state.packs.push_back({"opencl", "opencl-v1"});
    if (!SaveActiveRuntimeStateAtomic(f.runtime / "active-runtime.json", state,
                                      error))
      throw std::runtime_error(error);
    check(InspectRuntimeInstallation(f.runtime).condition ==
              C::RecoveryRequired,
          "Missing active optional runtime must not be shown as launchable");
    Write(f.runtime / "packs/opencl/opencl-v1/runtime" /
          CurrentArrayFireBackendPluginName("opencl"));
    check(InspectRuntimeInstallation(f.runtime).condition == C::Active,
          "Canonical optional runtime layout remains supported");
  }
  {
    Fixture f;
    fs::create_directories(f.runtime);
    std::error_code error;
    fs::create_symlink(f.product / "missing-state.json",
                       f.runtime / "active-runtime.json", error);
    if (error)
      std::cout << "SKIP: symlink creation unavailable: " << error.message()
                << '\n';
    else
      check(InspectRuntimeInstallation(f.runtime).condition ==
                C::RecoveryRequired,
            "Dangling activation links must not appear as missing/fresh state");
  }
  return failures == 0 ? 0 : 1;
}
