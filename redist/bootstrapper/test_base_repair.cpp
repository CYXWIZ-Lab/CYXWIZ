#include "backend_pack_installer.h"
#include "backend_pack_platform.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace {
namespace fs = std::filesystem;
using namespace cyxwiz::runtime;
constexpr const char *kHash =
    "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d";
void Write(const fs::path &path, char value) {
  fs::create_directories(path.parent_path());
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.put(value);
  if (!stream)
    throw std::runtime_error("Cannot write repair fixture");
}
char Read(const fs::path &path) {
  char value = '?';
  std::ifstream(path, std::ios::binary).get(value);
  return value;
}
struct Fixture {
  fs::path root =
      fs::temp_directory_path() /
      ("cyxwiz-base-repair-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::path runtime = root / "runtime";
  fs::path engine = runtime / "base/base-v1" / CurrentEngineExecutableName();
  Fixture() {
    Write(engine, 'x');
    Write(root / "source" / CurrentEngineExecutableName(), '\0');
    Write(root / "project.txt", 'p');
    Write(engine.parent_path() / "user-note.txt", 'u');
    ActiveRuntimeState state;
    state.runtime_set_id = "set-v1";
    state.base_pack_id = "base-v1";
    state.generation = 1;
    state.packs.push_back({"opencl", "opencl-v1"});
    std::string error;
    if (!SaveActiveRuntimeStateAtomic(runtime / "active-runtime.json", state,
                                      error))
      throw std::runtime_error(error);
  }
  ~Fixture() {
    std::error_code ignored;
    fs::remove_all(root, ignored);
  }
  VerifiedBackendPackPayload Payload() const {
    return {"set-v1",
            {},
            "cpu",
            "base-v1",
            root / "source",
            {{std::string(CurrentEngineExecutableName()), 1, kHash}}};
  }
  ActiveRuntimeState Active() const {
    ActiveRuntimeState state;
    std::string error;
    if (!LoadActiveRuntimeState(runtime / "active-runtime.json", state, error))
      throw std::runtime_error(error);
    return state;
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
  using S = BackendPackInstallStatus;
  using P = BackendPackInstallCheckpoint;
  {
    Fixture f;
    BackendPackInstaller installer(f.runtime, [] { return false; });
    const auto repaired = installer.RepairBase(f.Payload(), 1024);
    check(repaired.status == S::InstalledAndActivated &&
              Read(f.engine) == '\0' && f.Active().generation == 2 &&
              f.Active().packs.empty() && Read(f.root / "project.txt") == 'p',
          "CPU repair restores verified files and advances CPU-only activation "
          "without touching projects");
    const auto backups = f.runtime / "repair-backups";
    bool preserved = false;
    for (const auto &entry : fs::directory_iterator(backups)) {
      preserved =
          preserved ||
          (Read(entry.path() / "payload/user-note.txt") == 'u' &&
           Read(entry.path() / "payload" / CurrentEngineExecutableName()) ==
               'x' &&
           fs::is_regular_file(entry.path() / "previous-active-runtime.json"));
    }
    check(preserved, "The old base, including unknown user files, must remain "
                     "in a recovery backup");
  }
  for (const auto point :
       {P::AfterValidation, P::AfterCopy, P::BeforePackPublish,
        P::AfterQuarantine, P::AfterPackPublish, P::BeforeActivation}) {
    Fixture f;
    BackendPackInstaller installer(
        f.runtime, [] { return false; }, {},
        [point](P reached) { return reached != point; });
    const auto interrupted = installer.RepairBase(f.Payload(), 1024);
    check(interrupted.status == S::Interrupted && Read(f.engine) == 'x' &&
              f.Active().generation == 1 &&
              Read(f.engine.parent_path() / "user-note.txt") == 'u',
          "Interruption at each repair boundary must retain or restore the "
          "original base and activation");
  }
  {
    Fixture f;
    fs::remove_all(f.engine.parent_path());
    BackendPackInstaller installer(f.runtime, [] { return false; });
    check(installer.RepairBase(f.Payload(), 1024).status ==
                  S::InstalledAndActivated &&
              Read(f.engine) == '\0' && f.Active().generation == 2,
          "A missing base directory can be restored when activation still "
          "identifies the exact base");
  }
  {
    Fixture f;
    BackendPackInstaller installer(
        f.runtime, [] { return false; }, {},
        [&](P point) {
          if (point == P::AfterPackPublish)
            installer.Cancel();
          return true;
        });
    check(installer.RepairBase(f.Payload(), 1024).status == S::Interrupted &&
              Read(f.engine) == 'x',
          "Cancellation during publication must restore the old directory");
  }
  {
    Fixture f;
    BackendPackInstaller installer(
        f.runtime, [] { return false; }, {},
        [](P point) {
          if (point == P::AfterPackPublish)
            throw std::runtime_error("injected publication failure");
          return true;
        });
    check(installer.RepairBase(f.Payload(), 1024).status ==
                  S::FilesystemFailure &&
              Read(f.engine) == 'x' && f.Active().generation == 1,
          "An exception during publication must restore files before returning "
          "an error");
  }
  {
    Fixture f;
    BackendPackInstaller absent_guard(f.runtime);
    check(absent_guard.RepairBase(f.Payload(), 1024).status ==
              S::InvalidRequest,
          "Repair must never run without an execution-active guard");
    BackendPackInstaller running(f.runtime, [] { return true; });
    check(running.RepairBase(f.Payload(), 1024).status == S::ExecutionActive,
          "A running engine must block repair");
    BackendPackInstaller installer(f.runtime, [] { return false; });
    auto wrong = f.Payload();
    wrong.pack_id = "base-other";
    check(installer.RepairBase(wrong, 1024).status == S::InvalidRequest,
          "Repair must not replace another base identity");
    Write(f.root / "source" / CurrentEngineExecutableName(), 'b');
    check(installer.RepairBase(f.Payload(), 1024).status ==
                  S::IntegrityFailure &&
              Read(f.engine) == 'x',
          "A corrupt repair source must fail before replacing installed files");
  }
  {
    Fixture f;
    bool running = false;
    BackendPackInstaller installer(
        f.runtime, [&] { return running; }, {},
        [&](P point) {
          if (point == P::AfterPackPublish)
            running = true;
          return true;
        });
    check(installer.RepairBase(f.Payload(), 1024).status ==
                  S::ExecutionActive &&
              Read(f.engine) == 'x' && f.Active().generation == 1,
          "Execution appearing at commit must trigger restoration");
  }
  {
    Fixture f;
    BackendPackInstaller installer(
        f.runtime, [] { return false; }, {},
        [&](P point) {
          if (point == P::AfterCopy) {
            auto state = f.Active();
            ++state.generation;
            std::string error;
            if (!SaveActiveRuntimeStateAtomic(f.runtime / "active-runtime.json",
                                              state, error))
              throw std::runtime_error(error);
          }
          return true;
        });
    check(installer.RepairBase(f.Payload(), 1024).status == S::InvalidRequest &&
              Read(f.engine) == 'x' && f.Active().generation == 2,
          "Repair must not overwrite a newer activation state");
  }
#ifdef _WIN32
  {
    Fixture f;
    const HANDLE locked =
        ::CreateFileW((f.runtime / "active-runtime.json").c_str(), GENERIC_READ,
                      FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                      FILE_ATTRIBUTE_NORMAL, nullptr);
    if (locked == INVALID_HANDLE_VALUE)
      throw std::runtime_error("Cannot lock activation fixture");
    BackendPackInstaller installer(f.runtime, [] { return false; });
    const auto result = installer.RepairBase(f.Payload(), 1024);
    ::CloseHandle(locked);
    check(result.status == S::FilesystemFailure && Read(f.engine) == 'x' &&
              f.Active().generation == 1,
          "Failure to publish activation must restore the old base");
  }
#endif
  return failures == 0 ? 0 : 1;
}
