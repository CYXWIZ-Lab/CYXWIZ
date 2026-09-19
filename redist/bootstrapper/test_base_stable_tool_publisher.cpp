#include "backend_pack_hash.h"
#include "backend_pack_platform.h"
#include "base_stable_tool_publisher.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace {
namespace fs = std::filesystem;
using namespace cyxwiz::runtime;
void Check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}
void Write(const fs::path &path, char byte) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.put(byte);
  stream.close();
  Check(static_cast<bool>(stream), "Cannot write test fixture");
}
char Read(const fs::path &path) {
  std::ifstream stream(path, std::ios::binary);
  return static_cast<char>(stream.get());
}
struct Fixture {
  fs::path root =
      fs::temp_directory_path() /
      ("cyxwiz-stable-tools-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::path runtime = root / "runtime";
  fs::path base = runtime / "base/base-v2";
  std::string launcher =
      std::string(CurrentRuntimeBootstrapperExecutableName());
  std::string finalizer =
      std::string(CurrentProductRemovalFinalizerExecutableName());
  VerifiedBackendPackManifest manifest;
  Fixture() {
    fs::create_directories(base);
    manifest.kind = BackendPackManifestKind::Base;
    manifest.backend = "cpu";
    manifest.pack_id = "base-v2";
    for (const auto &name : {finalizer, launcher}) {
      Write(root / name, 'o');
      Write(base / name, 'n');
      std::string hash, error;
      Check(Sha256File(base / name, hash, error), error);
      manifest.components.push_back({name, 1, hash});
    }
  }
  ~Fixture() {
    std::error_code ignored;
    fs::remove_all(root, ignored);
  }
  void ExpectRejectedPair() {
    const auto result = PublishVerifiedBaseStableTools(manifest, base, runtime);
    Check(!result.published && !result.message.empty() &&
              result.launcher_path.empty() && result.finalizer_path.empty() &&
              Read(root / launcher) == 'o' && Read(root / finalizer) == 'o',
          "Invalid candidate pair must preserve both installed tools");
  }
};
void Run() {
  for (const bool accept : {false, true}) {
    Fixture f;
    bool called = false;
    const auto result = PublishVerifiedBaseStableTools(
        f.manifest, f.base, f.runtime, [&](std::string &error) {
          called = true;
          Check(Read(f.root / f.launcher) == 'n' &&
                    Read(f.root / f.finalizer) == 'n',
                "Activation must run after both tools are published");
          error = accept ? "" : "Injected activation rejection";
          return accept;
        });
    Check(called && result.published == accept && !result.commit_uncertain &&
              Read(f.root / f.launcher) == (accept ? 'n' : 'o') &&
              Read(f.root / f.finalizer) == (accept ? 'n' : 'o') &&
              result.recovery_directory.empty(),
          "Activation rejection must restore both tools; success must keep "
          "both new tools");
  }
  {
    Fixture f;
    fs::remove(f.root / f.launcher);
    fs::remove(f.root / f.finalizer);
    const auto result = PublishVerifiedBaseStableTools(
        f.manifest, f.base, f.runtime, [](std::string &error) {
          error = "Injected activation rejection";
          return false;
        });
    Check(!result.published && !fs::exists(f.root / f.launcher) &&
              !fs::exists(f.root / f.finalizer),
          "Rejected fresh activation must restore absence of both tools");
  }
  {
    Fixture f;
    const auto result = PublishVerifiedBaseStableTools(
        f.manifest, f.base, f.runtime, [](std::string &) -> bool {
          throw std::runtime_error("Uncertain activation");
        });
    Check(!result.published && result.commit_uncertain &&
              Read(f.root / f.launcher) == 'n' &&
              Read(f.root / f.finalizer) == 'n' &&
              fs::is_directory(result.recovery_directory),
          "Unknown commit outcome must retain new tools and recovery copies");
    int backups = 0;
    for (const auto &entry : fs::directory_iterator(f.root)) {
      if (entry.path().filename().string().find(".cyxwiz-tool-backup-") == 0)
        ++backups;
    }
    Check(backups == 2, "Unknown commit must retain both backups");
  }
  {
    Fixture f;
    // A directory at the second destination forces a real rename failure on
    // Windows and POSIX, after the first tool was successfully published.
    fs::remove(f.root / f.launcher);
    fs::create_directory(f.root / f.launcher);
    Write(f.root / f.launcher / "preserve", 'p');
    const auto result =
        PublishVerifiedBaseStableTools(f.manifest, f.base, f.runtime);
    Check(!result.published && result.finalizer_path.empty() &&
              result.recovery_directory.empty() &&
              result.message.find("previous finalizer state restored") !=
                  std::string::npos &&
              Read(f.root / f.finalizer) == 'o' &&
              Read(f.root / f.launcher / "preserve") == 'p',
          "Second-tool publication failure must restore the old finalizer and "
          "preserve the failed destination");
    for (const auto &entry : fs::directory_iterator(f.root))
      Check(entry.path().filename().string().find(".cyxwiz-tool-backup-") != 0,
            "Successful rollback must clean its owned backup");
  }
  {
    Fixture f;
    fs::remove(f.root / f.finalizer);
    fs::remove(f.root / f.launcher);
    fs::create_directory(f.root / f.launcher);
    const auto result =
        PublishVerifiedBaseStableTools(f.manifest, f.base, f.runtime);
    Check(!result.published && !fs::exists(f.root / f.finalizer) &&
              fs::is_directory(f.root / f.launcher),
          "Fresh-install rollback must restore the absence of the finalizer");
  }
  {
    Fixture f;
    fs::remove(f.root / f.finalizer);
    fs::create_directory(f.root / f.finalizer);
    Write(f.root / f.finalizer / "preserve", 'p');
    const auto result =
        PublishVerifiedBaseStableTools(f.manifest, f.base, f.runtime);
    Check(!result.published && Read(f.root / f.launcher) == 'o' &&
              Read(f.root / f.finalizer / "preserve") == 'p',
          "Invalid first destination must reject before changing the launcher");
  }
  {
    Fixture f;
    const auto result =
        PublishVerifiedBaseStableTools(f.manifest, f.base, f.runtime);
    Check(result.published && Read(f.root / f.launcher) == 'n' &&
              Read(f.root / f.finalizer) == 'n',
          "Valid tools must be published");
  }
  {
    Fixture f;
    f.manifest.components.pop_back();
    f.ExpectRejectedPair();
  }
  {
    Fixture f;
    fs::remove(f.base / f.launcher);
    f.ExpectRejectedPair();
  }
  {
    Fixture f;
    Write(f.base / f.launcher, 'x'); // Same size; hash must catch corruption.
    f.ExpectRejectedPair();
  }
  {
    Fixture f;
    f.manifest.components.back().size = 2;
    f.ExpectRejectedPair();
  }
  {
    Fixture f;
    fs::remove(f.base / f.launcher);
    fs::create_directory(f.base / f.launcher);
    f.ExpectRejectedPair();
  }
}
#ifdef _WIN32
class LoadedImage {
public:
  explicit LoadedImage(const fs::path &executable) {
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    std::wstring command = L"\"" + executable.native() + L"\"";
    Check(::CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr,
                           FALSE, CREATE_SUSPENDED | CREATE_NO_WINDOW, nullptr,
                           nullptr, &startup, &process) != FALSE,
          "Cannot load isolated executable image");
    handle_ = process.hProcess;
    ::CloseHandle(process.hThread);
  }
  LoadedImage(const LoadedImage &) = delete;
  LoadedImage &operator=(const LoadedImage &) = delete;
  ~LoadedImage() {
    if (handle_) {
      ::TerminateProcess(handle_, 0);
      ::WaitForSingleObject(handle_, 5000);
      ::CloseHandle(handle_);
    }
  }
  void Stop() {
    Check(::TerminateProcess(handle_, 0) != FALSE,
          "Cannot stop owned image probe");
    Check(::WaitForSingleObject(handle_, 5000) == WAIT_OBJECT_0,
          "Owned image probe did not exit");
    ::CloseHandle(handle_);
    handle_ = nullptr;
  }

private:
  HANDLE handle_ = nullptr;
};

void RunLoadedImage(const fs::path &executable) {
  for (const bool block_launcher : {false, true}) {
    Fixture f;
    const auto blocked = f.root / (block_launcher ? f.launcher : f.finalizer);
    fs::copy_file(executable, blocked, fs::copy_options::overwrite_existing);
    std::string original, actual, error;
    Check(Sha256File(blocked, original, error), error);
    LoadedImage image(blocked);
    bool activated = false;
    const auto result = PublishVerifiedBaseStableTools(
        f.manifest, f.base, f.runtime, [&](std::string &) {
          activated = true;
          return true;
        });
    Check(Sha256File(blocked, actual, error), error);
    Check(!result.published && !activated && original == actual &&
              Read(f.root / (block_launcher ? f.finalizer : f.launcher)) == 'o',
          "Loaded tool must block activation and preserve both previous tools");
    image.Stop();
    const auto retry = PublishVerifiedBaseStableTools(
        f.manifest, f.base, f.runtime, [&](std::string &) {
          activated = true;
          return true;
        });
    Check(retry.published && activated && Read(f.root / f.launcher) == 'n' &&
              Read(f.root / f.finalizer) == 'n',
          "Retry must succeed after the executable image is released");
  }
}
#endif

void RunBinaryMigration(const fs::path &previous, const fs::path &candidate) {
  for (const bool accept : {false, true}) {
    Fixture f;
    std::vector<std::string> old_hashes;
    for (auto &component : f.manifest.components) {
      const auto name = component.relative_path;
      fs::copy_file(previous / name, f.root / name,
                    fs::copy_options::overwrite_existing);
      fs::copy_file(candidate / name, f.base / name,
                    fs::copy_options::overwrite_existing);
      std::string error, old_hash;
      Check(Sha256File(f.root / name, old_hash, error), error);
      old_hashes.push_back(old_hash);
      component.size = fs::file_size(f.base / name);
      Check(Sha256File(f.base / name, component.sha256, error), error);
      Check(old_hash != component.sha256,
            "Migration must use distinct old and new binaries");
    }
    bool called = false;
    const auto result = PublishVerifiedBaseStableTools(
        f.manifest, f.base, f.runtime, [&](std::string &error) {
          called = true;
          error = accept ? "" : "Injected activation rejection";
          return accept;
        });
    Check(called && result.published == accept && !result.commit_uncertain &&
              result.recovery_directory.empty(),
          "Binary migration outcome mismatch");
    for (std::size_t i = 0; i < f.manifest.components.size(); ++i) {
      const auto &component = f.manifest.components[i];
      std::string actual, error;
      Check(Sha256File(f.root / component.relative_path, actual, error), error);
      Check(actual == (accept ? component.sha256 : old_hashes[i]),
            "Migration must preserve the exact committed or restored binary "
            "hash");
    }
  }
  std::cout << "Previous-binary migration and rejection rollback passed\n";
}
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc == 3) {
      RunBinaryMigration(fs::absolute(argv[1]), fs::absolute(argv[2]));
      return 0;
    }
    Check(argc == 1,
          "Expected no arguments or previous/candidate tool directories");
    Run();
#ifdef _WIN32
    RunLoadedImage(fs::absolute(argv[0]));
#endif
    std::cout << "Stable tool pair preflight contracts passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
