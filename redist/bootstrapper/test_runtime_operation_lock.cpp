#include "runtime_operation_lock.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <thread>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {
namespace fs = std::filesystem;
using namespace cyxwiz::runtime;
using namespace std::chrono_literals;

class Child {
public:
  Child(const fs::path &executable, const fs::path &runtime, bool hold,
        bool engine_use = false, bool inherited_only = false) {
    const auto mode = inherited_only ? "--inherited-only"
                      : engine_use   ? (hold ? "--hold-engine" : "--try-engine")
                                     : (hold ? "--hold" : "--try");
#ifdef _WIN32
    std::wstring command = L"\"" + executable.native() + L"\" " +
                           fs::path(mode).native() + L" \"" + runtime.native() +
                           L"\"";
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    if (::CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr,
                         inherited_only ? TRUE : FALSE, CREATE_NO_WINDOW,
                         nullptr, nullptr, &startup, &process)) {
      handle_ = process.hProcess;
      ::CloseHandle(process.hThread);
    }
#else
    pid_ = ::fork();
    if (pid_ == 0) {
      ::execl(executable.c_str(), executable.c_str(), mode, runtime.c_str(),
              nullptr);
      ::_exit(79);
    }
#endif
  }
  ~Child() {
#ifdef _WIN32
    if (handle_) {
      if (::WaitForSingleObject(handle_, 0) == WAIT_TIMEOUT) {
        ::TerminateProcess(handle_, 80);
        ::WaitForSingleObject(handle_, 5000);
      }
      ::CloseHandle(handle_);
    }
#else
    if (pid_ > 0) {
      ::kill(pid_, SIGKILL);
      ::waitpid(pid_, nullptr, 0);
    }
#endif
  }
  Child(const Child &) = delete;
  Child &operator=(const Child &) = delete;
  int Wait() {
#ifdef _WIN32
    DWORD result = 79;
    if (handle_ && ::WaitForSingleObject(handle_, 5000) == WAIT_OBJECT_0 &&
        ::GetExitCodeProcess(handle_, &result))
      return static_cast<int>(result);
#else
    for (int i = 0; pid_ > 0 && i < 100; ++i) {
      int status = 0;
      if (::waitpid(pid_, &status, WNOHANG) == pid_) {
        pid_ = -1;
        return WIFEXITED(status) ? WEXITSTATUS(status) : 79;
      }
      std::this_thread::sleep_for(50ms);
    }
#endif
    return 79;
  }

private:
#ifdef _WIN32
  HANDLE handle_ = nullptr;
#else
  pid_t pid_ = -1;
#endif
};
} // namespace

int main(int argc, char **argv) {
  if (argc == 3) {
    const std::string mode(argv[1]);
    if (mode == "--inherited-only") {
      {
        RuntimeOperationLock startup;
        std::string error;
        if (startup.AcquireEngineUse(fs::path(argv[2]), error) !=
                RuntimeOperationLockStatus::Acquired ||
            !startup.RestrictInheritedEngineOwnership(error) ||
            std::getenv("CYXWIZ_RUNTIME_USE_TOKEN"))
          return 4;
      }
      std::ofstream(fs::path(argv[2]) / "inherited-ready") << "ready";
      std::this_thread::sleep_for(20s);
      return 0;
    }
    RuntimeOperationLock lock;
    std::string error;
    const auto result = mode == "--try-engine" || mode == "--hold-engine"
                            ? lock.AcquireEngineUse(fs::path(argv[2]), error)
                            : lock.Acquire(fs::path(argv[2]), error);
    if (result == RuntimeOperationLockStatus::Busy)
      return 2;
    if (result != RuntimeOperationLockStatus::Acquired)
      return 3;
    if (mode == "--hold" || mode == "--hold-engine") {
      std::ofstream ready(fs::path(argv[2]) / "ready");
      ready << "locked";
      ready.close();
      std::this_thread::sleep_for(20s);
    }
    return 0;
  }
  const auto root =
      fs::temp_directory_path() /
      ("cyxwiz-operation-lock-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  const auto runtime = root / "runtime with spaces";
  const auto executable = fs::absolute(argv[0]);
  int failures = 0;
  const auto check = [&](bool ok, const char *message) {
    if (!ok) {
      ++failures;
      std::cerr << "FAIL: " << message << '\n';
    }
  };
  std::string error;
  {
    RuntimeOperationLock lock;
    check(lock.Acquire("relative", error) ==
                  RuntimeOperationLockStatus::Failed &&
              lock.Acquire(root.root_path(), error) ==
                  RuntimeOperationLockStatus::Failed,
          "Reject relative and drive-root operation locks");
    check(lock.Acquire(runtime, error) ==
                  RuntimeOperationLockStatus::Acquired &&
              lock.Owns(runtime) && !lock.Owns(root / "other"),
          "Bind ownership to runtime");
    check(lock.Acquire(runtime, error) == RuntimeOperationLockStatus::Failed &&
              lock.Owns(runtime),
          "Repeated acquisition must not leak or release ownership");
    RuntimeOperationLock overlap;
    check(overlap.Acquire(runtime, error) == RuntimeOperationLockStatus::Busy,
          "Overlapping local owners are rejected");
    Child child(executable, runtime, false);
    check(child.Wait() == 2,
          "Independent helper process must observe contention");
    Child engine_child(executable, runtime, false, true);
    check(engine_child.Wait() == 2,
          "Maintenance must block new Engine sessions");
  }
  {
    Child child(executable, runtime, false);
    check(child.Wait() == 0,
          "Owner destruction releases the persistent lock file");
  }
  {
    Child child(executable, runtime, true);
    for (int i = 0; i < 100 && !fs::exists(runtime / "ready"); ++i)
      std::this_thread::sleep_for(50ms);
    RuntimeOperationLock blocked;
    check(fs::exists(runtime / "ready") && blocked.Acquire(runtime, error) ==
                                               RuntimeOperationLockStatus::Busy,
          "Live child owns the lock before forced termination");
  }
  {
    RuntimeOperationLock after_crash;
    check(after_crash.Acquire(runtime, error) ==
              RuntimeOperationLockStatus::Acquired,
          "OS must release ownership after child termination");
  }
  {
    const auto product = root / "removal-product";
    const auto removal_runtime = product / "runtime";
    RuntimeOperationLock removal;
    check(removal.Acquire(removal_runtime, error) ==
              RuntimeOperationLockStatus::Acquired,
          "Removal fixture must acquire both identities");
    removal.PrepareForQuarantine();
    check(!removal.Owns(removal_runtime),
          "A removal-only guard cannot be borrowed for repair");
    Child before_move(executable, removal_runtime, false);
    check(before_move.Wait() == 2,
          "Sibling guard blocks a child after the inner handle closes");
    std::error_code rename_error;
    fs::rename(product, root / "quarantined-product", rename_error);
    check(!rename_error, "Product rename must succeed under sibling ownership");
    Child after_move(executable, removal_runtime, false);
    check(after_move.Wait() == 2 && !fs::exists(product),
          "A helper must not recreate the installed path during quarantine");
  }
  {
    RuntimeOperationLock engine, second, maintenance;
    check(engine.AcquireEngineUse(runtime, error) ==
                  RuntimeOperationLockStatus::Acquired &&
              second.AcquireEngineUse(runtime, error) ==
                  RuntimeOperationLockStatus::Acquired &&
              !engine.Owns(runtime),
          "Multiple Engines share ownership without repair authority");
    check(maintenance.Acquire(runtime, error) ==
              RuntimeOperationLockStatus::Busy,
          "A running Engine excludes maintenance");
    Child another_engine(executable, runtime, false, true);
    check(another_engine.Wait() == 0,
          "Another Engine process may share ownership");
  }
  {
    std::unique_ptr<Child> child;
    {
      RuntimeOperationLock launcher;
      check(launcher.AcquireEngineUse(runtime, error) ==
                    RuntimeOperationLockStatus::Acquired &&
                launcher.AllowEngineChildInheritance(error),
            "Launcher exports shared ownership");
      child = std::make_unique<Child>(executable, runtime, true, true, true);
      for (int i = 0; i < 100 && !fs::exists(runtime / "inherited-ready"); ++i)
        std::this_thread::sleep_for(50ms);
      check(fs::exists(runtime / "inherited-ready"),
            "Child validates and restricts inherited ownership");
    }
    RuntimeOperationLock maintenance;
    check(maintenance.Acquire(runtime, error) ==
              RuntimeOperationLockStatus::Busy,
          "Inherited child ownership survives launcher release");
    child.reset();
    check(maintenance.Acquire(runtime, error) ==
              RuntimeOperationLockStatus::Acquired,
          "Terminated Engine child releases inherited ownership");
#ifdef _WIN32
    ::_putenv_s("CYXWIZ_RUNTIME_USE_TOKEN", "");
#else
    ::unsetenv("CYXWIZ_RUNTIME_USE_TOKEN");
#endif
  }
  fs::create_directories(root / "redirected/staging/session/operation.lock");
  {
    RuntimeOperationLock invalid;
    check(invalid.Acquire(root / "redirected", error) ==
              RuntimeOperationLockStatus::Failed,
          "A directory must not serve as a lock file");
  }
  fs::create_directories(root / "linked/staging/session");
  std::ofstream(root / "outside.txt") << "preserve";
  fs::create_hard_link(root / "outside.txt",
                       root / "linked/staging/session/operation.lock");
  {
    RuntimeOperationLock invalid;
    check(invalid.Acquire(root / "linked", error) ==
                  RuntimeOperationLockStatus::Failed &&
              fs::file_size(root / "outside.txt") == 8,
          "Reject hard-linked lock identities");
  }
  fs::create_directories(root / "symlinked");
  fs::create_directories(root / "outside-directory");
  std::error_code symlink_error;
  fs::create_directory_symlink(root / "outside-directory",
                               root / "symlinked/staging", symlink_error);
  if (!symlink_error) {
    RuntimeOperationLock invalid;
    check(invalid.Acquire(root / "symlinked", error) ==
                  RuntimeOperationLockStatus::Failed &&
              fs::is_empty(root / "outside-directory"),
          "Redirected staging must fail before creating files outside the "
          "runtime");
  } else {
    std::cout << "SKIP: directory symlink fixture unavailable: "
              << symlink_error.message() << '\n';
  }
  std::error_code cleanup;
  fs::remove_all(root, cleanup);
  check(!cleanup, "Fixture cleanup must succeed after lock release");
  fs::remove(RuntimeOperationLock::LocationLockPath(runtime), cleanup);
  check(!cleanup,
        "Remove this isolated test's sibling lock after all children exit");
  if (!failures)
    std::cout << "runtime operation lock contracts passed\n";
  return failures ? 1 : 0;
}
