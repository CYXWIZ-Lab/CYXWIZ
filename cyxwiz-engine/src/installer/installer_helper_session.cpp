#include "installer_helper_session.h"

#include <limits>
#include <system_error>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace cyxwiz::installer {

std::uint64_t CurrentInstallerProcessId() {
#ifdef _WIN32
  return static_cast<std::uint64_t>(::GetCurrentProcessId());
#else
  return static_cast<std::uint64_t>(::getpid());
#endif
}

bool CleanupAbandonedInstallerStaging(
    const std::filesystem::path &runtime_root, const std::string &pack_id,
    std::string &error) {
  if (!runtime_root.is_absolute()) {
    error = "Abandoned staging cleanup requires an absolute runtime root";
    return false;
  }
  const auto staging = runtime_root / "staging";
  std::error_code filesystem_error;
  std::filesystem::remove_all(staging / "delivery", filesystem_error);
  if (filesystem_error) {
    error = "Cannot remove abandoned extraction staging: " +
            filesystem_error.message();
    return false;
  }
  std::filesystem::remove_all(staging / "progress", filesystem_error);
  if (filesystem_error) {
    error = "Cannot remove abandoned progress state: " +
            filesystem_error.message();
    return false;
  }
  if (pack_id.empty()) return true;
  const auto prefix = pack_id + "-";
  std::filesystem::directory_iterator iterator(staging, filesystem_error);
  const std::filesystem::directory_iterator end;
  while (!filesystem_error && iterator != end) {
    const auto name = iterator->path().filename().string();
    if (name.starts_with(prefix)) {
      std::filesystem::remove_all(iterator->path(), filesystem_error);
      if (filesystem_error) break;
    }
    iterator.increment(filesystem_error);
  }
  if (filesystem_error) {
    error = "Cannot remove abandoned installation staging: " +
            filesystem_error.message();
    return false;
  }
  return true;
}

InstallerHelperSession::~InstallerHelperSession() {
#ifdef _WIN32
  if (parent_handle_ != 0) {
    ::CloseHandle(reinterpret_cast<HANDLE>(parent_handle_));
  }
  if (operation_handle_ != 0) {
    ::CloseHandle(reinterpret_cast<HANDLE>(operation_handle_));
  }
#else
  if (operation_descriptor_ >= 0) {
    ::flock(operation_descriptor_, LOCK_UN);
    ::close(operation_descriptor_);
  }
#endif
}

bool InstallerHelperSession::Open(const std::filesystem::path &runtime_root,
                                  std::uint64_t parent_process_id,
                                  std::string &error) {
#ifdef _WIN32
  constexpr auto maximum_process_id =
      static_cast<std::uint64_t>(std::numeric_limits<DWORD>::max());
#else
  constexpr auto maximum_process_id =
      static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max());
#endif
  if (!runtime_root.is_absolute() ||
      parent_process_id > maximum_process_id) {
    error = "Helper session requires an absolute runtime and valid parent process";
    return false;
  }
  std::error_code filesystem_error;
  const auto session_root = runtime_root / "staging" / "session";
  std::filesystem::create_directories(session_root, filesystem_error);
  if (filesystem_error) {
    error = "Cannot create the installer session directory: " +
            filesystem_error.message();
    return false;
  }
  const auto lock_path = session_root / "operation.lock";
#ifdef _WIN32
  const HANDLE parent = parent_process_id == 0
      ? nullptr
      : ::OpenProcess(
            SYNCHRONIZE, FALSE, static_cast<DWORD>(parent_process_id));
  if (parent_process_id != 0 && !parent) {
    error = "Cannot bind the helper to its installer parent; Win32 error " +
            std::to_string(::GetLastError());
    return false;
  }
  const HANDLE operation = ::CreateFileW(
      lock_path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
      OPEN_ALWAYS, FILE_ATTRIBUTE_HIDDEN, nullptr);
  if (operation == INVALID_HANDLE_VALUE) {
    const auto code = ::GetLastError();
    if (parent) ::CloseHandle(parent);
    error = code == ERROR_SHARING_VIOLATION
                ? "Another CyxWiz installation operation is still active"
                : "Cannot acquire the installer operation lock; Win32 error " +
                      std::to_string(code);
    return false;
  }
  parent_handle_ = reinterpret_cast<std::uintptr_t>(parent);
  operation_handle_ = reinterpret_cast<std::uintptr_t>(operation);
#else
  const int descriptor = ::open(
      lock_path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
  if (descriptor < 0 || ::flock(descriptor, LOCK_EX | LOCK_NB) != 0) {
    if (descriptor >= 0) ::close(descriptor);
    error = errno == EWOULDBLOCK
                ? "Another CyxWiz installation operation is still active"
                : "Cannot acquire the installer operation lock";
    return false;
  }
  operation_descriptor_ = descriptor;
  parent_process_id_ = parent_process_id == 0
      ? 0 : static_cast<std::int64_t>(parent_process_id);
#endif
  return true;
}

bool InstallerHelperSession::ParentExited() const {
#ifdef _WIN32
  if (parent_handle_ == 0) return false;
  return ::WaitForSingleObject(
             reinterpret_cast<HANDLE>(parent_handle_), 0) != WAIT_TIMEOUT;
#else
  if (parent_process_id_ == 0) return false;
  if (parent_process_id_ < 0) return true;
  if (::kill(static_cast<pid_t>(parent_process_id_), 0) == 0) return false;
  return errno != EPERM;
#endif
}

} // namespace cyxwiz::installer
