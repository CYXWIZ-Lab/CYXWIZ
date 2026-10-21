#include "runtime_operation_lock.h"
#include <charconv>
#include <cstdlib>
#include <limits>
#include <string_view>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace cyxwiz::runtime {
namespace {
constexpr const char *kRuntimeBusyMessage =
    "CyxWiz is in use. Save your work and close CyxWiz Engine for this "
    "installation, then retry. If another installation or update is running, "
    "wait for it to finish before retrying.";

bool PlainPath(const std::filesystem::path &path, std::string &error) {
  std::error_code code;
  if (std::filesystem::weakly_canonical(path, code) !=
          path.lexically_normal() ||
      code) {
    error = "Installer operation lock path is unreadable or redirected";
    return false;
  }
  return true;
}
} // namespace

void RuntimeOperationLock::FileLock::Close() noexcept {
#ifdef _WIN32
  if (handle != 0)
    ::CloseHandle(reinterpret_cast<HANDLE>(handle));
  handle = 0;
#else
  if (descriptor >= 0)
    ::close(descriptor);
  descriptor = -1;
#endif
}

RuntimeOperationLockStatus
RuntimeOperationLock::FileLock::Acquire(const std::filesystem::path &path,
                                        std::string &error, bool shared) {
  if (!PlainPath(path, error))
    return RuntimeOperationLockStatus::Failed;
  std::error_code code;
  if (!shared)
    std::filesystem::create_directories(path.parent_path(), code);
  if (code) {
    error = "Cannot create the installer lock directory: " + code.message();
    return RuntimeOperationLockStatus::Failed;
  }
  if (!PlainPath(path, error))
    return RuntimeOperationLockStatus::Failed;
#ifdef _WIN32
  const HANDLE opened = ::CreateFileW(
      path.c_str(), shared ? GENERIC_READ : GENERIC_READ | GENERIC_WRITE,
      shared ? FILE_SHARE_READ : 0, nullptr, OPEN_ALWAYS,
      FILE_ATTRIBUTE_HIDDEN | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
  if (opened == INVALID_HANDLE_VALUE) {
    const auto failure = ::GetLastError();
    error = failure == ERROR_SHARING_VIOLATION
                ? kRuntimeBusyMessage
                : "Cannot acquire the installer operation lock; Win32 error " +
                      std::to_string(failure);
    return failure == ERROR_SHARING_VIOLATION
               ? RuntimeOperationLockStatus::Busy
               : RuntimeOperationLockStatus::Failed;
  }
  BY_HANDLE_FILE_INFORMATION info{};
  if (!::GetFileInformationByHandle(opened, &info) ||
      (info.dwFileAttributes &
       (FILE_ATTRIBUTE_REPARSE_POINT | FILE_ATTRIBUTE_DIRECTORY)) ||
      info.nNumberOfLinks != 1) {
    ::CloseHandle(opened);
    error = "Installer operation lock must be a plain, unshared file";
    return RuntimeOperationLockStatus::Failed;
  }
  handle = reinterpret_cast<std::uintptr_t>(opened);
#else
  const int opened = ::open(path.c_str(),
                            (shared ? O_RDONLY : O_RDWR) | O_CREAT | O_CLOEXEC |
                                O_NOFOLLOW | O_NONBLOCK,
                            0644);
  if (opened < 0) {
    error = "Cannot open the installer operation lock; errno " +
            std::to_string(errno);
    return RuntimeOperationLockStatus::Failed;
  }
  struct stat info{};
  if (::fstat(opened, &info) != 0 || !S_ISREG(info.st_mode) ||
      info.st_nlink != 1) {
    ::close(opened);
    error = "Installer operation lock must be a plain, unshared file";
    return RuntimeOperationLockStatus::Failed;
  }
  if (::flock(opened, (shared ? LOCK_SH : LOCK_EX) | LOCK_NB) != 0) {
    const int failure = errno;
    ::close(opened);
    const bool busy = failure == EWOULDBLOCK || failure == EAGAIN;
    error = busy ? kRuntimeBusyMessage
                 : "Cannot acquire the installer operation lock; errno " +
                       std::to_string(failure);
    return busy ? RuntimeOperationLockStatus::Busy
                : RuntimeOperationLockStatus::Failed;
  }
  descriptor = opened;
#endif
  return RuntimeOperationLockStatus::Acquired;
}

std::filesystem::path RuntimeOperationLock::LocationLockPath(
    const std::filesystem::path &runtime_root) {
  auto name = std::filesystem::path(".cyxwiz-operation-");
  name += runtime_root.parent_path().filename().native();
  name += ".lock";
  return runtime_root.parent_path().parent_path() / name;
}

RuntimeOperationLockStatus
RuntimeOperationLock::Acquire(const std::filesystem::path &runtime_root,
                              std::string &error) {
  return AcquireInternal(runtime_root, error, false);
}

RuntimeOperationLockStatus RuntimeOperationLock::AcquireEngineUse(
    const std::filesystem::path &runtime_root, std::string &error) {
  return AcquireInternal(runtime_root, error, true);
}

RuntimeOperationLockStatus
RuntimeOperationLock::AcquireInternal(const std::filesystem::path &runtime_root,
                                      std::string &error, bool engine_use) {
  error.clear();
  const auto normalized = runtime_root.lexically_normal();
  if (!runtime_root_.empty() || preparing_quarantine_ ||
      !normalized.is_absolute() ||
      normalized.parent_path() == normalized.root_path() ||
      normalized == normalized.root_path()) {
    error = "An unused lock and an absolute runtime within a non-root product "
            "are required";
    return RuntimeOperationLockStatus::Failed;
  }
  std::error_code code;
  const auto status = std::filesystem::symlink_status(normalized, code);
  if (code == std::errc::no_such_file_or_directory)
    code.clear();
  if (code ||
      (engine_use && status.type() != std::filesystem::file_type::directory) ||
      (status.type() != std::filesystem::file_type::not_found &&
       status.type() != std::filesystem::file_type::directory)) {
    error = "Installer runtime must be a plain directory";
    return RuntimeOperationLockStatus::Failed;
  }
  auto owned_root = std::filesystem::weakly_canonical(normalized, code);
  if (code) {
    error = "Cannot resolve the installer runtime: " + code.message();
    return RuntimeOperationLockStatus::Failed;
  }
  // Lock order: product location, then the legacy runtime identity.
  const auto location =
      location_lock_.Acquire(LocationLockPath(owned_root), error, engine_use);
  if (location != RuntimeOperationLockStatus::Acquired)
    return location;
  if (engine_use) {
    engine_use_ = true;
    runtime_root_.swap(owned_root);
    return RuntimeOperationLockStatus::Acquired;
  }
  const auto legacy = runtime_lock_.Acquire(
      owned_root / "staging/session/operation.lock", error);
  if (legacy != RuntimeOperationLockStatus::Acquired) {
    location_lock_.Close();
    return legacy;
  }
  runtime_root_.swap(owned_root);
  return RuntimeOperationLockStatus::Acquired;
}

void RuntimeOperationLock::PrepareForQuarantine() {
  runtime_lock_.Close();
  preparing_quarantine_ = true;
}

bool RuntimeOperationLock::Owns(
    const std::filesystem::path &runtime_root) const {
  if (runtime_root_.empty() || preparing_quarantine_ || engine_use_ ||
      !runtime_root.is_absolute())
    return false;
  std::error_code error;
  const auto canonical = std::filesystem::canonical(runtime_root, error);
  return !error && canonical == runtime_root_;
}

bool RuntimeOperationLock::AllowEngineChildInheritance(std::string &error) {
  if (!engine_use_ || runtime_root_.empty()) {
    error = "Only live Engine ownership may be inherited";
    return false;
  }
#ifdef _WIN32
  if (::SetHandleInformation(reinterpret_cast<HANDLE>(location_lock_.handle),
                             HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT) &&
      ::_putenv_s("CYXWIZ_RUNTIME_USE_TOKEN",
                  std::to_string(location_lock_.handle).c_str()) == 0)
    return true;
#else
  const int flags = ::fcntl(location_lock_.descriptor, F_GETFD);
  if (flags >= 0 &&
      ::fcntl(location_lock_.descriptor, F_SETFD, flags & ~FD_CLOEXEC) == 0 &&
      ::setenv("CYXWIZ_RUNTIME_USE_TOKEN",
               std::to_string(location_lock_.descriptor).c_str(), 1) == 0)
    return true;
#endif
  error = "Cannot preserve Engine ownership across child launch";
  return false;
}

bool RuntimeOperationLock::RestrictInheritedEngineOwnership(
    std::string &error) {
  const char *value = std::getenv("CYXWIZ_RUNTIME_USE_TOKEN");
  if (!value || !*value)
    return true;
  const std::string_view text(value);
  std::uintptr_t token = 0;
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), token);
  if (!engine_use_ || parsed.ec != std::errc{} ||
      parsed.ptr != text.data() + text.size()) {
    error = "Invalid inherited Engine ownership token";
    return false;
  }
#ifdef _WIN32
  BY_HANDLE_FILE_INFORMATION own{}, inherited{};
  const HANDLE native = reinterpret_cast<HANDLE>(token);
  const bool matches =
      ::GetFileInformationByHandle(
          reinterpret_cast<HANDLE>(location_lock_.handle), &own) &&
      ::GetFileInformationByHandle(native, &inherited) &&
      own.dwVolumeSerialNumber == inherited.dwVolumeSerialNumber &&
      own.nFileIndexHigh == inherited.nFileIndexHigh &&
      own.nFileIndexLow == inherited.nFileIndexLow;
  if (matches && ::SetHandleInformation(native, HANDLE_FLAG_INHERIT, 0) &&
      ::_putenv_s("CYXWIZ_RUNTIME_USE_TOKEN", "") == 0)
    return true;
#else
  struct stat own{}, inherited{};
  if (token <= static_cast<std::uintptr_t>(std::numeric_limits<int>::max())) {
    const int native = static_cast<int>(token);
    const int flags = ::fcntl(native, F_GETFD);
    if (::fstat(location_lock_.descriptor, &own) == 0 &&
        ::fstat(native, &inherited) == 0 && own.st_dev == inherited.st_dev &&
        own.st_ino == inherited.st_ino && flags >= 0 &&
        ::fcntl(native, F_SETFD, flags | FD_CLOEXEC) == 0 &&
        ::unsetenv("CYXWIZ_RUNTIME_USE_TOKEN") == 0)
      return true;
  }
#endif
  error = "Inherited Engine ownership does not match this installation or "
          "cannot be restricted";
  return false;
}
} // namespace cyxwiz::runtime
