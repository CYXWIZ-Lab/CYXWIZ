#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

namespace cyxwiz::runtime {

enum class RuntimeOperationLockStatus { Acquired, Busy, Failed };

// Cooperative product ownership: exclusive maintenance or shared Engine use.
// Keep lock files in place: removing them can split POSIX flock ownership.
// Noncopyable, nonmovable; a borrowed owner must outlive its synchronous
// caller.
class RuntimeOperationLock {
public:
  RuntimeOperationLock() = default;
  ~RuntimeOperationLock() = default;
  RuntimeOperationLock(const RuntimeOperationLock &) = delete;
  RuntimeOperationLock &operator=(const RuntimeOperationLock &) = delete;

  RuntimeOperationLockStatus Acquire(const std::filesystem::path &runtime_root,
                                     std::string &error);
  // Shared read ownership excludes maintenance but allows multiple Engines.
  RuntimeOperationLockStatus
  AcquireEngineUse(const std::filesystem::path &runtime_root,
                   std::string &error);
  // Launcher only: retain ownership in the child even if the launcher exits.
  bool AllowEngineChildInheritance(std::string &error);
  // Validate the inherited token, then stop it propagating to helpers.
  // The OS retains the inherited handle until Engine process exit.
  bool RestrictInheritedEngineOwnership(std::string &error);
  // Close the legacy in-product handle before renaming/deleting the product.
  // The sibling product-location lock remains held until destruction.
  void PrepareForQuarantine();
  static std::filesystem::path
  LocationLockPath(const std::filesystem::path &runtime_root);
  bool Owns(const std::filesystem::path &runtime_root) const;

private:
  RuntimeOperationLockStatus AcquireInternal(const std::filesystem::path &,
                                             std::string &, bool);
  struct FileLock {
    FileLock() = default;
    ~FileLock() { Close(); }
    FileLock(const FileLock &) = delete;
    FileLock &operator=(const FileLock &) = delete;
    RuntimeOperationLockStatus Acquire(const std::filesystem::path &,
                                       std::string &, bool shared = false);
    void Close() noexcept;
#ifdef _WIN32
    std::uintptr_t handle = 0;
#else
    int descriptor = -1;
#endif
  };
  std::filesystem::path runtime_root_;
  FileLock location_lock_;
  FileLock runtime_lock_;
  bool preparing_quarantine_ = false;
  bool engine_use_ = false;
};

} // namespace cyxwiz::runtime
