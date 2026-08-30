#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

namespace cyxwiz::installer {

std::uint64_t CurrentInstallerProcessId();
bool CleanupAbandonedInstallerStaging(
    const std::filesystem::path &runtime_root, const std::string &pack_id,
    std::string &error);

class InstallerHelperSession {
public:
  InstallerHelperSession() = default;
  ~InstallerHelperSession();

  InstallerHelperSession(const InstallerHelperSession &) = delete;
  InstallerHelperSession &operator=(const InstallerHelperSession &) = delete;

  bool Open(const std::filesystem::path &runtime_root,
            std::uint64_t parent_process_id, std::string &error);
  bool ParentExited() const;

private:
#ifdef _WIN32
  std::uintptr_t operation_handle_ = 0;
  std::uintptr_t parent_handle_ = 0;
#else
  int operation_descriptor_ = -1;
  std::int64_t parent_process_id_ = -1;
#endif
};

} // namespace cyxwiz::installer
