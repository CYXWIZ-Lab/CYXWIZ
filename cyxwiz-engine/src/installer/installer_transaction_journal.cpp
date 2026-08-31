#include "installer_transaction_journal.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <system_error>

namespace cyxwiz::installer {
namespace {

std::string PathText(const std::filesystem::path &path) {
  const auto value = path.u8string();
  return {reinterpret_cast<const char *>(value.data()), value.size()};
}

} // namespace

std::filesystem::path DefaultInstallerTransactionJournalPath() {
#ifdef _WIN32
  if (const char *local = std::getenv("LOCALAPPDATA")) {
    return std::filesystem::path(local) / "CyxWiz" / "Installer" /
           "last-transaction.json";
  }
#elif defined(__APPLE__)
  if (const char *home = std::getenv("HOME")) {
    return std::filesystem::path(home) / "Library" / "Logs" / "CyxWiz" /
           "Installer" / "last-transaction.json";
  }
#else
  if (const char *state = std::getenv("XDG_STATE_HOME")) {
    return std::filesystem::path(state) / "cyxwiz" / "installer" /
           "last-transaction.json";
  }
  if (const char *home = std::getenv("HOME")) {
    return std::filesystem::path(home) / ".local" / "state" / "cyxwiz" /
           "installer" / "last-transaction.json";
  }
#endif
  std::error_code error;
  const auto temporary = std::filesystem::temp_directory_path(error);
  return error ? std::filesystem::path{}
               : temporary / "CyxWiz" / "Installer" /
                     "last-transaction.json";
}

std::string CurrentInstallerTransactionUtc() {
  const auto now = std::chrono::system_clock::to_time_t(
      std::chrono::system_clock::now());
  std::tm utc{};
#ifdef _WIN32
  gmtime_s(&utc, &now);
#else
  gmtime_r(&now, &utc);
#endif
  std::ostringstream output;
  output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
  return output.str();
}

bool WriteInstallerTransactionJournal(
    const std::filesystem::path &path,
    const InstallerTransactionRecord &record,
    std::string &error) {
  if (!path.is_absolute() || record.transaction_id.empty() ||
      !record.runtime_root.is_absolute() || record.status.empty()) {
    error = "Installer transaction journal requires absolute bounded state";
    return false;
  }
  std::error_code filesystem_error;
  std::filesystem::create_directories(path.parent_path(), filesystem_error);
  if (filesystem_error) {
    error = "Cannot create the installer journal directory: " +
            filesystem_error.message();
    return false;
  }
  const nlohmann::json value = {
      {"schema_version", 1},
      {"kind", "cyxwiz-installer-transaction"},
      {"transaction_id", record.transaction_id},
      {"started_utc", record.started_utc},
      {"completed_utc", record.completed_utc.empty()
                            ? nlohmann::json(nullptr)
                            : nlohmann::json(record.completed_utc)},
      {"status", record.status},
      {"message", record.message},
      {"runtime_root", PathText(record.runtime_root)},
      {"scope", record.scope == CyxWizInstallScope::AllUsers
                    ? "all_users"
                    : "current_user"},
      {"plan",
       {{"install_base", record.plan.install_base},
        {"update_base", record.plan.update_base},
        {"base_pack_id", record.plan.base_pack_id},
        {"install_pack_ids", record.plan.pack_ids},
        {"remove_pack_ids", record.plan.remove_pack_ids},
        {"deactivate_backends", record.plan.deactivate_backends}}}};
  auto next = path;
  next += ".next";
  {
    std::ofstream output(next, std::ios::binary | std::ios::trunc);
    if (!output) {
      error = "Cannot open the installer transaction journal";
      return false;
    }
    output << value.dump(2) << '\n';
    if (!output) {
      error = "Cannot write the installer transaction journal";
      return false;
    }
  }
#ifdef _WIN32
  std::filesystem::remove(path, filesystem_error);
  filesystem_error.clear();
#endif
  std::filesystem::rename(next, path, filesystem_error);
  if (filesystem_error) {
    const auto rename_error = filesystem_error.message();
    filesystem_error.clear();
    std::filesystem::remove(next, filesystem_error);
    error = "Cannot publish the installer transaction journal: " +
            rename_error;
    return false;
  }
  return true;
}

} // namespace cyxwiz::installer
