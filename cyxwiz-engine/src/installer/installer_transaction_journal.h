#pragma once

#include "../core/backend_pack_manager_model.h"

#include <filesystem>
#include <string>

namespace cyxwiz::installer {

struct InstallerTransactionRecord {
  std::string transaction_id;
  std::string started_utc;
  std::string completed_utc;
  std::string status;
  std::string message;
  std::filesystem::path runtime_root;
  CyxWizInstallScope scope = CyxWizInstallScope::CurrentUser;
  BackendPackInstallerPlan plan;
};

std::filesystem::path DefaultInstallerTransactionJournalPath();
std::string CurrentInstallerTransactionUtc();
bool WriteInstallerTransactionJournal(
    const std::filesystem::path &path,
    const InstallerTransactionRecord &record,
    std::string &error);

} // namespace cyxwiz::installer
