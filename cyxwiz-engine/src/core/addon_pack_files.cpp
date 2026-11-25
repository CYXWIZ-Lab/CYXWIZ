#include "addon_pack_files.h"

#include "../plugin/plugin_pack_checks.h"

#include "backend_pack_installed_files.h"
#include "backend_pack_metadata_verifier.h"
#include "runtime_layout.h"

#include <chrono>
#include <ctime>
#include <map>
#include <optional>

namespace cyxwiz {
namespace {

std::string ClockNow() {
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    char text[16];
    std::strftime(text, sizeof(text), "%H:%M", &local);
    return text;
}

struct FileCheck {
    std::size_t listed = 0;
    std::vector<plugin::PackFileIssue> issues;
    std::string error;
};

FileCheck CheckOne(const std::filesystem::path& runtime_root, const BackendPackManagerRecord& record,
                   const runtime::BackendPackMetadataVerifier& verifier) {
    FileCheck result;
    runtime::VerifiedBackendPackManifest manifest;
    std::string error;
    if (!runtime::VerifyInstalledPluginPack(runtime_root, record.plugin_id, record.installed_pack_id, verifier,
                                            manifest, error)) {
        result.error = error;
        return result;
    }
    result.listed = manifest.components.size();
    const auto folder = plugin::PackFolderCheckOf(runtime::CheckInstalledPackFiles(
        runtime::PluginPackDirectory(runtime_root, record.plugin_id, record.installed_pack_id),
        manifest.components));
    result.issues = folder.issues;
    result.error = folder.error;  // a check that could not run (a folder it could not list)
    return result;
}

bool IsCheckedAddOn(const BackendPackManagerRecord& record) {
    return record.pack_kind == BackendPackRecordKind::Plugin && record.installed &&
           !record.installed_for_other_base && !record.installed_pack_id.empty() && !record.plugin_id.empty();
}

}  // namespace

void MarkAddOnFileChecksPending(std::vector<BackendPackManagerRecord>& records) {
    for (auto& record : records) {
        if (IsCheckedAddOn(record)) record.files_check_pending = true;
    }
}

void CheckInstalledAddOnFiles(std::vector<BackendPackManagerRecord>& records,
                              const std::filesystem::path& runtime_root, const std::string& client_version) {
    if (runtime_root.empty()) return;
    std::string trust_error;
    const auto verifier = runtime::LoadBackendPackMetadataVerifier(runtime_root / "trust" / "trusted-keys.json",
                                                                   client_version, trust_error);
    std::map<std::string, FileCheck> checked;  // installed pack id -> its check
    const std::string now = ClockNow();
    for (auto& record : records) {
        if (!IsCheckedAddOn(record)) continue;
        record.files_check_pending = false;
        if (!verifier) {
            record.files_check_note =
                "Its files could not be checked: the installation's trust file could not be read (" +
                trust_error + ").";
            continue;
        }
        auto found = checked.find(record.installed_pack_id);
        if (found == checked.end())
            found = checked.emplace(record.installed_pack_id, CheckOne(runtime_root, record, *verifier)).first;
        record.files_checked = true;
        record.files_listed = found->second.listed;
        record.file_issues = found->second.issues;
        record.files_error = found->second.error;
        record.files_checked_at = now;
        record.integrity_verified = record.file_issues.empty() && record.files_error.empty();
    }
}

void MergeAddOnFileChecks(std::vector<BackendPackManagerRecord>& records,
                          const std::vector<BackendPackManagerRecord>& checked) {
    for (auto& record : records) {
        if (!IsCheckedAddOn(record)) continue;
        for (const auto& result : checked) {
            if (result.pack_id != record.pack_id || result.installed_pack_id != record.installed_pack_id) continue;
            record.files_check_pending = result.files_check_pending;
            record.files_check_note = result.files_check_note;
            record.files_checked = result.files_checked;
            record.files_listed = result.files_listed;
            record.file_issues = result.file_issues;
            record.files_error = result.files_error;
            record.files_checked_at = result.files_checked_at;
            record.integrity_verified = result.integrity_verified;
            break;
        }
    }
}

}  // namespace cyxwiz
