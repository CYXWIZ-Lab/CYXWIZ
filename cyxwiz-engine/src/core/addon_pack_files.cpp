#include "addon_pack_files.h"

#include "backend_pack_installed_files.h"
#include "backend_pack_metadata_verifier.h"
#include "runtime_layout.h"

#include <chrono>
#include <ctime>
#include <map>
#include <string>

namespace cyxwiz {
namespace {

plugin::PackFileProblem ProblemOf(runtime::PackFileFault fault) {
    switch (fault) {
        case runtime::PackFileFault::Missing: return plugin::PackFileProblem::Missing;
        case runtime::PackFileFault::SizeDiffers: return plugin::PackFileProblem::SizeDiffers;
        case runtime::PackFileFault::Unreadable: return plugin::PackFileProblem::Unreadable;
        case runtime::PackFileFault::NotListed: return plugin::PackFileProblem::NotListed;
        case runtime::PackFileFault::ContentDiffers:
        case runtime::PackFileFault::None: break;
    }
    return plugin::PackFileProblem::ContentDiffers;
}

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
    const auto check = runtime::CheckInstalledPackFiles(
        runtime::PluginPackDirectory(runtime_root, record.plugin_id, record.installed_pack_id),
        manifest.components);
    for (const auto& finding : check.findings)
        result.issues.push_back({finding.relative_path, ProblemOf(finding.fault)});
    if (result.issues.empty()) result.error = check.error;  // a check that could not run
    return result;
}

}  // namespace

void CheckInstalledAddOnFiles(std::vector<BackendPackManagerRecord>& records,
                              const std::filesystem::path& runtime_root,
                              const runtime::BackendPackMetadataVerifier& verifier) {
    if (runtime_root.empty()) return;
    std::map<std::string, FileCheck> checked;  // installed pack id -> its check
    const std::string now = ClockNow();
    for (auto& record : records) {
        if (record.pack_kind != BackendPackRecordKind::Plugin || !record.installed ||
            record.installed_for_other_base || record.installed_pack_id.empty() || record.plugin_id.empty())
            continue;
        auto found = checked.find(record.installed_pack_id);
        if (found == checked.end())
            found = checked.emplace(record.installed_pack_id, CheckOne(runtime_root, record, verifier)).first;
        record.files_checked = true;
        record.files_listed = found->second.listed;
        record.file_issues = found->second.issues;
        record.files_error = found->second.error;
        record.files_checked_at = now;
        record.integrity_verified = record.file_issues.empty() && record.files_error.empty();
    }
}

}  // namespace cyxwiz
