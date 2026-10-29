#pragma once

// Disk caches of prepared data (TOFIX160 P7): the Parquet copies the Engine
// keeps so a dataset loads fast next time (the temp cache, a project's
// cache/ingestion) and the files a streamed preparation writes
// (cache/materialized/streamed). One rule for all of them: files older than
// the age go, then the oldest until the folder fits the size; a file a
// loaded dataset still reads is never removed. No ImGui.

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace cyxwiz {

// The limits the owner set (2026-10-21), as for the temp cache.
constexpr uint64_t kCacheFolderMaxBytes = 10ULL * 1024 * 1024 * 1024;
constexpr int kCacheFolderMaxAgeDays = 30;

struct CacheFolderUsage {
    int files = 0;
    uint64_t bytes = 0;
};

struct CachePruneResult {
    int removed = 0;
    uint64_t freed_bytes = 0;
    int kept = 0;
    uint64_t kept_bytes = 0;
    int in_use = 0;  // kept because a loaded dataset reads them
};

// Which files the rule covers: `extension` (".parquet"; empty: every file),
// directly in the folder.
struct CacheFolder {
    std::filesystem::path path;
    std::string extension = ".parquet";
    // True for a file a loaded dataset reads (never removed); unset: none.
    std::function<bool(const std::filesystem::path&)> in_use;
};

CacheFolderUsage MeasureCacheFolder(const CacheFolder& folder);

// Files older than `max_age_days` go, then the oldest until the folder holds
// at most `max_bytes`. A temp file (".tmp." in its name) a day old was left
// by a crash and goes too. A file that cannot be removed (open elsewhere)
// stays for the next prune.
CachePruneResult PruneCacheFolder(const CacheFolder& folder, uint64_t max_bytes = kCacheFolderMaxBytes,
                                  int max_age_days = kCacheFolderMaxAgeDays);

// Every file the rule covers, except those in use.
CachePruneResult ClearCacheFolder(const CacheFolder& folder);

// The prepared-data caches: this PC's temp cache (Parquet copies of large
// files), and with a project its cache/ingestion (Data Input copies, reused
// next time) and cache/materialized/streamed (a streamed preparation's
// output, written again each run). `in_use` is set on each.
struct NamedCacheFolder {
    std::string label;
    std::string detail;
    CacheFolder folder;
};
std::vector<NamedCacheFolder> PreparedDataCacheFolders(
    const std::filesystem::path& project_root,
    const std::function<bool(const std::filesystem::path&)>& in_use = {});

}  // namespace cyxwiz
