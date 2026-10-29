#include "cache_prune.h"

#include "parquet_backed_dataset.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <mutex>
#include <vector>

namespace cyxwiz {

namespace fs = std::filesystem;

namespace {

struct CacheFile {
    fs::path path;
    uint64_t size = 0;
    fs::file_time_type written{};
};

bool IsTemp(const fs::path& p) { return p.filename().string().find(".tmp.") != std::string::npos; }

// The files the rule covers, with their size and time; a day-old temp file
// is removed on the way (it was left by a crash).
std::vector<CacheFile> ListFiles(const CacheFolder& folder, bool drop_stale_temps) {
    std::vector<CacheFile> files;
    std::error_code ec;
    if (folder.path.empty() || !fs::is_directory(folder.path, ec)) return files;
    for (fs::directory_iterator it(folder.path, ec), end; !ec && it != end; it.increment(ec)) {
        std::error_code entry_ec;
        if (!it->is_regular_file(entry_ec)) continue;
        const fs::path& p = it->path();
        if (IsTemp(p)) {
            const auto written = fs::last_write_time(p, entry_ec);
            if (drop_stale_temps && !entry_ec && fs::file_time_type::clock::now() - written > std::chrono::hours(24))
                fs::remove(p, entry_ec);
            continue;
        }
        if (!folder.extension.empty() && p.extension() != folder.extension) continue;
        CacheFile f;
        f.path = p;
        f.size = it->file_size(entry_ec);
        if (entry_ec) f.size = 0;
        f.written = fs::last_write_time(p, entry_ec);
        if (entry_ec) continue;
        files.push_back(std::move(f));
    }
    return files;
}

bool InUse(const CacheFolder& folder, const fs::path& p) { return folder.in_use && folder.in_use(p); }

}  // namespace

CacheFolderUsage MeasureCacheFolder(const CacheFolder& folder) {
    CacheFolderUsage usage;
    for (const auto& f : ListFiles(folder, false)) {
        ++usage.files;
        usage.bytes += f.size;
    }
    return usage;
}

CachePruneResult PruneCacheFolder(const CacheFolder& folder, uint64_t max_bytes, int max_age_days) {
    // Loads on other threads prune too; one walk at a time.
    static std::mutex prune_mutex;
    std::lock_guard<std::mutex> lock(prune_mutex);
    CachePruneResult result;
    auto files = ListFiles(folder, true);
    if (files.empty()) return result;

    // Oldest first: the age pass removes from the front, the size pass then
    // continues there.
    std::sort(files.begin(), files.end(), [](const CacheFile& a, const CacheFile& b) { return a.written < b.written; });
    uint64_t total = 0;
    for (const auto& f : files) total += f.size;
    const auto cutoff = fs::file_time_type::clock::now() - std::chrono::hours(24) * max_age_days;
    for (const auto& f : files) {
        const bool expired = f.written < cutoff;
        if (!expired && total <= max_bytes) {
            ++result.kept;
            result.kept_bytes += f.size;
            continue;
        }
        std::error_code ec;
        if (InUse(folder, f.path)) {
            ++result.in_use;
        } else if (fs::remove(f.path, ec) && !ec) {
            ++result.removed;
            result.freed_bytes += f.size;
            total -= std::min(total, f.size);
            continue;
        }
        ++result.kept;
        result.kept_bytes += f.size;
    }
    if (result.removed > 0)
        spdlog::info("Cache {}: removed {} file(s) ({:.1f} MB), {} kept ({:.1f} MB)", folder.path.string(),
                     result.removed, result.freed_bytes / (1024.0 * 1024.0), result.kept,
                     result.kept_bytes / (1024.0 * 1024.0));
    return result;
}

CachePruneResult ClearCacheFolder(const CacheFolder& folder) {
    CachePruneResult result;
    for (const auto& f : ListFiles(folder, true)) {
        std::error_code ec;
        if (!InUse(folder, f.path) && fs::remove(f.path, ec) && !ec) {
            ++result.removed;
            result.freed_bytes += f.size;
        } else {
            if (InUse(folder, f.path)) ++result.in_use;
            ++result.kept;
            result.kept_bytes += f.size;
        }
    }
    return result;
}

std::vector<NamedCacheFolder> PreparedDataCacheFolders(
    const fs::path& project_root, const std::function<bool(const fs::path&)>& in_use) {
    std::vector<NamedCacheFolder> folders;
    const auto add = [&](std::string label, std::string detail, fs::path path) {
        NamedCacheFolder f;
        f.label = std::move(label);
        f.detail = std::move(detail);
        f.folder.path = std::move(path);
        f.folder.in_use = in_use;
        folders.push_back(std::move(f));
    };
    add("Temporary (this PC)", "Parquet copies of files too large for memory", ParquetBackedDataset::GetCacheDir());
    if (!project_root.empty()) {
        add("Prepared inputs (this project)", "Data Input copies, reused the next time a file loads",
            project_root / "cache" / "ingestion");
        add("Streamed preparations (this project)", "the output of preparing data kept on disk, written again each run",
            project_root / "cache" / "materialized" / "streamed");
    }
    return folders;
}

}  // namespace cyxwiz
