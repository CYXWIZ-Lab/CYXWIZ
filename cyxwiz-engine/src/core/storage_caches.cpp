#include "storage_caches.h"

#include "cache_prune.h"
#include "materialization_cache.h"

namespace cyxwiz {

namespace fs = std::filesystem;

namespace {

constexpr const char* kRuleLimit = "kept 30 days, at most 10 GB";

// The ids of PreparedDataCacheFolders, in its order.
const char* const kFolderIds[] = {"temp", "ingestion", "streamed"};

MaterializationCacheConfig PreparedConfig(const fs::path& project_root) {
    // As the training launcher sets it (graph_training_launcher.cpp).
    MaterializationCacheConfig config;
    config.cache_root = project_root;
    config.max_total_bytes = 10ull * 1024ull * 1024ull * 1024ull;
    config.max_entries = 32;
    return config;
}

CacheFolder QueriesFolder(const fs::path& project_root) {
    CacheFolder folder;
    folder.path = project_root / "cache" / "query_results";  // SessionQueryService::SetProjectRoot
    return folder;
}

}  // namespace

std::string StorageItemsText(int items, const std::string& unit) {
    const std::string plural = unit == "entry" ? "entries" : unit + "s";
    if (items == 0) return "no " + plural;
    return std::to_string(items) + " " + (items == 1 ? unit : plural);
}

std::vector<StorageCache> MeasureStorageCaches(const fs::path& project_root,
                                               const std::function<bool(const fs::path&)>& in_use) {
    std::vector<StorageCache> caches;
    const auto folders = PreparedDataCacheFolders(project_root, in_use);
    for (size_t i = 0; i < folders.size() && i < std::size(kFolderIds); ++i) {
        const auto& f = folders[i];
        StorageCache c;
        c.id = kFolderIds[i];
        c.label = f.label;
        c.detail = f.detail;
        c.path = f.folder.path.string();
        c.limit = kRuleLimit;
        const auto usage = MeasureCacheFolder(f.folder);
        c.bytes = usage.bytes;
        c.items = usage.files;
        caches.push_back(std::move(c));
    }
    if (project_root.empty()) return caches;

    StorageCache prepared;
    prepared.id = "prepared";
    prepared.label = "Prepared datasets (this project)";
    prepared.detail = "training data after its preparation steps, reused when the steps are unchanged";
    const auto config = PreparedConfig(project_root);
    prepared.path = MaterializationCacheDirectory(config).string();
    prepared.limit = "at most 32 entries and 10 GB (least recently used go)";
    prepared.unit = "entry";
    const auto usage = MeasureMaterializationCache(config);
    prepared.bytes = usage.total_bytes;
    prepared.items = usage.entries;
    caches.push_back(std::move(prepared));

    StorageCache queries;
    queries.id = "queries";
    queries.label = "Saved query answers (this project)";
    queries.detail = "Data Studio and dashboard query results, reused when the data is unchanged";
    const auto folder = QueriesFolder(project_root);
    queries.path = folder.path.string();
    queries.limit = "at most 512 MB (least recently used go)";
    const auto q = MeasureCacheFolder(folder);
    queries.bytes = q.bytes;
    queries.items = q.files;
    caches.push_back(std::move(queries));

    // Files a loaded dataset reads, per folder (Clear keeps them).
    for (auto& c : caches) {
        if (!in_use || c.id == "prepared") continue;
        std::error_code ec;
        if (!fs::is_directory(c.path, ec)) continue;
        for (fs::directory_iterator it(c.path, ec), end; !ec && it != end; it.increment(ec))
            if (in_use(it->path())) ++c.in_use;
    }
    return caches;
}

StorageClearResult ClearStorageCache(const std::string& id, const fs::path& project_root,
                                     const std::function<bool(const fs::path&)>& in_use) {
    StorageClearResult out;
    const auto take = [&out](const CachePruneResult& r) {
        out.removed = r.removed;
        out.freed_bytes = r.freed_bytes;
        out.kept = r.kept;
    };
    const auto folders = PreparedDataCacheFolders(project_root, in_use);
    for (size_t i = 0; i < folders.size() && i < std::size(kFolderIds); ++i) {
        if (id == kFolderIds[i]) {
            take(ClearCacheFolder(folders[i].folder));
            return out;
        }
    }
    if (project_root.empty()) return out;
    if (id == "prepared") {
        const auto r = ClearMaterializationCache(PreparedConfig(project_root));
        out.removed = r.removed_entries;
        out.freed_bytes = r.freed_bytes;
        out.kept = r.remaining_entries;
    } else if (id == "queries") {
        // Deleting saved answers is safe: a query runs again and saves anew.
        auto folder = QueriesFolder(project_root);
        folder.in_use = in_use;
        take(ClearCacheFolder(folder));
    }
    return out;
}

}  // namespace cyxwiz
