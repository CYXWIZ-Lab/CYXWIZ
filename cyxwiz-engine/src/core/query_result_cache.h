#pragma once

// Saved query results (TOFIX134, owner 2026-10-05: "only once and next
// should load"). A query that asks for it (QueryRequest::cache: dashboard
// widgets, KPIs, the summary strip, profiles) has its result saved as
// Parquet in the project's cache/query_results folder, keyed by the SQL, its
// values and the identity of every table it reads (the content of an
// in-memory table, the file of a disk-backed one). The same query on the
// same data, in this session or a later one, then reads the file instead of
// running. Two callers asking for the same key at once run it once.
// The folder is kept under kMaxBytes (least recently used first); deleting
// it is safe.

#include "session_query_engine.h"

#include <arrow/api.h>

#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace cyxwiz {

class QueryResultCache {
public:
    static constexpr uint64_t kMaxBytes = 512ull * 1024 * 1024;       // the folder
    static constexpr uint64_t kMaxResultBytes = 32ull * 1024 * 1024;  // one result (bigger ones are not saved)

    // Empty: saving and loading are off (no project open).
    void SetFolder(const std::filesystem::path& folder);
    std::filesystem::path Folder() const;

    // The key for `request` reading tables with these identities (in any order).
    static std::string Key(const QueryRequest& request, std::vector<std::string> identities);
    // A table's identity from its content (every byte of every column; the
    // same values in a new table give the same identity).
    static std::string ContentIdentity(const arrow::Table& table);
    // A file's identity: its path, size and last change.
    static std::string FileIdentity(const std::string& path);

    // The saved result, or null. A hit counts as a use (pruning keeps it).
    // `truncated` (optional) gets whether the result was cut at its row limit.
    std::shared_ptr<arrow::Table> Load(const std::string& key, bool* truncated = nullptr) const;
    // Saves `table` (a temporary file, then renamed); false when off, too big or not written.
    bool Save(const std::string& key, const std::shared_ptr<arrow::Table>& table, bool truncated = false);
    // Removes the least recently used results until the folder fits `max_bytes`.
    void Prune(uint64_t max_bytes = kMaxBytes);

    // The same key asked for twice at once: the first caller runs it (leader),
    // the others wait for its result (Wait; null when stopped or the leader
    // did not finish with a result).
    struct Flight {
        std::mutex mutex;
        std::condition_variable cv;
        bool done = false;
        QueryResult result;
    };
    std::shared_ptr<Flight> Join(const std::string& key, bool* leader);
    void Finish(const std::string& key, const std::shared_ptr<Flight>& flight, const QueryResult& result);
    static bool Wait(Flight& flight, const std::function<bool()>& stopped, QueryResult* out);

private:
    std::filesystem::path PathOf(const std::string& key) const;

    mutable std::mutex mutex_;
    std::filesystem::path folder_;
    std::map<std::string, std::shared_ptr<Flight>> flights_;
    size_t saves_ = 0;  // prune every few saves
};

}  // namespace cyxwiz
