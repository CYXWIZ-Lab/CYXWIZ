#pragma once

// The session query service (TOFIX134 P3 foundation, dashboard_architecture.md
// L2): the one place the Engine's UI (Data Studio Query / Profile /
// Visualize, dashboards) runs SQL. Tables are named as the dataset catalog
// names them ("Spotify", "MNIST"); the ones a query names are attached
// before it runs. Every query runs off the UI thread as a task (Task View
// shows it, Cancel stops it) and its result comes back on the UI thread as
// an Arrow table. A few queries run side by side (SessionQueryEngine's
// connections); each has its own stop.
//
// A request that asks for it (QueryRequest::cache) has its result saved in
// the open project (cache/query_results, see query_result_cache.h) and read
// back when the same query meets the same data again, also in a later
// session. The same key asked for twice at once runs once.
//
// Graph execution keeps its own SqlTransform (the SQL step); this service
// is for looking at data, not for the pipeline.

#include "query_result_cache.h"
#include "session_query_engine.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <map>
#include <string>
#include <vector>

namespace cyxwiz {

class SessionQueryService {
public:
    static SessionQueryService& Instance();

    using Done = std::function<void(const QueryResult&)>;
    // Runs `request` in the background; `done` runs on the UI thread (not when
    // `owner` is gone). Returns the task id (Cancel).
    uint64_t Submit(QueryRequest request, Done done, std::weak_ptr<const void> owner = {});
    void Cancel(uint64_t task_id);

    // The datasets a query can name now: tabular catalog entries (in-memory
    // Arrow and disk-backed Parquet). Image / audio / text file tables come
    // with their dashboard templates.
    std::vector<std::string> QueryableNames() const;

    // Runs on the calling thread (workers that already are off the UI thread).
    // `token` (optional) lets Interrupt stop this query alone.
    QueryResult RunNow(const QueryRequest& request, QueryToken* token = nullptr);
    // Thread-safe: stops the query run with `token` (or keeps it from starting).
    void Interrupt(QueryToken& token);

    // The open project's root (saved results live under it); empty: none open,
    // results are not saved.
    void SetProjectRoot(const std::string& root);
    QueryResultCache& Cache() { return cache_; }

    // A Parquet table only the Engine's own queries name (a dashboard's text
    // words): attached when a query names it, never listed as a dataset.
    void SetSideTable(const std::string& name, const std::string& parquet_path, size_t rows);

private:
    SessionQueryService();
    // Attaches the catalog tables the request names (or lists); false + error.
    // `identities` (optional) gets one identity per table (the cache key);
    // with `attach` false only those are collected (a saved result needs no
    // table: attaching copies an in-memory table once per session).
    bool AttachInputs(const QueryRequest& request, std::string* error, std::vector<std::string>* identities = nullptr,
                      bool attach = true);
    // The identity of an in-memory table's content (computed once per table).
    std::string ArrowIdentity(const std::shared_ptr<arrow::Table>& table);
    struct SideTable {
        std::string path;
        size_t rows = 0;
    };
    std::mutex side_mutex_;
    std::map<std::string, SideTable> side_tables_;

    std::unique_ptr<SessionQueryEngine> engine_;
    QueryResultCache cache_;
    struct ArrowIdentityEntry {
        std::weak_ptr<arrow::Table> table;  // the address may be reused by a new table
        std::string identity;
    };
    std::mutex identity_mutex_;
    std::map<const arrow::Table*, ArrowIdentityEntry> arrow_identities_;
};

}  // namespace cyxwiz
