#pragma once

// The query engine behind the session query service (TOFIX134 P3 foundation,
// dashboard_architecture.md L2): one restricted DuckDB, tables attached by
// name as read-only views, read-only SQL with bound parameters, Arrow
// results. Synchronous and self-contained (tested on its own); the service
// (session_query_service.h) runs it off the UI thread with the catalog.
//
// Safety: user SQL must parse to one SELECT (WITH allowed); DuckDB runs with
// external access off, extensions not auto-loaded and the configuration
// locked, so read_csv / COPY / ATTACH / network fail inside DuckDB. Only the
// engine's own folders (the attachment folder, Parquet caches) are readable.
// Values are bound as parameters; identifiers the engine writes are quoted.
//
// Attachment: an in-memory table is written once per version to a Parquet
// file in the attachment folder (DuckDB reads Parquet natively and fast,
// where row-by-row copying would take minutes on a wide table); a
// disk-backed dataset is viewed straight from its cache file.
//
// Queries run side by side on a few connections to the one database
// (kConnections); attaching and reopening wait for running queries. A
// query's QueryToken stops it, also while it waits for a connection.

#include <arrow/api.h>

#include <cstdint>
#include <map>
#include <memory>
#include <condition_variable>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <string>
#include <vector>

namespace cyxwiz {

class DuckDBConnector;

struct QueryParam {
    enum class Type { Null, Bool, Int, Double, Text };
    Type type = Type::Null;
    bool b = false;
    int64_t i = 0;
    double d = 0;
    std::string s;
    static QueryParam Of(bool v) { QueryParam p; p.type = Type::Bool; p.b = v; return p; }
    static QueryParam Of(int64_t v) { QueryParam p; p.type = Type::Int; p.i = v; return p; }
    static QueryParam Of(double v) { QueryParam p; p.type = Type::Double; p.d = v; return p; }
    static QueryParam Of(std::string v) { QueryParam p; p.type = Type::Text; p.s = std::move(v); return p; }
};

struct QueryRequest {
    std::string sql;
    std::vector<QueryParam> params;   // bound to ? / $1 in order
    std::vector<std::string> inputs;  // tables to attach first (the service adds the ones the SQL names)
    size_t row_limit = 0;             // 0: all rows
    std::string label;                // shown in Task View ("Query: Spotify")
    bool cache = false;               // the result may be saved and read back (query_result_cache.h)
    // Set: the rows go to this Parquet file instead of the result (DuckDB
    // writes them itself, every column type kept); the result has one row
    // with their count. The SQL stays a read-only SELECT.
    std::string export_path;
};

struct QueryResult {
    bool ok = false;
    bool cancelled = false;
    std::string error;
    std::shared_ptr<arrow::Table> table;
    bool truncated = false;           // more rows than row_limit
    size_t rows_in_inputs = 0;        // rows of the tables the SQL names
    std::vector<std::string> inputs;  // those tables
    double elapsed_ms = 0;
    bool from_cache = false;          // read from a saved result, not run
};

// A query's handle for stopping it (Interrupt): before it starts, it does
// not start; while it runs, DuckDB stops it.
struct QueryToken {
    std::mutex mutex;
    DuckDBConnector* conn = nullptr;  // while running
    bool stop = false;
};

class SessionQueryEngine {
public:
    static constexpr size_t kConnections = 3;  // queries at once

    // `attach_dir` is created and owned (Parquet copies of in-memory tables).
    explicit SessionQueryEngine(std::string attach_dir);
    ~SessionQueryEngine();
    SessionQueryEngine(const SessionQueryEngine&) = delete;
    SessionQueryEngine& operator=(const SessionQueryEngine&) = delete;

    bool Ready() const;
    // Attach `table` as `name` (once per `version`: the same version is a no-op).
    bool AttachArrow(const std::string& name, uint64_t version, const std::shared_ptr<arrow::Table>& table, std::string* error);
    // Attach a Parquet file the engine owns (a disk-backed dataset's cache).
    bool AttachParquet(const std::string& name, uint64_t version, const std::string& path, size_t rows, std::string* error);
    void Detach(const std::string& name);
    bool IsAttached(const std::string& name, uint64_t version) const;
    std::vector<std::string> Attached() const;

    QueryResult Run(const QueryRequest& request, QueryToken* token = nullptr);
    // Thread-safe: stops every running or waiting query (Run returns cancelled).
    void Interrupt();
    // Thread-safe: stops the query run with `token` (or keeps it from starting).
    void Interrupt(QueryToken& token);

    // "a""b" style quoting for identifiers the engine writes.
    static std::string QuoteIdentifier(const std::string& name);
    // The attached names a SQL text mentions (case-insensitive, whole words or quoted).
    std::vector<std::string> NamesIn(const std::string& sql) const;

private:
    struct Attachment {
        uint64_t version = 0;
        std::string path;
        size_t rows = 0;
        bool owned_file = false;  // written by AttachArrow (removed on Detach)
    };
    bool Open(std::string* error);  // (re)opens DuckDB with the allowed folders and re-creates the views
    bool CreateView(const std::string& name, const std::string& path, std::string* error);
    // Under the exclusive lock: `folder` becomes one DuckDB may read and write.
    bool AllowFolderLocked(const std::string& folder, std::string* error);
    std::vector<std::string> NamesInLocked(const std::string& sql) const;
    // A free connection for `token` (waits); null when the token was stopped.
    DuckDBConnector* Acquire(QueryToken& token);
    void Release(DuckDBConnector* conn, QueryToken& token);

    std::string dir_;
    std::vector<std::string> allowed_;  // folders DuckDB may read
    std::unique_ptr<DuckDBConnector> db_;                 // the database (its connection is one of the pool)
    std::vector<std::unique_ptr<DuckDBConnector>> extra_;  // more connections to it
    std::map<std::string, Attachment> attached_;
    mutable std::shared_mutex mutex_;   // queries shared; attaching and reopening exclusive
    std::mutex pool_mutex_;             // free_ and running_ (taken before a token's mutex)
    std::condition_variable pool_cv_;
    std::vector<DuckDBConnector*> free_;
    std::set<QueryToken*> running_;     // waiting or running
};

}  // namespace cyxwiz
