#include "session_query_service.h"

#include "arrow_dataset.h"
#include "async_task_manager.h"
#include "data_registry.h"
#include "dataset_catalog.h"
#include "parquet_backed_dataset.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>

#include <spdlog/spdlog.h>

#ifdef _WIN32
#include <process.h>
#define CYXWIZ_GETPID _getpid
#else
#include <unistd.h>
#define CYXWIZ_GETPID getpid
#endif

namespace cyxwiz {

namespace fs = std::filesystem;

namespace {

std::string Lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// Whether `sql` names `name` as a whole word (or inside quotes).
bool Mentions(const std::string& lower_sql, const std::string& name) {
    const std::string n = Lower(name);
    if (n.empty()) return false;
    const auto word = [](char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; };
    for (size_t at = lower_sql.find(n); at != std::string::npos; at = lower_sql.find(n, at + 1)) {
        const bool start = at == 0 || !word(lower_sql[at - 1]);
        const bool end = at + n.size() >= lower_sql.size() || !word(lower_sql[at + n.size()]);
        if (start && end) return true;
    }
    return false;
}

bool Queryable(const DatasetEntry& e) {
    return e.storage == DatasetStorageKind::InMemoryArrow || e.storage == DatasetStorageKind::DiskBackedParquet;
}

}  // namespace

SessionQueryService& SessionQueryService::Instance() {
    static SessionQueryService service;
    return service;
}

SessionQueryService::SessionQueryService() {
    // This session's own folder for table copies; old sessions' folders are removed.
    const fs::path base = fs::temp_directory_path() / "cyxwiz" / "query_sessions";
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(base, ec)) fs::remove_all(entry.path(), ec);
    engine_ = std::make_unique<SessionQueryEngine>((base / std::to_string(CYXWIZ_GETPID())).string());
    if (!engine_->Ready()) spdlog::error("Session query service: the query engine did not start");
}

std::vector<std::string> SessionQueryService::QueryableNames() const {
    std::vector<std::string> out;
    for (const auto& e : DatasetCatalog::Instance().List())
        if (Queryable(e)) out.push_back(e.Shown());
    return out;
}

std::string SessionQueryService::ArrowIdentity(const std::shared_ptr<arrow::Table>& table) {
    {
        std::lock_guard<std::mutex> lock(identity_mutex_);
        auto it = arrow_identities_.find(table.get());
        if (it != arrow_identities_.end() && it->second.table.lock() == table) return it->second.identity;
    }
    const std::string identity = QueryResultCache::ContentIdentity(*table);
    std::lock_guard<std::mutex> lock(identity_mutex_);
    // Tables that are gone are forgotten.
    for (auto it = arrow_identities_.begin(); it != arrow_identities_.end();)
        it = it->second.table.expired() ? arrow_identities_.erase(it) : std::next(it);
    arrow_identities_[table.get()] = ArrowIdentityEntry{table, identity};
    return identity;
}

bool SessionQueryService::AttachInputs(const QueryRequest& request, std::string* error, std::vector<std::string>* identities,
                                       bool attach) {
    const std::string sql = Lower(request.sql);
    const auto entries = DatasetCatalog::Instance().List();
    std::vector<std::string> catalog_names;
    for (const auto& e : entries) {
        catalog_names.push_back(e.name);
        if (!e.label.empty()) catalog_names.push_back(e.label);
    }
    std::map<std::string, SideTable> side;
    {
        std::lock_guard<std::mutex> lock(side_mutex_);
        side = side_tables_;
    }
    for (const auto& [name, t] : side) catalog_names.push_back(name);  // kept attached
    // Copies of datasets that are gone are dropped.
    if (attach)
        for (const auto& name : engine_->Attached())
            if (std::find(catalog_names.begin(), catalog_names.end(), name) == catalog_names.end()) engine_->Detach(name);
    auto& registry = DataRegistry::Instance();
    for (const auto& e : entries) {
        // A table is named by its dataset name or by its label (the Data Input node's name).
        std::vector<std::string> as;
        for (const std::string* n : {&e.name, &e.label}) {
            if (n->empty()) continue;
            const bool listed = std::find(request.inputs.begin(), request.inputs.end(), *n) != request.inputs.end();
            if (listed || Mentions(sql, *n)) as.push_back(*n);
            if (listed && !Queryable(e)) {
                if (error) *error = "'" + *n + "' is " + StorageText(e.storage) + ": only tables can be queried for now.";
                return false;
            }
        }
        if (as.empty() || !Queryable(e)) continue;
        for (const auto& view : as) {
            if (e.storage == DatasetStorageKind::InMemoryArrow) {
                auto ds = registry.GetArrowDataset(e.name);
                auto table = ds ? ds->GetArrowTable() : nullptr;
                if (!table) break;  // removed meanwhile
                // The table object's address is its version: a re-load is a new table.
                if (attach && !engine_->AttachArrow(view, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(table.get())), table, error))
                    return false;
                if (identities) identities->push_back(view + "=" + ArrowIdentity(table));
            } else {
                auto ds = registry.GetParquetBackedDataset(e.name);
                if (!ds) break;
                const uint64_t version = std::hash<std::string>{}(ds->GetFilePath()) ^ static_cast<uint64_t>(ds->GetNumRows());
                if (attach && !engine_->AttachParquet(view, version, ds->GetFilePath(), static_cast<size_t>(ds->GetNumRows()), error))
                    return false;
                if (identities) identities->push_back(view + "=parquet:" + QueryResultCache::FileIdentity(ds->GetFilePath()));
            }
        }
    }
    // Side tables the request names or lists.
    for (const auto& [name, t] : side) {
        const bool listed = std::find(request.inputs.begin(), request.inputs.end(), name) != request.inputs.end();
        if (!listed && !Mentions(sql, Lower(name))) continue;
        const uint64_t version = std::hash<std::string>{}(t.path) ^ static_cast<uint64_t>(t.rows);
        if (attach && !engine_->AttachParquet(name, version, t.path, t.rows, error)) return false;
        if (identities) identities->push_back(name + "=side:" + QueryResultCache::FileIdentity(t.path));
    }
    return true;
}

void SessionQueryService::SetSideTable(const std::string& name, const std::string& parquet_path, size_t rows) {
    std::lock_guard<std::mutex> lock(side_mutex_);
    side_tables_[name] = SideTable{parquet_path, rows};
}

void SessionQueryService::SetProjectRoot(const std::string& root) {
    cache_.SetFolder(root.empty() ? fs::path() : fs::path(root) / "cache" / "query_results");
}

void SessionQueryService::Interrupt(QueryToken& token) {
    engine_->Interrupt(token);
}

QueryResult SessionQueryService::RunNow(const QueryRequest& request, QueryToken* token) {
    const auto start = std::chrono::steady_clock::now();
    QueryResult r;
    std::string error;
    const bool saved = request.cache && !cache_.Folder().empty();
    if (!saved) {
        if (!AttachInputs(request, &error)) {
            r.error = error;
            return r;
        }
        return engine_->Run(request, token);
    }
    // A saved result first (its key needs the tables' identities, not the tables).
    std::vector<std::string> identities;
    if (!AttachInputs(request, &error, &identities, false)) {
        r.error = error;
        return r;
    }
    const std::string key = QueryResultCache::Key(request, identities);
    if (auto table = cache_.Load(key, &r.truncated)) {
        r.ok = true;
        r.from_cache = true;
        r.table = table;
        r.elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        return r;
    }
    if (!AttachInputs(request, &error)) {
        r.error = error;
        return r;
    }
    const auto stopped = [token] {
        if (!token) return false;
        std::lock_guard<std::mutex> lock(token->mutex);
        return token->stop;
    };
    bool leader = false;
    auto flight = cache_.Join(key, &leader);
    if (!leader) {
        // Someone is running this very query: take their result.
        if (QueryResultCache::Wait(*flight, stopped, &r)) return r;
        if (stopped()) {
            r.cancelled = true;
            r.error = "Cancelled.";
            return r;
        }
        return engine_->Run(request, token);  // theirs failed or stopped: run it here
    }
    r = engine_->Run(request, token);
    if (r.ok) cache_.Save(key, r.table, r.truncated);
    cache_.Finish(key, flight, r);
    return r;
}

uint64_t SessionQueryService::Submit(QueryRequest request, Done done, std::weak_ptr<const void> owner) {
    const std::string label = request.label.empty() ? std::string("Query") : request.label;
    const bool owned = !owner.expired();
    return AsyncTaskManager::Instance().RunAsync(
        label,
        [this, request = std::move(request), done = std::move(done), owner, owned](LambdaTask& task) {
            // Cancel stops this query alone (the token outlives the task: the callback may run late).
            auto token = std::make_shared<QueryToken>();
            task.SetCancellationCallback([this, token] { engine_->Interrupt(*token); });
            QueryResult result;
            if (task.ShouldStop()) {
                result.cancelled = true;
                result.error = "Cancelled.";
            } else {
                task.ReportProgress(0.2f, "Running");
                result = RunNow(request, token.get());
            }
            if (result.cancelled) task.MarkCancelled("Cancelled");
            else if (!result.ok) task.MarkFailed(result.error);
            else task.MarkCompleted(std::to_string(result.table ? result.table->num_rows() : 0) + (result.from_cache ? " rows (saved)" : " rows"));
            const auto deliver = [done, result] {
                if (done) done(result);
            };
            if (owned) AsyncTaskManager::Instance().PostToMainThread(owner, deliver);
            else AsyncTaskManager::Instance().PostToMainThread(deliver);
        },
        nullptr, nullptr, owner);
}

void SessionQueryService::Cancel(uint64_t task_id) {
    AsyncTaskManager::Instance().Cancel(task_id);
}

}  // namespace cyxwiz
