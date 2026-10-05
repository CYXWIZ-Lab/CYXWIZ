#include "session_query_service.h"

#include "arrow_dataset.h"
#include "async_task_manager.h"
#include "data_registry.h"
#include "dataset_catalog.h"
#include "parquet_backed_dataset.h"

#include <algorithm>
#include <cctype>
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

bool SessionQueryService::AttachInputs(const QueryRequest& request, std::string* error) {
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
                if (!engine_->AttachArrow(view, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(table.get())), table, error)) return false;
            } else {
                auto ds = registry.GetParquetBackedDataset(e.name);
                if (!ds) break;
                const uint64_t version = std::hash<std::string>{}(ds->GetFilePath()) ^ static_cast<uint64_t>(ds->GetNumRows());
                if (!engine_->AttachParquet(view, version, ds->GetFilePath(), static_cast<size_t>(ds->GetNumRows()), error)) return false;
            }
        }
    }
    // Side tables the request names or lists.
    for (const auto& [name, t] : side) {
        const bool listed = std::find(request.inputs.begin(), request.inputs.end(), name) != request.inputs.end();
        if (!listed && !Mentions(sql, Lower(name))) continue;
        const uint64_t version = std::hash<std::string>{}(t.path) ^ static_cast<uint64_t>(t.rows);
        if (!engine_->AttachParquet(name, version, t.path, t.rows, error)) return false;
    }
    return true;
}

void SessionQueryService::SetSideTable(const std::string& name, const std::string& parquet_path, size_t rows) {
    std::lock_guard<std::mutex> lock(side_mutex_);
    side_tables_[name] = SideTable{parquet_path, rows};
}

QueryResult SessionQueryService::RunNow(const QueryRequest& request) {
    std::lock_guard<std::mutex> lock(run_mutex_);
    QueryResult r;
    std::string error;
    if (!AttachInputs(request, &error)) {
        r.error = error;
        return r;
    }
    return engine_->Run(request);
}

uint64_t SessionQueryService::Submit(QueryRequest request, Done done, std::weak_ptr<const void> owner) {
    const std::string label = request.label.empty() ? std::string("Query") : request.label;
    const bool owned = !owner.expired();
    return AsyncTaskManager::Instance().RunAsync(
        label,
        [this, request = std::move(request), done = std::move(done), owner, owned](LambdaTask& task) {
            const uint64_t id = task.GetId();
            // Cancel interrupts this query only while it is the one running.
            task.SetCancellationCallback([this, id] {
                std::lock_guard<std::mutex> lock(tasks_mutex_);
                if (running_task_ == id) engine_->Interrupt();
            });
            task.ReportProgress(0.05f, "Waiting for the query engine");
            QueryResult result;
            {
                std::lock_guard<std::mutex> lock(run_mutex_);
                if (task.ShouldStop()) {
                    result.cancelled = true;
                    result.error = "Cancelled.";
                } else {
                    {
                        std::lock_guard<std::mutex> tl(tasks_mutex_);
                        running_task_ = id;
                    }
                    task.ReportProgress(0.2f, "Attaching tables");
                    std::string error;
                    if (!AttachInputs(request, &error)) {
                        result.error = error;
                    } else {
                        task.ReportProgress(0.4f, "Running");
                        result = engine_->Run(request);
                    }
                    std::lock_guard<std::mutex> tl(tasks_mutex_);
                    running_task_ = 0;
                }
            }
            if (result.cancelled) task.MarkCancelled("Cancelled");
            else if (!result.ok) task.MarkFailed(result.error);
            else task.MarkCompleted(std::to_string(result.table ? result.table->num_rows() : 0) + " rows");
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
