#pragma once

// The session query service (TOFIX134 P3 foundation, dashboard_architecture.md
// L2): the one place the Engine's UI (Data Studio Query / Profile /
// Visualize, dashboards) runs SQL. Tables are named as the dataset catalog
// names them ("Spotify", "MNIST"); the ones a query names are attached
// before it runs. Every query runs off the UI thread as a task (Task View
// shows it, Cancel stops it) and its result comes back on the UI thread as
// an Arrow table. One query runs at a time.
//
// Graph execution keeps its own SqlTransform (the SQL step); this service
// is for looking at data, not for the pipeline.

#include "session_query_engine.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
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
    QueryResult RunNow(const QueryRequest& request);

private:
    SessionQueryService();
    // Attaches the catalog tables the request names (or lists); false + error.
    bool AttachInputs(const QueryRequest& request, std::string* error);

    std::unique_ptr<SessionQueryEngine> engine_;
    std::mutex run_mutex_;  // one query at a time
    std::mutex tasks_mutex_;
    uint64_t running_task_ = 0;
};

}  // namespace cyxwiz
