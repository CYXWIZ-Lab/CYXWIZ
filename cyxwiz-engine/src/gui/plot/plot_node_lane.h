#pragma once

// The Plot node's result lane (TOFIX134 P2 step 2.3b, approved boards 4-5):
// gets each Plot node the table at its position. A loaded Data Input is read
// directly (and again when it changes); otherwise only the nodes above the
// plot run, in the background through the task system (Task View shows the
// progress, Cancel stops it), on Refresh. An edit above a plot marks its
// data out of date; nothing re-runs by itself. UI thread only.

#include "../../core/graph_model.h"
#include "../../core/plot/node_result_plan.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace arrow {
class Table;
}

namespace cyxwiz {
class PipelineExecutor;
}

namespace cyxwiz::plot {

class PlotNodeLane {
public:
    struct Status {
        enum class State {
            NotConnected,  // nothing wired into the plot
            Unavailable,   // a node above runs only inside training
            Idle,          // can be read; not read yet (Refresh)
            Running,       // the nodes above are running
            Ready,         // `table` is the data at the plot
            OutOfDate,     // `table` is from before an edit above
            Failed,        // the run failed (`error`)
        };
        State state = State::NotConnected;
        std::string feeder_name;
        std::string reason;            // Unavailable
        int alternative_id = -1;       // Unavailable: the blocking node's input
        std::string alternative_name;
        std::string error;             // Failed
        bool needs_run = false;        // Refresh runs nodes (not a direct read)
        int run_node_count = 0;
        float progress = 0.0f;         // Running
        std::string progress_text;
        std::shared_ptr<arrow::Table> table;
        std::string dataset_name;      // the registry name of `table` (a loaded Data Input or the run's result)
        std::string read_at;           // "16:42" when the table was read
        uint64_t data_version = 0;     // bumps when `table` changes
    };

    // Each frame: finishes runs, re-plans twice a second (out of date,
    // unavailable, a loaded Data Input read again when it changes).
    void Poll(const std::vector<gui::MLNode>& nodes, const std::vector<gui::NodeLink>& links,
              const std::shared_ptr<const void>& owner);
    // Reads or runs now (the Plot window's Refresh, or its first open).
    void Refresh(int plot_id, const std::vector<gui::MLNode>& nodes, const std::vector<gui::NodeLink>& links,
                 const std::shared_ptr<const void>& owner);
    void Cancel(int plot_id);
    void Forget(int plot_id);
    const Status& StatusOf(int plot_id);

private:
    struct Entry {
        Status status;
        NodeResultPlan plan;
        std::string result_fingerprint;  // plan fingerprint the table came from
        std::string run_fingerprint;     // of the running run
        uint64_t task_id = 0;
        std::shared_ptr<PipelineExecutor> executor;
    };
    void Plan(int plot_id, Entry& e, const std::vector<gui::MLNode>& nodes, const std::vector<gui::NodeLink>& links);
    void ReadLoaded(Entry& e);
    void Finish(Entry& e);
    void Start(int plot_id, Entry& e, const std::shared_ptr<const void>& owner);

    std::map<int, Entry> entries_;
    double last_plan_time_ = -1.0;
};

}  // namespace cyxwiz::plot
