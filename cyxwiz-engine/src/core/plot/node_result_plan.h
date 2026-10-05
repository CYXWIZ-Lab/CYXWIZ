#pragma once

// What a Plot node needs to show its data (TOFIX134 P2, the node result
// lane): the table at the node's position. Pure: decided from a frozen copy
// of the graph, no registry or executor calls (the caller says which
// datasets are loaded).

#include "../graph_model.h"

#include <functional>
#include <string>
#include <vector>

namespace cyxwiz::plot {

struct NodeResultPlan {
    enum class State {
        NotConnected,  // nothing wired into the plot
        Loaded,        // a loaded Data Input: read dataset_name directly
        Run,           // run pipeline_json (only the nodes above the plot)
        Unavailable,   // a node above cannot run outside training (yet)
    };
    State state = State::NotConnected;
    int feeder_id = -1;          // the node wired into the plot
    std::string feeder_name;
    int feeder_pin = -1;         // which of its outputs (index)
    // The output is a fitted model (a trainer's Model pin, TOFIX134 P4.7):
    // the run's model file is read as rows (plot_tree_model), not a table.
    bool model = false;
    // The feeder is a Count / TF-IDF Vectorizer with sparse output (TOFIX134
    // P3, board 19): the Data Studio run cannot make it, so the lane loads the
    // Data Input above (`input_pipeline_json`, or `dataset_name` when loaded)
    // and runs `closure_ids` through the materializer, as training does.
    bool sparse = false;
    int source_input_id = -1;
    std::string input_pipeline_json;
    std::vector<int> closure_ids;
    std::string dataset_name;    // Loaded
    std::string pipeline_json;   // Run: PipelineExecutor JSON of the closure
    int run_node_count = 0;      // Run: nodes it runs
    std::string reason;          // Unavailable: why, in words
    int alternative_id = -1;     // Unavailable: the blocking node's input
    std::string alternative_name;
    // Changes when any node above the plot (type, name, parameters) or a
    // link between them changes: a result made before is out of date.
    std::string fingerprint;
};

// `dataset_loaded(name)` says whether a dataset is in memory (Arrow).
NodeResultPlan PlanNodeResult(int plot_node_id, const std::vector<gui::MLNode>& nodes,
                              const std::vector<gui::NodeLink>& links,
                              const std::function<bool(const std::string&)>& dataset_loaded);

}  // namespace cyxwiz::plot
