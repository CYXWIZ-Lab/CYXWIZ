#pragma once

// The data steps' result outside the canvas (TOFIX178 S0-f2): checkpoint
// testing, Smoke Run and the Plot node need what the steps before a node
// make, from a background task, without the canvas's executor. The table kept
// on disk is used when it is there (prepared_data_cache); otherwise the steps
// run in a pipeline run of their own - the same engine as Execute Pipeline.

#include "graph_model.h"
#include "materialization_cache.h"
#include "materialization_memory_types.h"
#include "node_executors/pipeline_operator.h"

#include <nlohmann/json.hpp>

#include <functional>
#include <string>
#include <vector>

namespace cyxwiz {

class DataRegistry;
class PipelineExecutor;

// The node wired into the Data Split's Training Dataset input (the Data
// Loader's input without a Split) from its main result (output 0); -1 if none.
int TrainingInputFeeder(const std::vector<gui::MLNode>& nodes, const std::vector<gui::NodeLink>& links);

// The graph as the PipelineExecutor reads it: nodes (type names, parameters)
// and links with pin indices. Plot and Dashboard nodes are left out (they
// show data). False with the reason when a link names a missing node or pin.
bool PipelineJsonFromGraph(const std::vector<gui::MLNode>& nodes,
                           const std::vector<gui::NodeLink>& links,
                           nlohmann::json& pipeline_json,
                           std::string& error);

// Where a run of the steps reads and writes (the project's folders).
struct StepRunRoots {
    std::string project;
    std::string artifacts;
    std::string exports;
    std::string ingestion_cache;
};

struct PreparedStepResult {
    bool ok = false;
    bool cancelled = false;
    std::string dataset;  // the registered result (a table or sparse features)
    std::string error;
    bool from_disk = false;
    bool ran = false;  // the steps ran here
};

// The result of `node` (and every step before it): from disk when kept there,
// else by running those steps. A node that is a Data Input, or a path with a
// step the pipeline run cannot make (images), gives the Data Input's dataset
// unchanged. `keep_owner`, when given, takes over the names the run keeps
// (released with that executor); otherwise they stay registered.
PreparedStepResult PrepareStepResult(const std::vector<gui::MLNode>& nodes,
                             const std::vector<gui::NodeLink>& links,
                             int node,
                             DataRegistry& registry,
                             const StepRunRoots& roots,
                             const MaterializationCacheConfig& cache_config,
                             const std::function<bool()>& cancelled = {},
                             const std::function<void(float, const std::string&)>& progress = {},
                             PipelineExecutor* keep_owner = nullptr);

struct StepMemoryEstimate {
    bool available = false;  // the step published an estimate
    std::string error;       // its settings are invalid
    PipelineOperatorProgress evidence;
};

// The memory estimate of the first step after the Data Input on the way to
// `node`, before anything runs: the step's own estimator on the source
// table, stopped as soon as it publishes (no step is applied). Nothing when
// that step has no estimator (SQL and the other table steps).
StepMemoryEstimate EstimateFirstStepMemory(const std::vector<gui::MLNode>& nodes,
                                           const std::vector<gui::NodeLink>& links,
                                           int node,
                                           const std::shared_ptr<arrow::Table>& source_table,
                                           const MaterializationMemoryContext& memory_context);

}  // namespace cyxwiz
