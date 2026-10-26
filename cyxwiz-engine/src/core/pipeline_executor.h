#pragma once

#include <mutex>
#include <filesystem>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <atomic>
#include <set>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

#include "operator_figures.h"
#include "pipeline_node_run_facts.h"

namespace gui {
enum class NodeType;
}

namespace arrow {
class Schema;
class Table;
}

namespace cyxwiz {

// Forward declaration
class SqlTransform;
struct SqlTransformInput;
class IPipelineOperator;
struct PipelineRuntimeSupport;
struct TabularExport;

enum class PipelineNodeExecutionEvent : uint8_t {
    Pending,
    Executing,
    Completed,
    Error
};

/**
 * PipelineExecutor - Executes data transformation pipelines
 *
 * Phase 1, Week 1: Core Infrastructure
 *
 * This component takes a pipeline graph from the Node Editor (Data Pipeline mode) and executes
 * the data transformations using DuckDB and Arrow. It handles:
 * - Topological sorting of pipeline nodes
 * - Data flow between nodes
 * - Execution monitoring and progress reporting
 * - Error handling and recovery
 *
 * Architecture:
 *   Node Editor (Graph) -> PipelineExecutor (Engine) -> DuckDB/Arrow (Backend)
 *
 * Execution Model:
 *   1. Validate pipeline (no cycles, all nodes connected)
 *   2. Topologically sort nodes (execution order)
 *   3. For each node in order:
 *      - Fetch input data from upstream nodes
 *      - Apply transformation (SQL query or custom operator)
 *      - Store output for downstream nodes
 *   4. Return final result dataset
 */
class PipelineExecutor {
public:
    PipelineExecutor();
    ~PipelineExecutor();

    /**
     * Execute a pipeline from JSON representation
     * @param pipeline_json Serialized pipeline from the Node Editor
     * @return true if execution succeeded
     */
    bool ExecutePipeline(const std::string& pipeline_json);

    /**
     * Set the project-owned root used for automatically generated pipeline
     * artifacts. Empty falls back to ./artifacts for headless callers.
     */
    void SetArtifactRoot(std::string artifact_root) {
        artifact_root_ = std::move(artifact_root);
    }

    /**
     * Set the project-owned root used when an export node has no explicit
     * output path. Empty uses a process-independent temporary export root.
     */
    void SetExportRoot(std::string export_root) {
        export_root_ = std::move(export_root);
    }

    /**
     * Set the project-owned persistent CSV ingestion cache directory.
     * Empty uses DataRegistry's process-independent temporary fallback.
     */
    void SetIngestionCacheRoot(std::string ingestion_cache_root) {
        ingestion_cache_root_ = std::move(ingestion_cache_root);
    }

    /**
     * Project root for source paths: a relative Data Input / File Input
     * path is resolved against it (see ResolveProjectDataPath). Empty keeps
     * the previous behaviour (relative to the working directory).
     */
    void SetProjectRoot(std::string project_root) {
        project_root_ = std::move(project_root);
    }

    /**
     * Stop the currently running pipeline
     */
    void StopExecution();

    /**
     * Check if a pipeline is currently executing
     */
    bool IsExecuting() const { return executing_.load(); }

    // What this executor's runs registered for these nodes - their results,
    // extra outputs, and the in-memory copies of inputs they read - is
    // unregistered when the nodes leave the graph (TOFIX160 P2). A name a
    // kept node still uses stays, and so does a Save / Deploy / Data Output
    // result (the user named it). Not while executing (the caller waits).
    // `keep`: names another executor now owns too (the same node ids).
    void ReleaseNodes(const std::set<int>& node_ids, const std::set<std::string>& keep = {});
    // Everything its runs registered (the graph was cleared or replaced).
    void ReleaseAll(const std::set<std::string>& keep = {});
    // The names its runs registered and would release (results of nodes that
    // are not Save / Deploy / Data Output, and in-memory input copies).
    std::set<std::string> OwnedNames();
    // Only the in-memory copies of disk inputs (a one-shot run whose result
    // has been read).
    void ReleaseAdmittedInputs();
    // Takes over what `from`'s runs registered (its owned names, kept nodes
    // and in-memory input copies), so this executor releases them later.
    // A one-shot run (a plot's data) hands its names to one that outlives it.
    void AdoptOwnership(PipelineExecutor& from);

    /**
     * Get execution progress (0.0 to 1.0)
     */
    float GetProgress() const { return progress_.load(); }

    /**
     * Get the last error message
     */
    const std::string& GetLastError() const { return last_error_; }

    /**
     * Get deployment status (Phase 5 Week 7)
     */
    bool IsDeploymentReady() const { return deployment_ready_.load(); }
    const std::string& GetDeploymentDataset() const { return deployment_dataset_; }

    /**
     * Clear deployment status (called after deployment is complete)
     */
    void ClearDeploymentStatus() {
        deployment_ready_ = false;
        deployment_dataset_.clear();
    }

    /**
     * Register a progress callback (Phase 8: Enhanced with status messages)
     * Called periodically during execution with progress updates
     */
    void SetProgressCallback(std::function<void(float, const std::string&)> callback);

    /** Register typed per-node lifecycle events for canvas visualization. */
    void SetNodeExecutionCallback(
        std::function<void(int, PipelineNodeExecutionEvent, const std::string&)> callback);

    /** Figures a step reports about its own run (Word Embeddings epochs), with its node. */
    void SetOperatorFiguresCallback(std::function<void(const OperatorFigures&)> callback);

    /** Facts about each node's latest run (thread-safe; readable during a run). */
    std::map<int, PipelineNodeRunFacts> GetNodeRunFacts() const;

    /**
     * Register a completion callback
     * Called when pipeline execution finishes (success or failure)
     */
    void SetCompletionCallback(std::function<void(bool)> callback);

    /**
     * Request cancellation of the current pipeline execution (Phase 8)
     */
    void RequestCancel();

    /**
     * Check if execution was cancelled (Phase 8)
     */
    bool IsCancelled() const { return cancel_requested_; }

    // Node id -> DataRegistry dataset of the last successful run (the Plot
    // node's result lane reads the table of the node wired into it).
    const std::map<int, std::string>& NodeResults() const { return last_node_results_; }
    // (node ID, output pin >= 1) -> Arrow table name of the last run, for
    // operators with more than one table output.
    const std::map<std::pair<int, int>, std::string>& NodePinResults() const { return last_node_pin_results_; }
    // Node ID -> fitted model file of the last run (nodes with a Model output).
    const std::map<int, std::string>& ModelArtifacts() const { return last_model_artifacts_; }

    // The table node `node_id` left in this executor's runs, when the node and
    // everything upstream of it is unchanged in `pipeline_json` and the table is
    // still loaded; nothing while a run executes or when it would run again.
    // Runs nothing (TOFIX178 S0-d: training uses the canvas run's result).
    std::optional<std::string> CurrentNodeResult(const std::string& pipeline_json, int node_id);

private:
    struct InputLink {
        int node_id = -1;
        std::optional<int> source_pin_index;
        std::optional<int> target_pin_index;
    };

    struct Node {
        int id;
        std::string type;
        std::optional<gui::NodeType> runtime_type;
        std::string name;
        std::map<std::string, std::string> parameters;
        std::vector<int> inputs;   // Input node IDs
        std::vector<int> outputs;  // Output node IDs
        std::vector<InputLink> input_links; // Typed artifact/table routing
        std::string recipe;        // the Preparation Recipe the step came from ("" outside one)

        // Phase 8: Lazy evaluation support
        bool needs_execution = true;        // Dirty flag for change tracking
        uint64_t last_execution_hash = 0;   // Hash of parameters for change detection
        std::string cached_output_dataset;  // Result from last successful execution
    };

    // What a node left in its latest successful run on this executor, so an
    // unchanged node is not run again (lazy evaluation, TOFIX154 item 10).
    // Nodes that write files or prepare a deployment always run.
    struct NodeRunCache {
        uint64_t hash = 0;
        std::string output;                     // node_results entry
        std::map<int, std::string> pin_outputs; // output pin >= 1 -> dataset
        std::string model_artifact;
        std::string outcome;
        // What the hash covered, so a dirty node can say which key changed.
        std::map<std::string, std::string> parameters;
    };
    std::map<int, NodeRunCache> run_cache_;
    // Restores each unchanged node's hash and output from run_cache_ (when
    // its output still exists), so MarkDirtyNodes can skip it.
    void RestoreRunCache(std::vector<Node>& nodes);

    // The names each node's completed runs registered (results and extra
    // outputs), for ReleaseNodes; kept_nodes_ are Save / Deploy / Data Output
    // nodes, whose results stay. Under owned_mutex_ (nodes complete on worker
    // threads).
    std::mutex owned_mutex_;
    std::map<int, std::set<std::string>> owned_names_;
    std::set<int> kept_nodes_;
    std::map<int, std::string> last_node_results_;
    std::map<std::pair<int, int>, std::string> last_node_pin_results_;
    std::map<int, std::string> last_model_artifacts_;

    struct ExecutionContext {
        std::map<int, std::string> node_results;  // Node ID -> Arrow table name
        // (node ID, output pin >= 1) -> Arrow table name (extra operator outputs)
        std::map<std::pair<int, int>, std::string> node_pin_results;
        std::map<int, std::string> model_artifacts; // Node ID -> fitted model path
        std::map<int, std::string> node_outcomes;   // Node ID -> operator outcome summary
        std::map<int, std::string> node_written;    // Node ID -> file an export node wrote
        std::string input_dataset;                // Initial dataset name
        std::string output_dataset;               // Final result dataset name
        std::string deployment_dataset;           // Dataset ready for Node Editor deployment
        bool deployment_ready = false;            // Deployment flag
    };

    // Execution state
    std::atomic<bool> executing_;
    std::atomic<float> progress_;
    std::string last_error_;
    bool stop_requested_;
    std::atomic<bool> cancel_requested_;  // Phase 8: Atomic cancellation flag
    std::string current_status_;          // Phase 8: Current execution status message
    std::mutex status_mutex_;             // current_status_: nodes report progress from their own threads
    std::string artifact_root_;           // Project artifacts directory
    std::string project_root_;            // Project root for relative source paths
    std::string graph_path_;              // the graph file the run came from (lineage)
    // In-memory copies of disk inputs (AdmitInputTable), kept until the next run.
    std::vector<std::string> admitted_inputs_;
    const std::vector<Node>* run_nodes_ = nullptr;  // the run's nodes while ExecuteParallel runs (lineage)
    // Writes <output>.lineage.json for an export node that wrote `output_path`
    // (TOFIX148 Publish); a failure is logged, the export stands.
    void WriteExportLineage(const Node& node, const ExecutionContext& ctx, const TabularExport& written,
                            const std::string& output_path, const std::string& format);
    std::string export_root_;             // Project exports directory
    std::string ingestion_cache_root_;    // Project cache/ingestion directory

    // Deployment state (Phase 5 Week 7)
    std::atomic<bool> deployment_ready_;
    std::string deployment_dataset_;

    // Callbacks
    std::function<void(float, const std::string&)> progress_callback_;  // Phase 8: Enhanced with status message
    std::function<void(bool)> completion_callback_;
    std::function<void(int, PipelineNodeExecutionEvent, const std::string&)>
        node_execution_callback_;
    std::function<void(const OperatorFigures&)> operator_figures_callback_;

    // Latest run facts per node (GetNodeRunFacts); start times of running nodes.
    mutable std::mutex node_run_facts_mutex_;
    std::map<int, PipelineNodeRunFacts> node_run_facts_;
    std::map<int, double> node_started_seconds_;
    void RecordNodeRunFacts(const Node& node, PipelineNodeExecutionEvent event,
                            const std::string& output_dataset, const std::string& outcome,
                            const std::string& error,
                            const std::string& written_path = {});

    // SQL steps running now (sibling branches run at once), so RequestCancel
    // can interrupt each DuckDB.
    std::mutex active_sql_mutex_;
    std::set<SqlTransform*> active_sql_;
    void StartSql(SqlTransform& transform);
    void EndSql(SqlTransform& transform);

    // The error of the node this thread runs: a node's task writes its own
    // (TOFIX157 A2) and the scheduler publishes the failing one as last_error_;
    // outside a task it is last_error_.
    std::string& Error();

    // Pipeline execution steps
    bool ParsePipeline(const std::string& pipeline_json, std::vector<Node>& nodes);
    bool ResolveAutomaticPreprocessingStatePaths(std::vector<Node>& nodes);
    bool ValidatePipeline(const std::vector<Node>& nodes);
    std::vector<int> TopologicalSort(const std::vector<Node>& nodes);
    bool ExecuteNode(const Node& node, ExecutionContext& ctx);
    PipelineRuntimeSupport ResolveNodeRuntimeSupport(const Node& node) const;
    bool ExecuteTypedLegacyNode(const Node& node,
                                ExecutionContext& ctx,
                                bool& handled);

    // Node type executors
    bool ExecuteFileInput(const Node& node, ExecutionContext& ctx);
    bool ExecuteDataInput(const Node& node, ExecutionContext& ctx);   // Smart universal data input
    bool ExecuteDataOutput(const Node& node, ExecutionContext& ctx);  // Smart universal data output
    bool ExecuteDataConvert(const Node& node, ExecutionContext& ctx); // Smart file conversion utility
    bool ExecuteFilterRows(const Node& node, ExecutionContext& ctx);
    bool ExecuteSqlTransform(const Node& node, ExecutionContext& ctx);
    bool ExecuteSelectColumns(const Node& node, ExecutionContext& ctx);
    bool ExecuteRemoveDuplicates(const Node& node, ExecutionContext& ctx);
    bool ExecuteSaveDataset(const Node& node, ExecutionContext& ctx);

    // Phase 2 Week 4 - Additional tabular transformations
    bool ExecuteFillMissing(const Node& node, ExecutionContext& ctx);
    bool ExecuteSortRows(const Node& node, ExecutionContext& ctx);
    bool ExecuteJoin(const Node& node, ExecutionContext& ctx);
    bool ExecuteGroupBy(const Node& node, ExecutionContext& ctx);

    // Phase 5 Week 7 - Node Editor Handoff
    bool ExecuteDeployToNodeEditor(const Node& node, ExecutionContext& ctx);

    // Phase 6 Week 8-9 - Text Processing Nodes
    bool ExecuteTextClean(const Node& node, ExecutionContext& ctx);
    bool ExecuteTextTokenize(const Node& node, ExecutionContext& ctx);
    bool ExecuteTextVectorize(const Node& node, ExecutionContext& ctx);
    bool ExecuteNERSequenceBuilder(const Node& node, ExecutionContext& ctx);
    bool ExecuteSequenceVocabulary(const Node& node, ExecutionContext& ctx);

    // Phase 6 Week 8-9 - Time-Series Nodes
    bool ExecuteTSWindow(const Node& node, ExecutionContext& ctx);
    bool ExecuteTSFeatures(const Node& node, ExecutionContext& ctx);
    bool ExecuteTSLag(const Node& node, ExecutionContext& ctx);
    bool ExecuteTSDiff(const Node& node, ExecutionContext& ctx);

    // Phase 6 Week 8-9 - Feature Engineering Nodes
    bool ExecutePolynomialFeatures(const Node& node, ExecutionContext& ctx);
    bool ExecuteBinning(const Node& node, ExecutionContext& ctx);

    // KNIME-Style Table Manipulation Nodes
    bool ExecuteRowToColumnNames(const Node& node, ExecutionContext& ctx);
    bool ExecuteTableCropper(const Node& node, ExecutionContext& ctx);
    bool ExecuteStringManipulation(const Node& node, ExecutionContext& ctx);
    bool ExecuteMathFormula(const Node& node, ExecutionContext& ctx);
    bool ExecuteRenameColumns(const Node& node, ExecutionContext& ctx);
    bool ExecuteCellExtractor(const Node& node, ExecutionContext& ctx);
    bool ExecuteCellUpdater(const Node& node, ExecutionContext& ctx);
    bool ExecuteRowAppender(const Node& node, ExecutionContext& ctx);
    bool ExecuteColumnAppender(const Node& node, ExecutionContext& ctx);
    bool ExecuteUnpivot(const Node& node, ExecutionContext& ctx);
    bool ExecuteExportCSV(const Node& node, ExecutionContext& ctx);
    bool ExecuteExportJSON(const Node& node, ExecutionContext& ctx);
    bool ExecuteExportParquet(const Node& node, ExecutionContext& ctx);
    bool ExecuteRuleEngine(const Node& node, ExecutionContext& ctx);
    bool ExecuteUnitConverter(const Node& node, ExecutionContext& ctx);
    bool ExecuteCalculatorNode(const Node& node, ExecutionContext& ctx);
    bool ExecuteJSONPathExtractor(const Node& node, ExecutionContext& ctx);
    bool ExecuteRegexTester(const Node& node, ExecutionContext& ctx);
    bool ExecuteDataProfiler(const Node& node, ExecutionContext& ctx);
    bool ExecuteRegressionMetrics(const Node& node, ExecutionContext& ctx);
    bool ExecuteClassificationMetrics(const Node& node, ExecutionContext& ctx);
    // A test's per-row table to Results > Test's figures (TOFIX157):
    // Metrics (pin 0), Confusion, Per class, Worst rows (pins 1-3).
    bool ExecuteTestEvaluation(const Node& node, ExecutionContext& ctx);
    bool ExecuteConfusionMatrix(const Node& node, ExecutionContext& ctx);
    bool ExecuteROCCurve(const Node& node, ExecutionContext& ctx);
    bool ExecutePRCurve(const Node& node, ExecutionContext& ctx);
    bool ExecuteDataValidator(const Node& node, ExecutionContext& ctx);
    bool ExecuteSampleRows(const Node& node, ExecutionContext& ctx);
    bool ExecuteValueCounts(const Node& node, ExecutionContext& ctx);
    bool ExecuteDescribeStats(const Node& node, ExecutionContext& ctx);
    bool ExecuteCorrelationMatrix(const Node& node, ExecutionContext& ctx);
    // Helper methods
    void UpdateProgress(float progress, const std::string& status = "");  // Phase 8: Added status parameter
    void ReportError(const std::string& error);
    void NotifyNodeExecution(const Node& node,
                             PipelineNodeExecutionEvent event,
                             const std::string& status = "");
    void NotifyCompletion(bool success);
    // The table a link from (node_id, source_pin) carries: the pin's own table
    // for output pins >= 1 when the node published one, else the node's result.
    static const std::string* ResultForOutput(const ExecutionContext& ctx, int node_id,
                                              std::optional<int> source_pin);
    std::string GetInputDatasetName(const Node& node, ExecutionContext& ctx);
    std::vector<std::string> GetInputDatasetNames(const Node& node,
                                                  ExecutionContext& ctx,
                                                  size_t expected_count);
    std::string GetInputDatasetForPin(const Node& node,
                                      ExecutionContext& ctx,
                                      int target_pin_index);
    std::string GetInputModelArtifactForPin(const Node& node,
                                            ExecutionContext& ctx,
                                            int target_pin_index);
    bool FailUnsupportedNode(const Node& node, const std::string& reason);
    bool ExecutePipelineOperatorNode(const Node& node, ExecutionContext& ctx, gui::NodeType type);

    // Phase 7: Improved error messages with suggestions
    std::string GetImprovedErrorMessage(const std::string& node_type, const std::string& error_category, const std::string& details = "");

    // Phase 8: Performance optimization helpers
    uint64_t ComputeNodeHash(const Node& node) const;
    void MarkDirtyNodes(std::vector<Node>& nodes);
    std::vector<int> FindReadyNodes(const std::vector<Node>& nodes, const std::set<int>& completed) const;
    bool ExecuteParallel(std::vector<Node>& nodes);
    // The input table a step reads in memory: name itself, or an in-memory
    // copy of a disk table that fits the data budget; "" after ReportError.
    std::string AdmitInputTable(const Node& node, const std::string& name);
    // A step without its input table: keeps the lookup's own reason.
    void ReportMissingInput(const std::string& step);

    // SQL over files (TOFIX154 step 3). A step's inputs are Parquet files
    // DuckDB reads in place: a disk table's own file, or a memory table
    // written once (StepInputFile). The result comes back as a table, or,
    // when an input is on disk or large, DuckDB writes it to the pipeline
    // cache; a result above the data budget's threshold stays there on disk.
    // A check's output is its input (the same table or file) under output_name.
    bool PassInputThrough(const Node& node, const std::string& input_dataset, const std::string& output_name);
    struct StepSqlInput {
        std::string alias;
        std::string dataset;
    };
    // `exact_types`: the SQL step's contract 1; the pipeline's own SQL
    // (FilterRows, GroupBy, ...) takes any type DuckDB reads. Errors are
    // reported as error_prefix + DuckDB's message.
    bool RunStepSql(const Node& node, const std::vector<StepSqlInput>& inputs, const std::string& sql,
                    const std::string& output_name, bool exact_types, const std::string& error_prefix);
    // A result file a step wrote: read back into memory up to the threshold,
    // else registered on disk (D1); false after ReportError.
    bool RegisterStepResultFile(const Node& node, const std::string& output_name, const std::string& path,
                                int64_t rows);
    // A result file's path for this run (cache/pipeline/results).
    std::string NewStepResultPath(const std::string& output_name) const;
    // An operator that streams (TOFIX154 step 4) run over its input one row
    // group at a time: fit pass, apply pass, the output written as it comes.
    bool RunStreamingOperator(const Node& node, IPipelineOperator& op, const std::string& input_dataset,
                              const std::string& output_name);
    // A step's input for checking columns and types: the table in memory, or
    // an empty table with the file's schema (its rows stay on disk).
    std::shared_ptr<arrow::Table> InputSchemaTable(const std::string& dataset) const;
    // A dataset's rows wherever it lives (0 when it is not a table).
    int64_t DatasetRows(const std::string& dataset) const;
    // A small query's result as a table (a fit step's statistics), over the
    // same files; null after ReportError.
    std::shared_ptr<arrow::Table> FetchStepSql(const std::vector<StepSqlInput>& inputs, const std::string& sql,
                                               const std::string& error_prefix);
    // The SqlTransform inputs for a step: one file per input; false after
    // ReportError.
    bool PrepareStepSqlInputs(const std::vector<StepSqlInput>& inputs, const std::string& sql,
                              std::vector<SqlTransformInput>& files, uint64_t& input_bytes, bool& any_on_disk);
    // The Parquet file a dataset is read from ("" after ReportError).
    std::string StepInputFile(const std::string& dataset, std::shared_ptr<arrow::Schema>& schema,
                              uint64_t& bytes, bool& on_disk);
    // Steps that read their inputs from the files, never whole: SQL steps
    // (RunStepSql) and operators that stream (RunStreamingOperator). They are
    // not admitted first.
    static bool StepReadsInputFiles(const Node& node);
    // <project>/cache/pipeline, or <temp>/cyxwiz/pipeline outside a project.
    std::filesystem::path PipelineCacheRoot() const;
    // Parquet copies of memory tables, kept while their table lives.
    struct StepInputCopy {
        std::weak_ptr<arrow::Table> table;
        std::string path;
    };
    std::map<std::string, StepInputCopy> step_input_copies_;
    // Result files this executor wrote (output dataset -> file); one name per
    // run, since Windows cannot replace a file that is still mapped.
    std::map<std::string, std::string> step_result_files_;
    // Sibling branches run at once (TOFIX157 A2): step_input_copies_,
    // step_result_files_ and admitted_inputs_ change under step_files_mutex_,
    // and a table's copy is made under its own name lock, so two steps reading
    // one table write (or read in) one copy, the second waiting for the first.
    std::mutex step_files_mutex_;
    std::map<std::string, std::shared_ptr<std::mutex>> name_locks_;
    std::shared_ptr<std::mutex> NameLock(const std::string& key);
    // Held while a node writes an export file: two branches exporting to one
    // file write it one after the other (none for an empty path).
    std::unique_lock<std::mutex> LockExportPath(const std::string& path);
    // Removes the copies whose table is gone and the result files no dataset
    // uses any more (at run start and when the executor goes).
    void RemoveUnusedStepFiles();
    Node* FindNodeById(std::vector<Node>& nodes, int node_id);
    const Node* FindNodeById(const std::vector<Node>& nodes, int node_id) const;

    // Phase 8: Memory optimization (future enhancement)
    // These settings control streaming behavior for large datasets
    bool streaming_mode_ = false;        // Enable chunk-based processing
    int64_t chunk_size_ = 100000;        // Rows per chunk for streaming
    int64_t memory_limit_ = 2147483648;  // 2GB max memory per operation
};

} // namespace cyxwiz
