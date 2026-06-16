#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <atomic>
#include <set>
#include <cstddef>
#include <optional>

namespace gui {
enum class NodeType;
}

namespace cyxwiz {

// Forward declaration
class DuckDBConnector;
enum class PipelineLegacyDispatchKind;
struct PipelineRuntimeSupport;

/**
 * PipelineExecutor - Executes data transformation pipelines
 *
 * Phase 1, Week 1: Core Infrastructure
 *
 * This component takes a pipeline graph from PipelineCanvas and executes
 * the data transformations using DuckDB and Arrow. It handles:
 * - Topological sorting of pipeline nodes
 * - Data flow between nodes
 * - Execution monitoring and progress reporting
 * - Error handling and recovery
 *
 * Architecture:
 *   PipelineCanvas (Graph) -> PipelineExecutor (Engine) -> DuckDB/Arrow (Backend)
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
     * @param pipeline_json Serialized pipeline from PipelineCanvas
     * @return true if execution succeeded
     */
    bool ExecutePipeline(const std::string& pipeline_json);

    /**
     * Stop the currently running pipeline
     */
    void StopExecution();

    /**
     * Check if a pipeline is currently executing
     */
    bool IsExecuting() const { return executing_; }

    /**
     * Get execution progress (0.0 to 1.0)
     */
    float GetProgress() const { return progress_; }

    /**
     * Get the last error message
     */
    const std::string& GetLastError() const { return last_error_; }

    /**
     * Get deployment status (Phase 5 Week 7)
     */
    bool IsDeploymentReady() const { return deployment_ready_; }
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

private:
    struct Node {
        int id;
        std::string type;
        std::optional<gui::NodeType> runtime_type;
        std::string name;
        std::map<std::string, std::string> parameters;
        std::vector<int> inputs;   // Input node IDs
        std::vector<int> outputs;  // Output node IDs

        // Phase 8: Lazy evaluation support
        bool needs_execution = true;        // Dirty flag for change tracking
        uint64_t last_execution_hash = 0;   // Hash of parameters for change detection
        std::string cached_output_dataset;  // Result from last successful execution
    };

    struct ExecutionContext {
        std::map<int, std::string> node_results;  // Node ID -> Arrow table name
        std::string input_dataset;                // Initial dataset name
        std::string output_dataset;               // Final result dataset name
        std::string deployment_dataset;           // Dataset ready for Node Editor deployment
        bool deployment_ready = false;            // Deployment flag
    };

    // Execution state
    bool executing_;
    float progress_;
    std::string last_error_;
    bool stop_requested_;
    std::atomic<bool> cancel_requested_;  // Phase 8: Atomic cancellation flag
    std::string current_status_;          // Phase 8: Current execution status message

    // Deployment state (Phase 5 Week 7)
    bool deployment_ready_;
    std::string deployment_dataset_;

    // Callbacks
    std::function<void(float, const std::string&)> progress_callback_;  // Phase 8: Enhanced with status message
    std::function<void(bool)> completion_callback_;

    // DuckDB connector for SQL transformations
    std::unique_ptr<DuckDBConnector> duckdb_;

    // Pipeline execution steps
    bool ParsePipeline(const std::string& pipeline_json, std::vector<Node>& nodes);
    bool ValidatePipeline(const std::vector<Node>& nodes);
    std::vector<int> TopologicalSort(const std::vector<Node>& nodes);
    bool ExecuteNode(const Node& node, ExecutionContext& ctx);
    PipelineRuntimeSupport ResolveNodeRuntimeSupport(const Node& node) const;
    bool ExecuteTypedLegacyNode(const Node& node,
                                ExecutionContext& ctx,
                                bool& handled);
    bool ExecuteLegacyDispatchKind(const Node& node,
                                   ExecutionContext& ctx,
                                   PipelineLegacyDispatchKind dispatch_kind);

    // Node type executors
    bool ExecuteFileInput(const Node& node, ExecutionContext& ctx);
    bool ExecuteDataInput(const Node& node, ExecutionContext& ctx);   // Smart universal data input
    bool ExecuteDataOutput(const Node& node, ExecutionContext& ctx);  // Smart universal data output
    bool ExecuteDataConvert(const Node& node, ExecutionContext& ctx); // Smart file conversion utility
    bool ExecuteFilterRows(const Node& node, ExecutionContext& ctx);
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
    // Helper methods
    void UpdateProgress(float progress, const std::string& status = "");  // Phase 8: Added status parameter
    void ReportError(const std::string& error);
    void NotifyCompletion(bool success);
    std::string GetInputDatasetName(const Node& node, ExecutionContext& ctx);
    std::vector<std::string> GetInputDatasetNames(const Node& node,
                                                  ExecutionContext& ctx,
                                                  size_t expected_count);
    bool FailUnsupportedNode(const Node& node, const std::string& reason);
    bool ExecutePipelineOperatorNode(const Node& node, ExecutionContext& ctx, gui::NodeType type);

    // Phase 7: Improved error messages with suggestions
    std::string GetImprovedErrorMessage(const std::string& node_type, const std::string& error_category, const std::string& details = "");

    // Phase 8: Performance optimization helpers
    uint64_t ComputeNodeHash(const Node& node) const;
    void MarkDirtyNodes(std::vector<Node>& nodes);
    std::vector<int> FindReadyNodes(const std::vector<Node>& nodes, const std::set<int>& completed) const;
    bool ExecuteParallel(std::vector<Node>& nodes);
    Node* FindNodeById(std::vector<Node>& nodes, int node_id);
    const Node* FindNodeById(const std::vector<Node>& nodes, int node_id) const;

    // Phase 8: Memory optimization (future enhancement)
    // These settings control streaming behavior for large datasets
    bool streaming_mode_ = false;        // Enable chunk-based processing
    int64_t chunk_size_ = 100000;        // Rows per chunk for streaming
    int64_t memory_limit_ = 2147483648;  // 2GB max memory per operation
};

} // namespace cyxwiz
