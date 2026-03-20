#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>

namespace cyxwiz {

// Forward declaration
class DuckDBConnector;

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
     * Register a progress callback
     * Called periodically during execution with progress updates
     */
    void SetProgressCallback(std::function<void(float)> callback);

    /**
     * Register a completion callback
     * Called when pipeline execution finishes (success or failure)
     */
    void SetCompletionCallback(std::function<void(bool)> callback);

private:
    struct Node {
        int id;
        std::string type;
        std::string name;
        std::map<std::string, std::string> parameters;
        std::vector<int> inputs;   // Input node IDs
        std::vector<int> outputs;  // Output node IDs
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

    // Deployment state (Phase 5 Week 7)
    bool deployment_ready_;
    std::string deployment_dataset_;

    // Callbacks
    std::function<void(float)> progress_callback_;
    std::function<void(bool)> completion_callback_;

    // DuckDB connector for SQL transformations
    std::unique_ptr<DuckDBConnector> duckdb_;

    // Pipeline execution steps
    bool ParsePipeline(const std::string& pipeline_json, std::vector<Node>& nodes);
    bool ValidatePipeline(const std::vector<Node>& nodes);
    std::vector<int> TopologicalSort(const std::vector<Node>& nodes);
    bool ExecuteNode(const Node& node, ExecutionContext& ctx);

    // Node type executors
    bool ExecuteFileInput(const Node& node, ExecutionContext& ctx);
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
    bool ExecutePCA(const Node& node, ExecutionContext& ctx);
    bool ExecutePolynomialFeatures(const Node& node, ExecutionContext& ctx);
    bool ExecuteBinning(const Node& node, ExecutionContext& ctx);

    // Helper methods
    void UpdateProgress(float progress);
    void ReportError(const std::string& error);
    void NotifyCompletion(bool success);
    std::string GetInputDatasetName(const Node& node, ExecutionContext& ctx);
};

} // namespace cyxwiz
