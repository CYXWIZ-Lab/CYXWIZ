#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>

namespace cyxwiz {

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
    ~PipelineExecutor() = default;

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
    };

    // Execution state
    bool executing_;
    float progress_;
    std::string last_error_;
    bool stop_requested_;

    // Callbacks
    std::function<void(float)> progress_callback_;
    std::function<void(bool)> completion_callback_;

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

    // Helper methods
    void UpdateProgress(float progress);
    void ReportError(const std::string& error);
    void NotifyCompletion(bool success);
};

} // namespace cyxwiz
