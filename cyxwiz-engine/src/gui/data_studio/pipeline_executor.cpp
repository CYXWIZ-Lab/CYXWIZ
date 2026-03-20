#include "pipeline_executor.h"
#include "../../core/duckdb_connector.h"
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <queue>

namespace cyxwiz {

PipelineExecutor::PipelineExecutor()
    : executing_(false)
    , progress_(0.0f)
    , stop_requested_(false)
{
    spdlog::info("[Data Studio] PipelineExecutor initialized");
}

bool PipelineExecutor::ExecutePipeline(const std::string& pipeline_json) {
    if (executing_) {
        last_error_ = "Pipeline is already executing";
        return false;
    }

    executing_ = true;
    progress_ = 0.0f;
    stop_requested_ = false;
    last_error_ = "";

    spdlog::info("[Data Studio] Starting pipeline execution");

    // Parse pipeline
    std::vector<Node> nodes;
    if (!ParsePipeline(pipeline_json, nodes)) {
        ReportError("Failed to parse pipeline");
        executing_ = false;
        NotifyCompletion(false);
        return false;
    }

    UpdateProgress(0.1f);

    // Validate pipeline
    if (!ValidatePipeline(nodes)) {
        ReportError("Pipeline validation failed");
        executing_ = false;
        NotifyCompletion(false);
        return false;
    }

    UpdateProgress(0.2f);

    // Topological sort
    auto execution_order = TopologicalSort(nodes);
    if (execution_order.empty()) {
        ReportError("Failed to determine execution order (pipeline may have cycles)");
        executing_ = false;
        NotifyCompletion(false);
        return false;
    }

    UpdateProgress(0.3f);

    // Execute nodes in order
    ExecutionContext ctx;
    float progress_per_node = 0.7f / execution_order.size();

    for (size_t i = 0; i < execution_order.size(); i++) {
        if (stop_requested_) {
            ReportError("Execution stopped by user");
            executing_ = false;
            NotifyCompletion(false);
            return false;
        }

        int node_id = execution_order[i];
        auto it = std::find_if(nodes.begin(), nodes.end(),
                              [node_id](const Node& n) { return n.id == node_id; });

        if (it == nodes.end()) {
            ReportError("Node not found in pipeline");
            executing_ = false;
            NotifyCompletion(false);
            return false;
        }

        if (!ExecuteNode(*it, ctx)) {
            executing_ = false;
            NotifyCompletion(false);
            return false;
        }

        UpdateProgress(0.3f + (i + 1) * progress_per_node);
    }

    executing_ = false;
    UpdateProgress(1.0f);
    NotifyCompletion(true);

    spdlog::info("[Data Studio] Pipeline execution completed successfully");
    return true;
}

void PipelineExecutor::StopExecution() {
    if (executing_) {
        stop_requested_ = true;
        spdlog::info("[Data Studio] Stop requested for pipeline execution");
    }
}

void PipelineExecutor::SetProgressCallback(std::function<void(float)> callback) {
    progress_callback_ = callback;
}

void PipelineExecutor::SetCompletionCallback(std::function<void(bool)> callback) {
    completion_callback_ = callback;
}

bool PipelineExecutor::ParsePipeline(const std::string& pipeline_json,
                                    std::vector<Node>& nodes) {
    try {
        auto j = nlohmann::json::parse(pipeline_json);

        for (const auto& node_json : j["nodes"]) {
            Node node;
            node.id = node_json["id"];
            node.type = node_json["type"];
            node.name = node_json["name"];
            node.parameters = node_json["parameters"].get<std::map<std::string, std::string>>();
            nodes.push_back(node);
        }

        // Build input/output connections
        for (const auto& link_json : j["links"]) {
            int start_node = link_json["start_node"];
            int end_node = link_json["end_node"];

            auto start_it = std::find_if(nodes.begin(), nodes.end(),
                                        [start_node](const Node& n) { return n.id == start_node; });
            auto end_it = std::find_if(nodes.begin(), nodes.end(),
                                      [end_node](const Node& n) { return n.id == end_node; });

            if (start_it != nodes.end() && end_it != nodes.end()) {
                start_it->outputs.push_back(end_node);
                end_it->inputs.push_back(start_node);
            }
        }

        return true;

    } catch (const std::exception& e) {
        last_error_ = std::string("JSON parse error: ") + e.what();
        return false;
    }
}

bool PipelineExecutor::ValidatePipeline(const std::vector<Node>& nodes) {
    // Check that there's at least one node
    if (nodes.empty()) {
        last_error_ = "Pipeline is empty";
        return false;
    }

    // TODO: Phase 1 Week 2 - More thorough validation
    // - Check for disconnected nodes
    // - Verify node parameters are valid
    // - Check data type compatibility

    return true;
}

std::vector<int> PipelineExecutor::TopologicalSort(const std::vector<Node>& nodes) {
    std::vector<int> result;
    std::map<int, int> in_degree;
    std::map<int, std::vector<int>> adj_list;

    // Build adjacency list and in-degree map
    for (const auto& node : nodes) {
        in_degree[node.id] = node.inputs.size();
        adj_list[node.id] = node.outputs;
    }

    // Queue for nodes with no dependencies
    std::queue<int> q;
    for (const auto& [id, degree] : in_degree) {
        if (degree == 0) {
            q.push(id);
        }
    }

    // Process nodes
    while (!q.empty()) {
        int current = q.front();
        q.pop();
        result.push_back(current);

        // Reduce in-degree for neighbors
        for (int neighbor : adj_list[current]) {
            in_degree[neighbor]--;
            if (in_degree[neighbor] == 0) {
                q.push(neighbor);
            }
        }
    }

    // Check if all nodes were processed (cycle detection)
    if (result.size() != nodes.size()) {
        return {};  // Cycle detected
    }

    return result;
}

bool PipelineExecutor::ExecuteNode(const Node& node, ExecutionContext& ctx) {
    spdlog::debug("[Data Studio] Executing node: {} (type: {})", node.name, node.type);

    if (node.type == "FileInput") {
        return ExecuteFileInput(node, ctx);
    } else if (node.type == "FilterRows") {
        return ExecuteFilterRows(node, ctx);
    } else if (node.type == "SelectColumns") {
        return ExecuteSelectColumns(node, ctx);
    } else if (node.type == "RemoveDuplicates") {
        return ExecuteRemoveDuplicates(node, ctx);
    } else if (node.type == "SaveDataset") {
        return ExecuteSaveDataset(node, ctx);
    } else {
        ReportError("Unknown node type: " + node.type);
        return false;
    }
}

bool PipelineExecutor::ExecuteFileInput(const Node& node, ExecutionContext& ctx) {
    // TODO: Phase 1 Week 2 - Load file and create Arrow table
    spdlog::info("[Data Studio] FileInput node executed (placeholder)");
    ctx.node_results[node.id] = "temp_table_" + std::to_string(node.id);
    return true;
}

bool PipelineExecutor::ExecuteFilterRows(const Node& node, ExecutionContext& ctx) {
    // TODO: Phase 1 Week 2 - Execute SQL WHERE clause
    spdlog::info("[Data Studio] FilterRows node executed (placeholder)");
    ctx.node_results[node.id] = "temp_table_" + std::to_string(node.id);
    return true;
}

bool PipelineExecutor::ExecuteSelectColumns(const Node& node, ExecutionContext& ctx) {
    // TODO: Phase 1 Week 2 - Execute SQL SELECT columns
    spdlog::info("[Data Studio] SelectColumns node executed (placeholder)");
    ctx.node_results[node.id] = "temp_table_" + std::to_string(node.id);
    return true;
}

bool PipelineExecutor::ExecuteRemoveDuplicates(const Node& node, ExecutionContext& ctx) {
    // TODO: Phase 1 Week 2 - Execute SQL DISTINCT
    spdlog::info("[Data Studio] RemoveDuplicates node executed (placeholder)");
    ctx.node_results[node.id] = "temp_table_" + std::to_string(node.id);
    return true;
}

bool PipelineExecutor::ExecuteSaveDataset(const Node& node, ExecutionContext& ctx) {
    // TODO: Phase 1 Week 2 - Save Arrow table to file
    spdlog::info("[Data Studio] SaveDataset node executed (placeholder)");
    return true;
}

void PipelineExecutor::UpdateProgress(float progress) {
    progress_ = progress;
    if (progress_callback_) {
        progress_callback_(progress);
    }
}

void PipelineExecutor::ReportError(const std::string& error) {
    last_error_ = error;
    spdlog::error("[Data Studio] Pipeline execution error: {}", error);
}

void PipelineExecutor::NotifyCompletion(bool success) {
    if (completion_callback_) {
        completion_callback_(success);
    }
}

} // namespace cyxwiz
