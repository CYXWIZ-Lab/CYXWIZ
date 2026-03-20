#include "pipeline_executor.h"
#include "../../core/duckdb_connector.h"
#include "../../core/data_registry.h"
#include "../../core/arrow_dataset.h"
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
    // Create DuckDB connector for SQL transformations
    duckdb_ = std::make_unique<DuckDBConnector>();
    spdlog::info("[Data Studio] PipelineExecutor initialized with DuckDB");
}

PipelineExecutor::~PipelineExecutor() = default;

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
    auto path_it = node.parameters.find("path");
    if (path_it == node.parameters.end() || path_it->second.empty()) {
        ReportError("FileInput node missing 'path' parameter");
        return false;
    }

    const std::string& file_path = path_it->second;
    std::string dataset_name = "ds_input_" + std::to_string(node.id);

    spdlog::info("[Data Studio] Loading file: {} as dataset '{}'", file_path, dataset_name);

    try {
        // Use DataRegistry's Arrow support to load the file
        auto& registry = DataRegistry::Instance();
        auto arrow_dataset = registry.LoadArrowTable(file_path, dataset_name);

        if (!arrow_dataset) {
            ReportError("Failed to load file: " + file_path);
            return false;
        }

        // Store the dataset name for downstream nodes
        ctx.node_results[node.id] = dataset_name;
        if (ctx.input_dataset.empty()) {
            ctx.input_dataset = dataset_name;
        }

        spdlog::info("[Data Studio] FileInput loaded {} rows, {} columns",
                    arrow_dataset->GetNumRows(), arrow_dataset->GetNumColumns());
        return true;

    } catch (const std::exception& e) {
        ReportError("FileInput error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteFilterRows(const Node& node, ExecutionContext& ctx) {
    // Get input dataset from upstream node
    if (node.inputs.empty()) {
        ReportError("FilterRows node has no input connection");
        return false;
    }

    int input_node_id = node.inputs[0];
    auto result_it = ctx.node_results.find(input_node_id);
    if (result_it == ctx.node_results.end()) {
        ReportError("FilterRows: Input dataset not found");
        return false;
    }

    // Get filter condition from parameters
    auto condition_it = node.parameters.find("condition");
    if (condition_it == node.parameters.end() || condition_it->second.empty()) {
        ReportError("FilterRows: Missing 'condition' parameter");
        return false;
    }

    const std::string& input_dataset_name = result_it->second;
    const std::string& condition = condition_it->second;
    std::string output_dataset_name = "ds_filter_" + std::to_string(node.id);

    spdlog::info("[Data Studio] Filtering rows from '{}' with condition: {}",
                input_dataset_name, condition);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("FilterRows: Input dataset not found in registry");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();

        // Register input table with DuckDB
        std::string temp_table = "temp_" + std::to_string(node.id);
        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("FilterRows: Failed to register table with DuckDB");
            return false;
        }

        // Execute WHERE query
        std::string sql = "SELECT * FROM " + temp_table + " WHERE " + condition;
        auto result_table = duckdb_->Query(sql);

        // Unregister temp table
        duckdb_->UnregisterTable(temp_table);

        if (!result_table) {
            ReportError("FilterRows: Query execution failed");
            return false;
        }

        // Register result in DataRegistry
        registry.RegisterArrowTable(result_table, output_dataset_name);

        // Store result for downstream nodes
        ctx.node_results[node.id] = output_dataset_name;

        spdlog::info("[Data Studio] FilterRows: {} -> {} rows",
                    input_table->num_rows(), result_table->num_rows());
        return true;

    } catch (const std::exception& e) {
        ReportError("FilterRows error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteSelectColumns(const Node& node, ExecutionContext& ctx) {
    // Get input dataset from upstream node
    if (node.inputs.empty()) {
        ReportError("SelectColumns node has no input connection");
        return false;
    }

    int input_node_id = node.inputs[0];
    auto result_it = ctx.node_results.find(input_node_id);
    if (result_it == ctx.node_results.end()) {
        ReportError("SelectColumns: Input dataset not found");
        return false;
    }

    // Get columns from parameters
    auto columns_it = node.parameters.find("columns");
    if (columns_it == node.parameters.end() || columns_it->second.empty()) {
        ReportError("SelectColumns: Missing 'columns' parameter");
        return false;
    }

    const std::string& input_dataset_name = result_it->second;
    const std::string& columns = columns_it->second;
    std::string output_dataset_name = "ds_select_" + std::to_string(node.id);

    spdlog::info("[Data Studio] Selecting columns from '{}': {}",
                input_dataset_name, columns);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("SelectColumns: Input dataset not found in registry");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();

        // Register input table with DuckDB
        std::string temp_table = "temp_" + std::to_string(node.id);
        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("SelectColumns: Failed to register table with DuckDB");
            return false;
        }

        // Execute SELECT columns query
        std::string sql = "SELECT " + columns + " FROM " + temp_table;
        auto result_table = duckdb_->Query(sql);

        // Unregister temp table
        duckdb_->UnregisterTable(temp_table);

        if (!result_table) {
            ReportError("SelectColumns: Query execution failed");
            return false;
        }

        // Register result in DataRegistry
        registry.RegisterArrowTable(result_table, output_dataset_name);

        // Store result for downstream nodes
        ctx.node_results[node.id] = output_dataset_name;

        spdlog::info("[Data Studio] SelectColumns: {} -> {} columns",
                    input_table->num_columns(), result_table->num_columns());
        return true;

    } catch (const std::exception& e) {
        ReportError("SelectColumns error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteRemoveDuplicates(const Node& node, ExecutionContext& ctx) {
    // Get input dataset from upstream node
    if (node.inputs.empty()) {
        ReportError("RemoveDuplicates node has no input connection");
        return false;
    }

    int input_node_id = node.inputs[0];
    auto result_it = ctx.node_results.find(input_node_id);
    if (result_it == ctx.node_results.end()) {
        ReportError("RemoveDuplicates: Input dataset not found");
        return false;
    }

    const std::string& input_dataset_name = result_it->second;
    std::string output_dataset_name = "ds_dedup_" + std::to_string(node.id);

    spdlog::info("[Data Studio] Removing duplicates from '{}'", input_dataset_name);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("RemoveDuplicates: Input dataset not found in registry");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();

        // Register input table with DuckDB
        std::string temp_table = "temp_" + std::to_string(node.id);
        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("RemoveDuplicates: Failed to register table with DuckDB");
            return false;
        }

        // Execute DISTINCT query
        std::string sql = "SELECT DISTINCT * FROM " + temp_table;
        auto result_table = duckdb_->Query(sql);

        // Unregister temp table
        duckdb_->UnregisterTable(temp_table);

        if (!result_table) {
            ReportError("RemoveDuplicates: Query execution failed");
            return false;
        }

        // Register result in DataRegistry
        registry.RegisterArrowTable(result_table, output_dataset_name);

        // Store result for downstream nodes
        ctx.node_results[node.id] = output_dataset_name;

        spdlog::info("[Data Studio] RemoveDuplicates: {} -> {} rows",
                    input_table->num_rows(), result_table->num_rows());
        return true;

    } catch (const std::exception& e) {
        ReportError("RemoveDuplicates error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteSaveDataset(const Node& node, ExecutionContext& ctx) {
    // Get the input dataset from the upstream node
    if (node.inputs.empty()) {
        ReportError("SaveDataset node has no input connection");
        return false;
    }

    int input_node_id = node.inputs[0];
    auto result_it = ctx.node_results.find(input_node_id);
    if (result_it == ctx.node_results.end()) {
        ReportError("SaveDataset: Input dataset not found");
        return false;
    }

    const std::string& input_dataset_name = result_it->second;

    // Get the desired output name from parameters
    auto name_it = node.parameters.find("name");
    std::string output_name = (name_it != node.parameters.end() && !name_it->second.empty())
                              ? name_it->second
                              : "ds_output_" + std::to_string(node.id);

    spdlog::info("[Data Studio] Saving dataset '{}' as '{}'", input_dataset_name, output_name);

    try {
        auto& registry = DataRegistry::Instance();

        // Get the Arrow dataset from the input
        auto arrow_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!arrow_dataset) {
            ReportError("SaveDataset: Input dataset not found in registry: " + input_dataset_name);
            return false;
        }

        // If the user specified a different name, register it again with the new name
        if (output_name != input_dataset_name) {
            auto arrow_table = arrow_dataset->GetArrowTable();
            registry.RegisterArrowTable(arrow_table, output_name);
        }

        // Store the output dataset name in context
        ctx.output_dataset = output_name;

        spdlog::info("[Data Studio] Dataset saved successfully as '{}'", output_name);
        return true;

    } catch (const std::exception& e) {
        ReportError("SaveDataset error: " + std::string(e.what()));
        return false;
    }
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
