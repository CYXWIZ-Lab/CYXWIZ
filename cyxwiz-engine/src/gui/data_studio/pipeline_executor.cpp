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
    , deployment_ready_(false)
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
    deployment_ready_ = false;
    deployment_dataset_.clear();

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

    // Transfer deployment status from context to executor state
    if (ctx.deployment_ready) {
        deployment_ready_ = true;
        deployment_dataset_ = ctx.deployment_dataset;
        spdlog::info("[Data Studio] Deployment ready: '{}'", deployment_dataset_);
    }

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
    } else if (node.type == "FillMissing") {
        return ExecuteFillMissing(node, ctx);
    } else if (node.type == "SortRows") {
        return ExecuteSortRows(node, ctx);
    } else if (node.type == "Join") {
        return ExecuteJoin(node, ctx);
    } else if (node.type == "GroupBy") {
        return ExecuteGroupBy(node, ctx);
    } else if (node.type == "DeployToNodeEditor") {
        return ExecuteDeployToNodeEditor(node, ctx);
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

// ============================================================================
// Phase 2 Week 4 - Additional Tabular Transformation Nodes
// ============================================================================

bool PipelineExecutor::ExecuteFillMissing(const Node& node, ExecutionContext& ctx) {
    // Get input dataset from upstream node
    if (node.inputs.empty()) {
        ReportError("FillMissing node has no input connection");
        return false;
    }

    int input_node_id = node.inputs[0];
    auto result_it = ctx.node_results.find(input_node_id);
    if (result_it == ctx.node_results.end()) {
        ReportError("FillMissing: Input dataset not found");
        return false;
    }

    // Get parameters
    auto strategy_it = node.parameters.find("strategy");
    std::string strategy = (strategy_it != node.parameters.end()) ? strategy_it->second : "mean";

    auto value_it = node.parameters.find("value");
    std::string fill_value = (value_it != node.parameters.end()) ? value_it->second : "0";

    const std::string& input_dataset_name = result_it->second;
    std::string output_dataset_name = "ds_fillmissing_" + std::to_string(node.id);

    spdlog::info("[Data Studio] Filling missing values in '{}' with strategy: {}",
                input_dataset_name, strategy);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("FillMissing: Input dataset not found in registry");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();

        // Register input table with DuckDB
        std::string temp_table = "temp_" + std::to_string(node.id);
        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("FillMissing: Failed to register table with DuckDB");
            return false;
        }

        // Build COALESCE query based on strategy
        std::string sql;
        if (strategy == "constant") {
            // Replace NULL with constant value
            sql = "SELECT * REPLACE (COALESCE(*, " + fill_value + ") AS *) FROM " + temp_table;
        } else if (strategy == "mean" || strategy == "median" || strategy == "mode") {
            // For MVP, use 0 as fallback (full implementation would compute statistics)
            // TODO: Implement proper mean/median/mode calculation
            sql = "SELECT * REPLACE (COALESCE(*, 0) AS *) FROM " + temp_table;
        } else {
            // Default to 0
            sql = "SELECT * REPLACE (COALESCE(*, 0) AS *) FROM " + temp_table;
        }

        auto result_table = duckdb_->Query(sql);

        // Unregister temp table
        duckdb_->UnregisterTable(temp_table);

        if (!result_table) {
            ReportError("FillMissing: Query execution failed");
            return false;
        }

        // Register result in DataRegistry
        registry.RegisterArrowTable(result_table, output_dataset_name);

        // Store result for downstream nodes
        ctx.node_results[node.id] = output_dataset_name;

        spdlog::info("[Data Studio] FillMissing completed: {} rows", result_table->num_rows());
        return true;

    } catch (const std::exception& e) {
        ReportError("FillMissing error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteSortRows(const Node& node, ExecutionContext& ctx) {
    // Get input dataset from upstream node
    if (node.inputs.empty()) {
        ReportError("SortRows node has no input connection");
        return false;
    }

    int input_node_id = node.inputs[0];
    auto result_it = ctx.node_results.find(input_node_id);
    if (result_it == ctx.node_results.end()) {
        ReportError("SortRows: Input dataset not found");
        return false;
    }

    // Get parameters
    auto columns_it = node.parameters.find("columns");
    if (columns_it == node.parameters.end() || columns_it->second.empty()) {
        ReportError("SortRows: Missing 'columns' parameter");
        return false;
    }

    auto order_it = node.parameters.find("order");
    std::string order = (order_it != node.parameters.end()) ? order_it->second : "ASC";

    const std::string& input_dataset_name = result_it->second;
    const std::string& sort_columns = columns_it->second;
    std::string output_dataset_name = "ds_sort_" + std::to_string(node.id);

    spdlog::info("[Data Studio] Sorting '{}' by columns: {} {}",
                input_dataset_name, sort_columns, order);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("SortRows: Input dataset not found in registry");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();

        // Register input table with DuckDB
        std::string temp_table = "temp_" + std::to_string(node.id);
        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("SortRows: Failed to register table with DuckDB");
            return false;
        }

        // Execute ORDER BY query
        std::string sql = "SELECT * FROM " + temp_table + " ORDER BY " + sort_columns + " " + order;
        auto result_table = duckdb_->Query(sql);

        // Unregister temp table
        duckdb_->UnregisterTable(temp_table);

        if (!result_table) {
            ReportError("SortRows: Query execution failed");
            return false;
        }

        // Register result in DataRegistry
        registry.RegisterArrowTable(result_table, output_dataset_name);

        // Store result for downstream nodes
        ctx.node_results[node.id] = output_dataset_name;

        spdlog::info("[Data Studio] SortRows completed: {} rows", result_table->num_rows());
        return true;

    } catch (const std::exception& e) {
        ReportError("SortRows error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteJoin(const Node& node, ExecutionContext& ctx) {
    // Join node requires two inputs
    if (node.inputs.size() < 2) {
        ReportError("Join node requires two input connections");
        return false;
    }

    // Get left and right datasets
    int left_node_id = node.inputs[0];
    int right_node_id = node.inputs[1];

    auto left_it = ctx.node_results.find(left_node_id);
    auto right_it = ctx.node_results.find(right_node_id);

    if (left_it == ctx.node_results.end() || right_it == ctx.node_results.end()) {
        ReportError("Join: One or both input datasets not found");
        return false;
    }

    // Get parameters
    auto join_type_it = node.parameters.find("join_type");
    std::string join_type = (join_type_it != node.parameters.end()) ? join_type_it->second : "INNER";

    auto on_column_it = node.parameters.find("on_column");
    if (on_column_it == node.parameters.end() || on_column_it->second.empty()) {
        ReportError("Join: Missing 'on_column' parameter");
        return false;
    }

    const std::string& left_dataset_name = left_it->second;
    const std::string& right_dataset_name = right_it->second;
    const std::string& on_column = on_column_it->second;
    std::string output_dataset_name = "ds_join_" + std::to_string(node.id);

    spdlog::info("[Data Studio] Joining '{}' and '{}' on column: {} ({})",
                left_dataset_name, right_dataset_name, on_column, join_type);

    try {
        auto& registry = DataRegistry::Instance();
        auto left_dataset = registry.GetArrowDataset(left_dataset_name);
        auto right_dataset = registry.GetArrowDataset(right_dataset_name);

        if (!left_dataset || !right_dataset) {
            ReportError("Join: Input datasets not found in registry");
            return false;
        }

        auto left_table = left_dataset->GetArrowTable();
        auto right_table = right_dataset->GetArrowTable();

        // Register both tables with DuckDB
        std::string left_temp = "temp_left_" + std::to_string(node.id);
        std::string right_temp = "temp_right_" + std::to_string(node.id);

        if (!duckdb_->RegisterTable(left_temp, left_table) ||
            !duckdb_->RegisterTable(right_temp, right_table)) {
            ReportError("Join: Failed to register tables with DuckDB");
            return false;
        }

        // Execute JOIN query
        std::string sql = "SELECT * FROM " + left_temp + " " + join_type + " JOIN " +
                         right_temp + " ON " + left_temp + "." + on_column +
                         " = " + right_temp + "." + on_column;

        auto result_table = duckdb_->Query(sql);

        // Unregister temp tables
        duckdb_->UnregisterTable(left_temp);
        duckdb_->UnregisterTable(right_temp);

        if (!result_table) {
            ReportError("Join: Query execution failed");
            return false;
        }

        // Register result in DataRegistry
        registry.RegisterArrowTable(result_table, output_dataset_name);

        // Store result for downstream nodes
        ctx.node_results[node.id] = output_dataset_name;

        spdlog::info("[Data Studio] Join completed: {} rows, {} columns",
                    result_table->num_rows(), result_table->num_columns());
        return true;

    } catch (const std::exception& e) {
        ReportError("Join error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteGroupBy(const Node& node, ExecutionContext& ctx) {
    // Get input dataset from upstream node
    if (node.inputs.empty()) {
        ReportError("GroupBy node has no input connection");
        return false;
    }

    int input_node_id = node.inputs[0];
    auto result_it = ctx.node_results.find(input_node_id);
    if (result_it == ctx.node_results.end()) {
        ReportError("GroupBy: Input dataset not found");
        return false;
    }

    // Get parameters
    auto group_columns_it = node.parameters.find("group_columns");
    if (group_columns_it == node.parameters.end() || group_columns_it->second.empty()) {
        ReportError("GroupBy: Missing 'group_columns' parameter");
        return false;
    }

    auto agg_it = node.parameters.find("aggregations");
    if (agg_it == node.parameters.end() || agg_it->second.empty()) {
        ReportError("GroupBy: Missing 'aggregations' parameter");
        return false;
    }

    const std::string& input_dataset_name = result_it->second;
    const std::string& group_columns = group_columns_it->second;
    const std::string& aggregations = agg_it->second;
    std::string output_dataset_name = "ds_groupby_" + std::to_string(node.id);

    spdlog::info("[Data Studio] GroupBy on '{}': columns={}, agg={}",
                input_dataset_name, group_columns, aggregations);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("GroupBy: Input dataset not found in registry");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();

        // Register input table with DuckDB
        std::string temp_table = "temp_" + std::to_string(node.id);
        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("GroupBy: Failed to register table with DuckDB");
            return false;
        }

        // Execute GROUP BY query
        // Aggregations format: "COUNT(*) as count, SUM(amount) as total"
        std::string sql = "SELECT " + group_columns + ", " + aggregations +
                         " FROM " + temp_table + " GROUP BY " + group_columns;

        auto result_table = duckdb_->Query(sql);

        // Unregister temp table
        duckdb_->UnregisterTable(temp_table);

        if (!result_table) {
            ReportError("GroupBy: Query execution failed");
            return false;
        }

        // Register result in DataRegistry
        registry.RegisterArrowTable(result_table, output_dataset_name);

        // Store result for downstream nodes
        ctx.node_results[node.id] = output_dataset_name;

        spdlog::info("[Data Studio] GroupBy completed: {} groups", result_table->num_rows());
        return true;

    } catch (const std::exception& e) {
        ReportError("GroupBy error: " + std::string(e.what()));
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

// ============================================================================
// Phase 5 Week 7 - Node Editor Handoff
// ============================================================================

bool PipelineExecutor::ExecuteDeployToNodeEditor(const Node& node, ExecutionContext& ctx) {
    // Get the input dataset from the upstream node
    if (node.inputs.empty()) {
        ReportError("DeployToNodeEditor node has no input connection");
        return false;
    }

    int input_node_id = node.inputs[0];
    auto result_it = ctx.node_results.find(input_node_id);
    if (result_it == ctx.node_results.end()) {
        ReportError("DeployToNodeEditor: Input dataset not found");
        return false;
    }

    const std::string& input_dataset_name = result_it->second;

    // Get the desired output name from parameters (optional)
    auto name_it = node.parameters.find("name");
    std::string deployment_name = (name_it != node.parameters.end() && !name_it->second.empty())
                                  ? name_it->second
                                  : "deployed_" + std::to_string(node.id);

    spdlog::info("[Data Studio] Preparing dataset '{}' for Node Editor deployment as '{}'",
                input_dataset_name, deployment_name);

    try {
        auto& registry = DataRegistry::Instance();

        // Get the Arrow dataset from the input
        auto arrow_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!arrow_dataset) {
            ReportError("DeployToNodeEditor: Input dataset not found in registry: " + input_dataset_name);
            return false;
        }

        // If the user specified a different name, register it again with the new name
        if (deployment_name != input_dataset_name) {
            auto arrow_table = arrow_dataset->GetArrowTable();
            registry.RegisterArrowTable(arrow_table, deployment_name);
        }

        // Tag dataset for deployment
        ctx.deployment_dataset = deployment_name;
        ctx.deployment_ready = true;

        // Also store in output_dataset for consistency
        ctx.output_dataset = deployment_name;

        spdlog::info("[Data Studio] Dataset ready for deployment: '{}'", deployment_name);
        return true;

    } catch (const std::exception& e) {
        ReportError("DeployToNodeEditor error: " + std::string(e.what()));
        return false;
    }
}

} // namespace cyxwiz
