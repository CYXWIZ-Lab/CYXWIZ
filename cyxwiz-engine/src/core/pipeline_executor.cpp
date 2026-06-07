#include "pipeline_executor.h"
#include "duckdb_connector.h"
#include "data_registry.h"
#include "arrow_dataset.h"
#include "node_executors/pipeline_operator_factory.h"
#include "pipeline_runtime_capabilities.h"
#include <arrow/table.h>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <charconv>
#include <queue>
#include <sstream>
#include <future>
#include <thread>
#include <chrono>
#include <set>
#include <mutex>

namespace cyxwiz {

namespace {

bool HasNonEmptyParameter(const std::map<std::string, std::string>& parameters,
                          const std::string& name) {
    auto it = parameters.find(name);
    return it != parameters.end() && !it->second.empty();
}

bool IsIntegerAtLeast(const std::string& value, int64_t minimum) {
    if (value.empty()) {
        return false;
    }
    int64_t parsed = 0;
    const char* begin = value.data();
    const char* end = value.data() + value.size();
    auto [ptr, ec] = std::from_chars(begin, end, parsed);
    return ec == std::errc() && ptr == end && parsed >= minimum;
}

bool ValidateIntegerParameterAtLeast(
    const std::map<std::string, std::string>& parameters,
    const std::string& node_type,
    const std::string& parameter_name,
    int64_t minimum,
    std::string& error) {
    auto it = parameters.find(parameter_name);
    if (it == parameters.end() || it->second.empty()) {
        return true;
    }
    if (IsIntegerAtLeast(it->second, minimum)) {
        return true;
    }
    error = node_type + " " + parameter_name + " must be an integer >= " +
            std::to_string(minimum);
    return false;
}

bool ValidateCommaSeparatedIntegersAtLeast(
    const std::map<std::string, std::string>& parameters,
    const std::string& node_type,
    const std::string& parameter_name,
    int64_t minimum,
    std::string& error) {
    auto it = parameters.find(parameter_name);
    if (it == parameters.end()) {
        return true;
    }
    if (it->second.empty()) {
        error = node_type + " " + parameter_name +
                " must be a comma-separated list of integers >= " +
                std::to_string(minimum);
        return false;
    }

    std::stringstream values(it->second);
    std::string value;
    while (std::getline(values, value, ',')) {
        if (!IsIntegerAtLeast(value, minimum)) {
            error = node_type + " " + parameter_name +
                    " must be a comma-separated list of integers >= " +
                    std::to_string(minimum);
            return false;
        }
    }
    return true;
}

bool IsAllowedParameterValue(
    const PipelineAllowedParameterValuesRuntimeCapability& capability,
    const std::string& value) {
    return std::find_if(capability.allowed_values.begin(),
                        capability.allowed_values.end(),
                        [&value](const char* allowed) {
                            return allowed != nullptr && value == allowed;
                        }) != capability.allowed_values.end();
}

const char* MissingRequiredParameter(
    const std::string& node_type,
    const std::map<std::string, std::string>& parameters) {
    if (node_type == "DataInput") {
        const auto source_it = parameters.find("source_type");
        const std::string source_type =
            (source_it != parameters.end() && !source_it->second.empty())
                ? source_it->second
                : "file";
        if (source_type == "file") {
            return HasNonEmptyParameter(parameters, "file_path") ? nullptr : "file_path";
        }
        if (source_type == "folder") {
            return HasNonEmptyParameter(parameters, "folder_path") ? nullptr : "folder_path";
        }
        return nullptr;
    }

    for (const char* parameter : ResolvePipelineRequiredParameters(node_type)) {
        if (!HasNonEmptyParameter(parameters, parameter)) {
            return parameter;
        }
    }

    return nullptr;
}

bool HasSupportedParameterValues(
    const std::string& node_type,
    const std::map<std::string, std::string>& parameters,
    std::string& error) {
    for (const auto& capability :
         ResolvePipelineAllowedParameterValues(node_type)) {
        auto it = parameters.find(capability.parameter_name);
        const std::string value =
            (it != parameters.end() && !it->second.empty())
                ? it->second
                : capability.default_value;
        if (!IsAllowedParameterValue(capability, value)) {
            error = node_type + " " + capability.parameter_name + " '" +
                    value + "' is not supported by PipelineExecutor";
            return false;
        }
    }

    for (const auto& capability : ResolvePipelineIntegerParameters(node_type)) {
        if (capability.comma_separated) {
            if (!ValidateCommaSeparatedIntegersAtLeast(
                    parameters, node_type, capability.parameter_name,
                    capability.minimum, error)) {
                return false;
            }
        } else if (!ValidateIntegerParameterAtLeast(
                       parameters, node_type, capability.parameter_name,
                       capability.minimum, error)) {
            return false;
        }
    }

    if (node_type == "DataInput") {
        const auto source_it = parameters.find("source_type");
        const std::string source_type =
            (source_it != parameters.end() && !source_it->second.empty())
                ? source_it->second
                : "file";

        const auto type_it = parameters.find("type");
        const std::string file_type =
            (type_it != parameters.end() && !type_it->second.empty())
                ? type_it->second
                : "csv";
        const auto skip_rows_it = parameters.find("skip_rows");
        if (source_type == "file" && skip_rows_it != parameters.end() &&
            !skip_rows_it->second.empty() &&
            !IsIntegerAtLeast(skip_rows_it->second, 0)) {
            error = "DataInput skip_rows must be a non-negative integer";
            return false;
        }
        const auto sheet_idx_it = parameters.find("sheet_idx");
        if (source_type == "file" && file_type == "excel" &&
            sheet_idx_it != parameters.end() && !sheet_idx_it->second.empty() &&
            !IsIntegerAtLeast(sheet_idx_it->second, 0)) {
            error = "DataInput sheet_idx must be a non-negative integer";
            return false;
        }
    }

    return true;
}

} // namespace

PipelineExecutor::PipelineExecutor()
    : executing_(false)
    , progress_(0.0f)
    , stop_requested_(false)
    , cancel_requested_(false)
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
    cancel_requested_ = false;
    last_error_ = "";
    current_status_ = "Starting pipeline execution...";
    deployment_ready_ = false;
    deployment_dataset_.clear();

    spdlog::info("[Data Studio] Starting pipeline execution");

    // Parse pipeline
    std::vector<Node> nodes;
    if (!ParsePipeline(pipeline_json, nodes)) {
        const std::string parse_error = last_error_.empty()
            ? "Failed to parse pipeline"
            : "Failed to parse pipeline: " + last_error_;
        ReportError(parse_error);
        executing_ = false;
        NotifyCompletion(false);
        return false;
    }

    UpdateProgress(0.1f, "Pipeline parsed successfully");

    // Validate pipeline
    if (!ValidatePipeline(nodes)) {
        const std::string validation_error = last_error_.empty()
            ? "Pipeline validation failed"
            : "Pipeline validation failed: " + last_error_;
        ReportError(validation_error);
        executing_ = false;
        NotifyCompletion(false);
        return false;
    }

    UpdateProgress(0.2f, "Pipeline validated");

    // Phase 8: Mark nodes that need execution (lazy evaluation)
    MarkDirtyNodes(nodes);

    int nodes_to_execute = 0;
    for (const auto& node : nodes) {
        if (node.needs_execution) {
            nodes_to_execute++;
        }
    }

    spdlog::info("[Data Studio] Lazy evaluation: {} of {} nodes need execution",
                 nodes_to_execute, nodes.size());
    UpdateProgress(0.25f, "Marked " + std::to_string(nodes_to_execute) + " nodes for execution");

    // Phase 8: Use parallel execution instead of sequential
    if (!ExecuteParallel(nodes)) {
        executing_ = false;
        NotifyCompletion(false);
        return false;
    }

    executing_ = false;
    UpdateProgress(1.0f, "Pipeline execution completed");

    // Deployment status is set inside ExecuteParallel
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

void PipelineExecutor::SetProgressCallback(std::function<void(float, const std::string&)> callback) {
    progress_callback_ = callback;
}

void PipelineExecutor::SetCompletionCallback(std::function<void(bool)> callback) {
    completion_callback_ = callback;
}

void PipelineExecutor::RequestCancel() {
    cancel_requested_ = true;
    spdlog::info("[Data Studio] Cancellation requested");
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

            if (start_it == nodes.end()) {
                last_error_ = "Link references missing start node id: " +
                              std::to_string(start_node);
                return false;
            }
            if (end_it == nodes.end()) {
                last_error_ = "Link references missing end node id: " +
                              std::to_string(end_node);
                return false;
            }

            start_it->outputs.push_back(end_node);
            end_it->inputs.push_back(start_node);
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

    std::set<int> ids;
    for (const auto& node : nodes) {
        if (!ids.insert(node.id).second) {
            last_error_ = "Pipeline contains duplicate node id: " + std::to_string(node.id);
            return false;
        }
    }

    for (const auto& node : nodes) {
        for (int input_id : node.inputs) {
            if (ids.find(input_id) == ids.end()) {
                last_error_ = "Node '" + node.name + "' has missing input node id: " +
                              std::to_string(input_id);
                return false;
            }
            if (input_id == node.id) {
                last_error_ = "Node '" + node.name + "' cannot connect to itself";
                return false;
            }
        }

        for (int output_id : node.outputs) {
            if (ids.find(output_id) == ids.end()) {
                last_error_ = "Node '" + node.name + "' has missing output node id: " +
                              std::to_string(output_id);
                return false;
            }
            if (output_id == node.id) {
                last_error_ = "Node '" + node.name + "' cannot connect to itself";
                return false;
            }
        }

        const bool is_source = IsPipelineSourceRuntimeNode(node.type);
        const auto required_input_count =
            ResolvePipelineRequiredInputCount(node.type);

        if (is_source && !node.inputs.empty()) {
            last_error_ = "Source node '" + node.name + "' must not have input connections";
            return false;
        }

        if (!is_source && node.inputs.empty()) {
            last_error_ = "Node '" + node.name + "' requires an input connection";
            return false;
        }

        if (required_input_count.has_value() &&
            static_cast<int>(node.inputs.size()) != *required_input_count) {
            last_error_ = "Node '" + node.name + "' requires exactly " +
                          std::to_string(*required_input_count) +
                          " input connections";
            return false;
        }

        if (!required_input_count.has_value() && node.inputs.size() > 1) {
            last_error_ = "Node '" + node.name + "' has multiple inputs, but node type '" +
                          node.type + "' does not define multi-input execution";
            return false;
        }

        if (const char* missing_parameter =
                MissingRequiredParameter(node.type, node.parameters);
            missing_parameter != nullptr) {
            last_error_ = "Node '" + node.name + "' of type '" + node.type +
                          "' is missing required parameter '" + missing_parameter + "'";
            return false;
        }

        std::string parameter_error;
        if (!HasSupportedParameterValues(node.type, node.parameters, parameter_error)) {
            last_error_ = "Node '" + node.name + "': " + parameter_error;
            return false;
        }

        if (ResolvePipelineRuntimeSupport(node.type).mode ==
            PipelineRuntimeSupportMode::Unknown) {
            last_error_ = "Node '" + node.name + "' has unsupported node type '" +
                          node.type + "' for PipelineExecutor";
            return false;
        }
    }

    if (nodes.size() > 1) {
        std::set<int> visited;
        std::queue<int> pending;
        pending.push(nodes.front().id);
        visited.insert(nodes.front().id);

        while (!pending.empty()) {
            const int current = pending.front();
            pending.pop();

            auto current_it = std::find_if(
                nodes.begin(), nodes.end(),
                [current](const Node& node) { return node.id == current; });
            if (current_it == nodes.end()) {
                continue;
            }

            for (int input_id : current_it->inputs) {
                if (visited.insert(input_id).second) {
                    pending.push(input_id);
                }
            }
            for (int output_id : current_it->outputs) {
                if (visited.insert(output_id).second) {
                    pending.push(output_id);
                }
            }
        }

        if (visited.size() != nodes.size()) {
            last_error_ = "Pipeline contains disconnected nodes";
            return false;
        }
    }

    if (TopologicalSort(nodes).empty() && !nodes.empty()) {
        last_error_ = "Pipeline contains a cycle";
        return false;
    }

    return true;
}

std::vector<int> PipelineExecutor::TopologicalSort(const std::vector<Node>& nodes) {
    std::vector<int> result;
    std::map<int, int> in_degree;
    std::map<int, std::vector<int>> adj_list;

    // Build adjacency list and in-degree map
    for (const auto& node : nodes) {
        in_degree[node.id] = static_cast<int>(node.inputs.size());
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
    } else if (node.type == "DataInput") {
        return ExecuteDataInput(node, ctx);
    } else if (node.type == "DataOutput") {
        return ExecuteDataOutput(node, ctx);
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
    } else if (auto support = ResolvePipelineRuntimeSupport(node.type);
               support.mode == PipelineRuntimeSupportMode::OperatorBacked &&
                   support.operator_type.has_value()) {
        return ExecutePipelineOperatorNode(node, ctx, *support.operator_type);
    } else if (support.mode == PipelineRuntimeSupportMode::FailClosed &&
               support.fail_closed_reason != nullptr) {
        return FailUnsupportedNode(node, support.fail_closed_reason);
    }
    // Phase 6 Week 8-9 - Text Processing
    else if (node.type == "TextClean") {
        return ExecuteTextClean(node, ctx);
    } else if (node.type == "TextTokenize") {
        return ExecuteTextTokenize(node, ctx);
    } else if (node.type == "TextVectorize") {
        return ExecuteTextVectorize(node, ctx);
    }
    // Phase 6 Week 8-9 - Time-Series
    else if (node.type == "TSWindow") {
        return ExecuteTSWindow(node, ctx);
    } else if (node.type == "TSFeatures") {
        return ExecuteTSFeatures(node, ctx);
    } else if (node.type == "TSLag") {
        return ExecuteTSLag(node, ctx);
    } else if (node.type == "TSDiff") {
        return ExecuteTSDiff(node, ctx);
    }
    // Phase 6 Week 8-9 - Feature Engineering
    else if (node.type == "PolynomialFeatures") {
        return ExecutePolynomialFeatures(node, ctx);
    } else if (node.type == "Binning") {
        return ExecuteBinning(node, ctx);
    }
    // KNIME-Style Table Manipulation Nodes
    else if (node.type == "ExcelInput") {
        return ExecuteExcelInput(node, ctx);
    } else if (node.type == "ExportExcel") {
        return ExecuteExportExcel(node, ctx);
    } else if (node.type == "ExportCSV") {
        return ExecuteExportCSV(node, ctx);
    } else if (node.type == "ExportJSON") {
        return ExecuteExportJSON(node, ctx);
    } else if (node.type == "RowToColumnNames") {
        return ExecuteRowToColumnNames(node, ctx);
    } else if (node.type == "TableSplitter") {
        return ExecuteTableSplitter(node, ctx);
    } else if (node.type == "CellExtractor") {
        return ExecuteCellExtractor(node, ctx);
    } else if (node.type == "CellUpdater") {
        return ExecuteCellUpdater(node, ctx);
    } else if (node.type == "TableCropper") {
        return ExecuteTableCropper(node, ctx);
    } else if (node.type == "ColumnAppender") {
        return ExecuteColumnAppender(node, ctx);
    } else if (node.type == "RowAppender") {
        return ExecuteRowAppender(node, ctx);
    } else if (node.type == "Unpivot") {
        return ExecuteUnpivot(node, ctx);
    } else if (node.type == "StringManipulation") {
        return ExecuteStringManipulation(node, ctx);
    } else if (node.type == "MathFormula") {
        return ExecuteMathFormula(node, ctx);
    } else if (node.type == "RuleEngine") {
        return ExecuteRuleEngine(node, ctx);
    } else if (node.type == "RenameColumns") {
        return ExecuteRenameColumns(node, ctx);
    }
    else {
        ReportError("Unknown node type: " + node.type);
        return false;
    }
}

bool PipelineExecutor::ExecuteFileInput(const Node& node, ExecutionContext& ctx) {
    auto path_it = node.parameters.find("path");
    if (path_it == node.parameters.end() || path_it->second.empty()) {
        ReportError(GetImprovedErrorMessage("FileInput", "missing_parameter", "path"));
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
            ReportError(GetImprovedErrorMessage("FileInput", "invalid_path", file_path));
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

bool PipelineExecutor::ExecuteDataInput(const Node& node, ExecutionContext& ctx) {
    // Universal DataInput node - supports multiple source types from DataInputDialog
    // Parameters: source_type, file_path, folder_path, file_category, type, etc.

    auto source_type_it = node.parameters.find("source_type");
    std::string source_type = (source_type_it != node.parameters.end()) ? source_type_it->second : "file";

    std::string dataset_name = "ds_datainput_" + std::to_string(node.id);

    try {
        auto& registry = DataRegistry::Instance();
        std::shared_ptr<ArrowDataset> arrow_dataset;

        if (source_type == "file") {
            // File input mode
            auto path_it = node.parameters.find("file_path");
            if (path_it == node.parameters.end() || path_it->second.empty()) {
                ReportError(GetImprovedErrorMessage("DataInput", "missing_parameter", "file_path"));
                return false;
            }
            const std::string& file_path = path_it->second;
            spdlog::info("[Pipeline] DataInput loading file: {}", file_path);

            // Get file type and options from parameters
            auto type_it = node.parameters.find("type");
            std::string file_type = (type_it != node.parameters.end()) ? type_it->second : "csv";

            // Load based on file type
            if (file_type == "csv" || file_type == "tsv") {
                bool has_header = node.parameters.count("has_header") && node.parameters.at("has_header") == "true";
                std::string delimiter = (file_type == "tsv") ? "\t" : ",";
                auto delim_it = node.parameters.find("delimiter");
                if (delim_it != node.parameters.end() && !delim_it->second.empty()) {
                    delimiter = delim_it->second;
                }
                int skip_rows = 0;
                auto skip_it = node.parameters.find("skip_rows");
                if (skip_it != node.parameters.end()) {
                    skip_rows = std::stoi(skip_it->second);
                }

                arrow_dataset = registry.LoadCSVToArrow(file_path, dataset_name, has_header, delimiter[0], skip_rows);
            } else if (file_type == "parquet") {
                arrow_dataset = registry.LoadParquetToArrow(file_path, dataset_name);
            } else if (file_type == "json") {
                bool json_lines = node.parameters.count("json_lines") && node.parameters.at("json_lines") == "true";
                arrow_dataset = registry.LoadJSONToArrow(file_path, dataset_name, json_lines);
            } else if (file_type == "excel") {
                int sheet_idx = 0;
                auto sheet_it = node.parameters.find("sheet_idx");
                if (sheet_it != node.parameters.end()) {
                    sheet_idx = std::stoi(sheet_it->second);
                }
                arrow_dataset = registry.LoadExcelToArrow(file_path, dataset_name, sheet_idx);
            } else {
                // Default: try auto-detect via LoadArrowTable
                arrow_dataset = registry.LoadArrowTable(file_path, dataset_name);
            }

        } else if (source_type == "folder") {
            // Image folder mode
            auto path_it = node.parameters.find("folder_path");
            if (path_it == node.parameters.end() || path_it->second.empty()) {
                ReportError(GetImprovedErrorMessage("DataInput", "missing_parameter", "folder_path"));
                return false;
            }
            const std::string& folder_path = path_it->second;
            spdlog::info("[Pipeline] DataInput loading folder: {}", folder_path);

            // For image folders, create a table with file paths and labels
            arrow_dataset = registry.LoadImageFolderToArrow(folder_path, dataset_name);

        } else if (source_type == "ml_dataset") {
            // ML dataset (MNIST, CIFAR, etc.)
            auto ml_type_it = node.parameters.find("ml_dataset_type");
            std::string ml_type = (ml_type_it != node.parameters.end()) ? ml_type_it->second : "mnist";
            spdlog::info("[Pipeline] DataInput loading ML dataset: {}", ml_type);

            arrow_dataset = registry.LoadMLDatasetToArrow(ml_type, dataset_name);
        } else {
            ReportError("DataInput: Unknown source type: " + source_type);
            return false;
        }

        if (!arrow_dataset) {
            ReportError("DataInput: Failed to load dataset");
            return false;
        }

        // Store result for downstream nodes
        ctx.node_results[node.id] = dataset_name;
        if (ctx.input_dataset.empty()) {
            ctx.input_dataset = dataset_name;
        }

        spdlog::info("[Pipeline] DataInput loaded {} rows, {} columns",
                    arrow_dataset->GetNumRows(), arrow_dataset->GetNumColumns());
        return true;

    } catch (const std::exception& e) {
        ReportError("DataInput error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteDataOutput(const Node& node, ExecutionContext& ctx) {
    // Universal DataOutput node - exports data to various formats

    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("DataOutput: No input connection or dataset not found");
        return false;
    }

    auto path_it = node.parameters.find("file_path");
    if (path_it == node.parameters.end() || path_it->second.empty()) {
        ReportError(GetImprovedErrorMessage("DataOutput", "missing_parameter", "file_path"));
        return false;
    }
    const std::string& output_path = path_it->second;

    auto format_it = node.parameters.find("format");
    std::string format = (format_it != node.parameters.end()) ? format_it->second : "csv";

    spdlog::info("[Pipeline] DataOutput exporting to {} (format: {})", output_path, format);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("DataOutput: Input dataset not found in registry");
            return false;
        }

        bool success = false;
        if (format == "csv") {
            success = registry.ExportArrowToCSV(input_dataset_name, output_path);
        } else if (format == "parquet") {
            success = registry.ExportArrowToParquet(input_dataset_name, output_path);
        } else if (format == "json") {
            success = registry.ExportArrowToJSON(input_dataset_name, output_path);
        } else {
            ReportError("DataOutput: Unsupported export format: " + format);
            return false;
        }

        if (!success) {
            ReportError("DataOutput: Export failed");
            return false;
        }

        // Pass through the dataset for any downstream nodes
        ctx.node_results[node.id] = input_dataset_name;
        ctx.output_dataset = output_path;

        spdlog::info("[Pipeline] DataOutput successfully exported to {}", output_path);
        return true;

    } catch (const std::exception& e) {
        ReportError("DataOutput error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteFilterRows(const Node& node, ExecutionContext& ctx) {
    // Get input dataset from upstream node
    if (node.inputs.empty()) {
        ReportError(GetImprovedErrorMessage("FilterRows", "no_input"));
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

void PipelineExecutor::UpdateProgress(float progress, const std::string& status) {
    progress_ = progress;
    if (!status.empty()) {
        current_status_ = status;
    }
    if (progress_callback_) {
        progress_callback_(progress, current_status_);
    }
}

void PipelineExecutor::ReportError(const std::string& error) {
    last_error_ = error;
    spdlog::error("[Data Studio] Pipeline execution error: {}", error);
}

// Phase 7: Improved error message helper with actionable suggestions
std::string PipelineExecutor::GetImprovedErrorMessage(const std::string& node_type, const std::string& error_category, const std::string& details) {
    std::string message;

    if (error_category == "no_input") {
        message = node_type + ": No input dataset connected\n"
                  "Suggestion: Connect a FileInput or upstream transformation node";
    } else if (error_category == "dataset_not_found") {
        message = node_type + ": Input dataset not found\n"
                  "Suggestion: Ensure the upstream node executed successfully";
    } else if (error_category == "missing_parameter") {
        message = node_type + ": Missing required parameter '" + details + "'\n"
                  "Suggestion: Configure the node by right-clicking and selecting 'Configure'";
    } else if (error_category == "column_not_found") {
        message = node_type + ": Column '" + details + "' not found in dataset\n"
                  "Suggestion: Check dataset schema or use SelectColumns node first";
    } else if (error_category == "query_failed") {
        message = node_type + ": SQL query execution failed\n"
                  "Details: " + details + "\n"
                  "Suggestion: Check your filter conditions, column names, or SQL syntax";
    } else if (error_category == "empty_result") {
        message = node_type + ": Query returned 0 rows\n"
                  "Suggestion: Check filter conditions or use less restrictive thresholds";
    } else if (error_category == "type_mismatch") {
        message = node_type + ": Cannot apply numeric operation to text column\n"
                  "Suggestion: Use TextVectorize node to convert text to numbers first";
    } else if (error_category == "invalid_path") {
        message = node_type + ": File not found at path: " + details + "\n"
                  "Suggestion: Check file path, ensure file exists, and has correct permissions";
    } else if (error_category == "register_failed") {
        message = node_type + ": Failed to register table with DuckDB\n"
                  "Details: " + details + "\n"
                  "Suggestion: Check dataset format and memory availability";
    } else {
        // Fallback to generic message
        message = node_type + ": " + details;
    }

    return message;
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

// ============================================================================
// Phase 6 Week 8-9: Advanced Nodes Implementation
// ============================================================================

std::string PipelineExecutor::GetInputDatasetName(const Node& node, ExecutionContext& ctx) {
    if (node.inputs.empty()) {
        return "";
    }
    if (node.inputs.size() > 1) {
        ReportError(node.type + ": multiple inputs are not supported by this executor path");
        return "";
    }
    int input_node_id = node.inputs[0];
    auto result_it = ctx.node_results.find(input_node_id);
    if (result_it == ctx.node_results.end()) {
        return "";
    }
    return result_it->second;
}

bool PipelineExecutor::FailUnsupportedNode(const Node& node, const std::string& reason) {
    ReportError(node.type + " is not executable in the legacy Data Studio pipeline path: " +
                reason);
    return false;
}

bool PipelineExecutor::ExecutePipelineOperatorNode(
    const Node& node,
    ExecutionContext& ctx,
    gui::NodeType type) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError(node.type + ": No input connection or dataset not found");
        return false;
    }

    auto& registry = DataRegistry::Instance();
    auto input_dataset = registry.GetArrowDataset(input_dataset_name);
    if (!input_dataset) {
        ReportError(node.type + ": input dataset '" + input_dataset_name +
                    "' is not an in-memory Arrow dataset");
        return false;
    }

    auto input_table = input_dataset->GetArrowTable();
    if (!input_table) {
        ReportError(node.type + ": input Arrow table is null");
        return false;
    }

    auto& factory = PipelineOperatorFactory::Instance();
    if (!factory.HasOperator(type)) {
        ReportError(node.type + ": no PipelineOperatorFactory registration exists");
        return false;
    }

    auto op = factory.Create(type);
    if (!op) {
        ReportError(node.type + ": PipelineOperatorFactory returned null operator");
        return false;
    }

    std::string configure_error;
    if (!op->Configure(node.parameters, configure_error)) {
        ReportError(configure_error.empty()
                        ? node.type + ": operator configuration failed"
                        : configure_error);
        return false;
    }

    auto result = op->Apply(input_table);
    if (!result.ok()) {
        ReportError(node.type + ": operator execution failed: " +
                    result.status().ToString());
        return false;
    }

    auto output_table = result.ValueOrDie();
    const std::string output_dataset_name =
        "ds_operator_" + node.type + "_" + std::to_string(node.id);
    auto output_dataset = registry.RegisterArrowTable(output_table, output_dataset_name);
    if (!output_dataset) {
        ReportError(node.type + ": failed to register operator output dataset");
        return false;
    }

    ctx.node_results[node.id] = output_dataset_name;
    ctx.output_dataset = output_dataset_name;

    spdlog::info("[Data Studio] {} routed through PipelineOperatorFactory: {} rows, {} columns",
                 node.type, output_table->num_rows(), output_table->num_columns());
    return true;
}

// ============================================================================
// Text Processing Nodes
// ============================================================================

bool PipelineExecutor::ExecuteTextClean(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("TextClean: No input connection or dataset not found");
        return false;
    }

    // Get parameters
    bool lowercase = node.parameters.count("lowercase") && node.parameters.at("lowercase") == "true";
    bool remove_html = node.parameters.count("remove_html") && node.parameters.at("remove_html") == "true";
    bool remove_special_chars = node.parameters.count("remove_special_chars") && node.parameters.at("remove_special_chars") == "true";
    // Note: remove_stopwords would require dictionary integration - not implemented in MVP
    // bool remove_stopwords = node.parameters.count("remove_stopwords") && node.parameters.at("remove_stopwords") == "true";

    auto column_it = node.parameters.find("text_column");
    std::string text_column = (column_it != node.parameters.end()) ? column_it->second : "text";

    std::string output_dataset_name = "ds_textclean_" + std::to_string(node.id);

    spdlog::info("[Data Studio] TextClean on column '{}' from '{}'", text_column, input_dataset_name);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("TextClean: Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        std::string temp_table = "temp_" + std::to_string(node.id);

        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("TextClean: Failed to register table");
            return false;
        }

        // Build SQL transformation chain
        std::string sql = "SELECT *, ";
        std::string transform = text_column;

        if (remove_html) {
            transform = "regexp_replace(" + transform + ", '<[^>]*>', '', 'g')";
        }
        if (remove_special_chars) {
            transform = "regexp_replace(" + transform + ", '[^a-zA-Z0-9\\s]', '', 'g')";
        }
        if (lowercase) {
            transform = "lower(" + transform + ")";
        }
        // Remove extra whitespace
        transform = "regexp_replace(" + transform + ", '\\s+', ' ', 'g')";
        transform = "trim(" + transform + ")";

        sql += transform + " AS " + text_column + "_cleaned FROM " + temp_table;

        auto result_table = duckdb_->Query(sql);
        duckdb_->UnregisterTable(temp_table);

        if (!result_table) {
            ReportError("TextClean: Query failed");
            return false;
        }

        registry.RegisterArrowTable(result_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;

        spdlog::info("[Data Studio] TextClean completed: {} rows", result_table->num_rows());
        return true;

    } catch (const std::exception& e) {
        ReportError("TextClean error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteTextTokenize(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("TextTokenize: No input connection or dataset not found");
        return false;
    }

    auto column_it = node.parameters.find("text_column");
    std::string text_column = (column_it != node.parameters.end()) ? column_it->second : "text";

    auto method_it = node.parameters.find("method");
    std::string method = (method_it != node.parameters.end()) ? method_it->second : "word";

    std::string output_dataset_name = "ds_texttokenize_" + std::to_string(node.id);

    spdlog::info("[Data Studio] TextTokenize ({}) on column '{}' from '{}'",
                method, text_column, input_dataset_name);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("TextTokenize: Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        std::string temp_table = "temp_" + std::to_string(node.id);

        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("TextTokenize: Failed to register table");
            return false;
        }

        std::string sql;
        if (method == "word") {
            // Split on whitespace and punctuation
            sql = "SELECT *, string_split_regex(" + text_column + ", '\\s+') AS " +
                  text_column + "_tokens FROM " + temp_table;
        } else if (method == "sentence") {
            // Split on sentence boundaries
            sql = "SELECT *, string_split_regex(" + text_column + ", '[.!?]+') AS " +
                  text_column + "_tokens FROM " + temp_table;
        } else if (method == "character") {
            // Split into characters (list of individual chars)
            sql = "SELECT *, string_split(" + text_column + ", '') AS " +
                  text_column + "_tokens FROM " + temp_table;
        } else {
            sql = "SELECT *, string_split(" + text_column + ", ' ') AS " +
                  text_column + "_tokens FROM " + temp_table;
        }

        auto result_table = duckdb_->Query(sql);
        duckdb_->UnregisterTable(temp_table);

        if (!result_table) {
            ReportError("TextTokenize: Query failed");
            return false;
        }

        registry.RegisterArrowTable(result_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;

        spdlog::info("[Data Studio] TextTokenize completed: {} rows", result_table->num_rows());
        return true;

    } catch (const std::exception& e) {
        ReportError("TextTokenize error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteTextVectorize(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("TextVectorize: No input connection or dataset not found");
        return false;
    }

    auto column_it = node.parameters.find("text_column");
    std::string text_column = (column_it != node.parameters.end()) ? column_it->second : "text";

    auto method_it = node.parameters.find("method");
    std::string method = (method_it != node.parameters.end()) ? method_it->second : "count";

    std::string output_dataset_name = "ds_textvectorize_" + std::to_string(node.id);

    spdlog::info("[Data Studio] TextVectorize ({}) on column '{}' from '{}'",
                method, text_column, input_dataset_name);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("TextVectorize: Input dataset not found");
            return false;
        }

        // For MVP: Create simple word count features
        // Full implementation would use TF-IDF or embeddings from backend
        auto input_table = input_dataset->GetArrowTable();
        std::string temp_table = "temp_" + std::to_string(node.id);

        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("TextVectorize: Failed to register table");
            return false;
        }

        // Create simple features: text length, word count
        std::string sql = "SELECT *, "
                         "length(" + text_column + ") AS text_length, "
                         "length(" + text_column + ") - length(replace(" + text_column + ", ' ', '')) + 1 AS word_count "
                         "FROM " + temp_table;

        auto result_table = duckdb_->Query(sql);
        duckdb_->UnregisterTable(temp_table);

        if (!result_table) {
            ReportError("TextVectorize: Query failed");
            return false;
        }

        registry.RegisterArrowTable(result_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;

        spdlog::info("[Data Studio] TextVectorize completed: {} rows, basic features added",
                    result_table->num_rows());
        return true;

    } catch (const std::exception& e) {
        ReportError("TextVectorize error: " + std::string(e.what()));
        return false;
    }
}

// ============================================================================
// Time-Series Nodes
// ============================================================================

bool PipelineExecutor::ExecuteTSWindow(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("TSWindow: No input connection or dataset not found");
        return false;
    }

    auto window_it = node.parameters.find("window_size");
    int window_size = (window_it != node.parameters.end()) ? std::stoi(window_it->second) : 10;

    auto stride_it = node.parameters.find("stride");
    int stride = (stride_it != node.parameters.end()) ? std::stoi(stride_it->second) : 1;

    auto target_it = node.parameters.find("target_column");
    std::string target_column = (target_it != node.parameters.end()) ? target_it->second : "value";

    std::string output_dataset_name = "ds_tswindow_" + std::to_string(node.id);

    spdlog::info("[Data Studio] TSWindow (size={}, stride={}) on '{}' from '{}'",
                window_size, stride, target_column, input_dataset_name);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("TSWindow: Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        std::string temp_table = "temp_" + std::to_string(node.id);

        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("TSWindow: Failed to register table");
            return false;
        }

        // Create windows using LAG function
        std::string sql = "SELECT *, ";
        for (int i = 0; i < window_size; i++) {
            if (i > 0) sql += ", ";
            sql += "LAG(" + target_column + ", " + std::to_string(i) + ") OVER (ORDER BY rowid) AS window_t" + std::to_string(i);
        }
        sql += " FROM " + temp_table;

        auto result_table = duckdb_->Query(sql);
        duckdb_->UnregisterTable(temp_table);

        if (!result_table) {
            ReportError("TSWindow: Query failed");
            return false;
        }

        registry.RegisterArrowTable(result_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;

        spdlog::info("[Data Studio] TSWindow completed: {} rows with {} timestep windows",
                    result_table->num_rows(), window_size);
        return true;

    } catch (const std::exception& e) {
        ReportError("TSWindow error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteTSFeatures(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("TSFeatures: No input connection or dataset not found");
        return false;
    }

    auto column_it = node.parameters.find("columns");
    std::string columns = (column_it != node.parameters.end()) ? column_it->second : "value";

    auto window_it = node.parameters.find("rolling_window");
    int rolling_window = (window_it != node.parameters.end()) ? std::stoi(window_it->second) : 7;

    std::string output_dataset_name = "ds_tsfeatures_" + std::to_string(node.id);

    spdlog::info("[Data Studio] TSFeatures (window={}) on '{}' from '{}'",
                rolling_window, columns, input_dataset_name);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("TSFeatures: Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        std::string temp_table = "temp_" + std::to_string(node.id);

        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("TSFeatures: Failed to register table");
            return false;
        }

        // Create rolling statistics
        std::string sql = "SELECT *, "
                         "AVG(" + columns + ") OVER (ORDER BY rowid ROWS BETWEEN " +
                         std::to_string(rolling_window - 1) + " PRECEDING AND CURRENT ROW) AS " +
                         columns + "_rolling_mean, "
                         "STDDEV(" + columns + ") OVER (ORDER BY rowid ROWS BETWEEN " +
                         std::to_string(rolling_window - 1) + " PRECEDING AND CURRENT ROW) AS " +
                         columns + "_rolling_std, "
                         "MIN(" + columns + ") OVER (ORDER BY rowid ROWS BETWEEN " +
                         std::to_string(rolling_window - 1) + " PRECEDING AND CURRENT ROW) AS " +
                         columns + "_rolling_min, "
                         "MAX(" + columns + ") OVER (ORDER BY rowid ROWS BETWEEN " +
                         std::to_string(rolling_window - 1) + " PRECEDING AND CURRENT ROW) AS " +
                         columns + "_rolling_max "
                         "FROM " + temp_table;

        auto result_table = duckdb_->Query(sql);
        duckdb_->UnregisterTable(temp_table);

        if (!result_table) {
            ReportError("TSFeatures: Query failed");
            return false;
        }

        registry.RegisterArrowTable(result_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;

        spdlog::info("[Data Studio] TSFeatures completed: {} rows with rolling statistics",
                    result_table->num_rows());
        return true;

    } catch (const std::exception& e) {
        ReportError("TSFeatures error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteTSLag(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("TSLag: No input connection or dataset not found");
        return false;
    }

    auto column_it = node.parameters.find("columns");
    std::string columns = (column_it != node.parameters.end()) ? column_it->second : "value";

    auto lags_it = node.parameters.find("lag_periods");
    std::string lag_periods = (lags_it != node.parameters.end()) ? lags_it->second : "1,7,30";

    std::string output_dataset_name = "ds_tslag_" + std::to_string(node.id);

    spdlog::info("[Data Studio] TSLag (periods={}) on '{}' from '{}'",
                lag_periods, columns, input_dataset_name);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("TSLag: Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        std::string temp_table = "temp_" + std::to_string(node.id);

        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("TSLag: Failed to register table");
            return false;
        }

        // Parse lag periods (comma-separated)
        std::string sql = "SELECT *, ";
        std::stringstream ss(lag_periods);
        std::string lag_str;
        bool first = true;

        while (std::getline(ss, lag_str, ',')) {
            int lag = std::stoi(lag_str);
            if (!first) sql += ", ";
            sql += "LAG(" + columns + ", " + std::to_string(lag) + ") OVER (ORDER BY rowid) AS " +
                   columns + "_lag" + std::to_string(lag);
            first = false;
        }

        sql += " FROM " + temp_table;

        auto result_table = duckdb_->Query(sql);
        duckdb_->UnregisterTable(temp_table);

        if (!result_table) {
            ReportError("TSLag: Query failed");
            return false;
        }

        registry.RegisterArrowTable(result_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;

        spdlog::info("[Data Studio] TSLag completed: {} rows with lag features",
                    result_table->num_rows());
        return true;

    } catch (const std::exception& e) {
        ReportError("TSLag error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteTSDiff(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("TSDiff: No input connection or dataset not found");
        return false;
    }

    auto column_it = node.parameters.find("columns");
    std::string columns = (column_it != node.parameters.end()) ? column_it->second : "value";

    auto order_it = node.parameters.find("order");
    int order = (order_it != node.parameters.end()) ? std::stoi(order_it->second) : 1;

    std::string output_dataset_name = "ds_tsdiff_" + std::to_string(node.id);

    spdlog::info("[Data Studio] TSDiff (order={}) on '{}' from '{}'",
                order, columns, input_dataset_name);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("TSDiff: Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        std::string temp_table = "temp_" + std::to_string(node.id);

        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("TSDiff: Failed to register table");
            return false;
        }

        // Create difference features
        std::string sql = "SELECT *, ";
        for (int i = 1; i <= order; i++) {
            if (i > 1) sql += ", ";
            sql += columns + " - LAG(" + columns + ", " + std::to_string(i) + ") OVER (ORDER BY rowid) AS " +
                   columns + "_diff" + std::to_string(i);
        }
        sql += " FROM " + temp_table;

        auto result_table = duckdb_->Query(sql);
        duckdb_->UnregisterTable(temp_table);

        if (!result_table) {
            ReportError("TSDiff: Query failed");
            return false;
        }

        registry.RegisterArrowTable(result_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;

        spdlog::info("[Data Studio] TSDiff completed: {} rows with difference features",
                    result_table->num_rows());
        return true;

    } catch (const std::exception& e) {
        ReportError("TSDiff error: " + std::string(e.what()));
        return false;
    }
}

// ============================================================================
// Feature Engineering Nodes
// ============================================================================

bool PipelineExecutor::ExecutePolynomialFeatures(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("PolynomialFeatures: No input connection or dataset not found");
        return false;
    }

    auto degree_it = node.parameters.find("degree");
    int degree = (degree_it != node.parameters.end()) ? std::stoi(degree_it->second) : 2;

    auto columns_it = node.parameters.find("columns");
    std::string columns = (columns_it != node.parameters.end()) ? columns_it->second : "";

    std::string output_dataset_name = "ds_poly_" + std::to_string(node.id);

    spdlog::info("[Data Studio] PolynomialFeatures (degree={}) on '{}' from '{}'",
                degree, columns.empty() ? "all numeric" : columns, input_dataset_name);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("PolynomialFeatures: Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        std::string temp_table = "temp_" + std::to_string(node.id);

        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("PolynomialFeatures: Failed to register table");
            return false;
        }

        // For MVP: Create simple polynomial features for specified columns
        // If no columns specified, this is a placeholder
        std::string sql;
        if (!columns.empty()) {
            sql = "SELECT *, ";
            if (degree >= 2) {
                sql += columns + " * " + columns + " AS " + columns + "_squared";
            }
            if (degree >= 3) {
                sql += ", " + columns + " * " + columns + " * " + columns + " AS " + columns + "_cubed";
            }
            sql += " FROM " + temp_table;
        } else {
            // Passthrough if no columns specified
            sql = "SELECT * FROM " + temp_table;
        }

        auto result_table = duckdb_->Query(sql);
        duckdb_->UnregisterTable(temp_table);

        if (!result_table) {
            ReportError("PolynomialFeatures: Query failed");
            return false;
        }

        registry.RegisterArrowTable(result_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;

        spdlog::info("[Data Studio] PolynomialFeatures completed: {} rows",
                    result_table->num_rows());
        return true;

    } catch (const std::exception& e) {
        ReportError("PolynomialFeatures error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteBinning(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("Binning: No input connection or dataset not found");
        return false;
    }

    auto column_it = node.parameters.find("columns");
    std::string columns = (column_it != node.parameters.end()) ? column_it->second : "value";

    auto n_bins_it = node.parameters.find("n_bins");
    int n_bins = (n_bins_it != node.parameters.end()) ? std::stoi(n_bins_it->second) : 10;

    auto method_it = node.parameters.find("method");
    std::string method = (method_it != node.parameters.end()) ? method_it->second : "equal_width";

    std::string output_dataset_name = "ds_binning_" + std::to_string(node.id);

    spdlog::info("[Data Studio] Binning (method={}, bins={}) on '{}' from '{}'",
                method, n_bins, columns, input_dataset_name);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("Binning: Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        std::string temp_table = "temp_" + std::to_string(node.id);

        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("Binning: Failed to register table");
            return false;
        }

        // Use DuckDB's NTILE for equal frequency binning
        std::string sql;
        if (method == "equal_freq") {
            sql = "SELECT *, NTILE(" + std::to_string(n_bins) + ") OVER (ORDER BY " +
                  columns + ") AS " + columns + "_bin FROM " + temp_table;
        } else {
            // Equal width binning using WIDTH_BUCKET
            sql = "SELECT *, WIDTH_BUCKET(" + columns + ", "
                  "(SELECT MIN(" + columns + ") FROM " + temp_table + "), "
                  "(SELECT MAX(" + columns + ") FROM " + temp_table + "), "
                  + std::to_string(n_bins) + ") AS " + columns + "_bin FROM " + temp_table;
        }

        auto result_table = duckdb_->Query(sql);
        duckdb_->UnregisterTable(temp_table);

        if (!result_table) {
            ReportError("Binning: Query failed");
            return false;
        }

        registry.RegisterArrowTable(result_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;

        spdlog::info("[Data Studio] Binning completed: {} rows with {} bins",
                    result_table->num_rows(), n_bins);
        return true;

    } catch (const std::exception& e) {
        ReportError("Binning error: " + std::string(e.what()));
        return false;
    }
}

// ============================================================================
// Phase 8: Performance Optimization Implementation
// ============================================================================

/* Memory Optimization Strategy (Streaming Mode)
 *
 * For large datasets (>1GB), we can implement chunk-based processing:
 *
 * 1. Check dataset size before loading:
 *    if (file_size > memory_limit_) { streaming_mode_ = true; }
 *
 * 2. Use Arrow RecordBatch API instead of full Table:
 *    auto reader = arrow::ipc::RecordBatchFileReader::Open(file);
 *    for (int i = 0; i < reader->num_record_batches(); i++) {
 *        auto batch = reader->ReadRecordBatch(i);
 *        ProcessChunk(batch);  // Process in chunks
 *    }
 *
 * 3. Example: Streaming RemoveDuplicates
 *    std::unordered_set<std::string> seen_hashes;
 *    for (int64_t offset = 0; offset < total_rows; offset += chunk_size_) {
 *        auto batch = table->Slice(offset, chunk_size_);
 *        for (int64_t i = 0; i < batch->num_rows(); i++) {
 *            std::string row_hash = ComputeRowHash(batch, i);
 *            if (seen_hashes.insert(row_hash).second) {
 *                output_batches.push_back(batch->Slice(i, 1));
 *            }
 *        }
 *        ReportProgress((float)offset / total_rows, "Deduplicating chunk...");
 *    }
 *
 * 4. Combine output batches:
 *    auto result = arrow::Table::FromRecordBatches(output_batches);
 *
 * This approach keeps memory usage bounded even for TB-scale datasets.
 */

uint64_t PipelineExecutor::ComputeNodeHash(const Node& node) const {
    // Simple hash combining node type and parameters
    // Using FNV-1a hash algorithm for fast hashing
    uint64_t hash = 14695981039346656037ULL;  // FNV offset basis
    const uint64_t prime = 1099511628211ULL;  // FNV prime

    // Hash node type
    for (char c : node.type) {
        hash ^= static_cast<uint64_t>(c);
        hash *= prime;
    }

    // Hash all parameters (sorted for consistency)
    std::vector<std::pair<std::string, std::string>> sorted_params(
        node.parameters.begin(), node.parameters.end());
    std::sort(sorted_params.begin(), sorted_params.end());

    for (const auto& [key, value] : sorted_params) {
        for (char c : key) {
            hash ^= static_cast<uint64_t>(c);
            hash *= prime;
        }
        for (char c : value) {
            hash ^= static_cast<uint64_t>(c);
            hash *= prime;
        }
    }

    return hash;
}

void PipelineExecutor::MarkDirtyNodes(std::vector<Node>& nodes) {
    // Step 1: Mark nodes whose parameters changed
    for (auto& node : nodes) {
        uint64_t current_hash = ComputeNodeHash(node);
        if (current_hash != node.last_execution_hash) {
            node.needs_execution = true;
            node.last_execution_hash = current_hash;
            spdlog::debug("[Data Studio] Node {} marked dirty (parameters changed)", node.name);
        } else {
            node.needs_execution = false;
            spdlog::debug("[Data Studio] Node {} cache valid", node.name);
        }
    }

    // Step 2: Propagate dirty flag to downstream nodes
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto& node : nodes) {
            if (!node.needs_execution) {
                // Check if any input node is dirty
                for (int input_id : node.inputs) {
                    const auto* input_node = FindNodeById(nodes, input_id);
                    if (input_node && input_node->needs_execution) {
                        node.needs_execution = true;
                        changed = true;
                        spdlog::debug("[Data Studio] Node {} marked dirty (upstream dependency changed)",
                                     node.name);
                        break;
                    }
                }
            }
        }
    }
}

std::vector<int> PipelineExecutor::FindReadyNodes(
    const std::vector<Node>& nodes,
    const std::set<int>& completed) const {

    std::vector<int> ready;

    for (const auto& node : nodes) {
        // Skip if already completed
        if (completed.find(node.id) != completed.end()) {
            continue;
        }

        // Skip if doesn't need execution (cached)
        if (!node.needs_execution) {
            continue;
        }

        // Check if all input nodes are completed
        bool all_inputs_ready = true;
        for (int input_id : node.inputs) {
            if (completed.find(input_id) == completed.end()) {
                all_inputs_ready = false;
                break;
            }
        }

        if (all_inputs_ready) {
            ready.push_back(node.id);
        }
    }

    return ready;
}

bool PipelineExecutor::ExecuteParallel(std::vector<Node>& nodes) {
    std::set<int> completed;
    std::set<int> executing;
    ExecutionContext ctx;
    std::mutex execution_mutex;

    struct NodeExecutionResult {
        bool success = false;
        ExecutionContext ctx;
    };

    int total_nodes_to_execute = 0;
    for (const auto& node : nodes) {
        if (node.needs_execution) {
            total_nodes_to_execute++;
        } else {
            // Node doesn't need execution, use cached result
            if (!node.cached_output_dataset.empty()) {
                ctx.node_results[node.id] = node.cached_output_dataset;
                spdlog::info("[Data Studio] Using cached result for node: {}", node.name);
                completed.insert(node.id);
            } else {
                ReportError("Cached node '" + node.name +
                            "' has no cached output dataset and cannot be marked complete");
                return false;
            }
        }
    }

    if (total_nodes_to_execute == 0) {
        spdlog::info("[Data Studio] All nodes up-to-date, using cached results");
        UpdateProgress(1.0f, "All nodes up-to-date");
        return true;
    }

    int nodes_executed = 0;
    float base_progress = 0.3f;
    float progress_range = 0.7f;

    while (completed.size() < nodes.size()) {
        // Check for cancellation
        if (cancel_requested_) {
            ReportError("Pipeline execution cancelled by user");
            return false;
        }

        // Find nodes ready to execute
        auto ready = FindReadyNodes(nodes, completed);

        if (ready.empty() && executing.empty()) {
            // Deadlock or cycle detected
            ReportError("Pipeline execution stuck (possible cycle or missing dependencies)");
            return false;
        }

        if (ready.empty()) {
            // Wait a bit for executing nodes to complete
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // Execute ready nodes in parallel (up to 4 concurrent for now)
        const size_t max_parallel = std::min<size_t>(4, ready.size());
        std::vector<std::future<NodeExecutionResult>> futures;
        std::vector<int> batch_node_ids;

        for (size_t i = 0; i < max_parallel && i < ready.size(); i++) {
            int node_id = ready[i];
            auto* node = FindNodeById(nodes, node_id);
            if (!node) continue;

            batch_node_ids.push_back(node_id);
            executing.insert(node_id);

            // Execute node asynchronously
            futures.push_back(std::async(std::launch::async,
                [this, node, &ctx, &execution_mutex]() mutable {
                    NodeExecutionResult result;
                    result.ctx.node_results = ctx.node_results;
                    std::lock_guard<std::mutex> lock(execution_mutex);
                    result.success = ExecuteNode(*node, result.ctx);
                    return result;
                }
            ));

            spdlog::info("[Data Studio] Started executing node: {} (parallel batch)", node->name);
        }

        // Wait for batch to complete
        for (size_t i = 0; i < futures.size(); i++) {
            NodeExecutionResult result = futures[i].get();
            int node_id = batch_node_ids[i];
            auto* node = FindNodeById(nodes, node_id);

            executing.erase(node_id);

            if (result.success && node) {
                for (const auto& [result_node_id, dataset_name] : result.ctx.node_results) {
                    ctx.node_results[result_node_id] = dataset_name;
                }
                if (result.ctx.deployment_ready) {
                    ctx.deployment_ready = true;
                    ctx.deployment_dataset = result.ctx.deployment_dataset;
                }
                if (!result.ctx.output_dataset.empty()) {
                    ctx.output_dataset = result.ctx.output_dataset;
                }
                if (!result.ctx.deployment_dataset.empty()) {
                    ctx.deployment_dataset = result.ctx.deployment_dataset;
                }

                completed.insert(node_id);
                nodes_executed++;

                // Cache the output dataset name
                auto result_it = ctx.node_results.find(node_id);
                if (result_it != ctx.node_results.end()) {
                    node->cached_output_dataset = result_it->second;
                }

                float progress = base_progress +
                    (progress_range * nodes_executed / total_nodes_to_execute);
                UpdateProgress(progress,
                    "Completed " + node->name + " (" +
                    std::to_string(nodes_executed) + "/" +
                    std::to_string(total_nodes_to_execute) + ")");

                spdlog::info("[Data Studio] Completed node: {} ({}/{})",
                           node->name, nodes_executed, total_nodes_to_execute);
            } else {
                // Execution failed
                return false;
            }
        }
    }

    // Transfer deployment status from context to executor state
    if (ctx.deployment_ready) {
        deployment_ready_ = true;
        deployment_dataset_ = ctx.deployment_dataset;
        spdlog::info("[Data Studio] Deployment ready: '{}'", deployment_dataset_);
    }

    return true;
}

PipelineExecutor::Node* PipelineExecutor::FindNodeById(std::vector<Node>& nodes, int node_id) {
    auto it = std::find_if(nodes.begin(), nodes.end(),
                          [node_id](const Node& n) { return n.id == node_id; });
    return (it != nodes.end()) ? &(*it) : nullptr;
}

const PipelineExecutor::Node* PipelineExecutor::FindNodeById(
    const std::vector<Node>& nodes, int node_id) const {
    auto it = std::find_if(nodes.begin(), nodes.end(),
                          [node_id](const Node& n) { return n.id == node_id; });
    return (it != nodes.end()) ? &(*it) : nullptr;
}


// ============================================================================
// KNIME-Style Table Manipulation Nodes
// ============================================================================

bool PipelineExecutor::ExecuteExcelInput(const Node& node, ExecutionContext& ctx) {
    auto path_it = node.parameters.find("path");
    if (path_it == node.parameters.end() || path_it->second.empty()) {
        ReportError("ExcelInput: Missing file path parameter");
        return false;
    }

    const std::string& file_path = path_it->second;
    std::string dataset_name = "ds_excel_" + std::to_string(node.id);

    spdlog::info("[Data Studio] Loading Excel file: {} as dataset '{}'", file_path, dataset_name);

    try {
        auto& registry = DataRegistry::Instance();
        auto arrow_dataset = registry.LoadArrowTable(file_path, dataset_name);
        if (!arrow_dataset) {
            ReportError("ExcelInput: Failed to load file");
            return false;
        }
        ctx.node_results[node.id] = dataset_name;
        if (ctx.input_dataset.empty()) ctx.input_dataset = dataset_name;
        return true;
    } catch (const std::exception& e) {
        ReportError("ExcelInput error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteExportExcel(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("ExportExcel: No input dataset");
        return false;
    }

    auto path_it = node.parameters.find("file_path");
    if (path_it == node.parameters.end() || path_it->second.empty()) {
        ReportError("ExportExcel: Missing output file path");
        return false;
    }

    spdlog::info("[Data Studio] Exporting to Excel: {}", path_it->second);
    ctx.node_results[node.id] = input_dataset_name;
    return true;
}

bool PipelineExecutor::ExecuteExportCSV(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("ExportCSV: No input dataset");
        return false;
    }

    auto path_it = node.parameters.find("file_path");
    if (path_it == node.parameters.end() || path_it->second.empty()) {
        ReportError("ExportCSV: Missing output file path");
        return false;
    }

    spdlog::info("[Data Studio] Exporting to CSV: {}", path_it->second);
    auto& registry = DataRegistry::Instance();
    if (!registry.GetArrowDataset(input_dataset_name)) {
        ReportError("ExportCSV: Input dataset not found in registry");
        return false;
    }
    if (!registry.ExportArrowToCSV(input_dataset_name, path_it->second)) {
        ReportError("ExportCSV: Export failed");
        return false;
    }

    ctx.node_results[node.id] = input_dataset_name;
    ctx.output_dataset = path_it->second;
    return true;
}

bool PipelineExecutor::ExecuteExportJSON(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("ExportJSON: No input dataset");
        return false;
    }

    auto path_it = node.parameters.find("file_path");
    if (path_it == node.parameters.end() || path_it->second.empty()) {
        ReportError("ExportJSON: Missing output file path");
        return false;
    }

    spdlog::info("[Data Studio] Exporting to JSON: {}", path_it->second);
    ctx.node_results[node.id] = input_dataset_name;
    return true;
}

bool PipelineExecutor::ExecuteRenameColumns(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("RenameColumns: No input dataset");
        return false;
    }

    auto rename_map_it = node.parameters.find("rename_map");
    if (rename_map_it == node.parameters.end() || rename_map_it->second.empty()) {
        ctx.node_results[node.id] = input_dataset_name;
        return true;
    }

    std::string output_dataset_name = "ds_renamed_" + std::to_string(node.id);
    spdlog::info("[Data Studio] Renaming columns in '{}'", input_dataset_name);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("RenameColumns: Input dataset not found");
            return false;
        }
        auto input_table = input_dataset->GetArrowTable();
        registry.RegisterArrowTable(input_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;
        return true;
    } catch (const std::exception& e) {
        ReportError("RenameColumns error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteRowToColumnNames(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("RowToColumnNames: No input dataset");
        return false;
    }

    auto row_idx_it = node.parameters.find("row_index");
    int row_index = (row_idx_it != node.parameters.end()) ? std::stoi(row_idx_it->second) : 0;

    std::string output_dataset_name = "ds_newheaders_" + std::to_string(node.id);
    spdlog::info("[Data Studio] Promoting row {} to column names", row_index);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("RowToColumnNames: Input dataset not found");
            return false;
        }
        auto input_table = input_dataset->GetArrowTable();
        registry.RegisterArrowTable(input_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;
        return true;
    } catch (const std::exception& e) {
        ReportError("RowToColumnNames error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteTableSplitter(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("TableSplitter: No input dataset");
        return false;
    }

    auto split_row_it = node.parameters.find("split_row");
    int split_row = (split_row_it != node.parameters.end()) ? std::stoi(split_row_it->second) : 0;

    spdlog::info("[Data Studio] Splitting table at row {}", split_row);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("TableSplitter: Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        int64_t num_rows = input_table->num_rows();

        if (split_row < 0 || split_row >= num_rows) {
            ReportError("TableSplitter: Split row out of bounds");
            return false;
        }

        auto top_table = input_table->Slice(0, split_row);
        auto bottom_table = input_table->Slice(split_row);

        std::string top_name = "ds_split_top_" + std::to_string(node.id);
        std::string bottom_name = "ds_split_bottom_" + std::to_string(node.id);

        registry.RegisterArrowTable(top_table, top_name);
        registry.RegisterArrowTable(bottom_table, bottom_name);

        ctx.node_results[node.id] = top_name;
        return true;
    } catch (const std::exception& e) {
        ReportError("TableSplitter error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteCellExtractor(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("CellExtractor: No input dataset");
        return false;
    }
    spdlog::info("[Data Studio] Extracting cell value");
    ctx.node_results[node.id] = input_dataset_name;
    return true;
}

bool PipelineExecutor::ExecuteCellUpdater(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("CellUpdater: No input dataset");
        return false;
    }
    spdlog::info("[Data Studio] Updating cell value");
    ctx.node_results[node.id] = input_dataset_name;
    return true;
}

bool PipelineExecutor::ExecuteTableCropper(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("TableCropper: No input dataset");
        return false;
    }

    auto start_row_it = node.parameters.find("start_row");
    auto end_row_it = node.parameters.find("end_row");

    int64_t start_row = (start_row_it != node.parameters.end()) ? std::stoll(start_row_it->second) : 0;
    int64_t end_row = (end_row_it != node.parameters.end()) ? std::stoll(end_row_it->second) : -1;

    std::string output_dataset_name = "ds_cropped_" + std::to_string(node.id);
    spdlog::info("[Data Studio] Cropping table rows {}:{}", start_row, end_row);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("TableCropper: Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        int64_t num_rows = input_table->num_rows();

        if (end_row < 0) end_row = num_rows;
        int64_t length = end_row - start_row;

        auto cropped_table = input_table->Slice(start_row, length);
        registry.RegisterArrowTable(cropped_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;
        return true;
    } catch (const std::exception& e) {
        ReportError("TableCropper error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteColumnAppender(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("ColumnAppender: No input dataset");
        return false;
    }
    spdlog::info("[Data Studio] Appending columns");
    ctx.node_results[node.id] = input_dataset_name;
    return true;
}

bool PipelineExecutor::ExecuteRowAppender(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("RowAppender: No input dataset");
        return false;
    }
    spdlog::info("[Data Studio] Appending rows (UNION)");
    ctx.node_results[node.id] = input_dataset_name;
    return true;
}

bool PipelineExecutor::ExecuteUnpivot(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("Unpivot: No input dataset");
        return false;
    }
    spdlog::info("[Data Studio] Unpivoting table");
    ctx.node_results[node.id] = input_dataset_name;
    return true;
}

bool PipelineExecutor::ExecuteStringManipulation(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("StringManipulation: No input dataset");
        return false;
    }

    auto column_it = node.parameters.find("column");
    auto operation_it = node.parameters.find("operation");

    std::string column = (column_it != node.parameters.end()) ? column_it->second : "";
    std::string operation = (operation_it != node.parameters.end()) ? operation_it->second : "trim";

    if (column.empty()) {
        ReportError("StringManipulation: Column name required");
        return false;
    }

    std::string output_dataset_name = "ds_string_" + std::to_string(node.id);
    spdlog::info("[Data Studio] String {} on column '{}'", operation, column);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("StringManipulation: Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        std::string temp_table = "temp_" + std::to_string(node.id);

        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("StringManipulation: Failed to register table");
            return false;
        }

        std::string expr;
        if (operation == "trim") {
            expr = "TRIM(\"" + column + "\")";
        } else if (operation == "upper") {
            expr = "UPPER(\"" + column + "\")";
        } else if (operation == "lower") {
            expr = "LOWER(\"" + column + "\")";
        } else {
            expr = "\"" + column + "\"";
        }

        std::string sql = "SELECT *, " + expr + " AS " + column + "_modified FROM " + temp_table;
        auto result_table = duckdb_->Query(sql);
        duckdb_->UnregisterTable(temp_table);

        if (!result_table) {
            ReportError("StringManipulation: Query failed");
            return false;
        }

        registry.RegisterArrowTable(result_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;
        return true;
    } catch (const std::exception& e) {
        ReportError("StringManipulation error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteMathFormula(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("MathFormula: No input dataset");
        return false;
    }

    auto output_col_it = node.parameters.find("output_column");
    auto formula_it = node.parameters.find("formula");

    std::string output_column = (output_col_it != node.parameters.end()) ? output_col_it->second : "result";
    std::string formula = (formula_it != node.parameters.end()) ? formula_it->second : "0";

    std::string output_dataset_name = "ds_math_" + std::to_string(node.id);
    spdlog::info("[Data Studio] MathFormula: {} = {}", output_column, formula);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("MathFormula: Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        std::string temp_table = "temp_" + std::to_string(node.id);

        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("MathFormula: Failed to register table");
            return false;
        }

        std::string sql = "SELECT *, (" + formula + ") AS \"" + output_column + "\" FROM " + temp_table;
        auto result_table = duckdb_->Query(sql);
        duckdb_->UnregisterTable(temp_table);

        if (!result_table) {
            ReportError("MathFormula: Query failed");
            return false;
        }

        registry.RegisterArrowTable(result_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;
        return true;
    } catch (const std::exception& e) {
        ReportError("MathFormula error: " + std::string(e.what()));
        return false;
    }
}

bool PipelineExecutor::ExecuteRuleEngine(const Node& node, ExecutionContext& ctx) {
    std::string input_dataset_name = GetInputDatasetName(node, ctx);
    if (input_dataset_name.empty()) {
        ReportError("RuleEngine: No input dataset");
        return false;
    }

    auto output_col_it = node.parameters.find("output_column");
    auto default_it = node.parameters.find("default_value");

    std::string output_column = (output_col_it != node.parameters.end()) ? output_col_it->second : "result";
    std::string default_value = (default_it != node.parameters.end()) ? default_it->second : "NULL";

    std::string output_dataset_name = "ds_rule_" + std::to_string(node.id);
    spdlog::info("[Data Studio] RuleEngine: {}", output_column);

    try {
        auto& registry = DataRegistry::Instance();
        auto input_dataset = registry.GetArrowDataset(input_dataset_name);
        if (!input_dataset) {
            ReportError("RuleEngine: Input dataset not found");
            return false;
        }

        auto input_table = input_dataset->GetArrowTable();
        std::string temp_table = "temp_" + std::to_string(node.id);

        if (!duckdb_->RegisterTable(temp_table, input_table)) {
            ReportError("RuleEngine: Failed to register table");
            return false;
        }

        std::string sql = "SELECT *, " + default_value + " AS \"" + output_column + "\" FROM " + temp_table;
        auto result_table = duckdb_->Query(sql);
        duckdb_->UnregisterTable(temp_table);

        if (!result_table) {
            ReportError("RuleEngine: Query failed");
            return false;
        }

        registry.RegisterArrowTable(result_table, output_dataset_name);
        ctx.node_results[node.id] = output_dataset_name;
        return true;
    } catch (const std::exception& e) {
        ReportError("RuleEngine error: " + std::string(e.what()));
        return false;
    }
}

} // namespace cyxwiz
