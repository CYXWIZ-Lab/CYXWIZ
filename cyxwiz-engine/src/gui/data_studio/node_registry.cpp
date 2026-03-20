#include "node_registry.h"
#include <spdlog/spdlog.h>
#include <algorithm>

namespace cyxwiz {

NodeRegistry& NodeRegistry::Instance() {
    static NodeRegistry instance;
    return instance;
}

NodeRegistry::NodeRegistry() {
    RegisterBuiltInNodes();
    spdlog::info("[Data Studio] NodeRegistry initialized with {} node types",
                 node_types_.size());
}

void NodeRegistry::RegisterNodeType(const NodeType& node_type) {
    // Check for duplicates
    if (HasNodeType(node_type.type_id)) {
        spdlog::warn("[Data Studio] Node type already registered: {}", node_type.type_id);
        return;
    }

    node_types_.push_back(node_type);
    spdlog::debug("[Data Studio] Registered node type: {} ({})",
                  node_type.display_name, node_type.type_id);
}

const std::vector<NodeRegistry::NodeType>& NodeRegistry::GetAllNodeTypes() const {
    return node_types_;
}

std::vector<NodeRegistry::NodeType> NodeRegistry::GetNodeTypesByCategory(
    const std::string& category) const {

    std::vector<NodeType> result;
    for (const auto& node_type : node_types_) {
        if (node_type.category == category) {
            result.push_back(node_type);
        }
    }
    return result;
}

std::vector<std::string> NodeRegistry::GetCategories() const {
    std::vector<std::string> categories;
    for (const auto& node_type : node_types_) {
        if (std::find(categories.begin(), categories.end(), node_type.category) == categories.end()) {
            categories.push_back(node_type.category);
        }
    }
    return categories;
}

const NodeRegistry::NodeType* NodeRegistry::FindNodeType(const std::string& type_id) const {
    auto it = std::find_if(node_types_.begin(), node_types_.end(),
                          [&type_id](const NodeType& nt) { return nt.type_id == type_id; });

    return (it != node_types_.end()) ? &(*it) : nullptr;
}

bool NodeRegistry::HasNodeType(const std::string& type_id) const {
    return FindNodeType(type_id) != nullptr;
}

void NodeRegistry::RegisterBuiltInNodes() {
    // ===== Data Sources =====

    NodeType file_input;
    file_input.type_id = "FileInput";
    file_input.display_name = "File Input";
    file_input.category = "Data Sources";
    file_input.description = "Load data from a file (CSV, Parquet, JSON)";
    file_input.output_ports = {"data"};
    file_input.parameters = {
        {"path", "string"},
        {"format", "enum:csv,parquet,json"}
    };
    RegisterNodeType(file_input);

    NodeType arrow_dataset;
    arrow_dataset.type_id = "ArrowDataset";
    arrow_dataset.display_name = "Arrow Dataset";
    arrow_dataset.category = "Data Sources";
    arrow_dataset.description = "Load existing Arrow dataset from DataRegistry";
    arrow_dataset.output_ports = {"data"};
    arrow_dataset.parameters = {
        {"dataset_name", "string"}
    };
    RegisterNodeType(arrow_dataset);

    // ===== Transformations =====

    NodeType filter_rows;
    filter_rows.type_id = "FilterRows";
    filter_rows.display_name = "Filter Rows";
    filter_rows.category = "Transformations";
    filter_rows.description = "Filter rows based on condition (SQL WHERE)";
    filter_rows.input_ports = {"data"};
    filter_rows.output_ports = {"data"};
    filter_rows.parameters = {
        {"condition", "string"}
    };
    RegisterNodeType(filter_rows);

    NodeType select_columns;
    select_columns.type_id = "SelectColumns";
    select_columns.display_name = "Select Columns";
    select_columns.category = "Transformations";
    select_columns.description = "Select specific columns to keep";
    select_columns.input_ports = {"data"};
    select_columns.output_ports = {"data"};
    select_columns.parameters = {
        {"columns", "string"}  // Comma-separated list
    };
    RegisterNodeType(select_columns);

    NodeType join;
    join.type_id = "Join";
    join.display_name = "Join Tables";
    join.category = "Transformations";
    join.description = "Join two datasets on a key column";
    join.input_ports = {"left", "right"};
    join.output_ports = {"data"};
    join.parameters = {
        {"join_type", "enum:inner,left,right,outer"},
        {"left_key", "string"},
        {"right_key", "string"}
    };
    RegisterNodeType(join);

    NodeType aggregate;
    aggregate.type_id = "Aggregate";
    aggregate.display_name = "Aggregate";
    aggregate.category = "Transformations";
    aggregate.description = "Group by and aggregate (SQL GROUP BY)";
    aggregate.input_ports = {"data"};
    aggregate.output_ports = {"data"};
    aggregate.parameters = {
        {"group_by", "string"},
        {"aggregations", "string"}  // e.g., "COUNT(*), SUM(value)"
    };
    RegisterNodeType(aggregate);

    // ===== Data Quality =====

    NodeType remove_duplicates;
    remove_duplicates.type_id = "RemoveDuplicates";
    remove_duplicates.display_name = "Remove Duplicates";
    remove_duplicates.category = "Data Quality";
    remove_duplicates.description = "Remove duplicate rows (SQL DISTINCT)";
    remove_duplicates.input_ports = {"data"};
    remove_duplicates.output_ports = {"data"};
    remove_duplicates.parameters = {
        {"columns", "string"}  // Columns to check for duplicates
    };
    RegisterNodeType(remove_duplicates);

    NodeType fill_missing;
    fill_missing.type_id = "FillMissing";
    fill_missing.display_name = "Fill Missing Values";
    fill_missing.category = "Data Quality";
    fill_missing.description = "Fill null values with a constant or strategy";
    fill_missing.input_ports = {"data"};
    fill_missing.output_ports = {"data"};
    fill_missing.parameters = {
        {"strategy", "enum:constant,mean,median,mode"},
        {"value", "string"}
    };
    RegisterNodeType(fill_missing);

    NodeType detect_outliers;
    detect_outliers.type_id = "DetectOutliers";
    detect_outliers.display_name = "Detect Outliers";
    detect_outliers.category = "Data Quality";
    detect_outliers.description = "Mark or remove outliers using IQR or Z-score";
    detect_outliers.input_ports = {"data"};
    detect_outliers.output_ports = {"data", "outliers"};
    detect_outliers.parameters = {
        {"method", "enum:iqr,zscore"},
        {"action", "enum:mark,remove"}
    };
    RegisterNodeType(detect_outliers);

    // ===== Output =====

    NodeType save_dataset;
    save_dataset.type_id = "SaveDataset";
    save_dataset.display_name = "Save Dataset";
    save_dataset.category = "Output";
    save_dataset.description = "Save dataset to file";
    save_dataset.input_ports = {"data"};
    save_dataset.parameters = {
        {"path", "string"},
        {"format", "enum:csv,parquet,json,arrow"}
    };
    RegisterNodeType(save_dataset);

    NodeType export_csv;
    export_csv.type_id = "ExportCSV";
    export_csv.display_name = "Export CSV";
    export_csv.category = "Output";
    export_csv.description = "Export to CSV file";
    export_csv.input_ports = {"data"};
    export_csv.parameters = {
        {"path", "string"},
        {"delimiter", "string"},
        {"header", "bool"}
    };
    RegisterNodeType(export_csv);

    spdlog::info("[Data Studio] Registered {} built-in node types", node_types_.size());
}

} // namespace cyxwiz
