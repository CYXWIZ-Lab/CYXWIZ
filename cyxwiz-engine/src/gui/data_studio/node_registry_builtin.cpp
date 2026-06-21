#include "node_registry.h"
#include <spdlog/spdlog.h>

namespace cyxwiz {

void NodeRegistry::RegisterBuiltInNodes() {
    // ===== Data Sources =====

    NodeType file_input;
    file_input.type_id = "FileInput";
    file_input.display_name = "File Input";
    file_input.category = "Data Sources";
    file_input.description = "Load data from a file (CSV or Parquet)";
    file_input.output_ports = {"data"};
    file_input.parameters = {
        {"path", "string"},
        {"format", "enum:csv,parquet"}
    };
    RegisterNodeType(file_input);

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
        {"on_column", "string"}
    };
    RegisterNodeType(join);

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

    // ===== Output =====

    NodeType save_dataset;
    save_dataset.type_id = "SaveDataset";
    save_dataset.display_name = "Save Dataset";
    save_dataset.category = "Output";
    save_dataset.description = "Save dataset to file";
    save_dataset.input_ports = {"data"};
    save_dataset.parameters = {
        {"path", "string"},
        {"format", "enum:csv,parquet"}
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