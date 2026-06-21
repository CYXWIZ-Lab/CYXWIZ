#include "query_editor.h"
#include "../../core/duckdb_connector.h"
#include "../../core/data_registry.h"
#include "../../core/arrow_dataset.h"
#include <spdlog/spdlog.h>
#include <cstring>
#include <chrono>

namespace cyxwiz {

QueryEditor::QueryEditor()
    : query_running_(false)
    , duckdb_(std::make_unique<DuckDBConnector>())
{
    std::memset(query_buffer_, 0, sizeof(query_buffer_));
    std::strcpy(query_buffer_, "SELECT * FROM dataset LIMIT 100");

    spdlog::info("[Data Studio] QueryEditor initialized with DuckDB");
}

QueryEditor::~QueryEditor() = default;

bool QueryEditor::SaveResultAsDataset(const std::string& dataset_name) {
    if (current_query_.empty()) {
        last_error_ = "No query to save";
        spdlog::error("[Data Studio] QueryEditor: {}", last_error_);
        return false;
    }

    spdlog::info("[Data Studio] QueryEditor: Saving query result as dataset '{}'", dataset_name);

    try {
        // Re-execute the query to get the full result (not limited to 1000 rows)
        auto result_table = duckdb_->Query(current_query_);

        if (!result_table) {
            last_error_ = "Failed to re-execute query: " + duckdb_->GetLastError();
            spdlog::error("[Data Studio] QueryEditor: {}", last_error_);
            return false;
        }

        // Register the result as a new Arrow dataset
        auto& registry = DataRegistry::Instance();
        registry.RegisterArrowTable(result_table, dataset_name);

        spdlog::info("[Data Studio] QueryEditor: Saved query result as dataset '{}' ({} rows)",
                     dataset_name, result_table->num_rows());

        return true;

    } catch (const std::exception& e) {
        last_error_ = std::string("Failed to save dataset: ") + e.what();
        spdlog::error("[Data Studio] QueryEditor: {}", last_error_);
        return false;
    }
}

void QueryEditor::SetActiveDataset(const std::string& dataset_name) {
    current_dataset_ = dataset_name;
    spdlog::info("[Data Studio] QueryEditor: Setting active dataset: {}", dataset_name);

    if (dataset_name.empty()) {
        return;
    }

    try {
        // Get Arrow dataset from DataRegistry
        auto& registry = DataRegistry::Instance();
        auto arrow_dataset = registry.GetArrowDataset(dataset_name);

        if (!arrow_dataset) {
            spdlog::warn("[Data Studio] QueryEditor: dataset not found in registry");
            last_error_ = "Dataset not found: " + dataset_name;
            return;
        }

        auto arrow_table = arrow_dataset->GetArrowTable();
        if (!arrow_table) {
            spdlog::warn("[Data Studio] QueryEditor: dataset has no Arrow table");
            last_error_ = "Dataset has no Arrow table";
            return;
        }

        // Unregister previous dataset if exists
        if (duckdb_->HasTable("dataset")) {
            duckdb_->UnregisterTable("dataset");
        }

        // Register Arrow table with DuckDB as "dataset"
        if (!duckdb_->RegisterTable("dataset", arrow_table)) {
            last_error_ = "Failed to register table: " + duckdb_->GetLastError();
            spdlog::error("[Data Studio] QueryEditor: {}", last_error_);
            return;
        }

        // Get table schema for display
        auto schema = duckdb_->GetTableSchema("dataset");
        int64_t row_count = duckdb_->GetRowCount("dataset");

        spdlog::info("[Data Studio] QueryEditor: Registered table '{}' ({} rows, {} columns)",
                     dataset_name, row_count, schema.size());

        last_error_ = "";

    } catch (const std::exception& e) {
        last_error_ = std::string("Failed to register dataset: ") + e.what();
        spdlog::error("[Data Studio] QueryEditor: {}", last_error_);
    }
}

bool QueryEditor::ExecuteQuery() {
    current_query_ = std::string(query_buffer_);
    if (current_query_.empty()) {
        last_error_ = "Query is empty";
        return false;
    }

    if (current_dataset_.empty()) {
        last_error_ = "No dataset selected. Please select a dataset first.";
        return false;
    }

    spdlog::info("[Data Studio] QueryEditor: Executing query: {}", current_query_);

    // Clear previous results
    last_result_.column_names.clear();
    last_result_.rows.clear();
    last_result_.total_rows = 0;
    last_error_ = "";

    query_running_ = true;
    auto start = std::chrono::high_resolution_clock::now();

    try {
        // Execute SQL query via DuckDB
        auto result_table = duckdb_->Query(current_query_);

        if (!result_table) {
            last_error_ = "Query failed: " + duckdb_->GetLastError();
            spdlog::error("[Data Studio] QueryEditor: {}", last_error_);
            query_running_ = false;
            return false;
        }

        // Extract column names
        auto schema = result_table->schema();
        for (int i = 0; i < schema->num_fields(); i++) {
            last_result_.column_names.push_back(schema->field(i)->name());
        }

        // Extract rows (limit to 1000 for display)
        const int64_t max_display_rows = 1000;
        int64_t num_rows = std::min(result_table->num_rows(), max_display_rows);

        for (int64_t row_idx = 0; row_idx < num_rows; row_idx++) {
            std::vector<std::string> row_data;

            for (int col_idx = 0; col_idx < result_table->num_columns(); col_idx++) {
                auto column = result_table->column(col_idx);

                // Find the chunk containing this row
                int64_t chunk_offset = 0;
                std::shared_ptr<arrow::Array> chunk;
                for (int chunk_idx = 0; chunk_idx < column->num_chunks(); chunk_idx++) {
                    chunk = column->chunk(chunk_idx);
                    if (row_idx < chunk_offset + chunk->length()) {
                        break;
                    }
                    chunk_offset += chunk->length();
                }

                int64_t row_in_chunk = row_idx - chunk_offset;

                // Convert value to string
                std::string value_str;
                if (chunk->IsNull(row_in_chunk)) {
                    value_str = "NULL";
                } else {
                    auto type_id = chunk->type_id();

                    if (type_id == arrow::Type::DOUBLE) {
                        auto typed_array = std::static_pointer_cast<arrow::DoubleArray>(chunk);
                        value_str = std::to_string(typed_array->Value(row_in_chunk));
                    } else if (type_id == arrow::Type::FLOAT) {
                        auto typed_array = std::static_pointer_cast<arrow::FloatArray>(chunk);
                        value_str = std::to_string(typed_array->Value(row_in_chunk));
                    } else if (type_id == arrow::Type::INT64) {
                        auto typed_array = std::static_pointer_cast<arrow::Int64Array>(chunk);
                        value_str = std::to_string(typed_array->Value(row_in_chunk));
                    } else if (type_id == arrow::Type::INT32) {
                        auto typed_array = std::static_pointer_cast<arrow::Int32Array>(chunk);
                        value_str = std::to_string(typed_array->Value(row_in_chunk));
                    } else if (type_id == arrow::Type::STRING) {
                        auto typed_array = std::static_pointer_cast<arrow::StringArray>(chunk);
                        value_str = typed_array->GetString(row_in_chunk);
                    } else if (type_id == arrow::Type::BOOL) {
                        auto typed_array = std::static_pointer_cast<arrow::BooleanArray>(chunk);
                        value_str = typed_array->Value(row_in_chunk) ? "true" : "false";
                    } else {
                        // Fallback for other types
                        value_str = chunk->ToString();
                    }
                }

                row_data.push_back(value_str);
            }

            last_result_.rows.push_back(row_data);
        }

        last_result_.total_rows = result_table->num_rows();

        auto end = std::chrono::high_resolution_clock::now();
        last_result_.execution_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        spdlog::info("[Data Studio] QueryEditor: Query executed successfully - {} rows, {:.2f} ms",
                     last_result_.total_rows, last_result_.execution_time_ms);

        // Add to history
        query_history_.push_back(current_query_);
        if (query_history_.size() > static_cast<size_t>(max_history_size_)) {
            query_history_.erase(query_history_.begin());
        }

        query_running_ = false;
        return true;

    } catch (const std::exception& e) {
        last_error_ = std::string("Query execution failed: ") + e.what();
        spdlog::error("[Data Studio] QueryEditor: {}", last_error_);
        query_running_ = false;
        return false;
    }
}

} // namespace cyxwiz
