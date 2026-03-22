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

void QueryEditor::Render() {
    RenderQueryEditor();
    ImGui::Separator();
    RenderResultsTable();
}

void QueryEditor::RenderQueryEditor() {
    ImGui::Text("SQL Query Editor");
    ImGui::Separator();

    // Multi-line text input for SQL query
    ImGui::InputTextMultiline("##query", query_buffer_, sizeof(query_buffer_),
                              ImVec2(-1, 150), ImGuiInputTextFlags_AllowTabInput);

    // Execute button
    if (ImGui::Button("Execute Query")) {
        ExecuteQuery();
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear")) {
        std::memset(query_buffer_, 0, sizeof(query_buffer_));
    }
    ImGui::SameLine();
    if (ImGui::Button("Example Queries")) {
        ImGui::OpenPopup("Examples");
    }

    // Example queries popup
    if (ImGui::BeginPopup("Examples")) {
        if (ImGui::Selectable("Select all rows")) {
            std::strcpy(query_buffer_, "SELECT * FROM dataset LIMIT 100");
        }
        if (ImGui::Selectable("Count rows")) {
            std::strcpy(query_buffer_, "SELECT COUNT(*) as total FROM dataset");
        }
        if (ImGui::Selectable("Group by column")) {
            std::strcpy(query_buffer_, "SELECT column_name, COUNT(*) as count FROM dataset GROUP BY column_name");
        }
        if (ImGui::Selectable("Filter rows")) {
            std::strcpy(query_buffer_, "SELECT * FROM dataset WHERE column_name > 0");
        }
        ImGui::EndPopup();
    }

    // Display last error
    if (!last_error_.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Error: %s", last_error_.c_str());
    }

    // Display execution time
    if (last_result_.total_rows > 0) {
        ImGui::Text("Rows: %zu | Execution time: %.2f ms",
                   last_result_.total_rows, last_result_.execution_time_ms);
    }
}

void QueryEditor::RenderResultsTable() {
    if (last_result_.column_names.empty()) {
        ImGui::TextDisabled("No results. Execute a query to see results.");
        return;
    }

    ImGui::Text("Query Results");

    // Save as Dataset button
    if (ImGui::Button("Save as Dataset")) {
        ImGui::OpenPopup("SaveDataset");
    }
    ImGui::SameLine();
    ImGui::Text("(%zu rows displayed)", last_result_.rows.size());
    if (last_result_.total_rows > last_result_.rows.size()) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f),
                          "(showing %zu of %zu rows)",
                          last_result_.rows.size(), last_result_.total_rows);
    }

    // Save dataset dialog
    static char dataset_name_buffer[256] = "query_result";
    if (ImGui::BeginPopupModal("SaveDataset", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Save query result as new dataset");
        ImGui::Separator();
        ImGui::InputText("Dataset Name", dataset_name_buffer, sizeof(dataset_name_buffer));

        if (ImGui::Button("Save")) {
            std::string name = dataset_name_buffer;
            if (!name.empty()) {
                SaveResultAsDataset(name);
                ImGui::CloseCurrentPopup();
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    ImGui::Separator();

    // Begin table
    if (ImGui::BeginTable("##results", last_result_.column_names.size(),
                         ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY |
                         ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable)) {

        // Header row
        for (const auto& col_name : last_result_.column_names) {
            ImGui::TableSetupColumn(col_name.c_str());
        }
        ImGui::TableHeadersRow();

        // Data rows
        for (const auto& row : last_result_.rows) {
            ImGui::TableNextRow();
            for (size_t col = 0; col < row.size(); col++) {
                ImGui::TableSetColumnIndex(col);
                ImGui::TextUnformatted(row[col].c_str());
            }
        }

        ImGui::EndTable();
    }
}

void QueryEditor::RenderQueryHistory() {
    // TODO: Implement query history display
}

void QueryEditor::RenderExampleQueries() {
    // TODO: Implement example queries panel
}

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
