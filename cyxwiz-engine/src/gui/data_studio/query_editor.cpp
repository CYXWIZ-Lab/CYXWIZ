#include "query_editor.h"
#include "../../core/duckdb_connector.h"
#include <spdlog/spdlog.h>
#include <cstring>

namespace cyxwiz {

QueryEditor::QueryEditor()
    : query_running_(false)
{
    std::memset(query_buffer_, 0, sizeof(query_buffer_));
    std::strcpy(query_buffer_, "SELECT * FROM dataset LIMIT 100");

    spdlog::info("[Data Studio] QueryEditor initialized");
}

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

void QueryEditor::SetActiveDataset(const std::string& dataset_name) {
    // TODO: Register dataset with DuckDB
    spdlog::info("[Data Studio] Set active dataset: {}", dataset_name);
}

bool QueryEditor::ExecuteQuery() {
    current_query_ = std::string(query_buffer_);
    if (current_query_.empty()) {
        last_error_ = "Query is empty";
        return false;
    }

    spdlog::info("[Data Studio] Executing query: {}", current_query_);

    // TODO: Phase 1 Week 2 - Actually execute via DuckDB
    // For now, just show a placeholder result
    last_result_.column_names = {"Column1", "Column2", "Column3"};
    last_result_.rows = {
        {"Value1", "Value2", "Value3"},
        {"Value4", "Value5", "Value6"},
        {"Value7", "Value8", "Value9"}
    };
    last_result_.total_rows = 3;
    last_result_.execution_time_ms = 1.5;

    last_error_ = "";

    // Add to history
    query_history_.push_back(current_query_);
    if (query_history_.size() > max_history_size_) {
        query_history_.erase(query_history_.begin());
    }

    return true;
}

} // namespace cyxwiz
