#include "query_editor.h"
#include <cstring>
#include <limits>

namespace cyxwiz {

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

    if (last_result_.column_names.size() >
        static_cast<size_t>(std::numeric_limits<int>::max())) {
        ImGui::TextDisabled("Result has too many columns to display.");
        return;
    }
    const int column_count = static_cast<int>(last_result_.column_names.size());

    // Begin table
    if (ImGui::BeginTable("##results", column_count,
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
            for (int col = 0;
                 col < column_count && static_cast<size_t>(col) < row.size();
                 ++col) {
                ImGui::TableSetColumnIndex(col);
                ImGui::TextUnformatted(row[static_cast<size_t>(col)].c_str());
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


} // namespace cyxwiz
