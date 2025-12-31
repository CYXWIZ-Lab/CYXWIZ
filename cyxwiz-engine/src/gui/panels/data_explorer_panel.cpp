#include "data_explorer_panel.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <nfd.h>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <fstream>

namespace cyxwiz {

// Example queries for the dropdown
const std::vector<DataExplorerPanel::ExampleQuery> DataExplorerPanel::example_queries_ = {
    {"Select All", "SELECT * FROM 'file.csv'", "Load entire file"},
    {"Preview 100 Rows", "SELECT * FROM 'file.parquet' LIMIT 100", "Quick preview"},
    {"Filter by Condition", "SELECT * FROM 'data.csv' WHERE column > 100", "Filter rows"},
    {"Aggregate Stats", "SELECT COUNT(*), AVG(value), MIN(value), MAX(value) FROM 'data.parquet'", "Summary statistics"},
    {"Group By", "SELECT category, COUNT(*) as cnt FROM 'sales.csv' GROUP BY category ORDER BY cnt DESC", "Aggregate by category"},
    {"Join Files", "SELECT a.*, b.label FROM 'features.parquet' a JOIN 'labels.parquet' b ON a.id = b.id", "Join datasets"},
    {"Sample 10%", "SELECT * FROM 'data.parquet' USING SAMPLE 10%", "Random sampling"},
    {"Distinct Values", "SELECT DISTINCT category FROM 'data.csv' ORDER BY category", "Unique values"},
};

DataExplorerPanel::DataExplorerPanel()
    : Panel("Data Explorer", false) {

    // Check DuckDB availability
    duckdb_available_ = DataLoader::IsAvailable();

    if (duckdb_available_) {
        try {
            data_loader_ = std::make_unique<DataLoader>();
            spdlog::info("DataExplorerPanel: DuckDB {} initialized", DataLoader::GetVersion());
        } catch (const std::exception& e) {
            spdlog::error("DataExplorerPanel: Failed to initialize DataLoader: {}", e.what());
            duckdb_available_ = false;
        }
    } else {
        spdlog::warn("DataExplorerPanel: DuckDB not available");
    }

    // Set default query
    std::strcpy(query_buffer_, "SELECT * FROM 'file.csv' LIMIT 100");

    // Get current directory
    current_directory_ = std::filesystem::current_path().string();
}

DataExplorerPanel::~DataExplorerPanel() = default;

void DataExplorerPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(1000, 700), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(name_.c_str(), &visible_, ImGuiWindowFlags_MenuBar)) {
        focused_ = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);

        // Menu bar
        if (ImGui::BeginMenuBar()) {
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem(ICON_FA_FOLDER_OPEN " Open File...", "Ctrl+O")) {
                    OpenFileDialog();
                }
                ImGui::Separator();
                if (ImGui::MenuItem(ICON_FA_FILE_EXPORT " Export Results...", nullptr, false, current_result_.success)) {
                    ExportResultsToCSV();
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Query")) {
                if (ImGui::MenuItem(ICON_FA_PLAY " Execute", "Ctrl+Enter", false, duckdb_available_)) {
                    ExecuteCurrentQuery();
                }
                ImGui::Separator();
                if (ImGui::MenuItem(ICON_FA_CLOCK_ROTATE_LEFT " History...")) {
                    show_history_popup_ = true;
                }
                if (ImGui::MenuItem(ICON_FA_LIGHTBULB " Examples...")) {
                    show_examples_popup_ = true;
                }
                ImGui::EndMenu();
            }
            ImGui::EndMenuBar();
        }

        if (!duckdb_available_) {
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f),
                ICON_FA_TRIANGLE_EXCLAMATION " DuckDB not available. Build with CYXWIZ_HAS_DUCKDB=ON");
            ImGui::End();
            return;
        }

        // Main layout with splitters
        float available_width = ImGui::GetContentRegionAvail().x;
        float available_height = ImGui::GetContentRegionAvail().y - 25.0f; // Reserve for status bar

        // Left pane (File Browser)
        ImGui::BeginChild("FileBrowser", ImVec2(left_pane_width_, available_height), true);
        RenderFileBrowserPane();
        ImGui::EndChild();

        // Splitter
        ImGui::SameLine();
        ImGui::InvisibleButton("##VSplitter", ImVec2(4.0f, available_height));
        if (ImGui::IsItemActive()) {
            left_pane_width_ += ImGui::GetIO().MouseDelta.x;
            left_pane_width_ = std::clamp(left_pane_width_, 150.0f, 400.0f);
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
        }

        ImGui::SameLine();

        // Right side (Schema + Query + Results)
        float right_width = available_width - left_pane_width_ - 8.0f;
        ImGui::BeginChild("RightPane", ImVec2(right_width, available_height), false);
        {
            // Schema Viewer
            ImGui::BeginChild("SchemaViewer", ImVec2(0, schema_pane_height_), true);
            RenderSchemaViewerPane();
            ImGui::EndChild();

            // Splitter
            ImGui::InvisibleButton("##HSplitter1", ImVec2(-1, 4.0f));
            if (ImGui::IsItemActive()) {
                schema_pane_height_ += ImGui::GetIO().MouseDelta.y;
                schema_pane_height_ = std::clamp(schema_pane_height_, 80.0f, 300.0f);
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
            }

            // Query Editor
            ImGui::BeginChild("QueryEditor", ImVec2(0, query_pane_height_), true);
            RenderQueryEditorPane();
            ImGui::EndChild();

            // Splitter
            ImGui::InvisibleButton("##HSplitter2", ImVec2(-1, 4.0f));
            if (ImGui::IsItemActive()) {
                query_pane_height_ += ImGui::GetIO().MouseDelta.y;
                query_pane_height_ = std::clamp(query_pane_height_, 80.0f, 400.0f);
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
            }

            // Results Table
            ImGui::BeginChild("Results", ImVec2(0, 0), true);
            RenderResultsPane();
            ImGui::EndChild();
        }
        ImGui::EndChild();

        // Status bar
        RenderStatusBar();
    }
    ImGui::End();

    // Handle popups
    if (show_examples_popup_) {
        ImGui::OpenPopup("Examples");
        show_examples_popup_ = false;
    }
    if (show_history_popup_) {
        ImGui::OpenPopup("History");
        show_history_popup_ = false;
    }
    RenderExamplesPopup();
    RenderHistoryPopup();
}

void DataExplorerPanel::HandleKeyboardShortcuts() {
    if (!focused_) return;

    auto& io = ImGui::GetIO();

    if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_O)) {
        OpenFileDialog();
    }
    if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Enter)) {
        ExecuteCurrentQuery();
    }
}

void DataExplorerPanel::RenderFileBrowserPane() {
    ImGui::Text(ICON_FA_FOLDER_TREE " Files");
    ImGui::Separator();

    // Open file button
    if (ImGui::Button(ICON_FA_FOLDER_OPEN " Open File", ImVec2(-1, 0))) {
        OpenFileDialog();
    }

    ImGui::Spacing();

    // Recent files
    if (!recent_files_.empty()) {
        ImGui::Text(ICON_FA_CLOCK_ROTATE_LEFT " Recent");
        ImGui::Separator();

        for (const auto& path : recent_files_) {
            std::filesystem::path fs_path(path);
            std::string filename = fs_path.filename().string();
            std::string ext = fs_path.extension().string();

            const char* icon = GetFileIcon(ext);

            ImGui::PushID(path.c_str());
            if (ImGui::Selectable((std::string(icon) + " " + filename).c_str(),
                                  current_schema_.file_path == path)) {
                LoadSchema(path);
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("%s", path.c_str());
            }
            if (ImGui::IsItemClicked() && ImGui::IsMouseDoubleClicked(0)) {
                InsertFilePath(path);
            }
            ImGui::PopID();
        }
    }

    ImGui::Spacing();
    ImGui::TextDisabled("Double-click to insert path into query");
}

void DataExplorerPanel::RenderSchemaViewerPane() {
    if (current_schema_.file_path.empty()) {
        ImGui::TextDisabled("Select a file to view schema");
        return;
    }

    std::filesystem::path fs_path(current_schema_.file_path);
    ImGui::Text(ICON_FA_TABLE " %s", fs_path.filename().string().c_str());

    if (is_loading_schema_.load()) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.5f, 0.7f, 1.0f, 1.0f), "(Loading...)");
        return;
    }

    if (!schema_error_.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Error: %s", schema_error_.c_str());
        return;
    }

    if (!current_schema_.is_loaded) return;

    ImGui::SameLine();
    ImGui::TextDisabled("| %s rows", FormatNumber(current_schema_.row_count).c_str());

    ImGui::Separator();

    // Schema table
    if (ImGui::BeginTable("SchemaTable", 3,
        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY)) {

        ImGui::TableSetupColumn("Column", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed, 100.0f);
        ImGui::TableSetupColumn("Null", ImGuiTableColumnFlags_WidthFixed, 40.0f);
        ImGui::TableHeadersRow();

        for (const auto& col : current_schema_.columns) {
            ImGui::TableNextRow();

            ImGui::TableNextColumn();
            ImGui::Text("%s", col.name.c_str());

            ImGui::TableNextColumn();
            ImVec4 color = GetTypeColor(col.type);
            ImGui::TextColored(color, "%s", col.type.c_str());

            ImGui::TableNextColumn();
            if (col.nullable) {
                ImGui::TextDisabled("Yes");
            } else {
                ImGui::Text("No");
            }
        }

        ImGui::EndTable();
    }
}

void DataExplorerPanel::RenderQueryEditorPane() {
    RenderQueryToolbar();

    ImGui::Separator();

    // Query input
    float input_height = ImGui::GetContentRegionAvail().y;
    ImGui::InputTextMultiline("##QueryInput", query_buffer_, QUERY_BUFFER_SIZE,
        ImVec2(-1, input_height),
        ImGuiInputTextFlags_AllowTabInput);

    // Check for Ctrl+Enter
    if (ImGui::IsItemFocused() && ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Enter)) {
        ExecuteCurrentQuery();
    }
}

void DataExplorerPanel::RenderQueryToolbar() {
    // Execute button
    bool can_execute = !is_executing_query_.load() && strlen(query_buffer_) > 0;
    if (!can_execute) ImGui::BeginDisabled();
    if (ImGui::Button(ICON_FA_PLAY " Run")) {
        ExecuteCurrentQuery();
    }
    if (!can_execute) ImGui::EndDisabled();
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Execute query (Ctrl+Enter)");
    }

    ImGui::SameLine();

    // Examples dropdown
    if (ImGui::Button(ICON_FA_LIGHTBULB " Examples")) {
        show_examples_popup_ = true;
    }

    ImGui::SameLine();

    // History dropdown
    if (ImGui::Button(ICON_FA_CLOCK_ROTATE_LEFT " History")) {
        show_history_popup_ = true;
    }

    ImGui::SameLine();

    // Clear button
    if (ImGui::Button(ICON_FA_ERASER " Clear")) {
        query_buffer_[0] = '\0';
    }

    // Status
    if (is_executing_query_.load()) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.5f, 0.7f, 1.0f, 1.0f), ICON_FA_SPINNER " Executing...");
    }
}

void DataExplorerPanel::RenderExamplesPopup() {
    if (ImGui::BeginPopup("Examples")) {
        ImGui::Text("Example Queries");
        ImGui::Separator();

        for (const auto& example : example_queries_) {
            if (ImGui::Selectable(example.name)) {
                std::strcpy(query_buffer_, example.query);
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("%s\n\n%s", example.description, example.query);
            }
        }

        ImGui::EndPopup();
    }
}

void DataExplorerPanel::RenderHistoryPopup() {
    if (ImGui::BeginPopup("History")) {
        ImGui::Text("Query History");
        ImGui::Separator();

        if (query_history_.empty()) {
            ImGui::TextDisabled("No history yet");
        } else {
            for (size_t i = 0; i < query_history_.size(); i++) {
                const auto& item = query_history_[i];

                ImGui::PushID(static_cast<int>(i));

                // Truncate long queries for display
                std::string display = item.query;
                if (display.length() > 50) {
                    display = display.substr(0, 47) + "...";
                }
                // Replace newlines
                std::replace(display.begin(), display.end(), '\n', ' ');

                ImVec4 color = item.success ? ImVec4(0.4f, 1.0f, 0.4f, 1.0f) : ImVec4(1.0f, 0.4f, 0.4f, 1.0f);

                if (ImGui::Selectable(display.c_str())) {
                    std::strcpy(query_buffer_, item.query.c_str());
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("%s\n\n%s | %.1fms",
                        item.query.c_str(),
                        item.timestamp.c_str(),
                        item.execution_time_ms);
                }

                ImGui::PopID();
            }
        }

        ImGui::EndPopup();
    }
}

void DataExplorerPanel::RenderResultsPane() {
    RenderResultsToolbar();
    ImGui::Separator();

    if (is_executing_query_.load()) {
        ImGui::TextColored(ImVec4(0.5f, 0.7f, 1.0f, 1.0f), "Executing query...");
        return;
    }

    if (!current_result_.success && !current_result_.error_message.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
            ICON_FA_TRIANGLE_EXCLAMATION " %s", current_result_.error_message.c_str());
        return;
    }

    if (current_result_.column_names.empty()) {
        ImGui::TextDisabled("No results. Run a query to see data here.");
        return;
    }

    RenderResultsTable();
    RenderPagination();
}

void DataExplorerPanel::RenderResultsToolbar() {
    // Export button
    bool has_results = current_result_.success && !current_result_.rows.empty();

    if (!has_results) ImGui::BeginDisabled();
    if (ImGui::Button(ICON_FA_FILE_EXPORT " Export CSV")) {
        ExportResultsToCSV();
    }
    if (!has_results) ImGui::EndDisabled();

    ImGui::SameLine();

    if (!has_results) ImGui::BeginDisabled();
    if (ImGui::Button(ICON_FA_COPY " Copy")) {
        CopyResultsToClipboard();
    }
    if (!has_results) ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::Checkbox("Row #", &show_row_numbers_);

    // Stats
    if (current_result_.success) {
        ImGui::SameLine();
        ImGui::TextDisabled("| %s rows | %.1fms",
            FormatNumber(current_result_.total_rows).c_str(),
            current_result_.execution_time_ms);
    }
}

void DataExplorerPanel::RenderResultsTable() {
    int col_count = static_cast<int>(current_result_.column_names.size());
    if (show_row_numbers_) col_count++;

    ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                           ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY |
                           ImGuiTableFlags_Resizable;

    float table_height = ImGui::GetContentRegionAvail().y - 30.0f; // Reserve for pagination

    if (ImGui::BeginTable("ResultsTable", col_count, flags, ImVec2(0, table_height))) {
        // Setup columns
        if (show_row_numbers_) {
            ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 50.0f);
        }
        for (const auto& col_name : current_result_.column_names) {
            ImGui::TableSetupColumn(col_name.c_str(), ImGuiTableColumnFlags_WidthStretch);
        }
        ImGui::TableHeadersRow();

        // Calculate page range
        size_t start_row = static_cast<size_t>(current_page_) * rows_per_page_;
        size_t end_row = std::min(start_row + rows_per_page_, current_result_.rows.size());

        // Render visible rows
        ImGuiListClipper clipper;
        clipper.Begin(static_cast<int>(end_row - start_row));

        while (clipper.Step()) {
            for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; i++) {
                size_t row_idx = start_row + i;
                if (row_idx >= current_result_.rows.size()) break;

                const auto& row = current_result_.rows[row_idx];

                ImGui::TableNextRow();

                if (show_row_numbers_) {
                    ImGui::TableNextColumn();
                    ImGui::TextDisabled("%zu", row_idx + 1);
                }

                for (const auto& cell : row) {
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(cell.c_str());
                }
            }
        }

        ImGui::EndTable();
    }
}

void DataExplorerPanel::RenderPagination() {
    if (current_result_.rows.empty()) return;

    int total_pages = static_cast<int>((current_result_.rows.size() + rows_per_page_ - 1) / rows_per_page_);

    ImGui::Spacing();

    // First page
    if (ImGui::Button("<<")) {
        current_page_ = 0;
    }
    ImGui::SameLine();

    // Previous page
    if (ImGui::Button("<")) {
        if (current_page_ > 0) current_page_--;
    }
    ImGui::SameLine();

    // Page info
    ImGui::Text("Page %d / %d", current_page_ + 1, total_pages);
    ImGui::SameLine();

    // Next page
    if (ImGui::Button(">")) {
        if (current_page_ < total_pages - 1) current_page_++;
    }
    ImGui::SameLine();

    // Last page
    if (ImGui::Button(">>")) {
        current_page_ = total_pages - 1;
    }

    ImGui::SameLine();
    ImGui::SetNextItemWidth(80);
    if (ImGui::InputInt("##RowsPerPage", &rows_per_page_, 0, 0)) {
        rows_per_page_ = std::clamp(rows_per_page_, 10, 1000);
        current_page_ = 0;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("rows/page");
}

void DataExplorerPanel::RenderStatusBar() {
    ImGui::Separator();

    if (duckdb_available_) {
        ImGui::TextDisabled("DuckDB %s", DataLoader::GetVersion().c_str());
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "DuckDB unavailable");
    }

    if (!current_schema_.file_path.empty()) {
        ImGui::SameLine();
        ImGui::TextDisabled("| %s", current_schema_.file_path.c_str());
    }
}

void DataExplorerPanel::OpenFile(const std::string& path) {
    AddRecentFile(path);
    LoadSchema(path);
}

void DataExplorerPanel::OpenFileDialog() {
    nfdchar_t* outPath = nullptr;
    nfdfilteritem_t filters[3] = {
        {"CSV Files", "csv"},
        {"Parquet Files", "parquet"},
        {"JSON Files", "json"}
    };

    nfdresult_t result = NFD_OpenDialog(&outPath, filters, 3, current_directory_.c_str());

    if (result == NFD_OKAY) {
        std::string path(outPath);
        NFD_FreePath(outPath);
        OpenFile(path);
    }
}

void DataExplorerPanel::LoadSchema(const std::string& path) {
    if (!data_loader_ || is_loading_schema_.load()) return;

    current_schema_.file_path = path;
    current_schema_.is_loaded = false;
    current_schema_.columns.clear();
    schema_error_.clear();
    is_loading_schema_ = true;

    // Load schema synchronously for now (usually fast)
    try {
        auto columns = data_loader_->GetSchema(path);
        auto row_count = data_loader_->GetRowCount(path);

        std::lock_guard<std::mutex> lock(schema_mutex_);
        current_schema_.columns = columns;
        current_schema_.row_count = row_count;
        current_schema_.is_loaded = true;

        AddRecentFile(path);
        spdlog::info("DataExplorerPanel: Loaded schema for {} ({} columns, {} rows)",
            path, columns.size(), row_count);
    } catch (const std::exception& e) {
        schema_error_ = e.what();
        spdlog::error("DataExplorerPanel: Failed to load schema: {}", e.what());
    }

    is_loading_schema_ = false;
}

void DataExplorerPanel::ExecuteQuery(const std::string& sql) {
    std::strcpy(query_buffer_, sql.c_str());
    ExecuteCurrentQuery();
}

void DataExplorerPanel::ExecuteCurrentQuery() {
    if (!data_loader_ || is_executing_query_.load()) return;

    std::string sql = query_buffer_;
    if (sql.empty()) return;

    is_executing_query_ = true;
    current_result_ = QueryResult{};
    current_page_ = 0;

    auto start = std::chrono::steady_clock::now();

    try {
        // Execute query
        Tensor result = data_loader_->Query(sql);

        auto end = std::chrono::steady_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

        // Convert tensor to strings for display
        std::lock_guard<std::mutex> lock(result_mutex_);

        auto shape = result.Shape();
        size_t rows = shape.size() > 0 ? shape[0] : 0;
        size_t cols = shape.size() > 1 ? shape[1] : (rows > 0 ? 1 : 0);

        // Generate column names (we don't have them from tensor, use Col1, Col2, etc.)
        current_result_.column_names.clear();
        for (size_t c = 0; c < cols; c++) {
            current_result_.column_names.push_back("Col" + std::to_string(c + 1));
        }

        // Convert tensor data to strings
        current_result_.rows.clear();
        current_result_.rows.reserve(rows);

        // Get pointer to tensor data
        const float* data_ptr = result.Data<float>();
        size_t num_elements = result.NumElements();

        for (size_t r = 0; r < rows; r++) {
            std::vector<std::string> row;
            row.reserve(cols);
            for (size_t c = 0; c < cols; c++) {
                size_t idx = r * cols + c;
                float val = (idx < num_elements) ? data_ptr[idx] : 0.0f;
                std::ostringstream ss;
                ss << std::setprecision(6) << val;
                row.push_back(ss.str());
            }
            current_result_.rows.push_back(std::move(row));
        }

        current_result_.total_rows = rows;
        current_result_.execution_time_ms = elapsed_ms;
        current_result_.success = true;

        // Add to history
        QueryHistoryItem history_item;
        history_item.query = sql;
        history_item.timestamp = GetCurrentTimestamp();
        history_item.success = true;
        history_item.execution_time_ms = elapsed_ms;
        query_history_.push_front(history_item);
        if (query_history_.size() > MAX_QUERY_HISTORY) {
            query_history_.pop_back();
        }

        spdlog::info("DataExplorerPanel: Query returned {} rows in {:.1f}ms", rows, elapsed_ms);

    } catch (const std::exception& e) {
        auto end = std::chrono::steady_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::lock_guard<std::mutex> lock(result_mutex_);
        current_result_.success = false;
        current_result_.error_message = e.what();
        current_result_.execution_time_ms = elapsed_ms;

        // Add failed query to history
        QueryHistoryItem history_item;
        history_item.query = sql;
        history_item.timestamp = GetCurrentTimestamp();
        history_item.success = false;
        history_item.execution_time_ms = elapsed_ms;
        query_history_.push_front(history_item);
        if (query_history_.size() > MAX_QUERY_HISTORY) {
            query_history_.pop_back();
        }

        spdlog::error("DataExplorerPanel: Query failed: {}", e.what());
    }

    is_executing_query_ = false;
}

void DataExplorerPanel::ExportResultsToCSV() {
    if (!current_result_.success || current_result_.rows.empty()) return;

    nfdchar_t* outPath = nullptr;
    nfdfilteritem_t filters[1] = {{"CSV Files", "csv"}};

    nfdresult_t result = NFD_SaveDialog(&outPath, filters, 1, nullptr, "export.csv");

    if (result == NFD_OKAY) {
        std::string path(outPath);
        NFD_FreePath(outPath);

        try {
            std::ofstream file(path);

            // Header
            for (size_t i = 0; i < current_result_.column_names.size(); i++) {
                if (i > 0) file << ",";
                file << current_result_.column_names[i];
            }
            file << "\n";

            // Data
            for (const auto& row : current_result_.rows) {
                for (size_t i = 0; i < row.size(); i++) {
                    if (i > 0) file << ",";
                    file << row[i];
                }
                file << "\n";
            }

            spdlog::info("DataExplorerPanel: Exported {} rows to {}", current_result_.rows.size(), path);
        } catch (const std::exception& e) {
            spdlog::error("DataExplorerPanel: Export failed: {}", e.what());
        }
    }
}

void DataExplorerPanel::CopyResultsToClipboard() {
    if (!current_result_.success || current_result_.rows.empty()) return;

    std::ostringstream ss;

    // Header
    for (size_t i = 0; i < current_result_.column_names.size(); i++) {
        if (i > 0) ss << "\t";
        ss << current_result_.column_names[i];
    }
    ss << "\n";

    // Data (current page only)
    size_t start_row = static_cast<size_t>(current_page_) * rows_per_page_;
    size_t end_row = std::min(start_row + rows_per_page_, current_result_.rows.size());

    for (size_t r = start_row; r < end_row; r++) {
        const auto& row = current_result_.rows[r];
        for (size_t i = 0; i < row.size(); i++) {
            if (i > 0) ss << "\t";
            ss << row[i];
        }
        ss << "\n";
    }

    ImGui::SetClipboardText(ss.str().c_str());
    spdlog::info("DataExplorerPanel: Copied {} rows to clipboard", end_row - start_row);
}

void DataExplorerPanel::AddRecentFile(const std::string& path) {
    // Remove if already exists
    auto it = std::find(recent_files_.begin(), recent_files_.end(), path);
    if (it != recent_files_.end()) {
        recent_files_.erase(it);
    }

    // Add to front
    recent_files_.push_front(path);

    // Limit size
    while (recent_files_.size() > MAX_RECENT_FILES) {
        recent_files_.pop_back();
    }
}

void DataExplorerPanel::InsertFilePath(const std::string& path) {
    // Normalize path for SQL
    std::string normalized = path;
    std::replace(normalized.begin(), normalized.end(), '\\', '/');

    // Find cursor position and insert
    std::string current = query_buffer_;
    std::string insert_text = "'" + normalized + "'";

    // Simple append for now
    if (!current.empty() && current.back() != ' ' && current.back() != '\n') {
        insert_text = " " + insert_text;
    }
    current += insert_text;

    std::strncpy(query_buffer_, current.c_str(), QUERY_BUFFER_SIZE - 1);
    query_buffer_[QUERY_BUFFER_SIZE - 1] = '\0';
}

bool DataExplorerPanel::IsDataFile(const std::string& path) const {
    std::filesystem::path fs_path(path);
    std::string ext = fs_path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    return ext == ".csv" || ext == ".parquet" || ext == ".json" ||
           ext == ".tsv" || ext == ".txt";
}

const char* DataExplorerPanel::GetFileIcon(const std::string& ext) const {
    std::string lower_ext = ext;
    std::transform(lower_ext.begin(), lower_ext.end(), lower_ext.begin(), ::tolower);

    if (lower_ext == ".csv" || lower_ext == ".tsv") return ICON_FA_FILE_CSV;
    if (lower_ext == ".parquet") return ICON_FA_DATABASE;
    if (lower_ext == ".json") return ICON_FA_FILE_CODE;
    return ICON_FA_FILE;
}

ImVec4 DataExplorerPanel::GetTypeColor(const std::string& type) const {
    std::string upper_type = type;
    std::transform(upper_type.begin(), upper_type.end(), upper_type.begin(), ::toupper);

    if (upper_type.find("INT") != std::string::npos)
        return ImVec4(0.4f, 0.7f, 1.0f, 1.0f);  // Blue
    if (upper_type.find("FLOAT") != std::string::npos || upper_type.find("DOUBLE") != std::string::npos)
        return ImVec4(0.4f, 1.0f, 0.7f, 1.0f);  // Green
    if (upper_type.find("VARCHAR") != std::string::npos || upper_type.find("STRING") != std::string::npos)
        return ImVec4(1.0f, 0.8f, 0.4f, 1.0f);  // Yellow
    if (upper_type.find("BOOL") != std::string::npos)
        return ImVec4(1.0f, 0.5f, 0.5f, 1.0f);  // Red
    if (upper_type.find("DATE") != std::string::npos || upper_type.find("TIME") != std::string::npos)
        return ImVec4(0.8f, 0.6f, 1.0f, 1.0f);  // Purple

    return ImVec4(0.7f, 0.7f, 0.7f, 1.0f);  // Gray
}

std::string DataExplorerPanel::FormatFileSize(std::uintmax_t bytes) const {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_idx = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024.0 && unit_idx < 4) {
        size /= 1024.0;
        unit_idx++;
    }

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1) << size << " " << units[unit_idx];
    return ss.str();
}

std::string DataExplorerPanel::FormatNumber(size_t num) const {
    std::string str = std::to_string(num);
    int n = static_cast<int>(str.length()) - 3;
    while (n > 0) {
        str.insert(n, ",");
        n -= 3;
    }
    return str;
}

std::string DataExplorerPanel::GetCurrentTimestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &time);
#else
    localtime_r(&time, &tm_buf);
#endif

    std::ostringstream ss;
    ss << std::put_time(&tm_buf, "%H:%M:%S");
    return ss.str();
}

} // namespace cyxwiz
