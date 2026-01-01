#include "data_explorer_panel.h"
#include "../../core/project_manager.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <nfd.h>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <fstream>
#include <cmath>

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
    // Tab bar for different result views
    if (ImGui::BeginTabBar("ResultTabs", ImGuiTabBarFlags_None)) {
        if (ImGui::BeginTabItem(ICON_FA_TABLE " Results")) {
            current_tab_ = DataExplorerTab::Results;
            RenderResultsTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_CHART_SIMPLE " Quick Stats")) {
            current_tab_ = DataExplorerTab::QuickStats;
            RenderQuickStatsTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_CHART_LINE " Visualize")) {
            current_tab_ = DataExplorerTab::Visualize;
            RenderVisualizeTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_ERASER " Clean")) {
            current_tab_ = DataExplorerTab::Clean;
            RenderCleanTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_NETWORK_WIRED " Hub")) {
            current_tab_ = DataExplorerTab::Hub;
            RenderHubTab();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

// ===== Phase 1: Results Tab (Original Behavior) =====
void DataExplorerPanel::RenderResultsTab() {
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
        // Use smart path formatting - shows relative path if in project
        std::string display_path = SmartFormatPath(current_schema_.file_path);
        if (IsInProjectFolder(current_schema_.file_path)) {
            ImGui::TextDisabled("| %s (project)", display_path.c_str());
        } else {
            ImGui::TextDisabled("| %s", display_path.c_str());
        }
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

// ===== Phase 1: Smart Path Methods =====

std::string DataExplorerPanel::SmartFormatPath(const std::string& absolute_path) const {
    auto& pm = ProjectManager::Instance();
    if (pm.HasActiveProject() && IsInProjectFolder(absolute_path)) {
        return pm.MakeRelativePath(absolute_path);
    }
    return absolute_path;
}

bool DataExplorerPanel::IsInProjectFolder(const std::string& path) const {
    auto& pm = ProjectManager::Instance();
    if (!pm.HasActiveProject()) return false;

    std::string project_root = pm.GetProjectRoot();
    // Normalize paths for comparison
    std::filesystem::path fs_path(path);
    std::filesystem::path fs_root(project_root);

    try {
        std::string norm_path = std::filesystem::weakly_canonical(fs_path).string();
        std::string norm_root = std::filesystem::weakly_canonical(fs_root).string();

        // Check if path starts with project root
        return norm_path.find(norm_root) == 0;
    } catch (...) {
        return false;
    }
}

std::string DataExplorerPanel::GetProjectDatasetsPath() const {
    auto& pm = ProjectManager::Instance();
    if (pm.HasActiveProject()) {
        return pm.GetDatasetsPath();
    }
    return "";
}

// ===== Phase 2: Quick Stats Tab =====

void DataExplorerPanel::RenderQuickStatsTab() {
    if (!current_result_.success || current_result_.rows.empty()) {
        ImGui::TextDisabled("Run a query first to see statistics");
        return;
    }

    // Column selector
    ImGui::AlignTextToFramePadding();
    ImGui::Text("Column:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(200);
    if (ImGui::BeginCombo("##StatsColumn",
        selected_stats_column_ >= 0 && selected_stats_column_ < (int)current_result_.column_names.size()
            ? current_result_.column_names[selected_stats_column_].c_str()
            : "Select...")) {
        for (int i = 0; i < (int)current_result_.column_names.size(); i++) {
            if (ImGui::Selectable(current_result_.column_names[i].c_str(), selected_stats_column_ == i)) {
                if (selected_stats_column_ != i) {
                    selected_stats_column_ = i;
                    ComputeQuickStats(i);
                }
            }
        }
        ImGui::EndCombo();
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_CALCULATOR " Compute")) {
        ComputeQuickStats(selected_stats_column_);
    }

    ImGui::Separator();

    if (quick_stats_cache_.is_computing) {
        ImGui::TextColored(ImVec4(0.5f, 0.7f, 1.0f, 1.0f), ICON_FA_SPINNER " Computing statistics...");
        return;
    }

    if (!quick_stats_cache_.is_valid) {
        ImGui::TextDisabled("Select a column and click Compute to see statistics");
        return;
    }

    // Three-column layout: Stats | Histogram | Correlations
    float avail_width = ImGui::GetContentRegionAvail().x;
    float col_width = avail_width / 3.0f - 10.0f;

    // Statistics Table
    ImGui::BeginChild("StatsTable", ImVec2(col_width, 0), true);
    RenderStatsTable();
    ImGui::EndChild();

    ImGui::SameLine();

    // Mini Histogram
    ImGui::BeginChild("MiniHistogram", ImVec2(col_width, 0), true);
    RenderMiniHistogram();
    ImGui::EndChild();

    ImGui::SameLine();

    // Top Correlations
    ImGui::BeginChild("TopCorrelations", ImVec2(col_width, 0), true);
    RenderTopCorrelations();
    ImGui::EndChild();
}

void DataExplorerPanel::ComputeQuickStats(int column_index) {
    if (column_index < 0 || column_index >= (int)current_result_.column_names.size()) return;
    if (!data_loader_) return;

    quick_stats_cache_.is_computing = true;
    quick_stats_cache_.is_valid = false;
    quick_stats_cache_.column_index = column_index;

    // Get column data as doubles
    std::vector<double> col_data = GetColumnAsDoubles(column_index);

    if (col_data.empty()) {
        quick_stats_cache_.is_computing = false;
        return;
    }

    // Compute statistics synchronously for now (usually fast)
    try {
        quick_stats_cache_.stats = DataAnalyzer::ComputeDescriptiveStats(col_data);
        quick_stats_cache_.column_data = std::move(col_data);

        // Compute correlations with other numeric columns
        quick_stats_cache_.top_correlations.clear();
        for (int i = 0; i < (int)current_result_.column_names.size(); i++) {
            if (i == column_index) continue;

            std::vector<double> other_col = GetColumnAsDoubles(i);
            if (other_col.empty() || other_col.size() != quick_stats_cache_.column_data.size()) continue;

            // Compute Pearson correlation inline
            const auto& x = quick_stats_cache_.column_data;
            const auto& y = other_col;
            size_t n = x.size();

            double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0, sum_y2 = 0;
            for (size_t j = 0; j < n; j++) {
                sum_x += x[j];
                sum_y += y[j];
                sum_xy += x[j] * y[j];
                sum_x2 += x[j] * x[j];
                sum_y2 += y[j] * y[j];
            }

            double num = n * sum_xy - sum_x * sum_y;
            double den = std::sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));
            double corr = (den != 0) ? num / den : std::nan("");

            if (!std::isnan(corr)) {
                quick_stats_cache_.top_correlations.push_back({current_result_.column_names[i], corr});
            }
        }

        // Sort by absolute correlation value
        std::sort(quick_stats_cache_.top_correlations.begin(),
                  quick_stats_cache_.top_correlations.end(),
                  [](const auto& a, const auto& b) {
                      return std::abs(a.second) > std::abs(b.second);
                  });

        // Keep top 5
        if (quick_stats_cache_.top_correlations.size() > 5) {
            quick_stats_cache_.top_correlations.resize(5);
        }

        quick_stats_cache_.is_valid = true;
    } catch (const std::exception& e) {
        spdlog::error("DataExplorerPanel: Failed to compute stats: {}", e.what());
    }

    quick_stats_cache_.is_computing = false;
}

void DataExplorerPanel::RenderStatsTable() {
    ImGui::Text(ICON_FA_CHART_COLUMN " Statistics");
    ImGui::Separator();

    const auto& stats = quick_stats_cache_.stats;

    if (ImGui::BeginTable("StatsValues", 2, ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Stat", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

        auto AddRow = [](const char* label, double value, const char* fmt = "%.4f") {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextDisabled("%s", label);
            ImGui::TableNextColumn();
            ImGui::Text(fmt, value);
        };

        AddRow("Count", static_cast<double>(stats.count), "%.0f");
        AddRow("Mean", stats.mean);
        AddRow("Median", stats.median);
        AddRow("Std Dev", stats.std_dev);
        AddRow("Min", stats.min);
        AddRow("Max", stats.max);
        AddRow("Q1 (25%)", stats.q1);
        AddRow("Q3 (75%)", stats.q3);
        AddRow("Skewness", stats.skewness);
        AddRow("Kurtosis", stats.kurtosis);

        // Missing values
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextDisabled("Missing");
        ImGui::TableNextColumn();
        size_t missing = current_result_.rows.size() - stats.count;
        if (missing > 0) {
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.4f, 1.0f), "%zu (%.1f%%)",
                missing, 100.0 * missing / current_result_.rows.size());
        } else {
            ImGui::Text("0");
        }

        ImGui::EndTable();
    }
}

void DataExplorerPanel::RenderMiniHistogram() {
    ImGui::Text(ICON_FA_CHART_BAR " Distribution");
    ImGui::Separator();

    if (quick_stats_cache_.column_data.empty()) {
        ImGui::TextDisabled("No data");
        return;
    }

    // Create histogram using ImPlot
    if (ImPlot::BeginPlot("##Histogram", ImVec2(-1, -1), ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect)) {
        ImPlot::SetupAxes("Value", "Count", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);

        // Use ImPlot's histogram
        ImPlot::PlotHistogram("##hist", quick_stats_cache_.column_data.data(),
                              (int)quick_stats_cache_.column_data.size(), 20);

        // Draw mean line
        double mean = quick_stats_cache_.stats.mean;
        ImPlot::DragLineX(0, &mean, ImVec4(1, 0.5f, 0, 1), 1, ImPlotDragToolFlags_NoInputs);

        ImPlot::EndPlot();
    }
}

void DataExplorerPanel::RenderTopCorrelations() {
    ImGui::Text(ICON_FA_LINK " Top Correlations");
    ImGui::Separator();

    if (quick_stats_cache_.top_correlations.empty()) {
        ImGui::TextDisabled("No correlations computed");
        return;
    }

    for (const auto& [name, corr] : quick_stats_cache_.top_correlations) {
        // Color based on correlation strength
        ImVec4 color;
        if (corr > 0.7) color = ImVec4(0.4f, 1.0f, 0.4f, 1.0f);      // Strong positive - green
        else if (corr > 0.3) color = ImVec4(0.7f, 1.0f, 0.7f, 1.0f); // Moderate positive - light green
        else if (corr < -0.7) color = ImVec4(1.0f, 0.4f, 0.4f, 1.0f); // Strong negative - red
        else if (corr < -0.3) color = ImVec4(1.0f, 0.7f, 0.7f, 1.0f); // Moderate negative - light red
        else color = ImVec4(0.7f, 0.7f, 0.7f, 1.0f);                  // Weak - gray

        ImGui::TextColored(color, "%.3f", corr);
        ImGui::SameLine();
        ImGui::Text("%s", name.c_str());
    }
}

std::vector<double> DataExplorerPanel::GetColumnAsDoubles(int col_index) const {
    std::vector<double> result;
    if (col_index < 0 || col_index >= (int)current_result_.column_names.size()) return result;

    result.reserve(current_result_.rows.size());

    for (const auto& row : current_result_.rows) {
        if (col_index >= (int)row.size()) continue;

        try {
            double val = std::stod(row[col_index]);
            if (!std::isnan(val) && !std::isinf(val)) {
                result.push_back(val);
            }
        } catch (...) {
            // Skip non-numeric values
        }
    }

    return result;
}

// ===== Phase 3: Visualize Tab =====

void DataExplorerPanel::RenderVisualizeTab() {
    if (!current_result_.success || current_result_.rows.empty()) {
        ImGui::TextDisabled("Run a query first to create visualizations");
        return;
    }

    // Chart type selector and column selectors
    RenderChartSelector();

    ImGui::Separator();

    // Render the selected chart
    switch (chart_type_) {
        case ChartType::Histogram: RenderHistogramChart(); break;
        case ChartType::Scatter: RenderScatterChart(); break;
        case ChartType::Bar: RenderBarChart(); break;
        case ChartType::Box: RenderBoxChart(); break;
    }
}

void DataExplorerPanel::RenderChartSelector() {
    // Chart type radio buttons
    ImGui::Text("Chart Type:");
    ImGui::SameLine();
    if (ImGui::RadioButton("Histogram", chart_type_ == ChartType::Histogram)) chart_type_ = ChartType::Histogram;
    ImGui::SameLine();
    if (ImGui::RadioButton("Scatter", chart_type_ == ChartType::Scatter)) chart_type_ = ChartType::Scatter;
    ImGui::SameLine();
    if (ImGui::RadioButton("Bar", chart_type_ == ChartType::Bar)) chart_type_ = ChartType::Bar;
    ImGui::SameLine();
    if (ImGui::RadioButton("Box", chart_type_ == ChartType::Box)) chart_type_ = ChartType::Box;

    ImGui::Spacing();

    // Column selectors based on chart type
    auto ColumnCombo = [&](const char* label, int& selected) {
        ImGui::SetNextItemWidth(150);
        if (ImGui::BeginCombo(label,
            selected >= 0 && selected < (int)current_result_.column_names.size()
                ? current_result_.column_names[selected].c_str()
                : "Select...")) {
            for (int i = 0; i < (int)current_result_.column_names.size(); i++) {
                if (ImGui::Selectable(current_result_.column_names[i].c_str(), selected == i)) {
                    selected = i;
                }
            }
            ImGui::EndCombo();
        }
    };

    ImGui::Text("X:");
    ImGui::SameLine();
    ColumnCombo("##XCol", viz_x_column_);

    if (chart_type_ == ChartType::Scatter) {
        ImGui::SameLine();
        ImGui::Text("Y:");
        ImGui::SameLine();
        ColumnCombo("##YCol", viz_y_column_);
    }
}

void DataExplorerPanel::RenderHistogramChart() {
    std::vector<double> data = GetColumnAsDoubles(viz_x_column_);

    if (data.empty()) {
        ImGui::TextDisabled("Selected column has no numeric data");
        return;
    }

    if (ImPlot::BeginPlot("Histogram", ImVec2(-1, -1))) {
        ImPlot::SetupAxes(current_result_.column_names[viz_x_column_].c_str(), "Count");
        ImPlot::PlotHistogram("##hist", data.data(), (int)data.size(), 30);
        ImPlot::EndPlot();
    }
}

void DataExplorerPanel::RenderScatterChart() {
    std::vector<double> x_data = GetColumnAsDoubles(viz_x_column_);
    std::vector<double> y_data = GetColumnAsDoubles(viz_y_column_);

    if (x_data.empty() || y_data.empty()) {
        ImGui::TextDisabled("Selected columns have no numeric data");
        return;
    }

    // Match sizes
    size_t n = std::min(x_data.size(), y_data.size());
    x_data.resize(n);
    y_data.resize(n);

    if (ImPlot::BeginPlot("Scatter Plot", ImVec2(-1, -1))) {
        ImPlot::SetupAxes(current_result_.column_names[viz_x_column_].c_str(),
                          current_result_.column_names[viz_y_column_].c_str());
        ImPlot::PlotScatter("Data", x_data.data(), y_data.data(), (int)n);
        ImPlot::EndPlot();
    }
}

void DataExplorerPanel::RenderBarChart() {
    std::vector<double> data = GetColumnAsDoubles(viz_x_column_);

    if (data.empty()) {
        ImGui::TextDisabled("Selected column has no numeric data");
        return;
    }

    // Limit bars for display
    int n = std::min((int)data.size(), 50);

    if (ImPlot::BeginPlot("Bar Chart", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("Index", current_result_.column_names[viz_x_column_].c_str());
        ImPlot::PlotBars("##bars", data.data(), n, 0.67);
        ImPlot::EndPlot();
    }
}

void DataExplorerPanel::RenderBoxChart() {
    std::vector<double> data = GetColumnAsDoubles(viz_x_column_);

    if (data.empty()) {
        ImGui::TextDisabled("Selected column has no numeric data");
        return;
    }

    // Compute box plot statistics manually
    std::sort(data.begin(), data.end());
    size_t n = data.size();

    double min_val = data.front();
    double max_val = data.back();
    double q1 = data[n / 4];
    double median = data[n / 2];
    double q3 = data[3 * n / 4];
    double iqr = q3 - q1;
    double lower_fence = q1 - 1.5 * iqr;
    double upper_fence = q3 + 1.5 * iqr;

    if (ImPlot::BeginPlot("Box Plot", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("", current_result_.column_names[viz_x_column_].c_str());
        ImPlot::SetupAxisLimits(ImAxis_X1, -1, 1);

        // Draw box
        double xs[] = {0, 0};
        double box_lo[] = {q1, q1};
        double box_hi[] = {q3, q3};

        // IQR box
        ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(0.4f, 0.6f, 1.0f, 0.5f));
        double box_x[] = {-0.3, 0.3, 0.3, -0.3, -0.3};
        double box_y[] = {q1, q1, q3, q3, q1};
        ImPlot::PlotLine("Box", box_x, box_y, 5);
        ImPlot::PopStyleColor();

        // Median line
        double med_x[] = {-0.3, 0.3};
        double med_y[] = {median, median};
        ImPlot::PlotLine("Median", med_x, med_y, 2);

        // Whiskers
        double whisker_x[] = {0, 0};
        double whisker_lo[] = {std::max(min_val, lower_fence), q1};
        double whisker_hi[] = {q3, std::min(max_val, upper_fence)};
        ImPlot::PlotLine("Lower", whisker_x, whisker_lo, 2);
        ImPlot::PlotLine("Upper", whisker_x, whisker_hi, 2);

        ImPlot::EndPlot();
    }
}

// ===== Phase 4: Integration Hub Tab =====

void DataExplorerPanel::RenderHubTab() {
    ImGui::Text(ICON_FA_NETWORK_WIRED " Send Query Results to Analysis Panels");
    ImGui::Separator();

    if (!current_result_.success || current_result_.rows.empty()) {
        ImGui::TextDisabled("Run a query first to send data to other panels");
        return;
    }

    ImGui::Spacing();
    ImGui::Text("Current dataset: %s rows x %zu columns",
                FormatNumber(current_result_.total_rows).c_str(),
                current_result_.column_names.size());
    ImGui::Spacing();

    // Grid of destination panels
    float button_width = 180.0f;
    float button_height = 80.0f;

    ImGui::BeginGroup();

    RenderHubButton(ICON_FA_CHART_COLUMN, "Descriptive Stats",
                    "Detailed statistics for all columns",
                    [this]() { SendToDescriptiveStats(); });

    ImGui::SameLine();

    RenderHubButton(ICON_FA_TABLE_CELLS, "Correlation Matrix",
                    "Correlations between all numeric columns",
                    [this]() { SendToCorrelationMatrix(); });

    ImGui::SameLine();

    RenderHubButton(ICON_FA_CHART_LINE, "Regression",
                    "Linear/polynomial regression analysis",
                    [this]() { SendToRegression(); });

    ImGui::Spacing();

    RenderHubButton(ICON_FA_MAGNIFYING_GLASS_CHART, "Outlier Detection",
                    "Find outliers using IQR/Z-Score",
                    [this]() { SendToOutlierDetection(); });

    ImGui::SameLine();

    RenderHubButton(ICON_FA_QUESTION, "Missing Values",
                    "Analyze and impute missing data",
                    [this]() { SendToMissingValuePanel(); });

    ImGui::SameLine();

    RenderHubButton(ICON_FA_TABLE, "Data Profiler",
                    "Full data quality report",
                    [this]() { SendToDataProfiler(); });

    ImGui::EndGroup();
}

void DataExplorerPanel::RenderHubButton(const char* icon, const char* label,
                                         const char* description,
                                         std::function<void()> on_click) {
    ImVec2 button_size(180.0f, 80.0f);

    ImGui::BeginGroup();
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10, 10));

    if (ImGui::Button((std::string(icon) + "\n" + label).c_str(), button_size)) {
        if (on_click) on_click();
    }

    ImGui::PopStyleVar();

    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("%s", description);
    }

    ImGui::EndGroup();
}

void DataExplorerPanel::SendToDescriptiveStats() {
    spdlog::info("DataExplorerPanel: Sending to Descriptive Stats panel (TODO: implement)");
    // TODO: Use DataTableRegistry to share data with Stats panel
}

void DataExplorerPanel::SendToCorrelationMatrix() {
    spdlog::info("DataExplorerPanel: Sending to Correlation Matrix panel (TODO: implement)");
}

void DataExplorerPanel::SendToRegression() {
    spdlog::info("DataExplorerPanel: Sending to Regression panel (TODO: implement)");
}

void DataExplorerPanel::SendToOutlierDetection() {
    spdlog::info("DataExplorerPanel: Sending to Outlier Detection panel (TODO: implement)");
}

void DataExplorerPanel::SendToMissingValuePanel() {
    spdlog::info("DataExplorerPanel: Sending to Missing Value panel (TODO: implement)");
}

void DataExplorerPanel::SendToDataProfiler() {
    spdlog::info("DataExplorerPanel: Sending to Data Profiler panel (TODO: implement)");
}

// ===== Phase 5: Data Cleaning Tab =====

void DataExplorerPanel::RenderCleanTab() {
    if (!current_result_.success || current_result_.rows.empty()) {
        ImGui::TextDisabled("Run a query first to access cleaning tools");
        return;
    }

    // Data quality summary bar
    ImGui::TextColored(ImVec4(0.5f, 0.7f, 1.0f, 1.0f), ICON_FA_STETHOSCOPE " DATA QUALITY:");
    ImGui::SameLine();
    ImGui::Text("%d missing | %d potential outliers", missing_count_, outlier_count_);

    if (ImGui::Button(ICON_FA_ROTATE " Analyze Quality")) {
        AnalyzeDataQuality();
    }

    ImGui::Separator();

    // Tabs for different cleaning operations
    if (ImGui::BeginTabBar("CleaningTabs")) {
        if (ImGui::BeginTabItem("Missing Values")) {
            RenderMissingValueSection();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Outliers")) {
            RenderOutlierSection();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Type Conversion")) {
            RenderTypeConversionSection();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void DataExplorerPanel::AnalyzeDataQuality() {
    missing_count_ = 0;
    outlier_count_ = 0;
    columns_with_missing_.clear();

    // Count missing values per column
    for (int col = 0; col < (int)current_result_.column_names.size(); col++) {
        int col_missing = 0;
        for (const auto& row : current_result_.rows) {
            if (col < (int)row.size()) {
                const std::string& val = row[col];
                if (val.empty() || val == "NULL" || val == "null" || val == "NaN" || val == "nan") {
                    col_missing++;
                }
            }
        }
        if (col_missing > 0) {
            columns_with_missing_.push_back(col);
            missing_count_ += col_missing;
        }
    }

    // Count potential outliers (using IQR method)
    for (int col = 0; col < (int)current_result_.column_names.size(); col++) {
        std::vector<double> data = GetColumnAsDoubles(col);
        if (data.size() < 4) continue;

        std::sort(data.begin(), data.end());
        double q1 = data[data.size() / 4];
        double q3 = data[3 * data.size() / 4];
        double iqr = q3 - q1;
        double lower = q1 - 1.5 * iqr;
        double upper = q3 + 1.5 * iqr;

        for (double v : data) {
            if (v < lower || v > upper) outlier_count_++;
        }
    }

    spdlog::info("DataExplorerPanel: Quality analysis - {} missing, {} outliers",
                 missing_count_, outlier_count_);
}

void DataExplorerPanel::RenderMissingValueSection() {
    ImGui::Text(ICON_FA_QUESTION " Missing Value Handling");
    ImGui::Separator();

    // Column selector
    ImGui::Text("Column:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(200);
    if (ImGui::BeginCombo("##MissingCol",
        clean_selected_column_ >= 0 && clean_selected_column_ < (int)current_result_.column_names.size()
            ? current_result_.column_names[clean_selected_column_].c_str()
            : "Select...")) {
        for (int i = 0; i < (int)current_result_.column_names.size(); i++) {
            bool has_missing = std::find(columns_with_missing_.begin(),
                                         columns_with_missing_.end(), i) != columns_with_missing_.end();
            std::string label = current_result_.column_names[i];
            if (has_missing) label += " (*)";

            if (ImGui::Selectable(label.c_str(), clean_selected_column_ == i)) {
                clean_selected_column_ = i;
            }
        }
        ImGui::EndCombo();
    }

    ImGui::Spacing();

    // Fill method
    ImGui::Text("Fill with:");
    ImGui::RadioButton("Mean", &missing_fill_method_, 0);
    ImGui::SameLine();
    ImGui::RadioButton("Median", &missing_fill_method_, 1);
    ImGui::SameLine();
    ImGui::RadioButton("Mode", &missing_fill_method_, 2);
    ImGui::SameLine();
    ImGui::RadioButton("Drop rows", &missing_fill_method_, 3);
    ImGui::SameLine();
    ImGui::RadioButton("Custom", &missing_fill_method_, 4);

    if (missing_fill_method_ == 4) {
        ImGui::SetNextItemWidth(100);
        ImGui::InputDouble("##CustomValue", &missing_custom_value_, 0.0, 0.0, "%.4f");
    }

    ImGui::Spacing();

    if (ImGui::Button(ICON_FA_EYE " Preview")) {
        PreviewMissingValueFix();
    }
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_WAND_MAGIC_SPARKLES " Generate SQL")) {
        std::string sql = GenerateCleaningSQL();
        std::strcpy(query_buffer_, sql.c_str());
    }
}

void DataExplorerPanel::RenderOutlierSection() {
    ImGui::Text(ICON_FA_MAGNIFYING_GLASS_CHART " Outlier Detection");
    ImGui::Separator();

    // Method selector
    ImGui::Text("Method:");
    ImGui::RadioButton("IQR (1.5x)", &outlier_method_, 0);
    ImGui::SameLine();
    ImGui::RadioButton("Z-Score", &outlier_method_, 1);
    ImGui::SameLine();
    ImGui::RadioButton("Modified Z", &outlier_method_, 2);

    ImGui::SetNextItemWidth(100);
    ImGui::SliderFloat("Threshold", &outlier_threshold_, 1.0f, 3.0f, "%.1f");

    ImGui::Spacing();

    // Action selector
    ImGui::Text("Action:");
    ImGui::RadioButton("Remove", &outlier_action_, 0);
    ImGui::SameLine();
    ImGui::RadioButton("Cap (Winsorize)", &outlier_action_, 1);
    ImGui::SameLine();
    ImGui::RadioButton("Replace with Mean", &outlier_action_, 2);

    ImGui::Spacing();

    if (ImGui::Button(ICON_FA_EYE " Preview")) {
        // TODO: Preview outlier handling
    }
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_WAND_MAGIC_SPARKLES " Generate SQL")) {
        // TODO: Generate SQL for outlier handling
    }
}

void DataExplorerPanel::RenderTypeConversionSection() {
    ImGui::Text(ICON_FA_REPEAT " Type Conversion");
    ImGui::Separator();

    ImGui::TextDisabled("Convert column types in your query using DuckDB CAST:");
    ImGui::Spacing();

    ImGui::BulletText("CAST(column AS INTEGER)");
    ImGui::BulletText("CAST(column AS DOUBLE)");
    ImGui::BulletText("CAST(column AS VARCHAR)");
    ImGui::BulletText("CAST(column AS DATE)");
    ImGui::BulletText("strptime(column, '%%Y-%%m-%%d') -- Parse date string");

    ImGui::Spacing();
    ImGui::Separator();

    // Quick conversion buttons
    if (ImGui::Button("Insert CAST template")) {
        std::string current = query_buffer_;
        current += "\n-- CAST(column_name AS NEW_TYPE)";
        std::strncpy(query_buffer_, current.c_str(), QUERY_BUFFER_SIZE - 1);
    }
}

void DataExplorerPanel::PreviewMissingValueFix() {
    // TODO: Show preview of rows that would be affected
    spdlog::info("DataExplorerPanel: Preview missing value fix (TODO)");
}

void DataExplorerPanel::ApplyMissingValueFix() {
    // TODO: Apply the fix by executing generated SQL
    spdlog::info("DataExplorerPanel: Apply missing value fix (TODO)");
}

std::string DataExplorerPanel::GenerateCleaningSQL() const {
    if (clean_selected_column_ < 0 || clean_selected_column_ >= (int)current_result_.column_names.size()) {
        return "";
    }

    std::string col_name = current_result_.column_names[clean_selected_column_];
    std::ostringstream sql;

    switch (missing_fill_method_) {
        case 0: // Mean
            sql << "SELECT COALESCE(" << col_name << ", AVG(" << col_name << ") OVER()) AS " << col_name;
            break;
        case 1: // Median
            sql << "SELECT COALESCE(" << col_name << ", MEDIAN(" << col_name << ") OVER()) AS " << col_name;
            break;
        case 2: // Mode
            sql << "SELECT COALESCE(" << col_name << ", MODE(" << col_name << ") OVER()) AS " << col_name;
            break;
        case 3: // Drop
            sql << "SELECT * FROM table_name WHERE " << col_name << " IS NOT NULL";
            break;
        case 4: // Custom
            sql << "SELECT COALESCE(" << col_name << ", " << missing_custom_value_ << ") AS " << col_name;
            break;
    }

    sql << "\n-- Replace 'table_name' with your actual table/file path";

    return sql.str();
}

} // namespace cyxwiz
