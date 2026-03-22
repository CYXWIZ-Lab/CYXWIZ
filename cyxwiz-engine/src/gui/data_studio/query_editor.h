#pragma once

#include <imgui.h>
#include <string>
#include <vector>
#include <memory>

namespace cyxwiz {

// Forward declaration
class DuckDBConnector;

/**
 * QueryEditor - SQL query interface for Data Studio
 *
 * Phase 1, Week 1: Core Infrastructure
 *
 * This component provides SQL query editing and execution using DuckDB.
 * It wraps the existing DataExplorerPanel functionality in a Data Studio tab.
 *
 * Features:
 * - SQL editor with syntax highlighting (TODO: add later)
 * - Query execution button
 * - Results table display
 * - Query history
 * - Example queries dropdown
 *
 * Architecture:
 *   QueryEditor -> DuckDBConnector -> Arrow Result -> Display Table
 */
class QueryEditor {
public:
    QueryEditor();
    ~QueryEditor();

    /**
     * Render the query editor UI
     * Called from DataStudioPanel's Query tab
     */
    void Render();

    /**
     * Set the active dataset for queries
     * Registers it with DuckDB as a table
     */
    void SetActiveDataset(const std::string& dataset_name);

    /**
     * Execute the current query
     */
    bool ExecuteQuery();

    /**
     * Save the current query result as a new dataset
     */
    bool SaveResultAsDataset(const std::string& dataset_name);

private:
    // Query state
    char query_buffer_[4096];
    std::string current_query_;
    std::string last_error_;
    bool query_running_;
    std::string current_dataset_;

    // DuckDB connector for SQL execution
    std::unique_ptr<DuckDBConnector> duckdb_;

    // Results
    struct QueryResult {
        std::vector<std::string> column_names;
        std::vector<std::vector<std::string>> rows;
        size_t total_rows;
        double execution_time_ms;
    };
    QueryResult last_result_;

    // UI helpers
    void RenderQueryEditor();
    void RenderResultsTable();
    void RenderQueryHistory();
    void RenderExampleQueries();

    // Query history
    std::vector<std::string> query_history_;
    int max_history_size_ = 50;
};

} // namespace cyxwiz
