#pragma once

#include <arrow/api.h>
#include <duckdb.hpp>
#include <memory>
#include <string>
#include <vector>
#include <map>

namespace cyxwiz {

/**
 * DuckDB Connector (Phase 0 Day 4)
 *
 * Provides SQL queries on Arrow tables using DuckDB.
 * This is the Data Studio query engine foundation.
 *
 * Key Features:
 * - Arrow table registration into an in-memory DuckDB table
 * - SQL query execution with result as Arrow table
 * - Multiple table joins without data duplication
 * - Window functions, aggregations, CTEs
 * - In-memory analytics at 100M+ rows/second
 *
 * Architecture:
 *   ArrowDataset -> Register -> DuckDB -> SQL Query -> Arrow Result
 *
 * Example:
 *   DuckDBConnector db;
 *   db.RegisterTable("sales", sales_table);
 *   db.RegisterTable("customers", customer_table);
 *   auto result = db.Query(R"(
 *       SELECT c.name, SUM(s.amount) as total
 *       FROM sales s
 *       JOIN customers c ON s.customer_id = c.id
 *       GROUP BY c.name
 *       ORDER BY total DESC
 *       LIMIT 10
 *   )");
 */
class DuckDBConnector {
public:
    /**
     * Constructor - creates in-memory DuckDB instance
     */
    DuckDBConnector();

    /**
     * Destructor - closes connection
     */
    ~DuckDBConnector();

    // Prevent copying (DuckDB connection is not copyable)
    DuckDBConnector(const DuckDBConnector&) = delete;
    DuckDBConnector& operator=(const DuckDBConnector&) = delete;

    /**
     * Register Arrow table for SQL queries.
     *
     * @param table_name SQL table identifier
     * @param arrow_table Arrow table to register
     * @return true on success
     *
     * Note: The current DuckDB 1.4.x adapter copies supported Arrow scalar
     * values into an in-memory DuckDB table. This avoids the version-specific
     * arrow_scan pointer ABI while preserving SQL transform behavior.
     */
    bool RegisterTable(const std::string& table_name,
                       std::shared_ptr<arrow::Table> arrow_table);

    /**
     * Unregister table (removes from DuckDB)
     */
    bool UnregisterTable(const std::string& table_name);

    /**
     * Execute SQL query and return result as Arrow table
     *
     * @param sql SQL query (SELECT, WITH, etc.)
     * @return Arrow table with query results, or nullptr on error
     *
     * Supports:
     * - SELECT, FROM, WHERE, GROUP BY, ORDER BY, LIMIT
     * - JOINs (INNER, LEFT, RIGHT, FULL)
     * - Window functions (ROW_NUMBER, RANK, LAG, LEAD)
     * - CTEs (WITH clauses)
     * - Aggregations (SUM, AVG, COUNT, MIN, MAX, STDDEV, etc.)
     * - String functions (LIKE, REGEXP, SUBSTRING, etc.)
     * - Date/time functions
     */
    std::shared_ptr<arrow::Table> Query(const std::string& sql);

    /**
     * Execute SQL query that doesn't return data (CREATE, INSERT, UPDATE, DELETE)
     *
     * @param sql SQL statement
     * @return true on success
     */
    bool Execute(const std::string& sql);

    /**
     * Get list of registered tables
     */
    std::vector<std::string> GetTableNames() const;

    /**
     * Get table schema (column names and types)
     */
    struct ColumnInfo {
        std::string name;
        std::string type;
        bool nullable;
    };
    std::vector<ColumnInfo> GetTableSchema(const std::string& table_name);

    /**
     * Get row count for a table
     */
    int64_t GetRowCount(const std::string& table_name);

    /**
     * Check if table exists
     */
    bool HasTable(const std::string& table_name) const;

    /**
     * Get last error message
     */
    std::string GetLastError() const { return last_error_; }

    /**
     * Query profiling information
     */
    struct QueryProfile {
        std::string query;
        double execution_time_ms;
        int64_t rows_scanned;
        int64_t rows_returned;
        std::string plan;
    };

    /**
     * Get query execution plan (EXPLAIN)
     */
    std::string ExplainQuery(const std::string& sql);

    /**
     * Profile query execution
     */
    QueryProfile ProfileQuery(const std::string& sql);

    /**
     * Enable/disable query profiling
     */
    void SetProfilingEnabled(bool enabled);

    /**
     * Common Data Studio Operations (convenience methods)
     */

    /**
     * Filter rows by condition
     * @return Filtered Arrow table
     */
    std::shared_ptr<arrow::Table> FilterRows(
        const std::string& table_name,
        const std::string& condition
    );

    /**
     * Select specific columns
     */
    std::shared_ptr<arrow::Table> SelectColumns(
        const std::string& table_name,
        const std::vector<std::string>& columns
    );

    /**
     * Group by and aggregate
     *
     * @param table_name Source table
     * @param group_by_cols Columns to group by
     * @param aggregations Map of: output_column -> "AGG(input_column)"
     *
     * Example:
     *   auto result = db.GroupBy("sales", {"region"},
     *       {{"total_sales", "SUM(amount)"},
     *        {"avg_price", "AVG(price)"}});
     */
    std::shared_ptr<arrow::Table> GroupBy(
        const std::string& table_name,
        const std::vector<std::string>& group_by_cols,
        const std::map<std::string, std::string>& aggregations
    );

    /**
     * Join two tables
     */
    enum class JoinType { Inner, Left, Right, Full };

    std::shared_ptr<arrow::Table> Join(
        const std::string& left_table,
        const std::string& right_table,
        const std::string& on_clause,
        JoinType join_type = JoinType::Inner
    );

    /**
     * Sort table
     */
    std::shared_ptr<arrow::Table> Sort(
        const std::string& table_name,
        const std::vector<std::string>& order_by_cols,
        bool ascending = true
    );

    /**
     * Distinct rows
     */
    std::shared_ptr<arrow::Table> Distinct(
        const std::string& table_name,
        const std::vector<std::string>& columns = {}
    );

    /**
     * Sample rows (for data preview)
     */
    std::shared_ptr<arrow::Table> Sample(
        const std::string& table_name,
        int64_t n,
        bool random = false
    );

    /**
     * Compute column statistics
     */
    struct ColumnStats {
        std::string column_name;
        int64_t count;
        int64_t null_count;
        double min;
        double max;
        double mean;
        double median;
        double std_dev;
        double percentile_25;
        double percentile_75;
    };

    std::vector<ColumnStats> ComputeStats(const std::string& table_name);

private:
    std::unique_ptr<duckdb::DuckDB> db_;
    std::unique_ptr<duckdb::Connection> conn_;
    std::map<std::string, std::shared_ptr<arrow::Table>> registered_tables_;
    std::string last_error_;
    bool profiling_enabled_ = false;

    // Helper: Convert DuckDB result to Arrow table
    std::shared_ptr<arrow::Table> ResultToArrow(duckdb::unique_ptr<duckdb::QueryResult> result);

    // Helper: Convert JoinType enum to SQL string
    std::string JoinTypeToSQL(JoinType type) const;
};

} // namespace cyxwiz
