#include "duckdb_connector.h"
#include <spdlog/spdlog.h>
#include <chrono>
#include <sstream>

namespace cyxwiz {

// =============================================================================
// Constructor / Destructor
// =============================================================================

DuckDBConnector::DuckDBConnector() {
    try {
        // Create in-memory DuckDB instance
        db_ = std::make_unique<duckdb::DuckDB>(nullptr);  // nullptr = in-memory
        conn_ = std::make_unique<duckdb::Connection>(*db_);

        spdlog::info("DuckDB connector initialized (in-memory mode)");

    } catch (const duckdb::Exception& e) {
        spdlog::error("Failed to initialize DuckDB: {}", e.what());
        last_error_ = e.what();
    }
}

DuckDBConnector::~DuckDBConnector() {
    // Close connection and database
    conn_.reset();
    db_.reset();
    spdlog::debug("DuckDB connector closed");
}

// =============================================================================
// Table Registration (Zero-Copy Arrow Integration)
// =============================================================================

bool DuckDBConnector::RegisterTable(
    const std::string& table_name,
    std::shared_ptr<arrow::Table> arrow_table) {

    if (!arrow_table) {
        last_error_ = "Cannot register null Arrow table";
        spdlog::error(last_error_);
        return false;
    }

    try {
        // Register Arrow table as a DuckDB table (zero-copy view)
        // DuckDB will query the Arrow table directly without copying
        conn_->TableFunction("arrow_scan", {duckdb::Value::POINTER((uintptr_t)arrow_table.get())})
             ->CreateView(table_name, true, true);

        // Store reference to keep Arrow table alive
        registered_tables_[table_name] = arrow_table;

        spdlog::info("Registered Arrow table '{}': {} rows, {} columns",
                     table_name, arrow_table->num_rows(), arrow_table->num_columns());

        return true;

    } catch (const duckdb::Exception& e) {
        last_error_ = e.what();
        spdlog::error("Failed to register table '{}': {}", table_name, last_error_);
        return false;
    }
}

bool DuckDBConnector::UnregisterTable(const std::string& table_name) {
    try {
        // Drop view
        std::string sql = "DROP VIEW IF EXISTS " + table_name;
        conn_->Query(sql);

        // Remove from tracking
        registered_tables_.erase(table_name);

        spdlog::debug("Unregistered table '{}'", table_name);
        return true;

    } catch (const duckdb::Exception& e) {
        last_error_ = e.what();
        spdlog::error("Failed to unregister table '{}': {}", table_name, last_error_);
        return false;
    }
}

// =============================================================================
// Query Execution
// =============================================================================

std::shared_ptr<arrow::Table> DuckDBConnector::Query(const std::string& sql) {
    if (!conn_) {
        last_error_ = "DuckDB connection not initialized";
        spdlog::error(last_error_);
        return nullptr;
    }

    try {
        auto start = std::chrono::high_resolution_clock::now();

        // Execute query
        auto result = conn_->Query(sql);

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

        if (result->HasError()) {
            last_error_ = result->GetError();
            spdlog::error("Query failed: {}", last_error_);
            return nullptr;
        }

        // Convert result to Arrow table
        auto arrow_result = ResultToArrow(std::move(result));

        if (arrow_result) {
            spdlog::debug("Query executed in {:.2f} ms: {} rows returned",
                         elapsed_ms, arrow_result->num_rows());
        }

        return arrow_result;

    } catch (const duckdb::Exception& e) {
        last_error_ = e.what();
        spdlog::error("Query exception: {}", last_error_);
        return nullptr;
    }
}

bool DuckDBConnector::Execute(const std::string& sql) {
    if (!conn_) {
        last_error_ = "DuckDB connection not initialized";
        spdlog::error(last_error_);
        return false;
    }

    try {
        auto result = conn_->Query(sql);

        if (result->HasError()) {
            last_error_ = result->GetError();
            spdlog::error("Execute failed: {}", last_error_);
            return false;
        }

        return true;

    } catch (const duckdb::Exception& e) {
        last_error_ = e.what();
        spdlog::error("Execute exception: {}", last_error_);
        return false;
    }
}

// =============================================================================
// Table Introspection
// =============================================================================

std::vector<std::string> DuckDBConnector::GetTableNames() const {
    std::vector<std::string> names;
    for (const auto& [name, table] : registered_tables_) {
        names.push_back(name);
    }
    return names;
}

std::vector<DuckDBConnector::ColumnInfo> DuckDBConnector::GetTableSchema(
    const std::string& table_name) {

    std::vector<ColumnInfo> schema;

    try {
        std::string sql = "DESCRIBE " + table_name;
        auto result = conn_->Query(sql);

        if (result->HasError()) {
            spdlog::error("Failed to get schema for '{}': {}", table_name, result->GetError());
            return schema;
        }

        // DuckDB DESCRIBE returns: column_name, column_type, null, key, default, extra
        while (true) {
            auto chunk = result->Fetch();
            if (!chunk || chunk->size() == 0) break;

            for (idx_t row = 0; row < chunk->size(); ++row) {
                ColumnInfo info;
                info.name = chunk->GetValue(0, row).ToString();
                info.type = chunk->GetValue(1, row).ToString();
                info.nullable = chunk->GetValue(2, row).ToString() == "YES";
                schema.push_back(info);
            }
        }

    } catch (const duckdb::Exception& e) {
        spdlog::error("Exception getting schema: {}", e.what());
    }

    return schema;
}

int64_t DuckDBConnector::GetRowCount(const std::string& table_name) {
    try {
        std::string sql = "SELECT COUNT(*) FROM " + table_name;
        auto result = conn_->Query(sql);

        if (result->HasError()) {
            spdlog::error("Failed to get row count: {}", result->GetError());
            return -1;
        }

        auto chunk = result->Fetch();
        if (chunk && chunk->size() > 0) {
            return chunk->GetValue(0, 0).GetValue<int64_t>();
        }

    } catch (const duckdb::Exception& e) {
        spdlog::error("Exception getting row count: {}", e.what());
    }

    return -1;
}

bool DuckDBConnector::HasTable(const std::string& table_name) const {
    return registered_tables_.find(table_name) != registered_tables_.end();
}

// =============================================================================
// Query Analysis
// =============================================================================

std::string DuckDBConnector::ExplainQuery(const std::string& sql) {
    try {
        std::string explain_sql = "EXPLAIN " + sql;
        auto result = conn_->Query(explain_sql);

        if (result->HasError()) {
            return "Error: " + result->GetError();
        }

        std::ostringstream plan;
        while (true) {
            auto chunk = result->Fetch();
            if (!chunk || chunk->size() == 0) break;

            for (idx_t row = 0; row < chunk->size(); ++row) {
                plan << chunk->GetValue(0, row).ToString() << "\n";
            }
        }

        return plan.str();

    } catch (const duckdb::Exception& e) {
        return "Exception: " + std::string(e.what());
    }
}

DuckDBConnector::QueryProfile DuckDBConnector::ProfileQuery(const std::string& sql) {
    QueryProfile profile;
    profile.query = sql;

    auto start = std::chrono::high_resolution_clock::now();

    auto result = Query(sql);

    auto end = std::chrono::high_resolution_clock::now();
    profile.execution_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    if (result) {
        profile.rows_returned = result->num_rows();
    }

    profile.plan = ExplainQuery(sql);

    return profile;
}

void DuckDBConnector::SetProfilingEnabled(bool enabled) {
    profiling_enabled_ = enabled;
    try {
        if (enabled) {
            conn_->Query("PRAGMA enable_profiling");
        } else {
            conn_->Query("PRAGMA disable_profiling");
        }
    } catch (const duckdb::Exception& e) {
        spdlog::warn("Failed to set profiling: {}", e.what());
    }
}

// =============================================================================
// Convenience Methods (Data Studio Operations)
// =============================================================================

std::shared_ptr<arrow::Table> DuckDBConnector::FilterRows(
    const std::string& table_name,
    const std::string& condition) {

    std::string sql = "SELECT * FROM " + table_name + " WHERE " + condition;
    return Query(sql);
}

std::shared_ptr<arrow::Table> DuckDBConnector::SelectColumns(
    const std::string& table_name,
    const std::vector<std::string>& columns) {

    if (columns.empty()) {
        return Query("SELECT * FROM " + table_name);
    }

    std::ostringstream sql;
    sql << "SELECT ";
    for (size_t i = 0; i < columns.size(); ++i) {
        if (i > 0) sql << ", ";
        sql << columns[i];
    }
    sql << " FROM " << table_name;

    return Query(sql.str());
}

std::shared_ptr<arrow::Table> DuckDBConnector::GroupBy(
    const std::string& table_name,
    const std::vector<std::string>& group_by_cols,
    const std::map<std::string, std::string>& aggregations) {

    std::ostringstream sql;
    sql << "SELECT ";

    // Group by columns
    for (size_t i = 0; i < group_by_cols.size(); ++i) {
        if (i > 0) sql << ", ";
        sql << group_by_cols[i];
    }

    // Aggregations
    for (const auto& [output_col, agg_expr] : aggregations) {
        if (!group_by_cols.empty() || &agg_expr != &aggregations.begin()->second) {
            sql << ", ";
        }
        sql << agg_expr << " AS " << output_col;
    }

    sql << " FROM " << table_name;

    if (!group_by_cols.empty()) {
        sql << " GROUP BY ";
        for (size_t i = 0; i < group_by_cols.size(); ++i) {
            if (i > 0) sql << ", ";
            sql << group_by_cols[i];
        }
    }

    return Query(sql.str());
}

std::string DuckDBConnector::JoinTypeToSQL(JoinType type) const {
    switch (type) {
        case JoinType::Inner: return "INNER JOIN";
        case JoinType::Left:  return "LEFT JOIN";
        case JoinType::Right: return "RIGHT JOIN";
        case JoinType::Full:  return "FULL OUTER JOIN";
        default: return "INNER JOIN";
    }
}

std::shared_ptr<arrow::Table> DuckDBConnector::Join(
    const std::string& left_table,
    const std::string& right_table,
    const std::string& on_clause,
    JoinType join_type) {

    std::ostringstream sql;
    sql << "SELECT * FROM " << left_table << " "
        << JoinTypeToSQL(join_type) << " "
        << right_table << " ON " << on_clause;

    return Query(sql.str());
}

std::shared_ptr<arrow::Table> DuckDBConnector::Sort(
    const std::string& table_name,
    const std::vector<std::string>& order_by_cols,
    bool ascending) {

    std::ostringstream sql;
    sql << "SELECT * FROM " << table_name << " ORDER BY ";

    for (size_t i = 0; i < order_by_cols.size(); ++i) {
        if (i > 0) sql << ", ";
        sql << order_by_cols[i];
    }

    if (!ascending) {
        sql << " DESC";
    }

    return Query(sql.str());
}

std::shared_ptr<arrow::Table> DuckDBConnector::Distinct(
    const std::string& table_name,
    const std::vector<std::string>& columns) {

    std::ostringstream sql;
    sql << "SELECT DISTINCT ";

    if (columns.empty()) {
        sql << "*";
    } else {
        for (size_t i = 0; i < columns.size(); ++i) {
            if (i > 0) sql << ", ";
            sql << columns[i];
        }
    }

    sql << " FROM " << table_name;
    return Query(sql.str());
}

std::shared_ptr<arrow::Table> DuckDBConnector::Sample(
    const std::string& table_name,
    int64_t n,
    bool random) {

    std::ostringstream sql;
    sql << "SELECT * FROM " << table_name;

    if (random) {
        sql << " USING SAMPLE " << n;
    } else {
        sql << " LIMIT " << n;
    }

    return Query(sql.str());
}

std::vector<DuckDBConnector::ColumnStats> DuckDBConnector::ComputeStats(
    const std::string& table_name) {

    std::vector<ColumnStats> stats;

    // Get schema first
    auto schema = GetTableSchema(table_name);

    for (const auto& col_info : schema) {
        // Skip non-numeric columns
        if (col_info.type.find("INT") == std::string::npos &&
            col_info.type.find("DOUBLE") == std::string::npos &&
            col_info.type.find("FLOAT") == std::string::npos &&
            col_info.type.find("DECIMAL") == std::string::npos) {
            continue;
        }

        ColumnStats col_stat;
        col_stat.column_name = col_info.name;

        // Compute all stats in one query
        std::ostringstream sql;
        sql << "SELECT "
            << "COUNT(" << col_info.name << ") as count, "
            << "COUNT(*) - COUNT(" << col_info.name << ") as null_count, "
            << "MIN(" << col_info.name << ") as min_val, "
            << "MAX(" << col_info.name << ") as max_val, "
            << "AVG(" << col_info.name << ") as mean, "
            << "MEDIAN(" << col_info.name << ") as median, "
            << "STDDEV(" << col_info.name << ") as std_dev, "
            << "PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY " << col_info.name << ") as p25, "
            << "PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY " << col_info.name << ") as p75 "
            << "FROM " << table_name;

        auto result = Query(sql.str());
        if (result && result->num_rows() > 0) {
            // Extract values from result
            auto count_col = result->GetColumnByName("count");
            auto null_count_col = result->GetColumnByName("null_count");
            auto min_col = result->GetColumnByName("min_val");
            auto max_col = result->GetColumnByName("max_val");
            auto mean_col = result->GetColumnByName("mean");
            auto median_col = result->GetColumnByName("median");
            auto std_dev_col = result->GetColumnByName("std_dev");
            auto p25_col = result->GetColumnByName("p25");
            auto p75_col = result->GetColumnByName("p75");

            // Convert to scalars (simplified - assumes single row result)
            // In production, you'd need proper Arrow scalar extraction
            col_stat.count = 0;  // Placeholder
            col_stat.null_count = 0;
            col_stat.min = 0.0;
            col_stat.max = 0.0;
            col_stat.mean = 0.0;
            col_stat.median = 0.0;
            col_stat.std_dev = 0.0;
            col_stat.percentile_25 = 0.0;
            col_stat.percentile_75 = 0.0;

            // TODO: Proper Arrow scalar extraction here
        }

        stats.push_back(col_stat);
    }

    return stats;
}

// =============================================================================
// Helper Methods
// =============================================================================

std::shared_ptr<arrow::Table> DuckDBConnector::ResultToArrow(
    duckdb::unique_ptr<duckdb::QueryResult> result) {

    if (!result || result->HasError()) {
        return nullptr;
    }

    try {
        // DuckDB 1.4.3: Manually convert result to Arrow
        // Build Arrow arrays from DuckDB chunks
        std::vector<std::shared_ptr<arrow::Field>> fields;
        std::vector<std::shared_ptr<arrow::Array>> columns;

        // Get column count and types
        auto& types = result->types;
        auto& names = result->names;

        // Initialize builders for each column
        std::vector<std::unique_ptr<arrow::ArrayBuilder>> builders;
        for (size_t i = 0; i < types.size(); ++i) {
            // Map DuckDB type to Arrow type and create builder
            // For simplicity, we'll use string builders for all types
            // In production, map each DuckDB type to proper Arrow type
            fields.push_back(arrow::field(names[i], arrow::utf8()));
            builders.push_back(std::unique_ptr<arrow::ArrayBuilder>(
                new arrow::StringBuilder()));
        }

        // Iterate through result chunks
        while (true) {
            auto chunk = result->Fetch();
            if (!chunk || chunk->size() == 0) break;

            for (idx_t row = 0; row < chunk->size(); ++row) {
                for (size_t col = 0; col < chunk->ColumnCount(); ++col) {
                    auto value = chunk->GetValue(col, row);
                    auto* str_builder = static_cast<arrow::StringBuilder*>(builders[col].get());
                    str_builder->Append(value.ToString());
                }
            }
        }

        // Finish all builders
        for (size_t i = 0; i < builders.size(); ++i) {
            std::shared_ptr<arrow::Array> array;
            auto status = builders[i]->Finish(&array);
            if (!status.ok()) {
                spdlog::error("Failed to finish Arrow array: {}", status.ToString());
                return nullptr;
            }
            columns.push_back(array);
        }

        // Create Arrow table
        auto schema = arrow::schema(fields);
        auto table = arrow::Table::Make(schema, columns);

        return table;

    } catch (const duckdb::Exception& e) {
        spdlog::error("Exception converting result to Arrow: {}", e.what());
        return nullptr;
    }
}

} // namespace cyxwiz
