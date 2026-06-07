#include "duckdb_connector.h"
#include <spdlog/spdlog.h>
#include <chrono>
#include <sstream>

namespace cyxwiz {
namespace {

std::string QuoteIdentifier(const std::string& identifier) {
    std::string quoted = "\"";
    for (char c : identifier) {
        if (c == '"') {
            quoted += "\"\"";
        } else {
            quoted += c;
        }
    }
    quoted += '"';
    return quoted;
}

std::string DuckDBTypeForArrowType(const std::shared_ptr<arrow::DataType>& type) {
    switch (type->id()) {
        case arrow::Type::BOOL:
            return "BOOLEAN";
        case arrow::Type::INT8:
        case arrow::Type::UINT8:
        case arrow::Type::INT16:
        case arrow::Type::UINT16:
        case arrow::Type::INT32:
            return "INTEGER";
        case arrow::Type::UINT32:
        case arrow::Type::INT64:
            return "BIGINT";
        case arrow::Type::UINT64:
            return "HUGEINT";
        case arrow::Type::FLOAT:
            return "FLOAT";
        case arrow::Type::DOUBLE:
            return "DOUBLE";
        case arrow::Type::STRING:
        case arrow::Type::LARGE_STRING:
            return "VARCHAR";
        default:
            return "VARCHAR";
    }
}

duckdb::Value DuckDBValueFromArrowScalar(const std::shared_ptr<arrow::Scalar>& scalar) {
    if (!scalar || !scalar->is_valid) {
        return duckdb::Value();
    }

    switch (scalar->type->id()) {
        case arrow::Type::BOOL:
            return duckdb::Value(
                std::static_pointer_cast<arrow::BooleanScalar>(scalar)->value);
        case arrow::Type::INT8:
            return duckdb::Value::INTEGER(
                std::static_pointer_cast<arrow::Int8Scalar>(scalar)->value);
        case arrow::Type::UINT8:
            return duckdb::Value::INTEGER(
                std::static_pointer_cast<arrow::UInt8Scalar>(scalar)->value);
        case arrow::Type::INT16:
            return duckdb::Value::INTEGER(
                std::static_pointer_cast<arrow::Int16Scalar>(scalar)->value);
        case arrow::Type::UINT16:
            return duckdb::Value::INTEGER(
                std::static_pointer_cast<arrow::UInt16Scalar>(scalar)->value);
        case arrow::Type::INT32:
            return duckdb::Value::INTEGER(
                std::static_pointer_cast<arrow::Int32Scalar>(scalar)->value);
        case arrow::Type::UINT32:
            return duckdb::Value::BIGINT(
                std::static_pointer_cast<arrow::UInt32Scalar>(scalar)->value);
        case arrow::Type::INT64:
            return duckdb::Value::BIGINT(
                std::static_pointer_cast<arrow::Int64Scalar>(scalar)->value);
        case arrow::Type::UINT64:
            return duckdb::Value::UHUGEINT(
                std::static_pointer_cast<arrow::UInt64Scalar>(scalar)->value);
        case arrow::Type::FLOAT:
            return duckdb::Value::FLOAT(
                std::static_pointer_cast<arrow::FloatScalar>(scalar)->value);
        case arrow::Type::DOUBLE:
            return duckdb::Value::DOUBLE(
                std::static_pointer_cast<arrow::DoubleScalar>(scalar)->value);
        case arrow::Type::STRING:
        case arrow::Type::LARGE_STRING:
            return duckdb::Value(scalar->ToString());
        default:
            return duckdb::Value(scalar->ToString());
    }
}

std::shared_ptr<arrow::DataType> ArrowTypeForDuckDBType(const duckdb::LogicalType& type) {
    switch (type.id()) {
        case duckdb::LogicalTypeId::BOOLEAN:
            return arrow::boolean();
        case duckdb::LogicalTypeId::TINYINT:
        case duckdb::LogicalTypeId::SMALLINT:
        case duckdb::LogicalTypeId::INTEGER:
            return arrow::int32();
        case duckdb::LogicalTypeId::BIGINT:
            return arrow::int64();
        case duckdb::LogicalTypeId::UTINYINT:
        case duckdb::LogicalTypeId::USMALLINT:
        case duckdb::LogicalTypeId::UINTEGER:
            return arrow::uint32();
        case duckdb::LogicalTypeId::UBIGINT:
            return arrow::uint64();
        case duckdb::LogicalTypeId::FLOAT:
            return arrow::float32();
        case duckdb::LogicalTypeId::DOUBLE:
            return arrow::float64();
        default:
            return arrow::utf8();
    }
}

std::unique_ptr<arrow::ArrayBuilder>
MakeArrowBuilderForDuckDBType(const duckdb::LogicalType& type) {
    switch (type.id()) {
        case duckdb::LogicalTypeId::BOOLEAN:
            return std::make_unique<arrow::BooleanBuilder>();
        case duckdb::LogicalTypeId::TINYINT:
        case duckdb::LogicalTypeId::SMALLINT:
        case duckdb::LogicalTypeId::INTEGER:
            return std::make_unique<arrow::Int32Builder>();
        case duckdb::LogicalTypeId::BIGINT:
            return std::make_unique<arrow::Int64Builder>();
        case duckdb::LogicalTypeId::UTINYINT:
        case duckdb::LogicalTypeId::USMALLINT:
        case duckdb::LogicalTypeId::UINTEGER:
            return std::make_unique<arrow::UInt32Builder>();
        case duckdb::LogicalTypeId::UBIGINT:
            return std::make_unique<arrow::UInt64Builder>();
        case duckdb::LogicalTypeId::FLOAT:
            return std::make_unique<arrow::FloatBuilder>();
        case duckdb::LogicalTypeId::DOUBLE:
            return std::make_unique<arrow::DoubleBuilder>();
        default:
            return std::make_unique<arrow::StringBuilder>();
    }
}

arrow::Status AppendDuckDBValueToArrowBuilder(
    const duckdb::Value& value,
    const duckdb::LogicalType& type,
    arrow::ArrayBuilder* builder) {

    if (value.IsNull()) {
        return builder->AppendNull();
    }

    switch (type.id()) {
        case duckdb::LogicalTypeId::BOOLEAN:
            return static_cast<arrow::BooleanBuilder*>(builder)->Append(
                value.GetValue<bool>());
        case duckdb::LogicalTypeId::TINYINT:
            return static_cast<arrow::Int32Builder*>(builder)->Append(
                value.GetValue<int8_t>());
        case duckdb::LogicalTypeId::SMALLINT:
            return static_cast<arrow::Int32Builder*>(builder)->Append(
                value.GetValue<int16_t>());
        case duckdb::LogicalTypeId::INTEGER:
            return static_cast<arrow::Int32Builder*>(builder)->Append(
                value.GetValue<int32_t>());
        case duckdb::LogicalTypeId::BIGINT:
            return static_cast<arrow::Int64Builder*>(builder)->Append(
                value.GetValue<int64_t>());
        case duckdb::LogicalTypeId::UTINYINT:
            return static_cast<arrow::UInt32Builder*>(builder)->Append(
                value.GetValue<uint8_t>());
        case duckdb::LogicalTypeId::USMALLINT:
            return static_cast<arrow::UInt32Builder*>(builder)->Append(
                value.GetValue<uint16_t>());
        case duckdb::LogicalTypeId::UINTEGER:
            return static_cast<arrow::UInt32Builder*>(builder)->Append(
                value.GetValue<uint32_t>());
        case duckdb::LogicalTypeId::UBIGINT:
            return static_cast<arrow::UInt64Builder*>(builder)->Append(
                value.GetValue<uint64_t>());
        case duckdb::LogicalTypeId::FLOAT:
            return static_cast<arrow::FloatBuilder*>(builder)->Append(
                value.GetValue<float>());
        case duckdb::LogicalTypeId::DOUBLE:
            return static_cast<arrow::DoubleBuilder*>(builder)->Append(
                value.GetValue<double>());
        default:
            return static_cast<arrow::StringBuilder*>(builder)->Append(value.ToString());
    }
}

} // namespace

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
        const std::string quoted_table_name = QuoteIdentifier(table_name);
        auto drop_view = conn_->Query("DROP VIEW IF EXISTS " + quoted_table_name);
        if (drop_view->HasError()) {
            last_error_ = drop_view->GetError();
            spdlog::error("Failed to drop existing view '{}': {}", table_name, last_error_);
            return false;
        }
        auto drop_table = conn_->Query("DROP TABLE IF EXISTS " + quoted_table_name);
        if (drop_table->HasError()) {
            last_error_ = drop_table->GetError();
            spdlog::error("Failed to drop existing table '{}': {}", table_name, last_error_);
            return false;
        }

        std::ostringstream create_sql;
        create_sql << "CREATE TABLE " << quoted_table_name << " (";
        for (int i = 0; i < arrow_table->num_columns(); ++i) {
            if (i > 0) {
                create_sql << ", ";
            }
            const auto& field = arrow_table->schema()->field(i);
            create_sql << QuoteIdentifier(field->name()) << " "
                       << DuckDBTypeForArrowType(field->type());
        }
        create_sql << ")";

        auto create_result = conn_->Query(create_sql.str());
        if (create_result->HasError()) {
            last_error_ = create_result->GetError();
            spdlog::error("Failed to create table '{}': {}", table_name, last_error_);
            return false;
        }

        duckdb::Appender appender(*conn_, table_name);
        for (int64_t row = 0; row < arrow_table->num_rows(); ++row) {
            appender.BeginRow();
            for (int col = 0; col < arrow_table->num_columns(); ++col) {
                auto scalar_result = arrow_table->column(col)->GetScalar(row);
                if (!scalar_result.ok()) {
                    last_error_ = scalar_result.status().ToString();
                    spdlog::error("Failed to read Arrow scalar at row {}, column {}: {}",
                                  row, col, last_error_);
                    return false;
                }
                appender.Append(DuckDBValueFromArrowScalar(*scalar_result));
            }
            appender.EndRow();
        }
        appender.Close();

        // Store reference to keep Arrow table alive
        registered_tables_[table_name] = arrow_table;

        spdlog::info("Registered Arrow table '{}' as DuckDB table: {} rows, {} columns",
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
        const std::string quoted_table_name = QuoteIdentifier(table_name);
        conn_->Query("DROP VIEW IF EXISTS " + quoted_table_name);
        conn_->Query("DROP TABLE IF EXISTS " + quoted_table_name);

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
        std::vector<std::shared_ptr<arrow::Field>> fields;
        std::vector<std::shared_ptr<arrow::Array>> columns;

        // Get column count and types
        auto& types = result->types;
        auto& names = result->names;

        // Initialize builders for each column
        std::vector<std::unique_ptr<arrow::ArrayBuilder>> builders;
        for (size_t i = 0; i < types.size(); ++i) {
            fields.push_back(arrow::field(names[i], ArrowTypeForDuckDBType(types[i])));
            builders.push_back(MakeArrowBuilderForDuckDBType(types[i]));
        }

        // Iterate through result chunks
        while (true) {
            auto chunk = result->Fetch();
            if (!chunk || chunk->size() == 0) break;

            for (idx_t row = 0; row < chunk->size(); ++row) {
                for (size_t col = 0; col < chunk->ColumnCount(); ++col) {
                    auto value = chunk->GetValue(col, row);
                    auto status = AppendDuckDBValueToArrowBuilder(
                        value, types[col], builders[col].get());
                    if (!status.ok()) {
                        spdlog::error("Failed to append DuckDB result value: {}",
                                      status.ToString());
                        return nullptr;
                    }
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
