#include "duckdb_connector.h"
#include <arrow/c/bridge.h>
#include <duckdb/common/arrow/arrow_converter.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <unordered_map>
#include <chrono>
#include <sstream>

namespace cyxwiz {
namespace {

// DuckDB 1.x exceptions carry their message as JSON in what(); ErrorData
// turns it back into the readable "Parser Error: ..." text.
std::string ErrorText(const std::exception& e) {
    return duckdb::ErrorData(e).Message();
}

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

// "VIEW", "BASE TABLE", or empty when no object in the main schema has this
// name. DROP VIEW IF EXISTS fails when the name is a table (and DROP TABLE
// when it is a view), so callers drop by the kind found here.
std::string ExistingObjectKind(duckdb::Connection& conn, const std::string& name) {
    auto lookup = conn.Prepare(
        "SELECT table_type FROM information_schema.tables WHERE table_schema = 'main' AND table_name = $1");
    if (lookup->HasError()) return {};
    auto found = lookup->Execute(name);
    if (found->HasError()) return {};
    auto chunk = found->Fetch();
    return chunk && chunk->size() > 0 ? chunk->GetValue(0, 0).ToString() : std::string();
}

// The one table of result types (TOFIX148): how a result column of each
// DuckDB type comes back as Arrow. `native`: DuckDB's Arrow export gives
// this exact Arrow type, so whole chunks go through the C interface;
// otherwise values are appended one by one. `exact`: every value comes
// back unchanged (the SQL step refuses result columns that would not).
struct ResultType {
    std::shared_ptr<arrow::DataType> arrow;
    bool native;
    bool exact;
};

ResultType ResultTypeFor(const duckdb::LogicalType& type) {
    switch (type.id()) {
        case duckdb::LogicalTypeId::BOOLEAN: return {arrow::boolean(), true, true};
        case duckdb::LogicalTypeId::TINYINT:
        case duckdb::LogicalTypeId::SMALLINT: return {arrow::int32(), false, true};
        case duckdb::LogicalTypeId::INTEGER: return {arrow::int32(), true, true};
        case duckdb::LogicalTypeId::BIGINT: return {arrow::int64(), true, true};
        case duckdb::LogicalTypeId::UTINYINT:
        case duckdb::LogicalTypeId::USMALLINT:
        case duckdb::LogicalTypeId::UINTEGER: return {arrow::uint32(), false, true};
        case duckdb::LogicalTypeId::UBIGINT: return {arrow::uint64(), true, true};
        case duckdb::LogicalTypeId::FLOAT: return {arrow::float32(), true, true};
        case duckdb::LogicalTypeId::DOUBLE: return {arrow::float64(), true, true};
        case duckdb::LogicalTypeId::VARCHAR: return {arrow::utf8(), true, true};
        // 0.1, 1234567.5, SUM of decimals: numbers, as double (not exact past 15 digits).
        case duckdb::LogicalTypeId::DECIMAL: return {arrow::float64(), false, false};
        // Dates, timestamps, HUGEINT, BLOB, nested types: their text.
        default: return {arrow::utf8(), false, false};
    }
}

bool ConvertsNatively(const duckdb::LogicalType& type) { return ResultTypeFor(type).native; }

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
        // Nested columns (TOFIX143): lists and structs keep their structure so
        // SQL can read them; previously they arrived as their text form.
        case arrow::Type::LIST:
            return DuckDBTypeForArrowType(
                std::static_pointer_cast<arrow::ListType>(type)->value_type()) + "[]";
        case arrow::Type::STRUCT: {
            std::string sql = "STRUCT(";
            for (int i = 0; i < type->num_fields(); ++i) {
                if (i > 0) sql += ", ";
                sql += QuoteIdentifier(type->field(i)->name()) + " " +
                       DuckDBTypeForArrowType(type->field(i)->type());
            }
            return sql + ")";
        }
        default:
            return "VARCHAR";
    }
}

duckdb::LogicalType DuckDBLogicalTypeForArrowType(const std::shared_ptr<arrow::DataType>& type) {
    switch (type->id()) {
        case arrow::Type::BOOL:
            return duckdb::LogicalType::BOOLEAN;
        case arrow::Type::INT8:
        case arrow::Type::UINT8:
        case arrow::Type::INT16:
        case arrow::Type::UINT16:
        case arrow::Type::INT32:
            return duckdb::LogicalType::INTEGER;
        case arrow::Type::UINT32:
        case arrow::Type::INT64:
            return duckdb::LogicalType::BIGINT;
        case arrow::Type::UINT64:
            return duckdb::LogicalType::HUGEINT;
        case arrow::Type::FLOAT:
            return duckdb::LogicalType::FLOAT;
        case arrow::Type::DOUBLE:
            return duckdb::LogicalType::DOUBLE;
        case arrow::Type::LIST:
            return duckdb::LogicalType::LIST(DuckDBLogicalTypeForArrowType(
                std::static_pointer_cast<arrow::ListType>(type)->value_type()));
        case arrow::Type::STRUCT: {
            duckdb::child_list_t<duckdb::LogicalType> children;
            for (const auto& field : type->fields())
                children.emplace_back(field->name(), DuckDBLogicalTypeForArrowType(field->type()));
            return duckdb::LogicalType::STRUCT(std::move(children));
        }
        default:
            return duckdb::LogicalType::VARCHAR;
    }
}

duckdb::Value DuckDBValueFromArrowScalar(const std::shared_ptr<arrow::Scalar>& scalar);

// Elements and fields carry typed NULLs so a list/struct value matches its column type.
duckdb::Value DuckDBNestedValue(const std::shared_ptr<arrow::Scalar>& scalar,
                                const std::shared_ptr<arrow::DataType>& type) {
    if (!scalar || !scalar->is_valid) return duckdb::Value(DuckDBLogicalTypeForArrowType(type));
    return DuckDBValueFromArrowScalar(scalar);
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
        case arrow::Type::LIST: {
            const auto& list = static_cast<const arrow::ListScalar&>(*scalar);
            const auto child_type = std::static_pointer_cast<arrow::ListType>(scalar->type)->value_type();
            std::vector<duckdb::Value> values;
            values.reserve(static_cast<size_t>(list.value->length()));
            for (int64_t i = 0; i < list.value->length(); ++i) {
                auto element = list.value->GetScalar(i);
                if (!element.ok()) throw std::runtime_error(element.status().ToString());
                values.push_back(DuckDBNestedValue(*element, child_type));
            }
            return duckdb::Value::LIST(DuckDBLogicalTypeForArrowType(child_type), std::move(values));
        }
        case arrow::Type::STRUCT: {
            const auto& item = static_cast<const arrow::StructScalar&>(*scalar);
            duckdb::child_list_t<duckdb::Value> children;
            for (int i = 0; i < scalar->type->num_fields(); ++i)
                children.emplace_back(scalar->type->field(i)->name(),
                                      DuckDBNestedValue(item.value[static_cast<size_t>(i)],
                                                        scalar->type->field(i)->type()));
            return duckdb::Value::STRUCT(std::move(children));
        }
        default:
            return duckdb::Value(scalar->ToString());
    }
}

// One value into the builder ResultTypeFor chose for its column.
arrow::Status AppendDuckDBValueToArrowBuilder(const duckdb::Value& value, arrow::ArrayBuilder* builder) {
    if (value.IsNull()) return builder->AppendNull();
    switch (builder->type()->id()) {
        case arrow::Type::BOOL: return static_cast<arrow::BooleanBuilder*>(builder)->Append(value.GetValue<bool>());
        case arrow::Type::INT32: return static_cast<arrow::Int32Builder*>(builder)->Append(value.GetValue<int32_t>());
        case arrow::Type::INT64: return static_cast<arrow::Int64Builder*>(builder)->Append(value.GetValue<int64_t>());
        case arrow::Type::UINT32: return static_cast<arrow::UInt32Builder*>(builder)->Append(value.GetValue<uint32_t>());
        case arrow::Type::UINT64: return static_cast<arrow::UInt64Builder*>(builder)->Append(value.GetValue<uint64_t>());
        case arrow::Type::FLOAT: return static_cast<arrow::FloatBuilder*>(builder)->Append(value.GetValue<float>());
        case arrow::Type::DOUBLE: return static_cast<arrow::DoubleBuilder*>(builder)->Append(value.GetValue<double>());
        default: return static_cast<arrow::StringBuilder*>(builder)->Append(value.ToString());
    }
}

} // namespace

// =============================================================================
// Constructor / Destructor
// =============================================================================

DuckDBConnector::DuckDBConnector(std::vector<std::string> allowed_directories) {
    try {
        duckdb::DBConfig config;
        config.SetOptionByName("enable_external_access", duckdb::Value::BOOLEAN(false));
        config.SetOptionByName("autoinstall_known_extensions", duckdb::Value::BOOLEAN(false));
        config.SetOptionByName("autoload_known_extensions", duckdb::Value::BOOLEAN(false));
        for (auto& dir : allowed_directories) config.options.allowed_directories.insert(std::move(dir));
        config.SetOptionByName("lock_configuration", duckdb::Value::BOOLEAN(true));
        db_ = std::make_unique<duckdb::DuckDB>(nullptr, &config);
        conn_ = std::make_unique<duckdb::Connection>(*db_);
        spdlog::debug("DuckDB connector initialized (in-memory, no file or network access)");
    } catch (const std::exception& e) {
        spdlog::error("Failed to initialize DuckDB: {}", ErrorText(e));
        last_error_ = ErrorText(e);
        conn_.reset();
        db_.reset();
    }
}

std::string DuckDBConnector::ReadOnlySelectRejection(const std::string& sql) {
    if (!conn_) return "DuckDB connection not initialized";
    try {
        auto statements = conn_->ExtractStatements(sql);
        if (statements.size() != 1) {
            return "the SQL step takes exactly one statement; found " +
                   std::to_string(statements.size());
        }
        if (statements.front()->type != duckdb::StatementType::SELECT_STATEMENT) {
            return "only a read-only SELECT (or WITH ... SELECT) is allowed in a SQL step";
        }
        return {};
    } catch (const std::exception& e) {
        return std::string("the SQL does not parse: ") + ErrorText(e);
    }
}

std::string DuckDBConnector::UnsupportedResultColumns(const std::string& sql) {
    if (!conn_) return "DuckDB connection not initialized";
    try {
        auto prepared = conn_->Prepare(sql);
        if (prepared->HasError()) return prepared->GetError();
        const auto& names = prepared->GetNames();
        const auto& types = prepared->GetTypes();
        std::string unsupported;
        for (size_t i = 0; i < types.size(); ++i) {
            if (ResultTypeFor(types[i]).exact) continue;
            unsupported += (unsupported.empty() ? "" : ", ") + std::string("'") + names[i] + "' (" +
                           types[i].ToString() + ")";
        }
        if (!unsupported.empty()) {
            return "result column(s) " + unsupported +
                   " have a type that cannot be returned exactly; CAST them in the query "
                   "(for example to VARCHAR, BIGINT or DOUBLE)";
        }
        return {};
    } catch (const std::exception& e) {
        return ErrorText(e);
    }
}

void DuckDBConnector::Interrupt() {
    if (conn_) conn_->Interrupt();
}

DuckDBConnector::DuckDBConnector(SharedTag, std::shared_ptr<duckdb::DuckDB> db) : db_(std::move(db)) {
    try {
        conn_ = std::make_unique<duckdb::Connection>(*db_);
    } catch (const std::exception& e) {
        last_error_ = e.what();
        conn_.reset();
    }
}

std::unique_ptr<DuckDBConnector> DuckDBConnector::NewConnection() {
    if (!db_ || !conn_) return nullptr;
    std::unique_ptr<DuckDBConnector> c(new DuckDBConnector(SharedTag{}, db_));
    return c->IsReady() ? std::move(c) : nullptr;
}

DuckDBConnector::~DuckDBConnector() {
    // Close connection and database
    conn_.reset();
    db_.reset();
    spdlog::debug("DuckDB connector closed");
}

// =============================================================================
// Table Registration (the rows are copied into a DuckDB table)
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
        // Replace whatever holds the name (a table registered earlier on
        // this connection, or a view).
        const std::string existing = ExistingObjectKind(*conn_, table_name);
        if (!existing.empty()) {
            const char* kind = existing == "VIEW" ? "VIEW" : "TABLE";
            auto drop = conn_->Query(std::string("DROP ") + kind + " " + quoted_table_name);
            if (drop->HasError()) {
                last_error_ = drop->GetError();
                spdlog::error("Failed to drop existing {} '{}': {}", existing, table_name, last_error_);
                return false;
            }
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

        spdlog::info("Registered Arrow table '{}' as DuckDB table: {} rows, {} columns",
                     table_name, arrow_table->num_rows(), arrow_table->num_columns());

        return true;

    } catch (const duckdb::Exception& e) {
        last_error_ = ErrorText(e);
        spdlog::error("Failed to register table '{}': {}", table_name, last_error_);
        return false;
    }
}

bool DuckDBConnector::UnregisterTable(const std::string& table_name) {
    try {
        const std::string existing = ExistingObjectKind(*conn_, table_name);
        if (!existing.empty()) {
            conn_->Query(std::string(existing == "VIEW" ? "DROP VIEW " : "DROP TABLE ") +
                         QuoteIdentifier(table_name));
        }

        spdlog::debug("Unregistered table '{}'", table_name);
        return true;

    } catch (const duckdb::Exception& e) {
        last_error_ = ErrorText(e);
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
        last_error_ = ErrorText(e);
        spdlog::error("Query exception: {}", last_error_);
        return nullptr;
    }
}

std::shared_ptr<arrow::Table> DuckDBConnector::QueryWithParams(const std::string& sql, std::vector<duckdb::Value> params) {
    if (!conn_) {
        last_error_ = "DuckDB connection not initialized";
        return nullptr;
    }
    try {
        auto prepared = conn_->Prepare(sql);
        if (prepared->HasError()) {
            last_error_ = prepared->GetError();
            return nullptr;
        }
        duckdb::vector<duckdb::Value> values(params.begin(), params.end());
        auto result = prepared->Execute(values, false);
        if (!result || result->HasError()) {
            last_error_ = result ? result->GetError() : std::string("no result");
            return nullptr;
        }
        auto table = ResultToArrow(std::move(result));
        if (!table) last_error_ = "the result could not be converted to a table";
        return table;
    } catch (const std::exception& e) {
        last_error_ = e.what();
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
        last_error_ = ErrorText(e);
        spdlog::error("Execute exception: {}", last_error_);
        return false;
    }
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

        // Booleans, integers, floats, doubles and text: export whole chunks
        // through the Arrow C interface (same Arrow types as below).
        if (std::all_of(types.begin(), types.end(), ConvertsNatively)) {
            duckdb::ClientProperties options = result->client_properties;
            options.arrow_offset_size = duckdb::ArrowOffsetSize::REGULAR;
            options.produce_arrow_string_view = false;
            ArrowSchema c_schema;
            duckdb::ArrowConverter::ToArrowSchema(&c_schema, types, names, options);
            auto schema = arrow::ImportSchema(&c_schema);
            if (!schema.ok()) {
                spdlog::error("Arrow schema import failed: {}", schema.status().ToString());
                return nullptr;
            }
            const std::unordered_map<duckdb::idx_t, const duckdb::shared_ptr<duckdb::ArrowTypeExtensionData>> no_extensions;
            std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
            while (true) {
                auto chunk = result->Fetch();
                if (!chunk || chunk->size() == 0) break;
                ArrowArray c_array;
                duckdb::ArrowConverter::ToArrowArray(*chunk, &c_array, options, no_extensions);
                auto batch = arrow::ImportRecordBatch(&c_array, *schema);
                if (!batch.ok()) {
                    spdlog::error("Arrow batch import failed: {}", batch.status().ToString());
                    return nullptr;
                }
                batches.push_back(std::move(*batch));
            }
            auto table = arrow::Table::FromRecordBatches(*schema, batches);
            if (!table.ok()) {
                spdlog::error("Arrow table assembly failed: {}", table.status().ToString());
                return nullptr;
            }
            return *table;
        }

        // Initialize builders for each column
        std::vector<std::unique_ptr<arrow::ArrayBuilder>> builders;
        for (size_t i = 0; i < types.size(); ++i) {
            const auto arrow_type = ResultTypeFor(types[i]).arrow;
            fields.push_back(arrow::field(names[i], arrow_type));
            std::unique_ptr<arrow::ArrayBuilder> builder;
            const arrow::Status made = arrow::MakeBuilder(arrow::default_memory_pool(), arrow_type, &builder);
            if (!made.ok()) {
                spdlog::error("Arrow builder for '{}': {}", names[i], made.ToString());
                return nullptr;
            }
            builders.push_back(std::move(builder));
        }

        // Iterate through result chunks
        while (true) {
            auto chunk = result->Fetch();
            if (!chunk || chunk->size() == 0) break;

            for (idx_t row = 0; row < chunk->size(); ++row) {
                for (size_t col = 0; col < chunk->ColumnCount(); ++col) {
                    auto value = chunk->GetValue(col, row);
                    auto status = AppendDuckDBValueToArrowBuilder(value, builders[col].get());
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
        spdlog::error("Exception converting result to Arrow: {}", ErrorText(e));
        return nullptr;
    }
}

} // namespace cyxwiz
