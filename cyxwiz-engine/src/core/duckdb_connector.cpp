#include "duckdb_connector.h"
#include "data_budget.h"
#include <arrow/c/bridge.h>
#include <duckdb.h>
#include <duckdb/common/arrow/arrow_converter.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <unordered_map>
#include <chrono>
#include <filesystem>
#include <thread>
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
        // Integers keep their width (a uint8 pixel stays uint8, TOFIX178 0.5).
        case duckdb::LogicalTypeId::TINYINT: return {arrow::int8(), true, true};
        case duckdb::LogicalTypeId::SMALLINT: return {arrow::int16(), true, true};
        case duckdb::LogicalTypeId::INTEGER: return {arrow::int32(), true, true};
        case duckdb::LogicalTypeId::BIGINT: return {arrow::int64(), true, true};
        case duckdb::LogicalTypeId::UTINYINT: return {arrow::uint8(), true, true};
        case duckdb::LogicalTypeId::USMALLINT: return {arrow::uint16(), true, true};
        case duckdb::LogicalTypeId::UINTEGER: return {arrow::uint32(), true, true};
        case duckdb::LogicalTypeId::UBIGINT: return {arrow::uint64(), true, true};
        case duckdb::LogicalTypeId::FLOAT: return {arrow::float32(), true, true};
        case duckdb::LogicalTypeId::DOUBLE: return {arrow::float64(), true, true};
        case duckdb::LogicalTypeId::VARCHAR: return {arrow::utf8(), true, true};
        // Dates and timestamps stay dates (owner 2026-10-14).
        case duckdb::LogicalTypeId::DATE: return {arrow::date32(), true, true};
        case duckdb::LogicalTypeId::TIMESTAMP: return {arrow::timestamp(arrow::TimeUnit::MICRO), true, true};
        // 0.1, 1234567.5, SUM of decimals: numbers, as double; exact up to 15
        // digits, so the SQL step accepts those (owner 2026-10-14).
        case duckdb::LogicalTypeId::DECIMAL:
            return {arrow::float64(), false, duckdb::DecimalType::GetWidth(type) <= 15};
        // Timestamps with a time zone or another unit, HUGEINT, BLOB, nested
        // types: their text.
        default: return {arrow::utf8(), false, false};
    }
}

bool ConvertsNatively(const duckdb::LogicalType& type) { return ResultTypeFor(type).native; }

// One value into the builder ResultTypeFor chose for its column.
arrow::Status AppendDuckDBValueToArrowBuilder(const duckdb::Value& value, arrow::ArrayBuilder* builder) {
    if (value.IsNull()) return builder->AppendNull();
    switch (builder->type()->id()) {
        case arrow::Type::BOOL: return static_cast<arrow::BooleanBuilder*>(builder)->Append(value.GetValue<bool>());
        case arrow::Type::INT8: return static_cast<arrow::Int8Builder*>(builder)->Append(value.GetValue<int8_t>());
        case arrow::Type::INT16: return static_cast<arrow::Int16Builder*>(builder)->Append(value.GetValue<int16_t>());
        case arrow::Type::INT32: return static_cast<arrow::Int32Builder*>(builder)->Append(value.GetValue<int32_t>());
        case arrow::Type::INT64: return static_cast<arrow::Int64Builder*>(builder)->Append(value.GetValue<int64_t>());
        case arrow::Type::UINT8: return static_cast<arrow::UInt8Builder*>(builder)->Append(value.GetValue<uint8_t>());
        case arrow::Type::UINT16: return static_cast<arrow::UInt16Builder*>(builder)->Append(value.GetValue<uint16_t>());
        case arrow::Type::UINT32: return static_cast<arrow::UInt32Builder*>(builder)->Append(value.GetValue<uint32_t>());
        case arrow::Type::UINT64: return static_cast<arrow::UInt64Builder*>(builder)->Append(value.GetValue<uint64_t>());
        case arrow::Type::FLOAT: return static_cast<arrow::FloatBuilder*>(builder)->Append(value.GetValue<float>());
        case arrow::Type::DOUBLE: return static_cast<arrow::DoubleBuilder*>(builder)->Append(value.GetValue<double>());
        case arrow::Type::DATE32:
            return static_cast<arrow::Date32Builder*>(builder)->Append(value.GetValue<duckdb::date_t>().days);
        case arrow::Type::TIMESTAMP:
            return static_cast<arrow::TimestampBuilder*>(builder)->Append(value.GetValue<duckdb::timestamp_t>().value);
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
        // The data budget bounds DuckDB too; over it, sorts, joins and
        // group-bys spill to the spill directory instead of failing. Set
        // before external access is switched off, which locks temp_directory.
        opened_budget_ = DataBudgetShareBytes();
        opened_spill_ = DataSpillDirectory();
        config.SetOptionByName("memory_limit",
                               duckdb::Value(std::to_string(std::max<uint64_t>(64, opened_budget_ >> 20)) + "MB"));
        // Each DuckDB thread needs working memory of its own: one per 256 MB
        // of budget (a 5 GB sort under 244 MB failed with two threads, ran
        // with one); more threads than the budget feeds fail instead of
        // spilling.
        const uint64_t threads = std::clamp<uint64_t>(opened_budget_ / (256ULL << 20), 1,
                                                      std::max(1u, std::thread::hardware_concurrency()));
        config.SetOptionByName("threads", duckdb::Value::BIGINT(static_cast<int64_t>(threads)));
        // A folder of its own under the spill directory: DuckDBs running at
        // once (sibling branches) would otherwise share spill file names.
        static std::atomic<uint64_t> opened{0};
        const std::string spill = opened_spill_.empty() ? std::string() :
            (std::filesystem::path(opened_spill_) /
             (std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "_" + std::to_string(++opened))).string();
        std::error_code spill_error;
        if (!spill.empty() && (std::filesystem::create_directories(spill, spill_error), !spill_error)) {
            config.SetOptionByName("temp_directory", duckdb::Value(spill));
            // With external access off DuckDB touches only allowed folders,
            // its own spill folder included.
            std::string folder = std::filesystem::path(spill).generic_string();
            if (folder.back() != '/') folder += '/';
            config.options.allowed_directories.insert(folder);
        }
        config.SetOptionByName("enable_external_access", duckdb::Value::BOOLEAN(false));
        config.SetOptionByName("autoinstall_known_extensions", duckdb::Value::BOOLEAN(false));
        config.SetOptionByName("autoload_known_extensions", duckdb::Value::BOOLEAN(false));
        for (auto& dir : allowed_directories) config.options.allowed_directories.insert(std::move(dir));
        config.SetOptionByName("lock_configuration", duckdb::Value::BOOLEAN(true));
        // The spill folder goes with the database (connections made by
        // NewConnection share it).
        db_ = std::shared_ptr<duckdb::DuckDB>(new duckdb::DuckDB(nullptr, &config), [spill](duckdb::DuckDB* db) {
            delete db;
            std::error_code ignored;
            if (!spill.empty()) std::filesystem::remove_all(spill, ignored);
        });
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

std::vector<std::string> DuckDBConnector::ResultColumnNames(const std::string& sql) {
    if (!conn_) return {};
    try {
        auto prepared = conn_->Prepare(sql);
        if (prepared->HasError()) {
            last_error_ = prepared->GetError();
            return {};
        }
        return prepared->GetNames();
    } catch (const std::exception& e) {
        last_error_ = ErrorText(e);
        return {};
    }
}

namespace {
// 'a''b' quoting for a file path in SQL (forward slashes).
std::string SqlPathLiteral(const std::string& path) {
    std::string text = "'";
    for (char c : path) {
        if (c == '\\') text += '/';
        else if (c == '\'') text += "''";
        else text += c;
    }
    return text + "'";
}
}  // namespace

bool DuckDBConnector::RegisterParquetView(const std::string& name, const std::string& path) {
    if (!conn_) {
        last_error_ = "DuckDB connection not initialized";
        return false;
    }
    return Execute("CREATE OR REPLACE VIEW " + QuoteIdentifier(name) +
                   " AS SELECT * FROM read_parquet(" + SqlPathLiteral(path) + ")");
}

bool DuckDBConnector::RegisterParquetTable(const std::string& name, const std::string& path) {
    if (!conn_) {
        last_error_ = "DuckDB connection not initialized";
        return false;
    }
    return Execute("CREATE OR REPLACE TABLE " + QuoteIdentifier(name) +
                   " AS SELECT * FROM read_parquet(" + SqlPathLiteral(path) + ")");
}

int64_t DuckDBConnector::CopyQueryToParquet(const std::string& sql, const std::string& path) {
    if (!conn_) {
        last_error_ = "DuckDB connection not initialized";
        return -1;
    }
    try {
        auto result = conn_->Query("COPY (" + sql + ") TO " + SqlPathLiteral(path) +
                                   " (FORMAT PARQUET, ROW_GROUP_SIZE 131072, COMPRESSION ZSTD)");
        if (result->HasError()) {
            last_error_ = result->GetError();
            return -1;
        }
        auto chunk = result->Fetch();
        return chunk && chunk->size() > 0 ? chunk->GetValue(0, 0).GetValue<int64_t>() : 0;
    } catch (const std::exception& e) {
        last_error_ = ErrorText(e);
        return -1;
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

bool DuckDBConnector::MatchesDataBudget() const {
    return opened_budget_ == DataBudgetShareBytes() && opened_spill_ == DataSpillDirectory();
}

std::unique_ptr<DuckDBConnector> DuckDBConnector::NewConnection() {
    if (!db_ || !conn_) return nullptr;
    std::unique_ptr<DuckDBConnector> c(new DuckDBConnector(SharedTag{}, db_));
    c->opened_budget_ = opened_budget_;
    c->opened_spill_ = opened_spill_;
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

    if (!conn_) {
        last_error_ = "DuckDB connection not initialized";
        spdlog::error(last_error_);
        return false;
    }
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

        // One pass over the table's batches through DuckDB's Arrow scan (the
        // C API's: the C++ one needs ArrowArrayStreamWrapper, which duckdb.dll
        // does not export). The scan view reads its stream once, so the rows
        // are kept in a DuckDB table any query can read again (and it has a
        // rowid). Over the memory limit DuckDB spills it to its spill folder.
        ArrowArrayStream stream;
        const auto exported = arrow::ExportRecordBatchReader(
            std::make_shared<arrow::TableBatchReader>(*arrow_table), &stream);
        if (!exported.ok()) {
            last_error_ = exported.ToString();
            spdlog::error("Failed to export Arrow table '{}': {}", table_name, last_error_);
            return false;
        }
        const std::string scan = "__cyxwiz_scan_" + table_name;
        // In DuckDB 1.4 a duckdb_connection is a duckdb::Connection*
        // (duckdb_connect, duckdb-c.cpp); test_arrow_integration pins it.
        const bool scanned =
            duckdb_arrow_scan(reinterpret_cast<duckdb_connection>(conn_.get()), scan.c_str(),
                              reinterpret_cast<duckdb_arrow_stream>(&stream)) == DuckDBSuccess;
        auto created = scanned ? conn_->Query("CREATE TABLE " + quoted_table_name + " AS SELECT * FROM " +
                                              QuoteIdentifier(scan))
                               : nullptr;
        conn_->Query("DROP VIEW IF EXISTS " + QuoteIdentifier(scan));
        stream.release(&stream);
        if (!created || created->HasError()) {
            last_error_ = created ? created->GetError() : "DuckDB could not scan the Arrow table";
            spdlog::error("Failed to create table '{}': {}", table_name, last_error_);
            return false;
        }

        spdlog::debug("Registered Arrow table '{}' as DuckDB table: {} rows, {} columns",
                     table_name, arrow_table->num_rows(), arrow_table->num_columns());

        return true;

    } catch (const duckdb::Exception& e) {
        last_error_ = ErrorText(e);
        spdlog::error("Failed to register table '{}': {}", table_name, last_error_);
        return false;
    }
}

bool DuckDBConnector::UnregisterTable(const std::string& table_name) {
    if (!conn_) return false;
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
