#include "sql_transform.h"

#include "duckdb_connector.h"

#include <spdlog/spdlog.h>

#include <cctype>
#include <set>

namespace cyxwiz {
namespace {

bool IsExactRoundTripType(const arrow::DataType& type) {
    switch (type.id()) {
    case arrow::Type::BOOL:
    case arrow::Type::INT8:
    case arrow::Type::INT16:
    case arrow::Type::INT32:
    case arrow::Type::INT64:
    case arrow::Type::UINT8:
    case arrow::Type::UINT16:
    case arrow::Type::UINT32:
    case arrow::Type::FLOAT:
    case arrow::Type::DOUBLE:
    case arrow::Type::STRING:
    case arrow::Type::LARGE_STRING:
        return true;
    // Nested input (e.g. JSON Lines records, TOFIX143): Arrow lists and structs
    // map one-to-one onto DuckDB LIST and STRUCT, so they are exact when every
    // element/field type is. Results stay scalar (UnsupportedResultColumns).
    case arrow::Type::LIST:
        return IsExactRoundTripType(*static_cast<const arrow::ListType&>(type).value_type());
    case arrow::Type::STRUCT:
        for (const auto& field : type.fields()) {
            if (!IsExactRoundTripType(*field->type())) return false;
        }
        return true;
    default:
        return false;
    }
}

}  // namespace

std::string SqlInputAliasRejection(const std::string& alias) {
    if (alias.empty()) return "input alias is empty";
    if (!(std::isalpha(static_cast<unsigned char>(alias[0])) || alias[0] == '_')) {
        return "input alias '" + alias + "' must start with a letter or underscore";
    }
    for (unsigned char c : alias) {
        if (!(std::isalnum(c) || c == '_')) {
            return "input alias '" + alias + "' may only contain letters, digits and underscores";
        }
    }
    return {};
}

SqlTransform::SqlTransform(std::vector<std::string> readable_folders) {
    connector_ = std::make_unique<DuckDBConnector>(std::move(readable_folders));
}

SqlTransform::~SqlTransform() = default;

void SqlTransform::Interrupt() {
    std::lock_guard<std::mutex> lock(interrupt_mutex_);
    interrupted_ = true;
    if (connector_) connector_->Interrupt();
}

SqlTransformResult SqlTransform::Run(const std::string& query,
                                     const std::vector<SqlTransformInput>& inputs,
                                     SqlRunMode mode) {
    SqlTransformResult result;
    const auto fail = [&result](std::string message) {
        result.error = "SQL: " + std::move(message);
        return result;
    };
    if (!connector_ || !connector_->IsReady()) {
        return fail("the restricted DuckDB connection could not be opened: " +
                    (connector_ ? connector_->GetLastError() : std::string()));
    }
    if (query.find_first_not_of(" \t\r\n") == std::string::npos) {
        return fail("the query is empty");
    }

    std::set<std::string> aliases;
    for (const auto& input : inputs) {
        if (const auto why = SqlInputAliasRejection(input.alias); !why.empty()) return fail(why);
        if (!aliases.insert(input.alias).second) return fail("input alias '" + input.alias + "' is used twice");
        const auto schema = input.table ? input.table->schema() : input.schema;
        if (!schema || (!input.table && input.parquet_path.empty()))
            return fail("input '" + input.alias + "' has no table");
        for (const auto& field : schema->fields()) {
            if (exact_types_ && !IsExactRoundTripType(*field->type())) {
                return fail("input '" + input.alias + "' column '" + field->name() + "' has type " +
                            field->type()->ToString() +
                            ", which contract 1 cannot pass through DuckDB exactly; convert it "
                            "before the SQL step");
            }
        }
    }

    if (const auto why = connector_->ReadOnlySelectRejection(query); !why.empty()) {
        return fail(why);
    }

    for (const auto& input : inputs) {
        const bool registered = input.parquet_path.empty() ? connector_->RegisterTable(input.alias, input.table)
                                : input.load_file ? connector_->RegisterParquetTable(input.alias, input.parquet_path)
                                                  : connector_->RegisterParquetView(input.alias, input.parquet_path);
        if (!registered) {
            return fail("could not register input '" + input.alias + "': " + connector_->GetLastError());
        }
    }

    // Bind before running: unknown tables/columns and unsupported result
    // types are reported without executing the query.
    if (const auto why = exact_types_ ? connector_->UnsupportedResultColumns(query)
                                      : std::string();
        !why.empty()) {
        return fail(why);
    }
    if (mode == SqlRunMode::Check) {
        result.ok = true;
        return result;
    }
    if (mode == SqlRunMode::Explain || mode == SqlRunMode::Profile) {
        // The query passed the read-only single-SELECT check above. EXPLAIN
        // plans it; EXPLAIN ANALYZE also runs it once and times it.
        const auto plan_value = [&](const std::string& prefix, std::string& out) {
            auto plan = connector_->Query(prefix + query);
            if (!plan) return false;
            for (int64_t row = 0; row < plan->num_rows(); ++row) {
                for (int column = 0; column < plan->num_columns(); ++column) {
                    if (plan->schema()->field(column)->name() == "explain_key") continue;
                    auto value = plan->column(column)->GetScalar(row);
                    if (value.ok() && (*value)->is_valid) out += (*value)->ToString();
                }
            }
            return true;
        };
        const char* json_prefix = mode == SqlRunMode::Profile ? "EXPLAIN (ANALYZE, FORMAT JSON) " : "EXPLAIN (FORMAT JSON) ";
        if (!plan_value(json_prefix, result.plan)) {
            std::lock_guard<std::mutex> lock(interrupt_mutex_);
            if (interrupted_) {
                result.cancelled = true;
                return fail("the query was cancelled");
            }
            return fail(connector_->GetLastError());
        }
        if (!plan_value("EXPLAIN ", result.plan_text)) return fail(connector_->GetLastError());
        result.ok = true;
        return result;
    }

    if (!output_path_.empty()) {
        const auto names = connector_->ResultColumnNames(query);
        if (names.empty()) return fail(connector_->GetLastError());
        std::set<std::string> seen;
        bool repeated = false;
        for (const auto& name : names) {
            if (seen.insert(name).second) continue;
            if (exact_types_) {
                return fail("the result has two columns named '" + name +
                            "'; give each output column a unique name (AS ...)");
            }
            repeated = true;
        }
        // A file needs unique column names: the pipeline's own SQL (a join's
        // SELECT *) gets DuckDB's renaming of a subquery's repeated names.
        const std::string written = repeated ? "SELECT * FROM (" + query + ")" : query;
        result.rows_written = connector_->CopyQueryToParquet(written, output_path_);
        {
            std::lock_guard<std::mutex> lock(interrupt_mutex_);
            if (interrupted_) {
                result.cancelled = true;
                return fail("the query was cancelled");
            }
        }
        if (result.rows_written < 0) return fail(connector_->GetLastError());
        result.ok = true;
        return result;
    }

    auto table = connector_->Query(query);
    {
        std::lock_guard<std::mutex> lock(interrupt_mutex_);
        if (interrupted_) {
            result.cancelled = true;
            return fail("the query was cancelled");
        }
    }
    if (!table) {
        return fail(connector_->GetLastError());
    }
    std::set<std::string> names;
    for (const auto& field : table->schema()->fields()) {
        if (exact_types_ && !names.insert(field->name()).second) {
            return fail("the result has two columns named '" + field->name() +
                        "'; give each output column a unique name (AS ...)");
        }
    }
    result.ok = true;
    result.table = std::move(table);
    return result;
}

}  // namespace cyxwiz
