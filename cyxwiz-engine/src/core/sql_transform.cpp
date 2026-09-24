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

SqlTransform::SqlTransform() {
    DuckDBConnectorPolicy policy;
    policy.allow_external_access = false;
    connector_ = std::make_unique<DuckDBConnector>(policy);
}

SqlTransform::~SqlTransform() = default;

void SqlTransform::Interrupt() {
    std::lock_guard<std::mutex> lock(interrupt_mutex_);
    interrupted_ = true;
    if (connector_) connector_->Interrupt();
}

SqlTransformResult SqlTransform::Run(const std::string& query,
                                     const std::vector<SqlTransformInput>& inputs) {
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
        if (!input.table) return fail("input '" + input.alias + "' has no table");
        for (const auto& field : input.table->schema()->fields()) {
            if (!IsExactRoundTripType(*field->type())) {
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
        if (!connector_->RegisterTable(input.alias, input.table)) {
            return fail("could not register input '" + input.alias + "': " + connector_->GetLastError());
        }
    }

    // Bind before running: unknown tables/columns and unsupported result
    // types are reported without executing the query.
    if (const auto why = connector_->UnsupportedResultColumns(query); !why.empty()) {
        return fail(why);
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
        if (!names.insert(field->name()).second) {
            return fail("the result has two columns named '" + field->name() +
                        "'; give each output column a unique name (AS ...)");
        }
    }
    result.ok = true;
    result.table = std::move(table);
    return result;
}

}  // namespace cyxwiz
