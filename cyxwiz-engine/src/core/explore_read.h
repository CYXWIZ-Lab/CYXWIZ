#pragma once

// Reading chosen columns of a table for Data Studio's Explore models (Tests,
// Regression, Groups, Projection; TOFIX129 D3 / D5). The columns are read with
// one SELECT through a QueryRunner, so a table kept on disk works the same way.
// A value is missing when it is NULL, is not a number (number columns), or is
// one of the texts the project marks as missing for its column.

#include "data_explore.h"

#include <arrow/api.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz::explore_read {

// SELECT expressions: a number column as DOUBLE (NULL when missing or not a
// number), a label column as VARCHAR (NULL when missing), named `as`.
std::string NumberExpr(const std::string& column, const explore::MissingTexts& texts, std::vector<QueryParam>& params,
                       const std::string& as);
std::string LabelExpr(const std::string& column, const explore::MissingTexts& texts, std::vector<QueryParam>& params,
                      const std::string& as);

struct Read {
    std::shared_ptr<arrow::Table> table;
    std::string error;  // "Cancelled." when the query was cancelled
};
// SELECT exprs FROM table.
Read Select(const std::string& table, const std::vector<std::string>& exprs, std::vector<QueryParam> params,
            const QueryRunner& run, const char* label);

// A read column back: numbers (nullopt when missing or not finite), labels.
std::vector<std::optional<double>> Numbers(const arrow::Table& t, const std::string& name);
std::vector<std::optional<std::string>> Labels(const arrow::Table& t, const std::string& name);

}  // namespace cyxwiz::explore_read
