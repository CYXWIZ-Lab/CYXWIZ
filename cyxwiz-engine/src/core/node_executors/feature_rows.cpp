#include "feature_rows.h"

#include <arrow/compute/api.h>

#include <cmath>

namespace cyxwiz {

arrow::Status ReadFeatureRows(const std::shared_ptr<arrow::Table>& table, const std::vector<std::string>& names,
                              const std::string& op_name, bool standardise, FeatureRows& out,
                              const std::function<bool()>& cancelled) {
    out = FeatureRows{};
    if (!table) return arrow::Status::Invalid(op_name + ": input table is null");
    out.total = table->num_rows();
    const size_t d = names.size();
    // Each feature as one float64 array (a cast keeps nulls; NaN counts as missing below).
    std::vector<std::shared_ptr<arrow::DoubleArray>> columns;
    for (const auto& name : names) {
        auto column = table->GetColumnByName(name);
        if (!column) return arrow::Status::KeyError(op_name + ": feature column '" + name + "' not found");
        ARROW_ASSIGN_OR_RAISE(auto cast, arrow::compute::Cast(arrow::Datum(column), arrow::float64()));
        const auto& chunks = cast.chunked_array()->chunks();
        std::shared_ptr<arrow::Array> joined;
        if (chunks.empty()) {
            ARROW_ASSIGN_OR_RAISE(joined, arrow::MakeArrayOfNull(arrow::float64(), 0));
        } else {
            ARROW_ASSIGN_OR_RAISE(joined, arrow::Concatenate(chunks));
        }
        if (joined->length() != out.total) return arrow::Status::Invalid(op_name + ": feature '" + name + "' has a different row count");
        columns.push_back(std::static_pointer_cast<arrow::DoubleArray>(joined));
    }
    for (int64_t r = 0; r < out.total; ++r) {
        if ((r & 4095) == 0 && cancelled && cancelled()) return arrow::Status::Cancelled(op_name + ": cancelled while reading features");
        std::vector<double> row(d);
        bool complete = true;
        for (size_t j = 0; j < d && complete; ++j) {
            complete = columns[j]->IsValid(r) && std::isfinite(columns[j]->Value(r));
            if (complete) row[j] = columns[j]->Value(r);
        }
        if (!complete) continue;
        out.data.push_back(std::move(row));
        out.rows.push_back(r);
    }
    if (!standardise || out.data.empty()) return arrow::Status::OK();
    const double n = static_cast<double>(out.data.size());
    for (size_t j = 0; j < d; ++j) {
        double mean = 0.0, ss = 0.0;
        for (const auto& row : out.data) mean += row[j];
        mean /= n;
        for (const auto& row : out.data) ss += (row[j] - mean) * (row[j] - mean);
        const double sd = std::sqrt(ss / n);  // population SD, as scikit-learn's StandardScaler
        if (!(sd > 0.0))
            return arrow::Status::Invalid(op_name + ": feature '" + names[j] + "' has the same value in every row, so it cannot be standardised");
        for (auto& row : out.data) row[j] = (row[j] - mean) / sd;
    }
    return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::Table>> AppendFittedInt32(const std::shared_ptr<arrow::Table>& table, const std::string& name,
                                                               const std::vector<int64_t>& rows, const std::vector<int>& values) {
    if (rows.size() != values.size()) return arrow::Status::Invalid("AppendFittedInt32: " + name + " has a value per fitted row");
    arrow::Int32Builder builder;
    ARROW_RETURN_NOT_OK(builder.Reserve(table->num_rows()));
    size_t next = 0;
    for (int64_t r = 0; r < table->num_rows(); ++r) {
        if (next < rows.size() && rows[next] == r) ARROW_RETURN_NOT_OK(builder.Append(values[next++]));
        else ARROW_RETURN_NOT_OK(builder.AppendNull());
    }
    ARROW_ASSIGN_OR_RAISE(auto array, builder.Finish());
    return table->AddColumn(table->num_columns(), arrow::field(name, arrow::int32()), std::make_shared<arrow::ChunkedArray>(array));
}

arrow::Result<std::shared_ptr<arrow::Table>> AppendFittedColumns(const std::shared_ptr<arrow::Table>& table,
                                                                 const std::vector<std::string>& names,
                                                                 const std::vector<int64_t>& rows,
                                                                 const std::vector<std::vector<double>>& values) {
    if (rows.size() != values.size()) return arrow::Status::Invalid("AppendFittedColumns: a value row per fitted row");
    std::shared_ptr<arrow::Table> result = table;
    for (size_t k = 0; k < names.size(); ++k) {
        arrow::DoubleBuilder builder;
        ARROW_RETURN_NOT_OK(builder.Reserve(table->num_rows()));
        size_t next = 0;
        for (int64_t r = 0; r < table->num_rows(); ++r) {
            if (next < rows.size() && rows[next] == r) {
                const auto& v = values[next++];
                ARROW_RETURN_NOT_OK(k < v.size() ? builder.Append(v[k]) : builder.AppendNull());
            } else {
                ARROW_RETURN_NOT_OK(builder.AppendNull());
            }
        }
        ARROW_ASSIGN_OR_RAISE(auto array, builder.Finish());
        ARROW_ASSIGN_OR_RAISE(result, result->AddColumn(result->num_columns(), arrow::field(names[k], arrow::float64()),
                                                         std::make_shared<arrow::ChunkedArray>(array)));
    }
    return result;
}

}  // namespace cyxwiz
