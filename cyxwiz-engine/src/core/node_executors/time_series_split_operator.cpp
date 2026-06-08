#include "time_series_split_operator.h"

#include <arrow/api.h>
#include <arrow/builder.h>
#include <spdlog/spdlog.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

namespace {

bool ReadFloatParam(
    const std::map<std::string, std::string>& params,
    const char* key,
    float default_value,
    float& out,
    std::string& error) {

    auto it = params.find(key);
    if (it == params.end() || it->second.empty()) {
        out = default_value;
        return true;
    }
    try {
        out = std::stof(it->second);
    } catch (...) {
        error = std::string("TimeSeriesSplit: '") + key +
                "' is not a valid float: " + it->second;
        return false;
    }
    return true;
}

} // namespace

bool TimeSeriesSplitOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {

    train_ratio_ = 0.8f;
    val_ratio_ = 0.1f;
    test_ratio_ = 0.1f;

    if (!ReadFloatParam(params, "train_ratio", 0.8f, train_ratio_, error)) return false;
    if (!ReadFloatParam(params, "val_ratio",   0.1f, val_ratio_,   error)) return false;
    if (!ReadFloatParam(params, "test_ratio",  0.1f, test_ratio_,  error)) return false;

    if (train_ratio_ < 0.0f || val_ratio_ < 0.0f || test_ratio_ < 0.0f) {
        error = "TimeSeriesSplit: all ratios must be >= 0";
        return false;
    }
    if (train_ratio_ <= 0.0f) {
        error = "TimeSeriesSplit: train_ratio must be > 0";
        return false;
    }

    const float sum = train_ratio_ + val_ratio_ + test_ratio_;
    if (std::fabs(sum - 1.0f) > 0.01f) {
        error = "TimeSeriesSplit: ratios must sum to 1.0 (got " +
                std::to_string(sum) + ")";
        return false;
    }

    return true;
}

arrow::Result<std::shared_ptr<arrow::Schema>>
TimeSeriesSplitOperator::InferOutputSchema(
    const std::shared_ptr<arrow::Schema>& input_schema) {

    if (!input_schema) {
        return arrow::Status::Invalid("TimeSeriesSplit: input schema is null");
    }
    auto fields = input_schema->fields();
    fields.push_back(arrow::field(kPartitionColumnName, arrow::int8()));
    return arrow::schema(fields);
}

arrow::Result<std::shared_ptr<arrow::Table>>
TimeSeriesSplitOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    if (!input) {
        return arrow::Status::Invalid("TimeSeriesSplit: input table is null");
    }

    const int64_t n = input->num_rows();
    if (n == 0) {
        return arrow::Status::Invalid(
            "TimeSeriesSplit: input table has zero rows - upstream TimeSeriesWindow "
            "probably produced no windows");
    }

    int64_t train_count = static_cast<int64_t>(std::floor(n * train_ratio_));
    int64_t val_count   = static_cast<int64_t>(std::floor(n * val_ratio_));
    // Give any remainder from floor rounding to the train split so train
    // always gets the largest slice. Test takes whatever is left over so
    // train_count + val_count + test_count == n exactly.
    if (train_count + val_count > n) val_count = n - train_count;
    int64_t test_count = n - train_count - val_count;

    // Guard: if val or test is zero because n is too small for the
    // requested ratio, warn and steal a single row from train. This
    // keeps the smoke test viable on tiny datasets (airline passengers
    // has 144 rows; with input_width=12 + shift=1 that's 132 windows,
    // so 80/10/10 gives train=105 val=13 test=14 — fine. But a user
    // testing with 20 rows + input_width=16 would hit 4 windows and
    // get train=3 val=0 test=1 without this guard).
    if (val_count == 0 && train_count > 1) {
        --train_count;
        ++val_count;
        spdlog::warn("TimeSeriesSplit: val_count rounded to 0 - stealing 1 row from train");
    }

    arrow::MemoryPool* pool = arrow::default_memory_pool();
    arrow::Int8Builder builder(pool);
    ARROW_RETURN_NOT_OK(builder.Reserve(n));

    for (int64_t i = 0; i < train_count; ++i) {
        ARROW_RETURN_NOT_OK(builder.Append(0));
    }
    for (int64_t i = 0; i < val_count; ++i) {
        ARROW_RETURN_NOT_OK(builder.Append(1));
    }
    for (int64_t i = 0; i < test_count; ++i) {
        ARROW_RETURN_NOT_OK(builder.Append(2));
    }

    std::shared_ptr<arrow::Array> partition_array;
    ARROW_RETURN_NOT_OK(builder.Finish(&partition_array));

    // Append the new column to the existing schema. Arrow's AddColumn
    // takes the column index to insert at (we append at the end) and a
    // ChunkedArray wrapping our single-chunk builder result.
    auto partition_chunked = std::make_shared<arrow::ChunkedArray>(partition_array);
    auto partition_field = arrow::field(kPartitionColumnName, arrow::int8());

    ARROW_ASSIGN_OR_RAISE(
        auto out_table,
        input->AddColumn(input->num_columns(), partition_field, partition_chunked));

    spdlog::info("TimeSeriesSplit: {} rows -> train={}, val={}, test={} "
                 "(ratios {:.2f}/{:.2f}/{:.2f})",
                 n, train_count, val_count, test_count,
                 train_ratio_, val_ratio_, test_ratio_);

    return out_table;
}

} // namespace cyxwiz
