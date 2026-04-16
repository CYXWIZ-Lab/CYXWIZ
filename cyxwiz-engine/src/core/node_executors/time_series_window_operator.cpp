#include "time_series_window_operator.h"

#include <arrow/api.h>
#include <arrow/builder.h>
#include <spdlog/spdlog.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

namespace {

// Read an entire ChunkedArray column into a std::vector<float>, casting
// supported numeric types to float32. Returns false on unsupported types
// (caller surfaces as an Arrow error). Null values become NaN so they
// propagate through windowing and surface as loss NaN rather than
// silently defaulting to zero.
bool ReadColumnAsFloat(
    const std::shared_ptr<arrow::ChunkedArray>& column,
    std::vector<float>& out,
    std::string& error_type_name) {

    out.clear();
    out.reserve(static_cast<size_t>(column->length()));

    for (int c = 0; c < column->num_chunks(); ++c) {
        auto chunk = column->chunk(c);
        const int64_t chunk_len = chunk->length();

        auto read_null = [&](int64_t i) -> bool { return chunk->IsNull(i); };

        switch (chunk->type_id()) {
            case arrow::Type::DOUBLE: {
                auto arr = std::static_pointer_cast<arrow::DoubleArray>(chunk);
                const double* data = arr->raw_values();
                for (int64_t i = 0; i < chunk_len; ++i) {
                    out.push_back(read_null(i) ? std::numeric_limits<float>::quiet_NaN()
                                               : static_cast<float>(data[i]));
                }
                break;
            }
            case arrow::Type::FLOAT: {
                auto arr = std::static_pointer_cast<arrow::FloatArray>(chunk);
                const float* data = arr->raw_values();
                for (int64_t i = 0; i < chunk_len; ++i) {
                    out.push_back(read_null(i) ? std::numeric_limits<float>::quiet_NaN()
                                               : data[i]);
                }
                break;
            }
            case arrow::Type::INT64: {
                auto arr = std::static_pointer_cast<arrow::Int64Array>(chunk);
                for (int64_t i = 0; i < chunk_len; ++i) {
                    out.push_back(read_null(i) ? std::numeric_limits<float>::quiet_NaN()
                                               : static_cast<float>(arr->Value(i)));
                }
                break;
            }
            case arrow::Type::INT32: {
                auto arr = std::static_pointer_cast<arrow::Int32Array>(chunk);
                for (int64_t i = 0; i < chunk_len; ++i) {
                    out.push_back(read_null(i) ? std::numeric_limits<float>::quiet_NaN()
                                               : static_cast<float>(arr->Value(i)));
                }
                break;
            }
            case arrow::Type::INT16: {
                auto arr = std::static_pointer_cast<arrow::Int16Array>(chunk);
                for (int64_t i = 0; i < chunk_len; ++i) {
                    out.push_back(read_null(i) ? std::numeric_limits<float>::quiet_NaN()
                                               : static_cast<float>(arr->Value(i)));
                }
                break;
            }
            case arrow::Type::INT8: {
                auto arr = std::static_pointer_cast<arrow::Int8Array>(chunk);
                for (int64_t i = 0; i < chunk_len; ++i) {
                    out.push_back(read_null(i) ? std::numeric_limits<float>::quiet_NaN()
                                               : static_cast<float>(arr->Value(i)));
                }
                break;
            }
            case arrow::Type::UINT64: {
                auto arr = std::static_pointer_cast<arrow::UInt64Array>(chunk);
                for (int64_t i = 0; i < chunk_len; ++i) {
                    out.push_back(read_null(i) ? std::numeric_limits<float>::quiet_NaN()
                                               : static_cast<float>(arr->Value(i)));
                }
                break;
            }
            case arrow::Type::UINT32: {
                auto arr = std::static_pointer_cast<arrow::UInt32Array>(chunk);
                for (int64_t i = 0; i < chunk_len; ++i) {
                    out.push_back(read_null(i) ? std::numeric_limits<float>::quiet_NaN()
                                               : static_cast<float>(arr->Value(i)));
                }
                break;
            }
            case arrow::Type::UINT16: {
                auto arr = std::static_pointer_cast<arrow::UInt16Array>(chunk);
                for (int64_t i = 0; i < chunk_len; ++i) {
                    out.push_back(read_null(i) ? std::numeric_limits<float>::quiet_NaN()
                                               : static_cast<float>(arr->Value(i)));
                }
                break;
            }
            case arrow::Type::UINT8: {
                auto arr = std::static_pointer_cast<arrow::UInt8Array>(chunk);
                const uint8_t* data = arr->raw_values();
                for (int64_t i = 0; i < chunk_len; ++i) {
                    out.push_back(read_null(i) ? std::numeric_limits<float>::quiet_NaN()
                                               : static_cast<float>(data[i]));
                }
                break;
            }
            default:
                error_type_name = chunk->type()->ToString();
                return false;
        }
    }

    return true;
}

} // namespace

bool TimeSeriesWindowOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {

    auto it = params.find("value_col");
    if (it == params.end() || it->second.empty()) {
        error = "TimeSeriesWindow: 'value_col' parameter is required";
        return false;
    }
    value_col_ = it->second;

    auto read_int = [&](const char* key, int default_value, int& out) -> bool {
        auto p = params.find(key);
        if (p == params.end() || p->second.empty()) {
            out = default_value;
            return true;
        }
        try {
            out = std::stoi(p->second);
        } catch (...) {
            error = std::string("TimeSeriesWindow: '") + key +
                    "' is not a valid integer: " + p->second;
            return false;
        }
        return true;
    };

    if (!read_int("input_width", 12, input_width_)) return false;
    if (!read_int("label_width", 1,  label_width_)) return false;
    if (!read_int("shift",       1,  shift_))       return false;

    if (input_width_ < 1) {
        error = "TimeSeriesWindow: input_width must be >= 1 (got " +
                std::to_string(input_width_) + ")";
        return false;
    }
    if (label_width_ != 1) {
        error = "TimeSeriesWindow v1 supports label_width=1 only (got " +
                std::to_string(label_width_) +
                "). Multi-step forecasting is deferred to Phase 4.x.";
        return false;
    }
    if (shift_ < 1) {
        error = "TimeSeriesWindow: shift must be >= 1 (got " +
                std::to_string(shift_) + ")";
        return false;
    }

    return true;
}

arrow::Result<std::shared_ptr<arrow::Schema>>
TimeSeriesWindowOperator::InferOutputSchema(
    const std::shared_ptr<arrow::Schema>& /*input_schema*/) {

    std::vector<std::shared_ptr<arrow::Field>> fields;
    fields.reserve(input_width_ + 1);
    for (int i = 0; i < input_width_; ++i) {
        fields.push_back(arrow::field("x_" + std::to_string(i), arrow::float32()));
    }
    fields.push_back(arrow::field("y", arrow::float32()));
    return arrow::schema(fields);
}

arrow::Result<std::shared_ptr<arrow::Table>>
TimeSeriesWindowOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    if (!input) {
        return arrow::Status::Invalid("TimeSeriesWindow: input table is null");
    }
    if (value_col_.empty()) {
        return arrow::Status::Invalid(
            "TimeSeriesWindow: Apply called before Configure succeeded");
    }

    auto column = input->GetColumnByName(value_col_);
    if (!column) {
        return arrow::Status::KeyError(
            "TimeSeriesWindow: column '" + value_col_ + "' not found in input table");
    }

    std::vector<float> values;
    std::string bad_type;
    if (!ReadColumnAsFloat(column, values, bad_type)) {
        return arrow::Status::TypeError(
            "TimeSeriesWindow: column '" + value_col_ +
            "' has unsupported Arrow type '" + bad_type +
            "' (need a numeric type)");
    }

    const int64_t n = static_cast<int64_t>(values.size());
    // Each window consumes input_width + shift consecutive values (inputs
    // 0..input_width-1 followed by target at input_width+shift-1). Number
    // of windows = floor(n - input_width - shift + 1), iterated with
    // stride 1. stride > 1 (decimation) is deferred.
    const int64_t span = static_cast<int64_t>(input_width_) + shift_;
    const int64_t num_windows = (n >= span) ? (n - span + 1) : 0;

    if (num_windows <= 0) {
        return arrow::Status::Invalid(
            "TimeSeriesWindow: not enough rows to build any window. "
            "Have " + std::to_string(n) + " values, need at least " +
            std::to_string(span) +
            " (input_width=" + std::to_string(input_width_) +
            " + shift=" + std::to_string(shift_) + ")");
    }

    // Build output columns as float32. Memory pool is the default global
    // pool — matches how CompactIntegerColumns allocates. One FloatBuilder
    // per output column; labels go in a separate column named `y`.
    arrow::MemoryPool* pool = arrow::default_memory_pool();
    std::vector<std::unique_ptr<arrow::FloatBuilder>> feature_builders;
    feature_builders.reserve(input_width_);
    for (int i = 0; i < input_width_; ++i) {
        feature_builders.push_back(std::make_unique<arrow::FloatBuilder>(pool));
        ARROW_RETURN_NOT_OK(feature_builders.back()->Reserve(num_windows));
    }
    arrow::FloatBuilder label_builder(pool);
    ARROW_RETURN_NOT_OK(label_builder.Reserve(num_windows));

    for (int64_t w = 0; w < num_windows; ++w) {
        for (int i = 0; i < input_width_; ++i) {
            ARROW_RETURN_NOT_OK(
                feature_builders[i]->Append(values[w + i]));
        }
        // label_width_ is enforced to 1 in Configure. Target is the first
        // value AFTER the window, offset by (shift-1) additional steps.
        const int64_t target_idx = w + input_width_ + (shift_ - 1);
        ARROW_RETURN_NOT_OK(label_builder.Append(values[target_idx]));
    }

    std::vector<std::shared_ptr<arrow::Array>> arrays;
    arrays.reserve(input_width_ + 1);
    for (int i = 0; i < input_width_; ++i) {
        std::shared_ptr<arrow::Array> arr;
        ARROW_RETURN_NOT_OK(feature_builders[i]->Finish(&arr));
        arrays.push_back(std::move(arr));
    }
    {
        std::shared_ptr<arrow::Array> arr;
        ARROW_RETURN_NOT_OK(label_builder.Finish(&arr));
        arrays.push_back(std::move(arr));
    }

    ARROW_ASSIGN_OR_RAISE(auto out_schema, InferOutputSchema(input->schema()));
    auto out_table = arrow::Table::Make(out_schema, arrays, num_windows);

    spdlog::info("TimeSeriesWindow: {} input values -> {} windows "
                 "(input_width={}, label_width=1, shift={})",
                 n, num_windows, input_width_, shift_);
    return out_table;
}

} // namespace cyxwiz
