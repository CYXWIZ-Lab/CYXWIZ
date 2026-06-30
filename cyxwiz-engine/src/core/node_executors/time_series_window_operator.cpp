#include "time_series_window_operator.h"
#include "../profiler_trace.h"
#include "ts_column_utils.h"

#include <arrow/api.h>
#include <arrow/builder.h>
#include <spdlog/spdlog.h>

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace cyxwiz {

namespace {

// Parse a comma-separated list of column names. Empty string -> empty vector.
// Whitespace around commas is trimmed.
void ParseColList(const std::string& s, std::vector<std::string>& out) {
    out.clear();
    if (s.empty()) return;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        size_t start = token.find_first_not_of(" \t");
        size_t end = token.find_last_not_of(" \t");
        if (start == std::string::npos) continue;
        out.push_back(token.substr(start, end - start + 1));
    }
}

} // namespace

bool TimeSeriesWindowOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {

    value_col_.clear();
    feature_cols_.clear();
    time_col_.clear();
    input_width_ = 12;
    label_width_ = 1;
    shift_ = 1;

    auto it = params.find("value_col");
    if (it == params.end() || it->second.empty()) {
        error = "TimeSeriesWindow: 'value_col' parameter is required";
        return false;
    }
    value_col_ = it->second;

    auto fc_it = params.find("feature_cols");
    const std::string fc_str = (fc_it != params.end()) ? fc_it->second : "";
    ParseColList(fc_str, feature_cols_);

    // Optional time column for forecast-plotting alignment. Empty = off.
    auto tc_it = params.find("time_col");
    time_col_ = (tc_it != params.end()) ? tc_it->second : "";

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
    // value_col's windowed columns: x_0 .. x_{input_width-1}
    for (int i = 0; i < input_width_; ++i) {
        fields.push_back(arrow::field("x_" + std::to_string(i), arrow::float32()));
    }
    // Each extra feature column contributes its own windowed block:
    // <feat>_x_0 .. <feat>_x_{input_width-1}
    for (const std::string& feat : feature_cols_) {
        for (int i = 0; i < input_width_; ++i) {
            fields.push_back(arrow::field(
                feat + "_x_" + std::to_string(i), arrow::float32()));
        }
    }
    fields.push_back(arrow::field("y", arrow::float32()));
    if (!time_col_.empty()) {
        // __-prefix marks this as internal metadata — ArrowDatasetBatcher
        // skips __-prefixed columns in feature auto-detect so training
        // doesn't inhale the timestamp as an extra feature. Plotting /
        // inspection code reads it by exact name.
        fields.push_back(arrow::field("__window_start_time", arrow::float64()));
    }
    return arrow::schema(fields);
}

arrow::Result<std::shared_ptr<arrow::Table>>
TimeSeriesWindowOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz TimeSeriesWindow Materializer");

    if (!input) {
        return arrow::Status::Invalid("TimeSeriesWindow: input table is null");
    }
    if (value_col_.empty()) {
        return arrow::Status::Invalid(
            "TimeSeriesWindow: Apply called before Configure succeeded");
    }

    auto report_progress = [&](std::string stage,
                               std::string message,
                               double progress,
                               uint64_t processed = 0,
                               uint64_t total = 0,
                               uint64_t memory = 0) {
        if (!progress_callback_) {
            return;
        }
        PipelineOperatorProgress event;
        event.stage = std::move(stage);
        event.message = std::move(message);
        event.progress = static_cast<float>(progress);
        event.processed_items = processed;
        event.total_items = total;
        event.estimated_memory_bytes = memory;
        progress_callback_(event);
    };

    // Read value column (target source + first feature block).
    auto column = input->GetColumnByName(value_col_);
    if (!column) {
        return arrow::Status::KeyError(
            "TimeSeriesWindow: column '" + value_col_ + "' not found in input table");
    }

    report_progress("Reading value column",
                    "Reading time-series value column '" + value_col_ + "'",
                    0.05);

    std::vector<float> values;
    std::string bad_type;
    if (!ReadColumnAsFloat(column, values, bad_type)) {
        return arrow::Status::TypeError(
            "TimeSeriesWindow: column '" + value_col_ +
            "' has unsupported Arrow type '" + bad_type +
            "' (need a numeric type)");
    }
    report_progress("Value column ready",
                    "Read " + std::to_string(values.size()) +
                    " time-series values",
                    0.15,
                    static_cast<uint64_t>(values.size()),
                    static_cast<uint64_t>(values.size()));

    // Read optional time column (forecast-plot alignment). Numeric only
    // in v1 — timestamp/string types are deferred. Stored as float64 in
    // the output to preserve unix-epoch precision.
    std::vector<float> time_values;
    if (!time_col_.empty()) {
        report_progress("Reading time column",
                        "Reading time-series metadata column '" + time_col_ + "'",
                        0.20,
                        0,
                        static_cast<uint64_t>(values.size()));
        auto tcol = input->GetColumnByName(time_col_);
        if (!tcol) {
            return arrow::Status::KeyError(
                "TimeSeriesWindow: time_col '" + time_col_ + "' not found");
        }
        std::string tbad;
        if (!ReadColumnAsFloat(tcol, time_values, tbad)) {
            return arrow::Status::TypeError(
                "TimeSeriesWindow: time_col '" + time_col_ +
                "' has unsupported type '" + tbad + "' — numeric only in v1");
        }
        if (static_cast<int64_t>(time_values.size()) != static_cast<int64_t>(values.size())) {
            return arrow::Status::Invalid(
                "TimeSeriesWindow: time_col '" + time_col_ + "' row count (" +
                std::to_string(time_values.size()) + ") differs from value_col");
        }
    }

    // Read each extra feature column into a parallel vector. They must
    // all have the same length as the value column (they came from the
    // same Arrow table).
    std::vector<std::vector<float>> feature_values;
    feature_values.reserve(feature_cols_.size());
    for (size_t feature_index = 0; feature_index < feature_cols_.size(); ++feature_index) {
        const std::string& feat = feature_cols_[feature_index];
        report_progress("Reading feature columns",
                        "Reading time-series feature column '" + feat + "'",
                        0.25 + (feature_cols_.empty()
                            ? 0.0
                            : 0.15 * static_cast<double>(feature_index) /
                              static_cast<double>(feature_cols_.size())),
                        static_cast<uint64_t>(feature_index),
                        static_cast<uint64_t>(feature_cols_.size()));
        auto fcol = input->GetColumnByName(feat);
        if (!fcol) {
            return arrow::Status::KeyError(
                "TimeSeriesWindow: feature_col '" + feat + "' not found in input table");
        }
        std::vector<float> fvals;
        std::string fbad;
        if (!ReadColumnAsFloat(fcol, fvals, fbad)) {
            return arrow::Status::TypeError(
                "TimeSeriesWindow: feature_col '" + feat +
                "' has unsupported Arrow type '" + fbad + "'");
        }
        if (static_cast<int64_t>(fvals.size()) != static_cast<int64_t>(values.size())) {
            return arrow::Status::Invalid(
                "TimeSeriesWindow: feature_col '" + feat + "' row count (" +
                std::to_string(fvals.size()) + ") differs from value_col '" +
                value_col_ + "' row count (" + std::to_string(values.size()) + ")");
        }
        feature_values.push_back(std::move(fvals));
    }

    const int64_t n = static_cast<int64_t>(values.size());
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

    // Total columns in output: value_col windows + feature_col windows + 1 label.
    const size_t num_features = 1 + feature_cols_.size();
    const size_t total_x_cols = num_features * input_width_;
    const uint64_t estimated_window_matrix_bytes =
        static_cast<uint64_t>(num_windows) *
        static_cast<uint64_t>(total_x_cols + 1 + (time_col_.empty() ? 0 : 1)) *
        sizeof(float);
    report_progress("Planning windows",
                    "Planning " + std::to_string(num_windows) +
                    " windows x " + std::to_string(total_x_cols) +
                    " feature columns",
                    0.40,
                    0,
                    static_cast<uint64_t>(num_windows),
                    estimated_window_matrix_bytes);

    report_progress("Allocating Arrow columns",
                    "Allocating time-series window output columns",
                    0.45,
                    0,
                    static_cast<uint64_t>(num_windows),
                    estimated_window_matrix_bytes);
    arrow::MemoryPool* pool = arrow::default_memory_pool();
    std::vector<std::unique_ptr<arrow::FloatBuilder>> builders;
    builders.reserve(total_x_cols);
    for (size_t i = 0; i < total_x_cols; ++i) {
        builders.push_back(std::make_unique<arrow::FloatBuilder>(pool));
        ARROW_RETURN_NOT_OK(builders.back()->Reserve(num_windows));
    }
    arrow::FloatBuilder label_builder(pool);
    ARROW_RETURN_NOT_OK(label_builder.Reserve(num_windows));

    arrow::DoubleBuilder time_builder(pool);
    if (!time_col_.empty()) {
        ARROW_RETURN_NOT_OK(time_builder.Reserve(num_windows));
    }

    // Layout: value_col gets indices [0, input_width_), then feature_cols[0]
    // gets [input_width_, 2*input_width_), etc. Matches the schema's
    // per-feature column block ordering.
    for (int64_t w = 0; w < num_windows; ++w) {
        // value_col block
        for (int i = 0; i < input_width_; ++i) {
            ARROW_RETURN_NOT_OK(builders[i]->Append(values[w + i]));
        }
        // feature_col blocks
        for (size_t f = 0; f < feature_cols_.size(); ++f) {
            const size_t block_start = (f + 1) * input_width_;
            for (int i = 0; i < input_width_; ++i) {
                ARROW_RETURN_NOT_OK(
                    builders[block_start + i]->Append(feature_values[f][w + i]));
            }
        }
        // Target: value at step input_width + shift - 1 past window start.
        const int64_t target_idx = w + input_width_ + (shift_ - 1);
        ARROW_RETURN_NOT_OK(label_builder.Append(values[target_idx]));

        // Metadata: time at the window's first input step (index w).
        if (!time_col_.empty()) {
            ARROW_RETURN_NOT_OK(
                time_builder.Append(static_cast<double>(time_values[w])));
        }
        if ((w + 1) == num_windows || ((w + 1) % 1024) == 0) {
            const double write_progress =
                0.50 + (0.40 * static_cast<double>(w + 1) /
                        static_cast<double>(num_windows));
            report_progress("Building windows",
                            "Writing time-series windows to Arrow columns",
                            write_progress,
                            static_cast<uint64_t>(w + 1),
                            static_cast<uint64_t>(num_windows),
                            estimated_window_matrix_bytes);
        }
    }

    report_progress("Finishing Arrow table",
                    "Finalizing time-series window table",
                    0.95,
                    static_cast<uint64_t>(num_windows),
                    static_cast<uint64_t>(num_windows),
                    estimated_window_matrix_bytes);
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    arrays.reserve(total_x_cols + 1 + (time_col_.empty() ? 0 : 1));
    for (size_t i = 0; i < total_x_cols; ++i) {
        std::shared_ptr<arrow::Array> arr;
        ARROW_RETURN_NOT_OK(builders[i]->Finish(&arr));
        arrays.push_back(std::move(arr));
    }
    {
        std::shared_ptr<arrow::Array> arr;
        ARROW_RETURN_NOT_OK(label_builder.Finish(&arr));
        arrays.push_back(std::move(arr));
    }
    if (!time_col_.empty()) {
        std::shared_ptr<arrow::Array> arr;
        ARROW_RETURN_NOT_OK(time_builder.Finish(&arr));
        arrays.push_back(std::move(arr));
    }

    ARROW_ASSIGN_OR_RAISE(auto out_schema, InferOutputSchema(input->schema()));
    auto out_table = arrow::Table::Make(out_schema, arrays, num_windows);

    spdlog::info("TimeSeriesWindow: {} input values, {} feature_cols -> {} windows "
                 "(input_width={}, label_width=1, shift={}, total_x_cols={}{})",
                 n, feature_cols_.size(), num_windows,
                 input_width_, shift_, total_x_cols,
                 time_col_.empty() ? std::string()
                                   : ", time_col='" + time_col_ + "'");
    report_progress("Complete",
                    "TimeSeriesWindow materialization complete",
                    1.0,
                    static_cast<uint64_t>(num_windows),
                    static_cast<uint64_t>(num_windows),
                    estimated_window_matrix_bytes);
    return out_table;
}

} // namespace cyxwiz
