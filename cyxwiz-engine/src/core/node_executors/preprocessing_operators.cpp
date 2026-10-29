#include "preprocessing_operators.h"
#include "../split_partitioning.h"
#include "../materialization_memory_guard.h"
#include "../preprocessing_state.h"
#include "../profiler_trace.h"
#include "feature_matrix_utils.h"
#include "text_column_utils.h"
#include "ts_column_utils.h"

#include <cyxwiz/data_transform.h>
#include <cyxwiz/stats_utils.h>

#include <arrow/api.h>
#include <arrow/builder.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdint>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace cyxwiz {

namespace {

void ReportProgress(const PipelineOperatorProgressCallback& callback,
                    std::string stage,
                    std::string message,
                    double progress,
                    uint64_t processed = 0,
                    uint64_t total = 0,
                    uint64_t memory = 0) {
    if (!callback) {
        return;
    }
    PipelineOperatorProgress event;
    event.stage = std::move(stage);
    event.message = std::move(message);
    event.status = "running";
    event.progress = static_cast<float>(progress);
    event.processed_items = processed;
    event.total_items = total;
    event.estimated_memory_bytes = memory;
    callback(event);
}

std::string TrimString(const std::string& value) {
    auto begin = std::find_if_not(value.begin(), value.end(),
                                  [](unsigned char c) { return std::isspace(c); });
    auto end = std::find_if_not(value.rbegin(), value.rend(),
                                [](unsigned char c) { return std::isspace(c); }).base();
    if (begin >= end) {
        return {};
    }
    return std::string(begin, end);
}

std::string ToLowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return value;
}

// Shared Configure helper: parse columns (csv) + label_col.
void ParseColumnsAndLabel(
    const std::map<std::string, std::string>& params,
    std::vector<std::string>& out_cols,
    std::string& out_label,
    const std::string& cols_key = "columns") {
    out_label.clear();
    auto c = params.find(cols_key);
    const std::string cols_str = (c != params.end()) ? c->second : "";
    ParseCommaList(cols_str, out_cols);

    auto lc = params.find("label_col");
    if (lc != params.end()) out_label = lc->second;
}

bool ReadBoolParam(const std::map<std::string, std::string>& params,
                   const std::string& key,
                   bool default_value,
                   const std::string& op_name,
                   bool& out,
                   std::string& error) {
    auto it = params.find(key);
    if (it == params.end() || it->second.empty()) {
        out = default_value;
        return true;
    }
    if (it->second == "true") {
        out = true;
        return true;
    }
    if (it->second == "false") {
        out = false;
        return true;
    }
    error = op_name + ": '" + key + "' must be 'true' or 'false' (got '" +
            it->second + "')";
    return false;
}

// Read a numeric column as vector<double> (via float intermediate).
arrow::Status ReadNumericDouble(
    const std::shared_ptr<arrow::Table>& input,
    const std::string& col_name,
    const std::string& op_name,
    std::vector<double>& out,
    int& out_idx) {

    out_idx = input->schema()->GetFieldIndex(col_name);
    if (out_idx < 0) {
        return arrow::Status::KeyError(
            op_name + ": column '" + col_name + "' not found");
    }
    auto col = input->column(out_idx);
    std::vector<float> floats;
    std::string bad;
    if (!ReadColumnAsFloat(col, floats, bad)) {
        return arrow::Status::TypeError(
            op_name + ": column '" + col_name +
            "' must be numeric (got '" + bad + "')");
    }
    out.assign(floats.begin(), floats.end());
    return arrow::Status::OK();
}

// Replace a column with transformed float values (reuses
// ReplaceColumnWithFloat from ts_column_utils.h).
arrow::Result<std::shared_ptr<arrow::Table>> ReplaceWithDoubles(
    const std::shared_ptr<arrow::Table>& table,
    int col_idx,
    const std::vector<double>& values) {
    std::vector<float> floats;
    floats.reserve(values.size());
    for (double value : values) {
        floats.push_back(static_cast<float>(value));
    }
    return ReplaceColumnWithFloat(table, col_idx, floats,
                                   static_cast<int64_t>(floats.size()));
}

// Resolve columns param: explicit list OR auto-detect numeric.
// Shared between StandardScaler / MinMaxScaler / RobustScaler /
// OutlierDetector.
arrow::Status ResolveNumericColumnList(
    const std::shared_ptr<arrow::Table>& table,
    const std::vector<std::string>& explicit_names,
    const std::string& label_col,
    const std::vector<std::string>& excluded_names,
    const std::string& op_name,
    std::vector<std::string>& out) {
    ARROW_RETURN_NOT_OK(ResolveFeatureColumns(
        table, explicit_names, label_col, op_name, out));
    if (!explicit_names.empty() || excluded_names.empty()) {
        return arrow::Status::OK();
    }

    out.erase(
        std::remove_if(
            out.begin(), out.end(),
            [&excluded_names](const std::string& name) {
                return std::find(excluded_names.begin(), excluded_names.end(),
                                 name) != excluded_names.end();
            }),
        out.end());
    if (out.empty()) {
        return arrow::Status::Invalid(
            op_name +
            ": no numeric feature columns remain after exclusions. "
            "Remove an excluded column or specify Columns explicitly.");
    }
    return arrow::Status::OK();
}

arrow::Status ResolveNumericColumnList(
    const std::shared_ptr<arrow::Table>& table,
    const std::vector<std::string>& explicit_names,
    const std::string& label_col,
    const std::string& op_name,
    std::vector<std::string>& out) {
    return ResolveNumericColumnList(
        table, explicit_names, label_col, {}, op_name, out);
}

// Encode a string column to stable int32 codes using alphabetical
// ordering. Returns new int32 column + the ordered category map so
// the caller can log it.
arrow::Status EncodeStringColumn(
    const std::shared_ptr<arrow::ChunkedArray>& col,
    const std::string& col_name,
    const std::string& op_name,
    std::vector<int>& out_codes,
    std::vector<std::string>& out_categories) {

    out_codes.clear();
    out_categories.clear();
    out_codes.reserve(static_cast<size_t>(col->length()));

    // First pass: collect unique values in alphabetical order.
    std::set<std::string> unique;
    for (int c = 0; c < col->num_chunks(); ++c) {
        auto chunk = col->chunk(c);
        const int64_t chunk_len = chunk->length();
        if (chunk->type_id() == arrow::Type::STRING) {
            auto arr = std::static_pointer_cast<arrow::StringArray>(chunk);
            for (int64_t i = 0; i < chunk_len; ++i) {
                if (!arr->IsNull(i)) unique.insert(arr->GetString(i));
            }
        } else if (chunk->type_id() == arrow::Type::LARGE_STRING) {
            auto arr = std::static_pointer_cast<arrow::LargeStringArray>(chunk);
            for (int64_t i = 0; i < chunk_len; ++i) {
                if (!arr->IsNull(i)) unique.insert(arr->GetString(i));
            }
        } else {
            return arrow::Status::TypeError(
                op_name + ": column '" + col_name +
                "' must be string/large_string (got '" +
                chunk->type()->ToString() + "')");
        }
    }
    out_categories.assign(unique.begin(), unique.end());
    if (out_categories.size() >
        static_cast<size_t>((std::numeric_limits<int>::max)())) {
        return arrow::Status::Invalid(
            op_name + ": column '" + col_name +
            "' has too many categories for int32 encoding");
    }

    std::unordered_map<std::string, int> code_map;
    for (size_t i = 0; i < out_categories.size(); ++i) {
        code_map[out_categories[i]] = static_cast<int>(i);
    }

    // Second pass: encode. Null maps to -1.
    for (int c = 0; c < col->num_chunks(); ++c) {
        auto chunk = col->chunk(c);
        const int64_t chunk_len = chunk->length();
        if (chunk->type_id() == arrow::Type::STRING) {
            auto arr = std::static_pointer_cast<arrow::StringArray>(chunk);
            for (int64_t i = 0; i < chunk_len; ++i) {
                if (arr->IsNull(i)) out_codes.push_back(-1);
                else out_codes.push_back(code_map[arr->GetString(i)]);
            }
        } else {
            auto arr = std::static_pointer_cast<arrow::LargeStringArray>(chunk);
            for (int64_t i = 0; i < chunk_len; ++i) {
                if (arr->IsNull(i)) out_codes.push_back(-1);
                else out_codes.push_back(code_map[arr->GetString(i)]);
            }
        }
    }
    return arrow::Status::OK();
}

// Replace a column with int32 codes (new field type = int32).
arrow::Result<std::shared_ptr<arrow::Table>> ReplaceWithInts(
    const std::shared_ptr<arrow::Table>& table,
    int col_idx,
    const std::vector<int>& codes) {
    const int64_t n = static_cast<int64_t>(codes.size());
    arrow::Int32Builder builder;
    ARROW_RETURN_NOT_OK(builder.Reserve(n));
    for (int c : codes) ARROW_RETURN_NOT_OK(builder.Append(c));
    std::shared_ptr<arrow::Array> arr;
    ARROW_RETURN_NOT_OK(builder.Finish(&arr));

    const std::string col_name = table->schema()->field(col_idx)->name();
    auto field = arrow::field(col_name, arrow::int32());
    auto chunked = std::make_shared<arrow::ChunkedArray>(arr);
    return table->SetColumn(col_idx, field, chunked);
}

std::string BuildPreprocessingMemoryPreflightMessage(
    const std::string& op_name,
    const MaterializationMemoryEstimate& estimate,
    const MaterializationMemoryDecision& decision,
    const std::string& suggestion) {
    std::ostringstream ss;
    ss << op_name << " memory preflight: risk="
       << MaterializationMemoryRiskName(decision.risk)
       << ", rows=" << estimate.rows
       << ", planned_columns=" << estimate.output_features
       << ", raw=" << FormatMaterializationBytes(estimate.raw_output_bytes)
       << ", estimated_peak="
       << FormatMaterializationBytes(estimate.estimated_peak_bytes)
       << ", available="
       << FormatMaterializationBytes(decision.available_bytes)
       << ", safe_budget="
       << FormatMaterializationBytes(decision.safe_budget_bytes)
       << ". " << decision.reason
       << ". Suggestion: " << suggestion;
    return ss.str();
}

arrow::Result<MaterializationMemoryEstimate> EmitPreprocessingMemoryPreflight(
    const std::shared_ptr<arrow::Table>& input,
    const std::string& op_name,
    uint64_t planned_columns,
    uint64_t bytes_per_value,
    const std::string& suggestion,
    const MaterializationMemoryContext& memory_context,
    const PipelineOperatorProgressCallback& callback) {
    const uint64_t planned_rows =
        static_cast<uint64_t>(std::max<int64_t>(0, input->num_rows()));
    if (planned_rows == 0) {
        return arrow::Status::Invalid(op_name + ": input table has no rows");
    }
    if (planned_columns == 0) {
        return arrow::Status::Invalid(
            op_name + ": no columns selected for preprocessing");
    }

    const auto estimate = EstimateDenseMaterializationMemory(
        planned_rows, planned_columns, bytes_per_value);
    const auto decision = EvaluateMaterializationMemory(
        estimate, memory_context);
    const std::string preflight_message =
        BuildPreprocessingMemoryPreflightMessage(
            op_name, estimate, decision, suggestion);

    uint64_t planned_cells = 0;
    if (!CheckedMulU64(planned_rows, planned_columns, planned_cells)) {
        planned_cells = (std::numeric_limits<uint64_t>::max)();
    }

    if (callback) {
        PipelineOperatorProgress event;
        event.stage = op_name + " memory preflight";
        event.message = preflight_message;
        event.status = MaterializationMemoryRiskToProgressStatus(decision.risk);
        event.progress = 0.12f;
        event.processed_items = 0;
        event.total_items = planned_cells;
        event.estimated_memory_bytes = estimate.estimated_peak_bytes;
        event.memory_risk_level = MaterializationMemoryRiskName(decision.risk);
        callback(event);
    }
    if (decision.blocked) {
        return arrow::Status::CapacityError(
            "Materialization blocked: " + preflight_message);
    }
    return estimate;
}

} // namespace

// ============================================================================
// ColumnTransformOperator: what the fitted column transforms share
// ============================================================================

namespace {

// The present (finite) values of a column and where they sit.
void SplitPresent(const std::vector<double>& all, std::vector<double>& present, std::vector<size_t>& rows) {
    present.clear();
    rows.clear();
    for (size_t i = 0; i < all.size(); ++i) {
        if (!std::isfinite(all[i])) continue;
        present.push_back(all[i]);
        rows.push_back(i);
    }
}

// A column of transformed values, missing (null) where the input was.
arrow::Result<std::shared_ptr<arrow::Table>> ReplaceKeepingMissing(
    const std::shared_ptr<arrow::Table>& table, int col_idx, size_t rows,
    const std::vector<size_t>& present_rows, const std::vector<double>& values) {
    arrow::FloatBuilder builder(arrow::default_memory_pool());
    ARROW_RETURN_NOT_OK(builder.Reserve(static_cast<int64_t>(rows)));
    size_t next = 0;
    for (size_t row = 0; row < rows; ++row) {
        if (next < present_rows.size() && present_rows[next] == row) {
            ARROW_RETURN_NOT_OK(builder.Append(static_cast<float>(values[next++])));
        } else {
            ARROW_RETURN_NOT_OK(builder.AppendNull());
        }
    }
    std::shared_ptr<arrow::Array> array;
    ARROW_RETURN_NOT_OK(builder.Finish(&array));
    const std::string name = table->schema()->field(col_idx)->name();
    return table->SetColumn(col_idx, arrow::field(name, arrow::float32()), std::make_shared<arrow::ChunkedArray>(array));
}

bool ReadDoubleParam(const std::map<std::string, std::string>& params, const std::string& key, double& out,
                     const std::string& op_name, std::string& error) {
    auto it = params.find(key);
    if (it == params.end() || TrimString(it->second).empty()) return true;
    try {
        size_t used = 0;
        out = std::stod(it->second, &used);
        if (used != TrimString(it->second).size() || !std::isfinite(out)) throw std::invalid_argument(key);
        return true;
    } catch (...) {
        error = op_name + ": '" + key + "' is not a number: " + it->second;
        return false;
    }
}

double Saved(const PreprocessingFeatureState& feature, const char* key) {
    auto it = feature.numeric_values.find(key);
    return it == feature.numeric_values.end() ? NAN : it->second;
}

}  // namespace

bool ColumnTransformOperator::Configure(const std::map<std::string, std::string>& params, std::string& error) {
    ParseColumnsAndLabel(params, columns_, label_col_);
    const auto excluded = params.find("exclude_columns");
    ParseCommaList(excluded == params.end() ? "" : excluded->second, exclude_columns_);
    if (!ParseFittedPreprocessingOptions(params, GetName(), options_, error)) return false;
    return ConfigureOwn(params, error);
}

arrow::Status ColumnTransformOperator::BeginStream(const std::shared_ptr<arrow::Schema>& schema) {
    if (!schema) return arrow::Status::Invalid(GetName() + ": input is null");
    schema_ = schema;
    resolved_.clear();
    state_ = FittedPreprocessingState{};
    fit_rows_ = 0;
    if (options_.IsTransformOnly()) {
        std::string state_error;
        if (!LoadFittedPreprocessingState(options_.state_path, GetName(), state_, state_error)) {
            return arrow::Status::Invalid(GetName() + ": " + state_error);
        }
        if (!ValidateFittedPreprocessingStateSchema(state_, schema, state_error)) {
            return arrow::Status::Invalid(state_error);
        }
        if (!ValidateFittedPreprocessingConfiguration(state_, Settings(), state_error)) {
            return arrow::Status::Invalid(GetName() + ": " + state_error);
        }
        for (const auto& feature : state_.features) resolved_.push_back(feature.name);
        if (!columns_.empty() && columns_ != resolved_) {
            return arrow::Status::Invalid(
                GetName() + ": the columns differ from the saved values' columns. Keep the columns used when "
                "fitting, or leave Columns empty.");
        }
        BeginFit(resolved_.size());
        for (size_t i = 0; i < state_.features.size(); ++i) {
            ARROW_RETURN_NOT_OK(LoadColumn(i, state_.features[i]));
        }
        spdlog::info("[Preprocessing] {} uses saved values '{}' (fit_rows={}, columns={})", GetName(),
                     options_.state_path, state_.fit_rows, resolved_.size());
        return arrow::Status::OK();
    }
    for (const auto& name : exclude_columns_) {
        if (schema->GetFieldIndex(name) < 0) {
            return arrow::Status::KeyError(GetName() + ": excluded column '" + name + "' not found");
        }
    }
    ARROW_ASSIGN_OR_RAISE(auto empty, arrow::Table::MakeEmpty(schema));
    ARROW_RETURN_NOT_OK(ResolveNumericColumnList(empty, columns_, label_col_, exclude_columns_, GetName(), resolved_));
    BeginFit(resolved_.size());
    state_.operator_name = GetName();
    state_.input_schema_fingerprint = FingerprintPreprocessingSchema(schema);
    state_.configuration = Settings();
    state_.configuration["label_col"] = label_col_;
    std::string excluded;
    for (const auto& name : exclude_columns_) excluded += (excluded.empty() ? "" : ",") + name;
    state_.configuration["exclude_columns"] = excluded;
    return arrow::Status::OK();
}

arrow::Status ColumnTransformOperator::FitBatch(const std::shared_ptr<arrow::Table>& batch) {
    std::vector<double> all, present;
    std::vector<size_t> rows;
    for (size_t i = 0; i < resolved_.size(); ++i) {
        int idx = -1;
        ARROW_RETURN_NOT_OK(ReadNumericDouble(batch, resolved_[i], GetName(), all, idx));
        SplitPresent(all, present, rows);
        if (!present.empty()) ARROW_RETURN_NOT_OK(FitValues(i, present));
    }
    fit_rows_ += batch->num_rows();
    return arrow::Status::OK();
}

arrow::Status ColumnTransformOperator::EndFit() {
    state_.fit_rows = fit_rows_;
    state_.features.clear();
    for (size_t i = 0; i < resolved_.size(); ++i) {
        PreprocessingFeatureState feature;
        feature.name = resolved_[i];
        feature.data_type = schema_->field(schema_->GetFieldIndex(resolved_[i]))->type()->ToString();
        ARROW_RETURN_NOT_OK(EndColumnFit(i, feature));
        state_.features.push_back(std::move(feature));
    }
    return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::Table>> ColumnTransformOperator::ApplyBatch(const std::shared_ptr<arrow::Table>& batch) {
    auto out = batch;
    std::vector<double> all, present;
    std::vector<size_t> rows;
    for (size_t i = 0; i < resolved_.size(); ++i) {
        int idx = -1;
        ARROW_RETURN_NOT_OK(ReadNumericDouble(out, resolved_[i], GetName(), all, idx));
        SplitPresent(all, present, rows);
        std::vector<double> transformed;
        if (!present.empty()) {
            ARROW_ASSIGN_OR_RAISE(transformed, TransformValues(i, std::move(present)));
            if (transformed.size() != rows.size()) {
                return arrow::Status::Invalid(GetName() + ": column '" + resolved_[i] + "' changed length");
            }
        }
        ARROW_ASSIGN_OR_RAISE(out, ReplaceKeepingMissing(out, idx, all.size(), rows, transformed));
    }
    return out;
}

arrow::Status ColumnTransformOperator::EndStream() {
    if (!options_.IsTransformOnly() && options_.save_state) {
        std::string state_error;
        if (!SaveFittedPreprocessingState(options_.state_path, state_, options_.state_overwrite, state_error)) {
            return arrow::Status::IOError(GetName() + ": failed to save the fitted values: " + state_error);
        }
        spdlog::info("[Preprocessing] {} saved its fitted values '{}' (fit_rows={}, columns={})", GetName(),
                     options_.state_path, state_.fit_rows, state_.features.size());
    }
    spdlog::info("[Preprocessing] {} done: {} columns ({})", GetName(), resolved_.size(),
                 options_.IsTransformOnly() ? "saved values" : "fitted here");
    return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::Table>> ColumnTransformOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz ColumnTransform Materializer");
    if (!input) return arrow::Status::Invalid(GetName() + ": input is null");
    ReportProgress(progress_callback_, "Resolving columns", "Resolving the columns for " + Describe(), 0.05);
    ARROW_RETURN_NOT_OK(BeginStream(input->schema()));
    ARROW_ASSIGN_OR_RAISE(auto preflight, EmitPreprocessingMemoryPreflight(
        input, GetName(), static_cast<uint64_t>(resolved_.size()), static_cast<uint64_t>(sizeof(double)),
        "reduce the selected rows or columns, or transform a sample first.", GetMaterializationMemoryContext(),
        progress_callback_));
    const uint64_t estimated_bytes = preflight.estimated_peak_bytes;
    ReportProgress(progress_callback_, Describe(), Describe() + " of " + std::to_string(resolved_.size()) + " columns",
                   0.20, 0, static_cast<uint64_t>(resolved_.size()), estimated_bytes);
    // The whole table is one batch of the stream; it learns from the Train
    // rows when the pipeline run marked them (TOFIX178 S0-g3).
    if (NeedsFitPass()) {
        ARROW_ASSIGN_OR_RAISE(auto train_rows, TrainRowsOf(input));
        ARROW_RETURN_NOT_OK(FitBatch(train_rows));
        ARROW_RETURN_NOT_OK(EndFit());
    }
    ARROW_ASSIGN_OR_RAISE(auto out, ApplyBatch(input));
    ARROW_RETURN_NOT_OK(EndStream());
    ReportProgress(progress_callback_, "Complete", GetName() + " complete", 1.0, static_cast<uint64_t>(resolved_.size()),
                   static_cast<uint64_t>(resolved_.size()), estimated_bytes);
    return out;
}

// ============================================================================
// StandardScalerOperator
// ============================================================================

bool StandardScalerOperator::ConfigureOwn(const std::map<std::string, std::string>& params, std::string& error) {
    return ReadBoolParam(params, "with_mean", true, GetName(), with_mean_, error) &&
           ReadBoolParam(params, "with_std", true, GetName(), with_std_, error);
}

std::map<std::string, std::string> StandardScalerOperator::Settings() const {
    return {{"with_mean", with_mean_ ? "true" : "false"}, {"with_std", with_std_ ? "true" : "false"}};
}

void StandardScalerOperator::BeginFit(size_t columns) {
    moments_.assign(columns, {});
    centers_.assign(columns, 0.0);
    scales_.assign(columns, 1.0);
}

arrow::Status StandardScalerOperator::FitValues(size_t column, const std::vector<double>& present) {
    // Each batch's moments on the device, merged exactly (population std, scikit-learn).
    std::vector<DataTransform::ColumnMoments> moments;
    std::string error;
    if (!DataTransform::Moments({present}, moments, error)) {
        return arrow::Status::Invalid(GetName() + ": column '" + ColumnName(column) + "': " + error);
    }
    DataTransform::CombineMoments(moments_[column], moments[0]);
    return arrow::Status::OK();
}

arrow::Status StandardScalerOperator::EndColumnFit(size_t column, PreprocessingFeatureState& feature) {
    StandardizeOptions options;
    options.with_mean = with_mean_;
    options.with_std = with_std_;
    if (moments_[column].count < 1 ||
        !DataTransform::StandardizeParameters(moments_[column], options, centers_[column], scales_[column])) {
        return arrow::Status::Invalid(GetName() + ": column '" + ColumnName(column) + "' has no values to fit on");
    }
    feature.numeric_values["mean"] = centers_[column];
    feature.numeric_values["scale"] = scales_[column];
    return arrow::Status::OK();
}

arrow::Status StandardScalerOperator::LoadColumn(size_t column, const PreprocessingFeatureState& feature) {
    const double mean = Saved(feature, "mean"), scale = Saved(feature, "scale");
    if (!std::isfinite(mean) || !std::isfinite(scale) || scale <= 0.0) {
        return arrow::Status::Invalid(GetName() + ": the saved values of column '" + feature.name +
                                      "' are missing or invalid. Fit again.");
    }
    centers_[column] = mean;
    scales_[column] = scale;
    return arrow::Status::OK();
}

arrow::Result<std::vector<double>> StandardScalerOperator::TransformValues(size_t column, std::vector<double> present) {
    TransformResult scaled = DataTransform::ScaleColumns({std::move(present)}, {centers_[column]}, {scales_[column]});
    if (!scaled.success) return arrow::Status::Invalid(GetName() + ": column '" + ColumnName(column) + "': " + scaled.error_message);
    return std::move(scaled.transformed_data[0]);
}

// ============================================================================
// MinMaxScalerOperator
// ============================================================================

bool MinMaxScalerOperator::ConfigureOwn(const std::map<std::string, std::string>& params, std::string& error) {
    range_min_ = 0.0;
    range_max_ = 1.0;
    if (!ReadDoubleParam(params, "min", range_min_, GetName(), error)) return false;
    if (!ReadDoubleParam(params, "max", range_max_, GetName(), error)) return false;
    if (range_max_ <= range_min_) {
        error = GetName() + ": max (" + std::to_string(range_max_) + ") must be greater than min (" +
                std::to_string(range_min_) + ")";
        return false;
    }
    return true;
}

std::map<std::string, std::string> MinMaxScalerOperator::Settings() const {
    char lo[40], hi[40];
    std::snprintf(lo, sizeof(lo), "%.17g", range_min_);
    std::snprintf(hi, sizeof(hi), "%.17g", range_max_);
    return {{"min", lo}, {"max", hi}};
}

void MinMaxScalerOperator::BeginFit(size_t columns) {
    moments_.assign(columns, {});
    lows_.assign(columns, 0.0);
    highs_.assign(columns, 0.0);
}

arrow::Status MinMaxScalerOperator::FitValues(size_t column, const std::vector<double>& present) {
    std::vector<DataTransform::ColumnMoments> moments;
    std::string error;
    if (!DataTransform::Moments({present}, moments, error)) {
        return arrow::Status::Invalid(GetName() + ": column '" + ColumnName(column) + "': " + error);
    }
    DataTransform::CombineMoments(moments_[column], moments[0]);
    return arrow::Status::OK();
}

arrow::Status MinMaxScalerOperator::EndColumnFit(size_t column, PreprocessingFeatureState& feature) {
    if (moments_[column].count < 1) {
        return arrow::Status::Invalid(GetName() + ": column '" + ColumnName(column) + "' has no values to fit on");
    }
    lows_[column] = moments_[column].min;
    highs_[column] = moments_[column].max;
    feature.numeric_values["data_min"] = lows_[column];
    feature.numeric_values["data_max"] = highs_[column];
    return arrow::Status::OK();
}

arrow::Status MinMaxScalerOperator::LoadColumn(size_t column, const PreprocessingFeatureState& feature) {
    const double lo = Saved(feature, "data_min"), hi = Saved(feature, "data_max");
    if (!std::isfinite(lo) || !std::isfinite(hi) || hi < lo) {
        return arrow::Status::Invalid(GetName() + ": the saved values of column '" + feature.name +
                                      "' are missing or invalid. Fit again.");
    }
    lows_[column] = lo;
    highs_[column] = hi;
    return arrow::Status::OK();
}

arrow::Result<std::vector<double>> MinMaxScalerOperator::TransformValues(size_t column, std::vector<double> present) {
    // One implementation with the scaling tools, on the selected device.
    TransformResult scaled = DataTransform::NormalizeColumns({std::move(present)}, {lows_[column]}, {highs_[column]},
                                                             range_min_, range_max_);
    if (!scaled.success) return arrow::Status::Invalid(GetName() + ": column '" + ColumnName(column) + "': " + scaled.error_message);
    return std::move(scaled.transformed_data[0]);
}

// ============================================================================
// RobustScalerOperator
// ============================================================================

bool RobustScalerOperator::ConfigureOwn(const std::map<std::string, std::string>& params, std::string& error) {
    quantile_min_ = 25.0;
    quantile_max_ = 75.0;
    if (!ReadBoolParam(params, "with_centering", true, GetName(), with_centering_, error)) return false;
    if (!ReadBoolParam(params, "with_scaling", true, GetName(), with_scaling_, error)) return false;
    if (!ReadDoubleParam(params, "quantile_min", quantile_min_, GetName(), error)) return false;
    if (!ReadDoubleParam(params, "quantile_max", quantile_max_, GetName(), error)) return false;
    if (quantile_min_ < 0.0 || quantile_max_ > 100.0 || quantile_max_ <= quantile_min_) {
        error = GetName() + ": quantiles must satisfy 0 <= qmin < qmax <= 100 (got qmin=" + std::to_string(quantile_min_) +
                ", qmax=" + std::to_string(quantile_max_) + ")";
        return false;
    }
    return true;
}

std::map<std::string, std::string> RobustScalerOperator::Settings() const {
    char lo[40], hi[40];
    std::snprintf(lo, sizeof(lo), "%.17g", quantile_min_);
    std::snprintf(hi, sizeof(hi), "%.17g", quantile_max_);
    return {{"with_centering", with_centering_ ? "true" : "false"}, {"with_scaling", with_scaling_ ? "true" : "false"},
            {"quantile_min", lo}, {"quantile_max", hi}};
}

void RobustScalerOperator::BeginFit(size_t columns) {
    values_.assign(columns, {});
    centers_.assign(columns, 0.0);
    scales_.assign(columns, 1.0);
}

arrow::Status RobustScalerOperator::FitValues(size_t column, const std::vector<double>& present) {
    values_[column].insert(values_[column].end(), present.begin(), present.end());
    return arrow::Status::OK();
}

arrow::Status RobustScalerOperator::EndColumnFit(size_t column, PreprocessingFeatureState& feature) {
    if (values_[column].empty()) {
        return arrow::Status::Invalid(GetName() + ": column '" + ColumnName(column) + "' has no values to fit on");
    }
    RobustScaleOptions options;
    options.quantile_min = quantile_min_;
    options.quantile_max = quantile_max_;
    options.with_centering = with_centering_;
    options.with_scaling = with_scaling_;
    // The fit on the device: the centre and scale RobustScale uses.
    const TransformResult fitted = DataTransform::RobustScale({values_[column]}, options);
    if (!fitted.success) return arrow::Status::Invalid(GetName() + ": column '" + ColumnName(column) + "': " + fitted.error_message);
    centers_[column] = fitted.params.at("col0_median");
    scales_[column] = fitted.params.at("col0_iqr");
    values_[column].clear();
    values_[column].shrink_to_fit();
    feature.numeric_values["center"] = centers_[column];
    feature.numeric_values["scale"] = scales_[column];
    return arrow::Status::OK();
}

arrow::Status RobustScalerOperator::LoadColumn(size_t column, const PreprocessingFeatureState& feature) {
    const double center = Saved(feature, "center"), scale = Saved(feature, "scale");
    if (!std::isfinite(center) || !std::isfinite(scale) || scale <= 0.0) {
        return arrow::Status::Invalid(GetName() + ": the saved values of column '" + feature.name +
                                      "' are missing or invalid. Fit again.");
    }
    centers_[column] = center;
    scales_[column] = scale;
    return arrow::Status::OK();
}

arrow::Result<std::vector<double>> RobustScalerOperator::TransformValues(size_t column, std::vector<double> present) {
    TransformResult scaled = DataTransform::ScaleColumns({std::move(present)}, {centers_[column]}, {scales_[column]});
    if (!scaled.success) return arrow::Status::Invalid(GetName() + ": column '" + ColumnName(column) + "': " + scaled.error_message);
    return std::move(scaled.transformed_data[0]);
}

// ============================================================================
// MaxAbsScalerOperator
// ============================================================================

void MaxAbsScalerOperator::BeginFit(size_t columns) {
    moments_.assign(columns, {});
    scales_.assign(columns, 1.0);
}

arrow::Status MaxAbsScalerOperator::FitValues(size_t column, const std::vector<double>& present) {
    std::vector<DataTransform::ColumnMoments> moments;
    std::string error;
    if (!DataTransform::Moments({present}, moments, error)) {
        return arrow::Status::Invalid(GetName() + ": column '" + ColumnName(column) + "': " + error);
    }
    DataTransform::CombineMoments(moments_[column], moments[0]);
    return arrow::Status::OK();
}

arrow::Status MaxAbsScalerOperator::EndColumnFit(size_t column, PreprocessingFeatureState& feature) {
    if (moments_[column].count < 1) {
        return arrow::Status::Invalid(GetName() + ": column '" + ColumnName(column) + "' has no values to fit on");
    }
    const double max_abs = std::max(std::fabs(moments_[column].min), std::fabs(moments_[column].max));
    scales_[column] = max_abs > 0.0 ? max_abs : 1.0;  // a column of zeros stays as it is (scikit-learn)
    feature.numeric_values["max_abs"] = scales_[column];
    return arrow::Status::OK();
}

arrow::Status MaxAbsScalerOperator::LoadColumn(size_t column, const PreprocessingFeatureState& feature) {
    const double scale = Saved(feature, "max_abs");
    if (!std::isfinite(scale) || scale <= 0.0) {
        return arrow::Status::Invalid(GetName() + ": the saved values of column '" + feature.name +
                                      "' are missing or invalid. Fit again.");
    }
    scales_[column] = scale;
    return arrow::Status::OK();
}

arrow::Result<std::vector<double>> MaxAbsScalerOperator::TransformValues(size_t column, std::vector<double> present) {
    TransformResult scaled = DataTransform::ScaleColumns({std::move(present)}, {0.0}, {scales_[column]});
    if (!scaled.success) return arrow::Status::Invalid(GetName() + ": column '" + ColumnName(column) + "': " + scaled.error_message);
    return std::move(scaled.transformed_data[0]);
}

// ============================================================================
// QuantileTransformOperator
// ============================================================================

bool QuantileTransformOperator::ConfigureOwn(const std::map<std::string, std::string>& params, std::string& error) {
    distribution_ = "uniform";
    n_quantiles_ = 1000;
    if (auto it = params.find("output_distribution"); it != params.end() && !TrimString(it->second).empty()) {
        distribution_ = ToLowerAscii(TrimString(it->second));
    }
    if (distribution_ != "uniform" && distribution_ != "normal") {
        error = GetName() + ": 'output_distribution' must be 'uniform' or 'normal' (got '" + distribution_ + "')";
        return false;
    }
    double n = n_quantiles_;
    if (!ReadDoubleParam(params, "n_quantiles", n, GetName(), error)) return false;
    if (n < 2 || n > 100000 || n != std::floor(n)) {
        error = GetName() + ": 'n_quantiles' must be a whole number from 2 to 100000";
        return false;
    }
    n_quantiles_ = static_cast<int>(n);
    return true;
}

std::map<std::string, std::string> QuantileTransformOperator::Settings() const {
    return {{"output_distribution", distribution_}, {"n_quantiles", std::to_string(n_quantiles_)}};
}

void QuantileTransformOperator::BeginFit(size_t columns) {
    values_.assign(columns, {});
    quantiles_.assign(columns, {});
}

arrow::Status QuantileTransformOperator::FitValues(size_t column, const std::vector<double>& present) {
    values_[column].insert(values_[column].end(), present.begin(), present.end());
    return arrow::Status::OK();
}

arrow::Status QuantileTransformOperator::EndColumnFit(size_t column, PreprocessingFeatureState& feature) {
    std::string error;
    if (!DataTransform::QuantileFit(values_[column], n_quantiles_, quantiles_[column], error)) {
        return arrow::Status::Invalid(GetName() + ": column '" + ColumnName(column) + "': " + error);
    }
    values_[column].clear();
    values_[column].shrink_to_fit();
    feature.numeric_values["count"] = static_cast<double>(quantiles_[column].size());
    for (size_t q = 0; q < quantiles_[column].size(); ++q) {
        char key[16];
        std::snprintf(key, sizeof(key), "q%05zu", q);
        feature.numeric_values[key] = quantiles_[column][q];
    }
    return arrow::Status::OK();
}

arrow::Status QuantileTransformOperator::LoadColumn(size_t column, const PreprocessingFeatureState& feature) {
    const double count = Saved(feature, "count");
    std::vector<double> quantiles;
    for (size_t q = 0; std::isfinite(count) && q < static_cast<size_t>(count); ++q) {
        char key[16];
        std::snprintf(key, sizeof(key), "q%05zu", q);
        const double value = Saved(feature, key);
        if (!std::isfinite(value) || (!quantiles.empty() && value < quantiles.back())) {
            return arrow::Status::Invalid(GetName() + ": the saved quantiles of column '" + feature.name +
                                          "' are missing or out of order. Fit again.");
        }
        quantiles.push_back(value);
    }
    if (quantiles.size() < 2) {
        return arrow::Status::Invalid(GetName() + ": the saved values of column '" + feature.name + "' are missing. Fit again.");
    }
    quantiles_[column] = std::move(quantiles);
    return arrow::Status::OK();
}

arrow::Result<std::vector<double>> QuantileTransformOperator::TransformValues(size_t column, std::vector<double> present) {
    TransformResult mapped = DataTransform::QuantileApply({std::move(present)}, {quantiles_[column]}, distribution_);
    if (!mapped.success) return arrow::Status::Invalid(GetName() + ": column '" + ColumnName(column) + "': " + mapped.error_message);
    return std::move(mapped.transformed_data[0]);
}

// ============================================================================
// PowerTransformOperator
// ============================================================================

bool PowerTransformOperator::ConfigureOwn(const std::map<std::string, std::string>& params, std::string& error) {
    std::string family = "yeo-johnson", mode = "auto";
    if (auto it = params.find("family"); it != params.end() && !TrimString(it->second).empty()) family = ToLowerAscii(TrimString(it->second));
    if (auto it = params.find("lambda_mode"); it != params.end() && !TrimString(it->second).empty()) mode = ToLowerAscii(TrimString(it->second));
    if (family != "box-cox" && family != "yeo-johnson") {
        error = GetName() + ": 'family' must be 'box-cox' or 'yeo-johnson' (got '" + family + "')";
        return false;
    }
    if (mode != "auto" && mode != "fixed") {
        error = GetName() + ": 'lambda_mode' must be 'auto' or 'fixed' (got '" + mode + "')";
        return false;
    }
    box_cox_ = family == "box-cox";
    auto_lambda_ = mode == "auto";
    lambda_ = 0.0;
    if (!ReadDoubleParam(params, "lambda", lambda_, GetName(), error)) return false;
    if (std::fabs(lambda_) > 10.0) {
        error = GetName() + ": 'lambda' must be between -10 and 10";
        return false;
    }
    return true;
}

std::map<std::string, std::string> PowerTransformOperator::Settings() const {
    char lam[40];
    std::snprintf(lam, sizeof(lam), "%.17g", lambda_);
    return {{"family", box_cox_ ? "box-cox" : "yeo-johnson"}, {"lambda_mode", auto_lambda_ ? "auto" : "fixed"},
            {"lambda", auto_lambda_ ? "" : lam}};
}

void PowerTransformOperator::BeginFit(size_t columns) {
    values_.assign(columns, {});
    lambdas_.assign(columns, lambda_);
}

arrow::Status PowerTransformOperator::FitValues(size_t column, const std::vector<double>& present) {
    values_[column].insert(values_[column].end(), present.begin(), present.end());
    return arrow::Status::OK();
}

arrow::Status PowerTransformOperator::EndColumnFit(size_t column, PreprocessingFeatureState& feature) {
    const auto& x = values_[column];
    if (x.empty()) return arrow::Status::Invalid(GetName() + ": column '" + ColumnName(column) + "' has no values to fit on");
    if (box_cox_ && !DataTransform::CanApplyBoxCox(x)) {
        return arrow::Status::Invalid(GetName() + ": column '" + ColumnName(column) +
                                      "' has values of 0 or less; Box-Cox needs values above 0. Use Yeo-Johnson.");
    }
    if (auto_lambda_) {
        const BoxCoxLambdaResult search = box_cox_ ? DataTransform::FindOptimalLambda(x) : DataTransform::FindOptimalYeoJohnsonLambda(x);
        if (!search.success) return arrow::Status::Invalid(GetName() + ": column '" + ColumnName(column) + "': " + search.error_message);
        lambdas_[column] = search.optimal_lambda;
    }
    values_[column].clear();
    values_[column].shrink_to_fit();
    feature.numeric_values["lambda"] = lambdas_[column];
    return arrow::Status::OK();
}

arrow::Status PowerTransformOperator::LoadColumn(size_t column, const PreprocessingFeatureState& feature) {
    const double lambda = Saved(feature, "lambda");
    if (!std::isfinite(lambda)) {
        return arrow::Status::Invalid(GetName() + ": the saved lambda of column '" + feature.name + "' is missing. Fit again.");
    }
    lambdas_[column] = lambda;
    return arrow::Status::OK();
}

arrow::Result<std::vector<double>> PowerTransformOperator::TransformValues(size_t column, std::vector<double> present) {
    TransformResult y = box_cox_ ? DataTransform::BoxCoxColumn(present, lambdas_[column], false)
                                 : DataTransform::YeoJohnsonColumn(present, lambdas_[column], false);
    if (!y.success) {
        return arrow::Status::Invalid(GetName() + ": column '" + ColumnName(column) + "': " + y.error_message +
                                      (box_cox_ ? " (Box-Cox needs values above 0; use Yeo-Johnson)" : ""));
    }
    return std::move(y.transformed_data[0]);
}

// ============================================================================
// LogTransformOperator
// ============================================================================

bool LogTransformOperator::Configure(const std::map<std::string, std::string>& params, std::string& error) {
    ParseColumnsAndLabel(params, columns_, label_col_);
    const auto excluded = params.find("exclude_columns");
    ParseCommaList(excluded == params.end() ? "" : excluded->second, exclude_columns_);
    const auto base = params.find("base");
    base_ = base == params.end() || TrimString(base->second).empty() ? "natural" : ToLowerAscii(TrimString(base->second));
    if (base_ != "natural" && base_ != "log10" && base_ != "log2") {
        error = GetName() + ": 'base' must be natural, log10 or log2 (got '" + base->second + "')";
        return false;
    }
    return ReadBoolParam(params, "use_log1p", true, GetName(), log1p_, error);
}

arrow::Status LogTransformOperator::BeginStream(const std::shared_ptr<arrow::Schema>& schema) {
    if (!schema) return arrow::Status::Invalid(GetName() + ": input is null");
    for (const auto& name : exclude_columns_) {
        if (schema->GetFieldIndex(name) < 0) {
            return arrow::Status::KeyError(GetName() + ": excluded column '" + name + "' not found");
        }
    }
    ARROW_ASSIGN_OR_RAISE(auto empty, arrow::Table::MakeEmpty(schema));
    resolved_.clear();
    return ResolveNumericColumnList(empty, columns_, label_col_, exclude_columns_, GetName(), resolved_);
}

arrow::Result<std::shared_ptr<arrow::Table>> LogTransformOperator::ApplyBatch(const std::shared_ptr<arrow::Table>& batch) {
    auto out = batch;
    std::vector<double> all, present;
    std::vector<size_t> rows;
    for (const auto& name : resolved_) {
        ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
        int idx = -1;
        ARROW_RETURN_NOT_OK(ReadNumericDouble(out, name, GetName(), all, idx));
        SplitPresent(all, present, rows);
        std::vector<double> transformed;
        if (!present.empty()) {
            // One implementation with the backend's log transform, on the selected device.
            const TransformResult result = DataTransform::LogTransform({present}, base_, log1p_);
            if (!result.success) {
                const std::string advice = log1p_
                    ? " log(1 + x) needs values above -1; Power Transform (Yeo-Johnson) takes any value."
                    : " Turn on log(1 + x) for zeros, or use Power Transform (Yeo-Johnson) for negative values.";
                return arrow::Status::Invalid(GetName() + ": column '" + name + "': " + result.error_message + "." +
                                              advice);
            }
            transformed = result.transformed_data[0];
        }
        ARROW_ASSIGN_OR_RAISE(out, ReplaceKeepingMissing(out, idx, all.size(), rows, transformed));
    }
    return out;
}

arrow::Result<std::shared_ptr<arrow::Table>> LogTransformOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz LogTransform Materializer");
    if (!input) return arrow::Status::Invalid(GetName() + ": input is null");
    ARROW_RETURN_NOT_OK(BeginStream(input->schema()));
    ARROW_ASSIGN_OR_RAISE(auto preflight, EmitPreprocessingMemoryPreflight(
        input, GetName(), static_cast<uint64_t>(resolved_.size()), static_cast<uint64_t>(sizeof(double)),
        "reduce the selected rows or columns, or transform a sample first.", GetMaterializationMemoryContext(),
        progress_callback_));
    const uint64_t estimated_bytes = preflight.estimated_peak_bytes;
    const uint64_t columns = static_cast<uint64_t>(resolved_.size());
    ReportProgress(progress_callback_, "Log transform", "Log transform of " + std::to_string(columns) + " columns",
                   0.20, 0, columns, estimated_bytes);
    // The whole table is one batch of the stream.
    ARROW_ASSIGN_OR_RAISE(auto out, ApplyBatch(input));
    spdlog::info("[Preprocessing] LogTransform done: {} columns, base {}, {}", columns, base_,
                 log1p_ ? "log(1 + x)" : "log(x)");
    ReportProgress(progress_callback_, "Complete", "LogTransform complete", 1.0, columns, columns, estimated_bytes);
    return out;
}

// ============================================================================
// LabelEncoderOperator
// ============================================================================

bool LabelEncoderOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {
    column_.clear();

    auto c = params.find("column");
    if (c == params.end() || c->second.empty()) {
        error = GetName() + ": 'column' parameter is required";
        return false;
    }
    column_ = c->second;
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
LabelEncoderOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz LabelEncoder Materializer");

    if (!input) return arrow::Status::Invalid(GetName() + ": input is null");

    ReportProgress(progress_callback_, "Encoding column",
                   "Encoding label column '" + column_ + "'", 0.10,
                   0, 1);
    int idx = input->schema()->GetFieldIndex(column_);
    if (idx < 0) {
        return arrow::Status::KeyError(
            GetName() + ": column '" + column_ + "' not found");
    }
    auto col = input->column(idx);
    ARROW_ASSIGN_OR_RAISE(auto preflight_estimate,
        EmitPreprocessingMemoryPreflight(
            input, GetName(), 1ULL, static_cast<uint64_t>(sizeof(int)),
            "reduce selected rows, encode a sample first, or use a future chunked encoding path.",
            GetMaterializationMemoryContext(),
            progress_callback_));

    std::vector<int> codes;
    std::vector<std::string> categories;
    ARROW_RETURN_NOT_OK(EncodeStringColumn(col, column_, GetName(), codes, categories));

    ARROW_ASSIGN_OR_RAISE(auto out, ReplaceWithInts(input, idx, codes));

    spdlog::info("LabelEncoder: column '{}' -> int32 with {} categories",
                 column_, categories.size());
    ReportProgress(progress_callback_, "Complete",
                   "LabelEncoder materialization complete", 1.0, 1, 1,
                   preflight_estimate.estimated_peak_bytes);
    return out;
}

// ============================================================================
// OrdinalEncoderOperator
// ============================================================================

bool OrdinalEncoderOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {
    columns_.clear();

    auto c = params.find("columns");
    if (c == params.end() || c->second.empty()) {
        error = GetName() + ": 'columns' parameter is required (comma-sep list)";
        return false;
    }
    ParseCommaList(c->second, columns_);
    if (columns_.empty()) {
        error = GetName() + ": 'columns' parsed to empty list";
        return false;
    }
    auto cats = params.find("categories");
    if (cats != params.end() && !cats->second.empty() &&
        ToLowerAscii(TrimString(cats->second)) != "auto") {
        error = GetName() + ": only 'categories=auto' is supported in v1 "
                "(custom ordering deferred to tofix); got '" + cats->second + "'";
        return false;
    }
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
OrdinalEncoderOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz OrdinalEncoder Materializer");

    if (!input) return arrow::Status::Invalid(GetName() + ": input is null");

    ARROW_ASSIGN_OR_RAISE(auto preflight_estimate,
        EmitPreprocessingMemoryPreflight(
            input, GetName(), static_cast<uint64_t>(columns_.size()),
            static_cast<uint64_t>(sizeof(int)),
            "reduce selected rows or categorical columns, encode a sample first, or use a future chunked encoding path.",
            GetMaterializationMemoryContext(),
            progress_callback_));
    const uint64_t estimated_bytes = preflight_estimate.estimated_peak_bytes;

    auto out = input;
    size_t total_categories = 0;
    for (size_t column_index = 0; column_index < columns_.size(); ++column_index) {
        const auto& name = columns_[column_index];
        ReportProgress(progress_callback_, "Encoding columns",
                       "Ordinal encoding column '" + name + "'",
                       0.10 + (0.80 * static_cast<double>(column_index) /
                               static_cast<double>(columns_.size())),
                       static_cast<uint64_t>(column_index),
                       static_cast<uint64_t>(columns_.size()),
                       estimated_bytes);
        int idx = out->schema()->GetFieldIndex(name);
        if (idx < 0) {
            return arrow::Status::KeyError(
                GetName() + ": column '" + name + "' not found");
        }
        auto col = out->column(idx);
        std::vector<int> codes;
        std::vector<std::string> categories;
        ARROW_RETURN_NOT_OK(EncodeStringColumn(col, name, GetName(), codes, categories));
        ARROW_ASSIGN_OR_RAISE(out, ReplaceWithInts(out, idx, codes));
        total_categories += categories.size();
    }

    spdlog::info("OrdinalEncoder: encoded {} columns, {} total categories",
                 columns_.size(), total_categories);
    ReportProgress(progress_callback_, "Complete",
                   "OrdinalEncoder materialization complete", 1.0,
                   static_cast<uint64_t>(columns_.size()),
                   static_cast<uint64_t>(columns_.size()),
                   estimated_bytes);
    return out;
}

// ============================================================================
// TargetEncoderOperator
// ============================================================================

bool TargetEncoderOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {
    columns_.clear();
    target_col_.clear();
    smoothing_ = 1.0;

    auto c = params.find("columns");
    if (c == params.end() || c->second.empty()) {
        error = GetName() + ": 'columns' parameter is required (categorical columns)";
        return false;
    }
    ParseCommaList(c->second, columns_);
    if (columns_.empty()) {
        error = GetName() + ": 'columns' parsed to empty list";
        return false;
    }

    auto tc = params.find("target_col");
    if (tc == params.end() || tc->second.empty()) {
        error = GetName() + ": 'target_col' parameter is required";
        return false;
    }
    target_col_ = tc->second;

    auto s = params.find("smoothing");
    if (s != params.end() && !s->second.empty()) {
        try { smoothing_ = std::stod(s->second); }
        catch (...) {
            error = GetName() + ": 'smoothing' is not a valid float: " + s->second;
            return false;
        }
    }
    if (smoothing_ < 0.0) {
        error = GetName() + ": smoothing must be >= 0 (got " +
                std::to_string(smoothing_) + ")";
        return false;
    }
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
TargetEncoderOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz TargetEncoder Materializer");

    if (!input) return arrow::Status::Invalid(GetName() + ": input is null");

    ARROW_ASSIGN_OR_RAISE(auto preflight_estimate,
        EmitPreprocessingMemoryPreflight(
            input, GetName(), static_cast<uint64_t>(columns_.size()) + 1ULL,
            static_cast<uint64_t>(sizeof(double)),
            "reduce selected rows or categorical columns, encode a sample first, or use a future chunked target-encoding path.",
            GetMaterializationMemoryContext(),
            progress_callback_));
    const uint64_t estimated_bytes = preflight_estimate.estimated_peak_bytes;

    // Read target once.
    ReportProgress(progress_callback_, "Reading target",
                   "Reading target column '" + target_col_ +
                   "' for target encoding",
                   0.05, 0, 1, estimated_bytes);
    std::vector<double> target;
    int target_idx = -1;
    ARROW_RETURN_NOT_OK(ReadNumericDouble(input, target_col_, GetName(), target, target_idx));
    // The category means come from the Train rows when the pipeline run
    // marked them (TOFIX178 S0-g3): the validation and test labels never
    // reach the encoding. Every row is encoded.
    const std::vector<bool> fit_row = TrainRowMask(input);
    const auto fits = [&fit_row](size_t row) { return fit_row.empty() || (row < fit_row.size() && fit_row[row]); };
    double global_mean = 0.0;
    {
        double total = 0.0;
        size_t rows = 0;
        for (size_t i = 0; i < target.size(); ++i)
            if (fits(i)) total += target[i], ++rows;
        global_mean = rows ? total / static_cast<double>(rows) : 0.0;
    }

    auto out = input;
    int transformed = 0;
    for (const auto& name : columns_) {
        ReportProgress(progress_callback_, "Encoding columns",
                       "Target encoding column '" + name + "'",
                       0.15 + (0.75 * static_cast<double>(transformed) /
                               static_cast<double>(columns_.size())),
                       static_cast<uint64_t>(transformed),
                       static_cast<uint64_t>(columns_.size()),
                       estimated_bytes);
        int idx = out->schema()->GetFieldIndex(name);
        if (idx < 0) {
            return arrow::Status::KeyError(
                GetName() + ": column '" + name + "' not found");
        }
        auto col = out->column(idx);

        std::vector<int> codes;
        std::vector<std::string> categories;
        ARROW_RETURN_NOT_OK(EncodeStringColumn(col, name, GetName(), codes, categories));
        if (codes.size() != target.size()) {
            return arrow::Status::Invalid(
                GetName() + ": column '" + name + "' has " +
                std::to_string(codes.size()) + " rows, target has " +
                std::to_string(target.size()));
        }

        // Per-category sum + count.
        std::vector<double> sum(categories.size(), 0.0);
        std::vector<int> count(categories.size(), 0);
        for (size_t i = 0; i < codes.size(); ++i) {
            if (codes[i] < 0 || !fits(i)) continue;  // null category, or not a Train row
            sum[codes[i]] += target[i];
            count[codes[i]]++;
        }

        // Smoothed per-category mean.
        std::vector<double> cat_encoded(categories.size(), global_mean);
        for (size_t k = 0; k < categories.size(); ++k) {
            if (count[k] == 0) continue;
            const double cat_mean = sum[k] / count[k];
            cat_encoded[k] = (count[k] * cat_mean + smoothing_ * global_mean)
                              / (count[k] + smoothing_);
        }

        // Apply.
        std::vector<double> encoded(codes.size(), global_mean);
        for (size_t i = 0; i < codes.size(); ++i) {
            if (codes[i] >= 0) encoded[i] = cat_encoded[codes[i]];
        }
        ARROW_ASSIGN_OR_RAISE(out, ReplaceWithDoubles(out, idx, encoded));
        transformed++;
    }

    spdlog::info("TargetEncoder: encoded {} columns with target '{}' "
                 "(global_mean={:.4f}, smoothing={})",
                 transformed, target_col_, global_mean, smoothing_);
    ReportProgress(progress_callback_, "Complete",
                   "TargetEncoder materialization complete", 1.0,
                   static_cast<uint64_t>(transformed),
                   static_cast<uint64_t>(columns_.size()),
                   estimated_bytes);
    return out;
}

// ============================================================================
// OutlierDetectorOperator
// ============================================================================

bool OutlierDetectorOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {
    columns_.clear();
    label_col_.clear();
    method_ = "iqr";
    threshold_ = 1.5;

    // "columns" can be "all" or a csv list. "all" maps to empty =
    // auto-detect numeric (matches ResolveFeatureColumns semantic).
    auto c = params.find("columns");
    if (c != params.end() && !c->second.empty()) {
        const std::string columns_value = ToLowerAscii(TrimString(c->second));
        if (columns_value != "all") {
            ParseCommaList(c->second, columns_);
        }
    }

    auto lc = params.find("label_col");
    if (lc != params.end()) label_col_ = lc->second;

    auto m = params.find("method");
    if (m != params.end() && !m->second.empty()) {
        method_ = ToLowerAscii(TrimString(m->second));
        if (method_ != "iqr" && method_ != "zscore") {
            error = GetName() + ": only 'iqr' and 'zscore' methods supported "
                    "in v1 (isolation_forest / lof deferred to tofix); got '" +
                    method_ + "'";
            return false;
        }
    }

    auto t = params.find("threshold");
    if (t != params.end() && !t->second.empty()) {
        try { threshold_ = std::stod(t->second); }
        catch (...) {
            error = GetName() + ": 'threshold' is not a valid float: " + t->second;
            return false;
        }
    }
    if (threshold_ <= 0.0) {
        error = GetName() + ": threshold must be > 0 (got " +
                std::to_string(threshold_) + ")";
        return false;
    }
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
OutlierDetectorOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz OutlierDetector Materializer");

    if (!input) return arrow::Status::Invalid(GetName() + ": input is null");

    ReportProgress(progress_callback_, "Resolving columns",
                   "Resolving numeric columns for OutlierDetector", 0.05);
    std::vector<std::string> resolved;
    ARROW_RETURN_NOT_OK(ResolveNumericColumnList(
        input, columns_, label_col_, GetName(), resolved));
    ARROW_ASSIGN_OR_RAISE(auto preflight_estimate,
        EmitPreprocessingMemoryPreflight(
            input, GetName(), static_cast<uint64_t>(resolved.size()) + 1ULL,
            static_cast<uint64_t>(sizeof(double)),
            "reduce selected rows or feature columns, detect outliers on a sample first, or use a future chunked outlier path.",
            GetMaterializationMemoryContext(),
            progress_callback_));
    const uint64_t estimated_bytes = preflight_estimate.estimated_peak_bytes;

    const int64_t n = input->num_rows();
    std::vector<int> is_outlier(n, 0);

    // Per-column IQR or Z-score detection. A row is flagged if ANY
    // column marks it as an outlier.
    int total_flagged = 0;
    for (size_t column_index = 0; column_index < resolved.size(); ++column_index) {
        const auto& name = resolved[column_index];
        ReportProgress(progress_callback_, "Detecting outliers",
                       "Detecting outliers in column '" + name + "'",
                       0.15 + (0.70 * static_cast<double>(column_index) /
                               static_cast<double>(resolved.size())),
                       static_cast<uint64_t>(column_index),
                       static_cast<uint64_t>(resolved.size()),
                       estimated_bytes);
        std::vector<double> data;
        int idx = -1;
        ARROW_RETURN_NOT_OK(ReadNumericDouble(input, name, GetName(), data, idx));
        const OutlierFlags col_outliers = DataTransform::FindOutliers(data, method_, threshold_);
        if (!col_outliers.success) {
            return arrow::Status::Invalid(GetName() + ": column '" + name + "': " + col_outliers.error_message);
        }
        for (int64_t i = 0; i < n && i < static_cast<int64_t>(col_outliers.is_outlier.size()); ++i) {
            if (col_outliers.is_outlier[static_cast<size_t>(i)] && is_outlier[static_cast<size_t>(i)] == 0) {
                is_outlier[static_cast<size_t>(i)] = 1;
                total_flagged++;
            }
        }
    }

    // Append is_outlier column.
    ReportProgress(progress_callback_, "Appending outlier flag",
                   "Appending is_outlier column", 0.90,
                   static_cast<uint64_t>(resolved.size()),
                   static_cast<uint64_t>(resolved.size()),
                   estimated_bytes);
    arrow::Int32Builder builder;
    ARROW_RETURN_NOT_OK(builder.Reserve(n));
    for (int v : is_outlier) ARROW_RETURN_NOT_OK(builder.Append(v));
    std::shared_ptr<arrow::Array> arr;
    ARROW_RETURN_NOT_OK(builder.Finish(&arr));
    auto field = arrow::field("is_outlier", arrow::int32());
    auto chunked = std::make_shared<arrow::ChunkedArray>(arr);
    ARROW_ASSIGN_OR_RAISE(
        auto out,
        input->AddColumn(input->num_columns(), field, chunked));

    spdlog::info("OutlierDetector: {}/{} rows flagged (method={}, threshold={}, "
                 "cols checked={})",
                 total_flagged, n, method_, threshold_, resolved.size());
    ReportProgress(progress_callback_, "Complete",
                   "OutlierDetector materialization complete", 1.0,
                   static_cast<uint64_t>(n), static_cast<uint64_t>(n),
                   estimated_bytes);
    return out;
}

} // namespace cyxwiz
