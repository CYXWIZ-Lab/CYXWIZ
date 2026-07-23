#include "preprocessing_operators.h"
#include "../materialization_memory_guard.h"
#include "../preprocessing_state.h"
#include "../profiler_trace.h"
#include "feature_matrix_utils.h"
#include "text_column_utils.h"
#include "ts_column_utils.h"

#include <cyxwiz/stats_utils.h>

#include <arrow/api.h>
#include <arrow/builder.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdint>
#include <cctype>
#include <cmath>
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
    const std::string& op_name,
    std::vector<std::string>& out) {
    return ResolveFeatureColumns(table, explicit_names, label_col,
                                  op_name, out);
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

const char* MaterializationMemoryProgressStatus(
    MaterializationMemoryRisk risk) {
    switch (risk) {
    case MaterializationMemoryRisk::Safe:
        return "running";
    case MaterializationMemoryRisk::Warning:
        return "warning";
    case MaterializationMemoryRisk::Risky:
        return "risky";
    case MaterializationMemoryRisk::Blocked:
        return "blocked";
    }
    return "running";
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
        estimate, DetectMaterializationMemorySnapshot());
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
        event.status = MaterializationMemoryProgressStatus(decision.risk);
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
// StandardScalerOperator
// ============================================================================

bool StandardScalerOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {
    ParseColumnsAndLabel(params, columns_, label_col_);
    if (!ReadBoolParam(params, "with_mean", true, GetName(),
                       with_mean_, error)) return false;
    if (!ReadBoolParam(params, "with_std", true, GetName(),
                       with_std_, error)) return false;
    const auto mode = params.find("operation_mode");
    operation_mode_ =
        mode == params.end() || mode->second.empty()
            ? "fit_transform"
            : ToLowerAscii(TrimString(mode->second));
    if (operation_mode_ != "fit_transform" &&
        operation_mode_ != "transform_only") {
        error = GetName() +
                ": 'operation_mode' must be 'fit_transform' or "
                "'transform_only' (got '" + operation_mode_ + "')";
        return false;
    }
    const auto state_path = params.find("state_path");
    state_path_ =
        state_path == params.end() ? "" : TrimString(state_path->second);
    if (!ReadBoolParam(params, "save_state", false, GetName(),
                       save_state_, error)) return false;
    if (!ReadBoolParam(params, "state_overwrite", false, GetName(),
                       state_overwrite_, error)) return false;
    if (operation_mode_ == "transform_only" && state_path_.empty()) {
        error = GetName() +
                ": Transform Only requires 'state_path'. Fit this node on "
                "training data with 'Save fitted state' enabled, then select "
                "that artifact.";
        return false;
    }
    if (operation_mode_ == "fit_transform" && save_state_ &&
        state_path_.empty()) {
        error = GetName() +
                ": 'Save fitted state' is enabled but 'state_path' is empty. "
                "Choose a .cyxstate.json path or disable saving.";
        return false;
    }
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
StandardScalerOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz StandardScaler Materializer");

    if (!input) return arrow::Status::Invalid(GetName() + ": input is null");

    ReportProgress(
        progress_callback_, "Resolving columns",
        "Resolving numeric columns for StandardScaler (" +
            operation_mode_ + ")",
        0.05);
    std::vector<std::string> resolved;
    FittedPreprocessingState state;
    if (operation_mode_ == "transform_only") {
        std::string state_error;
        if (!LoadFittedPreprocessingState(
                state_path_, GetName(), state, state_error)) {
            return arrow::Status::Invalid(GetName() + ": " + state_error);
        }
        if (!ValidateFittedPreprocessingStateSchema(
                state, input->schema(), state_error)) {
            return arrow::Status::Invalid(state_error);
        }
        const auto mean_config = state.configuration.find("with_mean");
        const auto std_config = state.configuration.find("with_std");
        if ((mean_config != state.configuration.end() &&
             mean_config->second != (with_mean_ ? "true" : "false")) ||
            (std_config != state.configuration.end() &&
             std_config->second != (with_std_ ? "true" : "false"))) {
            return arrow::Status::Invalid(
                GetName() +
                ": node options do not match the fitted artifact. Set "
                "'Center data' and 'Scale to unit variance' to the values "
                "used during fitting, or choose the correct artifact.");
        }
        for (const auto& feature : state.features) {
            resolved.push_back(feature.name);
        }
        if (!columns_.empty() && columns_ != resolved) {
            return arrow::Status::Invalid(
                GetName() +
                ": configured columns do not match the fitted artifact. "
                "Keep the training column order or leave Columns empty.");
        }
        spdlog::info(
            "[Preprocessing] StandardScaler Transform Only loaded '{}' "
            "(fit_rows={}, features={}, schema={})",
            state_path_, state.fit_rows, state.features.size(),
            state.input_schema_fingerprint);
    } else {
        ARROW_RETURN_NOT_OK(ResolveNumericColumnList(
            input, columns_, label_col_, GetName(), resolved));
        state.operator_name = GetName();
        state.fit_rows = input->num_rows();
        state.input_schema_fingerprint =
            FingerprintPreprocessingSchema(input->schema());
        state.configuration["with_mean"] =
            with_mean_ ? "true" : "false";
        state.configuration["with_std"] =
            with_std_ ? "true" : "false";
        state.configuration["label_col"] = label_col_;
        spdlog::info(
            "[Preprocessing] StandardScaler Fit + Transform starting "
            "(rows={}, features={}, save_state={}, state_path='{}')",
            input->num_rows(), resolved.size(), save_state_, state_path_);
    }
    ARROW_ASSIGN_OR_RAISE(auto preflight_estimate,
        EmitPreprocessingMemoryPreflight(
            input, GetName(), static_cast<uint64_t>(resolved.size()),
            static_cast<uint64_t>(sizeof(double)),
            "reduce selected rows or columns, scale a sample first, or use a future chunked preprocessing path.",
            progress_callback_));
    const uint64_t estimated_bytes = preflight_estimate.estimated_peak_bytes;
    ReportProgress(progress_callback_, "Columns resolved",
                   "Resolved " + std::to_string(resolved.size()) +
                   " columns for StandardScaler",
                   0.15, 0, static_cast<uint64_t>(resolved.size()),
                   estimated_bytes);

    auto out = input;
    int transformed = 0;
    for (size_t feature_index = 0;
         feature_index < resolved.size(); ++feature_index) {
        const auto& name = resolved[feature_index];
        ReportProgress(progress_callback_, "Scaling columns",
                       "Standard scaling column '" + name + "'",
                       0.20 + (0.70 * static_cast<double>(transformed) /
                               static_cast<double>(resolved.size())),
                       static_cast<uint64_t>(transformed),
                       static_cast<uint64_t>(resolved.size()),
                       estimated_bytes);
        std::vector<double> data;
        int idx = -1;
        ARROW_RETURN_NOT_OK(ReadNumericDouble(out, name, GetName(), data, idx));

        double mean = 0.0;
        double denom = 1.0;
        if (operation_mode_ == "transform_only") {
            const auto& feature = state.features[feature_index];
            const auto mean_it = feature.numeric_values.find("mean");
            const auto scale_it = feature.numeric_values.find("scale");
            if (mean_it == feature.numeric_values.end() ||
                scale_it == feature.numeric_values.end()) {
                return arrow::Status::Invalid(
                    GetName() + ": state for column '" + name +
                    "' is missing mean/scale values. Refit the artifact.");
            }
            mean = mean_it->second;
            denom = scale_it->second;
            if (!std::isfinite(mean) || !std::isfinite(denom) ||
                denom <= 0.0) {
                return arrow::Status::Invalid(
                    GetName() + ": state for column '" + name +
                    "' contains invalid mean/scale values. Refit the "
                    "artifact.");
            }
        } else {
            mean = with_mean_ ? stats::Mean(data) : 0.0;
            const double sd = with_std_ ? stats::StdDev(data) : 1.0;
            denom = (sd == 0.0) ? 1.0 : sd;
            if (!std::isfinite(mean) || !std::isfinite(denom)) {
                return arrow::Status::Invalid(
                    GetName() + ": fitted non-finite statistics for column '" +
                    name + "'. Fill missing/non-finite values before "
                    "scaling.");
            }
            PreprocessingFeatureState feature;
            feature.name = name;
            feature.data_type =
                input->schema()
                    ->field(input->schema()->GetFieldIndex(name))
                    ->type()
                    ->ToString();
            feature.numeric_values["mean"] = mean;
            feature.numeric_values["scale"] = denom;
            state.features.push_back(std::move(feature));
        }

        for (auto& v : data) v = (v - mean) / denom;
        ARROW_ASSIGN_OR_RAISE(out, ReplaceWithDoubles(out, idx, data));
        transformed++;
    }

    if (operation_mode_ == "fit_transform" && save_state_) {
        std::string state_error;
        if (!SaveFittedPreprocessingState(
                state_path_, state, state_overwrite_, state_error)) {
            return arrow::Status::IOError(
                GetName() + ": failed to save fitted state: " + state_error);
        }
        spdlog::info(
            "[Preprocessing] StandardScaler saved fitted state '{}' "
            "(fit_rows={}, features={}, schema={})",
            state_path_, state.fit_rows, state.features.size(),
            state.input_schema_fingerprint);
    }

    spdlog::info(
        "[Preprocessing] StandardScaler completed mode={} columns={} "
        "(with_mean={}, with_std={})",
        operation_mode_, transformed, with_mean_, with_std_);
    ReportProgress(progress_callback_, "Complete",
                   "StandardScaler materialization complete", 1.0,
                   static_cast<uint64_t>(transformed),
                   static_cast<uint64_t>(resolved.size()), estimated_bytes);
    return out;
}

// ============================================================================
// MinMaxScalerOperator
// ============================================================================

bool MinMaxScalerOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {
    range_min_ = 0.0;
    range_max_ = 1.0;

    ParseColumnsAndLabel(params, columns_, label_col_);

    auto mn = params.find("min");
    if (mn != params.end() && !mn->second.empty()) {
        try { range_min_ = std::stod(mn->second); }
        catch (...) {
            error = GetName() + ": 'min' is not a valid float: " + mn->second;
            return false;
        }
    }
    auto mx = params.find("max");
    if (mx != params.end() && !mx->second.empty()) {
        try { range_max_ = std::stod(mx->second); }
        catch (...) {
            error = GetName() + ": 'max' is not a valid float: " + mx->second;
            return false;
        }
    }
    if (range_max_ <= range_min_) {
        error = GetName() + ": max (" + std::to_string(range_max_) +
                ") must be > min (" + std::to_string(range_min_) + ")";
        return false;
    }
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
MinMaxScalerOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz MinMaxScaler Materializer");

    if (!input) return arrow::Status::Invalid(GetName() + ": input is null");

    ReportProgress(progress_callback_, "Resolving columns",
                   "Resolving numeric columns for MinMaxScaler", 0.05);
    std::vector<std::string> resolved;
    ARROW_RETURN_NOT_OK(ResolveNumericColumnList(
        input, columns_, label_col_, GetName(), resolved));
    ARROW_ASSIGN_OR_RAISE(auto preflight_estimate,
        EmitPreprocessingMemoryPreflight(
            input, GetName(), static_cast<uint64_t>(resolved.size()),
            static_cast<uint64_t>(sizeof(double)),
            "reduce selected rows or columns, scale a sample first, or use a future chunked preprocessing path.",
            progress_callback_));
    const uint64_t estimated_bytes = preflight_estimate.estimated_peak_bytes;
    ReportProgress(progress_callback_, "Columns resolved",
                   "Resolved " + std::to_string(resolved.size()) +
                   " columns for MinMaxScaler",
                   0.15, 0, static_cast<uint64_t>(resolved.size()),
                   estimated_bytes);

    const double target_span = range_max_ - range_min_;
    auto out = input;
    int transformed = 0;
    for (const auto& name : resolved) {
        ReportProgress(progress_callback_, "Scaling columns",
                       "Min-max scaling column '" + name + "'",
                       0.20 + (0.70 * static_cast<double>(transformed) /
                               static_cast<double>(resolved.size())),
                       static_cast<uint64_t>(transformed),
                       static_cast<uint64_t>(resolved.size()),
                       estimated_bytes);
        std::vector<double> data;
        int idx = -1;
        ARROW_RETURN_NOT_OK(ReadNumericDouble(out, name, GetName(), data, idx));

        const double dmin = stats::Min(data);
        const double dmax = stats::Max(data);
        const double span = dmax - dmin;
        const double denom = (span == 0.0) ? 1.0 : span;

        for (auto& v : data) {
            v = (v - dmin) / denom * target_span + range_min_;
        }
        ARROW_ASSIGN_OR_RAISE(out, ReplaceWithDoubles(out, idx, data));
        transformed++;
    }

    spdlog::info("MinMaxScaler: transformed {} columns to [{}, {}]",
                 transformed, range_min_, range_max_);
    ReportProgress(progress_callback_, "Complete",
                   "MinMaxScaler materialization complete", 1.0,
                   static_cast<uint64_t>(transformed),
                   static_cast<uint64_t>(resolved.size()), estimated_bytes);
    return out;
}

// ============================================================================
// RobustScalerOperator
// ============================================================================

bool RobustScalerOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {
    quantile_min_ = 25.0;
    quantile_max_ = 75.0;

    ParseColumnsAndLabel(params, columns_, label_col_);

    if (!ReadBoolParam(params, "with_centering", true, GetName(),
                       with_centering_, error)) return false;
    if (!ReadBoolParam(params, "with_scaling", true, GetName(),
                       with_scaling_, error)) return false;

    auto qmin = params.find("quantile_min");
    if (qmin != params.end() && !qmin->second.empty()) {
        try { quantile_min_ = std::stod(qmin->second); }
        catch (...) {
            error = GetName() + ": 'quantile_min' is not a valid float: " + qmin->second;
            return false;
        }
    }
    auto qmax = params.find("quantile_max");
    if (qmax != params.end() && !qmax->second.empty()) {
        try { quantile_max_ = std::stod(qmax->second); }
        catch (...) {
            error = GetName() + ": 'quantile_max' is not a valid float: " + qmax->second;
            return false;
        }
    }
    if (quantile_min_ < 0.0 || quantile_min_ > 100.0 ||
        quantile_max_ < 0.0 || quantile_max_ > 100.0 ||
        quantile_max_ <= quantile_min_) {
        error = GetName() + ": quantiles must satisfy 0 <= qmin < qmax <= 100 "
                "(got qmin=" + std::to_string(quantile_min_) +
                ", qmax=" + std::to_string(quantile_max_) + ")";
        return false;
    }
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
RobustScalerOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz RobustScaler Materializer");

    if (!input) return arrow::Status::Invalid(GetName() + ": input is null");

    ReportProgress(progress_callback_, "Resolving columns",
                   "Resolving numeric columns for RobustScaler", 0.05);
    std::vector<std::string> resolved;
    ARROW_RETURN_NOT_OK(ResolveNumericColumnList(
        input, columns_, label_col_, GetName(), resolved));
    ARROW_ASSIGN_OR_RAISE(auto preflight_estimate,
        EmitPreprocessingMemoryPreflight(
            input, GetName(), static_cast<uint64_t>(resolved.size()),
            static_cast<uint64_t>(sizeof(double)),
            "reduce selected rows or columns, scale a sample first, or use a future chunked preprocessing path.",
            progress_callback_));
    const uint64_t estimated_bytes = preflight_estimate.estimated_peak_bytes;
    ReportProgress(progress_callback_, "Columns resolved",
                   "Resolved " + std::to_string(resolved.size()) +
                   " columns for RobustScaler",
                   0.15, 0, static_cast<uint64_t>(resolved.size()),
                   estimated_bytes);

    auto out = input;
    int transformed = 0;
    for (const auto& name : resolved) {
        ReportProgress(progress_callback_, "Scaling columns",
                       "Robust scaling column '" + name + "'",
                       0.20 + (0.70 * static_cast<double>(transformed) /
                               static_cast<double>(resolved.size())),
                       static_cast<uint64_t>(transformed),
                       static_cast<uint64_t>(resolved.size()),
                       estimated_bytes);
        std::vector<double> data;
        int idx = -1;
        ARROW_RETURN_NOT_OK(ReadNumericDouble(out, name, GetName(), data, idx));

        const double center = with_centering_ ? stats::Median(data) : 0.0;
        const double q_low = stats::Percentile(data, quantile_min_ / 100.0);
        const double q_high = stats::Percentile(data, quantile_max_ / 100.0);
        const double iqr = q_high - q_low;
        const double scale = (with_scaling_ && iqr != 0.0) ? iqr : 1.0;

        for (auto& v : data) v = (v - center) / scale;
        ARROW_ASSIGN_OR_RAISE(out, ReplaceWithDoubles(out, idx, data));
        transformed++;
    }

    spdlog::info("RobustScaler: transformed {} columns (center={}, scale={}, "
                 "q={}/{})",
                 transformed, with_centering_, with_scaling_,
                 quantile_min_, quantile_max_);
    ReportProgress(progress_callback_, "Complete",
                   "RobustScaler materialization complete", 1.0,
                   static_cast<uint64_t>(transformed),
                   static_cast<uint64_t>(resolved.size()), estimated_bytes);
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
    const double global_mean = stats::Mean(target);

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
            if (codes[i] < 0) continue;  // null category
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
        std::vector<size_t> col_outliers;
        if (method_ == "iqr") {
            col_outliers = stats::DetectOutliersIQR(data, threshold_);
        } else {
            col_outliers = stats::DetectOutliersZScore(data, threshold_);
        }
        for (size_t i : col_outliers) {
            if (static_cast<int64_t>(i) < n && is_outlier[i] == 0) {
                is_outlier[i] = 1;
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
