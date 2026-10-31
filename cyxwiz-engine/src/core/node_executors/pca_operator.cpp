#include "pca_operator.h"
#include "../materialization_memory_guard.h"
#include "../profiler_trace.h"
#include "text_column_utils.h"  // ReadLabelColumnAsInt
#include "feature_rows.h"

#include <cyxwiz/dimensionality_reduction.h>

#include <arrow/api.h>
#include <arrow/builder.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace cyxwiz {

namespace {

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

// True iff the column type is one we can read as float.
bool IsNumericArrowType(const std::shared_ptr<arrow::ChunkedArray>& col) {
    if (!col || col->num_chunks() == 0) return false;
    switch (col->chunk(0)->type_id()) {
        case arrow::Type::DOUBLE:
        case arrow::Type::FLOAT:
        case arrow::Type::INT64:
        case arrow::Type::INT32:
        case arrow::Type::INT16:
        case arrow::Type::INT8:
        case arrow::Type::UINT64:
        case arrow::Type::UINT32:
        case arrow::Type::UINT16:
        case arrow::Type::UINT8:
            return true;
        default:
            return false;
    }
}

bool ReadBoolParam(const std::map<std::string, std::string>& params,
                   const char* key,
                   bool default_value,
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
    error = std::string("PCA: '") + key +
            "' must be 'true' or 'false' (got '" + it->second + "')";
    return false;
}

std::string BuildPcaMemoryPreflightMessage(
    const MaterializationMemoryEstimate& estimate,
    const MaterializationMemoryDecision& decision,
    uint64_t source_features,
    uint64_t output_components) {
    std::ostringstream ss;
    ss << "PCA memory preflight: risk="
       << MaterializationMemoryRiskName(decision.risk)
       << ", rows=" << estimate.rows
       << ", source_features=" << source_features
       << ", output_components=" << output_components
       << ", raw=" << FormatMaterializationBytes(estimate.raw_output_bytes)
       << ", estimated_peak="
       << FormatMaterializationBytes(estimate.estimated_peak_bytes)
       << ", available="
       << FormatMaterializationBytes(decision.available_bytes)
       << ", safe_budget="
       << FormatMaterializationBytes(decision.safe_budget_bytes)
       << ". " << decision.reason
       << ". Suggestion: reduce feature_cols, source rows, n_components, "
          "or use a future chunked PCA/materialization path.";
    return ss.str();
}

} // namespace

bool PCAOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {

    feature_cols_.clear();
    label_col_.clear();
    n_components_ = 2;
    center_ = true;
    scale_ = false;
    keep_columns_ = false;

    auto fc = params.find("feature_cols");
    const std::string fc_str = (fc != params.end()) ? fc->second : "";
    ParseColList(fc_str, feature_cols_);

    auto lc = params.find("label_col");
    if (lc != params.end()) label_col_ = lc->second;

    auto p = params.find("n_components");
    if (p != params.end() && !p->second.empty()) {
        try { n_components_ = std::stoi(p->second); }
        catch (...) {
            error = std::string("PCA: 'n_components' is not a valid integer: ") + p->second;
            return false;
        }
    }
    if (n_components_ < 1) {
        error = "PCA: n_components must be >= 1 (got " +
                std::to_string(n_components_) + ")";
        return false;
    }

    if (!ReadBoolParam(params, "center", true, center_, error)) return false;
    if (!ReadBoolParam(params, "scale", false, scale_, error)) return false;
    if (!ReadBoolParam(params, "keep_columns", false, keep_columns_, error)) return false;

    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
PCAOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz PCA Materializer");

    if (!input) {
        return arrow::Status::Invalid("PCA: input table is null");
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
        event.status = "running";
        event.progress = static_cast<float>(progress);
        event.processed_items = processed;
        event.total_items = total;
        event.estimated_memory_bytes = memory;
        progress_callback_(event);
    };

    report_progress("Resolving features",
                    "Resolving numeric feature columns for PCA",
                    0.05);

    // Resolve which columns to use as features. Auto-detect = all
    // numeric columns except label_col and any internal `__*` metadata
    // columns (matches ArrowDatasetBatcher's feature detection).
    std::vector<std::string> resolved_features;
    if (!feature_cols_.empty()) {
        for (const auto& name : feature_cols_) {
            auto col = input->GetColumnByName(name);
            if (!col) {
                return arrow::Status::KeyError(
                    "PCA: feature column '" + name + "' not found");
            }
            if (!IsNumericArrowType(col)) {
                return arrow::Status::TypeError(
                    "PCA: feature column '" + name + "' is not numeric");
            }
            resolved_features.push_back(name);
        }
    } else {
        for (const auto& field : input->schema()->fields()) {
            const std::string& name = field->name();
            if (name == label_col_) continue;
            if (name.rfind("__", 0) == 0) continue;
            auto col = input->GetColumnByName(name);
            if (!col || !IsNumericArrowType(col)) continue;
            resolved_features.push_back(name);
        }
        if (resolved_features.empty()) {
            return arrow::Status::Invalid(
                "PCA: no numeric feature columns found (auto-detect). "
                "Specify feature_cols explicitly.");
        }
    }
    report_progress("Features resolved",
                    "Resolved " + std::to_string(resolved_features.size()) +
                    " numeric PCA feature columns",
                    0.15,
                    static_cast<uint64_t>(resolved_features.size()),
                    static_cast<uint64_t>(resolved_features.size()));

    const uint64_t planned_rows =
        static_cast<uint64_t>(std::max<int64_t>(0, input->num_rows()));
    const uint64_t planned_features =
        static_cast<uint64_t>(resolved_features.size());
    const uint64_t planned_components =
        static_cast<uint64_t>(std::max(1, n_components_));
    const uint64_t planned_columns =
        planned_features + planned_components + (label_col_.empty() ? 0ULL : 1ULL);
    const auto preflight_estimate = EstimateDenseMaterializationMemory(
        planned_rows, planned_columns, static_cast<uint64_t>(sizeof(double)));
    const auto preflight_decision = EvaluateMaterializationMemory(
        preflight_estimate, GetMaterializationMemoryContext());
    const std::string preflight_message = BuildPcaMemoryPreflightMessage(
        preflight_estimate, preflight_decision,
        planned_features, planned_components);
    uint64_t planned_cells = 0;
    if (!CheckedMulU64(planned_rows, planned_columns, planned_cells)) {
        planned_cells = std::numeric_limits<uint64_t>::max();
    }
    if (progress_callback_) {
        PipelineOperatorProgress event;
        event.stage = "PCA memory preflight";
        event.message = preflight_message;
        event.status = MaterializationMemoryRiskToProgressStatus(
            preflight_decision.risk);
        event.progress = 0.17f;
        event.estimated_memory_bytes = preflight_estimate.estimated_peak_bytes;
        event.memory_risk_level = MaterializationMemoryRiskName(
            preflight_decision.risk);
        event.processed_items = 0;
        event.total_items = planned_cells;
        progress_callback_(event);
    }
    if (preflight_decision.blocked) {
        return arrow::Status::CapacityError(
            "Materialization blocked: " + preflight_message);
    }

    // The rows with every feature, as doubles (rows missing one are left out).
    report_progress("Reading feature columns", "Reading the PCA feature columns", 0.20, 0,
                    static_cast<uint64_t>(resolved_features.size()), preflight_estimate.estimated_peak_bytes);
    const size_t n_features = resolved_features.size();
    FeatureRows fit;
    ARROW_RETURN_NOT_OK(ReadFeatureRows(input, resolved_features, "PCA", false, fit, GetCancellationQuery()));
    const int64_t n_samples = static_cast<int64_t>(fit.rows.size());
    const auto& data = fit.data;
    const uint64_t estimated_input_matrix_bytes = preflight_estimate.estimated_peak_bytes;
    const uint64_t estimated_output_matrix_bytes =
        EstimateDenseMaterializationMemory(static_cast<uint64_t>(n_samples),
                                           static_cast<uint64_t>(n_components_ + (label_col_.empty() ? 0 : 1)),
                                           static_cast<uint64_t>(sizeof(float)))
            .estimated_peak_bytes;
    if (n_samples < n_components_) {
        return arrow::Status::Invalid(
            "PCA: rows with every feature (" + std::to_string(n_samples) +
            ") < n_components (" + std::to_string(n_components_) + ")");
    }
    if (static_cast<int>(n_features) < n_components_) {
        return arrow::Status::Invalid(
            "PCA: n_features (" + std::to_string(n_features) +
            ") < n_components (" + std::to_string(n_components_) + ")");
    }

    report_progress("Computing PCA",
                    "Computing PCA projection",
                    0.65,
                    0,
                    static_cast<uint64_t>(n_components_),
                    estimated_input_matrix_bytes + estimated_output_matrix_bytes);
    auto result = DimensionalityReduction::ComputePCA(
        data, n_components_, center_, scale_);
    if (!result.success) {
        return arrow::Status::ExecutionError(
            "PCA: ComputePCA failed: " + result.error_message);
    }
    report_progress("PCA computed",
                    "PCA projection computed",
                    0.75,
                    static_cast<uint64_t>(n_components_),
                    static_cast<uint64_t>(n_components_),
                    estimated_input_matrix_bytes + estimated_output_matrix_bytes);

    // Keep the input: pc_* beside its columns, null on the rows left out.
    if (keep_columns_) {
        std::vector<std::string> names;
        for (int i = 0; i < n_components_; ++i) names.push_back("pc_" + std::to_string(i));
        spdlog::info("PCA: {} of {} rows x {} features -> {} components beside the input, explained variance {:.2f}%",
                     n_samples, fit.total, n_features, n_components_, result.total_variance_explained * 100.0);
        report_progress("Complete", "PCA materialization complete", 1.0, static_cast<uint64_t>(n_samples),
                        static_cast<uint64_t>(n_samples), estimated_output_matrix_bytes);
        return AppendFittedColumns(input, names, fit.rows, result.transformed);
    }

    // Optional label column passthrough.
    std::vector<int> labels;
    std::vector<std::string> class_names;
    if (!label_col_.empty()) {
        auto label_column = input->GetColumnByName(label_col_);
        if (!label_column) {
            return arrow::Status::KeyError(
                "PCA: label column '" + label_col_ + "' not found");
        }
        std::string lbad;
        if (!ReadLabelColumnAsInt(label_column, labels, class_names, lbad)) {
            return arrow::Status::TypeError(
                "PCA: label column '" + label_col_ +
                "' has unsupported type '" + lbad + "'");
        }
        if (static_cast<int64_t>(labels.size()) != fit.total) {
            return arrow::Status::Invalid(
                "PCA: label count (" + std::to_string(labels.size()) +
                ") differs from row count (" + std::to_string(fit.total) + ")");
        }
    }

    // Build wide output: pc_0..pc_{n-1}, y.
    report_progress("Building Arrow columns",
                    "Allocating PCA output columns",
                    0.80,
                    0,
                    static_cast<uint64_t>(n_samples),
                    estimated_output_matrix_bytes);
    arrow::MemoryPool* pool = arrow::default_memory_pool();
    std::vector<std::unique_ptr<arrow::FloatBuilder>> pc_builders;
    pc_builders.reserve(n_components_);
    for (int i = 0; i < n_components_; ++i) {
        pc_builders.push_back(std::make_unique<arrow::FloatBuilder>(pool));
        ARROW_RETURN_NOT_OK(pc_builders.back()->Reserve(n_samples));
    }
    arrow::Int32Builder label_builder(pool);
    if (!labels.empty()) {
        ARROW_RETURN_NOT_OK(label_builder.Reserve(n_samples));
    }

    for (int64_t r = 0; r < n_samples; ++r) {
        const auto& row = result.transformed[r];
        for (int c = 0; c < n_components_; ++c) {
            const float v = (c < static_cast<int>(row.size()))
                ? static_cast<float>(row[c]) : 0.0f;
            ARROW_RETURN_NOT_OK(pc_builders[c]->Append(v));
        }
        if (!labels.empty()) {
            ARROW_RETURN_NOT_OK(label_builder.Append(labels[static_cast<size_t>(fit.rows[static_cast<size_t>(r)])]));
        }
        if ((r + 1) == n_samples || ((r + 1) % 1024) == 0) {
            report_progress("Writing PCA rows",
                            "Writing PCA rows to Arrow columns",
                            0.82 + (0.13 * static_cast<double>(r + 1) /
                                    static_cast<double>(n_samples)),
                            static_cast<uint64_t>(r + 1),
                            static_cast<uint64_t>(n_samples),
                            estimated_output_matrix_bytes);
        }
    }

    report_progress("Finishing Arrow table",
                    "Finalizing PCA Arrow table",
                    0.97,
                    static_cast<uint64_t>(n_samples),
                    static_cast<uint64_t>(n_samples),
                    estimated_output_matrix_bytes);
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    std::vector<std::shared_ptr<arrow::Field>> fields;
    arrays.reserve(n_components_ + (labels.empty() ? 0 : 1));
    fields.reserve(n_components_ + (labels.empty() ? 0 : 1));
    for (int i = 0; i < n_components_; ++i) {
        std::shared_ptr<arrow::Array> arr;
        ARROW_RETURN_NOT_OK(pc_builders[i]->Finish(&arr));
        arrays.push_back(std::move(arr));
        fields.push_back(arrow::field("pc_" + std::to_string(i), arrow::float32()));
    }
    if (!labels.empty()) {
        std::shared_ptr<arrow::Array> arr;
        ARROW_RETURN_NOT_OK(label_builder.Finish(&arr));
        arrays.push_back(std::move(arr));
        fields.push_back(arrow::field("y", arrow::int32()));
    }

    auto out_schema = arrow::schema(fields);
    auto out_table = arrow::Table::Make(out_schema, arrays, n_samples);

    spdlog::info("PCA: {} samples x {} features -> {} components, "
                 "explained variance {:.2f}%, center={}, scale={}",
                 n_samples, n_features, n_components_,
                 result.total_variance_explained * 100.0,
                 center_, scale_);
    report_progress("Complete",
                    "PCA materialization complete",
                    1.0,
                    static_cast<uint64_t>(n_samples),
                    static_cast<uint64_t>(n_samples),
                    estimated_output_matrix_bytes);
    return out_table;
}

} // namespace cyxwiz
