#include "pca_operator.h"
#include "text_column_utils.h"  // ReadLabelColumnAsInt
#include "ts_column_utils.h"    // ReadColumnAsFloat

#include <cyxwiz/dimensionality_reduction.h>

#include <arrow/api.h>
#include <arrow/builder.h>
#include <spdlog/spdlog.h>

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

} // namespace

bool PCAOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {

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

    auto c = params.find("center");
    center_ = (c == params.end()) ? true : (c->second == "true");

    auto s = params.find("scale");
    scale_ = (s == params.end()) ? false : (s->second == "true");

    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
PCAOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    if (!input) {
        return arrow::Status::Invalid("PCA: input table is null");
    }

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

    // Read each feature column to a parallel float vector.
    const size_t n_features = resolved_features.size();
    std::vector<std::vector<float>> feat_cols(n_features);
    int64_t n_samples = 0;
    for (size_t f = 0; f < n_features; ++f) {
        auto col = input->GetColumnByName(resolved_features[f]);
        std::string bad_type;
        if (!ReadColumnAsFloat(col, feat_cols[f], bad_type)) {
            return arrow::Status::TypeError(
                "PCA: feature '" + resolved_features[f] +
                "' read failed (type='" + bad_type + "')");
        }
        if (f == 0) {
            n_samples = static_cast<int64_t>(feat_cols[f].size());
        } else if (static_cast<int64_t>(feat_cols[f].size()) != n_samples) {
            return arrow::Status::Invalid(
                "PCA: feature '" + resolved_features[f] +
                "' has different row count than first feature");
        }
    }

    if (n_samples < n_components_) {
        return arrow::Status::Invalid(
            "PCA: n_samples (" + std::to_string(n_samples) +
            ") < n_components (" + std::to_string(n_components_) + ")");
    }
    if (static_cast<int>(n_features) < n_components_) {
        return arrow::Status::Invalid(
            "PCA: n_features (" + std::to_string(n_features) +
            ") < n_components (" + std::to_string(n_components_) + ")");
    }

    // Repack as row-major double matrix [n_samples x n_features] for the
    // backend (which uses std::vector<std::vector<double>>).
    std::vector<std::vector<double>> data(n_samples,
                                           std::vector<double>(n_features, 0.0));
    for (size_t f = 0; f < n_features; ++f) {
        const auto& col = feat_cols[f];
        for (int64_t i = 0; i < n_samples; ++i) {
            data[i][f] = static_cast<double>(col[i]);
        }
    }

    auto result = DimensionalityReduction::ComputePCA(
        data, n_components_, center_, scale_);
    if (!result.success) {
        return arrow::Status::ExecutionError(
            "PCA: ComputePCA failed: " + result.error_message);
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
        if (static_cast<int64_t>(labels.size()) != n_samples) {
            return arrow::Status::Invalid(
                "PCA: label count (" + std::to_string(labels.size()) +
                ") differs from sample count (" + std::to_string(n_samples) + ")");
        }
    }

    // Build wide output: pc_0..pc_{n-1}, y.
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
            ARROW_RETURN_NOT_OK(label_builder.Append(labels[r]));
        }
    }

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
    return out_table;
}

} // namespace cyxwiz
