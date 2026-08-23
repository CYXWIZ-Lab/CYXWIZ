#pragma once

#include "ts_column_utils.h"

#include <arrow/api.h>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

// Auto-detect rule used by clustering / PCA / any multi-feature operator
// that treats the table as "all numeric columns are features".
// Mirrors ArrowDatasetBatcher's feature detection: skip label_col and
// any `__`-prefixed internal metadata column.
inline bool IsNumericChunked(const std::shared_ptr<arrow::ChunkedArray>& col) {
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

// Resolve the set of feature columns. If explicit_names is non-empty,
// validate each exists + is numeric. Otherwise auto-detect numeric
// columns, skipping label_col and __-prefixed metadata.
inline arrow::Status ResolveFeatureColumns(
    const std::shared_ptr<arrow::Table>& table,
    const std::vector<std::string>& explicit_names,
    const std::string& label_col,
    const std::string& op_name,
    std::vector<std::string>& out) {

    out.clear();
    if (!explicit_names.empty()) {
        for (const auto& name : explicit_names) {
            auto col = table->GetColumnByName(name);
            if (!col) {
                return arrow::Status::KeyError(
                    op_name + ": feature column '" + name + "' not found");
            }
            if (!IsNumericChunked(col)) {
                return arrow::Status::TypeError(
                    op_name + ": feature column '" + name + "' is not numeric");
            }
            out.push_back(name);
        }
        return arrow::Status::OK();
    }
    for (const auto& field : table->schema()->fields()) {
        const std::string& name = field->name();
        if (name == label_col) continue;
        if (name.rfind("__", 0) == 0) continue;
        auto col = table->GetColumnByName(name);
        if (!col || !IsNumericChunked(col)) continue;
        out.push_back(name);
    }
    if (out.empty()) {
        return arrow::Status::Invalid(
            op_name + ": no numeric feature columns found (auto-detect). "
            "Specify feature_cols explicitly.");
    }
    return arrow::Status::OK();
}

// Read resolved feature columns into a row-major double matrix
// suitable for Clustering::KMeans / DimensionalityReduction::ComputePCA
// / any backend expecting [n_samples][n_features] double.
inline arrow::Status ReadFeatureMatrix(
    const std::shared_ptr<arrow::Table>& table,
    const std::vector<std::string>& feature_names,
    const std::string& op_name,
    std::vector<std::vector<double>>& out_matrix,
    int64_t& out_n_samples,
    const std::function<bool()>& cancellation_requested = {}) {

    const size_t n_features = feature_names.size();
    std::vector<std::vector<float>> feat_cols(n_features);
    int64_t n_samples = 0;
    for (size_t f = 0; f < n_features; ++f) {
        auto col = table->GetColumnByName(feature_names[f]);
        std::string bad_type;
        bool cancelled = false;
        if (!ReadColumnAsFloat(
                col, feat_cols[f], bad_type, cancellation_requested,
                &cancelled)) {
            if (cancelled) {
                return arrow::Status::Cancelled(
                    op_name + ": materialization cancelled while reading features");
            }
            return arrow::Status::TypeError(
                op_name + ": feature '" + feature_names[f] +
                "' read failed (type='" + bad_type + "')");
        }
        if (f == 0) {
            n_samples = static_cast<int64_t>(feat_cols[f].size());
        } else if (static_cast<int64_t>(feat_cols[f].size()) != n_samples) {
            return arrow::Status::Invalid(
                op_name + ": feature '" + feature_names[f] +
                "' has different row count than first feature");
        }
    }

    out_matrix.assign(n_samples, std::vector<double>(n_features, 0.0));
    for (size_t f = 0; f < n_features; ++f) {
        const auto& col = feat_cols[f];
        for (int64_t i = 0; i < n_samples; ++i) {
            if ((i & 1023) == 0 && cancellation_requested &&
                cancellation_requested()) {
                return arrow::Status::Cancelled(
                    op_name + ": materialization cancelled while building features");
            }
            out_matrix[i][f] = static_cast<double>(col[i]);
        }
    }
    out_n_samples = n_samples;
    return arrow::Status::OK();
}

// Parse a comma-separated list of column names. Trims whitespace,
// skips empties. Used for `feature_cols` params across operators.
inline void ParseCommaList(const std::string& s, std::vector<std::string>& out) {
    out.clear();
    if (s.empty()) return;
    size_t pos = 0;
    while (pos <= s.size()) {
        size_t next = s.find(',', pos);
        if (next == std::string::npos) next = s.size();
        size_t start = s.find_first_not_of(" \t", pos);
        size_t end = s.find_last_not_of(" \t", next - 1);
        if (start != std::string::npos && start < next && end != std::string::npos && end >= start) {
            out.push_back(s.substr(start, end - start + 1));
        }
        pos = next + 1;
        if (next == s.size()) break;
    }
}

// Append a cluster_id int32 column to the input table and return the
// new table. Clustering operators share this — output semantic is
// "input + cluster annotation", not "replace features".
inline arrow::Result<std::shared_ptr<arrow::Table>> AppendClusterIdColumn(
    const std::shared_ptr<arrow::Table>& input,
    const std::vector<int>& cluster_ids,
    const std::string& column_name = "cluster_id") {

    const int64_t n = static_cast<int64_t>(cluster_ids.size());
    if (input->num_rows() != n) {
        return arrow::Status::Invalid(
            "AppendClusterIdColumn: cluster_ids.size() (" +
            std::to_string(n) + ") != input rows (" +
            std::to_string(input->num_rows()) + ")");
    }

    arrow::Int32Builder builder;
    ARROW_RETURN_NOT_OK(builder.Reserve(n));
    for (int id : cluster_ids) {
        ARROW_RETURN_NOT_OK(builder.Append(id));
    }
    std::shared_ptr<arrow::Array> arr;
    ARROW_RETURN_NOT_OK(builder.Finish(&arr));

    auto chunked = std::make_shared<arrow::ChunkedArray>(arr);
    auto field = arrow::field(column_name, arrow::int32());
    return input->AddColumn(input->num_columns(), field, chunked);
}

} // namespace cyxwiz
