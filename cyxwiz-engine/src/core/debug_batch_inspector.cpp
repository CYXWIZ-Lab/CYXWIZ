#include "debug_batch_inspector.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <sstream>

namespace cyxwiz {

namespace {

const char* TensorDataTypeName(DataType dtype) {
    switch (dtype) {
        case DataType::Float32: return "float32";
        case DataType::Float64: return "float64";
        case DataType::Int32: return "int32";
        case DataType::Int64: return "int64";
        case DataType::UInt8: return "uint8";
    }
    return "unknown";
}

bool IsClassificationLoss(gui::NodeType loss_type) {
    return loss_type == gui::NodeType::CrossEntropyLoss ||
           loss_type == gui::NodeType::FocalLoss ||
           loss_type == gui::NodeType::BCELoss ||
           loss_type == gui::NodeType::BCEWithLogits ||
           loss_type == gui::NodeType::NLLLoss;
}

nlohmann::json ColumnPreviewJson(
    const std::vector<BatchColumnInspection>& columns,
    const std::string& tensor_dtype) {
    nlohmann::json preview = nlohmann::json::array();
    for (const auto& column : columns) {
        preview.push_back({
            {"name", column.name},
            {"source_dtype", column.source_dtype},
            {"tensor_dtype", tensor_dtype},
        });
    }
    return preview;
}

nlohmann::json DTypeConversionSummary(
    const BatchInspectionMetadata& inspection,
    const std::string& feature_tensor_dtype,
    const std::string& label_tensor_dtype) {
    nlohmann::json summary = nlohmann::json::array();
    const auto append = [&](const char* role,
                            const std::vector<BatchColumnInspection>& columns,
                            const std::string& tensor_dtype) {
        std::set<std::string> source_types;
        for (const auto& column : columns) {
            source_types.insert(column.source_dtype);
        }
        if (source_types.empty()) {
            return;
        }
        std::ostringstream source;
        bool first = true;
        for (const auto& type : source_types) {
            if (!first) source << ", ";
            source << type;
            first = false;
        }
        summary.push_back({
            {"role", role},
            {"source_dtypes", source.str()},
            {"tensor_dtype", tensor_dtype},
        });
    };
    append("features", inspection.feature_columns_preview,
           feature_tensor_dtype);
    append("labels", inspection.label_columns_preview,
           label_tensor_dtype);
    return summary;
}

void AttachPaddingSummary(nlohmann::json& payload,
                          const Batch& batch,
                          int pad_value) {
    payload["padding_summary_available"] = false;
    if (!batch.inspection.token_sequence_columns ||
        batch.data.GetDataType() != DataType::Float32 ||
        batch.data.Shape().size() != 2 ||
        batch.data.Shape()[0] != batch.size ||
        batch.size > kDebugBatchInspectorMaxRows) {
        payload["padding_summary_reason"] =
            "The bounded batch is not a recognized float32 token-slot matrix.";
        return;
    }

    Tensor non_padding =
        (batch.data - static_cast<float>(pad_value)).Sign().Abs();
    Tensor row_lengths = non_padding.Sum(1);
    if (row_lengths.GetDataType() != DataType::Float32 ||
        row_lengths.NumElements() != batch.size) {
        payload["padding_summary_reason"] =
            "The Tensor row reduction did not produce one length per batch row.";
        return;
    }

    const float* values = row_lengths.ReadData<float>();
    nlohmann::json preview = nlohmann::json::array();
    size_t non_padding_count = 0;
    size_t min_length = batch.data.Shape()[1];
    size_t max_length = 0;
    for (size_t row = 0; row < batch.size; ++row) {
        const size_t length = static_cast<size_t>(
            std::max(0.0f, std::round(values[row])));
        preview.push_back(length);
        non_padding_count += length;
        min_length = std::min(min_length, length);
        max_length = std::max(max_length, length);
    }
    const size_t token_slots = batch.data.NumElements();
    const size_t pad_count = token_slots >= non_padding_count
        ? token_slots - non_padding_count
        : 0;
    payload["padding_summary_available"] = true;
    payload["pad_value"] = pad_value;
    payload["sequence_lengths"] = std::move(preview);
    payload["sequence_length_min"] = min_length;
    payload["sequence_length_max"] = max_length;
    payload["sequence_length_mean"] = batch.size > 0
        ? static_cast<double>(non_padding_count) /
              static_cast<double>(batch.size)
        : 0.0;
    payload["pad_count"] = pad_count;
    payload["pad_ratio"] = token_slots > 0
        ? static_cast<double>(pad_count) /
              static_cast<double>(token_slots)
        : 0.0;
    payload["padding_summary_scalar_read_count"] = batch.size;
}

void AttachClassBalance(nlohmann::json& payload,
                        const Batch& batch,
                        gui::NodeType loss_type) {
    payload["class_balance_available"] = false;
    if (!IsClassificationLoss(loss_type)) {
        payload["class_balance_reason"] =
            "The configured loss is not a classification loss.";
        return;
    }
    if (batch.labels.GetDataType() != DataType::Float32 ||
        batch.size == 0 || batch.size > kDebugBatchInspectorMaxRows) {
        payload["class_balance_reason"] =
            "Labels are not a bounded float32 debug target tensor.";
        return;
    }

    nlohmann::json counts = nlohmann::json::array();
    const auto& shape = batch.labels.Shape();
    if (shape.size() == 2 && shape[0] == batch.size && shape[1] > 1) {
        if (shape[1] > kDebugBatchInspectorMaxClasses) {
            payload["class_balance_reason"] =
                "The class width exceeds the bounded debug class limit.";
            return;
        }
        Tensor reduced = batch.labels.Sum(0);
        if (reduced.GetDataType() != DataType::Float32 ||
            reduced.NumElements() != shape[1]) {
            payload["class_balance_reason"] =
                "The Tensor class reduction returned an unexpected shape.";
            return;
        }
        const float* values = reduced.ReadData<float>();
        for (size_t class_index = 0; class_index < shape[1]; ++class_index) {
            counts.push_back({
                {"class_index", class_index},
                {"count", static_cast<size_t>(
                    std::max(0.0f, std::round(values[class_index])))},
            });
        }
        payload["class_balance_scalar_read_count"] = shape[1];
    } else if ((shape.size() == 1 && shape[0] == batch.size) ||
               (shape.size() == 2 && shape[0] == batch.size &&
                shape[1] == 1)) {
        if (batch.labels.NumElements() > kDebugBatchInspectorMaxRows) {
            payload["class_balance_reason"] =
                "The scalar label tensor exceeds the bounded debug row limit.";
            return;
        }
        const float* values = batch.labels.ReadData<float>();
        std::map<int64_t, size_t> histogram;
        for (size_t row = 0; row < batch.labels.NumElements(); ++row) {
            if (!std::isfinite(values[row])) {
                payload["class_balance_reason"] =
                    "The bounded label tensor contains a non-finite value.";
                return;
            }
            const int64_t class_index = static_cast<int64_t>(
                std::llround(values[row]));
            if (std::abs(values[row] - static_cast<float>(class_index)) >
                1.0e-4f) {
                payload["class_balance_reason"] =
                    "The bounded labels are not integer class ids.";
                return;
            }
            ++histogram[class_index];
        }
        if (histogram.size() > kDebugBatchInspectorMaxClasses) {
            payload["class_balance_reason"] =
                "The observed class count exceeds the bounded debug class limit.";
            return;
        }
        for (const auto& [class_index, count] : histogram) {
            counts.push_back({
                {"class_index", class_index},
                {"count", count},
            });
        }
        payload["class_balance_scalar_read_count"] =
            batch.labels.NumElements();
    } else {
        payload["class_balance_reason"] =
            "The label tensor shape is not scalar ids or one-hot classes.";
        return;
    }

    payload["class_balance_available"] = true;
    payload["class_balance_scope"] = "first_smoke_batch";
    payload["class_counts"] = std::move(counts);
}

} // namespace

void AttachDebugBatchInspection(
    DebugTraceRecord& trace,
    const Batch& batch,
    const TrainingConfiguration& config,
    const std::string& dataset_name,
    const std::string& batcher_source,
    size_t requested_batch_size) {
    auto& payload = trace.payload;
    payload["batch_inspector"] = true;
    payload["batch_inspection_scope"] = "first_smoke_batch";
    payload["bounded_batch"] = batch.size <= kDebugBatchInspectorMaxRows;
    payload["bounded_row_limit"] = kDebugBatchInspectorMaxRows;
    payload["requested_batch_size"] = requested_batch_size;
    payload["actual_row_count"] = batch.size;
    payload["dataset_name"] = dataset_name;
    payload["batcher_source"] = batcher_source;
    payload["feature_tensor_shape"] = batch.data.Shape();
    payload["label_tensor_shape"] = batch.labels.Shape();
    const std::string feature_tensor_dtype =
        TensorDataTypeName(batch.data.GetDataType());
    const std::string label_tensor_dtype =
        TensorDataTypeName(batch.labels.GetDataType());
    payload["feature_tensor_dtype"] = feature_tensor_dtype;
    payload["label_tensor_dtype"] = label_tensor_dtype;
    payload["feature_values_captured"] = false;
    payload["label_values_captured"] = false;

    payload["source_metadata_available"] = batch.inspection.available;
    payload["feature_column_count"] =
        batch.inspection.feature_column_count;
    payload["label_column_count"] =
        batch.inspection.label_column_count;
    payload["feature_columns_preview"] = ColumnPreviewJson(
        batch.inspection.feature_columns_preview, feature_tensor_dtype);
    payload["label_columns_preview"] = ColumnPreviewJson(
        batch.inspection.label_columns_preview, label_tensor_dtype);
    payload["feature_columns_truncated"] =
        batch.inspection.feature_columns_truncated;
    payload["label_columns_truncated"] =
        batch.inspection.label_columns_truncated;
    payload["dtype_conversions"] = DTypeConversionSummary(
        batch.inspection, feature_tensor_dtype, label_tensor_dtype);

    payload["null_summary_available"] =
        batch.inspection.null_summary_available;
    payload["null_summary_scope"] = "selected_source_cells_for_batch";
    payload["inspected_source_value_count"] =
        batch.inspection.inspected_value_count;
    payload["feature_null_count"] =
        batch.inspection.feature_null_count;
    payload["label_null_count"] = batch.inspection.label_null_count;
    payload["post_conversion_non_finite_summary_available"] = false;
    payload["post_conversion_non_finite_summary_reason"] =
        "Feature tensor values are not materialized for the batch inspector.";

    AttachPaddingSummary(
        payload, batch, config.text_preprocessing.pad_value);
    AttachClassBalance(payload, batch, config.loss_type);
}

} // namespace cyxwiz
