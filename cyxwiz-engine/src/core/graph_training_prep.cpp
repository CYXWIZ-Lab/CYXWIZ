// Headless training preparation shared by every host (TOFIX118 P2): the
// label/target/feature-width reconciliation and the supplied-role and
// sequence-column checks that the Engine runs before a graph trains, moved
// verbatim from gui/graph_training_launcher.cpp. Datasets are read through
// the GraphDatasetCatalog, so the Engine (DataRegistry) and a Server Node
// (its job's files) run identical checks.
#include "graph_training_prep.h"

#include "arrow_dataset.h"
#include "graph_compiler_dataset_hooks.h"
#include "label_column_resolver.h"
#include "parquet_backed_dataset.h"
#include "sparse_feature_dataset.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <sstream>
#include <unordered_set>

namespace cyxwiz {

using gui::MLNode;
using gui::NodeType;

namespace {

void SetBlockedStatus(LaunchBlock& result, std::string title, std::string detail) {
    result.title = std::move(title);
    result.detail = std::move(detail);
    result.error_message = result.detail.empty() ? result.title : result.detail;
}

}  // namespace

std::string FindGraphDatasetName(const std::vector<MLNode>& nodes) {
    for (const auto& node : nodes) {
        if (node.type != NodeType::DataInput &&
            node.type != NodeType::DatasetInput) {
            continue;
        }
        auto it = node.parameters.find("dataset_name");
        if (it != node.parameters.end() && !it->second.empty()) {
            return it->second;
        }
        it = node.parameters.find("dataset");
        if (it != node.parameters.end() && !it->second.empty()) {
            return it->second;
        }
    }
    return {};
}

std::string FindGraphLabelColumn(
    const std::vector<MLNode>& nodes,
    const std::string& dataset_name,
    int data_source_node_id) {

    const MLNode* fallback_data_input = nullptr;
    for (const auto& node : nodes) {
        if (node.type != NodeType::DataInput) {
            continue;
        }

        if (node.id == data_source_node_id) {
            auto label_it = node.parameters.find("label_column");
            if (label_it != node.parameters.end() && !label_it->second.empty()) {
                return label_it->second;
            }
            return {};
        }

        if (!fallback_data_input) {
            fallback_data_input = &node;
        }

        auto dataset_it = node.parameters.find("dataset_name");
        if (dataset_it == node.parameters.end() || dataset_it->second.empty()) {
            dataset_it = node.parameters.find("dataset");
        }
        if (!dataset_name.empty() &&
            dataset_it != node.parameters.end() &&
            dataset_it->second == dataset_name) {
            auto label_it = node.parameters.find("label_column");
            if (label_it != node.parameters.end() && !label_it->second.empty()) {
                return label_it->second;
            }
            return {};
        }
    }

    if (fallback_data_input) {
        auto label_it = fallback_data_input->parameters.find("label_column");
        if (label_it != fallback_data_input->parameters.end() &&
            !label_it->second.empty()) {
            return label_it->second;
        }
    }
    return {};
}

std::string ResolveRuntimeArrowLabelColumn(
    const std::string& dataset_name,
    const std::string& requested_label) {
    if (auto sparse_ds = GraphSparseDataset(dataset_name)) {
        if (!requested_label.empty() &&
            requested_label == sparse_ds->GetLabelName()) {
            return requested_label;
        }
        return sparse_ds->GetLabels()
            ? sparse_ds->GetLabelName()
            : requested_label;
    }
    auto arrow_ds = GraphArrowDataset(dataset_name);
    auto parquet_ds = GraphParquetDataset(dataset_name);
    auto schema = arrow_ds ? arrow_ds->GetSchema()
        : parquet_ds ? parquet_ds->GetSchema()
        : nullptr;
    if (!schema) return requested_label;

    if (!requested_label.empty() &&
        schema->GetFieldByName(requested_label) != nullptr) {
        return requested_label;
    }

    const int fallback_idx = cyxwiz::FindCommonLabelColumnIndex(schema);
    if (fallback_idx < 0) {
        return requested_label;
    }

    const std::string resolved = schema->field(fallback_idx)->name();
    if (resolved != requested_label) {
        spdlog::info("StartTrainingFromGraph: resolved tabular label column "
                     "'{}' -> '{}' for materialized dataset '{}'",
                     requested_label.empty() ? "<auto>" : requested_label,
                     resolved,
                     dataset_name);
    }
    return resolved;
}

void ReconcileRuntimeDatasetTarget(
    cyxwiz::TrainingConfiguration& config,
    const std::string& resolved_label,
    const std::string& dataset_name) {
    if (resolved_label.empty() ||
        config.target.origin != cyxwiz::TargetOrigin::DatasetColumn ||
        config.target.primary_column == resolved_label) {
        return;
    }

    spdlog::info(
        "StartTrainingFromGraph: reconciled runtime target column '{}' -> "
        "'{}' for dataset '{}'",
        config.target.primary_column.empty()
            ? "<auto>"
            : config.target.primary_column,
        resolved_label,
        dataset_name);
    config.target.primary_column = resolved_label;
}

void ReconcileRuntimeTabularFeatureWidth(
    const std::string& dataset_name,
    const std::string& label_column,
    cyxwiz::TrainingConfiguration& config) {
    if (config.sequence_batch.enabled || config.is_time_series) return;

    if (auto sparse_ds = GraphSparseDataset(dataset_name)) {
        const size_t feature_count = static_cast<size_t>(
            sparse_ds->GetNumFeatures());
        if (feature_count > 0 && config.input_size != feature_count) {
            spdlog::warn(
                "StartTrainingFromGraph: corrected compiled sparse input "
                "width {} -> {} from CSR artifact '{}'",
                config.input_size, feature_count, dataset_name);
            config.input_shape = {feature_count};
            config.input_size = feature_count;
        }
        return;
    }

    std::shared_ptr<arrow::Schema> schema;
    if (auto arrow_ds = GraphArrowDataset(dataset_name)) {
        schema = arrow_ds->GetSchema();
    } else if (auto parquet_ds =
                   GraphParquetDataset(dataset_name)) {
        schema = parquet_ds->GetSchema();
    }
    const int label_index = cyxwiz::ResolveLabelColumnIndex(
        schema, label_column);
    if (label_index < 0) return;

    const size_t feature_count = cyxwiz::CountNumericBatchFeatureColumns(
        schema, label_index);
    if (feature_count == 0 || config.input_size == feature_count) return;

    spdlog::warn(
        "StartTrainingFromGraph: corrected compiled tabular input width "
        "{} -> {} from runtime schema for dataset '{}'",
        config.input_size, feature_count, dataset_name);
    config.input_shape = {feature_count};
    config.input_size = feature_count;
}

std::string DefaultSequenceColumn(const std::string& value,
                                  const char* fallback) {
    return value.empty() ? std::string(fallback) : value;
}

std::shared_ptr<arrow::Schema> FindGraphDatasetSchema(
    const std::string& dataset_name) {
    if (auto arrow_ds = GraphArrowDataset(dataset_name)) {
        return arrow_ds->GetSchema();
    }
    if (auto parquet_ds = GraphParquetDataset(dataset_name)) {
        return parquet_ds->GetSchema();
    }
    if (auto sparse_ds = GraphSparseDataset(dataset_name)) {
        std::vector<std::shared_ptr<arrow::Field>> fields;
        fields.reserve(static_cast<size_t>(sparse_ds->GetNumFeatures()) + 1);
        const auto& names = sparse_ds->GetFeatureNames();
        for (int64_t feature = 0;
             feature < sparse_ds->GetNumFeatures(); ++feature) {
            fields.push_back(arrow::field(
                names.empty()
                    ? "feature_" + std::to_string(feature)
                    : names[static_cast<size_t>(feature)],
                arrow::float32(), false));
        }
        if (const auto& labels = sparse_ds->GetLabels()) {
            fields.push_back(arrow::field(
                sparse_ds->GetLabelName(), labels->type()));
        }
        return arrow::schema(std::move(fields));
    }
    return nullptr;
}

bool IsRoleFeatureType(const std::shared_ptr<arrow::DataType>& type) {
    if (!type) return false;
    switch (type->id()) {
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

bool IsInternalRoleColumn(const std::string& name) {
    return name.rfind("__", 0) == 0;
}

std::shared_ptr<arrow::Field> ResolveRoleLabelField(
    const std::shared_ptr<arrow::Schema>& schema,
    const std::string& requested_label) {
    if (!schema) return nullptr;
    if (!requested_label.empty()) {
        return schema->GetFieldByName(requested_label);
    }
    const int fallback_idx = cyxwiz::FindCommonLabelColumnIndex(schema);
    return fallback_idx >= 0 ? schema->field(fallback_idx) : nullptr;
}

std::vector<std::shared_ptr<arrow::Field>> RoleFeatureFields(
    const std::shared_ptr<arrow::Schema>& schema,
    const std::string& label_name) {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    if (!schema) return fields;
    for (int i = 0; i < schema->num_fields(); ++i) {
        auto field = schema->field(i);
        if (!field) continue;
        const std::string& name = field->name();
        if (name == label_name || IsInternalRoleColumn(name)) {
            continue;
        }
        if (IsRoleFeatureType(field->type())) {
            fields.push_back(field);
        }
    }
    return fields;
}
std::string LowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool IsStableIdentifierColumnName(const std::string& name) {
    const std::string lower = LowerAscii(name);
    if (lower == "id" || lower == "uid" || lower == "uuid" ||
        lower == "row_id" || lower == "record_id" ||
        lower == "sample_id" || lower == "example_id" ||
        lower == "instance_id") {
        return true;
    }
    return lower.size() > 3 &&
           lower.compare(lower.size() - 3, 3, "_id") == 0;
}

std::string FindSharedStableIdentifierColumn(
    const std::shared_ptr<arrow::Schema>& train_schema,
    const std::shared_ptr<arrow::Schema>& role_schema,
    const std::string& train_label_name,
    const std::string& role_label_name) {
    if (!train_schema || !role_schema) return {};
    for (int i = 0; i < train_schema->num_fields(); ++i) {
        auto train_field = train_schema->field(i);
        if (!train_field) continue;
        const std::string& name = train_field->name();
        if (name == train_label_name || IsInternalRoleColumn(name) ||
            !IsStableIdentifierColumnName(name)) {
            continue;
        }
        auto role_field = role_schema->GetFieldByName(name);
        if (role_field && name != role_label_name &&
            train_field->type()->Equals(role_field->type())) {
            return name;
        }
    }
    return {};
}

bool AppendColumnValues(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& column_name,
    std::unordered_set<std::string>& values,
    int64_t& rows_scanned) {
    if (!table) return false;
    auto column = table->GetColumnByName(column_name);
    if (!column) return false;
    for (const auto& chunk : column->chunks()) {
        if (!chunk) continue;
        for (int64_t i = 0; i < chunk->length(); ++i) {
            auto scalar_result = chunk->GetScalar(i);
            if (!scalar_result.ok()) return false;
            const auto scalar = scalar_result.ValueOrDie();
            if (scalar && scalar->is_valid) {
                values.insert(scalar->ToString());
            }
            ++rows_scanned;
        }
    }
    return true;
}

bool AppendDatasetColumnValues(
    const std::string& dataset_name,
    const std::string& column_name,
    int64_t max_rows,
    std::unordered_set<std::string>& values,
    int64_t& rows_scanned,
    std::string& reason) {
    if (auto arrow_ds = GraphArrowDataset(dataset_name)) {
        if (arrow_ds->GetNumRows() > max_rows) {
            reason = "row count exceeds bounded identifier scan limit";
            return false;
        }
        return AppendColumnValues(
            arrow_ds->GetArrowTable(), column_name, values, rows_scanned);
    }
    if (auto parquet_ds = GraphParquetDataset(dataset_name)) {
        if (parquet_ds->GetNumRows() > max_rows) {
            reason = "row count exceeds bounded identifier scan limit";
            return false;
        }
        for (int i = 0; i < parquet_ds->GetNumRowGroups(); ++i) {
            if (!AppendColumnValues(
                    parquet_ds->ReadRowGroup(i), column_name, values,
                    rows_scanned)) {
                reason = "could not read Parquet row group for identifier scan";
                return false;
            }
        }
        return true;
    }
    reason = "dataset is not a registered Arrow/Parquet table";
    return false;
}

std::string BuildRowSignature(
    const std::shared_ptr<arrow::Table>& table,
    int64_t row,
    const std::vector<std::string>& columns) {
    std::ostringstream out;
    for (const auto& column_name : columns) {
        auto column = table->GetColumnByName(column_name);
        if (!column) return {};
        int64_t offset = row;
        for (const auto& chunk : column->chunks()) {
            if (!chunk) continue;
            if (offset >= chunk->length()) {
                offset -= chunk->length();
                continue;
            }
            auto scalar_result = chunk->GetScalar(offset);
            if (!scalar_result.ok()) return {};
            const auto scalar = scalar_result.ValueOrDie();
            out << ((scalar && scalar->is_valid) ? scalar->ToString() : "<null>")
                << '\x1f';
            break;
        }
    }
    return out.str();
}

bool AppendRowSignatures(
    const std::shared_ptr<arrow::Table>& table,
    const std::vector<std::string>& columns,
    std::unordered_set<std::string>& values,
    std::string& example) {
    if (!table) return false;
    for (int64_t row = 0; row < table->num_rows(); ++row) {
        const std::string signature = BuildRowSignature(table, row, columns);
        if (signature.empty()) return false;
        if (example.empty()) example = signature;
        values.insert(signature);
    }
    return true;
}

bool AppendDatasetRowSignatures(
    const std::string& dataset_name,
    const std::vector<std::string>& columns,
    int64_t max_rows,
    std::unordered_set<std::string>& values,
    std::string& example,
    std::string& reason) {
    if (auto arrow_ds = GraphArrowDataset(dataset_name)) {
        if (arrow_ds->GetNumRows() > max_rows) {
            reason = "row count exceeds bounded exact-row scan limit";
            return false;
        }
        return AppendRowSignatures(
            arrow_ds->GetArrowTable(), columns, values, example);
    }
    if (auto parquet_ds = GraphParquetDataset(dataset_name)) {
        if (parquet_ds->GetNumRows() > max_rows) {
            reason = "row count exceeds bounded exact-row scan limit";
            return false;
        }
        for (int i = 0; i < parquet_ds->GetNumRowGroups(); ++i) {
            if (!AppendRowSignatures(
                    parquet_ds->ReadRowGroup(i), columns, values, example)) {
                reason = "could not read Parquet row group for exact-row scan";
                return false;
            }
        }
        return true;
    }
    reason = "dataset is not a registered Arrow/Parquet table";
    return false;
}

std::vector<std::string> ExactRowSignatureColumns(
    const std::vector<std::shared_ptr<arrow::Field>>& feature_fields,
    const std::string& label_name) {
    std::vector<std::string> columns;
    columns.reserve(feature_fields.size() + 1);
    for (const auto& field : feature_fields) {
        if (field) columns.push_back(field->name());
    }
    if (!label_name.empty()) columns.push_back(label_name);
    return columns;
}

bool ValidateSuppliedRoleLeakage(
    const cyxwiz::ResolvedDatasetRole& train_role,
    const cyxwiz::ResolvedDatasetRole& role,
    const char* role_name,
    LaunchBlock& launch_result,
    cyxwiz::PartitionLeakageStatus& status,
    std::string& status_reason) {
    status_reason.clear();
    if (!role.IsSupplied()) {
        status = cyxwiz::PartitionLeakageStatus::Passed;
        status_reason = "derived from Training Dataset by partition policy";
        return true;
    }

    constexpr int64_t kMaxIdentifierScanRows = 1000000;
    constexpr int64_t kMaxExactRowScanRows = 20000;

    auto train_schema = FindGraphDatasetSchema(train_role.dataset_name);
    auto role_schema = FindGraphDatasetSchema(role.dataset_name);
    auto train_label = ResolveRoleLabelField(train_schema, train_role.label_column);
    auto role_label = ResolveRoleLabelField(role_schema, role.label_column);
    if (!train_schema || !role_schema || !train_label || !role_label) {
        status = cyxwiz::PartitionLeakageStatus::Unavailable;
        status_reason = "schema or label metadata was unavailable";
        return true;
    }

    const std::string id_column = FindSharedStableIdentifierColumn(
        train_schema, role_schema, train_label->name(), role_label->name());
    if (!id_column.empty()) {
        std::unordered_set<std::string> train_ids;
        std::string reason;
        int64_t ignored_rows_scanned = 0;
        if (!AppendDatasetColumnValues(train_role.dataset_name,
                                       id_column, kMaxIdentifierScanRows,
                                       train_ids, ignored_rows_scanned, reason)) {
            spdlog::warn(
                "Track70: skipped {} overlap check against Training dataset '{}' using identifier '{}': {}",
                role_name, train_role.dataset_name, id_column, reason);
            status = cyxwiz::PartitionLeakageStatus::Unavailable;
            status_reason = reason;
            return true;
        }

        std::unordered_set<std::string> role_ids;
        reason.clear();
        if (!AppendDatasetColumnValues(role.dataset_name,
                                       id_column, kMaxIdentifierScanRows,
                                       role_ids, ignored_rows_scanned, reason)) {
            spdlog::warn(
                "Track70: skipped {} overlap check for dataset '{}' using identifier '{}': {}",
                role_name, role.dataset_name, id_column, reason);
            status = cyxwiz::PartitionLeakageStatus::Unavailable;
            status_reason = reason;
            return true;
        }

        for (const auto& id : role_ids) {
            if (train_ids.count(id) > 0) {
                SetBlockedStatus(
                    launch_result,
                    std::string(role_name) + " dataset leakage detected",
                    std::string(role_name) + " Dataset '" + role.dataset_name +
                        "' overlaps Training Dataset '" +
                        train_role.dataset_name + "' on identifier column '" +
                        id_column + "' with value " + id +
                        ". External " + role_name +
                        " data must not duplicate Training rows.");
                status = cyxwiz::PartitionLeakageStatus::Failed;
                status_reason = "overlap detected on identifier column '" +
                    id_column + "'";
                return false;
            }
        }
        status = cyxwiz::PartitionLeakageStatus::Passed;
        status_reason = "no overlap found on identifier column '" +
            id_column + "'";
        return true;
    }

    auto train_arrow = GraphArrowDataset(train_role.dataset_name);
    auto role_arrow = GraphArrowDataset(role.dataset_name);
    auto train_parquet = GraphParquetDataset(train_role.dataset_name);
    auto role_parquet = GraphParquetDataset(role.dataset_name);
    const int64_t train_rows = train_arrow ? train_arrow->GetNumRows()
        : train_parquet ? train_parquet->GetNumRows()
        : 0;
    const int64_t role_rows = role_arrow ? role_arrow->GetNumRows()
        : role_parquet ? role_parquet->GetNumRows()
        : 0;
    if (train_rows + role_rows > kMaxExactRowScanRows) {
        spdlog::warn(
            "Track70: {} overlap check for dataset '{}' could not find a shared stable identifier and skipped exact-row comparison for {} combined rows",
            role_name, role.dataset_name, train_rows + role_rows);
        status = cyxwiz::PartitionLeakageStatus::Unavailable;
        status_reason = "no shared stable identifier and exact-row scan limit exceeded";
        return true;
    }

    const auto train_features =
        RoleFeatureFields(train_schema, train_label->name());
    const std::vector<std::string> train_columns =
        ExactRowSignatureColumns(train_features, train_label->name());
    const auto role_features =
        RoleFeatureFields(role_schema, role_label->name());
    std::vector<std::string> role_columns;
    role_columns.reserve(role_features.size() + 1);
    for (const auto& field : role_features) {
        if (field) role_columns.push_back(field->name());
    }
    if (!role_label->name().empty()) role_columns.push_back(role_label->name());

    std::unordered_set<std::string> train_rows_seen;
    std::string reason;
    std::string ignored_example;
    if (!AppendDatasetRowSignatures(train_role.dataset_name,
                                    train_columns, kMaxExactRowScanRows,
                                    train_rows_seen, ignored_example, reason)) {
        spdlog::warn(
            "Track70: skipped exact-row {} overlap check for Training dataset '{}': {}",
            role_name, train_role.dataset_name, reason);
        status = cyxwiz::PartitionLeakageStatus::Unavailable;
        status_reason = reason;
        return true;
    }

    std::unordered_set<std::string> role_rows_seen;
    reason.clear();
    if (!AppendDatasetRowSignatures(role.dataset_name,
                                    role_columns, kMaxExactRowScanRows,
                                    role_rows_seen, ignored_example, reason)) {
        spdlog::warn(
            "Track70: skipped exact-row {} overlap check for dataset '{}': {}",
            role_name, role.dataset_name, reason);
        status = cyxwiz::PartitionLeakageStatus::Unavailable;
        status_reason = reason;
        return true;
    }

    for (const auto& signature : role_rows_seen) {
        if (train_rows_seen.count(signature) > 0) {
            SetBlockedStatus(
                launch_result,
                std::string(role_name) + " dataset leakage detected",
                std::string(role_name) + " Dataset '" + role.dataset_name +
                    "' contains an exact row also present in Training Dataset '" +
                    train_role.dataset_name +
                    "'. External " + role_name +
                    " data must not duplicate Training rows.");
            status = cyxwiz::PartitionLeakageStatus::Failed;
            status_reason = "an exact duplicate row overlaps Training Dataset";
            return false;
        }
    }

    status = cyxwiz::PartitionLeakageStatus::Passed;
    status_reason = "bounded exact-row comparison found no overlap";
    return true;
}

bool ValidateSuppliedRoleSchema(
    const cyxwiz::ResolvedDatasetRole& train_role,
    const cyxwiz::ResolvedDatasetRole& role,
    const char* role_name,
    LaunchBlock& launch_result) {
    if (!role.IsSupplied()) return true;

    auto train_schema = FindGraphDatasetSchema(train_role.dataset_name);
    if (!train_schema) {
        SetBlockedStatus(
            launch_result,
            "Training dataset schema unavailable",
            "Could not inspect schema for Training dataset '" +
                train_role.dataset_name +
                "' while validating supplied " + std::string(role_name) +
                " dataset '" + role.dataset_name + "'.");
        return false;
    }

    auto role_schema = FindGraphDatasetSchema(role.dataset_name);
    if (!role_schema) {
        SetBlockedStatus(
            launch_result,
            std::string(role_name) + " dataset schema unavailable",
            "Could not inspect schema for supplied " + std::string(role_name) +
                " dataset '" + role.dataset_name + "'. Apply its Data Input "
                "node as a tabular Arrow/Parquet source before training.");
        return false;
    }

    auto train_label = ResolveRoleLabelField(train_schema, train_role.label_column);
    if (!train_label) {
        SetBlockedStatus(
            launch_result,
            "Training dataset label unavailable",
            "Training dataset '" + train_role.dataset_name +
                "' does not contain label column '" +
                (train_role.label_column.empty() ? "<auto>" : train_role.label_column) +
                "'.");
        return false;
    }

    auto role_label = ResolveRoleLabelField(role_schema, role.label_column);
    if (!role_label) {
        SetBlockedStatus(
            launch_result,
            std::string(role_name) + " dataset label unavailable",
            "Supplied " + std::string(role_name) + " dataset '" +
                role.dataset_name + "' does not contain label column '" +
                (role.label_column.empty() ? "<auto>" : role.label_column) +
                "'.");
        return false;
    }

    if (!train_label->type()->Equals(role_label->type())) {
        SetBlockedStatus(
            launch_result,
            std::string(role_name) + " dataset label mismatch",
            std::string(role_name) + " Dataset label column '" +
                role_label->name() + "' in dataset '" + role.dataset_name +
                "' has type " + role_label->type()->ToString() +
                ", but Training Dataset label column '" + train_label->name() +
                "' has type " + train_label->type()->ToString() + ".");
        return false;
    }

    const auto train_features =
        RoleFeatureFields(train_schema, train_label->name());
    const auto role_features =
        RoleFeatureFields(role_schema, role_label->name());
    if (train_features.size() != role_features.size()) {
        SetBlockedStatus(
            launch_result,
            std::string(role_name) + " dataset schema mismatch",
            std::string(role_name) + " Dataset '" + role.dataset_name +
                "' has " + std::to_string(role_features.size()) +
                " numeric feature columns after excluding label/internal "
                "columns, but Training Dataset '" + train_role.dataset_name +
                "' has " + std::to_string(train_features.size()) + ".");
        return false;
    }

    for (size_t i = 0; i < train_features.size(); ++i) {
        const auto& train_field = train_features[i];
        const auto& role_field = role_features[i];
        if (train_field->name() != role_field->name()) {
            SetBlockedStatus(
                launch_result,
                std::string(role_name) + " dataset schema mismatch",
                std::string(role_name) + " Dataset schema mismatch at feature " +
                    std::to_string(i) + ": expected Training feature '" +
                    train_field->name() + "' from dataset '" +
                    train_role.dataset_name + "', found '" +
                    role_field->name() + "' in dataset '" + role.dataset_name +
                    "'.");
            return false;
        }
        if (!train_field->type()->Equals(role_field->type())) {
            SetBlockedStatus(
                launch_result,
                std::string(role_name) + " dataset schema mismatch",
                std::string(role_name) + " Dataset feature '" +
                    role_field->name() + "' in dataset '" + role.dataset_name +
                    "' has type " + role_field->type()->ToString() +
                    ", but Training Dataset feature '" +
                    train_field->name() + "' has type " +
                    train_field->type()->ToString() + ".");
            return false;
        }
    }

    return true;
}

bool ValidateSequenceLaunchColumns(
    const std::string& dataset_name,
    const cyxwiz::TrainingConfiguration& config, std::string& error_message);

bool ValidateSuppliedRolePreflight(
    cyxwiz::ResolvedDatasetRoles& roles,
    const cyxwiz::TrainingConfiguration& config,
    LaunchBlock& launch_result) {
    auto& manifest = roles.manifest;
    if (config.sequence_batch.enabled) {
        // Sequence inputs own token/tag/POS contracts, not numeric feature/label
        // columns. Keep this preflight allocation-light; the existing sequence
        // batcher validates typed IDs, frozen metadata and role overlap before
        // the training executor can perform any updates.
        if (!roles.dev.IsSupplied() && !roles.test.IsSupplied()) return true;
        for (const auto* role : {&roles.train, &roles.dev, &roles.test}) {
            if (role != &roles.train && !role->IsSupplied()) continue;
            std::string error;
            if (!ValidateSequenceLaunchColumns(role->dataset_name, config, error)) {
                SetBlockedStatus(launch_result, "Sequence dataset columns unavailable", error);
                return false;
            }
        }
        // Do not report tabular label/row checks as sequence compatibility proof.
        manifest.dev_compatibility = cyxwiz::PartitionCompatibility::Unknown;
        manifest.test_compatibility = cyxwiz::PartitionCompatibility::Unknown;
        manifest.dev_leakage = cyxwiz::PartitionLeakageStatus::NotChecked;
        manifest.test_leakage = cyxwiz::PartitionLeakageStatus::NotChecked;
        manifest.dev_status_reason = manifest.test_status_reason =
            "Sequence schema columns checked; full role validation is owned by sequence batch construction";
        return true;
    }
    if (!ValidateSuppliedRoleSchema(
            roles.train, roles.dev, "Dev", launch_result)) {
        manifest.dev_compatibility =
            cyxwiz::PartitionCompatibility::Incompatible;
        manifest.dev_status_reason = launch_result.error_message;
        return false;
    }
    manifest.dev_compatibility = cyxwiz::PartitionCompatibility::Compatible;
    if (!ValidateSuppliedRoleSchema(
            roles.train, roles.test, "Test", launch_result)) {
        manifest.test_compatibility =
            cyxwiz::PartitionCompatibility::Incompatible;
        manifest.test_status_reason = launch_result.error_message;
        return false;
    }
    manifest.test_compatibility = cyxwiz::PartitionCompatibility::Compatible;
    if (!ValidateSuppliedRoleLeakage(
            roles.train, roles.dev, "Dev", launch_result,
            manifest.dev_leakage, manifest.dev_status_reason)) {
        return false;
    }
    return ValidateSuppliedRoleLeakage(
        roles.train, roles.test, "Test", launch_result,
        manifest.test_leakage, manifest.test_status_reason);
}

bool ValidateSequenceLaunchColumns(
    const std::string& dataset_name,
    const cyxwiz::TrainingConfiguration& config,
    std::string& error_message) {
    if (!config.sequence_batch.enabled) {
        return true;
    }

    auto schema = FindGraphDatasetSchema(dataset_name);
    if (!schema) {
        error_message =
            "Sequence training could not inspect tabular schema for dataset '" +
            dataset_name + "'.";
        return false;
    }

    struct RequiredColumn {
        std::string role;
        std::string name;
    };

    std::vector<RequiredColumn> required = {
        {"token", DefaultSequenceColumn(config.sequence_batch.token_column,
                                         "tokens")},
    };
    if (!config.sequence_batch.create_causal_lm_targets) {
        required.push_back({"tag", DefaultSequenceColumn(
            config.sequence_batch.tag_column, "ner_tags")});
    }
    if (!config.sequence_batch.pos_column.empty()) {
        required.push_back({"POS", config.sequence_batch.pos_column});
    }
    if (config.has_data_split &&
        !config.sequence_batch.sentence_id_column.empty()) {
        required.push_back({"sentence id",
                            config.sequence_batch.sentence_id_column});
    }

    std::vector<std::string> missing;
    for (const auto& column : required) {
        if (schema->GetFieldIndex(column.name) < 0) {
            missing.push_back(column.role + " column '" + column.name + "'");
        }
    }

    if (missing.empty()) {
        return true;
    }

    error_message = "Sequence training is missing ";
    for (size_t i = 0; i < missing.size(); ++i) {
        if (i > 0) {
            error_message += (i + 1 == missing.size()) ? " and " : ", ";
        }
        error_message += missing[i];
    }
    error_message += " in dataset '" + dataset_name + "'.";
    return false;
}


}  // namespace cyxwiz
