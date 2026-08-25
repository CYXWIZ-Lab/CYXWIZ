#include "graph_training_launcher.h"

#include "../core/arrow_dataset.h"
#include "../core/async_task_manager.h"
#include "../core/data_registry.h"
#include "../core/label_column_resolver.h"
#include "../core/parquet_backed_dataset.h"
#include "../core/pipeline_materializer.h"
#include "../core/profiler_trace.h"
#include "../core/process_memory_snapshot.h"
#include "../core/training_trace_collector.h"
#include "panels/training_plot_panel.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <sstream>
#include <unordered_set>
#include <chrono>
#include <filesystem>
#include <future>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

namespace gui {

namespace {

constexpr const char* kGraphTrainingPreparationTaskName =
    "Prepare graph training";

void SetBlockedStatus(GraphTrainingLaunchResult& result,
                      std::string title,
                      std::string detail);

void PostPlotPanelUpdate(
    std::weak_ptr<cyxwiz::TrainingPlotPanel> plot_panel,
    std::function<void(cyxwiz::TrainingPlotPanel&)> update) {
    if (!update) {
        return;
    }
    cyxwiz::AsyncTaskManager::Instance().PostToMainThread(
        [plot_panel, update = std::move(update)]() mutable {
            if (auto panel = plot_panel.lock()) {
                update(*panel);
            }
        });
}

bool HasActiveGraphTrainingPreparation() {
    const auto tasks = cyxwiz::AsyncTaskManager::Instance().GetActiveTasks();
    return std::any_of(
        tasks.begin(), tasks.end(), [](const cyxwiz::TaskInfo& task) {
            return task.name == kGraphTrainingPreparationTaskName;
        });
}

void ReportMaterializationProgress(
    cyxwiz::LambdaTask& task,
    std::weak_ptr<cyxwiz::TrainingPlotPanel> plot_panel,
    const cyxwiz::PipelineOperatorProgress& event,
    float task_progress) {
    const std::string message = event.message.empty()
        ? event.stage
        : event.message;
    task.ReportProgress(task_progress, message);
    cyxwiz::TrainingTraceCollector::Instance().RecordTaskProgress(
        task.GetId(),
        task.GetName(),
        event.stage.empty() ? "Materializing" : event.stage,
        task_progress,
        message,
        event.status.empty() ? "running" : event.status,
        event.node_id,
        event.node_name,
        event.estimated_memory_bytes,
        event.processed_items,
        event.total_items,
        event.memory_risk_level,
        event.available_memory_bytes,
        event.safe_memory_budget_bytes,
        event.process_memory_detected,
        event.process_resident_memory_bytes,
        event.process_private_memory_bytes,
        event.process_resident_growth_bytes,
        event.process_private_memory_name,
        event.process_memory_source);
    PostPlotPanelUpdate(
        plot_panel,
        [event, message, task_progress](cyxwiz::TrainingPlotPanel& panel) {
            panel.SetPreparationState(true, message, task_progress);
            panel.RecordMaterializationProgress(
                event.stage,
                message,
                task_progress,
                event.estimated_memory_bytes,
                event.processed_items,
                event.total_items,
                event.node_id,
                event.node_name,
                event.memory_risk_level,
                event.status,
                event.available_memory_bytes,
                event.safe_memory_budget_bytes,
                event.process_memory_detected,
                event.process_resident_memory_bytes,
                event.process_private_memory_bytes,
                event.process_resident_growth_bytes,
                event.process_private_memory_name,
                event.process_memory_source);
        });
}

bool DispatchTrainingOnMainThread(
    cyxwiz::LambdaTask& task,
    GraphTrainingDispatch dispatch,
    cyxwiz::TrainingConfiguration config,
    const std::string& dataset_name,
    const std::string& label_column,
    int epochs,
    int batch_size,
    std::weak_ptr<cyxwiz::TrainingPlotPanel> plot_panel,
    std::function<void(bool)> callback,
    std::string& error_message) {
    auto promise = std::make_shared<std::promise<bool>>();
    auto error = std::make_shared<std::string>();
    auto cancelled = std::make_shared<std::atomic_bool>(false);
    auto config_ptr =
        std::make_shared<cyxwiz::TrainingConfiguration>(std::move(config));
    auto future = promise->get_future();

    cyxwiz::AsyncTaskManager::Instance().PostToMainThread(
        [dispatch = std::move(dispatch),
         config_ptr,
         dataset_name,
         label_column,
         epochs,
         batch_size,
         plot_panel,
         callback = std::move(callback),
         promise,
         error,
         cancelled]() mutable {
            try {
                if (cancelled->load()) {
                    promise->set_value(false);
                    return;
                }
                const bool started = dispatch(
                    std::move(*config_ptr),
                    dataset_name,
                    label_column,
                    epochs,
                    batch_size,
                    plot_panel,
                    std::move(callback));
                promise->set_value(started);
            } catch (const std::exception& e) {
                *error = e.what();
                promise->set_value(false);
            } catch (...) {
                *error = "Unknown training dispatch failure.";
                promise->set_value(false);
            }
        });

    while (future.wait_for(std::chrono::milliseconds(25)) !=
           std::future_status::ready) {
        if (task.ShouldStop()) {
            cancelled->store(true);
            error_message = "Training launch cancelled before dispatch.";
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    const bool started = future.get();
    if (!started && !error->empty()) {
        error_message = *error;
    }
    return started;
}

cyxwiz::MaterializationCacheConfig DefaultMaterializationCacheConfig(
    const std::filesystem::path& project_root = {}) {
    cyxwiz::MaterializationCacheConfig config;
    config.mode = cyxwiz::MaterializationCacheMode::Auto;
    config.cache_root = project_root.empty()
        ? std::filesystem::current_path() / ".cyxwiz"
        : project_root.lexically_normal();
    config.artifact_format = "parquet";
    return config;
}

std::string MaterializationCacheStatusLabel(
    cyxwiz::MaterializationCacheStatus status) {
    if (status == cyxwiz::MaterializationCacheStatus::Disabled) {
        return "completed";
    }
    return std::string("cache_") + cyxwiz::MaterializationCacheStatusName(status);
}

void ReportMaterializationCacheStatus(
    cyxwiz::LambdaTask& task,
    std::weak_ptr<cyxwiz::TrainingPlotPanel> plot_panel,
    const cyxwiz::MaterializeResult& materialize_result,
    const std::optional<cyxwiz::PipelineOperatorProgress>& preflight_evidence) {
    if (materialize_result.cache_status ==
        cyxwiz::MaterializationCacheStatus::Disabled) {
        return;
    }

    const std::string status = MaterializationCacheStatusLabel(
        materialize_result.cache_status);
    std::string message = materialize_result.cache_message.empty()
        ? (std::string("Materialization cache status: ") + status)
        : materialize_result.cache_message;
    if (!materialize_result.cache_key.empty()) {
        message += " (key " + materialize_result.cache_key.substr(0, 8) + ")";
    }

    constexpr float kCacheProgress = 0.64f;
    task.ReportProgress(kCacheProgress, message);
    const auto process = cyxwiz::DetectProcessMemorySnapshot();
    const auto preflight = preflight_evidence.value_or(
        cyxwiz::PipelineOperatorProgress{});
    const uint64_t resident_growth =
        process.detected && preflight.process_memory_detected &&
            process.resident_bytes >= preflight.process_resident_memory_bytes
        ? process.resident_bytes - preflight.process_resident_memory_bytes
        : 0;
    cyxwiz::TrainingTraceCollector::Instance().RecordTaskProgress(
        task.GetId(),
        task.GetName(),
        "MaterializationCache",
        kCacheProgress,
        message,
        status,
        preflight.node_id,
        preflight.node_name,
        preflight.estimated_memory_bytes,
        preflight.processed_items,
        preflight.total_items,
        preflight.memory_risk_level,
        preflight.available_memory_bytes,
        preflight.safe_memory_budget_bytes,
        process.detected,
        process.resident_bytes,
        process.private_bytes,
        resident_growth,
        process.private_metric_name,
        process.source);
    PostPlotPanelUpdate(
        plot_panel,
        [message,
         kCacheProgress,
         status,
         cache_key = materialize_result.cache_key,
         cache_artifact_path = materialize_result.cache_artifact_path,
         cache_manifest_path = materialize_result.cache_manifest_path,
         cache_row_count = materialize_result.cache_row_count,
         cache_column_count = materialize_result.cache_column_count,
         preflight,
         process,
         resident_growth](
            cyxwiz::TrainingPlotPanel& panel) {
            panel.SetPreparationState(true, message, kCacheProgress);
            panel.RecordMaterializationProgress(
                "MaterializationCache",
                message,
                kCacheProgress,
                preflight.estimated_memory_bytes,
                preflight.processed_items,
                preflight.total_items,
                preflight.node_id,
                preflight.node_name,
                preflight.memory_risk_level,
                status,
                preflight.available_memory_bytes,
                preflight.safe_memory_budget_bytes,
                process.detected,
                process.resident_bytes,
                process.private_bytes,
                resident_growth,
                process.private_metric_name,
                process.source,
                cache_key,
                cache_artifact_path,
                cache_manifest_path,
                cache_row_count,
                cache_column_count);
        });
}

std::string FindDatasetName(const std::vector<MLNode>& nodes) {
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

std::string FindLabelColumn(
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

void ApplyLegacyOptimizerLoopParams(
    const std::vector<MLNode>& nodes,
    const cyxwiz::TrainingConfiguration& config,
    int& epochs,
    int& batch_size) {
    for (const auto& node : nodes) {
        if (config.optimizer_node_id >= 0 && node.id != config.optimizer_node_id) {
            continue;
        }
        if (node.type != NodeType::Adam &&
            node.type != NodeType::SGD &&
            node.type != NodeType::AdamW &&
            node.type != NodeType::RMSprop &&
            node.type != NodeType::Adagrad &&
            node.type != NodeType::NAdam) {
            continue;
        }

        auto ep_it = node.parameters.find("epochs");
        if (ep_it != node.parameters.end() && !ep_it->second.empty()) {
            if (config.has_data_loader) {
                spdlog::warn("epochs is set on the optimizer node AND a "
                             "DataLoader node is present - using DataLoader's "
                             "value ({}). Remove epochs from the optimizer "
                             "node to clear this warning.", epochs);
            } else {
                spdlog::warn("epochs on the optimizer node is deprecated - "
                             "move it to a DataLoader node. Honoring legacy "
                             "value for now.");
                try { epochs = std::stoi(ep_it->second); } catch (...) {}
            }
        }

        auto bs_it = node.parameters.find("batch_size");
        if (bs_it != node.parameters.end() && !bs_it->second.empty()) {
            if (config.has_data_loader) {
                spdlog::warn("batch_size is set on the optimizer node AND a "
                             "DataLoader node is present - using DataLoader's "
                             "value ({}). Remove batch_size from the optimizer "
                             "node to clear this warning.", batch_size);
            } else {
                spdlog::warn("batch_size on the optimizer node is deprecated - "
                             "move it to a DataLoader node. Honoring legacy "
                             "value for now.");
                try { batch_size = std::stoi(bs_it->second); } catch (...) {}
            }
        }
        break;
    }
}

std::string ResolveRuntimeArrowLabelColumn(
    cyxwiz::DataRegistry& registry,
    const std::string& dataset_name,
    const std::string& requested_label) {
    auto arrow_ds = registry.GetArrowDataset(dataset_name);
    auto parquet_ds = registry.GetParquetBackedDataset(dataset_name);
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
    cyxwiz::DataRegistry& registry,
    const std::string& dataset_name,
    const std::string& label_column,
    cyxwiz::TrainingConfiguration& config) {
    if (config.sequence_batch.enabled || config.is_time_series) return;

    std::shared_ptr<arrow::Schema> schema;
    if (auto arrow_ds = registry.GetArrowDataset(dataset_name)) {
        schema = arrow_ds->GetSchema();
    } else if (auto parquet_ds =
                   registry.GetParquetBackedDataset(dataset_name)) {
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

std::shared_ptr<arrow::Schema> FindTabularSchema(
    cyxwiz::DataRegistry& registry,
    const std::string& dataset_name) {
    if (auto arrow_ds = registry.GetArrowDataset(dataset_name)) {
        return arrow_ds->GetSchema();
    }
    if (auto parquet_ds = registry.GetParquetBackedDataset(dataset_name)) {
        return parquet_ds->GetSchema();
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
    cyxwiz::DataRegistry& registry,
    const std::string& dataset_name,
    const std::string& column_name,
    int64_t max_rows,
    std::unordered_set<std::string>& values,
    int64_t& rows_scanned,
    std::string& reason) {
    if (auto arrow_ds = registry.GetArrowDataset(dataset_name)) {
        if (arrow_ds->GetNumRows() > max_rows) {
            reason = "row count exceeds bounded identifier scan limit";
            return false;
        }
        return AppendColumnValues(
            arrow_ds->GetArrowTable(), column_name, values, rows_scanned);
    }
    if (auto parquet_ds = registry.GetParquetBackedDataset(dataset_name)) {
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
    cyxwiz::DataRegistry& registry,
    const std::string& dataset_name,
    const std::vector<std::string>& columns,
    int64_t max_rows,
    std::unordered_set<std::string>& values,
    std::string& example,
    std::string& reason) {
    if (auto arrow_ds = registry.GetArrowDataset(dataset_name)) {
        if (arrow_ds->GetNumRows() > max_rows) {
            reason = "row count exceeds bounded exact-row scan limit";
            return false;
        }
        return AppendRowSignatures(
            arrow_ds->GetArrowTable(), columns, values, example);
    }
    if (auto parquet_ds = registry.GetParquetBackedDataset(dataset_name)) {
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
    cyxwiz::DataRegistry& registry,
    const cyxwiz::ResolvedDatasetRole& train_role,
    const cyxwiz::ResolvedDatasetRole& role,
    const char* role_name,
    GraphTrainingLaunchResult& launch_result,
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

    auto train_schema = FindTabularSchema(registry, train_role.dataset_name);
    auto role_schema = FindTabularSchema(registry, role.dataset_name);
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
        if (!AppendDatasetColumnValues(registry, train_role.dataset_name,
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
        if (!AppendDatasetColumnValues(registry, role.dataset_name,
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

    auto train_arrow = registry.GetArrowDataset(train_role.dataset_name);
    auto role_arrow = registry.GetArrowDataset(role.dataset_name);
    auto train_parquet = registry.GetParquetBackedDataset(train_role.dataset_name);
    auto role_parquet = registry.GetParquetBackedDataset(role.dataset_name);
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
    if (!AppendDatasetRowSignatures(registry, train_role.dataset_name,
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
    if (!AppendDatasetRowSignatures(registry, role.dataset_name,
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
    cyxwiz::DataRegistry& registry,
    const cyxwiz::ResolvedDatasetRole& train_role,
    const cyxwiz::ResolvedDatasetRole& role,
    const char* role_name,
    GraphTrainingLaunchResult& launch_result) {
    if (!role.IsSupplied()) return true;

    auto train_schema = FindTabularSchema(registry, train_role.dataset_name);
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

    auto role_schema = FindTabularSchema(registry, role.dataset_name);
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

bool ValidateSuppliedRolePreflight(
    cyxwiz::DataRegistry& registry,
    cyxwiz::ResolvedDatasetRoles& roles,
    GraphTrainingLaunchResult& launch_result) {
    auto& manifest = roles.manifest;
    if (!ValidateSuppliedRoleSchema(
            registry, roles.train, roles.dev, "Dev", launch_result)) {
        manifest.dev_compatibility =
            cyxwiz::PartitionCompatibility::Incompatible;
        manifest.dev_status_reason = launch_result.error_message;
        return false;
    }
    manifest.dev_compatibility = cyxwiz::PartitionCompatibility::Compatible;
    if (!ValidateSuppliedRoleSchema(
            registry, roles.train, roles.test, "Test", launch_result)) {
        manifest.test_compatibility =
            cyxwiz::PartitionCompatibility::Incompatible;
        manifest.test_status_reason = launch_result.error_message;
        return false;
    }
    manifest.test_compatibility = cyxwiz::PartitionCompatibility::Compatible;
    if (!ValidateSuppliedRoleLeakage(
            registry, roles.train, roles.dev, "Dev", launch_result,
            manifest.dev_leakage, manifest.dev_status_reason)) {
        return false;
    }
    return ValidateSuppliedRoleLeakage(
        registry, roles.train, roles.test, "Test", launch_result,
        manifest.test_leakage, manifest.test_status_reason);
}

bool ValidateSequenceLaunchColumns(
    cyxwiz::DataRegistry& registry,
    const std::string& dataset_name,
    const cyxwiz::TrainingConfiguration& config,
    std::string& error_message) {
    if (!config.sequence_batch.enabled) {
        return true;
    }

    auto schema = FindTabularSchema(registry, dataset_name);
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
        {"tag", DefaultSequenceColumn(config.sequence_batch.tag_column,
                                       "ner_tags")},
    };
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

void SetBlockedStatus(GraphTrainingLaunchResult& result,
                      std::string title,
                      std::string detail) {
    result.status_title = std::move(title);
    result.status_detail = std::move(detail);
    result.error_message = result.status_detail.empty()
        ? result.status_title
        : result.status_detail;
}

} // namespace

cyxwiz::MaterializationCacheConfig GraphMaterializationCacheConfig(
    const std::filesystem::path& project_root) {
    return DefaultMaterializationCacheConfig(project_root);
}

GraphMaterializationPreflightResult PreflightGraphMaterialization(
    const std::vector<MLNode>& nodes,
    const std::vector<NodeLink>& links,
    const cyxwiz::TrainingConfiguration& config,
    cyxwiz::DataRegistry& registry,
    cyxwiz::MaterializationMemoryContext memory_context) {
    GraphMaterializationPreflightResult result;
    result.checked = true;
    result.dataset_name = !config.dataset_name.empty()
        ? config.dataset_name
        : FindDatasetName(nodes);

    if (result.dataset_name.empty()) {
        result.blocked = true;
        result.status_title = "Materialization preflight blocked";
        result.status_detail =
            "No dataset is configured for the materialization memory check.";
        return result;
    }

    const auto source_kind = cyxwiz::ResolvePipelineMaterializerSourceKind(
        registry, result.dataset_name);
    if (source_kind != cyxwiz::PipelineMaterializerSourceKind::ArrowTable) {
        result.status_title = "Materialization estimate unavailable";
        result.status_detail =
            "A truthful pre-start estimate is unavailable for dataset '" +
            result.dataset_name + "' (" +
            cyxwiz::PipelineMaterializerSourceKindName(source_kind) +
            "). Runtime materialization guards remain active.";
        return result;
    }

    auto source_dataset = registry.GetArrowDataset(result.dataset_name);
    auto source_table = source_dataset ? source_dataset->GetArrowTable() : nullptr;
    if (!source_table) {
        result.blocked = true;
        result.status_title = "Materialization preflight blocked";
        result.status_detail =
            "The Arrow dataset '" + result.dataset_name +
            "' is unavailable or has no table.";
        return result;
    }

    auto table_result = cyxwiz::PipelineMaterializer::PreflightTable(
        nodes, links, source_table, result.dataset_name,
        std::move(memory_context));
    if (!table_result.success) {
        result.blocked = true;
        result.status_title = "Materialization preflight blocked";
        result.status_detail = table_result.error_message;
        return result;
    }
    if (!table_result.memory_preflight_observed) {
        result.status_title = "No materializing estimate required";
        result.status_detail =
            "No guarded Arrow materializer was found before training. Runtime "
            "guards remain active for any later data-dependent work.";
        return result;
    }

    result.estimate_available = true;
    result.evidence = std::move(table_result.memory_preflight);
    result.blocked = result.evidence.memory_risk_level == "blocked" ||
                     result.evidence.status == "blocked";
    result.requires_confirmation =
        result.evidence.memory_risk_level == "warning" ||
        result.evidence.memory_risk_level == "risky";
    result.status_title = result.blocked
        ? "Materialization memory check blocked"
        : result.requires_confirmation
            ? "Materialization memory confirmation required"
            : "Materialization memory check passed";
    result.status_detail = result.evidence.message;
    result.status_detail +=
        "\n\nThis is the first truthful operator estimate. Downstream shapes "
        "that depend on materialized data remain unknown and are protected by "
        "the runtime guard.";
    return result;
}

GraphTrainingLaunchResult StartGraphTrainingFromCompiledConfig(
    const std::vector<MLNode>& nodes,
    const std::vector<NodeLink>& links,
    cyxwiz::TrainingConfiguration config,
    cyxwiz::DataRegistry& registry,
    std::weak_ptr<cyxwiz::TrainingPlotPanel> plot_panel,
    std::function<void(bool)> node_editor_callback,
    GraphTrainingDispatch dispatch,
    cyxwiz::MaterializationMemoryPolicy materialization_memory_policy,
    std::optional<cyxwiz::PipelineOperatorProgress>
        materialization_preflight_evidence,
    std::filesystem::path project_root) {

    GraphTrainingLaunchResult result;

    if (HasActiveGraphTrainingPreparation()) {
        SetBlockedStatus(
            result,
            "Training preparation already active",
            "Wait for the current graph preparation to finish or cancel it "
            "before starting another training run.");
        return result;
    }

    if (!config.is_valid) {
        SetBlockedStatus(result,
                         "Training configuration blocked",
                         "Compiled training configuration is invalid.");
        return result;
    }
    if (!dispatch) {
        SetBlockedStatus(result,
                         "Training launch unavailable",
                         "Training dispatch callback is missing.");
        return result;
    }

    std::string dataset_name = !config.dataset_name.empty()
        ? config.dataset_name
        : FindDatasetName(nodes);
    if (dataset_name.empty()) {
        SetBlockedStatus(
            result,
            "Dataset not configured",
            "No dataset loaded. Please configure the Data Input node first.");
        spdlog::error(result.error_message);
        return result;
    }

    std::string label_column = FindLabelColumn(
        nodes, dataset_name, config.data_source_node_id);
    label_column = ResolveRuntimeArrowLabelColumn(
        registry, dataset_name, label_column);
    ReconcileRuntimeDatasetTarget(config, label_column, dataset_name);
    ReconcileRuntimeTabularFeatureWidth(
        registry, dataset_name, label_column, config);
    if (config.dataset_roles.train.dataset_name.empty()) {
        config.dataset_roles.train.dataset_name = dataset_name;
    }
    config.dataset_roles.train.label_column = label_column;
    if (config.dataset_roles.dev.IsSupplied()) {
        config.dataset_roles.dev.label_column = FindLabelColumn(
            nodes, config.dataset_roles.dev.dataset_name,
            config.dataset_roles.dev.source_node_id);
    }
    if (config.dataset_roles.test.IsSupplied()) {
        config.dataset_roles.test.label_column = FindLabelColumn(
            nodes, config.dataset_roles.test.dataset_name,
            config.dataset_roles.test.source_node_id);
    }

    auto validate_supplied_role = [&registry](const cyxwiz::ResolvedDatasetRole& role,
                                              const char* role_name,
                                              GraphTrainingLaunchResult& launch_result) {
        if (!role.IsSupplied()) return true;
        if (registry.GetArrowDataset(role.dataset_name) ||
            registry.GetParquetBackedDataset(role.dataset_name)) {
            return true;
        }
        SetBlockedStatus(launch_result,
                         std::string(role_name) + " dataset unavailable",
                         "The supplied " + std::string(role_name) +
                             " dataset '" + role.dataset_name +
                             "' is not registered. Apply its Data Input node first.");
        return false;
    };
    if (!validate_supplied_role(config.dataset_roles.dev, "Dev", result) ||
        !validate_supplied_role(config.dataset_roles.test, "Test", result)) {
        return result;
    }
    if (!ValidateSuppliedRolePreflight(registry, config.dataset_roles, result)) {
        return result;
    }

    if (config.sequence_batch.enabled) {
        const bool has_arrow = registry.GetArrowDataset(dataset_name) != nullptr;
        const bool has_parquet = registry.GetParquetBackedDataset(dataset_name) != nullptr;
        if (!has_arrow && !has_parquet) {
            SetBlockedStatus(
                result,
                "Sequence dataset unavailable",
                "Sequence training requires a registered Arrow or Parquet "
                "table. Apply the Data Input node before training.");
            spdlog::error("StartTrainingFromGraph: {}", result.error_message);
            return result;
        }
    }

    int batch_size = config.batch_size;
    int epochs = config.epochs;
    ApplyLegacyOptimizerLoopParams(nodes, config, epochs, batch_size);

    result.started = true;
    result.status_title = "Training launch queued";
    result.status_detail =
        "Preparing graph materialization and training loaders in the background.";
    result.effective_dataset_name = dataset_name;
    result.label_column = label_column;
    result.epochs = epochs;
    result.batch_size = batch_size;

    const auto materialization_cache_config =
        DefaultMaterializationCacheConfig(project_root);
    result.materialization_cache_enabled =
        materialization_cache_config.mode != cyxwiz::MaterializationCacheMode::Disabled;
    result.materialization_cache_mode = materialization_cache_config.mode;
    result.materialization_cache_root =
        materialization_cache_config.cache_root.string();
    spdlog::info(
        "Graph materialization cache: mode={}, root='{}', owner={}",
        cyxwiz::MaterializationCacheModeName(
            materialization_cache_config.mode),
        result.materialization_cache_root,
        project_root.empty() ? "standalone_runtime" : "active_project");

    if (auto panel = plot_panel.lock()) {
        panel->Clear();
        panel->SetPreparationState(
            true,
            "Preparing graph materialization and training loaders...",
            0.02f);
        panel->SetVisible(true);
    }
    auto preparation_node_editor_callback = node_editor_callback;
    if (preparation_node_editor_callback) {
        preparation_node_editor_callback(true);
    }

    auto launch_task = std::make_shared<cyxwiz::LambdaTask>(
        kGraphTrainingPreparationTaskName,
        [nodes,
         links,
         config = std::move(config),
         &registry,
         dataset_name,
         label_column,
         epochs,
         batch_size,
         cache_config = materialization_cache_config,
         materialization_memory_policy,
         materialization_preflight_evidence =
             std::move(materialization_preflight_evidence),
         plot_panel,
         callback = std::move(node_editor_callback),
         dispatch = std::move(dispatch)](cyxwiz::LambdaTask& task) mutable {
            CYXWIZ_PROFILE_ZONE("CyxWiz Prepare Graph Training");
            const auto now = std::chrono::system_clock::now().time_since_epoch();
            const auto run_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
            cyxwiz::TrainingTraceCollector::Instance().StartRun(
                "train-" + std::to_string(run_ms));
            cyxwiz::TrainingTraceCollector::Instance().RecordTaskProgress(
                task.GetId(),
                task.GetName(),
                "TrainingSetup",
                0.0f,
                "Preparing graph materialization and training loaders...",
                "running");
            if (materialization_preflight_evidence) {
                ReportMaterializationProgress(
                    task,
                    plot_panel,
                    *materialization_preflight_evidence,
                    0.04f);
            }
            task.ReportProgress(0.05f, "Preparing graph training launch...");
            PostPlotPanelUpdate(
                plot_panel,
                [](cyxwiz::TrainingPlotPanel& panel) {
                    panel.SetPreparationState(
                        true, "Preparing graph training launch...", 0.05f);
                });
            if (task.ShouldStop()) {
                return;
            }

            std::string effective_dataset_name = dataset_name;
            std::string effective_label_column = label_column;

            task.ReportProgress(0.15f, "Materializing graph preprocessing...");
            PostPlotPanelUpdate(
                plot_panel,
                [](cyxwiz::TrainingPlotPanel& panel) {
                    panel.SetPreparationState(
                        true, "Materializing graph preprocessing...", 0.15f);
                });
            cyxwiz::MaterializeResult materialize_result;
            {
                CYXWIZ_PROFILE_ZONE("CyxWiz Pipeline Materialization");
                cyxwiz::PipelineOperatorExecutionContext materialization_context;
                materialization_context.memory.policy =
                    materialization_memory_policy;
                materialization_context.cancellation_requested =
                    [&task]() { return task.ShouldStop(); };
                materialize_result = cyxwiz::PipelineMaterializer::Materialize(
                    nodes, links, registry, effective_dataset_name, cache_config,
                    [&task, plot_panel](const cyxwiz::PipelineOperatorProgress& event) {
                        const float task_progress =
                            0.15f + 0.50f * std::clamp(event.progress, 0.0f, 1.0f);
                        ReportMaterializationProgress(
                            task, plot_panel, event, task_progress);
                    },
                    std::move(materialization_context));
            }
            ReportMaterializationCacheStatus(
                task,
                plot_panel,
                materialize_result,
                materialization_preflight_evidence);
            if (materialize_result.failure_kind ==
                cyxwiz::MaterializationFailureKind::Cancelled) {
                cyxwiz::TrainingTraceCollector::Instance().RecordTaskProgress(
                    task.GetId(),
                    task.GetName(),
                    "Materialization cancelled",
                    0.15f,
                    materialize_result.error_message,
                    "cancelled",
                    materialize_result.failed_node_id,
                    materialize_result.failed_node_name);
                return;
            }
            if (!materialize_result.success) {
                throw std::runtime_error(
                    "Materializer failed for dataset '" +
                    effective_dataset_name + "': " +
                    materialize_result.error_message);
            }
            PostPlotPanelUpdate(
                plot_panel,
                [output_dataset = materialize_result.effective_dataset_name.empty()
                     ? effective_dataset_name
                     : materialize_result.effective_dataset_name,
                 operators_applied = materialize_result.operators_applied,
                 status = MaterializationCacheStatusLabel(
                     materialize_result.cache_status)](
                    cyxwiz::TrainingPlotPanel& panel) {
                    panel.SetMaterializationComplete(
                        output_dataset, operators_applied, status);
                });

            if (materialize_result.skipped_unsupported_source) {
                spdlog::info("StartTrainingFromGraph: materializer skipped '{}' "
                             "({}): {}",
                             effective_dataset_name,
                             cyxwiz::PipelineMaterializerSourceKindName(
                                 materialize_result.source_kind),
                             materialize_result.unsupported_source_reason.empty()
                                 ? "storage backend is unsupported"
                                 : materialize_result.unsupported_source_reason);
            }

            if (materialize_result.operators_applied > 0) {
                task.ReportProgress(0.65f, "Resolving materialized dataset...");
                PostPlotPanelUpdate(
                    plot_panel,
                    [](cyxwiz::TrainingPlotPanel& panel) {
                        panel.SetPreparationState(
                            true, "Resolving materialized dataset...", 0.65f);
                    });
                spdlog::info("StartTrainingFromGraph: materialized '{}' -> '{}' "
                             "({} Cat-1 ops)",
                             effective_dataset_name,
                             materialize_result.effective_dataset_name,
                             materialize_result.operators_applied);
                effective_dataset_name = materialize_result.effective_dataset_name;
                effective_label_column = ResolveRuntimeArrowLabelColumn(
                    registry, effective_dataset_name, effective_label_column);
                ReconcileRuntimeDatasetTarget(
                    config, effective_label_column, effective_dataset_name);
            }

            ReconcileRuntimeTabularFeatureWidth(
                registry, effective_dataset_name, effective_label_column,
                config);

            if (task.ShouldStop()) {
                return;
            }

            auto runtime_roles = config.dataset_roles;
            runtime_roles.train.dataset_name = effective_dataset_name;
            runtime_roles.train.label_column = effective_label_column;
            GraphTrainingLaunchResult role_validation;
            if (!ValidateSuppliedRolePreflight(registry, runtime_roles, role_validation)) {
                throw std::runtime_error(role_validation.error_message);
            }
            config.dataset_roles = runtime_roles;

            if (config.sequence_batch.enabled) {
                task.ReportProgress(0.75f, "Validating sequence launch columns...");
                PostPlotPanelUpdate(
                    plot_panel,
                    [](cyxwiz::TrainingPlotPanel& panel) {
                        panel.SetPreparationState(
                            true, "Validating sequence launch columns...", 0.75f);
                    });
                std::string sequence_column_error;
                if (!ValidateSequenceLaunchColumns(
                    registry, effective_dataset_name, config,
                        sequence_column_error)) {
                    throw std::runtime_error(sequence_column_error);
                }
            }

            config.dataset_name = effective_dataset_name;
            config.dataset_roles.train.dataset_name = effective_dataset_name;

            task.ReportProgress(0.9f, "Starting training...");
            PostPlotPanelUpdate(
                plot_panel,
                [](cyxwiz::TrainingPlotPanel& panel) {
                    panel.SetPreparationState(true, "Starting training...", 0.9f);
                });
            bool started = false;
            {
                CYXWIZ_PROFILE_ZONE("CyxWiz Dispatch Training");
                std::string dispatch_error;
                started = DispatchTrainingOnMainThread(
                    task,
                    std::move(dispatch),
                    std::move(config),
                    effective_dataset_name,
                    effective_label_column,
                    epochs,
                    batch_size,
                    plot_panel,
                    std::move(callback),
                    dispatch_error);
                if (!started && !dispatch_error.empty()) {
                    throw std::runtime_error(dispatch_error);
                }
            }

            if (!started) {
                throw std::runtime_error(
                    "Failed to start training. Another training session may be active "
                    "or the dataset could not be resolved.");
            }

            task.ReportProgress(1.0f, "Training started");
            task.MarkCompleted("Training started", "started");
            PostPlotPanelUpdate(
                plot_panel,
                [effective_dataset_name,
                 operators_applied = materialize_result.operators_applied,
                 status = MaterializationCacheStatusLabel(
                     materialize_result.cache_status)](
                    cyxwiz::TrainingPlotPanel& panel) {
                    if (operators_applied > 0) {
                        panel.SetMaterializationComplete(
                            effective_dataset_name, operators_applied, status);
                    }
                    panel.SetPreparationState(false);
                });
        });
    launch_task->SetCompletionCallback(
        [plot_panel, preparation_node_editor_callback](
            bool success,
            const std::string& error) {
            if (!success) {
                if (auto panel = plot_panel.lock()) {
                    panel->SetPreparationFailed(
                        error.empty() ? "Training preparation failed." : error);
                }
                if (preparation_node_editor_callback) {
                    preparation_node_editor_callback(false);
                }
            }
        });
    cyxwiz::AsyncTaskManager::Instance().Submit(launch_task);

    if (!result.started) {
        SetBlockedStatus(
            result,
            "Training launch blocked",
            "Failed to start training. Another training session may be active "
            "or the runtime batcher rejected the prepared data.");
        spdlog::error(result.error_message);
    }
    return result;
}

} // namespace gui
