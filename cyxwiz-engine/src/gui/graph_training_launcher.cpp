#include "graph_training_launcher.h"

#include "../core/arrow_dataset.h"
#include "../core/async_task_manager.h"
#include "../core/data_registry.h"
#include "../core/label_column_resolver.h"
#include "../core/parquet_backed_dataset.h"
#include "../core/pipeline_materializer.h"
#include "../core/profiler_trace.h"
#include "../core/training_trace_collector.h"
#include "panels/training_plot_panel.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <vector>

namespace gui {

namespace {

void SetBlockedStatus(GraphTrainingLaunchResult& result,
                      std::string title,
                      std::string detail);

cyxwiz::MaterializationCacheConfig DefaultMaterializationCacheConfig() {
    cyxwiz::MaterializationCacheConfig config;
    config.mode = cyxwiz::MaterializationCacheMode::Auto;
    config.cache_root = std::filesystem::current_path() / ".cyxwiz";
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
    const cyxwiz::MaterializeResult& materialize_result) {
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
    cyxwiz::TrainingTraceCollector::Instance().RecordTaskProgress(
        task.GetId(),
        task.GetName(),
        "MaterializationCache",
        kCacheProgress,
        message,
        status);
    if (auto panel = plot_panel.lock()) {
        panel->SetPreparationState(true, message, kCacheProgress);
        panel->RecordMaterializationProgress(
            "MaterializationCache",
            message,
            kCacheProgress,
            0,
            0,
            0,
            -1,
            "",
            "",
            status,
            materialize_result.cache_key,
            materialize_result.cache_artifact_path,
            materialize_result.cache_manifest_path,
            materialize_result.cache_row_count,
            materialize_result.cache_column_count);
    }
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
    if (!arrow_ds) return requested_label;

    auto schema = arrow_ds->GetSchema();
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
        spdlog::info("StartTrainingFromGraph: resolved Arrow label column "
                     "'{}' -> '{}' for materialized dataset '{}'",
                     requested_label.empty() ? "<auto>" : requested_label,
                     resolved,
                     dataset_name);
    }
    return resolved;
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

bool ValidateSuppliedRoleSchemas(
    cyxwiz::DataRegistry& registry,
    const cyxwiz::ResolvedDatasetRoles& roles,
    GraphTrainingLaunchResult& launch_result) {
    return ValidateSuppliedRoleSchema(
               registry, roles.train, roles.dev, "Dev", launch_result) &&
           ValidateSuppliedRoleSchema(
               registry, roles.train, roles.test, "Test", launch_result);
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

GraphTrainingLaunchResult StartGraphTrainingFromCompiledConfig(
    const std::vector<MLNode>& nodes,
    const std::vector<NodeLink>& links,
    cyxwiz::TrainingConfiguration config,
    cyxwiz::DataRegistry& registry,
    std::weak_ptr<cyxwiz::TrainingPlotPanel> plot_panel,
    std::function<void(bool)> node_editor_callback,
    GraphTrainingDispatch dispatch) {

    GraphTrainingLaunchResult result;

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
    if (!ValidateSuppliedRoleSchemas(registry, config.dataset_roles, result)) {
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

    const auto materialization_cache_config = DefaultMaterializationCacheConfig();
    result.materialization_cache_enabled =
        materialization_cache_config.mode != cyxwiz::MaterializationCacheMode::Disabled;
    result.materialization_cache_mode = materialization_cache_config.mode;
    result.materialization_cache_root =
        materialization_cache_config.cache_root.string();

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
        "Prepare graph training",
        [nodes,
         links,
         config = std::move(config),
         &registry,
         dataset_name,
         label_column,
         epochs,
         batch_size,
         cache_config = materialization_cache_config,
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
            task.ReportProgress(0.05f, "Preparing graph training launch...");
            if (auto panel = plot_panel.lock()) {
                panel->SetPreparationState(
                    true, "Preparing graph training launch...", 0.05f);
            }
            if (task.ShouldStop()) {
                return;
            }

            std::string effective_dataset_name = dataset_name;
            std::string effective_label_column = label_column;

            task.ReportProgress(0.15f, "Materializing graph preprocessing...");
            if (auto panel = plot_panel.lock()) {
                panel->SetPreparationState(
                    true, "Materializing graph preprocessing...", 0.15f);
            }
            cyxwiz::MaterializeResult materialize_result;
            {
                CYXWIZ_PROFILE_ZONE("CyxWiz Pipeline Materialization");
                materialize_result = cyxwiz::PipelineMaterializer::Materialize(
                    nodes, links, registry, effective_dataset_name, cache_config,
                    [&task, plot_panel](const cyxwiz::PipelineOperatorProgress& event) {
                        const float task_progress =
                            0.15f + 0.50f * std::clamp(event.progress, 0.0f, 1.0f);
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
                            event.memory_risk_level);
                        if (auto panel = plot_panel.lock()) {
                            panel->SetPreparationState(true, message, task_progress);
                            panel->RecordMaterializationProgress(
                                event.stage,
                                message,
                                task_progress,
                                event.estimated_memory_bytes,
                                event.processed_items,
                                event.total_items,
                                event.node_id,
                                event.node_name,
                                event.memory_risk_level,
                                event.status);
                        }
                    });
            }
            ReportMaterializationCacheStatus(task, plot_panel, materialize_result);
            if (!materialize_result.success) {
                throw std::runtime_error(
                    "Materializer failed for dataset '" +
                    effective_dataset_name + "': " +
                    materialize_result.error_message);
            }
            if (auto panel = plot_panel.lock()) {
                panel->SetMaterializationComplete(
                    materialize_result.effective_dataset_name.empty()
                        ? effective_dataset_name
                        : materialize_result.effective_dataset_name,
                    materialize_result.operators_applied,
                    MaterializationCacheStatusLabel(materialize_result.cache_status));
            }

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
                if (auto panel = plot_panel.lock()) {
                    panel->SetPreparationState(
                        true, "Resolving materialized dataset...", 0.65f);
                }
                spdlog::info("StartTrainingFromGraph: materialized '{}' -> '{}' "
                             "({} Cat-1 ops)",
                             effective_dataset_name,
                             materialize_result.effective_dataset_name,
                             materialize_result.operators_applied);
                effective_dataset_name = materialize_result.effective_dataset_name;
                effective_label_column = ResolveRuntimeArrowLabelColumn(
                    registry, effective_dataset_name, effective_label_column);
            }

            if (task.ShouldStop()) {
                return;
            }

            auto runtime_roles = config.dataset_roles;
            runtime_roles.train.dataset_name = effective_dataset_name;
            runtime_roles.train.label_column = effective_label_column;
            GraphTrainingLaunchResult role_validation;
            if (!ValidateSuppliedRoleSchemas(registry, runtime_roles, role_validation)) {
                throw std::runtime_error(role_validation.error_message);
            }
            config.dataset_roles = runtime_roles;

            if (config.sequence_batch.enabled) {
                task.ReportProgress(0.75f, "Validating sequence launch columns...");
                if (auto panel = plot_panel.lock()) {
                    panel->SetPreparationState(
                        true, "Validating sequence launch columns...", 0.75f);
                }
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
            if (auto panel = plot_panel.lock()) {
                panel->SetPreparationState(true, "Starting training...", 0.9f);
            }
            bool started = false;
            {
                CYXWIZ_PROFILE_ZONE("CyxWiz Dispatch Training");
                started = dispatch(
                    std::move(config), effective_dataset_name,
                    effective_label_column, epochs, batch_size, plot_panel,
                    std::move(callback));
            }

            if (!started) {
                throw std::runtime_error(
                    "Failed to start training. Another training session may be active "
                    "or the dataset could not be resolved.");
            }

            task.ReportProgress(1.0f, "Training started");
            task.MarkCompleted("Training started", "started");
            if (auto panel = plot_panel.lock()) {
                if (materialize_result.operators_applied > 0) {
                    panel->SetMaterializationComplete(
                        effective_dataset_name,
                        materialize_result.operators_applied,
                        MaterializationCacheStatusLabel(materialize_result.cache_status));
                }
                panel->SetPreparationState(false);
            }
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
