#include "graph_training_launcher.h"

#include "../core/graph_training_prep.h"

#include "../core/arrow_dataset.h"
#include "../core/async_task_manager.h"
#include "../core/data_registry.h"
#include "../core/label_column_resolver.h"
#include "../core/parquet_backed_dataset.h"
#include "../core/pipeline_materializer.h"
#include "../core/profiler_trace.h"
#include "../core/sparse_feature_dataset.h"
#include "../core/process_memory_snapshot.h"
#include "../core/training_trace_collector.h"
#include <cyxwiz/error_codes.h>
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
    return cyxwiz::FindGraphDatasetName(nodes);
}

std::string FindLabelColumn(
    const std::vector<MLNode>& nodes,
    const std::string& dataset_name,
    int data_source_node_id) {
    return cyxwiz::FindGraphLabelColumn(nodes, dataset_name, data_source_node_id);
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

// Preparation lives in core/graph_training_prep (shared with Server Nodes);
// these keep the launcher's call sites. The registry argument is the one the
// catalog is installed over (DataRegistry).
std::string ResolveRuntimeArrowLabelColumn(
    cyxwiz::DataRegistry&, const std::string& dataset_name, const std::string& requested_label) {
    return cyxwiz::ResolveRuntimeArrowLabelColumn(dataset_name, requested_label);
}


void ReconcileRuntimeTabularFeatureWidth(
    cyxwiz::DataRegistry&, const std::string& dataset_name, const std::string& label_column,
    cyxwiz::TrainingConfiguration& config) {
    cyxwiz::ReconcileRuntimeTabularFeatureWidth(dataset_name, label_column, config);
}

std::shared_ptr<arrow::Schema> FindTabularSchema(cyxwiz::DataRegistry&, const std::string& dataset_name) {
    return cyxwiz::FindGraphDatasetSchema(dataset_name);
}

bool ValidateSuppliedRolePreflight(
    cyxwiz::DataRegistry&, cyxwiz::ResolvedDatasetRoles& roles, const cyxwiz::TrainingConfiguration& config,
    GraphTrainingLaunchResult& launch_result) {
    cyxwiz::LaunchBlock block;
    if (cyxwiz::ValidateSuppliedRolePreflight(roles, config, block)) return true;
    SetBlockedStatus(launch_result, block.title, block.detail);
    return false;
}

bool ValidateSequenceLaunchColumns(
    cyxwiz::DataRegistry&, const std::string& dataset_name, const cyxwiz::TrainingConfiguration& config,
    std::string& error_message) {
    return cyxwiz::ValidateSequenceLaunchColumns(dataset_name, config, error_message);
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

std::vector<cyxwiz::ValidationIssue> CheckGraphLaunchReadiness(
    const std::vector<MLNode>& nodes,
    const cyxwiz::TrainingConfiguration& config,
    cyxwiz::DataRegistry& registry,
    const cyxwiz::RequestedRouteTrainingReadiness* route) {
    std::vector<cyxwiz::ValidationIssue> issues;
    const auto node_name = [&nodes](int node_id) {
        for (const auto& node : nodes) {
            if (node.id == node_id) return node.name;
        }
        return std::string();
    };

    // 1. Sequence columns on every loaded partition (train, dev, test).
    if (config.sequence_batch.enabled) {
        struct Partition {
            const char* role;
            std::string dataset_name;
            int node_id;
        };
        const std::string train_name = !config.dataset_roles.train.dataset_name.empty()
            ? config.dataset_roles.train.dataset_name : config.dataset_name;
        const Partition partitions[] = {
            {"Training", train_name, config.dataset_roles.train.source_node_id},
            {"Validation", config.dataset_roles.dev.dataset_name, config.dataset_roles.dev.source_node_id},
            {"Test", config.dataset_roles.test.dataset_name, config.dataset_roles.test.source_node_id},
        };
        for (const auto& partition : partitions) {
            // Unloaded inputs are reported by the compiler ("click Apply");
            // only a loaded schema can be checked here.
            if (partition.dataset_name.empty() || !FindTabularSchema(registry, partition.dataset_name)) {
                continue;
            }
            std::string error;
            if (!ValidateSequenceLaunchColumns(registry, partition.dataset_name, config, error)) {
                issues.push_back({cyxwiz::IssueLevel::Error, partition.node_id,
                                  node_name(partition.node_id),
                                  std::string(partition.role) + " data: " + error,
                                  cyxwiz::errors::Training::InvalidTrainingSetup});
            }
        }
    }

    // 2. Device route qualification (only graphs that train a model).
    if (!config.layers.empty() && route != nullptr) {
        const cyxwiz::RequestedRouteTrainingReadiness& preview = *route;
        if (preview.route_available && !preview.authorized) {
            const std::string device = preview.route_name.empty()
                ? std::string("the selected device") : "'" + preview.route_name + "'";
            const bool can_fall_back =
                !config.forbid_native_cpu_fallback && preview.cpu_recovery_qualified;
            if (can_fall_back) {
                issues.push_back({cyxwiz::IssueLevel::Warning, -1, "",
                    "Training device " + device + " is not qualified for training (" +
                        preview.message + "); the run would fall back to ArrayFire CPU. "
                        "Run Preferences > Devices > Verify Selected to train on it.",
                    cyxwiz::errors::Training::InvalidTrainingSetup});
            } else {
                issues.push_back({cyxwiz::IssueLevel::Error, -1, "",
                    "Training device " + device + " is not qualified for training on this machine (" +
                        preview.message + "). Run Preferences > Devices > Verify Selected before training.",
                    cyxwiz::errors::Training::InvalidTrainingSetup});
            }
        } else if (preview.route_available && preview.authorized &&
                   preview.type != cyxwiz::DeviceType::CPU &&
                   !config.forbid_native_cpu_fallback && !preview.cpu_recovery_qualified) {
            const std::string device = preview.route_name.empty()
                ? std::string("the selected device") : "'" + preview.route_name + "'";
            issues.push_back({cyxwiz::IssueLevel::Info, -1, "",
                "CPU recovery is not verified on this machine: if " + device +
                    " fails its launch check, the run stops instead of falling back to ArrayFire CPU. "
                    "Preferences > Devices > Verify Selected now verifies the CPU route with the device.",
                cyxwiz::errors::Training::InvalidTrainingSetup});
        }
    }

    // 3. Data Input row limits that training does not apply.
    for (const auto& node : nodes) {
        if (node.type != NodeType::DataInput) continue;
        const auto limit_it = node.parameters.find("max_rows");
        const auto name_it = node.parameters.find("dataset_name");
        if (limit_it == node.parameters.end() || name_it == node.parameters.end()) continue;
        long long limit = 0;
        try { limit = std::stoll(limit_it->second); } catch (...) { continue; }
        if (limit <= 0) continue;
        const auto dataset = registry.GetArrowDataset(name_it->second);
        const auto table = dataset ? dataset->GetArrowTable() : nullptr;
        if (table && table->num_rows() > limit) {
            issues.push_back({cyxwiz::IssueLevel::Warning, node.id, node.name,
                "max_rows=" + limit_it->second + " is not applied by training: the loaded dataset has " +
                    std::to_string(table->num_rows()) + " rows and training uses all of them.",
                cyxwiz::errors::Training::InvalidTrainingSetup});
        }
    }
    return issues;
}

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
    cyxwiz::ReconcileRuntimeDatasetTarget(config, label_column, dataset_name);
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
            registry.GetParquetBackedDataset(role.dataset_name) ||
            registry.GetSparseFeatureDataset(role.dataset_name)) {
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
    if (!ValidateSuppliedRolePreflight(registry, config.dataset_roles, config, result)) {
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
                cyxwiz::ReconcileRuntimeDatasetTarget(
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
            if (!ValidateSuppliedRolePreflight(registry, runtime_roles, config, role_validation)) {
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
