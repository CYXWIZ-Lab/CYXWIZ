#include "graph_training_launcher.h"

#include "../core/arrow_dataset.h"
#include "../core/data_registry.h"
#include "../core/label_column_resolver.h"
#include "../core/parquet_backed_dataset.h"
#include "../core/pipeline_materializer.h"

#include <spdlog/spdlog.h>

#include <memory>
#include <vector>

namespace gui {

namespace {

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

    auto materialize_result = cyxwiz::PipelineMaterializer::Materialize(
        nodes, links, registry, dataset_name);
    if (!materialize_result.success) {
        SetBlockedStatus(
            result,
            "Graph materialization blocked",
            "Materializer failed for dataset '" + dataset_name + "': " +
                materialize_result.error_message);
        spdlog::error(result.error_message);
        return result;
    }

    result.operators_applied = materialize_result.operators_applied;
    result.materializer_source_kind = materialize_result.source_kind;
    result.materializer_skipped_unsupported_source =
        materialize_result.skipped_unsupported_source;
    result.materializer_unsupported_source_reason =
        materialize_result.unsupported_source_reason;
    if (materialize_result.skipped_unsupported_source) {
        spdlog::info("StartTrainingFromGraph: materializer skipped '{}' "
                     "({}): {}",
                     dataset_name,
                     cyxwiz::PipelineMaterializerSourceKindName(
                         materialize_result.source_kind),
                     materialize_result.unsupported_source_reason.empty()
                         ? "storage backend is unsupported"
                         : materialize_result.unsupported_source_reason);
    }
    if (materialize_result.operators_applied > 0) {
        spdlog::info("StartTrainingFromGraph: materialized '{}' -> '{}' "
                     "({} Cat-1 ops)",
                     dataset_name, materialize_result.effective_dataset_name,
                     materialize_result.operators_applied);
        dataset_name = materialize_result.effective_dataset_name;
        label_column = ResolveRuntimeArrowLabelColumn(
            registry, dataset_name, label_column);
    }

    if (config.sequence_batch.enabled) {
        std::string sequence_column_error;
        if (!ValidateSequenceLaunchColumns(registry, dataset_name, config,
                                           sequence_column_error)) {
            SetBlockedStatus(result,
                             "Sequence materialization blocked",
                             sequence_column_error);
            spdlog::error("StartTrainingFromGraph: {}",
                          result.error_message);
            return result;
        }
    }

    config.dataset_name = dataset_name;

    result.effective_dataset_name = dataset_name;
    result.label_column = label_column;
    result.epochs = epochs;
    result.batch_size = batch_size;
    result.started = dispatch(
        std::move(config), dataset_name, label_column, epochs, batch_size,
        plot_panel, std::move(node_editor_callback));

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
