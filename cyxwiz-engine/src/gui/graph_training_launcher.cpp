#include "graph_training_launcher.h"

#include "../core/arrow_dataset.h"
#include "../core/data_registry.h"
#include "../core/label_column_resolver.h"
#include "../core/pipeline_materializer.h"

#include <spdlog/spdlog.h>

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
            node.type != NodeType::RMSprop) {
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
        result.error_message = "compiled training configuration is invalid";
        return result;
    }
    if (!dispatch) {
        result.error_message = "training dispatch callback is missing";
        return result;
    }

    std::string dataset_name = !config.dataset_name.empty()
        ? config.dataset_name
        : FindDatasetName(nodes);
    if (dataset_name.empty()) {
        result.error_message =
            "No dataset loaded. Please configure the Data Input node first.";
        spdlog::error(result.error_message);
        return result;
    }

    std::string label_column = FindLabelColumn(
        nodes, dataset_name, config.data_source_node_id);

    int batch_size = config.batch_size;
    int epochs = config.epochs;
    ApplyLegacyOptimizerLoopParams(nodes, config, epochs, batch_size);

    auto materialize_result = cyxwiz::PipelineMaterializer::Materialize(
        nodes, links, registry, dataset_name);
    if (!materialize_result.success) {
        result.error_message =
            "StartTrainingFromGraph: materializer failed - " +
            materialize_result.error_message;
        spdlog::error(result.error_message);
        return result;
    }

    result.operators_applied = materialize_result.operators_applied;
    if (materialize_result.operators_applied > 0) {
        spdlog::info("StartTrainingFromGraph: materialized '{}' -> '{}' "
                     "({} Cat-1 ops)",
                     dataset_name, materialize_result.effective_dataset_name,
                     materialize_result.operators_applied);
        dataset_name = materialize_result.effective_dataset_name;
        label_column = ResolveRuntimeArrowLabelColumn(
            registry, dataset_name, label_column);
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
        result.error_message =
            "Failed to start training - another training session may be active";
        spdlog::error(result.error_message);
    }
    return result;
}

} // namespace gui
