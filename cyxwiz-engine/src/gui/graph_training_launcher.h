#pragma once

#include "../core/graph_compiler.h"
#include "../core/pipeline_materializer.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {
class DataRegistry;
class TrainingPlotPanel;
} // namespace cyxwiz

namespace gui {

struct GraphTrainingLaunchResult {
    bool started = false;
    std::string status_title;
    std::string status_detail;
    std::string error_message;
    std::string effective_dataset_name;
    std::string label_column;
    int operators_applied = 0;
    cyxwiz::PipelineMaterializerSourceKind materializer_source_kind =
        cyxwiz::PipelineMaterializerSourceKind::Unknown;
    bool materializer_skipped_unsupported_source = false;
    std::string materializer_unsupported_source_reason;
    std::string materializer_diagnostic_message;
    bool materialization_cache_enabled = false;
    cyxwiz::MaterializationCacheMode materialization_cache_mode =
        cyxwiz::MaterializationCacheMode::Disabled;
    std::string materialization_cache_root;
    int epochs = 0;
    int batch_size = 0;
};

using GraphTrainingDispatch = std::function<bool(
    cyxwiz::TrainingConfiguration config,
    const std::string& dataset_name,
    const std::string& label_column,
    int epochs,
    int batch_size,
    std::weak_ptr<cyxwiz::TrainingPlotPanel> plot_panel,
    std::function<void(bool)> node_editor_callback)>;

// Shared cache policy for graph preprocessing used by Train and checkpoint
// evaluation preparation.
cyxwiz::MaterializationCacheConfig GraphMaterializationCacheConfig();

GraphTrainingLaunchResult StartGraphTrainingFromCompiledConfig(
    const std::vector<MLNode>& nodes,
    const std::vector<NodeLink>& links,
    cyxwiz::TrainingConfiguration config,
    cyxwiz::DataRegistry& registry,
    std::weak_ptr<cyxwiz::TrainingPlotPanel> plot_panel,
    std::function<void(bool)> node_editor_callback,
    GraphTrainingDispatch dispatch);

} // namespace gui
