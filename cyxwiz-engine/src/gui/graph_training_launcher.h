#pragma once

#include "../core/graph_compiler.h"
#include "../core/pipeline_materializer.h"
#include "../core/execution_device_preferences.h"

#include <functional>
#include <memory>
#include <optional>
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

struct GraphMaterializationPreflightResult {
    bool checked = false;
    bool estimate_available = false;
    bool blocked = false;
    bool requires_confirmation = false;
    std::string dataset_name;
    std::string status_title;
    std::string status_detail;
    cyxwiz::PipelineOperatorProgress evidence;
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
cyxwiz::MaterializationCacheConfig GraphMaterializationCacheConfig(
    const std::filesystem::path& project_root = {});

// Synchronous, allocation-light pre-start check. It executes the existing
// operator-owned estimator and stops at its first memory decision; it does not
// publish a dataset or approximate unknown downstream shapes.
GraphMaterializationPreflightResult PreflightGraphMaterialization(
    const std::vector<MLNode>& nodes,
    const std::vector<NodeLink>& links,
    const cyxwiz::TrainingConfiguration& config,
    cyxwiz::DataRegistry& registry,
    cyxwiz::MaterializationMemoryContext memory_context = {});

// Launch-readiness findings Compile reports before a run is started, so Train
// never starts a task that is already known to fail:
//  - sequence token / sentence-id / tag columns missing from a loaded
//    train, dev or test dataset (the same check Train runs);
//  - the requested device route is not qualified for training on this machine
//    (error when native CPU fallback is forbidden or CPU recovery is not
//    qualified, otherwise a fallback warning);
//  - a Data Input max_rows limit that training does not apply.
// `route` is the device decision previewed by the caller (Compile uses
// PreviewRequestedRouteTrainingReadiness, the run preflight's own route
// resolution); null skips the device check.
std::vector<cyxwiz::ValidationIssue> CheckGraphLaunchReadiness(
    const std::vector<MLNode>& nodes,
    const cyxwiz::TrainingConfiguration& config,
    cyxwiz::DataRegistry& registry,
    const cyxwiz::RequestedRouteTrainingReadiness* route = nullptr);

GraphTrainingLaunchResult StartGraphTrainingFromCompiledConfig(
    const std::vector<MLNode>& nodes,
    const std::vector<NodeLink>& links,
    cyxwiz::TrainingConfiguration config,
    cyxwiz::DataRegistry& registry,
    std::weak_ptr<cyxwiz::TrainingPlotPanel> plot_panel,
    std::function<void(bool)> node_editor_callback,
    GraphTrainingDispatch dispatch,
    cyxwiz::MaterializationMemoryPolicy materialization_memory_policy = {},
    std::optional<cyxwiz::PipelineOperatorProgress>
        materialization_preflight_evidence = std::nullopt,
    std::filesystem::path project_root = {});

} // namespace gui
