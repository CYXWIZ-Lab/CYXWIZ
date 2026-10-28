#include "training_input_preparation.h"
#include "training_input_graph.h"

// The executor-dependent half (TOFIX178 S0-f2): the graph as the executor
// reads it, and a run of the steps above a node.

#include "arrow_dataset.h"
#include "data_registry.h"
#include "pipeline_executor.h"
#include "pipeline_runtime_capabilities.h"
#include "pipeline_type_names.h"
#include "prepared_data_cache.h"
#include "sparse_feature_dataset.h"

#include <spdlog/spdlog.h>

#include <set>

namespace cyxwiz {

using namespace training_input;

bool PipelineJsonFromGraph(const std::vector<gui::MLNode>& nodes,
                           const std::vector<gui::NodeLink>& links,
                           nlohmann::json& pipeline_json,
                           std::string& error) {
    pipeline_json["nodes"] = nlohmann::json::array();
    for (const auto& node : nodes) {
        // Plot and Dashboard nodes show data; they are not pipeline steps (TOFIX134 P2, P3).
        if (node.type == gui::NodeType::Plot || node.type == gui::NodeType::Dashboard) continue;
        nlohmann::json node_json;
        node_json["id"] = node.id;
        node_json["type"] = PipelineTypeName(node.type);
        node_json["name"] = node.name;
        node_json["parameters"] = nlohmann::json::object();
        for (const auto& [key, value] : node.parameters) node_json["parameters"][key] = value;
        pipeline_json["nodes"].push_back(std::move(node_json));
    }
    // PipelineExecutor treats this top-level list as the canonical graph
    // connection contract, with each link's pin positions.
    pipeline_json["links"] = nlohmann::json::array();
    for (const auto& link : links) {
        const gui::MLNode* from_node = FindNode(nodes, link.from_node);
        const gui::MLNode* to_node = FindNode(nodes, link.to_node);
        if (!from_node || !to_node) {
            error = "Pipeline link " + std::to_string(link.id) + " references a missing node";
            return false;
        }
        if (to_node->type == gui::NodeType::Plot || to_node->type == gui::NodeType::Dashboard) continue;
        const int from_pin = PinIndex(from_node->outputs, link.from_pin);
        const int to_pin = PinIndex(to_node->inputs, link.to_pin);
        if (from_pin < 0 || to_pin < 0) {
            error = "Pipeline link " + std::to_string(link.id) + " references a missing pin";
            return false;
        }
        pipeline_json["links"].push_back({{"start_node", link.from_node},
                                          {"end_node", link.to_node},
                                          {"start_pin_index", from_pin},
                                          {"end_pin_index", to_pin}});
    }
    if (const auto policy = TrainingSplitPolicyOf(nodes, links))
        pipeline_json["training_split"] = TrainingSplitPolicyJson(*policy);
    return true;
}

PreparedStepResult PrepareStepResult(const std::vector<gui::MLNode>& nodes,
                             const std::vector<gui::NodeLink>& links,
                             int node,
                             DataRegistry& registry,
                             const StepRunRoots& roots,
                             const MaterializationCacheConfig& cache_config,
                             const std::function<bool()>& cancelled,
                             const std::function<void(float, const std::string&)>& progress,
                             PipelineExecutor* keep_owner) {
    PreparedStepResult out;
    const gui::MLNode* target = FindNode(nodes, node);
    if (!target) {
        out.error = "the step is not in the graph";
        return out;
    }
    const std::set<int> slice = Upstream(links, node);
    const gui::MLNode* source = nullptr;
    for (const int id : slice)
        if (const gui::MLNode* n = FindNode(nodes, id); n && IsSource(*n)) source = n;

    const PreparedDataKey key = ComputePreparedDataKey(nodes, links, node, registry, roots.project);
    // A Data Input, or a path the pipeline run cannot make (images): its data
    // unchanged, as training reads it.
    if (IsSource(*target) || !key.table_path) {
        out.dataset = source ? DatasetName(*source) : std::string();
        out.ok = !out.dataset.empty();
        if (!out.ok) out.error = "no Data Input feeds '" + target->name + "'";
        else if (!IsSource(*target)) spdlog::info("Steps before '{}' left to training: {}", target->name, key.reason);
        return out;
    }

    if (ProbePreparedData(cache_config, key)) {
        if (progress) progress(0.1f, "Loading the prepared data from disk");
        std::string error;
        if (const auto loaded =
                LoadPreparedData(cache_config, key, registry, "prepared_" + key.key.substr(0, 16), &error)) {
            out.ok = true;
            out.from_disk = true;
            out.dataset = *loaded;
            return out;
        }
        spdlog::warn("The prepared data on disk could not be used ({}); running the steps", error);
    }

    // The steps before `node` (training-only nodes never are), in a run of
    // their own.
    std::vector<gui::MLNode> steps;
    std::vector<gui::NodeLink> step_links;
    for (const auto& n : nodes)
        if (slice.count(n.id) && !IsTrainingOnlyNode(n.type)) steps.push_back(n);
    for (const auto& link : links)
        if (slice.count(link.from_node) && slice.count(link.to_node)) step_links.push_back(link);
    nlohmann::json pipeline_json;
    if (!PipelineJsonFromGraph(steps, step_links, pipeline_json, out.error)) return out;

    PipelineExecutor executor;
    executor.SetArtifactRoot(roots.artifacts);
    executor.SetExportRoot(roots.exports);
    executor.SetIngestionCacheRoot(roots.ingestion_cache);
    executor.SetProjectRoot(roots.project);
    executor.SetProgressCallback([&executor, &cancelled, &progress](float value, const std::string& message) {
        if (cancelled && cancelled()) executor.RequestCancel();
        if (progress) progress(value, message);
    });
    const bool ran = executor.ExecutePipeline(pipeline_json.dump());
    out.ran = true;
    if (cancelled && cancelled()) {
        out.cancelled = true;
        executor.ReleaseAll();
        return out;
    }
    const auto result = executor.NodeResults().find(node);
    if (!ran || result == executor.NodeResults().end() || result->second.empty()) {
        out.error = executor.GetLastError().empty() ? "'" + target->name + "' gave no result" : executor.GetLastError();
        executor.ReleaseAll();
        return out;
    }
    out.ok = true;
    out.dataset = result->second;

    if (key.cacheable) {
        std::string error;
        bool kept = false;
        if (const auto sparse = registry.GetSparseFeatureDataset(out.dataset))
            kept = SavePreparedSparseData(cache_config, key, *sparse, &error);
        else if (const auto table = registry.GetArrowDataset(out.dataset))
            kept = SavePreparedData(cache_config, key, table->GetArrowTable(), out.dataset, &error);
        if (kept) spdlog::info("'{}' is kept on disk for the next session", out.dataset);
        else spdlog::debug("'{}' not kept on disk: {}", out.dataset, error);
    }
    // The other steps' results go now; the result stays (with its owner).
    if (keep_owner) keep_owner->AdoptOwnership(executor);
    else executor.ReleaseAll({out.dataset});
    return out;
}

}  // namespace cyxwiz
