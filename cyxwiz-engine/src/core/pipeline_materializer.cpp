#include "pipeline_materializer.h"

#include "arrow_dataset.h"
#include "data_registry.h"
#include "node_executors/pipeline_operator.h"
#include "node_executors/pipeline_operator_factory.h"

#include <arrow/table.h>
#include <spdlog/spdlog.h>

#include <queue>
#include <unordered_set>

namespace cyxwiz {

namespace {

const gui::MLNode* FindDataInputNode(const std::vector<gui::MLNode>& nodes) {
    for (const auto& n : nodes) {
        if (n.type == gui::NodeType::DataInput ||
            n.type == gui::NodeType::DatasetInput) {
            return &n;
        }
    }
    return nullptr;
}

const gui::MLNode* FindNodeById(int id, const std::vector<gui::MLNode>& nodes) {
    for (const auto& n : nodes) {
        if (n.id == id) return &n;
    }
    return nullptr;
}

} // namespace

MaterializeResult PipelineMaterializer::Materialize(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    DataRegistry& registry,
    const std::string& source_dataset_name) {

    MaterializeResult result;
    result.effective_dataset_name = source_dataset_name;

    if (source_dataset_name.empty()) {
        result.success = false;
        result.error_message = "PipelineMaterializer: source_dataset_name is empty";
        return result;
    }

    // v1: only the in-memory Arrow path is materialized. Parquet-backed,
    // image, audio, and text sources fall through unchanged. This keeps the
    // legacy pipelines unaffected while we prove the integration on Arrow.
    if (!registry.IsArrowDataset(source_dataset_name)) {
        spdlog::debug("PipelineMaterializer: source '{}' is not an Arrow dataset, "
                      "skipping materialization", source_dataset_name);
        return result;
    }

    auto source_dataset = registry.GetArrowDataset(source_dataset_name);
    if (!source_dataset) {
        result.success = false;
        result.error_message = "PipelineMaterializer: failed to fetch Arrow dataset '" +
                               source_dataset_name + "' from registry";
        return result;
    }

    auto current_table = source_dataset->GetArrowTable();
    if (!current_table) {
        result.success = false;
        result.error_message = "PipelineMaterializer: Arrow dataset '" +
                               source_dataset_name + "' has a null table";
        return result;
    }

    const gui::MLNode* data_input = FindDataInputNode(nodes);
    if (!data_input) {
        // No DataInput in the graph — nothing to walk. Pass through.
        return result;
    }

    auto& factory = PipelineOperatorFactory::Instance();

    // Quick check: if no node in the graph has a registered Cat-1 operator,
    // skip the BFS entirely. Common case for graphs that haven't migrated
    // to the operator framework yet.
    bool any_registered = false;
    for (const auto& n : nodes) {
        if (n.id == data_input->id) continue;
        if (factory.HasOperator(n.type)) {
            any_registered = true;
            break;
        }
    }
    if (!any_registered) {
        return result;
    }

    // BFS forward from DataInput. Order isn't strictly topological, but
    // for a linear preprocessing chain (the v1 expected shape) BFS visits
    // upstream nodes before downstream, which matches the operator order.
    // Parallel preprocessing branches are a v1 limitation — document and
    // tighten in a later pass if real graphs need it.
    std::queue<int> queue;
    std::unordered_set<int> visited;
    queue.push(data_input->id);
    visited.insert(data_input->id);

    while (!queue.empty()) {
        const int node_id = queue.front();
        queue.pop();

        const gui::MLNode* node = FindNodeById(node_id, nodes);
        if (!node) continue;

        // Skip the DataInput itself — its output IS the source table.
        // Apply any Cat-1 operator registered for downstream nodes.
        if (node->id != data_input->id && factory.HasOperator(node->type)) {
            auto op = factory.Create(node->type);
            if (!op) {
                result.success = false;
                result.error_message = "PipelineMaterializer: factory returned null for node '" +
                                       node->name + "' (type registered but Create failed)";
                return result;
            }

            std::string err;
            if (!op->Configure(node->parameters, err)) {
                result.success = false;
                result.error_message = "PipelineMaterializer: Configure failed for node '" +
                                       node->name + "': " + err;
                return result;
            }

            auto applied = op->Apply(current_table);
            if (!applied.ok()) {
                result.success = false;
                result.error_message = "PipelineMaterializer: Apply failed for node '" +
                                       node->name + "': " + applied.status().message();
                return result;
            }

            current_table = applied.ValueOrDie();
            ++result.operators_applied;
            spdlog::info("PipelineMaterializer: applied operator '{}' on node '{}'",
                         op->GetName(), node->name);
        }

        // Enqueue forward neighbors via outgoing links.
        for (const auto& link : links) {
            if (link.from_node == node_id && visited.insert(link.to_node).second) {
                queue.push(link.to_node);
            }
        }
    }

    if (result.operators_applied == 0) {
        // Reachability check above was conservative — a node was registered
        // but never reached from DataInput. Treat as pass-through.
        return result;
    }

    const std::string materialized_name = source_dataset_name + kMaterializedSuffix;
    auto registered = registry.RegisterArrowTable(current_table, materialized_name);
    if (!registered) {
        result.success = false;
        result.error_message = "PipelineMaterializer: RegisterArrowTable failed for '" +
                               materialized_name + "'";
        return result;
    }

    result.effective_dataset_name = materialized_name;
    spdlog::info("PipelineMaterializer: materialized '{}' -> '{}' ({} operators applied)",
                 source_dataset_name, materialized_name, result.operators_applied);
    return result;
}

} // namespace cyxwiz
