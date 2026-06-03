#include "pipeline_materializer.h"

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

MaterializeTableResult PipelineMaterializer::MaterializeTable(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    const std::shared_ptr<arrow::Table>& source_table) {

    MaterializeTableResult result;
    result.table = source_table;

    if (!source_table) {
        result.success = false;
        result.error_message = "PipelineMaterializer: source Arrow table is null";
        return result;
    }

    const gui::MLNode* data_input = FindDataInputNode(nodes);
    if (!data_input) {
        return result;
    }

    auto& factory = PipelineOperatorFactory::Instance();

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

    auto current_table = source_table;
    std::queue<int> queue;
    std::unordered_set<int> visited;
    queue.push(data_input->id);
    visited.insert(data_input->id);

    while (!queue.empty()) {
        const int node_id = queue.front();
        queue.pop();

        const gui::MLNode* node = FindNodeById(node_id, nodes);
        if (!node) continue;

        if (node->id != data_input->id && factory.HasOperator(node->type)) {
            auto op = factory.Create(node->type);
            if (!op) {
                result.success = false;
                result.error_message =
                    "PipelineMaterializer: factory returned null for node '" +
                    node->name + "' (type registered but Create failed)";
                return result;
            }

            std::string err;
            if (!op->Configure(node->parameters, err)) {
                result.success = false;
                result.error_message =
                    "PipelineMaterializer: Configure failed for node '" +
                    node->name + "': " + err;
                return result;
            }

            auto applied = op->Apply(current_table);
            if (!applied.ok()) {
                result.success = false;
                result.error_message =
                    "PipelineMaterializer: Apply failed for node '" +
                    node->name + "': " + applied.status().message();
                return result;
            }

            current_table = applied.ValueOrDie();
            ++result.operators_applied;
            spdlog::info("PipelineMaterializer: applied operator '{}' on node '{}'",
                         op->GetName(), node->name);
        }

        for (const auto& link : links) {
            if (link.from_node == node_id &&
                visited.insert(link.to_node).second) {
                queue.push(link.to_node);
            }
        }
    }

    result.table = current_table;
    return result;
}

} // namespace cyxwiz
