#include "pipeline_materializer.h"

#include "node_executors/pipeline_operator.h"
#include "node_executors/pipeline_operator_factory.h"

#include <arrow/table.h>
#include <spdlog/spdlog.h>

#include <map>
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

bool IsFoldedTextConfigNode(gui::NodeType type) {
    return type == gui::NodeType::TextVocabulary ||
           type == gui::NodeType::TextPadding;
}

void FoldTextConfigNodeParams(
    const gui::MLNode& config_node,
    std::map<std::string, std::string>& params) {

    if (config_node.type == gui::NodeType::TextVocabulary) {
        auto min_freq = config_node.parameters.find("min_freq");
        if (min_freq != config_node.parameters.end()) {
            params["min_word_freq"] = min_freq->second;
        }
        auto max_vocab = config_node.parameters.find("max_vocab_size");
        if (max_vocab != config_node.parameters.end()) {
            params["max_vocab_size"] = max_vocab->second;
        }
        auto vocab_file = config_node.parameters.find("vocab_file");
        if (vocab_file != config_node.parameters.end()) {
            params["vocab_file"] = vocab_file->second;
        }
    } else if (config_node.type == gui::NodeType::TextPadding) {
        auto max_length = config_node.parameters.find("max_length");
        if (max_length != config_node.parameters.end()) {
            params["max_length"] = max_length->second;
        }
        // pad_value is intentionally not folded in v1. TextTokenizer uses
        // the tokenizer PAD id, which is fixed at 0.
    }
}

void FoldReachableTextConfigParams(
    const gui::MLNode& tokenizer_node,
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    std::map<std::string, std::string>& params) {

    std::queue<int> queue;
    std::unordered_set<int> visited;
    queue.push(tokenizer_node.id);
    visited.insert(tokenizer_node.id);

    while (!queue.empty()) {
        const int node_id = queue.front();
        queue.pop();

        for (const auto& link : links) {
            if (link.from_node != node_id ||
                !visited.insert(link.to_node).second) {
                continue;
            }

            const gui::MLNode* child = FindNodeById(link.to_node, nodes);
            if (!child || !IsFoldedTextConfigNode(child->type)) {
                continue;
            }

            FoldTextConfigNodeParams(*child, params);
            queue.push(child->id);
        }
    }
}

std::map<std::string, std::string> BuildOperatorParams(
    const gui::MLNode& node,
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links) {

    auto params = node.parameters;
    if (node.type == gui::NodeType::TextTokenizer) {
        FoldReachableTextConfigParams(node, nodes, links, params);
    }
    return params;
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
            auto params = BuildOperatorParams(*node, nodes, links);
            if (!op->Configure(params, err)) {
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
