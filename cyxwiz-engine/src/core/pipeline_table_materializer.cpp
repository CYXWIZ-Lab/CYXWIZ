#include "pipeline_materializer.h"

#include "node_executors/pipeline_operator.h"
#include "node_executors/pipeline_operator_factory.h"

#include <arrow/table.h>
#include <spdlog/spdlog.h>

#include <map>
#include <queue>
#include <unordered_set>
#include <vector>

namespace cyxwiz {

namespace {

bool IsDataInputNode(const gui::MLNode& node) {
    return node.type == gui::NodeType::DataInput ||
           node.type == gui::NodeType::DatasetInput;
}

std::string DatasetNameForNode(const gui::MLNode& node) {
    auto dataset_name = node.parameters.find("dataset_name");
    if (dataset_name != node.parameters.end() && !dataset_name->second.empty()) {
        return dataset_name->second;
    }

    auto legacy_dataset = node.parameters.find("dataset");
    if (legacy_dataset != node.parameters.end() && !legacy_dataset->second.empty()) {
        return legacy_dataset->second;
    }

    return {};
}

const gui::MLNode* FindDataInputNode(
    const std::vector<gui::MLNode>& nodes,
    const std::string& source_dataset_name) {

    if (!source_dataset_name.empty()) {
        for (const auto& n : nodes) {
            if (IsDataInputNode(n) &&
                DatasetNameForNode(n) == source_dataset_name) {
                return &n;
            }
        }
    }

    for (const auto& n : nodes) {
        if (IsDataInputNode(n)) {
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

bool HasReachableMaterializerOperator(
    int node_id,
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    const PipelineOperatorFactory& factory,
    std::unordered_set<int>& visiting) {

    if (!visiting.insert(node_id).second) {
        return false;
    }

    const gui::MLNode* node = FindNodeById(node_id, nodes);
    if (!node) {
        return false;
    }

    if (factory.HasOperator(node->type)) {
        return true;
    }

    for (const auto& link : links) {
        if (link.from_node != node_id) {
            continue;
        }
        if (HasReachableMaterializerOperator(
                link.to_node, nodes, links, factory, visiting)) {
            return true;
        }
    }

    return false;
}

bool ValidateLinearMaterializerOperatorPath(
    const gui::MLNode& data_input,
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    const PipelineOperatorFactory& factory,
    std::string& error) {

    std::queue<int> queue;
    std::unordered_set<int> visited;
    queue.push(data_input.id);
    visited.insert(data_input.id);

    while (!queue.empty()) {
        const int node_id = queue.front();
        queue.pop();

        std::vector<int> operator_relevant_children;
        for (const auto& link : links) {
            if (link.from_node != node_id) {
                continue;
            }

            std::unordered_set<int> visiting;
            if (HasReachableMaterializerOperator(
                    link.to_node, nodes, links, factory, visiting)) {
                operator_relevant_children.push_back(link.to_node);
            }

            if (visited.insert(link.to_node).second) {
                queue.push(link.to_node);
            }
        }

        if (operator_relevant_children.size() > 1) {
            const gui::MLNode* node = FindNodeById(node_id, nodes);
            const std::string node_name =
                node ? node->name : std::to_string(node_id);
            error = "PipelineMaterializer: branched operator paths from node '" +
                    node_name +
                    "' are not supported by the Arrow-table materializer";
            return false;
        }
    }

    return true;
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
    const std::shared_ptr<arrow::Table>& source_table,
    const std::string& source_dataset_name) {

    MaterializeTableResult result;
    result.table = source_table;

    if (!source_table) {
        result.success = false;
        result.error_message = "PipelineMaterializer: source Arrow table is null";
        return result;
    }

    const gui::MLNode* data_input = FindDataInputNode(nodes, source_dataset_name);
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

    std::string graph_shape_error;
    if (!ValidateLinearMaterializerOperatorPath(
            *data_input, nodes, links, factory, graph_shape_error)) {
        result.success = false;
        result.error_message = graph_shape_error;
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
