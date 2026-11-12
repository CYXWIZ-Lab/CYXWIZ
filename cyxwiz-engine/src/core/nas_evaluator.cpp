#include "nas_evaluator.h"
#include "node_metadata_registry.h"
#include "model_analyzer.h"
#include <spdlog/spdlog.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <set>

namespace cyxwiz {

namespace {

const NodeMetadata* MetadataFor(gui::NodeType type) {
    auto& registry = NodeMetadataRegistry::Instance();
    if (!registry.IsInitialized()) registry.Initialize();
    return registry.GetMetadata(type);
}

// A node of the model itself (a layer, activation, merge, ...), not the data
// pipeline around it (Data Input, Split, DataLoader, Normalize) or the loss.
bool IsModelNode(gui::NodeType type) {
    const NodeMetadata* metadata = MetadataFor(type);
    if (!metadata) return false;
    switch (metadata->category) {
        case gui::NodeCategory::Layers:
        case gui::NodeCategory::Activation:
        case gui::NodeCategory::Pooling:
        case gui::NodeCategory::Normalization:
        case gui::NodeCategory::Attention:
        case gui::NodeCategory::Recurrent:
        case gui::NodeCategory::ShapeOps:
        case gui::NodeCategory::MergeOps:
        case gui::NodeCategory::Upsampling:
        case gui::NodeCategory::Regularization:
            return true;
        default:
            return false;
    }
}

// NAS shapes carry the batch first ([1, features], [1, C, H, W]); the
// analysis takes one sample as the model receives it ([features], [H, W, C]).
std::vector<size_t> SampleShape(const std::vector<size_t>& input_shape) {
    if (input_shape.size() == 4) return {input_shape[2], input_shape[3], input_shape[1]};
    if (input_shape.size() >= 2) return {input_shape.begin() + 1, input_shape.end()};
    return input_shape;
}

}  // namespace

// ============================================================================
// Utility Methods
// ============================================================================

int NASEvaluator::GetNextNodeId(const std::vector<gui::MLNode>& nodes) {
    int max_id = 0;
    for (const auto& node : nodes) {
        max_id = std::max(max_id, node.id);
    }
    return max_id + 1;
}

int NASEvaluator::GetNextPinId(const std::vector<gui::MLNode>& nodes) {
    int max_id = 0;
    for (const auto& node : nodes) {
        for (const auto& pin : node.inputs) {
            max_id = std::max(max_id, pin.id);
        }
        for (const auto& pin : node.outputs) {
            max_id = std::max(max_id, pin.id);
        }
    }
    return max_id + 1;
}

int NASEvaluator::GetNextLinkId(const std::vector<gui::NodeLink>& links) {
    int max_id = 0;
    for (const auto& link : links) {
        max_id = std::max(max_id, link.id);
    }
    return max_id + 1;
}

bool NASEvaluator::IsTrainableLayer(gui::NodeType type) {
    switch (type) {
        case gui::NodeType::Dense:
        case gui::NodeType::Conv1D:
        case gui::NodeType::Conv2D:
        case gui::NodeType::Conv3D:
        case gui::NodeType::DepthwiseConv2D:
        case gui::NodeType::RNN:
        case gui::NodeType::LSTM:
        case gui::NodeType::GRU:
        case gui::NodeType::Embedding:
        case gui::NodeType::BatchNorm:
        case gui::NodeType::LayerNorm:
        case gui::NodeType::GroupNorm:
        case gui::NodeType::InstanceNorm:
        case gui::NodeType::SqueezeExcitation:
        case gui::NodeType::MultiHeadAttention:
        case gui::NodeType::TransformerEncoder:
        case gui::NodeType::TransformerDecoder:
            return true;
        default:
            return false;
    }
}

bool NASEvaluator::IsActivation(gui::NodeType type) {
    switch (type) {
        case gui::NodeType::ReLU:
        case gui::NodeType::LeakyReLU:
        case gui::NodeType::PReLU:
        case gui::NodeType::ELU:
        case gui::NodeType::SELU:
        case gui::NodeType::GELU:
        case gui::NodeType::Swish:
        case gui::NodeType::Mish:
        case gui::NodeType::Hardswish:
        case gui::NodeType::Hardsigmoid:
        case gui::NodeType::ReLU6:
        case gui::NodeType::Sigmoid:
        case gui::NodeType::Tanh:
        case gui::NodeType::Softmax:
            return true;
        default:
            return false;
    }
}

gui::NodeType NASEvaluator::GetRandomLayerType(std::mt19937& rng, gui::NodeType neighbour, gui::NodeType except) {
    // Shape-keeping choices only: a convolution stays a convolution of the
    // same kind (no Flatten or pooling that would change what follows).
    const bool spatial = neighbour == gui::NodeType::Conv1D || neighbour == gui::NodeType::Conv2D ||
                         neighbour == gui::NodeType::Conv3D || neighbour == gui::NodeType::DepthwiseConv2D;
    std::vector<gui::NodeType> types = {spatial ? neighbour : gui::NodeType::Dense, gui::NodeType::BatchNorm,
                                        gui::NodeType::Dropout};
    std::erase(types, except);
    std::uniform_int_distribution<int> dist(0, static_cast<int>(types.size()) - 1);
    return types[dist(rng)];
}

gui::NodeType NASEvaluator::GetRandomActivation(std::mt19937& rng) {
    std::vector<gui::NodeType> types = {
        gui::NodeType::ReLU,
        gui::NodeType::LeakyReLU,
        gui::NodeType::GELU,
        gui::NodeType::Swish,
        gui::NodeType::Mish
    };
    std::uniform_int_distribution<int> dist(0, static_cast<int>(types.size()) - 1);
    return types[dist(rng)];
}

// ============================================================================
// Architecture Scoring
// ============================================================================

ArchitectureScore NASEvaluator::ScoreArchitecture(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    const std::vector<size_t>& input_shape,
    const NASSearchConfig& config)
{
    ArchitectureScore score;

    if (nodes.empty()) {
        score.success = false;
        score.error_message = "Empty architecture";
        return score;
    }

    // Count layers and types
    for (const auto& node : nodes) {
        if (IsTrainableLayer(node.type)) {
            score.trainable_layer_count++;
            score.layer_type_counts[node.type]++;
        }
        if (IsModelNode(node.type)) score.layer_count++;
    }

    // Parameters and FLOPs layer by layer, each layer sized by what reaches
    // it (Model Summary's analysis).
    const ModelAnalysis analysis = ModelAnalyzer{}.AnalyzeGraph(nodes, links, 1, SampleShape(input_shape));
    score.trainable_params = analysis.trainable_parameters;
    score.total_flops = analysis.total_flops;
    score.total_params = score.trainable_params;

    // Calculate complexity score (prefer smaller models)
    // Log scale: 1M params -> 0.5, 10M -> 0.25, 100K -> 0.75
    if (score.total_params > 0) {
        double log_params = std::log10(static_cast<double>(score.total_params));
        score.complexity_score = std::max(0.0, 1.0 - log_params / 8.0);  // 8 = log10(100M)
    } else {
        score.complexity_score = 1.0;
    }

    // Efficiency score (FLOPs per parameter - prefer efficient models)
    if (score.total_params > 0 && score.total_flops > 0) {
        double flops_per_param = static_cast<double>(score.total_flops) / score.total_params;
        // Typical range: 1-100, normalize
        score.efficiency_score = std::min(1.0, flops_per_param / 50.0);
    }

    // Depth score (moderate depth is good)
    // Peak at ~10 layers, penalty for too shallow or too deep
    int optimal_depth = 10;
    double depth_diff = std::abs(score.trainable_layer_count - optimal_depth);
    score.depth_score = std::max(0.0, 1.0 - depth_diff / 20.0);

    // Diversity score (variety of layer types is good)
    int unique_types = static_cast<int>(score.layer_type_counts.size());
    score.diversity_score = std::min(1.0, unique_types / 5.0);

    // Connectivity score (skip connections are good): links drawn as skips,
    // and merge nodes (Add, Concatenate, ...) that join two or more paths,
    // the way the canvas builds residual and dense blocks.
    int skip_connections = 0;
    std::map<int, int> incoming;
    for (const auto& link : links) {
        if (link.type == gui::LinkType::ResidualSkip ||
            link.type == gui::LinkType::DenseSkip) {
            skip_connections++;
        }
        incoming[link.to_node]++;
    }
    for (const auto& node : nodes) {
        const NodeMetadata* metadata = MetadataFor(node.type);
        if (metadata && metadata->category == gui::NodeCategory::MergeOps && incoming[node.id] >= 2) {
            skip_connections++;
        }
    }
    score.skip_connections = skip_connections;
    score.connectivity_score = std::min(1.0, skip_connections / 3.0);

    // Overall score (weighted combination)
    score.overall_score =
        config.weight_complexity * score.complexity_score +
        config.weight_efficiency * score.efficiency_score +
        config.weight_depth * score.depth_score +
        config.weight_diversity * score.diversity_score +
        config.weight_connectivity * score.connectivity_score;

    // Generate summary
    score.architecture_summary = DescribeArchitecture(nodes, links);
    score.success = true;

    return score;
}

std::vector<int> NASEvaluator::ModelChain(const std::vector<gui::MLNode>& nodes,
                                          const std::vector<gui::NodeLink>& links) {
    // Kahn's order over the links, ties in the nodes' order.
    std::map<int, int> index_of;
    for (size_t i = 0; i < nodes.size(); ++i) index_of[nodes[i].id] = static_cast<int>(i);
    std::vector<int> waiting(nodes.size(), 0);
    std::vector<std::vector<int>> next(nodes.size());
    for (const auto& link : links) {
        const auto from = index_of.find(link.from_node);
        const auto to = index_of.find(link.to_node);
        if (from == index_of.end() || to == index_of.end() || from->second == to->second) continue;
        next[from->second].push_back(to->second);
        ++waiting[to->second];
    }
    std::set<int> ready;
    for (size_t i = 0; i < nodes.size(); ++i)
        if (waiting[i] == 0) ready.insert(static_cast<int>(i));
    std::vector<int> order;
    std::vector<bool> placed(nodes.size(), false);
    while (order.size() < nodes.size()) {
        int index = -1;
        if (!ready.empty()) {
            index = *ready.begin();
            ready.erase(ready.begin());
        } else {
            // A cycle: take the first node not placed yet.
            for (size_t i = 0; i < nodes.size() && index < 0; ++i)
                if (!placed[i]) index = static_cast<int>(i);
        }
        placed[index] = true;
        order.push_back(index);
        for (int to : next[index])
            if (!placed[to] && --waiting[to] == 0) ready.insert(to);
    }
    std::vector<int> chain;
    for (int index : order)
        if (IsModelNode(nodes[index].type)) chain.push_back(index);
    return chain;
}

std::string NASEvaluator::LayerKind(const gui::MLNode& node) {
    const NodeMetadata* metadata = MetadataFor(node.type);
    return metadata ? metadata->name : node.name;
}

std::string NASEvaluator::LayerText(const gui::MLNode& node) {
    const std::string text = LayerKind(node);
    for (const char* size : {"units", "filters", "out_channels", "hidden_size"}) {
        const auto it = node.parameters.find(size);
        if (it != node.parameters.end() && !it->second.empty()) return text + " " + it->second;
    }
    return text;
}

std::string NASEvaluator::DescribeArchitecture(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links)
{
    std::stringstream ss;

    // Model layers by type, in the order they first appear; the data pipeline
    // and the loss are counted in the node total only.
    std::vector<std::pair<std::string, int>> counts;
    for (const auto& node : nodes) {
        if (!IsModelNode(node.type)) continue;
        const NodeMetadata* metadata = MetadataFor(node.type);
        const std::string name = metadata ? metadata->name : node.name;
        auto it = std::find_if(counts.begin(), counts.end(), [&](const auto& c) { return c.first == name; });
        if (it == counts.end()) counts.emplace_back(name, 1);
        else ++it->second;
    }

    ss << nodes.size() << " nodes, " << links.size() << " links\n";
    ss << "Layers: ";
    if (counts.empty()) ss << "none";
    for (size_t i = 0; i < counts.size(); ++i) {
        if (i) ss << ", ";
        ss << counts[i].second << "x " << counts[i].first;
    }

    return ss.str();
}

// ============================================================================
// Architecture Validation
// ============================================================================

std::pair<bool, std::string> NASEvaluator::ValidateArchitecture(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links)
{
    if (nodes.empty()) {
        return {false, "Empty architecture"};
    }

    // Check for input node
    bool has_input = false;
    bool has_output = false;

    for (const auto& node : nodes) {
        if (node.type == gui::NodeType::DatasetInput) has_input = true;
        if (node.type == gui::NodeType::Output) has_output = true;
    }

    if (!has_input) {
        return {false, "Missing DatasetInput node"};
    }

    if (!has_output) {
        return {false, "Missing Output node"};
    }

    // Check connectivity (all nodes should be reachable)
    std::set<int> connected_nodes;
    for (const auto& link : links) {
        connected_nodes.insert(link.from_node);
        connected_nodes.insert(link.to_node);
    }

    if (connected_nodes.size() < nodes.size() - 1) {
        return {false, "Some nodes are not connected"};
    }

    return {true, ""};
}

// ============================================================================
// Architecture Mutation
// ============================================================================

gui::MLNode NASEvaluator::CreateLayerNode(
    gui::NodeType type,
    int node_id,
    int& pin_id,
    int units)
{
    gui::MLNode node;
    node.id = node_id;
    node.type = type;

    // Input pin
    gui::NodePin input_pin;
    input_pin.id = pin_id++;
    input_pin.type = gui::PinType::Tensor;
    input_pin.name = "in";
    input_pin.is_input = true;
    node.inputs.push_back(input_pin);

    // Output pin
    gui::NodePin output_pin;
    output_pin.id = pin_id++;
    output_pin.type = gui::PinType::Tensor;
    output_pin.name = "out";
    output_pin.is_input = false;
    node.outputs.push_back(output_pin);

    // Set default parameters based on type
    switch (type) {
        case gui::NodeType::Dense:
            node.name = "Dense";
            node.parameters["units"] = std::to_string(units);
            break;
        case gui::NodeType::Conv2D:
            node.name = "Conv2D";
            node.parameters["filters"] = std::to_string(units);
            node.parameters["kernel_size"] = "3";
            node.parameters["padding"] = "same";
            break;
        case gui::NodeType::MaxPool2D:
            node.name = "MaxPool2D";
            node.parameters["pool_size"] = "2";
            break;
        case gui::NodeType::BatchNorm:
            node.name = "BatchNorm";
            break;
        case gui::NodeType::Dropout:
            node.name = "Dropout";
            node.parameters["rate"] = "0.25";
            break;
        case gui::NodeType::Flatten:
            node.name = "Flatten";
            break;
        case gui::NodeType::ReLU:
            node.name = "ReLU";
            break;
        case gui::NodeType::GELU:
            node.name = "GELU";
            break;
        default:
            node.name = LayerKind(node);  // its name in the Nodes panel ("Mish")
            break;
    }

    return node;
}

int NASEvaluator::FindOutputNode(const std::vector<gui::MLNode>& nodes) {
    for (size_t i = 0; i < nodes.size(); i++) {
        if (nodes[i].type == gui::NodeType::Output) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

std::vector<int> NASEvaluator::FindTrainableLayers(const std::vector<gui::MLNode>& nodes) {
    std::vector<int> indices;
    for (size_t i = 0; i < nodes.size(); i++) {
        if (IsTrainableLayer(nodes[i].type)) {
            indices.push_back(static_cast<int>(i));
        }
    }
    return indices;
}

bool NASEvaluator::FeedsTrainableLayer(const std::vector<gui::MLNode>& nodes,
                                       const std::vector<gui::NodeLink>& links, int node_id) {
    std::map<int, const gui::MLNode*> by_id;
    for (const auto& node : nodes) by_id[node.id] = &node;
    // Walk downstream until a trainable layer is met.
    std::set<int> seen;
    std::vector<int> frontier = {node_id};
    while (!frontier.empty()) {
        const int id = frontier.back();
        frontier.pop_back();
        for (const auto& link : links) {
            if (link.from_node != id || !seen.insert(link.to_node).second) continue;
            const auto next = by_id.find(link.to_node);
            if (next == by_id.end()) continue;
            if (IsTrainableLayer(next->second->type)) return true;
            frontier.push_back(link.to_node);
        }
    }
    return false;
}

std::vector<int> NASEvaluator::FindBodyLayers(const std::vector<gui::MLNode>& nodes,
                                              const std::vector<gui::NodeLink>& links) {
    std::vector<int> body;
    for (const int index : FindTrainableLayers(nodes))
        if (FeedsTrainableLayer(nodes, links, nodes[index].id)) body.push_back(index);
    return body;
}

std::pair<std::vector<gui::MLNode>, std::vector<gui::NodeLink>> NASEvaluator::MutateArchitecture(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    MutationType mutation,
    unsigned int seed)
{
    if (nodes.empty()) {
        return {nodes, links};
    }

    std::mt19937 rng(seed == 0 ? std::random_device{}() : seed);
    std::vector<gui::MLNode> mutated_nodes = nodes;
    std::vector<gui::NodeLink> mutated_links = links;

    // If random, pick a mutation type
    if (mutation == MutationType::Random) {
        std::vector<MutationType> types = {
            MutationType::AddLayer,
            MutationType::RemoveLayer,
            MutationType::SwapLayer,
            MutationType::ChangeUnits,
            MutationType::ChangeActivation
        };
        std::uniform_int_distribution<int> dist(0, static_cast<int>(types.size()) - 1);
        mutation = types[dist(rng)];
    }

    std::vector<int> trainable = FindTrainableLayers(mutated_nodes);
    // Remove, resize and swap leave the head alone: it sets the output size.
    const std::vector<int> body = FindBodyLayers(mutated_nodes, mutated_links);

    switch (mutation) {
        case MutationType::AddLayer: {
            // Add a new layer before a random trainable layer
            if (!trainable.empty()) {
                std::uniform_int_distribution<int> dist(0, static_cast<int>(trainable.size()) - 1);
                int target_idx = trainable[dist(rng)];

                int new_node_id = GetNextNodeId(mutated_nodes);
                int pin_id = GetNextPinId(mutated_nodes);

                gui::NodeType new_type = GetRandomLayerType(rng, mutated_nodes[target_idx].type);
                std::uniform_int_distribution<int> units_dist(32, 256);
                gui::MLNode new_node = CreateLayerNode(new_type, new_node_id, pin_id, units_dist(rng));
                const int target_id = mutated_nodes[target_idx].id;

                // Splice the new layer into the links that fed the target:
                // they feed the new layer, which feeds the target's input.
                int next_link_id = GetNextLinkId(mutated_links);
                std::vector<gui::NodeLink> added;
                for (auto& link : mutated_links) {
                    if (link.to_node != target_id) continue;
                    added.push_back({next_link_id++, new_node_id, new_node.outputs[0].id, target_id, link.to_pin,
                                     link.type});
                    link.to_node = new_node_id;
                    link.to_pin = new_node.inputs[0].id;
                }
                if (added.empty()) break;  // nothing feeds the target: leave the graph as it was
                mutated_links.insert(mutated_links.end(), added.begin(), added.end());
                mutated_nodes.insert(mutated_nodes.begin() + target_idx, new_node);
            }
            break;
        }

        case MutationType::RemoveLayer: {
            // Remove a random trainable layer other than the head
            if (!body.empty()) {
                std::uniform_int_distribution<int> dist(0, static_cast<int>(body.size()) - 1);
                int remove_idx = body[dist(rng)];

                int node_id = mutated_nodes[remove_idx].id;

                // Remove the node
                mutated_nodes.erase(mutated_nodes.begin() + remove_idx);

                // Remove links to/from this node and reconnect: what fed it
                // feeds what it fed (the first feeder, for each consumer).
                const gui::NodeLink* feeder = nullptr;
                for (const auto& link : mutated_links)
                    if (link.to_node == node_id) {
                        feeder = &link;
                        break;
                    }
                std::vector<gui::NodeLink> new_links;
                int next_link_id = GetNextLinkId(mutated_links);
                for (const auto& link : mutated_links) {
                    if (link.from_node == node_id && feeder) {
                        new_links.push_back({next_link_id++, feeder->from_node, feeder->from_pin, link.to_node,
                                             link.to_pin, link.type});
                    } else if (link.from_node != node_id && link.to_node != node_id) {
                        new_links.push_back(link);
                    }
                }
                mutated_links = new_links;
            }
            break;
        }

        case MutationType::ChangeUnits: {
            // Change units/filters of a random trainable layer other than the head
            if (!body.empty()) {
                std::uniform_int_distribution<int> dist(0, static_cast<int>(body.size()) - 1);
                int target_idx = body[dist(rng)];

                auto& node = mutated_nodes[target_idx];

                // Modify units or filters
                if (node.parameters.count("units")) {
                    const int current = std::max(1, std::atoi(node.parameters["units"].c_str()));
                    std::uniform_real_distribution<double> factor(0.5, 2.0);
                    int new_units = static_cast<int>(current * factor(rng));
                    new_units = std::max(16, std::min(1024, new_units));
                    node.parameters["units"] = std::to_string(new_units);
                } else if (node.parameters.count("filters")) {
                    const int current = std::max(1, std::atoi(node.parameters["filters"].c_str()));
                    std::uniform_real_distribution<double> factor(0.5, 2.0);
                    int new_filters = static_cast<int>(current * factor(rng));
                    new_filters = std::max(8, std::min(512, new_filters));
                    node.parameters["filters"] = std::to_string(new_filters);
                }
            }
            break;
        }

        case MutationType::SwapLayer: {
            // Replace a layer other than the head with another type that fits
            // there; it keeps its id and its pins, so its links stay.
            if (!body.empty()) {
                std::uniform_int_distribution<int> dist(0, static_cast<int>(body.size()) - 1);
                auto& node = mutated_nodes[body[dist(rng)]];
                const gui::NodeType new_type = GetRandomLayerType(rng, node.type, node.type);
                int units = 64;
                const auto units_it = node.parameters.find("units");
                if (units_it != node.parameters.end()) units = std::max(1, std::atoi(units_it->second.c_str()));
                int unused_pin = 0;
                const gui::MLNode swapped = CreateLayerNode(new_type, node.id, unused_pin, units);
                node.type = swapped.type;
                node.name = swapped.name;
                node.parameters = swapped.parameters;
            }
            break;
        }

        case MutationType::ChangeActivation: {
            // A hidden activation: one that still feeds a trainable layer. The
            // output activation belongs to the head and the loss.
            std::vector<int> activation_indices;
            for (size_t i = 0; i < mutated_nodes.size(); i++) {
                if (IsActivation(mutated_nodes[i].type) &&
                    FeedsTrainableLayer(mutated_nodes, mutated_links, mutated_nodes[i].id)) {
                    activation_indices.push_back(static_cast<int>(i));
                }
            }

            if (!activation_indices.empty()) {
                std::uniform_int_distribution<int> dist(0, static_cast<int>(activation_indices.size()) - 1);
                int target_idx = activation_indices[dist(rng)];

                gui::NodeType new_activation = GetRandomActivation(rng);
                if (mutated_nodes[target_idx].type != new_activation) mutated_nodes[target_idx].parameters.clear();
                mutated_nodes[target_idx].type = new_activation;

                // Update name
                switch (new_activation) {
                    case gui::NodeType::ReLU: mutated_nodes[target_idx].name = "ReLU"; break;
                    case gui::NodeType::LeakyReLU: mutated_nodes[target_idx].name = "LeakyReLU"; break;
                    case gui::NodeType::GELU: mutated_nodes[target_idx].name = "GELU"; break;
                    case gui::NodeType::Swish: mutated_nodes[target_idx].name = "Swish"; break;
                    case gui::NodeType::Mish: mutated_nodes[target_idx].name = "Mish"; break;
                    default: break;
                }
            }
            break;
        }

        default:
            break;
    }

    return {mutated_nodes, mutated_links};
}

// ============================================================================
// Architecture Suggestions
// ============================================================================

std::vector<std::pair<std::vector<gui::MLNode>, std::vector<gui::NodeLink>>>
NASEvaluator::SuggestArchitectures(
    const std::string& task_type,
    const std::vector<size_t>& input_shape,
    int output_size,
    int num_suggestions)
{
    (void)input_shape;

    std::vector<std::pair<std::vector<gui::MLNode>, std::vector<gui::NodeLink>>> suggestions;

    std::mt19937 rng(42);

    for (int s = 0; s < num_suggestions; s++) {
        std::vector<gui::MLNode> nodes;
        std::vector<gui::NodeLink> links;
        int node_id = 1;
        int pin_id = 1;
        int link_id = 1;

        // Input node
        gui::MLNode input_node;
        input_node.id = node_id++;
        input_node.type = gui::NodeType::DatasetInput;
        input_node.name = "Input";
        gui::NodePin input_out;
        input_out.id = pin_id++;
        input_out.type = gui::PinType::Dataset;
        input_out.name = "data";
        input_out.is_input = false;
        input_node.outputs.push_back(input_out);
        nodes.push_back(input_node);

        int last_output_pin = input_out.id;
        int last_node_id = input_node.id;

        // Different architectures based on task
        if (task_type == "image_classification" || task_type == "image") {
            // CNN architecture
            std::vector<int> filters = {32, 64, 128};
            std::uniform_int_distribution<int> layer_dist(2, 4);
            int num_conv_blocks = layer_dist(rng);

            for (int i = 0; i < num_conv_blocks && i < 3; i++) {
                // Conv2D
                gui::MLNode conv = CreateLayerNode(gui::NodeType::Conv2D, node_id++, pin_id, filters[i]);
                nodes.push_back(conv);

                gui::NodeLink link;
                link.id = link_id++;
                link.from_node = last_node_id;
                link.from_pin = last_output_pin;
                link.to_node = conv.id;
                link.to_pin = conv.inputs[0].id;
                links.push_back(link);

                last_node_id = conv.id;
                last_output_pin = conv.outputs[0].id;

                // ReLU
                gui::MLNode relu = CreateLayerNode(gui::NodeType::ReLU, node_id++, pin_id);
                nodes.push_back(relu);

                link.id = link_id++;
                link.from_node = last_node_id;
                link.from_pin = last_output_pin;
                link.to_node = relu.id;
                link.to_pin = relu.inputs[0].id;
                links.push_back(link);

                last_node_id = relu.id;
                last_output_pin = relu.outputs[0].id;

                // MaxPool
                gui::MLNode pool = CreateLayerNode(gui::NodeType::MaxPool2D, node_id++, pin_id);
                nodes.push_back(pool);

                link.id = link_id++;
                link.from_node = last_node_id;
                link.from_pin = last_output_pin;
                link.to_node = pool.id;
                link.to_pin = pool.inputs[0].id;
                links.push_back(link);

                last_node_id = pool.id;
                last_output_pin = pool.outputs[0].id;
            }

            // Flatten
            gui::MLNode flatten = CreateLayerNode(gui::NodeType::Flatten, node_id++, pin_id);
            nodes.push_back(flatten);

            gui::NodeLink link;
            link.id = link_id++;
            link.from_node = last_node_id;
            link.from_pin = last_output_pin;
            link.to_node = flatten.id;
            link.to_pin = flatten.inputs[0].id;
            links.push_back(link);

            last_node_id = flatten.id;
            last_output_pin = flatten.outputs[0].id;

        } else {
            // MLP architecture for classification/regression
            std::vector<int> units = {256, 128, 64};
            std::uniform_int_distribution<int> layer_dist(2, 4);
            int num_hidden = layer_dist(rng);

            for (int i = 0; i < num_hidden && i < 3; i++) {
                // Dense
                gui::MLNode dense = CreateLayerNode(gui::NodeType::Dense, node_id++, pin_id, units[i]);
                nodes.push_back(dense);

                gui::NodeLink link;
                link.id = link_id++;
                link.from_node = last_node_id;
                link.from_pin = last_output_pin;
                link.to_node = dense.id;
                link.to_pin = dense.inputs[0].id;
                links.push_back(link);

                last_node_id = dense.id;
                last_output_pin = dense.outputs[0].id;

                // Activation
                gui::NodeType act_type = GetRandomActivation(rng);
                gui::MLNode act = CreateLayerNode(act_type, node_id++, pin_id);
                nodes.push_back(act);

                link.id = link_id++;
                link.from_node = last_node_id;
                link.from_pin = last_output_pin;
                link.to_node = act.id;
                link.to_pin = act.inputs[0].id;
                links.push_back(link);

                last_node_id = act.id;
                last_output_pin = act.outputs[0].id;

                // Optional dropout
                std::uniform_real_distribution<double> drop_prob(0, 1);
                if (drop_prob(rng) < 0.3) {
                    gui::MLNode dropout = CreateLayerNode(gui::NodeType::Dropout, node_id++, pin_id);
                    nodes.push_back(dropout);

                    link.id = link_id++;
                    link.from_node = last_node_id;
                    link.from_pin = last_output_pin;
                    link.to_node = dropout.id;
                    link.to_pin = dropout.inputs[0].id;
                    links.push_back(link);

                    last_node_id = dropout.id;
                    last_output_pin = dropout.outputs[0].id;
                }
            }
        }

        // Output Dense
        gui::MLNode output_dense = CreateLayerNode(gui::NodeType::Dense, node_id++, pin_id, output_size);
        nodes.push_back(output_dense);

        gui::NodeLink link;
        link.id = link_id++;
        link.from_node = last_node_id;
        link.from_pin = last_output_pin;
        link.to_node = output_dense.id;
        link.to_pin = output_dense.inputs[0].id;
        links.push_back(link);

        last_node_id = output_dense.id;
        last_output_pin = output_dense.outputs[0].id;

        // Output node
        gui::MLNode output_node;
        output_node.id = node_id++;
        output_node.type = gui::NodeType::Output;
        output_node.name = "Output";
        gui::NodePin output_in;
        output_in.id = pin_id++;
        output_in.type = gui::PinType::Tensor;
        output_in.name = "prediction";
        output_in.is_input = true;
        output_node.inputs.push_back(output_in);
        nodes.push_back(output_node);

        link.id = link_id++;
        link.from_node = last_node_id;
        link.from_pin = last_output_pin;
        link.to_node = output_node.id;
        link.to_pin = output_in.id;
        links.push_back(link);

        suggestions.push_back({nodes, links});
    }

    return suggestions;
}

// ============================================================================
// Evolutionary Search
// ============================================================================

std::pair<std::vector<gui::MLNode>, std::vector<gui::NodeLink>> NASEvaluator::Crossover(
    const std::vector<gui::MLNode>& parent1_nodes,
    const std::vector<gui::NodeLink>& parent1_links,
    const std::vector<gui::MLNode>& parent2_nodes,
    const std::vector<gui::NodeLink>& parent2_links,
    unsigned int seed)
{
    // Keeps one parent whole, chosen at random: splicing halves of two graphs
    // would need their links rewired, which this does not do.
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> prob(0, 1);

    if (prob(rng) < 0.5) {
        return {parent1_nodes, parent1_links};
    } else {
        return {parent2_nodes, parent2_links};
    }
}

NASSearchResult NASEvaluator::EvolveArchitecture(
    const std::vector<gui::MLNode>& initial_nodes,
    const std::vector<gui::NodeLink>& initial_links,
    const std::vector<size_t>& input_shape,
    const NASSearchConfig& config,
    std::function<bool(int, const ArchitectureScore&)> progress_callback)
{
    NASSearchResult result;

    if (initial_nodes.empty()) {
        result.success = false;
        result.error_message = "Empty initial architecture";
        return result;
    }

    spdlog::info("Starting NAS evolution: {} generations, population={}",
                 config.generations, config.population_size);

    std::mt19937 rng(42);

    // Initialize population
    std::vector<std::pair<std::vector<gui::MLNode>, std::vector<gui::NodeLink>>> population;
    population.push_back({initial_nodes, initial_links});

    // Generate rest of population through mutations
    while (population.size() < static_cast<size_t>(config.population_size)) {
        auto [mutated_nodes, mutated_links] = MutateArchitecture(
            initial_nodes, initial_links, MutationType::Random, rng());
        population.push_back({mutated_nodes, mutated_links});
    }

    // The best different architectures seen, keyed by their layers in order.
    std::map<std::string, NASCandidate> seen;
    const auto keep = [&](const std::vector<gui::MLNode>& nodes, const std::vector<gui::NodeLink>& links,
                          const ArchitectureScore& score) {
        std::string key;
        for (int index : ModelChain(nodes, links)) key += LayerText(nodes[index]) + ";";
        auto it = seen.find(key);
        if (it == seen.end() || it->second.score.overall_score < score.overall_score)
            seen[key] = NASCandidate{nodes, links, score};
    };

    for (int gen = 0; gen < config.generations; gen++) {
        // Evaluate population
        std::vector<std::pair<ArchitectureScore, int>> scores;

        for (size_t i = 0; i < population.size(); i++) {
            ArchitectureScore score = ScoreArchitecture(
                population[i].first, population[i].second, input_shape, config);
            if (score.success) keep(population[i].first, population[i].second, score);
            scores.push_back({score, static_cast<int>(i)});
            result.total_evaluations++;
        }

        // Sort by overall score (descending)
        std::sort(scores.begin(), scores.end(),
                  [](const auto& a, const auto& b) {
                      return a.first.overall_score > b.first.overall_score;
                  });
        result.total_generations = gen + 1;

        spdlog::debug("Generation {}: best score = {:.4f}", gen, scores[0].first.overall_score);
        if (progress_callback && !progress_callback(gen, scores[0].first)) {
            result.stopped = true;
            break;
        }

        // Create next generation
        std::vector<std::pair<std::vector<gui::MLNode>, std::vector<gui::NodeLink>>> new_population;

        // Elitism: keep top N
        for (int i = 0; i < config.elite_count && i < static_cast<int>(scores.size()); i++) {
            new_population.push_back(population[scores[i].second]);
        }

        // Fill rest with mutations and crossover
        std::uniform_real_distribution<double> prob(0, 1);

        while (new_population.size() < static_cast<size_t>(config.population_size)) {
            // Tournament selection
            std::uniform_int_distribution<int> select(0, static_cast<int>(scores.size()) / 2);
            int parent1_idx = scores[select(rng)].second;
            int parent2_idx = scores[select(rng)].second;

            auto [child_nodes, child_links] = population[parent1_idx];

            // Crossover
            if (prob(rng) < config.crossover_rate) {
                std::tie(child_nodes, child_links) = Crossover(
                    population[parent1_idx].first, population[parent1_idx].second,
                    population[parent2_idx].first, population[parent2_idx].second,
                    rng());
            }

            // Mutation
            if (prob(rng) < config.mutation_rate) {
                std::tie(child_nodes, child_links) = MutateArchitecture(
                    child_nodes, child_links, MutationType::Random, rng());
            }

            new_population.push_back({child_nodes, child_links});
        }

        population = new_population;
    }

    for (auto& [key, candidate] : seen) result.ranked.push_back(std::move(candidate));
    std::stable_sort(result.ranked.begin(), result.ranked.end(), [](const NASCandidate& a, const NASCandidate& b) {
        return a.score.overall_score > b.score.overall_score;
    });
    if (result.ranked.size() > static_cast<size_t>(kNASRanked)) result.ranked.resize(kNASRanked);
    result.success = !result.ranked.empty();
    if (!result.success) result.error_message = "no version of the architecture could be scored";

    if (result.success)
        spdlog::info("NAS complete: best score = {:.4f}, params = {}", result.ranked.front().score.overall_score,
                     result.ranked.front().score.total_params);

    return result;
}

} // namespace cyxwiz
