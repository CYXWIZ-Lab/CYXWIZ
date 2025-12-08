#include "nas_evaluator.h"
#include <spdlog/spdlog.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <set>

namespace cyxwiz {

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
        case gui::NodeType::MultiHeadAttention:
        case gui::NodeType::SelfAttention:
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
        case gui::NodeType::Sigmoid:
        case gui::NodeType::Tanh:
        case gui::NodeType::Softmax:
            return true;
        default:
            return false;
    }
}

gui::NodeType NASEvaluator::GetRandomLayerType(std::mt19937& rng) {
    std::vector<gui::NodeType> types = {
        gui::NodeType::Dense,
        gui::NodeType::Conv2D,
        gui::NodeType::MaxPool2D,
        gui::NodeType::AvgPool2D,
        gui::NodeType::BatchNorm,
        gui::NodeType::Dropout,
        gui::NodeType::Flatten
    };
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

int64_t NASEvaluator::EstimateLayerParams(
    gui::NodeType type,
    const std::map<std::string, std::string>& params,
    const std::vector<size_t>& input_shape)
{
    auto GetParam = [&params](const std::string& key, int default_val) -> int {
        auto it = params.find(key);
        if (it != params.end()) {
            try {
                return std::stoi(it->second);
            } catch (...) {}
        }
        return default_val;
    };

    int64_t param_count = 0;
    size_t in_features = input_shape.empty() ? 64 : input_shape.back();

    switch (type) {
        case gui::NodeType::Dense: {
            int units = GetParam("units", 64);
            param_count = static_cast<int64_t>(in_features) * units + units;  // W + b
            break;
        }
        case gui::NodeType::Conv2D: {
            int filters = GetParam("filters", 32);
            int kernel_size = GetParam("kernel_size", 3);
            size_t in_channels = input_shape.size() > 1 ? input_shape[1] : 3;
            param_count = static_cast<int64_t>(in_channels) * filters * kernel_size * kernel_size + filters;
            break;
        }
        case gui::NodeType::BatchNorm:
        case gui::NodeType::LayerNorm: {
            param_count = static_cast<int64_t>(in_features) * 4;  // gamma, beta, running_mean, running_var
            break;
        }
        case gui::NodeType::LSTM: {
            int units = GetParam("units", 64);
            param_count = 4 * ((static_cast<int64_t>(in_features) + units + 1) * units);  // 4 gates
            break;
        }
        case gui::NodeType::GRU: {
            int units = GetParam("units", 64);
            param_count = 3 * ((static_cast<int64_t>(in_features) + units + 1) * units);  // 3 gates
            break;
        }
        case gui::NodeType::Embedding: {
            int vocab_size = GetParam("vocab_size", 10000);
            int embed_dim = GetParam("embed_dim", 128);
            param_count = static_cast<int64_t>(vocab_size) * embed_dim;
            break;
        }
        case gui::NodeType::MultiHeadAttention:
        case gui::NodeType::SelfAttention: {
            int heads = GetParam("num_heads", 8);
            int dim = GetParam("embed_dim", 64);
            param_count = 4 * static_cast<int64_t>(dim) * dim;  // Q, K, V, O projections
            break;
        }
        default:
            param_count = 0;
            break;
    }

    return param_count;
}

int64_t NASEvaluator::EstimateLayerFLOPs(
    gui::NodeType type,
    const std::map<std::string, std::string>& params,
    const std::vector<size_t>& input_shape)
{
    auto GetParam = [&params](const std::string& key, int default_val) -> int {
        auto it = params.find(key);
        if (it != params.end()) {
            try {
                return std::stoi(it->second);
            } catch (...) {}
        }
        return default_val;
    };

    int64_t flops = 0;
    size_t batch = input_shape.empty() ? 1 : input_shape[0];
    size_t in_features = input_shape.size() > 1 ? input_shape.back() : 64;

    switch (type) {
        case gui::NodeType::Dense: {
            int units = GetParam("units", 64);
            flops = 2 * batch * static_cast<int64_t>(in_features) * units;  // multiply-add
            break;
        }
        case gui::NodeType::Conv2D: {
            int filters = GetParam("filters", 32);
            int kernel_size = GetParam("kernel_size", 3);
            size_t in_channels = input_shape.size() > 1 ? input_shape[1] : 3;
            size_t height = input_shape.size() > 2 ? input_shape[2] : 28;
            size_t width = input_shape.size() > 3 ? input_shape[3] : 28;
            flops = 2 * batch * static_cast<int64_t>(filters) * height * width *
                    in_channels * kernel_size * kernel_size;
            break;
        }
        case gui::NodeType::ReLU:
        case gui::NodeType::LeakyReLU:
        case gui::NodeType::Sigmoid:
        case gui::NodeType::Tanh: {
            flops = batch * static_cast<int64_t>(in_features);
            break;
        }
        case gui::NodeType::GELU:
        case gui::NodeType::Swish:
        case gui::NodeType::Mish: {
            flops = batch * static_cast<int64_t>(in_features) * 5;  // More complex activation
            break;
        }
        default:
            flops = 0;
            break;
    }

    return flops;
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

            // Estimate parameters
            score.trainable_params += EstimateLayerParams(node.type, node.parameters, input_shape);
            score.total_flops += EstimateLayerFLOPs(node.type, node.parameters, input_shape);
        }
        score.layer_count++;
    }

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

    // Connectivity score (skip connections are good)
    // Count links that skip layers
    int skip_connections = 0;
    for (const auto& link : links) {
        if (link.type == gui::LinkType::ResidualSkip ||
            link.type == gui::LinkType::DenseSkip) {
            skip_connections++;
        }
    }
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

std::string NASEvaluator::DescribeArchitecture(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links)
{
    std::stringstream ss;

    // Count layer types
    std::map<gui::NodeType, int> counts;
    for (const auto& node : nodes) {
        counts[node.type]++;
    }

    ss << nodes.size() << " nodes, " << links.size() << " links\n";
    ss << "Layers: ";

    bool first = true;
    for (const auto& [type, count] : counts) {
        if (!first) ss << ", ";
        first = false;

        // Get type name
        switch (type) {
            case gui::NodeType::Dense: ss << count << "x Dense"; break;
            case gui::NodeType::Conv2D: ss << count << "x Conv2D"; break;
            case gui::NodeType::MaxPool2D: ss << count << "x MaxPool"; break;
            case gui::NodeType::BatchNorm: ss << count << "x BatchNorm"; break;
            case gui::NodeType::Dropout: ss << count << "x Dropout"; break;
            case gui::NodeType::ReLU: ss << count << "x ReLU"; break;
            case gui::NodeType::LSTM: ss << count << "x LSTM"; break;
            case gui::NodeType::Flatten: ss << count << "x Flatten"; break;
            default: ss << count << "x Other"; break;
        }
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

    if (!has_output) {
        return {false, "Missing Output node"};
    }

    // Check connectivity (all nodes should be reachable)
    std::set<int> connected_nodes;
    for (const auto& link : links) {
        connected_nodes.insert(link.from_node);
        connected_nodes.insert(link.to_node);
    }

    // Single node graphs are valid if it's an output
    if (nodes.size() == 1 && nodes[0].type == gui::NodeType::Output) {
        return {true, ""};
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
            node.name = "Layer";
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
            MutationType::ChangeUnits,
            MutationType::ChangeActivation
        };
        std::uniform_int_distribution<int> dist(0, static_cast<int>(types.size()) - 1);
        mutation = types[dist(rng)];
    }

    std::vector<int> trainable = FindTrainableLayers(mutated_nodes);

    switch (mutation) {
        case MutationType::AddLayer: {
            // Add a new layer before a random trainable layer
            if (!trainable.empty()) {
                std::uniform_int_distribution<int> dist(0, static_cast<int>(trainable.size()) - 1);
                int target_idx = trainable[dist(rng)];

                int new_node_id = GetNextNodeId(mutated_nodes);
                int pin_id = GetNextPinId(mutated_nodes);

                gui::NodeType new_type = GetRandomLayerType(rng);
                std::uniform_int_distribution<int> units_dist(32, 256);
                gui::MLNode new_node = CreateLayerNode(new_type, new_node_id, pin_id, units_dist(rng));

                mutated_nodes.insert(mutated_nodes.begin() + target_idx, new_node);

                // TODO: Update links to include new node
            }
            break;
        }

        case MutationType::RemoveLayer: {
            // Remove a random trainable layer (if more than 1)
            if (trainable.size() > 1) {
                std::uniform_int_distribution<int> dist(0, static_cast<int>(trainable.size()) - 1);
                int remove_idx = trainable[dist(rng)];

                int node_id = mutated_nodes[remove_idx].id;

                // Remove the node
                mutated_nodes.erase(mutated_nodes.begin() + remove_idx);

                // Remove links to/from this node and reconnect
                std::vector<gui::NodeLink> new_links;
                for (const auto& link : mutated_links) {
                    if (link.from_node != node_id && link.to_node != node_id) {
                        new_links.push_back(link);
                    }
                }
                mutated_links = new_links;
            }
            break;
        }

        case MutationType::ChangeUnits: {
            // Change units/filters of a random trainable layer
            if (!trainable.empty()) {
                std::uniform_int_distribution<int> dist(0, static_cast<int>(trainable.size()) - 1);
                int target_idx = trainable[dist(rng)];

                auto& node = mutated_nodes[target_idx];

                // Modify units or filters
                if (node.parameters.count("units")) {
                    int current = std::stoi(node.parameters["units"]);
                    std::uniform_real_distribution<double> factor(0.5, 2.0);
                    int new_units = static_cast<int>(current * factor(rng));
                    new_units = std::max(16, std::min(1024, new_units));
                    node.parameters["units"] = std::to_string(new_units);
                } else if (node.parameters.count("filters")) {
                    int current = std::stoi(node.parameters["filters"]);
                    std::uniform_real_distribution<double> factor(0.5, 2.0);
                    int new_filters = static_cast<int>(current * factor(rng));
                    new_filters = std::max(8, std::min(512, new_filters));
                    node.parameters["filters"] = std::to_string(new_filters);
                }
            }
            break;
        }

        case MutationType::ChangeActivation: {
            // Find activation nodes and change one
            std::vector<int> activation_indices;
            for (size_t i = 0; i < mutated_nodes.size(); i++) {
                if (IsActivation(mutated_nodes[i].type)) {
                    activation_indices.push_back(static_cast<int>(i));
                }
            }

            if (!activation_indices.empty()) {
                std::uniform_int_distribution<int> dist(0, static_cast<int>(activation_indices.size()) - 1);
                int target_idx = activation_indices[dist(rng)];

                gui::NodeType new_activation = GetRandomActivation(rng);
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
    // Simple crossover: take first half from parent1, second half from parent2
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
    std::function<void(int, const ArchitectureScore&)> progress_callback)
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

    // Evolutionary loop
    ArchitectureScore best_score;
    best_score.overall_score = -1;

    for (int gen = 0; gen < config.generations; gen++) {
        // Evaluate population
        std::vector<std::pair<ArchitectureScore, int>> scores;

        for (size_t i = 0; i < population.size(); i++) {
            ArchitectureScore score = ScoreArchitecture(
                population[i].first, population[i].second, input_shape, config);
            scores.push_back({score, static_cast<int>(i)});
            result.all_scores.push_back(score);
            result.total_evaluations++;
        }

        // Sort by overall score (descending)
        std::sort(scores.begin(), scores.end(),
                  [](const auto& a, const auto& b) {
                      return a.first.overall_score > b.first.overall_score;
                  });

        // Track best
        if (scores[0].first.overall_score > best_score.overall_score) {
            best_score = scores[0].first;
            result.best_architecture = population[scores[0].second].first;
            result.best_links = population[scores[0].second].second;
        }

        result.generation_best.push_back(scores[0].first);

        if (progress_callback) {
            progress_callback(gen, scores[0].first);
        }

        spdlog::debug("Generation {}: best score = {:.4f}", gen, scores[0].first.overall_score);

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

    result.best_score = best_score;
    result.total_generations = config.generations;
    result.success = true;

    spdlog::info("NAS complete: best score = {:.4f}, params = {}",
                 best_score.overall_score, best_score.total_params);

    return result;
}

} // namespace cyxwiz
