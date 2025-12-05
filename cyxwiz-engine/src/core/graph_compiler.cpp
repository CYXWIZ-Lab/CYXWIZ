#include "graph_compiler.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <queue>
#include <set>
#include <stack>

namespace cyxwiz {

TrainingConfiguration GraphCompiler::Compile(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links)
{
    TrainingConfiguration config;

    // Validate graph first
    if (!ValidateGraph(nodes, links, config.error_message)) {
        config.is_valid = false;
        return config;
    }

    // Find key nodes
    const gui::MLNode* dataset_node = FindDatasetInputNode(nodes);
    const gui::MLNode* loss_node = FindLossNode(nodes);
    const gui::MLNode* optimizer_node = FindOptimizerNode(nodes);

    // Extract dataset configuration
    if (dataset_node) {
        config.dataset_name = dataset_node->parameters.count("dataset")
            ? dataset_node->parameters.at("dataset") : "";

        // Extract input shape from dataset node
        if (dataset_node->parameters.count("shape")) {
            // Parse shape string like "[28, 28, 1]"
            std::string shape_str = dataset_node->parameters.at("shape");
            // Simple parsing - remove brackets and split by comma
            shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), '['), shape_str.end());
            shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), ']'), shape_str.end());
            shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), ' '), shape_str.end());

            size_t pos = 0;
            while ((pos = shape_str.find(',')) != std::string::npos) {
                config.input_shape.push_back(std::stoul(shape_str.substr(0, pos)));
                shape_str.erase(0, pos + 1);
            }
            if (!shape_str.empty()) {
                config.input_shape.push_back(std::stoul(shape_str));
            }
        }

        // Calculate flattened input size
        config.input_size = 1;
        for (size_t dim : config.input_shape) {
            config.input_size *= dim;
        }
    }

    // Extract split ratios from DataSplit node if present
    for (const auto& node : nodes) {
        if (node.type == gui::NodeType::DataSplit) {
            if (node.parameters.count("train_ratio"))
                config.train_ratio = std::stof(node.parameters.at("train_ratio"));
            if (node.parameters.count("val_ratio"))
                config.val_ratio = std::stof(node.parameters.at("val_ratio"));
            if (node.parameters.count("test_ratio"))
                config.test_ratio = std::stof(node.parameters.at("test_ratio"));
            break;
        }
    }

    // Get topologically sorted node IDs
    std::vector<int> sorted_ids = TopologicalSort(nodes, links);

    // Extract model layers and preprocessing in execution order
    std::vector<size_t> current_shape = config.input_shape;

    for (int node_id : sorted_ids) {
        const gui::MLNode* node = FindNodeById(node_id, nodes);
        if (!node) continue;

        // Handle preprocessing nodes
        if (IsPreprocessing(node->type)) {
            ExtractPreprocessing(*node, config.preprocessing);
            continue;
        }

        // Handle model layers and activations
        if (IsModelLayer(node->type) || IsActivation(node->type)) {
            CompiledLayer layer = ExtractLayerConfig(*node);
            layer.input_shape = current_shape;

            // Infer output shape
            layer.output_shape = InferOutputShape(layer, current_shape);
            current_shape = layer.output_shape;

            config.layers.push_back(layer);

            // Track output size from last Dense layer
            if (node->type == gui::NodeType::Dense) {
                config.output_size = layer.units;
            }
        }
    }

    // Extract loss configuration
    if (loss_node) {
        config.loss_type = loss_node->type;
        config.loss_params = loss_node->parameters;
    }

    // Extract optimizer configuration
    if (optimizer_node) {
        config.optimizer_type = optimizer_node->type;

        if (optimizer_node->parameters.count("learning_rate"))
            config.learning_rate = std::stof(optimizer_node->parameters.at("learning_rate"));
        if (optimizer_node->parameters.count("lr"))
            config.learning_rate = std::stof(optimizer_node->parameters.at("lr"));
        if (optimizer_node->parameters.count("momentum"))
            config.momentum = std::stof(optimizer_node->parameters.at("momentum"));
        if (optimizer_node->parameters.count("beta1"))
            config.beta1 = std::stof(optimizer_node->parameters.at("beta1"));
        if (optimizer_node->parameters.count("beta2"))
            config.beta2 = std::stof(optimizer_node->parameters.at("beta2"));
        if (optimizer_node->parameters.count("weight_decay"))
            config.weight_decay = std::stof(optimizer_node->parameters.at("weight_decay"));
    }

    // Set one-hot encoding if we have classification (CrossEntropy loss)
    if (config.loss_type == gui::NodeType::CrossEntropyLoss) {
        config.preprocessing.has_onehot = true;
        config.preprocessing.num_classes = config.output_size;
    }

    config.is_valid = true;
    spdlog::info("GraphCompiler: Compiled {} layers, input_size={}, output_size={}",
                 config.layers.size(), config.input_size, config.output_size);

    return config;
}

bool GraphCompiler::ValidateGraph(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    std::string& error)
{
    // Check for empty graph
    if (nodes.empty()) {
        error = "Graph is empty - add nodes to create a model";
        return false;
    }

    // Check for dataset input
    if (!FindDatasetInputNode(nodes)) {
        error = "Graph must have a DatasetInput node";
        return false;
    }

    // Check for at least one model layer
    bool has_model_layer = false;
    for (const auto& node : nodes) {
        if (IsModelLayer(node.type)) {
            has_model_layer = true;
            break;
        }
    }
    if (!has_model_layer) {
        error = "Graph must have at least one model layer (Dense, Conv2D, etc.)";
        return false;
    }

    // Check for loss function
    if (!FindLossNode(nodes)) {
        error = "Graph must have a loss function (MSELoss or CrossEntropyLoss)";
        return false;
    }

    // Check for optimizer
    if (!FindOptimizerNode(nodes)) {
        error = "Graph must have an optimizer (SGD, Adam, or AdamW)";
        return false;
    }

    // Check for cycles
    if (HasCycle(nodes, links)) {
        error = "Graph contains a cycle - remove circular connections";
        return false;
    }

    return true;
}

std::vector<int> GraphCompiler::TopologicalSort(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links)
{
    // Build adjacency list and in-degree count
    std::map<int, std::vector<int>> adj;
    std::map<int, int> in_degree;

    for (const auto& node : nodes) {
        adj[node.id] = {};
        in_degree[node.id] = 0;
    }

    for (const auto& link : links) {
        adj[link.from_node].push_back(link.to_node);
        in_degree[link.to_node]++;
    }

    // Kahn's algorithm
    std::queue<int> queue;
    for (const auto& node : nodes) {
        if (in_degree[node.id] == 0) {
            queue.push(node.id);
        }
    }

    std::vector<int> sorted;
    while (!queue.empty()) {
        int node_id = queue.front();
        queue.pop();
        sorted.push_back(node_id);

        for (int neighbor : adj[node_id]) {
            in_degree[neighbor]--;
            if (in_degree[neighbor] == 0) {
                queue.push(neighbor);
            }
        }
    }

    return sorted;
}

const gui::MLNode* GraphCompiler::FindNodeById(int id, const std::vector<gui::MLNode>& nodes) const {
    for (const auto& node : nodes) {
        if (node.id == id) return &node;
    }
    return nullptr;
}

std::vector<int> GraphCompiler::GetConnectedNodes(
    int from_node_id,
    const std::vector<gui::NodeLink>& links) const
{
    std::vector<int> connected;
    for (const auto& link : links) {
        if (link.from_node == from_node_id) {
            connected.push_back(link.to_node);
        }
    }
    return connected;
}

std::vector<int> GraphCompiler::GetInputNodes(
    int to_node_id,
    const std::vector<gui::NodeLink>& links) const
{
    std::vector<int> inputs;
    for (const auto& link : links) {
        if (link.to_node == to_node_id) {
            inputs.push_back(link.from_node);
        }
    }
    return inputs;
}

const gui::MLNode* GraphCompiler::FindDatasetInputNode(const std::vector<gui::MLNode>& nodes) const {
    for (const auto& node : nodes) {
        if (node.type == gui::NodeType::DatasetInput) {
            return &node;
        }
    }
    return nullptr;
}

const gui::MLNode* GraphCompiler::FindLossNode(const std::vector<gui::MLNode>& nodes) const {
    for (const auto& node : nodes) {
        if (node.type == gui::NodeType::MSELoss ||
            node.type == gui::NodeType::CrossEntropyLoss ||
            node.type == gui::NodeType::BCELoss ||
            node.type == gui::NodeType::BCEWithLogits ||
            node.type == gui::NodeType::L1Loss ||
            node.type == gui::NodeType::SmoothL1Loss ||
            node.type == gui::NodeType::HuberLoss ||
            node.type == gui::NodeType::NLLLoss) {
            return &node;
        }
    }
    return nullptr;
}

const gui::MLNode* GraphCompiler::FindOptimizerNode(const std::vector<gui::MLNode>& nodes) const {
    for (const auto& node : nodes) {
        if (node.type == gui::NodeType::SGD ||
            node.type == gui::NodeType::Adam ||
            node.type == gui::NodeType::AdamW) {
            return &node;
        }
    }
    return nullptr;
}

const gui::MLNode* GraphCompiler::FindOutputNode(const std::vector<gui::MLNode>& nodes) const {
    for (const auto& node : nodes) {
        if (node.type == gui::NodeType::Output) {
            return &node;
        }
    }
    return nullptr;
}

bool GraphCompiler::IsModelLayer(gui::NodeType type) const {
    switch (type) {
        case gui::NodeType::Dense:
        case gui::NodeType::Conv2D:
        case gui::NodeType::MaxPool2D:
        case gui::NodeType::AvgPool2D:
        case gui::NodeType::GlobalMaxPool:
        case gui::NodeType::GlobalAvgPool:
        case gui::NodeType::Flatten:
        case gui::NodeType::Dropout:
        case gui::NodeType::BatchNorm:
            return true;
        default:
            return false;
    }
}

bool GraphCompiler::IsActivation(gui::NodeType type) const {
    switch (type) {
        case gui::NodeType::ReLU:
        case gui::NodeType::LeakyReLU:
        case gui::NodeType::ELU:
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

bool GraphCompiler::IsPreprocessing(gui::NodeType type) const {
    switch (type) {
        case gui::NodeType::Normalize:
        case gui::NodeType::TensorReshape:
        case gui::NodeType::OneHotEncode:
        case gui::NodeType::DataSplit:
        case gui::NodeType::DataLoader:
        case gui::NodeType::Augmentation:
            return true;
        default:
            return false;
    }
}

CompiledLayer GraphCompiler::ExtractLayerConfig(const gui::MLNode& node) const {
    CompiledLayer layer;
    layer.type = node.type;
    layer.node_id = node.id;
    layer.name = node.name;
    layer.parameters = node.parameters;

    // Extract specific parameters
    switch (node.type) {
        case gui::NodeType::Dense:
            if (node.parameters.count("units"))
                layer.units = std::stoi(node.parameters.at("units"));
            break;

        case gui::NodeType::Conv2D:
            if (node.parameters.count("filters"))
                layer.filters = std::stoi(node.parameters.at("filters"));
            if (node.parameters.count("kernel_size"))
                layer.kernel_size = std::stoi(node.parameters.at("kernel_size"));
            if (node.parameters.count("stride"))
                layer.stride = std::stoi(node.parameters.at("stride"));
            if (node.parameters.count("padding"))
                layer.padding = std::stoi(node.parameters.at("padding"));
            break;

        case gui::NodeType::MaxPool2D:
        case gui::NodeType::AvgPool2D:
            if (node.parameters.count("pool_size"))
                layer.pool_size = std::stoi(node.parameters.at("pool_size"));
            if (node.parameters.count("stride"))
                layer.stride = std::stoi(node.parameters.at("stride"));
            break;

        case gui::NodeType::BatchNorm:
            if (node.parameters.count("eps"))
                layer.eps = std::stof(node.parameters.at("eps"));
            if (node.parameters.count("momentum"))
                layer.momentum = std::stof(node.parameters.at("momentum"));
            break;

        case gui::NodeType::Dropout:
            if (node.parameters.count("rate"))
                layer.dropout_rate = std::stof(node.parameters.at("rate"));
            break;

        case gui::NodeType::LeakyReLU:
            if (node.parameters.count("negative_slope"))
                layer.negative_slope = std::stof(node.parameters.at("negative_slope"));
            break;

        case gui::NodeType::ELU:
            if (node.parameters.count("alpha"))
                layer.alpha = std::stof(node.parameters.at("alpha"));
            break;

        default:
            break;
    }

    return layer;
}

void GraphCompiler::ExtractPreprocessing(
    const gui::MLNode& node,
    PreprocessingConfig& config) const
{
    switch (node.type) {
        case gui::NodeType::Normalize:
            config.has_normalization = true;
            if (node.parameters.count("mean"))
                config.norm_mean = std::stof(node.parameters.at("mean"));
            if (node.parameters.count("std"))
                config.norm_std = std::stof(node.parameters.at("std"));
            break;

        case gui::NodeType::TensorReshape:
            config.has_reshape = true;
            // Parse reshape dimensions from parameter
            if (node.parameters.count("shape")) {
                std::string shape_str = node.parameters.at("shape");
                shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), '['), shape_str.end());
                shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), ']'), shape_str.end());
                shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), ' '), shape_str.end());

                size_t pos = 0;
                while ((pos = shape_str.find(',')) != std::string::npos) {
                    config.reshape_dims.push_back(std::stoi(shape_str.substr(0, pos)));
                    shape_str.erase(0, pos + 1);
                }
                if (!shape_str.empty()) {
                    config.reshape_dims.push_back(std::stoi(shape_str));
                }
            }
            break;

        case gui::NodeType::OneHotEncode:
            config.has_onehot = true;
            if (node.parameters.count("num_classes"))
                config.num_classes = std::stoul(node.parameters.at("num_classes"));
            break;

        default:
            break;
    }
}

std::vector<size_t> GraphCompiler::InferOutputShape(
    const CompiledLayer& layer,
    const std::vector<size_t>& input_shape) const
{
    std::vector<size_t> output_shape;

    switch (layer.type) {
        case gui::NodeType::Dense:
            // Dense: [...] -> [units]
            output_shape = {static_cast<size_t>(layer.units)};
            break;

        case gui::NodeType::Conv2D:
            // Conv2D: [H, W, C] -> [(H + 2*padding - kernel_size) / stride + 1, W', filters]
            if (input_shape.size() >= 2) {
                size_t out_h = (input_shape[0] + 2 * layer.padding - layer.kernel_size) / layer.stride + 1;
                size_t out_w = (input_shape[1] + 2 * layer.padding - layer.kernel_size) / layer.stride + 1;
                output_shape = {out_h, out_w, static_cast<size_t>(layer.filters)};
            }
            break;

        case gui::NodeType::MaxPool2D:
        case gui::NodeType::AvgPool2D:
            // Pool2D: [H, W, C] -> [H/pool_size, W/pool_size, C]
            if (input_shape.size() >= 3) {
                int stride = layer.stride > 0 ? layer.stride : layer.pool_size;
                size_t out_h = (input_shape[0] - layer.pool_size) / stride + 1;
                size_t out_w = (input_shape[1] - layer.pool_size) / stride + 1;
                output_shape = {out_h, out_w, input_shape[2]};
            }
            break;

        case gui::NodeType::GlobalMaxPool:
        case gui::NodeType::GlobalAvgPool:
            // Global pooling: [H, W, C] -> [C]
            if (input_shape.size() >= 3) {
                output_shape = {input_shape[2]};  // Just channels remain
            } else if (!input_shape.empty()) {
                output_shape = {input_shape.back()};
            }
            break;

        case gui::NodeType::Flatten:
            // Flatten: [H, W, C] -> [H*W*C]
            {
                size_t flat_size = 1;
                for (size_t dim : input_shape) flat_size *= dim;
                output_shape = {flat_size};
            }
            break;

        // Layers/activations that don't change shape
        case gui::NodeType::Dropout:
        case gui::NodeType::BatchNorm:
        case gui::NodeType::ReLU:
        case gui::NodeType::LeakyReLU:
        case gui::NodeType::ELU:
        case gui::NodeType::GELU:
        case gui::NodeType::Swish:
        case gui::NodeType::Mish:
        case gui::NodeType::Sigmoid:
        case gui::NodeType::Tanh:
        case gui::NodeType::Softmax:
            output_shape = input_shape;
            break;

        default:
            output_shape = input_shape;
            break;
    }

    return output_shape;
}

bool GraphCompiler::HasCycle(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links) const
{
    // Build adjacency list
    std::map<int, std::vector<int>> adj;
    for (const auto& node : nodes) {
        adj[node.id] = {};
    }
    for (const auto& link : links) {
        adj[link.from_node].push_back(link.to_node);
    }

    // DFS-based cycle detection
    std::set<int> white;  // Not visited
    std::set<int> gray;   // Currently visiting
    std::set<int> black;  // Fully processed

    for (const auto& node : nodes) {
        white.insert(node.id);
    }

    std::function<bool(int)> dfs = [&](int node_id) -> bool {
        white.erase(node_id);
        gray.insert(node_id);

        for (int neighbor : adj[node_id]) {
            if (gray.count(neighbor)) {
                // Back edge found - cycle exists
                return true;
            }
            if (white.count(neighbor) && dfs(neighbor)) {
                return true;
            }
        }

        gray.erase(node_id);
        black.insert(node_id);
        return false;
    };

    while (!white.empty()) {
        int node_id = *white.begin();
        if (dfs(node_id)) {
            return true;
        }
    }

    return false;
}

bool GraphCompiler::IsFullyConnected(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links) const
{
    if (nodes.empty()) return true;

    // BFS from first node
    std::set<int> visited;
    std::queue<int> queue;

    // Build undirected adjacency
    std::map<int, std::set<int>> adj;
    for (const auto& node : nodes) {
        adj[node.id] = {};
    }
    for (const auto& link : links) {
        adj[link.from_node].insert(link.to_node);
        adj[link.to_node].insert(link.from_node);
    }

    queue.push(nodes[0].id);
    visited.insert(nodes[0].id);

    while (!queue.empty()) {
        int node_id = queue.front();
        queue.pop();

        for (int neighbor : adj[node_id]) {
            if (!visited.count(neighbor)) {
                visited.insert(neighbor);
                queue.push(neighbor);
            }
        }
    }

    return visited.size() == nodes.size();
}

} // namespace cyxwiz
