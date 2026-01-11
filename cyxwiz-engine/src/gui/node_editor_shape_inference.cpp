#include "node_editor_shape_inference.h"
#include "../core/data_registry.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <queue>
#include <set>

namespace gui {

// ===========================================================================
// Public API
// ===========================================================================

void ShapeInferenceEngine::ComputeAllShapes(
    std::vector<MLNode>& nodes,
    const std::vector<NodeLink>& links)
{
    // Get topological sort order
    auto sorted_ids = TopologicalSortForShapes(nodes, links);
    if (sorted_ids.empty() && !nodes.empty()) {
        spdlog::warn("ShapeInferenceEngine: Cycle detected in graph, cannot compute shapes");
        return;
    }

    // Process nodes in topological order
    for (int node_id : sorted_ids) {
        MLNode* node = FindNode(nodes, node_id);
        if (!node) continue;

        // Collect input shapes from connected nodes
        std::vector<std::vector<size_t>> input_shapes;
        for (const auto& input_pin : node->inputs) {
            // Find the link connected to this input pin
            std::vector<size_t> input_shape;
            for (const auto& link : links) {
                if (link.to_pin == input_pin.id) {
                    // Find the source node's output pin
                    for (auto& src_node : nodes) {
                        for (auto& out_pin : src_node.outputs) {
                            if (out_pin.id == link.from_pin) {
                                if (out_pin.shape_valid) {
                                    input_shape = out_pin.shape;
                                }
                                break;
                            }
                        }
                        if (!input_shape.empty()) break;
                    }
                    break;
                }
            }
            input_shapes.push_back(input_shape);
        }

        // Compute output shape for this node
        std::vector<size_t> output_shape = InferNodeOutputShape(*node, input_shapes);

        // Store shape in all output pins
        for (auto& out_pin : node->outputs) {
            out_pin.shape = output_shape;
            out_pin.shape_valid = !output_shape.empty();
        }

        // Also update input pin shapes (for passthrough cases)
        for (size_t i = 0; i < node->inputs.size() && i < input_shapes.size(); ++i) {
            node->inputs[i].shape = input_shapes[i];
            node->inputs[i].shape_valid = !input_shapes[i].empty();
        }
    }

    shapes_valid_ = true;
}

void ShapeInferenceEngine::InvalidateShapes() {
    shapes_valid_ = false;
}

// ===========================================================================
// Private Helpers
// ===========================================================================

std::vector<size_t> ShapeInferenceEngine::InferNodeOutputShape(
    const MLNode& node,
    const std::vector<std::vector<size_t>>& input_shapes)
{
    std::vector<size_t> output_shape;

    // Get the first input shape (most nodes have single input)
    std::vector<size_t> input_shape;
    if (!input_shapes.empty() && !input_shapes[0].empty()) {
        input_shape = input_shapes[0];
    }

    const auto& params = node.parameters;

    switch (node.type) {
        case NodeType::DatasetInput:
            // Get shape from loaded dataset
            output_shape = GetDatasetInputShape();
            break;

        case NodeType::Dense: {
            // Dense layer: any input -> [units]
            int units = 64;
            if (params.count("units")) {
                units = std::stoi(params.at("units"));
            }
            output_shape = {static_cast<size_t>(units)};
            break;
        }

        case NodeType::Conv2D: {
            // Conv2D: [H, W, C] -> [H', W', filters]
            int filters = 32;
            int kernel_size = 3;
            int stride = 1;
            std::string padding = "same";

            if (params.count("filters")) filters = std::stoi(params.at("filters"));
            if (params.count("kernel_size")) kernel_size = std::stoi(params.at("kernel_size"));
            if (params.count("stride")) stride = std::stoi(params.at("stride"));
            if (params.count("padding")) padding = params.at("padding");

            if (input_shape.size() >= 2) {
                size_t h = input_shape[0];
                size_t w = input_shape[1];

                if (padding == "same") {
                    output_shape = {(h + stride - 1) / stride, (w + stride - 1) / stride, static_cast<size_t>(filters)};
                } else {
                    // valid padding
                    output_shape = {(h - kernel_size) / stride + 1, (w - kernel_size) / stride + 1, static_cast<size_t>(filters)};
                }
            }
            break;
        }

        case NodeType::MaxPool2D:
        case NodeType::AvgPool2D: {
            // Pooling: [H, W, C] -> [H/pool, W/pool, C]
            int pool_size = 2;
            int stride = 2;
            if (params.count("pool_size")) pool_size = std::stoi(params.at("pool_size"));
            if (params.count("stride")) stride = std::stoi(params.at("stride"));

            if (input_shape.size() >= 3) {
                output_shape = {
                    input_shape[0] / stride,
                    input_shape[1] / stride,
                    input_shape[2]
                };
            }
            break;
        }

        case NodeType::Flatten: {
            // Flatten: [H, W, C] -> [H*W*C]
            size_t flat_size = 1;
            for (size_t dim : input_shape) flat_size *= dim;
            output_shape = {flat_size};
            break;
        }

        case NodeType::GlobalMaxPool:
        case NodeType::GlobalAvgPool: {
            // Global pooling: [H, W, C] -> [C]
            if (input_shape.size() >= 3) {
                output_shape = {input_shape[2]};  // Keep channel dimension
            }
            break;
        }

        case NodeType::AdaptiveAvgPool: {
            // Adaptive pooling: [H, W, C] -> [target_h, target_w, C]
            int target_h = 1;
            int target_w = 1;
            if (params.count("output_size_h")) target_h = std::stoi(params.at("output_size_h"));
            if (params.count("output_size_w")) target_w = std::stoi(params.at("output_size_w"));

            if (input_shape.size() >= 3) {
                output_shape = {static_cast<size_t>(target_h), static_cast<size_t>(target_w), input_shape[2]};
            }
            break;
        }

        case NodeType::TensorReshape: {
            // Reshape: parse target shape from params
            if (params.count("shape")) {
                std::string shape_str = params.at("shape");
                shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), '['), shape_str.end());
                shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), ']'), shape_str.end());
                shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), ' '), shape_str.end());

                std::vector<int> dims;
                size_t pos = 0;
                while ((pos = shape_str.find(',')) != std::string::npos) {
                    dims.push_back(std::stoi(shape_str.substr(0, pos)));
                    shape_str.erase(0, pos + 1);
                }
                if (!shape_str.empty()) {
                    dims.push_back(std::stoi(shape_str));
                }

                // Handle -1 dimension (infer size)
                size_t total_elements = 1;
                for (size_t dim : input_shape) total_elements *= dim;

                size_t known_size = 1;
                for (int d : dims) {
                    if (d > 0) known_size *= d;
                }

                for (int d : dims) {
                    if (d == -1) {
                        output_shape.push_back(total_elements / known_size);
                    } else {
                        output_shape.push_back(static_cast<size_t>(d));
                    }
                }
            } else {
                output_shape = input_shape;
            }
            break;
        }

        case NodeType::OneHotEncode: {
            // OneHot: scalar -> [num_classes]
            int num_classes = 10;
            if (params.count("num_classes")) {
                num_classes = std::stoi(params.at("num_classes"));
            }
            output_shape = {static_cast<size_t>(num_classes)};
            break;
        }

        case NodeType::Concatenate: {
            // Concatenate: merge shapes along axis
            int axis = -1;  // Last axis by default
            if (params.count("axis")) {
                axis = std::stoi(params.at("axis"));
            }

            if (!input_shapes.empty()) {
                output_shape = input_shapes[0];
                if (!output_shape.empty()) {
                    // Normalize axis
                    if (axis < 0) axis = static_cast<int>(output_shape.size()) + axis;

                    // Sum dimension sizes along concat axis
                    for (size_t i = 1; i < input_shapes.size(); ++i) {
                        if (i < input_shapes.size() && !input_shapes[i].empty()) {
                            if (axis >= 0 && axis < static_cast<int>(output_shape.size())) {
                                output_shape[axis] += input_shapes[i][axis];
                            }
                        }
                    }
                }
            }
            break;
        }

        case NodeType::Add:
        case NodeType::Multiply: {
            // Element-wise ops: use broadcast rules (simplified - assume same shape)
            if (!input_shapes.empty() && !input_shapes[0].empty()) {
                output_shape = input_shapes[0];
            }
            break;
        }

        // Activation functions and normalization - preserve shape
        case NodeType::ReLU:
        case NodeType::LeakyReLU:
        case NodeType::Sigmoid:
        case NodeType::Tanh:
        case NodeType::Softmax:
        case NodeType::BatchNorm:
        case NodeType::LayerNorm:
        case NodeType::GroupNorm:
        case NodeType::InstanceNorm:
        case NodeType::Dropout:
        case NodeType::Normalize:
        case NodeType::Augmentation:
        case NodeType::DataLoader:
        case NodeType::DataSplit:
            output_shape = input_shape;
            break;

        // Loss and output nodes - preserve shape
        case NodeType::Output:
        case NodeType::MSELoss:
        case NodeType::CrossEntropyLoss:
            output_shape = input_shape;
            break;

        default:
            // Unknown node type - pass through input shape
            if (!input_shape.empty()) {
                output_shape = input_shape;
                spdlog::warn("ShapeInferenceEngine: Unknown node type, passing through input shape");
            }
            break;
    }

    return output_shape;
}

std::vector<size_t> ShapeInferenceEngine::GetDatasetInputShape() {
    auto& registry = cyxwiz::DataRegistry::Instance();

    // Try to get shape from currently loaded dataset
    auto datasets = registry.ListDatasets();
    if (!datasets.empty()) {
        // Use the first dataset (or most recently loaded)
        auto handle = registry.GetDataset(datasets[0].name);
        if (handle.IsValid()) {
            auto info = handle.GetInfo();
            if (!info.shape.empty()) {
                return info.shape;
            }
        }
    }

    // Default shape (MNIST as fallback)
    return {28, 28, 1};
}

std::vector<int> ShapeInferenceEngine::TopologicalSortForShapes(
    const std::vector<MLNode>& nodes,
    const std::vector<NodeLink>& links)
{
    // Build adjacency list and in-degree count
    std::map<int, std::vector<int>> adj;  // node_id -> list of dependent node_ids
    std::map<int, int> in_degree;         // node_id -> number of incoming edges

    // Initialize
    for (const auto& node : nodes) {
        in_degree[node.id] = 0;
        adj[node.id] = {};
    }

    // Build graph
    for (const auto& link : links) {
        // Find source and target nodes
        int from_node_id = -1;
        int to_node_id = -1;

        for (const auto& node : nodes) {
            for (const auto& pin : node.outputs) {
                if (pin.id == link.from_pin) {
                    from_node_id = node.id;
                    break;
                }
            }
            for (const auto& pin : node.inputs) {
                if (pin.id == link.to_pin) {
                    to_node_id = node.id;
                    break;
                }
            }
        }

        if (from_node_id != -1 && to_node_id != -1) {
            adj[from_node_id].push_back(to_node_id);
            in_degree[to_node_id]++;
        }
    }

    // Kahn's algorithm for topological sort
    std::queue<int> q;
    for (const auto& [node_id, degree] : in_degree) {
        if (degree == 0) {
            q.push(node_id);
        }
    }

    std::vector<int> result;
    while (!q.empty()) {
        int node_id = q.front();
        q.pop();
        result.push_back(node_id);

        // Decrease in-degree of neighbors
        for (int neighbor : adj[node_id]) {
            in_degree[neighbor]--;
            if (in_degree[neighbor] == 0) {
                q.push(neighbor);
            }
        }
    }

    // Check if all nodes were processed (no cycle)
    if (result.size() != nodes.size()) {
        spdlog::error("ShapeInferenceEngine: Cycle detected in graph");
        return {};  // Return empty to indicate cycle
    }

    return result;
}

// ===========================================================================
// Utility Functions
// ===========================================================================

MLNode* ShapeInferenceEngine::FindNode(std::vector<MLNode>& nodes, int node_id) {
    for (auto& node : nodes) {
        if (node.id == node_id) return &node;
    }
    return nullptr;
}

const MLNode* ShapeInferenceEngine::FindNode(const std::vector<MLNode>& nodes, int node_id) const {
    for (const auto& node : nodes) {
        if (node.id == node_id) return &node;
    }
    return nullptr;
}

NodePin* ShapeInferenceEngine::FindPin(MLNode& node, int pin_id) {
    for (auto& pin : node.inputs) {
        if (pin.id == pin_id) return &pin;
    }
    for (auto& pin : node.outputs) {
        if (pin.id == pin_id) return &pin;
    }
    return nullptr;
}

} // namespace gui
