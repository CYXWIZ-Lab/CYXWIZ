#include "properties.h"
#include "node_editor.h"
#include "../core/data_registry.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <cmath>
#include <queue>
#include <set>
#include <algorithm>

namespace gui {

Properties::Properties() : show_window_(true) {
}

Properties::~Properties() = default;

void Properties::SetSelectedNode(MLNode* node) {
    selected_node_ = node;
}

void Properties::ClearSelection() {
    selected_node_ = nullptr;
}

std::string Properties::FormatShape(const std::vector<size_t>& shape) {
    if (shape.empty()) return "[]";
    std::string result = "[";
    for (size_t i = 0; i < shape.size(); i++) {
        if (i > 0) result += ", ";
        result += std::to_string(shape[i]);
    }
    result += "]";
    return result;
}

size_t Properties::GetBatchSize() {
    // Try to get batch size from DataLoader node in the graph
    if (node_editor_) {
        const auto& nodes = node_editor_->GetNodes();
        for (const auto& node : nodes) {
            if (node.type == NodeType::DataLoader) {
                if (node.parameters.count("batch_size")) {
                    try {
                        return std::stoul(node.parameters.at("batch_size"));
                    } catch (...) {}
                }
            }
        }
    }
    // Default batch size
    return 32;
}

std::string Properties::FormatShapeMatrix(const std::vector<size_t>& shape, size_t batch_size) {
    if (shape.empty()) return "Scalar";

    // Calculate flattened feature count (N)
    size_t N = 1;
    for (size_t d : shape) N *= d;

    std::string batch_str = std::to_string(batch_size);

    if (shape.size() == 1) {
        // 1D: Already flattened features
        // Display as: batch x N (batch x features)
        return batch_str + " x " + std::to_string(shape[0]);
    }
    if (shape.size() == 2) {
        // 2D: [H, W] - grayscale image without channel
        // Unroll to N = H * W
        return batch_str + " x " + std::to_string(N) +
               "  (" + std::to_string(shape[0]) + " x " + std::to_string(shape[1]) + " unrolled)";
    }
    if (shape.size() == 3) {
        // 3D: [H, W, C] - image with channels
        // Unroll to N = H * W * C
        return batch_str + " x " + std::to_string(N) +
               "  (" + std::to_string(shape[0]) + " x " + std::to_string(shape[1]) +
               " x " + std::to_string(shape[2]) + " unrolled)";
    }
    // 4D+: Higher dimensional tensor
    std::string dims;
    for (size_t i = 0; i < shape.size(); i++) {
        if (i > 0) dims += " x ";
        dims += std::to_string(shape[i]);
    }
    return batch_str + " x " + std::to_string(N) + "  (" + dims + " unrolled)";
}

std::vector<size_t> Properties::GetInputShapeFromDataset() {
    // First, try to get the shape from the loaded dataset in DataRegistry
    auto& registry = cyxwiz::DataRegistry::Instance();
    auto datasets = registry.ListDatasets();

    if (!datasets.empty()) {
        // Get the first loaded dataset's shape
        const auto& info = datasets[0];
        if (!info.shape.empty()) {
            spdlog::debug("Properties: Got shape from loaded dataset '{}': {}",
                          info.name, FormatShape(info.shape));
            return info.shape;
        }
    }

    // If no dataset is loaded, check if there's a dataset node with shape parameter
    if (node_editor_) {
        const auto& nodes = node_editor_->GetNodes();
        for (const auto& node : nodes) {
            if (node.type == NodeType::DatasetInput) {
                // Check for dataset name and try to get shape from registry
                if (node.parameters.count("dataset_name")) {
                    const std::string& name = node.parameters.at("dataset_name");
                    if (registry.HasDataset(name)) {
                        auto handle = registry.GetDataset(name);
                        if (handle.IsValid()) {
                            auto info = handle.GetInfo();
                            if (!info.shape.empty()) {
                                return info.shape;
                            }
                        }
                    }
                }

                // Fallback: check for shape parameter in node
                if (node.parameters.count("shape")) {
                    std::string shape_str = node.parameters.at("shape");
                    std::vector<size_t> shape;

                    // Parse shape string like "[28, 28, 1]" or "28,28,1"
                    shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), '['), shape_str.end());
                    shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), ']'), shape_str.end());
                    shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), ' '), shape_str.end());

                    size_t pos = 0;
                    while ((pos = shape_str.find(',')) != std::string::npos) {
                        shape.push_back(std::stoul(shape_str.substr(0, pos)));
                        shape_str.erase(0, pos + 1);
                    }
                    if (!shape_str.empty()) {
                        shape.push_back(std::stoul(shape_str));
                    }

                    if (!shape.empty()) {
                        return shape;
                    }
                }
            }
        }
    }

    // Default MNIST shape as fallback
    return {28, 28, 1};
}

std::vector<size_t> Properties::InferOutputShape(
    NodeType type,
    const std::vector<size_t>& input_shape,
    const std::map<std::string, std::string>& params)
{
    std::vector<size_t> output_shape;

    switch (type) {
        case NodeType::Dense: {
            // Dense layer outputs [units]
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
            std::string padding = "same";

            if (params.count("filters")) filters = std::stoi(params.at("filters"));
            if (params.count("kernel_size")) kernel_size = std::stoi(params.at("kernel_size"));
            if (params.count("padding")) padding = params.at("padding");

            if (input_shape.size() >= 2) {
                size_t h = input_shape[0];
                size_t w = input_shape[1];

                if (padding == "same") {
                    output_shape = {h, w, static_cast<size_t>(filters)};
                } else {
                    // valid padding
                    output_shape = {h - kernel_size + 1, w - kernel_size + 1, static_cast<size_t>(filters)};
                }
            }
            break;
        }

        case NodeType::MaxPool2D: {
            // MaxPool2D: [H, W, C] -> [H/pool, W/pool, C]
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

                // Handle -1 dimension (batch size)
                size_t total_elements = 1;
                for (size_t dim : input_shape) total_elements *= dim;

                int neg_idx = -1;
                size_t known_size = 1;
                for (size_t i = 0; i < dims.size(); i++) {
                    if (dims[i] == -1) {
                        neg_idx = static_cast<int>(i);
                    } else {
                        known_size *= dims[i];
                    }
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

        // These operations don't change shape
        case NodeType::Dropout:
        case NodeType::BatchNorm:
        case NodeType::ReLU:
        case NodeType::Sigmoid:
        case NodeType::Tanh:
        case NodeType::Softmax:
        case NodeType::LeakyReLU:
        case NodeType::Normalize:
        case NodeType::Augmentation:
            output_shape = input_shape;
            break;

        case NodeType::DatasetInput:
            // Output shape is the dataset shape
            output_shape = GetInputShapeFromDataset();
            break;

        case NodeType::DataLoader:
        case NodeType::DataSplit:
            // These pass through the shape
            output_shape = input_shape;
            break;

        case NodeType::OneHotEncode: {
            // OneHot: scalar -> [num_classes]
            int num_classes = 10;
            if (params.count("num_classes")) {
                num_classes = std::stoi(params.at("num_classes"));
            }
            output_shape = {static_cast<size_t>(num_classes)};
            break;
        }

        case NodeType::Output: {
            // Output node - shape is the number of classes
            int classes = 10;
            if (params.count("classes")) {
                classes = std::stoi(params.at("classes"));
            }
            output_shape = {static_cast<size_t>(classes)};
            break;
        }

        // Loss and optimizer nodes don't have meaningful tensor shapes
        case NodeType::MSELoss:
        case NodeType::CrossEntropyLoss:
        case NodeType::SGD:
        case NodeType::Adam:
        case NodeType::AdamW:
            output_shape = {1};  // Scalar output (loss value)
            break;

        default:
            output_shape = input_shape;
            break;
    }

    return output_shape;
}

LayerParameters Properties::ComputeLayerParameters(
    NodeType type,
    const std::vector<size_t>& input_shape,
    const std::map<std::string, std::string>& params)
{
    LayerParameters layer_params;

    // Calculate flattened input size
    size_t input_features = 1;
    for (size_t d : input_shape) input_features *= d;

    switch (type) {
        case NodeType::Dense: {
            // Dense: weight = [input_features, units], bias = [units]
            int units = 64;
            if (params.count("units")) {
                try { units = std::stoi(params.at("units")); } catch (...) {}
            }

            layer_params.weight_shape = {input_features, static_cast<size_t>(units)};
            layer_params.bias_shape = {static_cast<size_t>(units)};
            layer_params.weight_count = input_features * units;
            layer_params.bias_count = units;
            layer_params.total_params = layer_params.weight_count + layer_params.bias_count;
            layer_params.has_parameters = true;
            break;
        }

        case NodeType::Conv2D: {
            // Conv2D: weight = [kernel_h, kernel_w, in_channels, filters], bias = [filters]
            int filters = 32;
            int kernel_size = 3;
            if (params.count("filters")) {
                try { filters = std::stoi(params.at("filters")); } catch (...) {}
            }
            if (params.count("kernel_size")) {
                try { kernel_size = std::stoi(params.at("kernel_size")); } catch (...) {}
            }

            // Input channels from input shape (last dimension for HWC format)
            size_t in_channels = input_shape.size() >= 3 ? input_shape[2] : 1;

            layer_params.weight_shape = {
                static_cast<size_t>(kernel_size),
                static_cast<size_t>(kernel_size),
                in_channels,
                static_cast<size_t>(filters)
            };
            layer_params.bias_shape = {static_cast<size_t>(filters)};
            layer_params.weight_count = kernel_size * kernel_size * in_channels * filters;
            layer_params.bias_count = filters;
            layer_params.total_params = layer_params.weight_count + layer_params.bias_count;
            layer_params.has_parameters = true;
            break;
        }

        case NodeType::BatchNorm: {
            // BatchNorm: gamma = [features], beta = [features], running_mean, running_var
            layer_params.weight_shape = {input_features};  // gamma (scale)
            layer_params.bias_shape = {input_features};    // beta (shift)
            layer_params.weight_count = input_features;
            layer_params.bias_count = input_features;
            layer_params.total_params = layer_params.weight_count + layer_params.bias_count;
            layer_params.has_parameters = true;
            break;
        }

        // Layers without learnable parameters
        case NodeType::ReLU:
        case NodeType::Sigmoid:
        case NodeType::Tanh:
        case NodeType::Softmax:
        case NodeType::LeakyReLU:
        case NodeType::Flatten:
        case NodeType::MaxPool2D:
        case NodeType::Dropout:
        case NodeType::Output:
        case NodeType::DatasetInput:
        case NodeType::DataLoader:
        case NodeType::Augmentation:
        case NodeType::DataSplit:
        case NodeType::TensorReshape:
        case NodeType::Normalize:
        case NodeType::OneHotEncode:
        case NodeType::MSELoss:
        case NodeType::CrossEntropyLoss:
        case NodeType::SGD:
        case NodeType::Adam:
        case NodeType::AdamW:
        default:
            layer_params.has_parameters = false;
            break;
    }

    return layer_params;
}

NodeShapeInfo Properties::ComputeNodeShape(int node_id) {
    NodeShapeInfo info;
    info.is_valid = false;

    if (!node_editor_) {
        info.error = "No graph context";
        return info;
    }

    const auto& nodes = node_editor_->GetNodes();
    const auto& links = node_editor_->GetLinks();

    // Find the node
    const MLNode* target_node = nullptr;
    for (const auto& node : nodes) {
        if (node.id == node_id) {
            target_node = &node;
            break;
        }
    }

    if (!target_node) {
        info.error = "Node not found";
        return info;
    }

    // Build topological order and compute shapes through the graph
    // Use BFS from dataset/input nodes

    // First find all nodes that feed into this node (traverse backwards)
    std::vector<int> predecessors;
    std::set<int> visited;
    std::queue<int> queue;
    queue.push(node_id);
    visited.insert(node_id);

    while (!queue.empty()) {
        int current = queue.front();
        queue.pop();

        // Find all nodes that connect TO this node
        for (const auto& link : links) {
            if (link.to_node == current && visited.find(link.from_node) == visited.end()) {
                visited.insert(link.from_node);
                queue.push(link.from_node);
                predecessors.push_back(link.from_node);
            }
        }
    }

    // Reverse to get topological order (from input to current node)
    std::reverse(predecessors.begin(), predecessors.end());
    predecessors.push_back(node_id);

    // Compute shapes through the chain
    std::map<int, std::vector<size_t>> node_output_shapes;

    for (int nid : predecessors) {
        const MLNode* node = nullptr;
        for (const auto& n : nodes) {
            if (n.id == nid) {
                node = &n;
                break;
            }
        }
        if (!node) continue;

        // Get input shape from predecessor
        std::vector<size_t> input_shape;

        // Find what connects to this node
        for (const auto& link : links) {
            if (link.to_node == nid) {
                if (node_output_shapes.count(link.from_node)) {
                    input_shape = node_output_shapes[link.from_node];
                    break;
                }
            }
        }

        // If no predecessor, this is a source node (Dataset, Input)
        if (input_shape.empty()) {
            if (node->type == NodeType::DatasetInput) {
                input_shape = GetInputShapeFromDataset();
            } else {
                // Try to get from dataset anyway as fallback
                input_shape = GetInputShapeFromDataset();
            }
        }

        // Compute output shape
        std::vector<size_t> output_shape = InferOutputShape(node->type, input_shape, node->parameters);
        node_output_shapes[nid] = output_shape;

        // If this is our target node, set the info
        if (nid == node_id) {
            info.input_shape = input_shape;
            info.output_shape = output_shape;

            // Calculate flattened sizes
            info.input_size = 1;
            for (size_t d : input_shape) info.input_size *= d;

            info.output_size = 1;
            for (size_t d : output_shape) info.output_size *= d;

            // Compute layer parameters (weights, biases)
            info.params = ComputeLayerParameters(node->type, input_shape, node->parameters);

            info.is_valid = true;
        }
    }

    return info;
}

void Properties::RenderShapeInfo(const NodeShapeInfo& shape_info) {
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    size_t batch_size = GetBatchSize();

    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Tensor Shape (batch = %zu)", batch_size);
    ImGui::Spacing();

    if (!shape_info.is_valid) {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "Cannot compute shape");
        if (!shape_info.error.empty()) {
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Error: %s", shape_info.error.c_str());
        }
        return;
    }

    // Calculate memory sizes (assuming Float32 = 4 bytes)
    size_t input_memory = batch_size * shape_info.input_size * sizeof(float);
    size_t output_memory = batch_size * shape_info.output_size * sizeof(float);

    // Input shape section
    ImGui::Text("Input:");
    ImGui::Indent();
    ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "%s", FormatShapeMatrix(shape_info.input_shape, batch_size).c_str());
    if (shape_info.input_shape.size() > 1) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Per-sample: %s", FormatShape(shape_info.input_shape).c_str());
    }
    // Show memory size
    if (input_memory >= 1024 * 1024) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Memory: %.2f MB (%zu elements)",
                           input_memory / (1024.0f * 1024.0f), batch_size * shape_info.input_size);
    } else if (input_memory >= 1024) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Memory: %.2f KB (%zu elements)",
                           input_memory / 1024.0f, batch_size * shape_info.input_size);
    } else {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Memory: %zu bytes (%zu elements)",
                           input_memory, batch_size * shape_info.input_size);
    }
    ImGui::Unindent();

    ImGui::Spacing();

    // Output shape section
    ImGui::Text("Output:");
    ImGui::Indent();
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "%s", FormatShapeMatrix(shape_info.output_shape, batch_size).c_str());
    if (shape_info.output_shape.size() > 1) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Per-sample: %s", FormatShape(shape_info.output_shape).c_str());
    }
    // Show memory size
    if (output_memory >= 1024 * 1024) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Memory: %.2f MB (%zu elements)",
                           output_memory / (1024.0f * 1024.0f), batch_size * shape_info.output_size);
    } else if (output_memory >= 1024) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Memory: %.2f KB (%zu elements)",
                           output_memory / 1024.0f, batch_size * shape_info.output_size);
    } else {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Memory: %zu bytes (%zu elements)",
                           output_memory, batch_size * shape_info.output_size);
    }
    ImGui::Unindent();

    // Shape transformation summary
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1.0f), "Transform: %zu x %zu -> %zu x %zu",
                       batch_size, shape_info.input_size, batch_size, shape_info.output_size);

    // Display learnable parameters if this layer has any
    if (shape_info.params.has_parameters) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), "Learnable Parameters");
        ImGui::Spacing();

        // Weight info
        ImGui::Text("Weight:");
        ImGui::Indent();
        ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "%s", FormatShape(shape_info.params.weight_shape).c_str());
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "%zu parameters", shape_info.params.weight_count);
        size_t weight_memory = shape_info.params.weight_count * sizeof(float);
        if (weight_memory >= 1024 * 1024) {
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Memory: %.2f MB", weight_memory / (1024.0f * 1024.0f));
        } else if (weight_memory >= 1024) {
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Memory: %.2f KB", weight_memory / 1024.0f);
        } else {
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Memory: %zu bytes", weight_memory);
        }
        ImGui::Unindent();

        ImGui::Spacing();

        // Bias info
        ImGui::Text("Bias:");
        ImGui::Indent();
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "%s", FormatShape(shape_info.params.bias_shape).c_str());
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "%zu parameters", shape_info.params.bias_count);
        ImGui::Unindent();

        ImGui::Spacing();
        ImGui::Separator();

        // Total parameters summary
        size_t total_memory = shape_info.params.total_params * sizeof(float);
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.3f, 1.0f), "Total: %zu params", shape_info.params.total_params);
        if (total_memory >= 1024 * 1024) {
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Parameter Memory: %.2f MB", total_memory / (1024.0f * 1024.0f));
        } else if (total_memory >= 1024) {
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Parameter Memory: %.2f KB", total_memory / 1024.0f);
        } else {
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Parameter Memory: %zu bytes", total_memory);
        }
    }
}

void Properties::Render() {
    if (!show_window_) return;

    if (ImGui::Begin("Properties", &show_window_)) {
        ImGui::Text("Node Properties");
        ImGui::Separator();

        if (!selected_node_) {
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "No node selected");
            ImGui::Text("Click on a node in the Node Editor to view its properties");
        } else {
            // Display selected node info
            ImGui::Text("Node: %s", selected_node_->name.c_str());
            ImGui::Text("ID: %d", selected_node_->id);
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // Render node-specific properties
            RenderNodeProperties(*selected_node_);

            // Compute and render shape information
            NodeShapeInfo shape_info = ComputeNodeShape(selected_node_->id);
            RenderShapeInfo(shape_info);
        }
    }
    ImGui::End();
}

void Properties::RenderNodeProperties(MLNode& node) {
    // Render editable parameters based on node type
    switch (node.type) {
        case NodeType::Dense: {
            // Units
            std::string& units = node.parameters["units"];
            if (units.empty()) units = "64";
            char u_buffer[16];
            strncpy(u_buffer, units.c_str(), sizeof(u_buffer) - 1);
            u_buffer[sizeof(u_buffer) - 1] = '\0';

            ImGui::Text("Units:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputText("##units", u_buffer, sizeof(u_buffer), ImGuiInputTextFlags_CharsDecimal)) {
                units = u_buffer;
                InvalidateShapes();
            }

            ImGui::Spacing();

            // Activation function
            std::string& activation = node.parameters["activation"];
            if (activation.empty()) activation = "relu";

            const char* activations[] = { "none", "relu", "sigmoid", "tanh", "softmax", "leaky_relu" };
            int current_activation = 0;
            for (int i = 0; i < 6; i++) {
                if (activation == activations[i]) {
                    current_activation = i;
                    break;
                }
            }

            ImGui::Text("Activation:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(150.0f);
            if (ImGui::Combo("##activation", &current_activation, activations, 6)) {
                activation = activations[current_activation];
            }
            break;
        }

        case NodeType::Conv2D: {
            // Filters
            std::string& filters = node.parameters["filters"];
            if (filters.empty()) filters = "32";
            char f_buffer[16];
            strncpy(f_buffer, filters.c_str(), sizeof(f_buffer) - 1);
            f_buffer[sizeof(f_buffer) - 1] = '\0';

            ImGui::Text("Filters:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputText("##filters", f_buffer, sizeof(f_buffer), ImGuiInputTextFlags_CharsDecimal)) {
                filters = f_buffer;
                InvalidateShapes();
            }

            ImGui::Spacing();

            // Kernel Size
            std::string& kernel = node.parameters["kernel_size"];
            if (kernel.empty()) kernel = "3";
            char k_buffer[16];
            strncpy(k_buffer, kernel.c_str(), sizeof(k_buffer) - 1);
            k_buffer[sizeof(k_buffer) - 1] = '\0';

            ImGui::Text("Kernel Size:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputText("##kernel", k_buffer, sizeof(k_buffer), ImGuiInputTextFlags_CharsDecimal)) {
                kernel = k_buffer;
                InvalidateShapes();
            }

            ImGui::Spacing();

            // Stride
            std::string& stride = node.parameters["stride"];
            if (stride.empty()) stride = "1";
            char s_buffer[16];
            strncpy(s_buffer, stride.c_str(), sizeof(s_buffer) - 1);
            s_buffer[sizeof(s_buffer) - 1] = '\0';

            ImGui::Text("Stride:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputText("##stride", s_buffer, sizeof(s_buffer), ImGuiInputTextFlags_CharsDecimal)) {
                stride = s_buffer;
                InvalidateShapes();
            }

            ImGui::Spacing();

            // Padding
            std::string& padding = node.parameters["padding"];
            if (padding.empty()) padding = "same";

            const char* paddings[] = { "same", "valid" };
            int current_padding = (padding == "valid") ? 1 : 0;

            ImGui::Text("Padding:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(150.0f);
            if (ImGui::Combo("##padding", &current_padding, paddings, 2)) {
                padding = paddings[current_padding];
                InvalidateShapes();
            }

            ImGui::Spacing();

            // Activation function
            std::string& activation = node.parameters["activation"];
            if (activation.empty()) activation = "relu";

            const char* activations[] = { "none", "relu", "sigmoid", "tanh", "softmax", "leaky_relu" };
            int current_activation = 0;
            for (int i = 0; i < 6; i++) {
                if (activation == activations[i]) {
                    current_activation = i;
                    break;
                }
            }

            ImGui::Text("Activation:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(150.0f);
            if (ImGui::Combo("##activation_conv", &current_activation, activations, 6)) {
                activation = activations[current_activation];
            }
            break;
        }

        case NodeType::MaxPool2D: {
            // Pool Size
            std::string& pool_size = node.parameters["pool_size"];
            if (pool_size.empty()) pool_size = "2";
            char p_buffer[16];
            strncpy(p_buffer, pool_size.c_str(), sizeof(p_buffer) - 1);
            p_buffer[sizeof(p_buffer) - 1] = '\0';

            ImGui::Text("Pool Size:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputText("##pool_size", p_buffer, sizeof(p_buffer), ImGuiInputTextFlags_CharsDecimal)) {
                pool_size = p_buffer;
                InvalidateShapes();
            }

            ImGui::Spacing();

            // Stride
            std::string& stride = node.parameters["stride"];
            if (stride.empty()) stride = "2";
            char s_buffer[16];
            strncpy(s_buffer, stride.c_str(), sizeof(s_buffer) - 1);
            s_buffer[sizeof(s_buffer) - 1] = '\0';

            ImGui::Text("Stride:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputText("##stride_pool", s_buffer, sizeof(s_buffer), ImGuiInputTextFlags_CharsDecimal)) {
                stride = s_buffer;
                InvalidateShapes();
            }
            break;
        }

        case NodeType::Dropout: {
            std::string& rate_str = node.parameters["rate"];
            if (rate_str.empty()) rate_str = "0.5";

            float rate = std::stof(rate_str);
            ImGui::Text("Drop Rate:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##rate", &rate, 0.0f, 0.9f, "%.2f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.2f", rate);
                rate_str = buf;
            }
            break;
        }

        case NodeType::BatchNorm: {
            // Momentum
            std::string& momentum_str = node.parameters["momentum"];
            if (momentum_str.empty()) momentum_str = "0.99";

            float momentum = std::stof(momentum_str);
            ImGui::Text("Momentum:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##momentum", &momentum, 0.0f, 1.0f, "%.3f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.3f", momentum);
                momentum_str = buf;
            }

            ImGui::Spacing();

            // Epsilon
            std::string& epsilon_str = node.parameters["epsilon"];
            if (epsilon_str.empty()) epsilon_str = "0.001";

            float epsilon = std::stof(epsilon_str);
            ImGui::Text("Epsilon:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##epsilon", &epsilon, 0.0001f, 0.01f, "%.4f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.4f", epsilon);
                epsilon_str = buf;
            }
            break;
        }

        case NodeType::Output: {
            std::string& classes = node.parameters["classes"];
            if (classes.empty()) classes = "10";
            char c_buffer[16];
            strncpy(c_buffer, classes.c_str(), sizeof(c_buffer) - 1);
            c_buffer[sizeof(c_buffer) - 1] = '\0';

            ImGui::Text("Classes:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputText("##classes", c_buffer, sizeof(c_buffer), ImGuiInputTextFlags_CharsDecimal)) {
                classes = c_buffer;
                InvalidateShapes();
            }
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Number of output classes");
            break;
        }

        // ========== Data Pipeline Nodes ==========

        case NodeType::DatasetInput: {
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 1.0f, 1.0f), "Dataset Input Node");
            ImGui::Separator();
            ImGui::Spacing();

            // Dataset name
            std::string& dataset_name = node.parameters["dataset_name"];
            char name_buffer[128];
            strncpy(name_buffer, dataset_name.c_str(), sizeof(name_buffer) - 1);
            name_buffer[sizeof(name_buffer) - 1] = '\0';

            ImGui::Text("Dataset Name:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::InputText("##dataset_name", name_buffer, sizeof(name_buffer))) {
                dataset_name = name_buffer;
                InvalidateShapes();
            }
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Name in DataRegistry");

            ImGui::Spacing();

            // Show loaded dataset info if available
            auto& registry = cyxwiz::DataRegistry::Instance();
            if (registry.HasDataset(dataset_name)) {
                auto handle = registry.GetDataset(dataset_name);
                if (handle.IsValid()) {
                    auto info = handle.GetInfo();
                    ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "Dataset loaded!");
                    ImGui::Text("Samples: %zu", info.num_samples);
                    ImGui::Text("Classes: %zu", info.num_classes);
                    ImGui::Text("Shape: %s", info.GetShapeString().c_str());
                }
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f), "Dataset not loaded");
            }

            ImGui::Spacing();

            // Split selection
            std::string& split = node.parameters["split"];
            if (split.empty()) split = "train";

            const char* splits[] = { "train", "val", "test" };
            int current_split = 0;
            for (int i = 0; i < 3; i++) {
                if (split == splits[i]) {
                    current_split = i;
                    break;
                }
            }

            ImGui::Text("Split:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(150.0f);
            if (ImGui::Combo("##split", &current_split, splits, 3)) {
                split = splits[current_split];
            }
            break;
        }

        case NodeType::DataLoader: {
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 1.0f, 1.0f), "Data Loader Node");
            ImGui::Separator();
            ImGui::Spacing();

            // Batch size
            std::string& batch_size = node.parameters["batch_size"];
            if (batch_size.empty()) batch_size = "32";
            char batch_buffer[16];
            strncpy(batch_buffer, batch_size.c_str(), sizeof(batch_buffer) - 1);
            batch_buffer[sizeof(batch_buffer) - 1] = '\0';

            ImGui::Text("Batch Size:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputText("##batch_size", batch_buffer, sizeof(batch_buffer), ImGuiInputTextFlags_CharsDecimal)) {
                batch_size = batch_buffer;
            }

            ImGui::Spacing();

            // Shuffle
            std::string& shuffle = node.parameters["shuffle"];
            if (shuffle.empty()) shuffle = "true";
            bool shuffle_val = (shuffle == "true");
            if (ImGui::Checkbox("Shuffle", &shuffle_val)) {
                shuffle = shuffle_val ? "true" : "false";
            }

            // Drop last
            std::string& drop_last = node.parameters["drop_last"];
            if (drop_last.empty()) drop_last = "false";
            bool drop_last_val = (drop_last == "true");
            if (ImGui::Checkbox("Drop Last Batch", &drop_last_val)) {
                drop_last = drop_last_val ? "true" : "false";
            }

            ImGui::Spacing();

            // Num workers
            std::string& num_workers = node.parameters["num_workers"];
            if (num_workers.empty()) num_workers = "4";
            char workers_buffer[16];
            strncpy(workers_buffer, num_workers.c_str(), sizeof(workers_buffer) - 1);
            workers_buffer[sizeof(workers_buffer) - 1] = '\0';

            ImGui::Text("Num Workers:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputText("##num_workers", workers_buffer, sizeof(workers_buffer), ImGuiInputTextFlags_CharsDecimal)) {
                num_workers = workers_buffer;
            }
            break;
        }

        case NodeType::Augmentation: {
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 1.0f, 1.0f), "Augmentation Node");
            ImGui::Separator();
            ImGui::Spacing();

            // Transforms
            std::string& transforms = node.parameters["transforms"];
            if (transforms.empty()) transforms = "RandomFlip,Normalize";
            char transform_buffer[256];
            strncpy(transform_buffer, transforms.c_str(), sizeof(transform_buffer) - 1);
            transform_buffer[sizeof(transform_buffer) - 1] = '\0';

            ImGui::Text("Transforms:");
            ImGui::SetNextItemWidth(250.0f);
            if (ImGui::InputText("##transforms", transform_buffer, sizeof(transform_buffer))) {
                transforms = transform_buffer;
            }
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Comma-separated list");

            ImGui::Spacing();

            // Flip probability
            std::string& flip_prob_str = node.parameters["flip_prob"];
            if (flip_prob_str.empty()) flip_prob_str = "0.5";
            float flip_prob = std::stof(flip_prob_str);

            ImGui::Text("Flip Probability:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##flip_prob", &flip_prob, 0.0f, 1.0f, "%.2f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.2f", flip_prob);
                flip_prob_str = buf;
            }

            ImGui::Spacing();

            // Normalize mean
            std::string& mean = node.parameters["normalize_mean"];
            if (mean.empty()) mean = "0.0";
            char mean_buffer[32];
            strncpy(mean_buffer, mean.c_str(), sizeof(mean_buffer) - 1);
            mean_buffer[sizeof(mean_buffer) - 1] = '\0';

            ImGui::Text("Normalize Mean:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(100.0f);
            if (ImGui::InputText("##norm_mean", mean_buffer, sizeof(mean_buffer))) {
                mean = mean_buffer;
            }

            // Normalize std
            std::string& std_val = node.parameters["normalize_std"];
            if (std_val.empty()) std_val = "1.0";
            char std_buffer[32];
            strncpy(std_buffer, std_val.c_str(), sizeof(std_buffer) - 1);
            std_buffer[sizeof(std_buffer) - 1] = '\0';

            ImGui::Text("Normalize Std:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(100.0f);
            if (ImGui::InputText("##norm_std", std_buffer, sizeof(std_buffer))) {
                std_val = std_buffer;
            }
            break;
        }

        case NodeType::DataSplit: {
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 1.0f, 1.0f), "Data Split Node");
            ImGui::Separator();
            ImGui::Spacing();

            // Train ratio
            std::string& train_ratio_str = node.parameters["train_ratio"];
            if (train_ratio_str.empty()) train_ratio_str = "0.8";
            float train_ratio = std::stof(train_ratio_str);

            ImGui::Text("Train Ratio:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##train_ratio", &train_ratio, 0.0f, 1.0f, "%.2f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.2f", train_ratio);
                train_ratio_str = buf;
            }

            // Validation ratio
            std::string& val_ratio_str = node.parameters["val_ratio"];
            if (val_ratio_str.empty()) val_ratio_str = "0.1";
            float val_ratio = std::stof(val_ratio_str);

            ImGui::Text("Validation Ratio:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##val_ratio", &val_ratio, 0.0f, 1.0f, "%.2f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.2f", val_ratio);
                val_ratio_str = buf;
            }

            // Test ratio
            std::string& test_ratio_str = node.parameters["test_ratio"];
            if (test_ratio_str.empty()) test_ratio_str = "0.1";
            float test_ratio = std::stof(test_ratio_str);

            ImGui::Text("Test Ratio:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##test_ratio", &test_ratio, 0.0f, 1.0f, "%.2f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.2f", test_ratio);
                test_ratio_str = buf;
            }

            // Show total
            float total = train_ratio + val_ratio + test_ratio;
            ImVec4 total_color = (std::abs(total - 1.0f) < 0.01f) ? ImVec4(0.0f, 1.0f, 0.0f, 1.0f) : ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
            ImGui::TextColored(total_color, "Total: %.2f (should be 1.0)", total);

            ImGui::Spacing();

            // Stratified
            std::string& stratified = node.parameters["stratified"];
            if (stratified.empty()) stratified = "true";
            bool stratified_val = (stratified == "true");
            if (ImGui::Checkbox("Stratified Split", &stratified_val)) {
                stratified = stratified_val ? "true" : "false";
            }
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Maintain class distribution");

            ImGui::Spacing();

            // Seed
            std::string& seed = node.parameters["seed"];
            if (seed.empty()) seed = "42";
            char seed_buffer[16];
            strncpy(seed_buffer, seed.c_str(), sizeof(seed_buffer) - 1);
            seed_buffer[sizeof(seed_buffer) - 1] = '\0';

            ImGui::Text("Random Seed:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(100.0f);
            if (ImGui::InputText("##seed", seed_buffer, sizeof(seed_buffer), ImGuiInputTextFlags_CharsDecimal)) {
                seed = seed_buffer;
            }
            break;
        }

        case NodeType::TensorReshape: {
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 1.0f, 1.0f), "Reshape Node");
            ImGui::Separator();
            ImGui::Spacing();

            // Shape
            std::string& shape = node.parameters["shape"];
            if (shape.empty()) shape = "-1,28,28,1";
            char shape_buffer[64];
            strncpy(shape_buffer, shape.c_str(), sizeof(shape_buffer) - 1);
            shape_buffer[sizeof(shape_buffer) - 1] = '\0';

            ImGui::Text("Target Shape:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::InputText("##reshape", shape_buffer, sizeof(shape_buffer))) {
                shape = shape_buffer;
                InvalidateShapes();
            }
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Use -1 for batch dimension");
            break;
        }

        case NodeType::Normalize: {
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 1.0f, 1.0f), "Normalize Node");
            ImGui::Separator();
            ImGui::Spacing();

            // Mean
            std::string& mean_str = node.parameters["mean"];
            if (mean_str.empty()) mean_str = "0.0";
            char mean_buffer[32];
            strncpy(mean_buffer, mean_str.c_str(), sizeof(mean_buffer) - 1);
            mean_buffer[sizeof(mean_buffer) - 1] = '\0';

            ImGui::Text("Mean:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(150.0f);
            if (ImGui::InputText("##mean", mean_buffer, sizeof(mean_buffer))) {
                mean_str = mean_buffer;
            }

            ImGui::Spacing();

            // Std
            std::string& std_str = node.parameters["std"];
            if (std_str.empty()) std_str = "1.0";
            char std_buffer[32];
            strncpy(std_buffer, std_str.c_str(), sizeof(std_buffer) - 1);
            std_buffer[sizeof(std_buffer) - 1] = '\0';

            ImGui::Text("Standard Deviation:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(150.0f);
            if (ImGui::InputText("##std", std_buffer, sizeof(std_buffer))) {
                std_str = std_buffer;
            }

            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Common values:");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "  MNIST: mean=0.1307, std=0.3081");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "  ImageNet: mean=0.485,0.456,0.406");
            break;
        }

        case NodeType::OneHotEncode: {
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 1.0f, 1.0f), "One-Hot Encode Node");
            ImGui::Separator();
            ImGui::Spacing();

            // Num classes
            std::string& num_classes = node.parameters["num_classes"];
            if (num_classes.empty()) num_classes = "10";
            char classes_buffer[16];
            strncpy(classes_buffer, num_classes.c_str(), sizeof(classes_buffer) - 1);
            classes_buffer[sizeof(classes_buffer) - 1] = '\0';

            ImGui::Text("Number of Classes:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputText("##num_classes", classes_buffer, sizeof(classes_buffer), ImGuiInputTextFlags_CharsDecimal)) {
                num_classes = classes_buffer;
                InvalidateShapes();
            }
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "MNIST=10, CIFAR-10=10, ImageNet=1000");
            break;
        }

        // ========== Activation Functions ==========
        case NodeType::ReLU:
            ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "ReLU Activation");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "f(x) = max(0, x)");
            break;

        case NodeType::Sigmoid:
            ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "Sigmoid Activation");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "f(x) = 1 / (1 + exp(-x))");
            break;

        case NodeType::Tanh:
            ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "Tanh Activation");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "f(x) = tanh(x)");
            break;

        case NodeType::Softmax:
            ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "Softmax Activation");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "f(x_i) = exp(x_i) / sum(exp(x))");
            break;

        case NodeType::LeakyReLU: {
            ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "Leaky ReLU Activation");

            std::string& slope_str = node.parameters["negative_slope"];
            if (slope_str.empty()) slope_str = "0.01";
            float slope = std::stof(slope_str);

            ImGui::Text("Negative Slope:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##neg_slope", &slope, 0.001f, 0.3f, "%.3f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.3f", slope);
                slope_str = buf;
            }
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "f(x) = max(slope*x, x)");
            break;
        }

        // ========== Loss Functions ==========
        case NodeType::MSELoss:
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "Mean Squared Error Loss");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "L = mean((y - y_hat)^2)");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Use for: Regression tasks");
            break;

        case NodeType::CrossEntropyLoss:
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "Cross Entropy Loss");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "L = -sum(y * log(y_hat))");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Use for: Classification tasks");
            break;

        // ========== Optimizers ==========
        case NodeType::SGD: {
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 1.0f, 1.0f), "SGD Optimizer");
            ImGui::Separator();
            ImGui::Spacing();

            std::string& lr_str = node.parameters["learning_rate"];
            if (lr_str.empty()) lr_str = "0.01";
            float lr = std::stof(lr_str);

            ImGui::Text("Learning Rate:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##lr_sgd", &lr, 0.0001f, 1.0f, "%.4f", ImGuiSliderFlags_Logarithmic)) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.4f", lr);
                lr_str = buf;
            }

            std::string& momentum_str = node.parameters["momentum"];
            if (momentum_str.empty()) momentum_str = "0.9";
            float momentum = std::stof(momentum_str);

            ImGui::Text("Momentum:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##momentum_sgd", &momentum, 0.0f, 0.99f, "%.2f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.2f", momentum);
                momentum_str = buf;
            }
            break;
        }

        case NodeType::Adam: {
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 1.0f, 1.0f), "Adam Optimizer");
            ImGui::Separator();
            ImGui::Spacing();

            std::string& lr_str = node.parameters["learning_rate"];
            if (lr_str.empty()) lr_str = "0.001";
            float lr = std::stof(lr_str);

            ImGui::Text("Learning Rate:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##lr_adam", &lr, 0.00001f, 0.1f, "%.5f", ImGuiSliderFlags_Logarithmic)) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.5f", lr);
                lr_str = buf;
            }

            std::string& beta1_str = node.parameters["beta1"];
            if (beta1_str.empty()) beta1_str = "0.9";
            float beta1 = std::stof(beta1_str);

            ImGui::Text("Beta1:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##beta1", &beta1, 0.0f, 0.999f, "%.3f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.3f", beta1);
                beta1_str = buf;
            }

            std::string& beta2_str = node.parameters["beta2"];
            if (beta2_str.empty()) beta2_str = "0.999";
            float beta2 = std::stof(beta2_str);

            ImGui::Text("Beta2:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##beta2", &beta2, 0.0f, 0.9999f, "%.4f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.4f", beta2);
                beta2_str = buf;
            }
            break;
        }

        case NodeType::AdamW: {
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 1.0f, 1.0f), "AdamW Optimizer");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Adam with decoupled weight decay");
            ImGui::Separator();
            ImGui::Spacing();

            std::string& lr_str = node.parameters["learning_rate"];
            if (lr_str.empty()) lr_str = "0.001";
            float lr = std::stof(lr_str);

            ImGui::Text("Learning Rate:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##lr_adamw", &lr, 0.00001f, 0.1f, "%.5f", ImGuiSliderFlags_Logarithmic)) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.5f", lr);
                lr_str = buf;
            }

            std::string& beta1_str = node.parameters["beta1"];
            if (beta1_str.empty()) beta1_str = "0.9";
            float beta1 = std::stof(beta1_str);

            ImGui::Text("Beta1:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##beta1_w", &beta1, 0.0f, 0.999f, "%.3f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.3f", beta1);
                beta1_str = buf;
            }

            std::string& beta2_str = node.parameters["beta2"];
            if (beta2_str.empty()) beta2_str = "0.999";
            float beta2 = std::stof(beta2_str);

            ImGui::Text("Beta2:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##beta2_w", &beta2, 0.0f, 0.9999f, "%.4f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.4f", beta2);
                beta2_str = buf;
            }

            std::string& wd_str = node.parameters["weight_decay"];
            if (wd_str.empty()) wd_str = "0.01";
            float wd = std::stof(wd_str);

            ImGui::Text("Weight Decay:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##wd", &wd, 0.0f, 0.1f, "%.4f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.4f", wd);
                wd_str = buf;
            }
            break;
        }

        case NodeType::Flatten:
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 1.0f, 1.0f), "Flatten Layer");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Flattens input to 1D vector");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "[H, W, C] -> [H * W * C]");
            break;

        default:
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No editable parameters for this node type");
            break;
    }
}

} // namespace gui
