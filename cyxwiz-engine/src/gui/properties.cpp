// Include Windows headers first, then undef conflicting macros
#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
// Undefine Windows macros that conflict with our method names
#ifdef CreateDialog
#undef CreateDialog
#endif
#ifdef CreateDialogA
#undef CreateDialogA
#endif
#ifdef CreateDialogW
#undef CreateDialogW
#endif
#endif

#include "properties.h"
#include "../core/node_metadata_registry.h"
#include "node_editor.h"
#include "../core/data_registry.h"
#include "../plugin/registries/plugin_node_registry.h"
#include "node_config_dialog.h"
#include <imgui.h>
#include <implot.h>
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
        if (!selected_node_) {
            // Placeholder when no node selected
            ImVec2 avail = ImGui::GetContentRegionAvail();
            ImGui::SetCursorPosY(avail.y * 0.3f);

            // Center icon
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 0.3f, 0.3f, 1.0f));
            ImGui::SetWindowFontScale(2.0f);
            float icon_width = ImGui::CalcTextSize("\xef\x80\x85").x;  // ICON_FA_SLIDERS
            ImGui::SetCursorPosX((avail.x - icon_width) * 0.5f);
            ImGui::Text("\xef\x80\x85");
            ImGui::SetWindowFontScale(1.0f);
            ImGui::PopStyleColor();

            ImGui::Spacing();

            const char* text = "Select a node to edit properties";
            ImVec2 text_size = ImGui::CalcTextSize(text);
            ImGui::SetCursorPosX((avail.x - text_size.x) * 0.5f);
            ImGui::TextDisabled("%s", text);
        } else {
            // Get metadata for this node type
            const cyxwiz::NodeMetadata* metadata =
                cyxwiz::NodeMetadataRegistry::Instance().GetMetadata(selected_node_->type);

            // Check if this is a dialog-only node (DataInput/DataOutput)
            bool is_dialog_only = (selected_node_->type == NodeType::DataInput ||
                                   selected_node_->type == NodeType::DataOutput);

            // Phase 3: Section-based rendering
            RenderGeneralSection(*selected_node_);

            // Skip other sections for dialog-only nodes
            if (!is_dialog_only) {
                ImGui::Spacing();

                // Parameters section - use metadata-driven rendering if available
                RenderParametersSection(*selected_node_, metadata);

                ImGui::Spacing();

                // Shape information section
                if (ImGui::CollapsingHeader("Shape Info", ImGuiTreeNodeFlags_DefaultOpen)) {
                    NodeShapeInfo shape_info = ComputeNodeShape(selected_node_->id);
                    RenderShapeInfo(shape_info);
                }

                ImGui::Spacing();

                // Advanced section
                RenderAdvancedSection(*selected_node_);

                ImGui::Spacing();

                // Presets section
                RenderPresetsSection(*selected_node_);

                ImGui::Spacing();

                // Node Executor section (for analytics nodes like KMeans, PCA, etc.)
                RenderExecutorSection(*selected_node_);
            }
        }
    }
    ImGui::End();

    // Render active configuration dialog (if open)
    if (active_dialog_ && active_dialog_->IsOpen()) {
        if (!active_dialog_->Render()) {
            // Dialog was closed
            active_dialog_.reset();
        }
    }
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

        case NodeType::SignalSlider: {
            ImGui::TextColored(ImVec4(0.0f, 0.9f, 0.8f, 1.0f), "Signal Slider");
            ImGui::Spacing();

            std::string& val_str = node.parameters["value"];
            std::string& min_str = node.parameters["min"];
            std::string& max_str = node.parameters["max"];
            float val = std::stof(val_str.empty() ? "0" : val_str);
            float mn = std::stof(min_str.empty() ? "-1" : min_str);
            float mx = std::stof(max_str.empty() ? "1" : max_str);

            ImGui::Text("Value:");
            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::SliderFloat("##slider_val", &val, mn, mx)) {
                char buf[32]; snprintf(buf, sizeof(buf), "%.4f", val);
                val_str = buf;
            }

            ImGui::Text("Range:");
            ImGui::SetNextItemWidth(90.0f);
            if (ImGui::InputFloat("##slider_min", &mn, 0, 0, "%.2f")) {
                char buf[32]; snprintf(buf, sizeof(buf), "%.2f", mn);
                min_str = buf;
            }
            ImGui::SameLine();
            ImGui::SetNextItemWidth(90.0f);
            if (ImGui::InputFloat("##slider_max", &mx, 0, 0, "%.2f")) {
                char buf[32]; snprintf(buf, sizeof(buf), "%.2f", mx);
                max_str = buf;
            }
            break;
        }

        case NodeType::SineWave: {
            ImGui::TextColored(ImVec4(0.0f, 0.9f, 0.8f, 1.0f), "Sine Wave Generator");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "A*sin(2*pi*f*t + phase) + offset");
            ImGui::Spacing();

            auto floatParam = [&](const char* label, const char* key, float step = 0.1f) {
                std::string& s = node.parameters[key];
                float v = std::stof(s.empty() ? "0" : s);
                ImGui::Text("%s:", label);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(120.0f);
                std::string id = std::string("##sine_") + key;
                if (ImGui::InputFloat(id.c_str(), &v, step, step * 10, "%.3f")) {
                    char buf[32]; snprintf(buf, sizeof(buf), "%.3f", v);
                    s = buf;
                }
            };
            floatParam("Amplitude", "amplitude");
            floatParam("Frequency", "frequency");
            floatParam("Phase", "phase");
            floatParam("Offset", "offset");
            break;
        }

        case NodeType::StepSignal: {
            ImGui::TextColored(ImVec4(0.0f, 0.9f, 0.8f, 1.0f), "Step Signal");
            ImGui::Spacing();

            auto floatParam = [&](const char* label, const char* key) {
                std::string& s = node.parameters[key];
                float v = std::stof(s.empty() ? "0" : s);
                ImGui::Text("%s:", label);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(120.0f);
                std::string id = std::string("##step_") + key;
                if (ImGui::InputFloat(id.c_str(), &v, 0.1f, 1.0f, "%.3f")) {
                    char buf[32]; snprintf(buf, sizeof(buf), "%.3f", v);
                    s = buf;
                }
            };
            floatParam("Step Time", "step_time");
            floatParam("Initial Value", "initial_value");
            floatParam("Final Value", "final_value");
            break;
        }

        case NodeType::RampSignal: {
            ImGui::TextColored(ImVec4(0.0f, 0.9f, 0.8f, 1.0f), "Ramp Signal");
            ImGui::Spacing();

            auto floatParam = [&](const char* label, const char* key) {
                std::string& s = node.parameters[key];
                float v = std::stof(s.empty() ? "0" : s);
                ImGui::Text("%s:", label);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(120.0f);
                std::string id = std::string("##ramp_") + key;
                if (ImGui::InputFloat(id.c_str(), &v, 0.1f, 1.0f, "%.3f")) {
                    char buf[32]; snprintf(buf, sizeof(buf), "%.3f", v);
                    s = buf;
                }
            };
            floatParam("Start Value", "start_value");
            floatParam("End Value", "end_value");
            floatParam("Duration", "duration");
            break;
        }

        case NodeType::SignalScope: {
            ImGui::TextColored(ImVec4(0.0f, 0.9f, 0.8f, 1.0f), "Signal Scope");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Plots incoming signal values in real-time");
            ImGui::Spacing();

            std::string& ws = node.parameters["window_size"];
            int win = std::stoi(ws.empty() ? "500" : ws);
            ImGui::Text("Window Size:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputInt("##scope_win", &win)) {
                if (win < 10) win = 10;
                ws = std::to_string(win);
            }

            std::string& as = node.parameters["auto_scale"];
            bool auto_s = (as == "true");
            if (ImGui::Checkbox("Auto Scale", &auto_s)) {
                as = auto_s ? "true" : "false";
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // Real-time signal plot
            auto& buf = scope_buffers_[node.id];
            buf.max_samples = win;

            // Generate demo data when no simulation is running
            // (will be replaced by real simulation data when connected)
            scope_demo_time_ += ImGui::GetIO().DeltaTime;
            buf.Push(scope_demo_time_, std::sin(2.0f * 3.14159f * 0.5f * scope_demo_time_));

            if (!buf.times.empty()) {
                // Copy deque to contiguous arrays for ImPlot
                std::vector<float> t_arr(buf.times.begin(), buf.times.end());
                std::vector<float> v_arr(buf.values.begin(), buf.values.end());

                ImPlotFlags plot_flags = ImPlotFlags_NoTitle;
                if (ImPlot::BeginPlot("##scope_plot", ImVec2(-1, 200), plot_flags)) {
                    ImPlotAxisFlags x_flags = ImPlotAxisFlags_NoLabel;
                    ImPlotAxisFlags y_flags = auto_s ? (ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_NoLabel) : ImPlotAxisFlags_NoLabel;
                    ImPlot::SetupAxes("Time (s)", "Value", x_flags, y_flags);

                    // Auto-scroll X axis to follow latest data
                    if (!t_arr.empty()) {
                        float t_max = t_arr.back();
                        float t_window = win * 0.016f;  // Approximate window in seconds
                        if (t_window < 2.0f) t_window = 2.0f;
                        ImPlot::SetupAxisLimits(ImAxis_X1, t_max - t_window, t_max, ImGuiCond_Always);
                    }

                    ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.0f, 0.9f, 0.8f, 1.0f));
                    ImPlot::PlotLine("Signal", t_arr.data(), v_arr.data(), static_cast<int>(t_arr.size()));
                    ImPlot::PopStyleColor();
                    ImPlot::EndPlot();
                }
            }

            // Controls
            if (ImGui::Button("Clear")) {
                buf.Clear();
                scope_demo_time_ = 0.0f;
            }
            ImGui::SameLine();
            ImGui::TextDisabled("Samples: %d", static_cast<int>(buf.times.size()));

            break;
        }

        case NodeType::PluginCustom: {
            // Get plugin info for display
            auto info_opt = cyxwiz::plugin::PluginNodeRegistry::Instance().GetNodeTypeInfoCopy(
                node.plugin_qualified_name);

            std::string node_type_name;
            if (info_opt.has_value()) {
                const auto& info = info_opt.value();
                node_type_name = info.type_name;
                ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.8f, 1.0f), "%s", info.display_name.c_str());
                if (!info.description.empty()) {
                    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "%s", info.description.c_str());
                }
                ImGui::Separator();
            }

            // ===== MuJoCo Plant — Custom Properties UI =====
            if (node_type_name == "MuJoCoPlant") {
                // MJCF File Path with Browse button
                ImGui::Text("MJCF Model:");
                std::string& mjcf_path = node.parameters["mjcf_path"];
                char path_buf[512];
                strncpy(path_buf, mjcf_path.c_str(), sizeof(path_buf) - 1);
                path_buf[sizeof(path_buf) - 1] = '\0';
                ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 70.0f);
                if (ImGui::InputText("##mjcf_path", path_buf, sizeof(path_buf), ImGuiInputTextFlags_EnterReturnsTrue)) {
                    mjcf_path = path_buf;
                    if (node.has_dynamic_pins && node_editor_) {
                        node_editor_->ResolveDynamicPins(node.id);
                    }
                    InvalidateShapes();
                }
                ImGui::SameLine();
                if (ImGui::Button("Browse")) {
#ifdef _WIN32
                    OPENFILENAMEA ofn = {};
                    char file[512] = {};
                    ofn.lStructSize = sizeof(ofn);
                    ofn.lpstrFilter = "MJCF Files (*.xml)\0*.xml\0All Files\0*.*\0";
                    ofn.lpstrFile = file;
                    ofn.nMaxFile = sizeof(file);
                    ofn.Flags = OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
                    if (GetOpenFileNameA(&ofn)) {
                        mjcf_path = file;
                        if (node.has_dynamic_pins && node_editor_) {
                            node_editor_->ResolveDynamicPins(node.id);
                        }
                        InvalidateShapes();
                    }
#endif
                }

                // Show loaded model status from Environment Library
                {
                    auto meta_path = node.parameters.find("_meta_loaded_path");
                    if (meta_path != node.parameters.end() && !meta_path->second.empty()) {
                        ImGui::TextColored(ImVec4(0.3f, 0.9f, 0.5f, 1.0f),
                            "Loaded from Environment Library:");
                        ImGui::TextWrapped("%s", meta_path->second.c_str());
                    } else if (mjcf_path.empty()) {
                        ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f),
                            "No model set.");
                        if (node.has_dynamic_pins && node_editor_) {
                            if (ImGui::Button("Sync from Environment Library")) {
                                node_editor_->ResolveDynamicPins(node.id);
                                InvalidateShapes();
                            }
                            ImGui::SameLine();
                            ImGui::TextDisabled("(Load a model in the Env Library first)");
                        }
                    }
                }

                ImGui::Spacing();

                // Interface mode dropdown
                std::string& iface = node.parameters["interface"];
                if (iface.empty()) iface = "bus";
                int iface_idx = (iface == "vector") ? 1 : 0;
                ImGui::Text("Interface:");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(120.0f);
                const char* iface_items[] = { "Bus (per-actuator)", "Vector (single array)" };
                if (ImGui::Combo("##iface_mode", &iface_idx, iface_items, 2)) {
                    iface = (iface_idx == 1) ? "vector" : "bus";
                    if (node.has_dynamic_pins && node_editor_) {
                        node_editor_->ResolveDynamicPins(node.id);
                    }
                    InvalidateShapes();
                }

                // Timestep
                std::string& ts = node.parameters["timestep"];
                if (ts.empty()) ts = "0.002";
                float timestep = std::stof(ts);
                ImGui::Text("Timestep:");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(100.0f);
                if (ImGui::InputFloat("##timestep", &timestep, 0.001f, 0.01f, "%.4f")) {
                    if (timestep < 0.0001f) timestep = 0.0001f;
                    ts = std::to_string(timestep);
                }

                // Frame skip
                std::string& fs = node.parameters["frame_skip"];
                if (fs.empty()) fs = "1";
                int frame_skip = std::stoi(fs);
                ImGui::Text("Frame Skip:");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(100.0f);
                if (ImGui::InputInt("##frame_skip", &frame_skip)) {
                    if (frame_skip < 1) frame_skip = 1;
                    fs = std::to_string(frame_skip);
                }

                // Model info (from dynamic pin metadata)
                bool has_meta = false;
                for (const auto& [key, value] : node.parameters) {
                    if (key.starts_with("_meta_")) {
                        if (!has_meta) {
                            ImGui::Spacing();
                            ImGui::Separator();
                            ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Model Info:");
                            has_meta = true;
                        }
                        std::string display_key = key.substr(6);
                        ImGui::Text("  %s: %s", display_key.c_str(), value.c_str());
                    }
                }

                // Pin summary
                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Text("Actuator Inputs: %d", static_cast<int>(node.inputs.size()));
                ImGui::Text("Sensor Outputs: %d", static_cast<int>(node.outputs.size()));

                // Actuator table
                if (!node.inputs.empty() && ImGui::TreeNode("Actuator Pins")) {
                    if (ImGui::BeginTable("##act_table", 2, ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_RowBg)) {
                        ImGui::TableSetupColumn("Pin", ImGuiTableColumnFlags_WidthStretch);
                        ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed, 80.0f);
                        ImGui::TableHeadersRow();
                        for (const auto& pin : node.inputs) {
                            ImGui::TableNextRow();
                            ImGui::TableNextColumn();
                            ImGui::Text("%s", pin.name.c_str());
                            ImGui::TableNextColumn();
                            ImGui::TextDisabled("Scalar");
                        }
                        ImGui::EndTable();
                    }
                    ImGui::TreePop();
                }

                if (!node.outputs.empty() && ImGui::TreeNode("Sensor Pins")) {
                    if (ImGui::BeginTable("##sens_table", 2, ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_RowBg)) {
                        ImGui::TableSetupColumn("Pin", ImGuiTableColumnFlags_WidthStretch);
                        ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed, 80.0f);
                        ImGui::TableHeadersRow();
                        for (const auto& pin : node.outputs) {
                            ImGui::TableNextRow();
                            ImGui::TableNextColumn();
                            ImGui::Text("%s", pin.name.c_str());
                            ImGui::TableNextColumn();
                            ImGui::TextDisabled("Tensor");
                        }
                        ImGui::EndTable();
                    }
                    ImGui::TreePop();
                }
            }
            // ===== Generic Plugin Node Properties =====
            else {
                // Render editable parameters (skip internal keys)
                for (auto& [key, value] : node.parameters) {
                    if (key == "plugin_qualified_name") continue;
                    if (key.starts_with("_meta_")) continue;

                    char buf[512];
                    strncpy(buf, value.c_str(), sizeof(buf) - 1);
                    buf[sizeof(buf) - 1] = '\0';

                    ImGui::Text("%s:", key.c_str());
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(200.0f);
                    std::string label = "##plugin_param_" + key;
                    if (ImGui::InputText(label.c_str(), buf, sizeof(buf), ImGuiInputTextFlags_EnterReturnsTrue)) {
                        value = buf;

                        if (node.has_dynamic_pins && key == node.dynamic_pin_trigger && node_editor_) {
                            node_editor_->ResolveDynamicPins(node.id);
                        }

                        InvalidateShapes();
                    }
                }

                // Show dynamic pin metadata if available
                bool has_meta = false;
                for (const auto& [key, value] : node.parameters) {
                    if (key.starts_with("_meta_")) {
                        if (!has_meta) {
                            ImGui::Separator();
                            ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Model Info:");
                            has_meta = true;
                        }
                        std::string display_key = key.substr(6);
                        ImGui::Text("  %s: %s", display_key.c_str(), value.c_str());
                    }
                }

                // Show pin summary
                ImGui::Separator();
                ImGui::Text("Inputs: %d  Outputs: %d",
                            static_cast<int>(node.inputs.size()),
                            static_cast<int>(node.outputs.size()));
            }
            break;
        }

        // ========== Smart I/O Nodes (Dialog-only configuration) ==========
        case NodeType::DataInput:
        case NodeType::DataOutput:
            // These nodes are configured via the Open Dialog button only
            ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "Use 'Open Dialog' to configure");
            break;

        default:
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No editable parameters for this node type");
            break;
    }
}

// ========== Phase 3: Enhanced Property Sections ==========

void Properties::RenderGeneralSection(MLNode& node) {
    ImGui::SetNextItemOpen(section_general_open_, ImGuiCond_Once);
    if (ImGui::CollapsingHeader("General", ImGuiTreeNodeFlags_DefaultOpen)) {
        section_general_open_ = true;

        // Node name (editable)
        char name_buf[128];
        strncpy(name_buf, node.name.c_str(), sizeof(name_buf) - 1);
        name_buf[sizeof(name_buf) - 1] = '\0';

        ImGui::Text("Name:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(180.0f);
        if (ImGui::InputText("##node_name", name_buf, sizeof(name_buf))) {
            node.name = name_buf;
        }

        // Node ID (read-only)
        ImGui::Text("ID: %d", node.id);

        // Node type
        auto* metadata = cyxwiz::NodeMetadataRegistry::Instance().GetMetadata(node.type);
        if (metadata) {
            ImGui::Text("Type: %s", metadata->name.c_str());

            // Category badge
            ImGui::SameLine();
            ImGui::TextDisabled("(%s)", cyxwiz::GetCategoryDisplayName(metadata->category).c_str());

            // Icon display
            if (!metadata->icon.empty()) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.7f, 1.0f, 1.0f));
                ImGui::Text("%s", metadata->icon.c_str());
                ImGui::PopStyleColor();
            }
        }

        // KNIME-style "Open Dialog" button for complex nodes
        RenderOpenDialogButton(node);
    } else {
        section_general_open_ = false;
    }
}

void Properties::RenderOpenDialogButton(MLNode& node) {
    // Check if this node type should have an "Open Dialog" button
    bool should_show = ShouldShowOpenDialogButton(node.type);
    spdlog::debug("RenderOpenDialogButton: node='{}', type={}, should_show={}",
                  node.name, static_cast<int>(node.type), should_show);
    if (!should_show) {
        return;
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Center the button
    float button_width = 150.0f;
    float avail_width = ImGui::GetContentRegionAvail().x;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (avail_width - button_width) * 0.5f);

    // Styled "Open Dialog" button (similar to KNIME)
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.4f, 0.6f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.5f, 0.7f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.15f, 0.35f, 0.55f, 1.0f));

    if (ImGui::Button("Open Dialog...", ImVec2(button_width, 0))) {
        // Create and open the dialog for this node
        active_dialog_ = NodeConfigDialogFactory::Instance().CreateDialog(&node);
        if (active_dialog_) {
            active_dialog_->Open();
            spdlog::info("Opened configuration dialog for node '{}'", node.name);
        }
    }

    ImGui::PopStyleColor(3);

    // Tooltip
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::Text("Open detailed configuration dialog");
        ImGui::TextDisabled("(Configure all settings with preview)");
        ImGui::EndTooltip();
    }
}

void Properties::RenderParametersSection(MLNode& node, const cyxwiz::NodeMetadata* metadata) {
    ImGui::SetNextItemOpen(section_parameters_open_, ImGuiCond_Once);
    if (ImGui::CollapsingHeader("Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
        section_parameters_open_ = true;

        if (metadata && !metadata->parameters.empty()) {
            for (const auto& param : metadata->parameters) {
                RenderParameter(node, param);
            }
        } else {
            // Fallback to existing node-specific rendering
            RenderNodeProperties(node);
        }
    } else {
        section_parameters_open_ = false;
    }
}

void Properties::RenderAdvancedSection(MLNode& node) {
    ImGui::SetNextItemOpen(section_advanced_open_, ImGuiCond_Once);
    if (ImGui::CollapsingHeader("Advanced")) {
        section_advanced_open_ = true;

        // Initial position (if set for pattern insertion)
        if (node.has_initial_position) {
            ImGui::Text("Initial Position: (%.1f, %.1f)", node.initial_pos_x, node.initial_pos_y);
        }

        // Connections info
        if (node_editor_) {
            const auto& links = node_editor_->GetLinks();
            int input_count = 0, output_count = 0;
            for (const auto& link : links) {
                if (link.to_node == node.id) input_count++;
                if (link.from_node == node.id) output_count++;
            }
            ImGui::Text("Connections: %d in, %d out", input_count, output_count);
        }

        // Raw parameters (debug)
        if (ImGui::TreeNode("Raw Parameters")) {
            for (const auto& [key, value] : node.parameters) {
                ImGui::Text("%s: %s", key.c_str(), value.c_str());
            }
            ImGui::TreePop();
        }
    } else {
        section_advanced_open_ = false;
    }
}

void Properties::RenderPresetsSection(MLNode& node) {
    ImGui::SetNextItemOpen(section_presets_open_, ImGuiCond_Once);
    if (ImGui::CollapsingHeader("Presets")) {
        section_presets_open_ = true;

        // List available presets
        auto presets = GetPresetsForNodeType(node.type);
        if (!presets.empty()) {
            ImGui::Text("Available Presets:");
            for (const auto& preset : presets) {
                if (ImGui::Button(preset.c_str())) {
                    LoadPreset(node, preset);
                }
                ImGui::SameLine();
            }
            ImGui::NewLine();
            ImGui::Separator();
        }

        // Save new preset
        ImGui::Text("Save Current as Preset:");
        ImGui::SetNextItemWidth(150.0f);
        ImGui::InputText("##preset_name", preset_name_buffer_, sizeof(preset_name_buffer_));
        ImGui::SameLine();
        if (ImGui::Button("Save") && preset_name_buffer_[0] != '\0') {
            SavePreset(node, preset_name_buffer_);
            preset_name_buffer_[0] = '\0';
        }
    } else {
        section_presets_open_ = false;
    }
}

void Properties::RenderParameter(MLNode& node, const cyxwiz::ParameterDefinition& param) {
    ImGui::PushID(param.name.c_str());

    std::string& value = node.parameters[param.name];

    // Initialize with default if empty
    if (value.empty() && !param.default_value.empty()) {
        value = param.default_value;
    }

    // Check for validation errors
    bool has_error = validation_errors_.count(param.name) > 0;
    if (has_error) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
    }

    // Label
    ImGui::Text("%s:", param.name.c_str());
    if (has_error) {
        ImGui::PopStyleColor();
    }

    // Tooltip with description
    if (!param.description.empty() && ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::TextUnformatted(param.description.c_str());
        if (!param.default_value.empty()) {
            ImGui::TextDisabled("Default: %s", param.default_value.c_str());
        }
        ImGui::EndTooltip();
    }

    ImGui::SameLine();

    bool changed = false;

    // Render based on parameter type
    if (param.type == "int") {
        int int_val = value.empty() ? 0 : std::stoi(value);
        ImGui::SetNextItemWidth(120.0f);
        if (ImGui::InputInt("##value", &int_val)) {
            value = std::to_string(int_val);
            changed = true;
        }
    }
    else if (param.type == "float") {
        float float_val = value.empty() ? 0.0f : std::stof(value);
        ImGui::SetNextItemWidth(120.0f);

        // Try to parse range from validation string (e.g., "0.0-1.0")
        float min_v = 0.0f, max_v = 1.0f;
        bool has_range = false;
        if (!param.validation.empty()) {
            size_t dash_pos = param.validation.find('-');
            // Handle negative numbers: look for dash that's not at the start
            if (dash_pos != std::string::npos && dash_pos > 0) {
                try {
                    min_v = std::stof(param.validation.substr(0, dash_pos));
                    max_v = std::stof(param.validation.substr(dash_pos + 1));
                    has_range = true;
                } catch (...) {}
            }
        }

        if (has_range) {
            if (ImGui::SliderFloat("##value", &float_val, min_v, max_v, "%.4f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.4f", float_val);
                value = buf;
                changed = true;
            }
        } else {
            if (ImGui::InputFloat("##value", &float_val, 0.01f, 0.1f, "%.4f")) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.4f", float_val);
                value = buf;
                changed = true;
            }
        }
    }
    else if (param.type == "bool") {
        bool bool_val = (value == "true" || value == "1");
        if (ImGui::Checkbox("##value", &bool_val)) {
            value = bool_val ? "true" : "false";
            changed = true;
        }
    }
    else if (param.type == "enum" && !param.enum_values.empty()) {
        // Find current index
        int current_idx = 0;
        for (size_t i = 0; i < param.enum_values.size(); i++) {
            if (param.enum_values[i] == value) {
                current_idx = static_cast<int>(i);
                break;
            }
        }

        // Build combo items
        std::vector<const char*> items;
        for (const auto& ev : param.enum_values) {
            items.push_back(ev.c_str());
        }

        ImGui::SetNextItemWidth(150.0f);
        if (ImGui::Combo("##value", &current_idx, items.data(), static_cast<int>(items.size()))) {
            value = param.enum_values[current_idx];
            changed = true;
        }
    }
    else if (param.type == "file") {
        char file_buf[512];
        strncpy(file_buf, value.c_str(), sizeof(file_buf) - 1);
        file_buf[sizeof(file_buf) - 1] = '\0';

        ImGui::SetNextItemWidth(180.0f);
        if (ImGui::InputText("##value", file_buf, sizeof(file_buf))) {
            value = file_buf;
            changed = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Browse")) {
#ifdef _WIN32
            OPENFILENAMEA ofn = {};
            char file[512] = {};
            strncpy(file, value.c_str(), sizeof(file) - 1);
            ofn.lStructSize = sizeof(ofn);
            ofn.lpstrFilter = "All Files\0*.*\0";
            ofn.lpstrFile = file;
            ofn.nMaxFile = sizeof(file);
            ofn.Flags = OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
            if (GetOpenFileNameA(&ofn)) {
                value = file;
                changed = true;
            }
#endif
        }
    }
    else {
        // Default: string input
        char str_buf[256];
        strncpy(str_buf, value.c_str(), sizeof(str_buf) - 1);
        str_buf[sizeof(str_buf) - 1] = '\0';

        ImGui::SetNextItemWidth(180.0f);
        if (ImGui::InputText("##value", str_buf, sizeof(str_buf))) {
            value = str_buf;
            changed = true;
        }
    }

    // Validation on change
    if (changed) {
        std::string error;
        if (!ValidateParameter(value, param, error)) {
            validation_errors_[param.name] = error;
        } else {
            validation_errors_.erase(param.name);
        }
        InvalidateShapes();
    }

    // Show validation error
    if (has_error) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "!");
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", validation_errors_[param.name].c_str());
            ImGui::EndTooltip();
        }
    }

    ImGui::PopID();
}

bool Properties::ValidateParameter(const std::string& value, const cyxwiz::ParameterDefinition& param, std::string& error) {
    // Empty value is typically allowed (uses default)
    if (value.empty()) {
        return true;
    }

    // Parse validation string for range (e.g., "0-100" or "0.0-1.0")
    auto parseRange = [](const std::string& validation, double& min_val, double& max_val) -> bool {
        if (validation.empty()) return false;
        size_t dash_pos = validation.find('-');
        if (dash_pos != std::string::npos && dash_pos > 0) {
            try {
                min_val = std::stod(validation.substr(0, dash_pos));
                max_val = std::stod(validation.substr(dash_pos + 1));
                return true;
            } catch (...) {}
        }
        return false;
    };

    // Type-specific validation
    if (param.type == "int") {
        try {
            int v = std::stoi(value);
            double min_val, max_val;
            if (parseRange(param.validation, min_val, max_val)) {
                if (v < static_cast<int>(min_val)) {
                    error = "Value must be >= " + std::to_string(static_cast<int>(min_val));
                    return false;
                }
                if (v > static_cast<int>(max_val)) {
                    error = "Value must be <= " + std::to_string(static_cast<int>(max_val));
                    return false;
                }
            }
        } catch (...) {
            error = "Invalid integer";
            return false;
        }
    }
    else if (param.type == "float") {
        try {
            float v = std::stof(value);
            double min_val, max_val;
            if (parseRange(param.validation, min_val, max_val)) {
                if (v < static_cast<float>(min_val)) {
                    error = "Value must be >= " + std::to_string(static_cast<float>(min_val));
                    return false;
                }
                if (v > static_cast<float>(max_val)) {
                    error = "Value must be <= " + std::to_string(static_cast<float>(max_val));
                    return false;
                }
            }
        } catch (...) {
            error = "Invalid number";
            return false;
        }
    }
    else if (param.type == "enum" && !param.enum_values.empty()) {
        bool found = false;
        for (const auto& ev : param.enum_values) {
            if (ev == value) {
                found = true;
                break;
            }
        }
        if (!found) {
            error = "Invalid option";
            return false;
        }
    }

    return true;
}

void Properties::SavePreset(const MLNode& node, const std::string& name) {
    // TODO: Implement preset persistence (save to JSON file)
    spdlog::info("Saved preset '{}' for node type {}", name, static_cast<int>(node.type));
}

void Properties::LoadPreset(MLNode& node, const std::string& name) {
    // TODO: Implement preset loading (read from JSON file)
    spdlog::info("Loading preset '{}' for node type {}", name, static_cast<int>(node.type));
    InvalidateShapes();
}

std::vector<std::string> Properties::GetPresetsForNodeType(NodeType type) {
    // TODO: Implement preset discovery (scan presets directory)
    std::vector<std::string> presets;

    // Return some built-in presets for common node types
    switch (type) {
        case NodeType::Conv2D:
            presets = {"VGG-style", "ResNet-style", "MobileNet-style"};
            break;
        case NodeType::Dense:
            presets = {"Small (64)", "Medium (256)", "Large (1024)"};
            break;
        case NodeType::Adam:
            presets = {"Default", "Fast Learning", "Fine-tuning"};
            break;
        default:
            break;
    }

    return presets;
}

// ==================== Node Executor Integration ====================

void Properties::RenderExecutorSection(MLNode& node) {
    // Check if this node type has an executor
    if (!HasNodeExecutor(node.type)) {
        return;  // No executor for this node type
    }

    auto* executor = cyxwiz::NodeExecutorFactory::Instance().GetExecutor(node.type);
    if (!executor) return;

    ImGui::Separator();

    // Executor header with name
    bool executor_open = ImGui::CollapsingHeader(
        (std::string(executor->GetName()) + " Configuration").c_str(),
        ImGuiTreeNodeFlags_DefaultOpen
    );

    if (executor_open) {
        ImGui::Indent(10.0f);

        // Setup input data from connected nodes if available
        SetupExecutorInputData(executor, node);

        // Render configuration UI
        ImGui::PushID("ExecutorConfig");
        executor->RenderConfigUI();
        ImGui::PopID();

        ImGui::Unindent(10.0f);
    }

    // Results section (shown after execution)
    if (executor->GetState() == cyxwiz::ExecutorState::Completed ||
        executor->GetState() == cyxwiz::ExecutorState::Executing ||
        executor->GetState() == cyxwiz::ExecutorState::Error) {

        bool results_open = ImGui::CollapsingHeader("Results", ImGuiTreeNodeFlags_DefaultOpen);

        if (results_open) {
            ImGui::Indent(10.0f);

            ImGui::PushID("ExecutorResults");
            executor->RenderResultsUI();
            ImGui::PopID();

            ImGui::Unindent(10.0f);
        }
    }
}

bool Properties::HasNodeExecutor(NodeType type) {
    return cyxwiz::NodeExecutorFactory::Instance().HasExecutor(type);
}

void Properties::SetupExecutorInputData(cyxwiz::INodeExecutor* executor, MLNode& node) {
    if (!node_editor_) return;

    // TODO: Get input data from connected upstream nodes
    // For now, check if there's a DatasetInput node connected
    // and load its data

    // This is a placeholder - in a real implementation:
    // 1. Find all nodes connected to this node's input pins
    // 2. Get their output data (Arrow tables or raw vectors)
    // 3. Pass to executor via SetInputData()

    // For demo/testing, we could generate sample data
    // if (executor->GetState() == cyxwiz::ExecutorState::Idle) {
    //     // Generate sample data for testing
    //     std::vector<std::vector<double>> sample_data;
    //     // ... generate random cluster data
    //     executor->SetInputData(sample_data);
    // }
}

} // namespace gui
