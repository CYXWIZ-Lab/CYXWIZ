#include "properties_shape_info.h"
#include "../core/data_registry.h"
#include "../core/arrow_dataset.h"
#include "../core/parquet_backed_dataset.h"
#include <imgui.h>
#include <algorithm>
#include <queue>
#include <set>

namespace gui::properties_shape {

std::string FormatShape(const std::vector<size_t>& shape) {
    if (shape.empty()) return "[]";
    std::string result = "[";
    for (size_t i = 0; i < shape.size(); i++) {
        if (i > 0) result += ", ";
        result += std::to_string(shape[i]);
    }
    result += "]";
    return result;
}

size_t GetBatchSize(NodeEditor* node_editor) {
    if (node_editor) {
        const auto& nodes = node_editor->GetNodes();
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
    return 32;
}

std::string FormatShapeMatrix(const std::vector<size_t>& shape, size_t batch_size) {
    if (shape.empty()) return "Scalar";

    size_t N = 1;
    for (size_t d : shape) N *= d;

    std::string batch_str = std::to_string(batch_size);

    if (shape.size() == 1) {
        return batch_str + " x " + std::to_string(shape[0]);
    }
    if (shape.size() == 2) {
        return batch_str + " x " + std::to_string(N) +
               "  (" + std::to_string(shape[0]) + " x " + std::to_string(shape[1]) + " unrolled)";
    }
    if (shape.size() == 3) {
        return batch_str + " x " + std::to_string(N) +
               "  (" + std::to_string(shape[0]) + " x " + std::to_string(shape[1]) +
               " x " + std::to_string(shape[2]) + " unrolled)";
    }

    std::string dims;
    for (size_t i = 0; i < shape.size(); i++) {
        if (i > 0) dims += " x ";
        dims += std::to_string(shape[i]);
    }
    return batch_str + " x " + std::to_string(N) + "  (" + dims + " unrolled)";
}

std::vector<size_t> GetInputShapeFromDataset(NodeEditor* node_editor) {
    auto& registry = cyxwiz::DataRegistry::Instance();

    if (!node_editor) return {};
    const auto& nodes = node_editor->GetNodes();

    for (const auto& node : nodes) {
        if (node.type != NodeType::DataInput &&
            node.type != NodeType::DatasetInput) {
            continue;
        }

        auto cat_it = node.parameters.find("file_category");
        if (cat_it != node.parameters.end() && cat_it->second == "image") {
            auto th = node.parameters.find("target_height");
            auto tw = node.parameters.find("target_width");
            auto rgb = node.parameters.find("rgb");
            if (th != node.parameters.end() && tw != node.parameters.end()) {
                try {
                    size_t h = std::stoul(th->second);
                    size_t w = std::stoul(tw->second);
                    size_t c = (rgb != node.parameters.end() &&
                                 rgb->second == "true") ? 3 : 1;
                    if (h > 0 && w > 0) return {h, w, c};
                } catch (...) {}
            }
        }

        auto name_it = node.parameters.find("dataset_name");
        if (name_it == node.parameters.end() || name_it->second.empty()) continue;
        const std::string& name = name_it->second;

        int64_t cols = 0;
        if (auto arrow_ds = registry.GetArrowDataset(name)) {
            cols = arrow_ds->GetNumColumns();
        } else if (auto pq_ds = registry.GetParquetBackedDataset(name)) {
            cols = pq_ds->GetNumColumns();
        } else if (registry.HasDataset(name)) {
            auto handle = registry.GetDataset(name);
            if (handle.IsValid()) {
                auto info = handle.GetInfo();
                if (!info.shape.empty()) return info.shape;
            }
        }

        if (cols > 0) {
            auto label_it = node.parameters.find("label_column");
            int64_t features = (label_it != node.parameters.end() &&
                                 !label_it->second.empty()) ? cols - 1 : cols;
            if (features > 0) return {static_cast<size_t>(features)};
        }

        auto shape_it = node.parameters.find("shape");
        if (shape_it != node.parameters.end()) {
            std::string shape_str = shape_it->second;
            shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), '['), shape_str.end());
            shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), ']'), shape_str.end());
            shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), ' '), shape_str.end());
            std::vector<size_t> shape;
            size_t pos = 0;
            while ((pos = shape_str.find(',')) != std::string::npos) {
                try { shape.push_back(std::stoul(shape_str.substr(0, pos))); } catch (...) {}
                shape_str.erase(0, pos + 1);
            }
            if (!shape_str.empty()) {
                try { shape.push_back(std::stoul(shape_str)); } catch (...) {}
            }
            if (!shape.empty()) return shape;
        }
    }

    return {};
}

std::vector<size_t> InferOutputShape(
    NodeEditor* node_editor,
    NodeType type,
    const std::vector<size_t>& input_shape,
    const std::map<std::string, std::string>& params)
{
    std::vector<size_t> output_shape;

    switch (type) {
        case NodeType::Dense: {
            int units = 64;
            if (params.count("units")) {
                units = std::stoi(params.at("units"));
            }
            output_shape = {static_cast<size_t>(units)};
            break;
        }

        case NodeType::Conv2D: {
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
                    output_shape = {h - kernel_size + 1, w - kernel_size + 1, static_cast<size_t>(filters)};
                }
            }
            break;
        }

        case NodeType::MaxPool2D: {
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
            size_t flat_size = 1;
            for (size_t dim : input_shape) flat_size *= dim;
            output_shape = {flat_size};
            break;
        }

        case NodeType::TensorReshape: {
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
            output_shape = GetInputShapeFromDataset(node_editor);
            break;

        case NodeType::DataLoader:
        case NodeType::DataSplit:
            output_shape = input_shape;
            break;

        case NodeType::OneHotEncode: {
            int num_classes = 10;
            if (params.count("num_classes")) {
                num_classes = std::stoi(params.at("num_classes"));
            }
            output_shape = {static_cast<size_t>(num_classes)};
            break;
        }

        case NodeType::Output: {
            int classes = 10;
            if (params.count("num_classes")) {
                classes = std::stoi(params.at("num_classes"));
            } else if (params.count("classes")) {
                classes = std::stoi(params.at("classes"));
            }
            output_shape = {static_cast<size_t>(classes)};
            break;
        }

        case NodeType::SequenceTagOutput: {
            int num_tags = 0;
            if (params.count("num_tags")) {
                num_tags = std::stoi(params.at("num_tags"));
            }
            output_shape = num_tags > 0
                               ? std::vector<size_t>{static_cast<size_t>(num_tags)}
                               : input_shape;
            break;
        }

        case NodeType::MSELoss:
        case NodeType::CrossEntropyLoss:
        case NodeType::SGD:
        case NodeType::Adam:
        case NodeType::AdamW:
            output_shape = {1};
            break;

        default:
            output_shape = input_shape;
            break;
    }

    return output_shape;
}

LayerParameters ComputeLayerParameters(
    NodeType type,
    const std::vector<size_t>& input_shape,
    const std::map<std::string, std::string>& params)
{
    LayerParameters layer_params;

    size_t input_features = 1;
    for (size_t d : input_shape) input_features *= d;

    switch (type) {
        case NodeType::Dense: {
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
            int filters = 32;
            int kernel_size = 3;
            if (params.count("filters")) {
                try { filters = std::stoi(params.at("filters")); } catch (...) {}
            }
            if (params.count("kernel_size")) {
                try { kernel_size = std::stoi(params.at("kernel_size")); } catch (...) {}
            }

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
            layer_params.weight_shape = {input_features};
            layer_params.bias_shape = {input_features};
            layer_params.weight_count = input_features;
            layer_params.bias_count = input_features;
            layer_params.total_params = layer_params.weight_count + layer_params.bias_count;
            layer_params.has_parameters = true;
            break;
        }

        case NodeType::ReLU:
        case NodeType::Sigmoid:
        case NodeType::Tanh:
        case NodeType::Softmax:
        case NodeType::LeakyReLU:
        case NodeType::Flatten:
        case NodeType::MaxPool2D:
        case NodeType::Dropout:
        case NodeType::Output:
        case NodeType::SequenceTagOutput:
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

NodeShapeInfo ComputeNodeShape(NodeEditor* node_editor, int node_id) {
    NodeShapeInfo info;
    info.is_valid = false;

    if (!node_editor) {
        info.error = "No graph context";
        return info;
    }

    const auto& nodes = node_editor->GetNodes();
    const auto& links = node_editor->GetLinks();

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

    std::vector<int> predecessors;
    std::set<int> visited;
    std::queue<int> queue;
    queue.push(node_id);
    visited.insert(node_id);

    while (!queue.empty()) {
        int current = queue.front();
        queue.pop();

        for (const auto& link : links) {
            if (link.to_node == current && visited.find(link.from_node) == visited.end()) {
                visited.insert(link.from_node);
                queue.push(link.from_node);
                predecessors.push_back(link.from_node);
            }
        }
    }

    std::reverse(predecessors.begin(), predecessors.end());
    predecessors.push_back(node_id);

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

        std::vector<size_t> input_shape;

        for (const auto& link : links) {
            if (link.to_node == nid) {
                if (node_output_shapes.count(link.from_node)) {
                    input_shape = node_output_shapes[link.from_node];
                    break;
                }
            }
        }

        if (input_shape.empty()) {
            input_shape = GetInputShapeFromDataset(node_editor);
        }

        std::vector<size_t> output_shape =
            InferOutputShape(node_editor, node->type, input_shape, node->parameters);
        node_output_shapes[nid] = output_shape;

        if (nid == node_id) {
            info.input_shape = input_shape;
            info.output_shape = output_shape;

            info.input_size = 1;
            for (size_t d : input_shape) info.input_size *= d;

            info.output_size = 1;
            for (size_t d : output_shape) info.output_size *= d;

            info.params = ComputeLayerParameters(node->type, input_shape, node->parameters);

            if (input_shape.empty()) {
                info.is_valid = false;
                info.error = "Input shape unknown - apply the DataInput node first";
            } else {
                info.is_valid = true;
            }
        }
    }

    return info;
}

void RenderShapeInfo(NodeEditor* node_editor, const NodeShapeInfo& shape_info) {
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    size_t batch_size = GetBatchSize(node_editor);

    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Tensor Shape (batch = %zu)", batch_size);
    ImGui::Spacing();

    if (!shape_info.is_valid) {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "Cannot compute shape");
        if (!shape_info.error.empty()) {
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Error: %s", shape_info.error.c_str());
        }
        return;
    }

    size_t input_memory = batch_size * shape_info.input_size * sizeof(float);
    size_t output_memory = batch_size * shape_info.output_size * sizeof(float);

    ImGui::Text("Input:");
    ImGui::Indent();
    ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "%s", FormatShapeMatrix(shape_info.input_shape, batch_size).c_str());
    if (shape_info.input_shape.size() > 1) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Per-sample: %s", FormatShape(shape_info.input_shape).c_str());
    }
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

    ImGui::Text("Output:");
    ImGui::Indent();
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "%s", FormatShapeMatrix(shape_info.output_shape, batch_size).c_str());
    if (shape_info.output_shape.size() > 1) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Per-sample: %s", FormatShape(shape_info.output_shape).c_str());
    }
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

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1.0f), "Transform: %zu x %zu -> %zu x %zu",
                       batch_size, shape_info.input_size, batch_size, shape_info.output_size);

    if (!shape_info.params.has_parameters) {
        return;
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), "Learnable Parameters");
    ImGui::Spacing();

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

    ImGui::Text("Bias:");
    ImGui::Indent();
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "%s", FormatShape(shape_info.params.bias_shape).c_str());
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "%zu parameters", shape_info.params.bias_count);
    ImGui::Unindent();

    ImGui::Spacing();
    ImGui::Separator();

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

} // namespace gui::properties_shape
