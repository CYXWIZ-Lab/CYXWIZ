#include "model_analyzer.h"
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <queue>
#include <set>
#include <cmath>

namespace cyxwiz {

// ===== Format Helpers Implementation =====

std::string FormatParameterCount(int64_t count) {
    if (count >= 1'000'000'000) {
        return fmt::format("{:.2f}B", count / 1'000'000'000.0);
    } else if (count >= 1'000'000) {
        return fmt::format("{:.2f}M", count / 1'000'000.0);
    } else if (count >= 1'000) {
        return fmt::format("{:.2f}K", count / 1'000.0);
    }
    return std::to_string(count);
}

std::string FormatFLOPs(int64_t flops) {
    if (flops >= 1'000'000'000'000LL) {
        return fmt::format("{:.2f} TFLOPs", flops / 1'000'000'000'000.0);
    } else if (flops >= 1'000'000'000) {
        return fmt::format("{:.2f} GFLOPs", flops / 1'000'000'000.0);
    } else if (flops >= 1'000'000) {
        return fmt::format("{:.2f} MFLOPs", flops / 1'000'000.0);
    } else if (flops >= 1'000) {
        return fmt::format("{:.2f} KFLOPs", flops / 1'000.0);
    }
    return fmt::format("{} FLOPs", flops);
}

std::string FormatMemory(int64_t bytes) {
    if (bytes >= 1'073'741'824) {  // 1 GB
        return fmt::format("{:.2f} GB", bytes / 1'073'741'824.0);
    } else if (bytes >= 1'048'576) {  // 1 MB
        return fmt::format("{:.2f} MB", bytes / 1'048'576.0);
    } else if (bytes >= 1024) {  // 1 KB
        return fmt::format("{:.2f} KB", bytes / 1024.0);
    }
    return fmt::format("{} B", bytes);
}

std::string FormatShape(const std::vector<size_t>& shape) {
    if (shape.empty()) return "()";
    std::ostringstream oss;
    oss << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape[i];
    }
    oss << ")";
    return oss.str();
}

std::string GetNodeTypeName(gui::NodeType type) {
    switch (type) {
        case gui::NodeType::Dense: return "Dense";
        case gui::NodeType::Conv1D: return "Conv1D";
        case gui::NodeType::Conv2D: return "Conv2D";
        case gui::NodeType::Conv3D: return "Conv3D";
        case gui::NodeType::DepthwiseConv2D: return "DepthwiseConv2D";
        case gui::NodeType::MaxPool2D: return "MaxPool2D";
        case gui::NodeType::AvgPool2D: return "AvgPool2D";
        case gui::NodeType::GlobalMaxPool: return "GlobalMaxPool";
        case gui::NodeType::GlobalAvgPool: return "GlobalAvgPool";
        case gui::NodeType::AdaptiveAvgPool: return "AdaptiveAvgPool";
        case gui::NodeType::BatchNorm: return "BatchNorm";
        case gui::NodeType::LayerNorm: return "LayerNorm";
        case gui::NodeType::GroupNorm: return "GroupNorm";
        case gui::NodeType::InstanceNorm: return "InstanceNorm";
        case gui::NodeType::Dropout: return "Dropout";
        case gui::NodeType::Flatten: return "Flatten";
        case gui::NodeType::RNN: return "RNN";
        case gui::NodeType::LSTM: return "LSTM";
        case gui::NodeType::GRU: return "GRU";
        case gui::NodeType::Bidirectional: return "Bidirectional";
        case gui::NodeType::TimeDistributed: return "TimeDistributed";
        case gui::NodeType::Embedding: return "Embedding";
        case gui::NodeType::MultiHeadAttention: return "MultiHeadAttention";
        case gui::NodeType::SelfAttention: return "SelfAttention";
        case gui::NodeType::CrossAttention: return "CrossAttention";
        case gui::NodeType::LinearAttention: return "LinearAttention";
        case gui::NodeType::TransformerEncoder: return "TransformerEncoder";
        case gui::NodeType::TransformerDecoder: return "TransformerDecoder";
        case gui::NodeType::PositionalEncoding: return "PositionalEncoding";
        case gui::NodeType::ReLU: return "ReLU";
        case gui::NodeType::LeakyReLU: return "LeakyReLU";
        case gui::NodeType::PReLU: return "PReLU";
        case gui::NodeType::ELU: return "ELU";
        case gui::NodeType::SELU: return "SELU";
        case gui::NodeType::GELU: return "GELU";
        case gui::NodeType::Swish: return "Swish";
        case gui::NodeType::Mish: return "Mish";
        case gui::NodeType::Sigmoid: return "Sigmoid";
        case gui::NodeType::Tanh: return "Tanh";
        case gui::NodeType::Softmax: return "Softmax";
        case gui::NodeType::Reshape: return "Reshape";
        case gui::NodeType::Permute: return "Permute";
        case gui::NodeType::Squeeze: return "Squeeze";
        case gui::NodeType::Unsqueeze: return "Unsqueeze";
        case gui::NodeType::View: return "View";
        case gui::NodeType::Split: return "Split";
        case gui::NodeType::Concatenate: return "Concatenate";
        case gui::NodeType::Add: return "Add";
        case gui::NodeType::Multiply: return "Multiply";
        case gui::NodeType::Average: return "Average";
        case gui::NodeType::Output: return "Output";
        case gui::NodeType::DatasetInput: return "DatasetInput";
        case gui::NodeType::DataLoader: return "DataLoader";
        case gui::NodeType::Augmentation: return "Augmentation";
        case gui::NodeType::DataSplit: return "DataSplit";
        case gui::NodeType::Normalize: return "Normalize";
        case gui::NodeType::OneHotEncode: return "OneHotEncode";
        default: return "Unknown";
    }
}

// ===== ModelAnalyzer Implementation =====

ModelAnalysis ModelAnalyzer::AnalyzeGraph(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    int batch_size
) {
    ModelAnalysis result;

    if (nodes.empty()) {
        result.error_message = "Graph is empty";
        return result;
    }

    // Get topological order
    std::vector<int> sorted_ids = TopologicalSort(nodes, links);
    if (sorted_ids.empty()) {
        result.error_message = "Graph has cycles or is not connected";
        return result;
    }

    // Find input shape from DatasetInput or first layer
    std::vector<size_t> current_shape;
    bool found_input = false;

    for (int node_id : sorted_ids) {
        const gui::MLNode* node = FindNodeById(node_id, nodes);
        if (!node) continue;

        if (node->type == gui::NodeType::DatasetInput) {
            // Try to get shape from node parameters
            int h = GetIntParam(*node, "height", 28);
            int w = GetIntParam(*node, "width", 28);
            int c = GetIntParam(*node, "channels", 1);
            current_shape = {static_cast<size_t>(h), static_cast<size_t>(w), static_cast<size_t>(c)};
            result.input_shape = current_shape;
            found_input = true;
            break;
        }
    }

    if (!found_input) {
        // Default to a common shape if no input node found
        current_shape = {28, 28, 1};  // MNIST default
        result.input_shape = current_shape;
    }

    // Analyze each layer in topological order
    for (int node_id : sorted_ids) {
        const gui::MLNode* node = FindNodeById(node_id, nodes);
        if (!node) continue;

        // Skip utility/data nodes
        if (IsUtilityNode(node->type)) continue;

        LayerAnalysis layer_analysis;
        layer_analysis.name = node->name;
        layer_analysis.type = node->type;
        layer_analysis.input_shape = current_shape;

        // Compute output shape
        std::vector<size_t> output_shape = InferOutputShape(*node, current_shape);
        layer_analysis.output_shape = output_shape;

        // Compute parameters
        if (IsModelLayer(node->type)) {
            switch (node->type) {
                case gui::NodeType::Dense: {
                    int64_t in_features = 1;
                    for (size_t dim : current_shape) in_features *= dim;
                    int64_t out_features = GetIntParam(*node, "units", 128);
                    bool bias = GetBoolParam(*node, "bias", true);
                    layer_analysis.parameters = ComputeDenseParams(in_features, out_features, bias);
                    layer_analysis.flops = ComputeDenseFLOPs(in_features, out_features, bias, batch_size);
                    break;
                }
                case gui::NodeType::Conv2D:
                case gui::NodeType::Conv1D:
                case gui::NodeType::Conv3D: {
                    int64_t in_channels = current_shape.size() >= 3 ? current_shape[2] : 1;
                    int64_t filters = GetIntParam(*node, "filters", 32);
                    int64_t kernel_size = GetIntParam(*node, "kernel_size", 3);
                    bool bias = GetBoolParam(*node, "bias", true);
                    layer_analysis.parameters = ComputeConv2DParams(in_channels, filters, kernel_size, bias);

                    int64_t out_h = output_shape.size() >= 1 ? output_shape[0] : 1;
                    int64_t out_w = output_shape.size() >= 2 ? output_shape[1] : 1;
                    int64_t in_h = current_shape.size() >= 1 ? current_shape[0] : 1;
                    int64_t in_w = current_shape.size() >= 2 ? current_shape[1] : 1;
                    layer_analysis.flops = ComputeConv2DFLOPs(in_h, in_w, in_channels, out_h, out_w, filters, kernel_size, batch_size);
                    break;
                }
                case gui::NodeType::BatchNorm:
                case gui::NodeType::LayerNorm:
                case gui::NodeType::GroupNorm:
                case gui::NodeType::InstanceNorm: {
                    int64_t features = current_shape.size() >= 3 ? current_shape[2] : (current_shape.empty() ? 1 : current_shape.back());
                    layer_analysis.parameters = ComputeBatchNormParams(features);
                    layer_analysis.non_trainable_params = ComputeBatchNormNonTrainableParams(features);
                    int64_t spatial_size = 1;
                    for (size_t i = 0; i < current_shape.size() - 1 && i < 2; ++i) {
                        spatial_size *= current_shape[i];
                    }
                    layer_analysis.flops = ComputeBatchNormFLOPs(features, spatial_size, batch_size);
                    break;
                }
                case gui::NodeType::LSTM:
                case gui::NodeType::GRU:
                case gui::NodeType::RNN: {
                    int64_t input_size = current_shape.empty() ? 1 : current_shape.back();
                    int64_t hidden_size = GetIntParam(*node, "hidden_size", 128);
                    int64_t seq_len = current_shape.size() >= 2 ? current_shape[0] : 1;
                    layer_analysis.parameters = ComputeLSTMParams(input_size, hidden_size);
                    layer_analysis.flops = ComputeLSTMFLOPs(input_size, hidden_size, seq_len, batch_size);
                    break;
                }
                case gui::NodeType::MultiHeadAttention:
                case gui::NodeType::SelfAttention:
                case gui::NodeType::CrossAttention: {
                    int64_t embed_dim = GetIntParam(*node, "embed_dim", 256);
                    int64_t num_heads = GetIntParam(*node, "num_heads", 8);
                    int64_t seq_len = current_shape.size() >= 2 ? current_shape[0] : 1;
                    layer_analysis.parameters = ComputeAttentionParams(embed_dim, num_heads);
                    layer_analysis.flops = ComputeAttentionFLOPs(seq_len, embed_dim, num_heads, batch_size);
                    break;
                }
                case gui::NodeType::Embedding: {
                    int64_t vocab_size = GetIntParam(*node, "vocab_size", 10000);
                    int64_t embed_dim = GetIntParam(*node, "embed_dim", 256);
                    layer_analysis.parameters = vocab_size * embed_dim;
                    layer_analysis.flops = 0;  // Embedding is a lookup, no FLOPs
                    break;
                }
                default:
                    break;
            }
        }

        // Compute activation FLOPs
        if (IsActivation(node->type)) {
            int64_t elements = batch_size;
            for (size_t dim : current_shape) elements *= dim;
            layer_analysis.flops = ComputeActivationFLOPs(elements, node->type);
        }

        // Compute pooling FLOPs
        if (IsPooling(node->type)) {
            int64_t pool_size = GetIntParam(*node, "pool_size", 2);
            int64_t out_h = output_shape.size() >= 1 ? output_shape[0] : 1;
            int64_t out_w = output_shape.size() >= 2 ? output_shape[1] : 1;
            int64_t channels = current_shape.size() >= 3 ? current_shape[2] : 1;
            layer_analysis.flops = ComputePoolFLOPs(out_h, out_w, channels, pool_size, batch_size);
        }

        // Compute activation memory
        layer_analysis.memory_bytes = ComputeActivationMemory(output_shape, batch_size);

        // Add to results
        result.layers.push_back(layer_analysis);
        result.total_parameters += layer_analysis.parameters;
        result.trainable_parameters += layer_analysis.parameters;
        result.non_trainable_parameters += layer_analysis.non_trainable_params;
        result.total_flops += layer_analysis.flops;
        result.total_memory_bytes += layer_analysis.memory_bytes;

        // Update current shape for next layer
        current_shape = output_shape;
    }

    result.output_shape = current_shape;
    result.is_valid = true;
    return result;
}

std::string ModelAnalyzer::GenerateSummary(const ModelAnalysis& analysis) const {
    std::ostringstream oss;

    // Header
    oss << "Model Summary\n";
    oss << std::string(80, '=') << "\n";
    oss << std::left << std::setw(25) << "Layer (type)"
        << std::setw(20) << "Output Shape"
        << std::setw(15) << "Param #"
        << std::setw(15) << "FLOPs"
        << "\n";
    oss << std::string(80, '-') << "\n";

    // Layers
    for (const auto& layer : analysis.layers) {
        std::string type_name = GetNodeTypeName(layer.type);
        std::string display_name = layer.name.empty() ? type_name : layer.name + " (" + type_name + ")";
        if (display_name.length() > 24) {
            display_name = display_name.substr(0, 21) + "...";
        }

        oss << std::left << std::setw(25) << display_name
            << std::setw(20) << FormatShape(layer.output_shape)
            << std::setw(15) << FormatParameterCount(layer.parameters)
            << std::setw(15) << FormatFLOPs(layer.flops)
            << "\n";
    }

    // Footer
    oss << std::string(80, '=') << "\n";
    oss << "Total params: " << FormatParameterCount(analysis.total_parameters) << "\n";
    oss << "Trainable params: " << FormatParameterCount(analysis.trainable_parameters) << "\n";
    oss << "Non-trainable params: " << FormatParameterCount(analysis.non_trainable_parameters) << "\n";
    oss << "Total FLOPs: " << FormatFLOPs(analysis.total_flops) << "\n";
    oss << "Estimated memory: " << FormatMemory(analysis.total_memory_bytes) << "\n";
    oss << std::string(80, '-') << "\n";
    oss << "Input shape: " << FormatShape(analysis.input_shape) << "\n";
    oss << "Output shape: " << FormatShape(analysis.output_shape) << "\n";

    return oss.str();
}

std::string ModelAnalyzer::ExportToJson(const ModelAnalysis& analysis) const {
    nlohmann::json j;

    j["is_valid"] = analysis.is_valid;
    j["error_message"] = analysis.error_message;
    j["total_parameters"] = analysis.total_parameters;
    j["trainable_parameters"] = analysis.trainable_parameters;
    j["non_trainable_parameters"] = analysis.non_trainable_parameters;
    j["total_flops"] = analysis.total_flops;
    j["total_memory_bytes"] = analysis.total_memory_bytes;
    j["input_shape"] = analysis.input_shape;
    j["output_shape"] = analysis.output_shape;

    nlohmann::json layers_json = nlohmann::json::array();
    for (const auto& layer : analysis.layers) {
        nlohmann::json layer_json;
        layer_json["name"] = layer.name;
        layer_json["type"] = GetNodeTypeName(layer.type);
        layer_json["input_shape"] = layer.input_shape;
        layer_json["output_shape"] = layer.output_shape;
        layer_json["parameters"] = layer.parameters;
        layer_json["non_trainable_params"] = layer.non_trainable_params;
        layer_json["flops"] = layer.flops;
        layer_json["memory_bytes"] = layer.memory_bytes;
        layers_json.push_back(layer_json);
    }
    j["layers"] = layers_json;

    return j.dump(2);
}

// ===== Helper Methods =====

std::vector<int> ModelAnalyzer::TopologicalSort(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links
) const {
    std::map<int, int> in_degree;
    std::map<int, std::vector<int>> adjacency;

    // Initialize
    for (const auto& node : nodes) {
        in_degree[node.id] = 0;
        adjacency[node.id] = {};
    }

    // Build adjacency list and compute in-degrees
    for (const auto& link : links) {
        adjacency[link.from_node].push_back(link.to_node);
        in_degree[link.to_node]++;
    }

    // Kahn's algorithm
    std::queue<int> queue;
    for (const auto& [id, degree] : in_degree) {
        if (degree == 0) queue.push(id);
    }

    std::vector<int> result;
    while (!queue.empty()) {
        int node_id = queue.front();
        queue.pop();
        result.push_back(node_id);

        for (int neighbor : adjacency[node_id]) {
            in_degree[neighbor]--;
            if (in_degree[neighbor] == 0) {
                queue.push(neighbor);
            }
        }
    }

    // Check for cycles
    if (result.size() != nodes.size()) {
        return {};  // Graph has cycles
    }

    return result;
}

const gui::MLNode* ModelAnalyzer::FindNodeById(int id, const std::vector<gui::MLNode>& nodes) const {
    for (const auto& node : nodes) {
        if (node.id == id) return &node;
    }
    return nullptr;
}

std::vector<int> ModelAnalyzer::GetInputNodeIds(
    int to_node_id,
    const std::vector<gui::NodeLink>& links
) const {
    std::vector<int> inputs;
    for (const auto& link : links) {
        if (link.to_node == to_node_id) {
            inputs.push_back(link.from_node);
        }
    }
    return inputs;
}

bool ModelAnalyzer::IsModelLayer(gui::NodeType type) const {
    switch (type) {
        case gui::NodeType::Dense:
        case gui::NodeType::Conv1D:
        case gui::NodeType::Conv2D:
        case gui::NodeType::Conv3D:
        case gui::NodeType::DepthwiseConv2D:
        case gui::NodeType::BatchNorm:
        case gui::NodeType::LayerNorm:
        case gui::NodeType::GroupNorm:
        case gui::NodeType::InstanceNorm:
        case gui::NodeType::LSTM:
        case gui::NodeType::GRU:
        case gui::NodeType::RNN:
        case gui::NodeType::Embedding:
        case gui::NodeType::MultiHeadAttention:
        case gui::NodeType::SelfAttention:
        case gui::NodeType::CrossAttention:
        case gui::NodeType::TransformerEncoder:
        case gui::NodeType::TransformerDecoder:
            return true;
        default:
            return false;
    }
}

bool ModelAnalyzer::IsActivation(gui::NodeType type) const {
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

bool ModelAnalyzer::IsPooling(gui::NodeType type) const {
    switch (type) {
        case gui::NodeType::MaxPool2D:
        case gui::NodeType::AvgPool2D:
        case gui::NodeType::GlobalMaxPool:
        case gui::NodeType::GlobalAvgPool:
        case gui::NodeType::AdaptiveAvgPool:
            return true;
        default:
            return false;
    }
}

bool ModelAnalyzer::IsNormalization(gui::NodeType type) const {
    switch (type) {
        case gui::NodeType::BatchNorm:
        case gui::NodeType::LayerNorm:
        case gui::NodeType::GroupNorm:
        case gui::NodeType::InstanceNorm:
            return true;
        default:
            return false;
    }
}

bool ModelAnalyzer::IsUtilityNode(gui::NodeType type) const {
    switch (type) {
        case gui::NodeType::DatasetInput:
        case gui::NodeType::DataLoader:
        case gui::NodeType::DataSplit:
        case gui::NodeType::Output:
        case gui::NodeType::MSELoss:
        case gui::NodeType::CrossEntropyLoss:
        case gui::NodeType::BCELoss:
        case gui::NodeType::BCEWithLogits:
        case gui::NodeType::L1Loss:
        case gui::NodeType::SmoothL1Loss:
        case gui::NodeType::HuberLoss:
        case gui::NodeType::NLLLoss:
        case gui::NodeType::SGD:
        case gui::NodeType::Adam:
        case gui::NodeType::AdamW:
        case gui::NodeType::RMSprop:
        case gui::NodeType::Adagrad:
        case gui::NodeType::NAdam:
        case gui::NodeType::StepLR:
        case gui::NodeType::CosineAnnealing:
        case gui::NodeType::ReduceOnPlateau:
        case gui::NodeType::ExponentialLR:
        case gui::NodeType::WarmupScheduler:
            return true;
        default:
            return false;
    }
}

// ===== FLOPs Calculation =====

int64_t ModelAnalyzer::ComputeDenseFLOPs(int64_t in_features, int64_t out_features, bool bias, int64_t batch_size) const {
    // FLOPs = 2 * in * out (for matrix multiply)
    // If bias: +out (for addition)
    int64_t flops = 2 * in_features * out_features;
    if (bias) flops += out_features;
    return flops * batch_size;
}

int64_t ModelAnalyzer::ComputeConv2DFLOPs(int64_t in_h, int64_t in_w, int64_t in_c,
                                          int64_t out_h, int64_t out_w, int64_t filters,
                                          int64_t kernel_size, int64_t batch_size) const {
    // FLOPs = 2 * K^2 * C_in * C_out * H_out * W_out
    // (K^2 * C_in multiplications + K^2 * C_in - 1 additions per output element)
    return 2 * kernel_size * kernel_size * in_c * filters * out_h * out_w * batch_size;
}

int64_t ModelAnalyzer::ComputeBatchNormFLOPs(int64_t features, int64_t spatial_size, int64_t batch_size) const {
    // Per element: subtract mean, divide by std, scale, shift = 4 ops
    return 4 * features * spatial_size * batch_size;
}

int64_t ModelAnalyzer::ComputePoolFLOPs(int64_t out_h, int64_t out_w, int64_t channels,
                                        int64_t pool_size, int64_t batch_size) const {
    // For max pool: pool_size^2 - 1 comparisons per output
    // For avg pool: pool_size^2 additions + 1 division per output
    return pool_size * pool_size * out_h * out_w * channels * batch_size;
}

int64_t ModelAnalyzer::ComputeActivationFLOPs(int64_t elements, gui::NodeType type) const {
    switch (type) {
        case gui::NodeType::ReLU:
            return elements;  // 1 comparison per element
        case gui::NodeType::LeakyReLU:
            return 2 * elements;  // comparison + multiply
        case gui::NodeType::Sigmoid:
        case gui::NodeType::Tanh:
            return 4 * elements;  // exp, div, etc.
        case gui::NodeType::GELU:
        case gui::NodeType::Swish:
        case gui::NodeType::Mish:
            return 8 * elements;  // more complex
        case gui::NodeType::Softmax:
            return 3 * elements;  // exp, sum, div
        default:
            return elements;
    }
}

int64_t ModelAnalyzer::ComputeAttentionFLOPs(int64_t seq_len, int64_t embed_dim, int64_t num_heads, int64_t batch_size) const {
    // Q, K, V projections: 3 * 2 * seq * embed^2
    // Attention scores: 2 * seq^2 * embed
    // Softmax: 3 * seq^2 * num_heads
    // Output projection: 2 * seq * embed^2
    int64_t projection_flops = 3 * 2 * seq_len * embed_dim * embed_dim;
    int64_t attention_flops = 2 * seq_len * seq_len * embed_dim;
    int64_t softmax_flops = 3 * seq_len * seq_len * num_heads;
    int64_t output_flops = 2 * seq_len * embed_dim * embed_dim;
    return (projection_flops + attention_flops + softmax_flops + output_flops) * batch_size;
}

int64_t ModelAnalyzer::ComputeLSTMFLOPs(int64_t input_size, int64_t hidden_size, int64_t seq_len, int64_t batch_size) const {
    // LSTM has 4 gates, each with input and hidden connections
    // Per timestep: 4 * (input_size + hidden_size) * hidden_size * 2 (multiply-add)
    // Plus activations: 4 * hidden_size (tanh, sigmoid)
    int64_t per_step = 4 * 2 * (input_size + hidden_size) * hidden_size + 4 * hidden_size;
    return per_step * seq_len * batch_size;
}

// ===== Parameter Counting =====

int64_t ModelAnalyzer::ComputeDenseParams(int64_t in_features, int64_t out_features, bool bias) const {
    int64_t params = in_features * out_features;
    if (bias) params += out_features;
    return params;
}

int64_t ModelAnalyzer::ComputeConv2DParams(int64_t in_channels, int64_t filters,
                                           int64_t kernel_size, bool bias) const {
    int64_t params = kernel_size * kernel_size * in_channels * filters;
    if (bias) params += filters;
    return params;
}

int64_t ModelAnalyzer::ComputeBatchNormParams(int64_t features) const {
    // gamma (scale) + beta (shift)
    return 2 * features;
}

int64_t ModelAnalyzer::ComputeBatchNormNonTrainableParams(int64_t features) const {
    // running_mean + running_var
    return 2 * features;
}

int64_t ModelAnalyzer::ComputeAttentionParams(int64_t embed_dim, int64_t num_heads) const {
    // Q, K, V projections + output projection
    // Each projection: embed_dim * embed_dim + embed_dim (bias)
    return 4 * (embed_dim * embed_dim + embed_dim);
}

int64_t ModelAnalyzer::ComputeLSTMParams(int64_t input_size, int64_t hidden_size) const {
    // 4 gates, each with input weights, hidden weights, and bias
    // W_ii, W_if, W_ig, W_io: 4 * input_size * hidden_size
    // W_hi, W_hf, W_hg, W_ho: 4 * hidden_size * hidden_size
    // b_i, b_f, b_g, b_o: 4 * hidden_size (or 8 if separate biases)
    return 4 * (input_size * hidden_size + hidden_size * hidden_size + hidden_size);
}

// ===== Shape Inference =====

std::vector<size_t> ModelAnalyzer::InferOutputShape(
    const gui::MLNode& node,
    const std::vector<size_t>& input_shape
) const {
    if (input_shape.empty()) return {};

    switch (node.type) {
        case gui::NodeType::Dense: {
            int units = GetIntParam(node, "units", 128);
            return {static_cast<size_t>(units)};
        }
        case gui::NodeType::Conv2D: {
            int filters = GetIntParam(node, "filters", 32);
            int kernel_size = GetIntParam(node, "kernel_size", 3);
            int stride = GetIntParam(node, "stride", 1);
            int padding = GetIntParam(node, "padding", 0);

            size_t h = input_shape.size() >= 1 ? input_shape[0] : 1;
            size_t w = input_shape.size() >= 2 ? input_shape[1] : 1;

            size_t out_h = (h + 2 * padding - kernel_size) / stride + 1;
            size_t out_w = (w + 2 * padding - kernel_size) / stride + 1;
            return {out_h, out_w, static_cast<size_t>(filters)};
        }
        case gui::NodeType::MaxPool2D:
        case gui::NodeType::AvgPool2D: {
            int pool_size = GetIntParam(node, "pool_size", 2);
            int stride = GetIntParam(node, "stride", pool_size);

            size_t h = input_shape.size() >= 1 ? input_shape[0] : 1;
            size_t w = input_shape.size() >= 2 ? input_shape[1] : 1;
            size_t c = input_shape.size() >= 3 ? input_shape[2] : 1;

            size_t out_h = (h - pool_size) / stride + 1;
            size_t out_w = (w - pool_size) / stride + 1;
            return {out_h, out_w, c};
        }
        case gui::NodeType::GlobalMaxPool:
        case gui::NodeType::GlobalAvgPool: {
            size_t c = input_shape.size() >= 3 ? input_shape[2] : (input_shape.empty() ? 1 : input_shape.back());
            return {c};
        }
        case gui::NodeType::Flatten: {
            size_t flat_size = 1;
            for (size_t dim : input_shape) flat_size *= dim;
            return {flat_size};
        }
        case gui::NodeType::LSTM:
        case gui::NodeType::GRU:
        case gui::NodeType::RNN: {
            int hidden_size = GetIntParam(node, "hidden_size", 128);
            bool return_sequences = GetBoolParam(node, "return_sequences", false);
            if (return_sequences && input_shape.size() >= 2) {
                return {input_shape[0], static_cast<size_t>(hidden_size)};
            }
            return {static_cast<size_t>(hidden_size)};
        }
        case gui::NodeType::Embedding: {
            int embed_dim = GetIntParam(node, "embed_dim", 256);
            // Input is sequence of indices, output adds embedding dimension
            if (!input_shape.empty()) {
                return {input_shape[0], static_cast<size_t>(embed_dim)};
            }
            return {static_cast<size_t>(embed_dim)};
        }
        case gui::NodeType::MultiHeadAttention:
        case gui::NodeType::SelfAttention:
        case gui::NodeType::CrossAttention: {
            // Output shape same as input for attention
            return input_shape;
        }
        case gui::NodeType::Dropout:
        case gui::NodeType::BatchNorm:
        case gui::NodeType::LayerNorm:
        case gui::NodeType::GroupNorm:
        case gui::NodeType::InstanceNorm:
            // Shape unchanged
            return input_shape;
        default:
            // Activations and others preserve shape
            if (IsActivation(node.type)) {
                return input_shape;
            }
            return input_shape;
    }
}

int64_t ModelAnalyzer::ComputeActivationMemory(const std::vector<size_t>& shape, int64_t batch_size) const {
    int64_t elements = batch_size;
    for (size_t dim : shape) elements *= dim;
    return elements * sizeof(float);  // Assuming float32
}

// ===== Parameter Extraction =====

int ModelAnalyzer::GetIntParam(const gui::MLNode& node, const std::string& key, int default_value) const {
    auto it = node.parameters.find(key);
    if (it != node.parameters.end()) {
        try {
            return std::stoi(it->second);
        } catch (...) {}
    }
    return default_value;
}

float ModelAnalyzer::GetFloatParam(const gui::MLNode& node, const std::string& key, float default_value) const {
    auto it = node.parameters.find(key);
    if (it != node.parameters.end()) {
        try {
            return std::stof(it->second);
        } catch (...) {}
    }
    return default_value;
}

bool ModelAnalyzer::GetBoolParam(const gui::MLNode& node, const std::string& key, bool default_value) const {
    auto it = node.parameters.find(key);
    if (it != node.parameters.end()) {
        const std::string& val = it->second;
        if (val == "true" || val == "1" || val == "yes") return true;
        if (val == "false" || val == "0" || val == "no") return false;
    }
    return default_value;
}

} // namespace cyxwiz
