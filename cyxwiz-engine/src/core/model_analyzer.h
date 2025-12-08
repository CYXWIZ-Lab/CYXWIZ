#pragma once

#include "../gui/node_editor.h"
#include <vector>
#include <string>
#include <cstdint>

namespace cyxwiz {

/**
 * Analysis result for a single layer
 */
struct LayerAnalysis {
    std::string name;
    gui::NodeType type;
    std::vector<size_t> input_shape;
    std::vector<size_t> output_shape;
    int64_t parameters = 0;           // Trainable parameters
    int64_t non_trainable_params = 0; // BatchNorm running stats, etc.
    int64_t flops = 0;                // Multiply-add operations (MACs * 2)
    int64_t memory_bytes = 0;         // Activation memory in bytes
};

/**
 * Complete model analysis result
 */
struct ModelAnalysis {
    std::vector<LayerAnalysis> layers;
    int64_t total_parameters = 0;
    int64_t trainable_parameters = 0;
    int64_t non_trainable_parameters = 0;
    int64_t total_flops = 0;
    int64_t total_memory_bytes = 0;
    std::vector<size_t> input_shape;
    std::vector<size_t> output_shape;
    bool is_valid = false;
    std::string error_message;
};

/**
 * ModelAnalyzer - Analyzes neural network graphs for parameter counts, FLOPs, and memory
 *
 * This class provides static analysis of node graphs to compute:
 * - Total and per-layer parameter counts
 * - FLOPs (floating-point operations) for inference
 * - Memory footprint for activations
 * - Shape flow through the network
 */
class ModelAnalyzer {
public:
    ModelAnalyzer() = default;

    /**
     * Analyze a node graph from the visual editor
     * @param nodes List of nodes from NodeEditor
     * @param links List of links from NodeEditor
     * @param batch_size Batch size for FLOPs/memory calculation (default 1)
     * @return ModelAnalysis with layer-wise and total statistics
     */
    ModelAnalysis AnalyzeGraph(
        const std::vector<gui::MLNode>& nodes,
        const std::vector<gui::NodeLink>& links,
        int batch_size = 1
    );

    /**
     * Get a text summary of the model (similar to Keras model.summary())
     * @param analysis The analysis result
     * @return Formatted text summary
     */
    std::string GenerateSummary(const ModelAnalysis& analysis) const;

    /**
     * Export analysis to JSON
     * @param analysis The analysis result
     * @return JSON string
     */
    std::string ExportToJson(const ModelAnalysis& analysis) const;

private:
    // Topological sort for execution order
    std::vector<int> TopologicalSort(
        const std::vector<gui::MLNode>& nodes,
        const std::vector<gui::NodeLink>& links
    ) const;

    // Node lookup helpers
    const gui::MLNode* FindNodeById(int id, const std::vector<gui::MLNode>& nodes) const;

    // Find nodes feeding into this node
    std::vector<int> GetInputNodeIds(
        int to_node_id,
        const std::vector<gui::NodeLink>& links
    ) const;

    // Check if node type is a model layer (has parameters/computes)
    bool IsModelLayer(gui::NodeType type) const;
    bool IsActivation(gui::NodeType type) const;
    bool IsPooling(gui::NodeType type) const;
    bool IsNormalization(gui::NodeType type) const;
    bool IsUtilityNode(gui::NodeType type) const;

    // FLOPs calculation per layer type
    int64_t ComputeDenseFLOPs(int64_t in_features, int64_t out_features, bool bias, int64_t batch_size) const;
    int64_t ComputeConv2DFLOPs(int64_t in_h, int64_t in_w, int64_t in_c,
                               int64_t out_h, int64_t out_w, int64_t filters,
                               int64_t kernel_size, int64_t batch_size) const;
    int64_t ComputeBatchNormFLOPs(int64_t features, int64_t spatial_size, int64_t batch_size) const;
    int64_t ComputePoolFLOPs(int64_t out_h, int64_t out_w, int64_t channels,
                             int64_t pool_size, int64_t batch_size) const;
    int64_t ComputeActivationFLOPs(int64_t elements, gui::NodeType type) const;
    int64_t ComputeAttentionFLOPs(int64_t seq_len, int64_t embed_dim, int64_t num_heads, int64_t batch_size) const;
    int64_t ComputeLSTMFLOPs(int64_t input_size, int64_t hidden_size, int64_t seq_len, int64_t batch_size) const;

    // Parameter counting
    int64_t ComputeDenseParams(int64_t in_features, int64_t out_features, bool bias) const;
    int64_t ComputeConv2DParams(int64_t in_channels, int64_t filters,
                                int64_t kernel_size, bool bias) const;
    int64_t ComputeBatchNormParams(int64_t features) const;
    int64_t ComputeBatchNormNonTrainableParams(int64_t features) const;
    int64_t ComputeAttentionParams(int64_t embed_dim, int64_t num_heads) const;
    int64_t ComputeLSTMParams(int64_t input_size, int64_t hidden_size) const;

    // Shape inference
    std::vector<size_t> InferOutputShape(
        const gui::MLNode& node,
        const std::vector<size_t>& input_shape
    ) const;

    // Memory estimation (activation memory)
    int64_t ComputeActivationMemory(const std::vector<size_t>& shape, int64_t batch_size) const;

    // Extract numeric parameter from node
    int GetIntParam(const gui::MLNode& node, const std::string& key, int default_value) const;
    float GetFloatParam(const gui::MLNode& node, const std::string& key, float default_value) const;
    bool GetBoolParam(const gui::MLNode& node, const std::string& key, bool default_value) const;
};

// ===== Format Helpers =====

/**
 * Format parameter count with appropriate suffix (K, M, B)
 * @param count Number of parameters
 * @return Formatted string (e.g., "1.2M", "345K")
 */
std::string FormatParameterCount(int64_t count);

/**
 * Format FLOPs with appropriate suffix (K, M, G, T)
 * @param flops Number of floating-point operations
 * @return Formatted string (e.g., "2.5 GFLOPs", "1.2 TFLOPs")
 */
std::string FormatFLOPs(int64_t flops);

/**
 * Format memory size with appropriate suffix (KB, MB, GB)
 * @param bytes Number of bytes
 * @return Formatted string (e.g., "128 MB", "2.5 GB")
 */
std::string FormatMemory(int64_t bytes);

/**
 * Format shape vector as string
 * @param shape Shape vector
 * @return Formatted string (e.g., "(28, 28, 1)")
 */
std::string FormatShape(const std::vector<size_t>& shape);

/**
 * Get human-readable name for node type
 * @param type Node type enum
 * @return Layer type name (e.g., "Dense", "Conv2D")
 */
std::string GetNodeTypeName(gui::NodeType type);

} // namespace cyxwiz
