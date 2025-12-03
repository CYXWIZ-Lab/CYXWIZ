#pragma once

#include <string>
#include <map>
#include <vector>
#include <functional>

namespace gui {

// Forward declarations
enum class NodeType;
struct MLNode;
struct NodeLink;
class NodeEditor;

/**
 * Parameter information for a layer
 */
struct LayerParameters {
    std::vector<size_t> weight_shape;   // Weight tensor shape
    std::vector<size_t> bias_shape;     // Bias tensor shape
    size_t weight_count = 0;            // Number of weight parameters
    size_t bias_count = 0;              // Number of bias parameters
    size_t total_params = 0;            // Total trainable parameters
    bool has_parameters = false;        // Whether this layer has learnable params
};

/**
 * Computed shape information for a node
 */
struct NodeShapeInfo {
    std::vector<size_t> input_shape;
    std::vector<size_t> output_shape;
    size_t input_size = 0;   // Flattened input size
    size_t output_size = 0;  // Flattened output size
    bool is_valid = false;
    std::string error;
    LayerParameters params;  // Learnable parameters info
};

class Properties {
public:
    Properties();
    ~Properties();

    void Render();

    // Set the currently selected node to display properties for
    void SetSelectedNode(MLNode* node);
    void ClearSelection();

    // Set the node editor reference for graph access
    void SetNodeEditor(NodeEditor* editor) { node_editor_ = editor; }

    // Visibility control for sidebar integration
    bool* GetVisiblePtr() { return &show_window_; }

    // Force recomputation of shapes (call when graph changes)
    void InvalidateShapes() { shapes_valid_ = false; }

private:
    void RenderNodeProperties(MLNode& node);
    void RenderShapeInfo(const NodeShapeInfo& shape_info);

    // Shape inference methods
    NodeShapeInfo ComputeNodeShape(int node_id);
    std::vector<size_t> GetInputShapeFromDataset();
    std::vector<size_t> InferOutputShape(
        NodeType type,
        const std::vector<size_t>& input_shape,
        const std::map<std::string, std::string>& params);
    LayerParameters ComputeLayerParameters(
        NodeType type,
        const std::vector<size_t>& input_shape,
        const std::map<std::string, std::string>& params);

    // Helper to format shape as string
    std::string FormatShape(const std::vector<size_t>& shape);
    std::string FormatShapeMatrix(const std::vector<size_t>& shape, size_t batch_size);
    size_t GetBatchSize();

    bool show_window_;
    MLNode* selected_node_ = nullptr;
    NodeEditor* node_editor_ = nullptr;

    // Shape caching
    bool shapes_valid_ = false;
    std::map<int, NodeShapeInfo> cached_shapes_;
};

} // namespace gui
