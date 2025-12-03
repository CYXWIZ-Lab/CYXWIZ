#pragma once

#include "../gui/node_editor.h"
#include <cyxwiz/tensor.h>
#include <cyxwiz/layer.h>
#include <cyxwiz/optimizer.h>
#include <memory>
#include <vector>
#include <map>
#include <string>
#include <optional>

namespace cyxwiz {

/**
 * Represents a compiled layer ready for execution
 */
struct CompiledLayer {
    gui::NodeType type;
    int node_id;
    std::string name;
    std::map<std::string, std::string> parameters;

    // For model layers that need to be instantiated
    int units = 0;              // Dense layer units
    int filters = 0;            // Conv2D filters
    int kernel_size = 3;        // Conv2D kernel size
    float dropout_rate = 0.0f;  // Dropout rate
    float negative_slope = 0.01f; // LeakyReLU slope

    // Computed shapes (after compilation)
    std::vector<size_t> input_shape;
    std::vector<size_t> output_shape;
};

/**
 * Preprocessing configuration extracted from graph
 */
struct PreprocessingConfig {
    bool has_normalization = false;
    float norm_mean = 0.0f;
    float norm_std = 1.0f;

    bool has_reshape = false;
    std::vector<int> reshape_dims;

    bool has_onehot = false;
    size_t num_classes = 0;
};

/**
 * Complete training configuration extracted from graph
 */
struct TrainingConfiguration {
    // Model architecture (in execution order)
    std::vector<CompiledLayer> layers;

    // Input/Output configuration
    std::vector<size_t> input_shape;    // e.g., [28, 28, 1] for MNIST
    size_t input_size = 0;              // Flattened size
    size_t output_size = 0;             // Number of classes

    // Dataset configuration
    std::string dataset_name;           // Name of dataset in DataRegistry
    float train_ratio = 0.8f;
    float val_ratio = 0.1f;
    float test_ratio = 0.1f;

    // Preprocessing
    PreprocessingConfig preprocessing;

    // Loss function
    gui::NodeType loss_type = gui::NodeType::CrossEntropyLoss;
    std::map<std::string, std::string> loss_params;

    // Optimizer
    gui::NodeType optimizer_type = gui::NodeType::Adam;
    float learning_rate = 0.001f;
    float momentum = 0.9f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float weight_decay = 0.0f;

    // Validation
    bool is_valid = false;
    std::string error_message;

    // Helper methods
    OptimizerType GetOptimizerType() const {
        switch (optimizer_type) {
            case gui::NodeType::SGD: return OptimizerType::SGD;
            case gui::NodeType::Adam: return OptimizerType::Adam;
            case gui::NodeType::AdamW: return OptimizerType::AdamW;
            default: return OptimizerType::Adam;
        }
    }

    std::string GetLossName() const {
        switch (loss_type) {
            case gui::NodeType::MSELoss: return "MSE";
            case gui::NodeType::CrossEntropyLoss: return "CrossEntropy";
            default: return "CrossEntropy";
        }
    }

    std::string GetOptimizerName() const {
        switch (optimizer_type) {
            case gui::NodeType::SGD: return "SGD";
            case gui::NodeType::Adam: return "Adam";
            case gui::NodeType::AdamW: return "AdamW";
            default: return "Adam";
        }
    }
};

/**
 * GraphCompiler - Compiles visual node graph into executable training configuration
 *
 * Takes the node graph from NodeEditor and produces a TrainingConfiguration
 * that can be used by TrainingExecutor to run actual training.
 */
class GraphCompiler {
public:
    GraphCompiler() = default;

    /**
     * Compile the node graph into a training configuration
     * @param nodes List of nodes from NodeEditor
     * @param links List of links from NodeEditor
     * @return TrainingConfiguration with is_valid flag and error_message if invalid
     */
    TrainingConfiguration Compile(
        const std::vector<gui::MLNode>& nodes,
        const std::vector<gui::NodeLink>& links
    );

    /**
     * Validate the graph without full compilation
     * @param nodes List of nodes from NodeEditor
     * @param links List of links from NodeEditor
     * @param error Output error message if invalid
     * @return true if graph is valid for training
     */
    bool ValidateGraph(
        const std::vector<gui::MLNode>& nodes,
        const std::vector<gui::NodeLink>& links,
        std::string& error
    );

private:
    // Topological sort for execution order
    std::vector<int> TopologicalSort(
        const std::vector<gui::MLNode>& nodes,
        const std::vector<gui::NodeLink>& links
    );

    // Node lookup helpers
    const gui::MLNode* FindNodeById(int id, const std::vector<gui::MLNode>& nodes) const;

    // Find connected nodes (outgoing edges)
    std::vector<int> GetConnectedNodes(
        int from_node_id,
        const std::vector<gui::NodeLink>& links
    ) const;

    // Find nodes feeding into this node (incoming edges)
    std::vector<int> GetInputNodes(
        int to_node_id,
        const std::vector<gui::NodeLink>& links
    ) const;

    // Find specific node types
    const gui::MLNode* FindDatasetInputNode(const std::vector<gui::MLNode>& nodes) const;
    const gui::MLNode* FindLossNode(const std::vector<gui::MLNode>& nodes) const;
    const gui::MLNode* FindOptimizerNode(const std::vector<gui::MLNode>& nodes) const;
    const gui::MLNode* FindOutputNode(const std::vector<gui::MLNode>& nodes) const;

    // Check if node is a model layer (trainable)
    bool IsModelLayer(gui::NodeType type) const;

    // Check if node is an activation function
    bool IsActivation(gui::NodeType type) const;

    // Check if node is preprocessing
    bool IsPreprocessing(gui::NodeType type) const;

    // Extract layer parameters from node
    CompiledLayer ExtractLayerConfig(const gui::MLNode& node) const;

    // Extract preprocessing config
    void ExtractPreprocessing(
        const gui::MLNode& node,
        PreprocessingConfig& config
    ) const;

    // Shape inference
    std::vector<size_t> InferOutputShape(
        const CompiledLayer& layer,
        const std::vector<size_t>& input_shape
    ) const;

    // Validation helpers
    bool HasCycle(
        const std::vector<gui::MLNode>& nodes,
        const std::vector<gui::NodeLink>& links
    ) const;

    bool IsFullyConnected(
        const std::vector<gui::MLNode>& nodes,
        const std::vector<gui::NodeLink>& links
    ) const;
};

} // namespace cyxwiz
