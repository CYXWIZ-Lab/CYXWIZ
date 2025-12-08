#pragma once

#include "../gui/node_editor.h"
#include <vector>
#include <string>
#include <functional>
#include <random>
#include <map>

namespace cyxwiz {

/**
 * ArchitectureScore - Metrics for evaluating a neural network architecture
 */
struct ArchitectureScore {
    // Scores (0-1 range, higher is better)
    double complexity_score = 0.0;   // Based on parameter count (lower is better -> inverted)
    double efficiency_score = 0.0;   // FLOPs per parameter ratio
    double depth_score = 0.0;        // Based on layer count
    double diversity_score = 0.0;    // Variety of layer types
    double connectivity_score = 0.0; // Skip connections, residual paths
    double overall_score = 0.0;      // Weighted combination

    // Raw metrics
    int64_t total_params = 0;
    int64_t trainable_params = 0;
    int64_t total_flops = 0;
    int layer_count = 0;
    int trainable_layer_count = 0;

    // Layer type distribution
    std::map<gui::NodeType, int> layer_type_counts;

    // Architecture description
    std::string architecture_summary;

    bool success = false;
    std::string error_message;
};

/**
 * MutationType - Types of architecture mutations
 */
enum class MutationType {
    AddLayer,           // Add a new layer
    RemoveLayer,        // Remove an existing layer
    SwapLayer,          // Replace a layer with different type
    ChangeUnits,        // Modify layer size (units, filters)
    ChangeActivation,   // Change activation function
    AddSkipConnection,  // Add residual connection
    RemoveSkipConnection,
    ChangeDropout,      // Modify dropout rate
    ChangeKernelSize,   // For conv layers
    Random              // Random mutation type
};

/**
 * NASSearchConfig - Configuration for architecture search
 */
struct NASSearchConfig {
    int population_size = 20;
    int generations = 10;
    int elite_count = 2;         // Top architectures kept unchanged
    double mutation_rate = 0.3;
    double crossover_rate = 0.5;

    // Architecture constraints
    int min_layers = 2;
    int max_layers = 20;
    int min_units = 16;
    int max_units = 1024;

    // Scoring weights
    double weight_complexity = 0.2;
    double weight_efficiency = 0.3;
    double weight_depth = 0.2;
    double weight_diversity = 0.15;
    double weight_connectivity = 0.15;
};

/**
 * NASSearchResult - Result of architecture search
 */
struct NASSearchResult {
    std::vector<gui::MLNode> best_architecture;
    std::vector<gui::NodeLink> best_links;
    ArchitectureScore best_score;

    std::vector<ArchitectureScore> generation_best;  // Best score per generation
    std::vector<ArchitectureScore> all_scores;       // All evaluated architectures

    int total_generations = 0;
    int total_evaluations = 0;

    bool success = false;
    std::string error_message;
};

/**
 * NASEvaluator - Neural Architecture Search utilities
 *
 * Provides tools for:
 * - Evaluating architecture complexity and efficiency
 * - Mutating architectures for search
 * - Generating architecture suggestions
 * - Simple evolutionary architecture search
 */
class NASEvaluator {
public:
    /**
     * Score an architecture without training
     *
     * Evaluates based on parameter count, FLOPs, depth, and other heuristics.
     *
     * @param nodes The node graph
     * @param links The connections
     * @param input_shape Input shape [batch, channels, height, width] or [batch, features]
     * @param config Scoring configuration (optional)
     * @return ArchitectureScore with all metrics
     */
    static ArchitectureScore ScoreArchitecture(
        const std::vector<gui::MLNode>& nodes,
        const std::vector<gui::NodeLink>& links,
        const std::vector<size_t>& input_shape,
        const NASSearchConfig& config = NASSearchConfig()
    );

    /**
     * Apply a mutation to an architecture
     *
     * @param nodes The current nodes (modified in place for returned copy)
     * @param links The current links
     * @param mutation Type of mutation to apply
     * @param seed Random seed
     * @return Pair of (mutated_nodes, mutated_links)
     */
    static std::pair<std::vector<gui::MLNode>, std::vector<gui::NodeLink>> MutateArchitecture(
        const std::vector<gui::MLNode>& nodes,
        const std::vector<gui::NodeLink>& links,
        MutationType mutation = MutationType::Random,
        unsigned int seed = 0
    );

    /**
     * Generate architecture suggestions based on task type
     *
     * @param task_type "classification", "regression", "image_classification", etc.
     * @param input_shape Input shape
     * @param output_size Number of output classes/values
     * @param num_suggestions Number of architectures to generate
     * @return Vector of (nodes, links) pairs
     */
    static std::vector<std::pair<std::vector<gui::MLNode>, std::vector<gui::NodeLink>>>
    SuggestArchitectures(
        const std::string& task_type,
        const std::vector<size_t>& input_shape,
        int output_size,
        int num_suggestions = 5
    );

    /**
     * Run evolutionary architecture search
     *
     * Note: This does NOT train models - it uses heuristic scoring only.
     * For actual NAS, you'd need to train candidate architectures.
     *
     * @param initial_nodes Starting architecture
     * @param initial_links Starting links
     * @param input_shape Input shape
     * @param config Search configuration
     * @param progress_callback Optional callback (generation, best_score)
     * @return NASSearchResult with best architecture found
     */
    static NASSearchResult EvolveArchitecture(
        const std::vector<gui::MLNode>& initial_nodes,
        const std::vector<gui::NodeLink>& initial_links,
        const std::vector<size_t>& input_shape,
        const NASSearchConfig& config = NASSearchConfig(),
        std::function<void(int, const ArchitectureScore&)> progress_callback = nullptr
    );

    /**
     * Get a human-readable description of an architecture
     */
    static std::string DescribeArchitecture(
        const std::vector<gui::MLNode>& nodes,
        const std::vector<gui::NodeLink>& links
    );

    /**
     * Validate an architecture for training
     * @return Pair of (is_valid, error_message)
     */
    static std::pair<bool, std::string> ValidateArchitecture(
        const std::vector<gui::MLNode>& nodes,
        const std::vector<gui::NodeLink>& links
    );

private:
    // Helper methods

    static int GetNextNodeId(const std::vector<gui::MLNode>& nodes);
    static int GetNextPinId(const std::vector<gui::MLNode>& nodes);
    static int GetNextLinkId(const std::vector<gui::NodeLink>& links);

    // Estimate parameters for a layer
    static int64_t EstimateLayerParams(
        gui::NodeType type,
        const std::map<std::string, std::string>& params,
        const std::vector<size_t>& input_shape
    );

    // Estimate FLOPs for a layer
    static int64_t EstimateLayerFLOPs(
        gui::NodeType type,
        const std::map<std::string, std::string>& params,
        const std::vector<size_t>& input_shape
    );

    // Check if a node type is a trainable layer
    static bool IsTrainableLayer(gui::NodeType type);

    // Check if a node type is an activation
    static bool IsActivation(gui::NodeType type);

    // Get random layer type for mutation
    static gui::NodeType GetRandomLayerType(std::mt19937& rng);

    // Get random activation type
    static gui::NodeType GetRandomActivation(std::mt19937& rng);

    // Create a new layer node
    static gui::MLNode CreateLayerNode(
        gui::NodeType type,
        int node_id,
        int& pin_id,
        int units = 64
    );

    // Find the output layer in a graph
    static int FindOutputNode(const std::vector<gui::MLNode>& nodes);

    // Find trainable layers
    static std::vector<int> FindTrainableLayers(const std::vector<gui::MLNode>& nodes);

    // Crossover two architectures (for evolutionary search)
    static std::pair<std::vector<gui::MLNode>, std::vector<gui::NodeLink>> Crossover(
        const std::vector<gui::MLNode>& parent1_nodes,
        const std::vector<gui::NodeLink>& parent1_links,
        const std::vector<gui::MLNode>& parent2_nodes,
        const std::vector<gui::NodeLink>& parent2_links,
        unsigned int seed
    );
};

} // namespace cyxwiz
