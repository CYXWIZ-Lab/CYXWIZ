#pragma once

#include "node_editor.h"
#include <vector>
#include <map>

namespace gui {

/**
 * Shape inference engine for computing tensor shapes throughout the graph.
 *
 * This class analyzes the node graph and computes the tensor shapes that flow
 * through each pin connection. Shape information is stored directly in NodePin
 * structs and is recomputed when the graph is modified.
 *
 * Key features:
 * - Topological sort to compute shapes in correct dependency order
 * - Cached results (only recompute when InvalidateShapes() is called)
 * - Handles all layer types (Dense, Conv2D, Flatten, etc.)
 * - Reads dataset shape from DataRegistry for DatasetInput nodes
 */
class ShapeInferenceEngine {
public:
    ShapeInferenceEngine() = default;

    /**
     * Compute shapes for all nodes in the graph.
     * This is the main entry point - call this after graph modifications.
     *
     * @param nodes Graph nodes (modified in place - pin shapes updated)
     * @param links Graph connections
     */
    void ComputeAllShapes(std::vector<MLNode>& nodes, const std::vector<NodeLink>& links);

    /**
     * Invalidate cached shapes (call when graph is modified).
     * Next call to ComputeAllShapes() will recompute everything.
     */
    void InvalidateShapes();

    /**
     * Check if shapes have been computed
     */
    bool AreShapesValid() const { return shapes_valid_; }

private:
    /**
     * Infer output shape for a single node based on its input shapes and parameters.
     *
     * @param node The node to compute output shape for
     * @param input_shapes Input shapes (one per input pin)
     * @return Output shape for the node
     */
    std::vector<size_t> InferNodeOutputShape(
        const MLNode& node,
        const std::vector<std::vector<size_t>>& input_shapes
    );

    /**
     * Get the input shape from the loaded dataset (for DatasetInput nodes).
     *
     * @return Dataset shape, or default shape if no dataset loaded
     */
    std::vector<size_t> GetDatasetInputShape();

    /**
     * Perform topological sort to compute shapes in correct dependency order.
     *
     * @param nodes Graph nodes
     * @param links Graph connections
     * @return Sorted node IDs (or empty if cycle detected)
     */
    std::vector<int> TopologicalSortForShapes(
        const std::vector<MLNode>& nodes,
        const std::vector<NodeLink>& links
    );

    /**
     * Find a node by ID
     */
    MLNode* FindNode(std::vector<MLNode>& nodes, int node_id);
    const MLNode* FindNode(const std::vector<MLNode>& nodes, int node_id) const;

    /**
     * Find the output pin for a given pin ID
     */
    NodePin* FindPin(MLNode& node, int pin_id);

    bool shapes_valid_ = false;  // Whether cached shapes are valid
};

} // namespace gui
