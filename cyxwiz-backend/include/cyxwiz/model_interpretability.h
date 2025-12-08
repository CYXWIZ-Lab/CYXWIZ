#pragma once

#include "cyxwiz/api_export.h"
#include <cyxwiz/sequential.h>
#include <cyxwiz/tensor.h>
#include <vector>
#include <string>
#include <map>
#include <functional>

namespace cyxwiz {

/**
 * GradCAMResult - Result of Grad-CAM computation
 */
struct CYXWIZ_API GradCAMResult {
    std::string layer_name;
    std::vector<size_t> heatmap_shape;     // [height, width]
    std::vector<float> heatmap;            // Normalized 0-1 values
    std::vector<float> overlay;            // Blended with input (if applicable)
    int target_class = -1;
    float class_score = 0.0f;
    bool success = false;
    std::string error_message;
};

/**
 * SaliencyMap - Gradient-based saliency visualization
 */
struct CYXWIZ_API SaliencyMap {
    std::vector<size_t> shape;
    std::vector<float> gradients;          // Raw gradients w.r.t. input
    std::vector<float> absolute_gradients; // |gradients|
    std::vector<float> smoothgrad;         // Averaged over noise (if computed)
    int target_class = -1;
    bool success = false;
    std::string error_message;
};

/**
 * LayerActivations - Intermediate layer outputs
 */
struct CYXWIZ_API LayerActivations {
    std::string layer_name;
    int layer_index;
    std::vector<size_t> shape;
    std::vector<float> activations;
    bool success = false;
    std::string error_message;
};

/**
 * ModelInterpretability - Tools for understanding model predictions
 *
 * Implements:
 * - Grad-CAM (Gradient-weighted Class Activation Mapping)
 * - Saliency Maps (Input gradient visualization)
 * - Layer Activation Extraction
 */
class CYXWIZ_API ModelInterpretability {
public:
    /**
     * Compute Grad-CAM heatmap for a specific layer
     *
     * Grad-CAM highlights important regions in the input that contribute
     * to a specific class prediction, by combining gradients and activations.
     *
     * @param model The trained SequentialModel
     * @param input Input tensor (usually an image)
     * @param target_layer Name or index of the target convolutional layer
     * @param target_class Class to explain (-1 = use predicted class)
     * @return GradCAMResult with heatmap
     */
    static GradCAMResult ComputeGradCAM(
        SequentialModel& model,
        const Tensor& input,
        const std::string& target_layer,
        int target_class = -1
    );

    /**
     * Compute Grad-CAM by layer index
     */
    static GradCAMResult ComputeGradCAM(
        SequentialModel& model,
        const Tensor& input,
        int layer_index,
        int target_class = -1
    );

    /**
     * Compute saliency map (input gradient visualization)
     *
     * Shows which input pixels have the most influence on the output class.
     *
     * @param model The trained SequentialModel
     * @param input Input tensor
     * @param target_class Class to explain (-1 = use predicted class)
     * @return SaliencyMap with gradients
     */
    static SaliencyMap ComputeSaliencyMap(
        SequentialModel& model,
        const Tensor& input,
        int target_class = -1
    );

    /**
     * Compute SmoothGrad - saliency with noise averaging
     *
     * Reduces noise in saliency maps by averaging gradients over
     * multiple noisy copies of the input.
     *
     * @param model The trained SequentialModel
     * @param input Input tensor
     * @param target_class Class to explain (-1 = use predicted class)
     * @param num_samples Number of noisy samples (default 50)
     * @param noise_std Standard deviation of Gaussian noise (default 0.1)
     * @return SaliencyMap with smoothgrad field populated
     */
    static SaliencyMap ComputeSmoothGrad(
        SequentialModel& model,
        const Tensor& input,
        int target_class = -1,
        int num_samples = 50,
        float noise_std = 0.1f
    );

    /**
     * Extract activations from a specific layer
     *
     * @param model The trained SequentialModel
     * @param input Input tensor
     * @param layer_index Index of the layer
     * @return LayerActivations with the layer output
     */
    static LayerActivations ExtractLayerActivations(
        SequentialModel& model,
        const Tensor& input,
        int layer_index
    );

    /**
     * Extract activations from a specific layer by name
     */
    static LayerActivations ExtractLayerActivations(
        SequentialModel& model,
        const Tensor& input,
        const std::string& layer_name
    );

    /**
     * Get list of layer names in the model
     */
    static std::vector<std::string> GetLayerNames(const SequentialModel& model);

    /**
     * Find layer index by name
     * @return Layer index or -1 if not found
     */
    static int FindLayerIndex(const SequentialModel& model, const std::string& name);

private:
    /**
     * Normalize heatmap to [0, 1] range
     */
    static std::vector<float> NormalizeHeatmap(const std::vector<float>& heatmap);

    /**
     * Resize heatmap using bilinear interpolation
     */
    static std::vector<float> ResizeHeatmap(
        const std::vector<float>& heatmap,
        size_t src_height, size_t src_width,
        size_t dst_height, size_t dst_width
    );

    /**
     * Apply ReLU to keep only positive contributions
     */
    static void ApplyReLU(std::vector<float>& data);

    /**
     * Compute global average pooling over spatial dimensions
     */
    static std::vector<float> GlobalAveragePool(
        const std::vector<float>& data,
        size_t channels, size_t height, size_t width
    );

    /**
     * Find the predicted class (argmax of output)
     */
    static int FindPredictedClass(const Tensor& output);

    /**
     * Create one-hot gradient for target class
     */
    static Tensor CreateOneHotGradient(int num_classes, int target_class);
};

} // namespace cyxwiz
