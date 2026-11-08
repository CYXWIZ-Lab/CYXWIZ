// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
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
 *
 * The map is ReLU(sum over the layer's channels c of alpha_c * A_c), alpha_c
 * the mean over the layer's height and width of d score / d A_c: the gradient
 * at the layer's OUTPUT (Selvaraju et al. 2017, as pytorch-grad-cam).
 */
struct CYXWIZ_API GradCAMResult {
    std::string layer_name;
    std::vector<size_t> layer_shape;       // [height, width] of the layer's map
    std::vector<float> layer_heatmap;      // the map at the layer's size, row-major
    std::vector<size_t> heatmap_shape;     // [height, width] of the input
    std::vector<float> raw_heatmap;        // the map resized to the input (bilinear, torch's align_corners=False)
    std::vector<float> heatmap;            // raw_heatmap scaled to 0-1
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
 * IntegratedGradientsResult - attribution of a score to each input value
 *
 * (x - x') * mean over k of d score / d x at x' + a_k (x - x'), a_k =
 * (k + 0.5) / steps (the midpoint Riemann sum, Sundararajan et al. 2017).
 * The attributions add up to about score(x) - score(x'): `attribution_sum`
 * against `score - baseline_score` is the completeness check.
 */
struct CYXWIZ_API IntegratedGradientsResult {
    std::vector<size_t> shape;             // the input's shape (batch 1)
    std::vector<float> attributions;       // same layout as the input
    int target_class = -1;
    int steps = 0;
    float score = 0.0f;                    // the target's score at the input
    float baseline_score = 0.0f;           // ... and at the baseline
    float attribution_sum = 0.0f;
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
 * A model as two functions, for any model that runs forward and backward (a
 * SequentialModel, a graph model): forward maps a batch of inputs to scores
 * [N, classes]; backward maps d / d scores [N, classes] to d / d inputs, in
 * the input's layout. The model should be in eval mode.
 */
struct CYXWIZ_API ExplainFunctions {
    std::function<Tensor(const Tensor&)> forward;
    std::function<Tensor(const Tensor&)> backward;
};

/**
 * ModelInterpretability - Tools for understanding model predictions
 *
 * Implements:
 * - Grad-CAM (Gradient-weighted Class Activation Mapping)
 * - Saliency maps and SmoothGrad (input gradients)
 * - Integrated Gradients
 * - Layer Activation Extraction
 *
 * Inputs carry a batch of one; the batch is the first dimension except for
 * spatial tensors, which use [H, W, C, N]. The score of a class is the
 * model's output for it (logits, before softmax). A backward pass leaves the
 * model's parameter gradients set.
 */
class CYXWIZ_API ModelInterpretability {
public:
    /**
     * Compute Grad-CAM heatmap for a specific layer
     *
     * @param model The trained SequentialModel
     * @param input Input tensor, as the model's first module takes it
     * @param target_layer Name of a layer whose output is [H, W, C, N]
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
     * @param target_class Class to explain (-1 = use predicted class)
     */
    static SaliencyMap ComputeSaliencyMap(
        SequentialModel& model,
        const Tensor& input,
        int target_class = -1
    );
    static SaliencyMap ComputeSaliencyMap(
        const ExplainFunctions& model,
        const Tensor& input,
        int target_class = -1
    );

    /**
     * Compute SmoothGrad - saliency averaged over noisy copies of the input
     *
     * @param num_samples Number of noisy samples (default 50)
     * @param noise_std Standard deviation of Gaussian noise (default 0.1)
     * @param seed Seed of the noise
     */
    static SaliencyMap ComputeSmoothGrad(
        SequentialModel& model,
        const Tensor& input,
        int target_class = -1,
        int num_samples = 50,
        float noise_std = 0.1f
    );
    static SaliencyMap ComputeSmoothGrad(
        const ExplainFunctions& model,
        const Tensor& input,
        int target_class = -1,
        int num_samples = 50,
        float noise_std = 0.1f,
        unsigned long long seed = 0
    );

    /**
     * Integrated Gradients from `baseline` (same shape as input: zeros, or a
     * black image as the model sees it) in `steps` steps, all in one batch.
     */
    static IntegratedGradientsResult ComputeIntegratedGradients(
        const ExplainFunctions& model,
        const Tensor& input,
        const Tensor& baseline,
        int target_class = -1,
        int steps = 64
    );

    /**
     * Extract activations from a specific layer
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

    /**
     * Resize a [src_height, src_width] row-major map to [dst_height,
     * dst_width] as torch.nn.functional.interpolate(mode="bilinear",
     * align_corners=False) does, on the device.
     */
    static std::vector<float> ResizeBilinear(
        const std::vector<float>& map,
        size_t src_height, size_t src_width,
        size_t dst_height, size_t dst_width
    );

    /**
     * A SequentialModel as ExplainFunctions (its whole stack).
     */
    static ExplainFunctions FunctionsOf(SequentialModel& model);
};

} // namespace cyxwiz
