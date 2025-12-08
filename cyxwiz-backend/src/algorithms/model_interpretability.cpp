// Fix Windows min/max macro conflicts
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "cyxwiz/model_interpretability.h"
#include <spdlog/spdlog.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <limits>

// Undef Windows min/max macros if they leaked through
#ifdef _WIN32
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#endif

namespace cyxwiz {

// ============================================================================
// Utility Methods
// ============================================================================

std::vector<float> ModelInterpretability::NormalizeHeatmap(const std::vector<float>& heatmap) {
    if (heatmap.empty()) return {};

    float min_val = *std::min_element(heatmap.begin(), heatmap.end());
    float max_val = *std::max_element(heatmap.begin(), heatmap.end());

    std::vector<float> normalized(heatmap.size());

    if (max_val - min_val < 1e-10f) {
        std::fill(normalized.begin(), normalized.end(), 0.5f);
    } else {
        for (size_t i = 0; i < heatmap.size(); i++) {
            normalized[i] = (heatmap[i] - min_val) / (max_val - min_val);
        }
    }

    return normalized;
}

std::vector<float> ModelInterpretability::ResizeHeatmap(
    const std::vector<float>& heatmap,
    size_t src_height, size_t src_width,
    size_t dst_height, size_t dst_width)
{
    std::vector<float> resized(dst_height * dst_width);

    float y_ratio = static_cast<float>(src_height) / dst_height;
    float x_ratio = static_cast<float>(src_width) / dst_width;

    for (size_t y = 0; y < dst_height; y++) {
        for (size_t x = 0; x < dst_width; x++) {
            float src_y = y * y_ratio;
            float src_x = x * x_ratio;

            // Bilinear interpolation
            size_t y0 = static_cast<size_t>(src_y);
            size_t x0 = static_cast<size_t>(src_x);
            size_t y1 = std::min(y0 + 1, src_height - 1);
            size_t x1 = std::min(x0 + 1, src_width - 1);

            float fy = src_y - y0;
            float fx = src_x - x0;

            float v00 = heatmap[y0 * src_width + x0];
            float v01 = heatmap[y0 * src_width + x1];
            float v10 = heatmap[y1 * src_width + x0];
            float v11 = heatmap[y1 * src_width + x1];

            float v0 = v00 * (1 - fx) + v01 * fx;
            float v1 = v10 * (1 - fx) + v11 * fx;

            resized[y * dst_width + x] = v0 * (1 - fy) + v1 * fy;
        }
    }

    return resized;
}

void ModelInterpretability::ApplyReLU(std::vector<float>& data) {
    for (auto& v : data) {
        if (v < 0) v = 0;
    }
}

std::vector<float> ModelInterpretability::GlobalAveragePool(
    const std::vector<float>& data,
    size_t channels, size_t height, size_t width)
{
    std::vector<float> pooled(channels);
    size_t spatial_size = height * width;

    for (size_t c = 0; c < channels; c++) {
        float sum = 0.0f;
        for (size_t s = 0; s < spatial_size; s++) {
            sum += data[c * spatial_size + s];
        }
        pooled[c] = sum / spatial_size;
    }

    return pooled;
}

int ModelInterpretability::FindPredictedClass(const Tensor& output) {
    const auto& shape = output.Shape();
    size_t num_classes = shape.empty() ? 1 : shape.back();

    // Get output data
    std::vector<float> output_data(output.NumElements());
    const float* data_ptr = output.Data<float>();
    if (data_ptr) {
        std::copy(data_ptr, data_ptr + output.NumElements(), output_data.begin());
    }

    // Find argmax
    int max_idx = 0;
    float max_val = -std::numeric_limits<float>::infinity();

    for (size_t i = 0; i < num_classes; i++) {
        if (output_data[i] > max_val) {
            max_val = output_data[i];
            max_idx = static_cast<int>(i);
        }
    }

    return max_idx;
}

Tensor ModelInterpretability::CreateOneHotGradient(int num_classes, int target_class) {
    std::vector<float> grad(num_classes, 0.0f);
    if (target_class >= 0 && target_class < num_classes) {
        grad[target_class] = 1.0f;
    }

    Tensor tensor({static_cast<size_t>(num_classes)}, DataType::Float32);
    float* data = tensor.Data<float>();
    if (data) {
        std::copy(grad.begin(), grad.end(), data);
    }

    return tensor;
}

// ============================================================================
// Layer Utilities
// ============================================================================

std::vector<std::string> ModelInterpretability::GetLayerNames(const SequentialModel& model) {
    std::vector<std::string> names;
    for (size_t i = 0; i < model.Size(); i++) {
        const Module* module = model.GetModule(i);
        if (module) {
            names.push_back(module->GetName());
        }
    }
    return names;
}

int ModelInterpretability::FindLayerIndex(const SequentialModel& model, const std::string& name) {
    for (size_t i = 0; i < model.Size(); i++) {
        const Module* module = model.GetModule(i);
        if (module && module->GetName() == name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

// ============================================================================
// Layer Activations
// ============================================================================

LayerActivations ModelInterpretability::ExtractLayerActivations(
    SequentialModel& model,
    const Tensor& input,
    int layer_index)
{
    LayerActivations result;

    if (layer_index < 0 || layer_index >= static_cast<int>(model.Size())) {
        result.success = false;
        result.error_message = "Invalid layer index";
        return result;
    }

    const Module* target_module = model.GetModule(layer_index);
    if (!target_module) {
        result.success = false;
        result.error_message = "Layer not found";
        return result;
    }

    result.layer_name = target_module->GetName();
    result.layer_index = layer_index;

    // Forward pass up to target layer
    Tensor current = input;
    for (int i = 0; i <= layer_index; i++) {
        Module* module = model.GetModule(i);
        if (module) {
            current = module->Forward(current);
        }
    }

    // Copy activations
    result.shape = current.Shape();
    result.activations.resize(current.NumElements());

    const float* data_ptr = current.Data<float>();
    if (data_ptr) {
        std::copy(data_ptr, data_ptr + current.NumElements(), result.activations.begin());
    }

    result.success = true;
    return result;
}

LayerActivations ModelInterpretability::ExtractLayerActivations(
    SequentialModel& model,
    const Tensor& input,
    const std::string& layer_name)
{
    int idx = FindLayerIndex(model, layer_name);
    if (idx < 0) {
        LayerActivations result;
        result.success = false;
        result.error_message = "Layer '" + layer_name + "' not found";
        return result;
    }
    return ExtractLayerActivations(model, input, idx);
}

// ============================================================================
// Grad-CAM Implementation
// ============================================================================

GradCAMResult ModelInterpretability::ComputeGradCAM(
    SequentialModel& model,
    const Tensor& input,
    const std::string& target_layer,
    int target_class)
{
    int idx = FindLayerIndex(model, target_layer);
    if (idx < 0) {
        GradCAMResult result;
        result.success = false;
        result.error_message = "Target layer '" + target_layer + "' not found";
        return result;
    }
    return ComputeGradCAM(model, input, idx, target_class);
}

GradCAMResult ModelInterpretability::ComputeGradCAM(
    SequentialModel& model,
    const Tensor& input,
    int layer_index,
    int target_class)
{
    GradCAMResult result;

    if (layer_index < 0 || layer_index >= static_cast<int>(model.Size())) {
        result.success = false;
        result.error_message = "Invalid layer index";
        return result;
    }

    const Module* target_module = model.GetModule(layer_index);
    if (!target_module) {
        result.success = false;
        result.error_message = "Layer not found";
        return result;
    }

    result.layer_name = target_module->GetName();

    spdlog::info("Computing Grad-CAM for layer '{}' (index {})", result.layer_name, layer_index);

    // Forward pass - store intermediate activations
    std::vector<Tensor> intermediate_outputs;
    Tensor current = input;

    for (size_t i = 0; i < model.Size(); i++) {
        Module* module = model.GetModule(i);
        if (module) {
            current = module->Forward(current);
            intermediate_outputs.push_back(current);
        }
    }

    Tensor output = current;

    // Determine target class
    if (target_class < 0) {
        target_class = FindPredictedClass(output);
    }

    result.target_class = target_class;

    // Get class score
    const float* output_data = output.Data<float>();
    if (output_data && target_class < static_cast<int>(output.NumElements())) {
        result.class_score = output_data[target_class];
    }

    // Create one-hot gradient for target class
    int num_classes = static_cast<int>(output.NumElements());
    Tensor grad_output = CreateOneHotGradient(num_classes, target_class);

    // Backward pass to get gradients
    Tensor grad = grad_output;
    std::vector<Tensor> gradients;

    for (int i = static_cast<int>(model.Size()) - 1; i >= 0; i--) {
        Module* module = model.GetModule(i);
        if (module) {
            grad = module->Backward(grad);
            gradients.insert(gradients.begin(), grad);
        }
    }

    // Get target layer activations and gradients
    if (layer_index >= static_cast<int>(intermediate_outputs.size())) {
        result.success = false;
        result.error_message = "Could not get layer activations";
        return result;
    }

    Tensor& activations = intermediate_outputs[layer_index];
    const auto& act_shape = activations.Shape();

    // For Grad-CAM, we need [C, H, W] or [N, C, H, W] shaped activations
    if (act_shape.size() < 3) {
        // Not a convolutional layer - return simple gradient
        result.heatmap_shape = act_shape;
        result.heatmap.resize(activations.NumElements());

        const float* act_ptr = activations.Data<float>();
        if (act_ptr) {
            std::copy(act_ptr, act_ptr + activations.NumElements(), result.heatmap.begin());
        }

        result.heatmap = NormalizeHeatmap(result.heatmap);
        result.success = true;
        return result;
    }

    // Extract dimensions (assume NCHW or CHW)
    size_t channels, height, width;
    if (act_shape.size() == 4) {
        channels = act_shape[1];
        height = act_shape[2];
        width = act_shape[3];
    } else {
        channels = act_shape[0];
        height = act_shape[1];
        width = act_shape[2];
    }

    // Get gradients for target layer
    Tensor& layer_grad = gradients[layer_index];
    const float* grad_ptr = layer_grad.Data<float>();
    const float* act_ptr = activations.Data<float>();

    if (!grad_ptr || !act_ptr) {
        result.success = false;
        result.error_message = "Could not access layer data";
        return result;
    }

    // Global average pooling of gradients to get importance weights
    std::vector<float> grad_data(layer_grad.NumElements());
    std::copy(grad_ptr, grad_ptr + layer_grad.NumElements(), grad_data.begin());

    std::vector<float> weights = GlobalAveragePool(grad_data, channels, height, width);

    // Weighted combination of activation maps
    std::vector<float> heatmap(height * width, 0.0f);

    for (size_t c = 0; c < channels; c++) {
        for (size_t y = 0; y < height; y++) {
            for (size_t x = 0; x < width; x++) {
                size_t idx = c * height * width + y * width + x;
                heatmap[y * width + x] += weights[c] * act_ptr[idx];
            }
        }
    }

    // Apply ReLU
    ApplyReLU(heatmap);

    // Normalize
    heatmap = NormalizeHeatmap(heatmap);

    result.heatmap_shape = {height, width};
    result.heatmap = std::move(heatmap);
    result.success = true;

    spdlog::info("Grad-CAM complete: class={}, score={:.4f}, heatmap={}x{}",
                 target_class, result.class_score, height, width);

    return result;
}

// ============================================================================
// Saliency Map Implementation
// ============================================================================

SaliencyMap ModelInterpretability::ComputeSaliencyMap(
    SequentialModel& model,
    const Tensor& input,
    int target_class)
{
    SaliencyMap result;
    result.shape = input.Shape();

    spdlog::info("Computing saliency map for input shape: {}",
                 input.Shape().empty() ? 0 : input.Shape()[0]);

    // Forward pass
    Tensor output = model.Forward(input);

    // Determine target class
    if (target_class < 0) {
        target_class = FindPredictedClass(output);
    }
    result.target_class = target_class;

    // Create one-hot gradient
    int num_classes = static_cast<int>(output.NumElements());
    Tensor grad_output = CreateOneHotGradient(num_classes, target_class);

    // Backward pass to get input gradients
    Tensor grad = model.Backward(grad_output);

    // Extract gradient values
    result.gradients.resize(grad.NumElements());
    const float* grad_ptr = grad.Data<float>();
    if (grad_ptr) {
        std::copy(grad_ptr, grad_ptr + grad.NumElements(), result.gradients.begin());
    }

    // Compute absolute gradients
    result.absolute_gradients.resize(result.gradients.size());
    for (size_t i = 0; i < result.gradients.size(); i++) {
        result.absolute_gradients[i] = std::abs(result.gradients[i]);
    }

    // Normalize
    result.absolute_gradients = NormalizeHeatmap(result.absolute_gradients);

    result.success = true;

    spdlog::info("Saliency map complete for class {}", target_class);

    return result;
}

SaliencyMap ModelInterpretability::ComputeSmoothGrad(
    SequentialModel& model,
    const Tensor& input,
    int target_class,
    int num_samples,
    float noise_std)
{
    SaliencyMap result;
    result.shape = input.Shape();

    spdlog::info("Computing SmoothGrad with {} samples, noise_std={}",
                 num_samples, noise_std);

    // Accumulate gradients
    std::vector<float> accumulated_grads(input.NumElements(), 0.0f);

    std::mt19937 rng(42);
    std::normal_distribution<float> noise_dist(0.0f, noise_std);

    // Get input data
    const float* input_ptr = input.Data<float>();
    if (!input_ptr) {
        result.success = false;
        result.error_message = "Could not access input data";
        return result;
    }

    std::vector<float> input_data(input_ptr, input_ptr + input.NumElements());

    for (int sample = 0; sample < num_samples; sample++) {
        // Create noisy input
        Tensor noisy_input(input.Shape(), DataType::Float32);
        float* noisy_ptr = noisy_input.Data<float>();

        for (size_t i = 0; i < input_data.size(); i++) {
            noisy_ptr[i] = input_data[i] + noise_dist(rng);
        }

        // Compute saliency for noisy input
        SaliencyMap noisy_saliency = ComputeSaliencyMap(model, noisy_input, target_class);

        if (sample == 0) {
            result.target_class = noisy_saliency.target_class;
        }

        // Accumulate
        for (size_t i = 0; i < accumulated_grads.size(); i++) {
            accumulated_grads[i] += noisy_saliency.absolute_gradients[i];
        }
    }

    // Average
    for (auto& g : accumulated_grads) {
        g /= num_samples;
    }

    result.smoothgrad = NormalizeHeatmap(accumulated_grads);

    // Also compute regular saliency
    SaliencyMap regular = ComputeSaliencyMap(model, input, target_class);
    result.gradients = std::move(regular.gradients);
    result.absolute_gradients = std::move(regular.absolute_gradients);

    result.success = true;

    spdlog::info("SmoothGrad complete");

    return result;
}

} // namespace cyxwiz
