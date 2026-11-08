// Fix Windows min/max macro conflicts
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "cyxwiz/model_interpretability.h"

#include <arrayfire.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

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

namespace {

// The batch axis of an input: the last of a spatial [H, W, C, N] tensor, else the first.
size_t BatchAxis(const std::vector<size_t>& shape) { return shape.size() == 4 ? 3 : 0; }

size_t SampleSize(const std::vector<size_t>& shape) {
    size_t size = 1;
    for (size_t d = 0; d < shape.size(); ++d)
        if (d != BatchAxis(shape)) size *= shape[d];
    return size;
}

std::vector<float> Host(const af::array& a) {
    std::vector<float> values(a.elements());
    if (!values.empty()) a.as(f32).host(values.data());
    return values;
}

// The scores of a batch as an af array [N, classes].
af::array Scores(const Tensor& output) {
    const auto& shape = output.Shape();
    if (shape.size() == 1) return af::moddims(output.GetRowMajorFlatArray(), 1, static_cast<dim_t>(shape[0]));
    if (shape.size() != 2) throw std::invalid_argument("the model's scores must be [N, classes]");
    return output.GetSemanticArray();
}

// d score_target / d scores for every sample of the batch.
Tensor OneHot(const Tensor& output, int target) {
    const af::array scores = Scores(output);
    af::array grad = af::constant(0.0f, scores.dims(0), scores.dims(1));
    grad(af::span, target) = 1.0f;
    return Tensor::FromSemanticArray(grad, {static_cast<size_t>(scores.dims(0)), static_cast<size_t>(scores.dims(1))});
}

int Predicted(const Tensor& output) {
    const af::array scores = Scores(output);
    af::array values, indices;
    af::max(values, indices, scores(0, af::span), 1);
    return static_cast<int>(indices.scalar<unsigned>());
}

float ScoreOf(const Tensor& output, int target, int sample = 0) {
    return Scores(output)(sample, target).scalar<float>();
}

int CheckTarget(const Tensor& output, int target) {
    const af::array scores = Scores(output);
    if (target < 0) return Predicted(output);
    if (target >= scores.dims(1))
        throw std::invalid_argument("target class " + std::to_string(target) + " is outside the model's " +
                                    std::to_string(scores.dims(1)) + " classes");
    return target;
}

// `count` samples, the input's values per sample in the rows of `samples`
// (af [count, sample size]), laid out as the input's batch.
Tensor Batch(const af::array& samples, std::vector<size_t> shape) {
    const size_t count = static_cast<size_t>(samples.dims(0));
    shape[BatchAxis(shape)] = count;
    // Row-major storage: the batch axis last varies fastest ([count, F]
    // column-major); first, each sample is contiguous ([F, count]).
    const af::array flat = BatchAxis(shape) == 3 ? af::flat(samples) : af::flat(samples.T());
    return Tensor::FromRowMajorFlatArray(flat, shape);
}

// The rows of a batch tensor as af [count, sample size].
af::array Samples(const Tensor& batch) {
    const auto& shape = batch.Shape();
    const dim_t count = static_cast<dim_t>(shape[BatchAxis(shape)]);
    const dim_t size = static_cast<dim_t>(SampleSize(shape));
    const af::array flat = batch.GetRowMajorFlatArray();
    return BatchAxis(shape) == 3 ? af::moddims(flat, count, size) : af::moddims(flat, size, count).T();
}

void CheckOneSample(const Tensor& input) {
    const auto& shape = input.Shape();
    if (shape.empty() || shape[BatchAxis(shape)] != 1)
        throw std::invalid_argument("explain one sample at a time (a batch of 1)");
}

}  // namespace

// ============================================================================
// Utilities
// ============================================================================

std::vector<float> ModelInterpretability::ResizeBilinear(const std::vector<float>& map, size_t src_height,
                                                         size_t src_width, size_t dst_height, size_t dst_width) {
    if (map.size() != src_height * src_width || src_height == 0 || src_width == 0) return {};
    // Row-major [H, W] host data is column-major [W, H].
    const af::array source = af::array(static_cast<dim_t>(src_width), static_cast<dim_t>(src_height), map.data()).T();
    const auto axis = [](size_t in, size_t out, af::array& low, af::array& high, af::array& weight) {
        const float scale = static_cast<float>(in) / static_cast<float>(out);
        af::array position = (af::range(af::dim4(static_cast<dim_t>(out))).as(f32) + 0.5f) * scale - 0.5f;
        position = af::max(position, 0.0f);
        low = af::min(af::floor(position), static_cast<double>(in - 1)).as(u32);
        high = af::min(low.as(f32) + 1.0f, static_cast<double>(in - 1)).as(u32);
        weight = position - low.as(f32);
    };
    af::array y0, y1, wy, x0, x1, wx;
    axis(src_height, dst_height, y0, y1, wy);
    axis(src_width, dst_width, x0, x1, wx);
    const af::array top = af::lookup(source, y0, 0), bottom = af::lookup(source, y1, 0);
    const af::array rows = af::tile(1.0f - wy, 1, static_cast<unsigned>(dst_width)) *
                               (af::lookup(top, x0, 1) * af::tile((1.0f - wx).T(), static_cast<unsigned>(dst_height)) +
                                af::lookup(top, x1, 1) * af::tile(wx.T(), static_cast<unsigned>(dst_height))) +
                           af::tile(wy, 1, static_cast<unsigned>(dst_width)) *
                               (af::lookup(bottom, x0, 1) * af::tile((1.0f - wx).T(), static_cast<unsigned>(dst_height)) +
                                af::lookup(bottom, x1, 1) * af::tile(wx.T(), static_cast<unsigned>(dst_height)));
    return Host(rows.T());  // [W, H] column-major = [H, W] row-major
}

ExplainFunctions ModelInterpretability::FunctionsOf(SequentialModel& model) {
    ExplainFunctions functions;
    functions.forward = [&model](const Tensor& input) { return model.Forward(input); };
    functions.backward = [&model](const Tensor& grad) { return model.Backward(grad); };
    return functions;
}

// ============================================================================
// Layer Utilities
// ============================================================================

std::vector<std::string> ModelInterpretability::GetLayerNames(const SequentialModel& model) {
    std::vector<std::string> names;
    for (size_t i = 0; i < model.Size(); i++) {
        const Module* module = model.GetModule(i);
        if (module) names.push_back(module->GetName());
    }
    return names;
}

int ModelInterpretability::FindLayerIndex(const SequentialModel& model, const std::string& name) {
    for (size_t i = 0; i < model.Size(); i++) {
        const Module* module = model.GetModule(i);
        if (module && module->GetName() == name) return static_cast<int>(i);
    }
    return -1;
}

LayerActivations ModelInterpretability::ExtractLayerActivations(SequentialModel& model, const Tensor& input,
                                                                int layer_index) {
    LayerActivations result;
    result.layer_index = layer_index;
    if (layer_index < 0 || layer_index >= static_cast<int>(model.Size())) {
        result.error_message = "Invalid layer index";
        return result;
    }
    result.layer_name = model.GetModule(static_cast<size_t>(layer_index))->GetName();
    Tensor current = input;
    for (int i = 0; i <= layer_index; i++) current = model.GetModule(static_cast<size_t>(i))->Forward(current);
    result.shape = current.Shape();
    result.activations = Host(current.GetRowMajorFlatArray());
    result.success = true;
    return result;
}

LayerActivations ModelInterpretability::ExtractLayerActivations(SequentialModel& model, const Tensor& input,
                                                                const std::string& layer_name) {
    const int idx = FindLayerIndex(model, layer_name);
    if (idx < 0) {
        LayerActivations result;
        result.error_message = "Layer '" + layer_name + "' not found";
        return result;
    }
    return ExtractLayerActivations(model, input, idx);
}

// ============================================================================
// Grad-CAM
// ============================================================================

GradCAMResult ModelInterpretability::ComputeGradCAM(SequentialModel& model, const Tensor& input,
                                                    const std::string& target_layer, int target_class) {
    const int idx = FindLayerIndex(model, target_layer);
    if (idx < 0) {
        GradCAMResult result;
        result.error_message = "Target layer '" + target_layer + "' not found";
        return result;
    }
    return ComputeGradCAM(model, input, idx, target_class);
}

GradCAMResult ModelInterpretability::ComputeGradCAM(SequentialModel& model, const Tensor& input, int layer_index,
                                                    int target_class) {
    GradCAMResult result;
    try {
        if (layer_index < 0 || layer_index >= static_cast<int>(model.Size()))
            throw std::invalid_argument("Invalid layer index");
        CheckOneSample(input);
        result.layer_name = model.GetModule(static_cast<size_t>(layer_index))->GetName();
        // Forward module by module, keeping the target layer's output A.
        Tensor current = input, activation;
        for (size_t i = 0; i < model.Size(); ++i) {
            current = model.GetModule(i)->Forward(current);
            if (static_cast<int>(i) == layer_index) activation = current;
        }
        const auto& a_shape = activation.Shape();
        if (a_shape.size() != 4 || a_shape[3] != 1)
            throw std::invalid_argument("layer '" + result.layer_name + "' does not give an image-shaped output [H, W, C]");
        result.target_class = CheckTarget(current, target_class);
        result.class_score = ScoreOf(current, result.target_class);
        // d score / d A: back through the modules after the layer.
        Tensor grad = OneHot(current, result.target_class);
        for (size_t i = model.Size(); i-- > static_cast<size_t>(layer_index) + 1;) grad = model.GetModule(i)->Backward(grad);
        if (grad.Shape() != a_shape) throw std::runtime_error("the gradient at the layer has another shape than its output");

        const af::array a = activation.GetArrayRowMajor4D();  // (h, w, c, 1)
        const af::array g = grad.GetArrayRowMajor4D();
        const af::array weights = af::mean(af::mean(g, 0), 1);  // (1, 1, c, 1)
        af::array cam = af::sum(a * af::tile(weights, static_cast<unsigned>(a.dims(0)), static_cast<unsigned>(a.dims(1))), 2);
        cam = af::max(cam, 0.0f);  // (h, w)
        const size_t h = a_shape[0], w = a_shape[1];
        result.layer_shape = {h, w};
        result.layer_heatmap = Host(cam.T());  // row-major [h, w]

        const auto& in_shape = input.Shape();
        const size_t in_h = in_shape.size() == 4 ? in_shape[0] : h, in_w = in_shape.size() == 4 ? in_shape[1] : w;
        result.heatmap_shape = {in_h, in_w};
        result.raw_heatmap = ResizeBilinear(result.layer_heatmap, h, w, in_h, in_w);
        const auto [low, high] = std::minmax_element(result.raw_heatmap.begin(), result.raw_heatmap.end());
        const float span = *high - *low;
        result.heatmap.resize(result.raw_heatmap.size());
        for (size_t i = 0; i < result.raw_heatmap.size(); ++i)
            result.heatmap[i] = span > 0.0f ? (result.raw_heatmap[i] - *low) / span : 0.0f;
        result.success = true;
    } catch (const std::exception& error) {
        result.success = false;
        result.error_message = error.what();
    }
    return result;
}

// ============================================================================
// Saliency and SmoothGrad
// ============================================================================

SaliencyMap ModelInterpretability::ComputeSaliencyMap(SequentialModel& model, const Tensor& input, int target_class) {
    return ComputeSaliencyMap(FunctionsOf(model), input, target_class);
}

SaliencyMap ModelInterpretability::ComputeSaliencyMap(const ExplainFunctions& model, const Tensor& input,
                                                      int target_class) {
    SaliencyMap result;
    try {
        CheckOneSample(input);
        const Tensor output = model.forward(input);
        result.target_class = CheckTarget(output, target_class);
        const Tensor grad = model.backward(OneHot(output, result.target_class));
        result.shape = input.Shape();
        const af::array flat = grad.GetRowMajorFlatArray();
        result.gradients = Host(flat);
        result.absolute_gradients = Host(af::abs(flat));
        result.success = true;
    } catch (const std::exception& error) {
        result.error_message = error.what();
    }
    return result;
}

SaliencyMap ModelInterpretability::ComputeSmoothGrad(SequentialModel& model, const Tensor& input, int target_class,
                                                     int num_samples, float noise_std) {
    return ComputeSmoothGrad(FunctionsOf(model), input, target_class, num_samples, noise_std, 0);
}

SaliencyMap ModelInterpretability::ComputeSmoothGrad(const ExplainFunctions& model, const Tensor& input,
                                                     int target_class, int num_samples, float noise_std,
                                                     unsigned long long seed) {
    SaliencyMap result = ComputeSaliencyMap(model, input, target_class);
    if (!result.success) return result;
    try {
        const int count = std::max(1, num_samples);
        const af::array x = Samples(input);  // [1, F]
        af::randomEngine engine(AF_RANDOM_ENGINE_DEFAULT, seed);
        const af::array noisy = af::tile(x, count) + noise_std * af::randn(af::dim4(count, x.dims(1)), f32, engine);
        const Tensor batch = Batch(noisy, input.Shape());
        const Tensor output = model.forward(batch);
        const Tensor grad = model.backward(OneHot(output, result.target_class));
        result.smoothgrad = Host(af::mean(Samples(grad), 0));
    } catch (const std::exception& error) {
        result.success = false;
        result.error_message = error.what();
    }
    return result;
}

// ============================================================================
// Integrated Gradients
// ============================================================================

IntegratedGradientsResult ModelInterpretability::ComputeIntegratedGradients(const ExplainFunctions& model,
                                                                             const Tensor& input,
                                                                             const Tensor& baseline,
                                                                             int target_class, int steps) {
    IntegratedGradientsResult result;
    try {
        CheckOneSample(input);
        if (baseline.Shape() != input.Shape()) throw std::invalid_argument("the baseline must have the input's shape");
        result.steps = std::max(1, steps);
        result.shape = input.Shape();
        const Tensor output = model.forward(input);
        result.target_class = CheckTarget(output, target_class);
        result.score = ScoreOf(output, result.target_class);
        result.baseline_score = ScoreOf(model.forward(baseline), result.target_class);

        const af::array x = Samples(input), x0 = Samples(baseline);  // [1, F]
        const af::array alphas = (af::range(af::dim4(result.steps)).as(f32) + 0.5f) / static_cast<float>(result.steps);
        const af::array path = af::tile(x0, result.steps) +
                               af::tile(alphas, 1, static_cast<unsigned>(x.dims(1))) * af::tile(x - x0, result.steps);
        const Tensor batch = Batch(path, input.Shape());
        const Tensor path_output = model.forward(batch);
        const Tensor grad = model.backward(OneHot(path_output, result.target_class));
        const af::array attributions = (x - x0) * af::mean(Samples(grad), 0);
        result.attributions = Host(attributions);
        result.attribution_sum = af::sum<float>(attributions);
        result.success = true;
    } catch (const std::exception& error) {
        result.error_message = error.what();
    }
    return result;
}

}  // namespace cyxwiz
