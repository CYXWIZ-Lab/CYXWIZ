#include <cyxwiz/sequential.h>
#include <cyxwiz/debug_hooks.h>
#include <cyxwiz/layers/linear.h>
#include <cyxwiz/activations/relu.h>
#include <cyxwiz/activations/sigmoid.h>
#include <cyxwiz/activations/tanh.h>
#include <cyxwiz/activation.h>  // For LeakyReLUActivation, ELUActivation, GELUActivation, etc.
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <atomic>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <utility>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {

namespace {

std::string ShapeToStringForTrace(const std::vector<size_t>& shape) {
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) {
            out << ',';
        }
        out << shape[i];
    }
    out << ']';
    return out.str();
}

std::vector<size_t> UnravelIndex(size_t index, const std::vector<size_t>& shape) {
    std::vector<size_t> indices(shape.size(), 0);
    for (size_t i = shape.size(); i-- > 0;) {
        indices[i] = index % shape[i];
        index /= shape[i];
    }
    return indices;
}

size_t RavelIndex(const std::vector<size_t>& indices,
                  const std::vector<size_t>& shape) {
    size_t linear = 0;
    for (size_t i = 0; i < shape.size(); ++i) {
        linear = linear * shape[i] + indices[i];
    }
    return linear;
}

void EmitModelLayerTrace(const char* stage,
                         size_t layer_index,
                         const std::string& layer_name,
                         const std::vector<size_t>& input_shape,
                         const std::vector<size_t>& output_shape,
                         float duration_ms) {
    std::ostringstream message;
    message << "layer=" << layer_index
            << " name=" << layer_name
            << " input=" << ShapeToStringForTrace(input_shape)
            << " output=" << ShapeToStringForTrace(output_shape)
            << " duration_ms=" << duration_ms;
    BackendDebugHooks::EmitDebugEvent(stage, message.str());
}

} // namespace

// ============================================================================
// TransformerEncoderModule Implementation
// ============================================================================

namespace {

bool IsGradientParameterKey(const std::string& key) {
    if (key.rfind("grad_", 0) == 0) {
        return true;
    }
    return key.find(".grad_") != std::string::npos;
}

std::string NormalizeGradientParameterKey(std::string key) {
    if (key.rfind("grad_", 0) == 0) {
        key.erase(0, 5);
    }

    size_t pos = 0;
    while ((pos = key.find(".grad_", pos)) != std::string::npos) {
        key.replace(pos, 6, ".");
        ++pos;
    }

    return key;
}

} // namespace

TransformerEncoderModule::TransformerEncoderModule(size_t d_model,
                                                   size_t num_heads,
                                                   size_t dim_feedforward,
                                                   float dropout,
                                                   bool norm_first)
    : d_model_(d_model)
    , num_heads_(num_heads)
    , dim_feedforward_(dim_feedforward)
    , dropout_(dropout)
    , norm_first_(norm_first)
{
    if (d_model_ < 1) d_model_ = 1;
    if (num_heads_ < 1) num_heads_ = 1;
    if (d_model_ % num_heads_ != 0) {
        spdlog::warn("TransformerEncoderModule: d_model={} is not divisible "
                     "by num_heads={}; falling back to one head",
                     d_model_, num_heads_);
        num_heads_ = 1;
    }
    if (dim_feedforward_ < 1) dim_feedforward_ = d_model_;
    if (dropout_ < 0.0f) dropout_ = 0.0f;
    if (dropout_ >= 1.0f) dropout_ = 0.999f;

    layer_ = std::make_unique<TransformerEncoderLayer>(
        static_cast<int>(d_model_),
        static_cast<int>(num_heads_),
        static_cast<int>(dim_feedforward_),
        dropout_,
        norm_first_);
}

Tensor TransformerEncoderModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    return layer_->Forward(input);
}

Tensor TransformerEncoderModule::Backward(const Tensor& grad_output) {
    return layer_->Backward(grad_output);
}

void TransformerEncoderModule::SetTraining(bool training) {
    Module::SetTraining(training);
    layer_->SetTraining(training);
}

std::map<std::string, Tensor> TransformerEncoderModule::GetParameters() {
    std::map<std::string, Tensor> params;
    for (const auto& [key, value] : layer_->GetParameters()) {
        if (!IsGradientParameterKey(key)) {
            params[key] = value;
        }
    }
    return params;
}

void TransformerEncoderModule::SetParameters(
    const std::map<std::string, Tensor>& params) {
    layer_->SetParameters(params);
}

std::map<std::string, Tensor> TransformerEncoderModule::GetGradients() {
    std::map<std::string, Tensor> grads;
    for (const auto& [key, value] : layer_->GetParameters()) {
        if (IsGradientParameterKey(key)) {
            grads[NormalizeGradientParameterKey(key)] = value;
        }
    }
    return grads;
}

std::string TransformerEncoderModule::GetName() const {
    return "TransformerEncoder(d_model=" + std::to_string(d_model_) +
           ", heads=" + std::to_string(num_heads_) + ")";
}

// ============================================================================
// TransformerDecoderModule Implementation
// ============================================================================

TransformerDecoderModule::TransformerDecoderModule(size_t d_model,
                                                   size_t num_heads,
                                                   size_t dim_feedforward,
                                                   float dropout,
                                                   bool norm_first)
    : d_model_(d_model)
    , num_heads_(num_heads)
    , dim_feedforward_(dim_feedforward)
    , dropout_(dropout)
    , norm_first_(norm_first)
{
    if (d_model_ < 1) d_model_ = 1;
    if (num_heads_ < 1) num_heads_ = 1;
    if (d_model_ % num_heads_ != 0) {
        spdlog::warn("TransformerDecoderModule: d_model={} is not divisible "
                     "by num_heads={}; falling back to one head",
                     d_model_, num_heads_);
        num_heads_ = 1;
    }
    if (dim_feedforward_ < 1) dim_feedforward_ = d_model_;
    if (dropout_ < 0.0f) dropout_ = 0.0f;
    if (dropout_ >= 1.0f) dropout_ = 0.999f;

    layer_ = std::make_unique<TransformerDecoderLayer>(
        static_cast<int>(d_model_),
        static_cast<int>(num_heads_),
        static_cast<int>(dim_feedforward_),
        dropout_,
        norm_first_);
}

Tensor TransformerDecoderModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    return layer_->Forward(input);
}

Tensor TransformerDecoderModule::Backward(const Tensor& grad_output) {
    return layer_->Backward(grad_output);
}

void TransformerDecoderModule::SetTraining(bool training) {
    Module::SetTraining(training);
    layer_->SetTraining(training);
}

std::map<std::string, Tensor> TransformerDecoderModule::GetParameters() {
    std::map<std::string, Tensor> params;
    for (const auto& [key, value] : layer_->GetParameters()) {
        if (!IsGradientParameterKey(key)) {
            params[key] = value;
        }
    }
    return params;
}

void TransformerDecoderModule::SetParameters(
    const std::map<std::string, Tensor>& params) {
    layer_->SetParameters(params);
}

std::map<std::string, Tensor> TransformerDecoderModule::GetGradients() {
    std::map<std::string, Tensor> grads;
    for (const auto& [key, value] : layer_->GetParameters()) {
        if (IsGradientParameterKey(key)) {
            grads[NormalizeGradientParameterKey(key)] = value;
        }
    }
    return grads;
}

std::string TransformerDecoderModule::GetName() const {
    return "TransformerDecoder(d_model=" + std::to_string(d_model_) +
           ", heads=" + std::to_string(num_heads_) + ")";
}

// ============================================================================
// BatchNormModule Implementation (BatchNorm1D for MLPs)
// ============================================================================

BatchNormModule::BatchNormModule(size_t num_features, float eps, float momentum)
    : num_features_(num_features)
    , eps_(eps)
    , momentum_(momentum)
{
    // Initialize gamma (scale) to 1, beta (shift) to 0
    gamma_ = Tensor({num_features}, DataType::Float32);
    beta_ = Tensor({num_features}, DataType::Float32);
    running_mean_ = Tensor({num_features}, DataType::Float32);
    running_var_ = Tensor({num_features}, DataType::Float32);
    grad_gamma_ = Tensor({num_features}, DataType::Float32);
    grad_beta_ = Tensor({num_features}, DataType::Float32);

#ifdef CYXWIZ_HAS_ARRAYFIRE
    gamma_ = Tensor(af::constant(1.0f, num_features));
    beta_ = Tensor(af::constant(0.0f, num_features));
    running_mean_ = Tensor(af::constant(0.0f, num_features));
    running_var_ = Tensor(af::constant(1.0f, num_features));
#else
    float* gamma_data = gamma_.Data<float>();
    float* beta_data = beta_.Data<float>();
    float* rm_data = running_mean_.Data<float>();
    float* rv_data = running_var_.Data<float>();
    for (size_t i = 0; i < num_features; ++i) {
        gamma_data[i] = 1.0f;
        beta_data[i] = 0.0f;
        rm_data[i] = 0.0f;
        rv_data[i] = 1.0f;
    }
#endif

    spdlog::debug("BatchNormModule({}) initialized", num_features);
}

Tensor BatchNormModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();

    const auto& shape = input.Shape();
    size_t batch_size = shape[0];
    size_t features = shape.size() > 1 ? shape[1] : shape[0];

#ifdef CYXWIZ_HAS_ARRAYFIRE
    af::array x = input.GetArray();
    // x is [batch, features] in our row-major view
    // ArrayFire sees it as [batch, features] with dims(0)=batch, dims(1)=features

    // gamma/beta are [features], we need them as [1, features] for broadcasting
    af::array gamma = af::moddims(gamma_.GetArray(), 1, features);
    af::array beta = af::moddims(beta_.GetArray(), 1, features);

    if (is_training_) {
        // Compute mean and variance per feature (across batch = dim 0)
        af::array mean = af::mean(x, 0);  // [1, features]
        af::array var = af::var(x, AF_VARIANCE_POPULATION, 0);  // [1, features]

        // Update running statistics
        af::array rm = af::moddims(running_mean_.GetArray(), 1, features);
        af::array rv = af::moddims(running_var_.GetArray(), 1, features);
        rm = (1.0f - momentum_) * rm + momentum_ * mean;
        rv = (1.0f - momentum_) * rv + momentum_ * var;
        running_mean_ = Tensor(af::flat(rm));
        running_var_ = Tensor(af::flat(rv));

        // Normalize: (x - mean) / sqrt(var + eps)
        af::array std_inv = 1.0f / af::sqrt(var + eps_);
        // Tile mean and std_inv to [batch, features]
        af::array mean_tiled = af::tile(mean, batch_size, 1);
        af::array std_inv_tiled = af::tile(std_inv, batch_size, 1);
        af::array x_norm = (x - mean_tiled) * std_inv_tiled;

        // Scale and shift: gamma * x_norm + beta
        af::array gamma_tiled = af::tile(gamma, batch_size, 1);
        af::array beta_tiled = af::tile(beta, batch_size, 1);
        af::array out = gamma_tiled * x_norm + beta_tiled;

        // Cache for backward
        normalized_ = Tensor(x_norm);
        std_inv_ = Tensor(af::flat(std_inv));
        batch_mean_ = Tensor(af::flat(mean));

        return Tensor(out);
    } else {
        // Inference mode: use running statistics
        af::array rm = af::moddims(running_mean_.GetArray(), 1, features);
        af::array rv = af::moddims(running_var_.GetArray(), 1, features);
        af::array std_inv = 1.0f / af::sqrt(rv + eps_);

        af::array rm_tiled = af::tile(rm, batch_size, 1);
        af::array std_inv_tiled = af::tile(std_inv, batch_size, 1);
        af::array x_norm = (x - rm_tiled) * std_inv_tiled;

        af::array gamma_tiled = af::tile(gamma, batch_size, 1);
        af::array beta_tiled = af::tile(beta, batch_size, 1);
        af::array out = gamma_tiled * x_norm + beta_tiled;

        return Tensor(out);
    }
#else
    // CPU fallback
    Tensor output({batch_size, features}, DataType::Float32);
    const float* x_data = input.Data<float>();
    float* out_data = output.Data<float>();
    const float* gamma_data = gamma_.Data<float>();
    const float* beta_data = beta_.Data<float>();

    if (is_training_) {
        // Compute mean per feature
        std::vector<float> mean(features, 0.0f);
        std::vector<float> var(features, 0.0f);

        for (size_t f = 0; f < features; ++f) {
            for (size_t b = 0; b < batch_size; ++b) {
                mean[f] += x_data[b * features + f];
            }
            mean[f] /= batch_size;
        }

        // Compute variance per feature
        for (size_t f = 0; f < features; ++f) {
            for (size_t b = 0; b < batch_size; ++b) {
                float diff = x_data[b * features + f] - mean[f];
                var[f] += diff * diff;
            }
            var[f] /= batch_size;
        }

        // Update running statistics
        float* rm_data = running_mean_.Data<float>();
        float* rv_data = running_var_.Data<float>();
        for (size_t f = 0; f < features; ++f) {
            rm_data[f] = (1.0f - momentum_) * rm_data[f] + momentum_ * mean[f];
            rv_data[f] = (1.0f - momentum_) * rv_data[f] + momentum_ * var[f];
        }

        // Normalize, scale, shift
        normalized_ = Tensor({batch_size, features}, DataType::Float32);
        std_inv_ = Tensor({features}, DataType::Float32);
        float* norm_data = normalized_.Data<float>();
        float* std_inv_data = std_inv_.Data<float>();

        for (size_t f = 0; f < features; ++f) {
            std_inv_data[f] = 1.0f / std::sqrt(var[f] + eps_);
        }

        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t f = 0; f < features; ++f) {
                float x_norm = (x_data[b * features + f] - mean[f]) * std_inv_data[f];
                norm_data[b * features + f] = x_norm;
                out_data[b * features + f] = x_norm * gamma_data[f] + beta_data[f];
            }
        }
    } else {
        // Inference mode
        const float* rm_data = running_mean_.Data<float>();
        const float* rv_data = running_var_.Data<float>();

        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t f = 0; f < features; ++f) {
                float std_inv = 1.0f / std::sqrt(rv_data[f] + eps_);
                float x_norm = (x_data[b * features + f] - rm_data[f]) * std_inv;
                out_data[b * features + f] = x_norm * gamma_data[f] + beta_data[f];
            }
        }
    }

    return output;
#endif
}

Tensor BatchNormModule::Backward(const Tensor& grad_output) {
    const auto& shape = grad_output.Shape();
    size_t batch_size = shape[0];
    size_t features = shape.size() > 1 ? shape[1] : shape[0];

#ifdef CYXWIZ_HAS_ARRAYFIRE
    af::array grad = grad_output.GetArray();
    // grad is [batch, features]

    // Get cached values - x_norm is [batch, features]
    af::array x_norm = normalized_.GetArray();

    // gamma is [features], reshape to [1, features] for broadcasting
    af::array gamma = af::moddims(gamma_.GetArray(), 1, features);
    af::array std_inv = af::moddims(std_inv_.GetArray(), 1, features);

    // Gradient w.r.t gamma and beta (sum along batch dimension = dim 0)
    af::array d_gamma = af::sum(grad * x_norm, 0);  // [1, features]
    af::array d_beta = af::sum(grad, 0);  // [1, features]

    grad_gamma_ = Tensor(af::flat(d_gamma));
    grad_beta_ = Tensor(af::flat(d_beta));

    // Gradient w.r.t input
    // d_x = (1/N) * std_inv * (N * d_y * gamma - sum(d_y * gamma) - x_norm * sum(d_y * gamma * x_norm))
    float N = static_cast<float>(batch_size);

    af::array gamma_tiled = af::tile(gamma, batch_size, 1);  // [batch, features]
    af::array d_x_norm = grad * gamma_tiled;

    af::array sum_d_x_norm = af::tile(af::sum(d_x_norm, 0), batch_size, 1);  // [batch, features]
    af::array sum_d_x_norm_x_norm = af::tile(af::sum(d_x_norm * x_norm, 0), batch_size, 1);  // [batch, features]

    af::array std_inv_tiled = af::tile(std_inv, batch_size, 1);  // [batch, features]
    af::array d_x = (1.0f / N) * std_inv_tiled *
                    (N * d_x_norm - sum_d_x_norm - x_norm * sum_d_x_norm_x_norm);

    return Tensor(d_x);
#else
    // CPU fallback
    Tensor grad_input({batch_size, features}, DataType::Float32);
    const float* grad_data = grad_output.Data<float>();
    const float* x_norm_data = normalized_.Data<float>();
    const float* std_inv_data = std_inv_.Data<float>();
    const float* gamma_data = gamma_.Data<float>();
    float* grad_input_data = grad_input.Data<float>();
    float* d_gamma_data = grad_gamma_.Data<float>();
    float* d_beta_data = grad_beta_.Data<float>();

    // Initialize gradients
    for (size_t f = 0; f < features; ++f) {
        d_gamma_data[f] = 0.0f;
        d_beta_data[f] = 0.0f;
    }

    // Compute d_gamma, d_beta
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t f = 0; f < features; ++f) {
            d_gamma_data[f] += grad_data[b * features + f] * x_norm_data[b * features + f];
            d_beta_data[f] += grad_data[b * features + f];
        }
    }

    // Compute d_x
    float N = static_cast<float>(batch_size);
    for (size_t f = 0; f < features; ++f) {
        float sum_d_x_norm = 0.0f;
        float sum_d_x_norm_x_norm = 0.0f;

        for (size_t b = 0; b < batch_size; ++b) {
            float d_x_norm = grad_data[b * features + f] * gamma_data[f];
            sum_d_x_norm += d_x_norm;
            sum_d_x_norm_x_norm += d_x_norm * x_norm_data[b * features + f];
        }

        for (size_t b = 0; b < batch_size; ++b) {
            float d_x_norm = grad_data[b * features + f] * gamma_data[f];
            grad_input_data[b * features + f] = (1.0f / N) * std_inv_data[f] *
                (N * d_x_norm - sum_d_x_norm - x_norm_data[b * features + f] * sum_d_x_norm_x_norm);
        }
    }

    return grad_input;
#endif
}

std::map<std::string, Tensor> BatchNormModule::GetParameters() {
    return {
        {"gamma", gamma_},
        {"beta", beta_}
    };
}

void BatchNormModule::SetParameters(const std::map<std::string, Tensor>& params) {
    if (params.count("gamma")) gamma_ = params.at("gamma");
    if (params.count("beta")) beta_ = params.at("beta");
    if (params.count("running_mean")) running_mean_ = params.at("running_mean");
    if (params.count("running_var")) running_var_ = params.at("running_var");
}

std::map<std::string, Tensor> BatchNormModule::GetGradients() {
    return {
        {"gamma", grad_gamma_},
        {"beta", grad_beta_}
    };
}

std::string BatchNormModule::GetName() const {
    return "BatchNorm(" + std::to_string(num_features_) + ")";
}

// ============================================================================
// SequentialModel Implementation
// ============================================================================

Tensor SequentialModel::Forward(const Tensor& input) {
    intermediate_outputs_.clear();
    intermediate_outputs_.reserve(modules_.size() + 1);

    Tensor current = input.Clone();
    intermediate_outputs_.push_back(input.Clone());  // Store input

    const bool trace_layers = BackendDebugHooks::HasDebugEventCallback();
    for (size_t i = 0; i < modules_.size(); ++i) {
        auto& module = modules_[i];
        const auto input_shape = trace_layers ? current.Shape() : std::vector<size_t>{};
        const auto layer_start = std::chrono::steady_clock::now();
        current = module->Forward(current);
        if (trace_layers) {
            const auto duration_ms = std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - layer_start).count();
            EmitModelLayerTrace("ModelForward", i, module->GetName(),
                                input_shape, current.Shape(), duration_ms);
        }
        intermediate_outputs_.push_back(current.Clone());
    }

    return current;
}

Tensor SequentialModel::Backward(const Tensor& grad_output) {
    Tensor grad = grad_output.Clone();

    // Backward through modules in reverse order
    const bool trace_layers = BackendDebugHooks::HasDebugEventCallback();
    for (int i = static_cast<int>(modules_.size()) - 1; i >= 0; --i) {
        const auto input_shape = trace_layers ? grad.Shape() : std::vector<size_t>{};
        const auto layer_start = std::chrono::steady_clock::now();
        grad = modules_[i]->Backward(grad);
        if (trace_layers) {
            const auto duration_ms = std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - layer_start).count();
            EmitModelLayerTrace("ModelBackward", static_cast<size_t>(i),
                                modules_[i]->GetName(), input_shape, grad.Shape(),
                                duration_ms);
        }
    }

    return grad;
}

std::map<std::string, Tensor> SequentialModel::GetParameters() {
    std::map<std::string, Tensor> all_params;

    for (size_t i = 0; i < modules_.size(); ++i) {
        // Skip frozen layers - their parameters won't be updated
        if (modules_[i]->HasParameters() && modules_[i]->IsTrainable()) {
            auto params = modules_[i]->GetParameters();
            for (auto& [key, tensor] : params) {
                all_params["layer" + std::to_string(i) + "." + key] = tensor;
            }
        }
    }

    return all_params;
}

void SequentialModel::SetParameters(const std::map<std::string, Tensor>& params) {
    // Group parameters by layer index
    std::map<size_t, std::map<std::string, Tensor>> layer_params;

    for (const auto& [key, tensor] : params) {
        // Parse "layerN.param_name"
        if (key.substr(0, 5) == "layer") {
            size_t dot_pos = key.find('.');
            if (dot_pos != std::string::npos) {
                size_t layer_idx = std::stoul(key.substr(5, dot_pos - 5));
                std::string param_name = key.substr(dot_pos + 1);
                layer_params[layer_idx][param_name] = tensor;
            }
        }
    }

    // Set parameters for each layer
    for (auto& [layer_idx, layer_param_map] : layer_params) {
        if (layer_idx < modules_.size()) {
            modules_[layer_idx]->SetParameters(layer_param_map);
        }
    }
}

std::map<std::string, Tensor> SequentialModel::GetGradients() {
    std::map<std::string, Tensor> all_grads;

    for (size_t i = 0; i < modules_.size(); ++i) {
        // Skip frozen layers - don't need their gradients
        if (modules_[i]->HasParameters() && modules_[i]->IsTrainable()) {
            auto grads = modules_[i]->GetGradients();
            for (auto& [key, tensor] : grads) {
                all_grads["layer" + std::to_string(i) + "." + key] = tensor;
            }
        }
    }

    return all_grads;
}

void SequentialModel::UpdateParameters(Optimizer* optimizer) {
    if (!optimizer) {
        spdlog::error("SequentialModel::UpdateParameters: No optimizer provided");
        return;
    }

    auto params = GetParameters();
    auto grads = GetGradients();

    optimizer->Step(params, grads);

    SetParameters(params);
}

void SequentialModel::SetTraining(bool training) {
    for (auto& module : modules_) {
        module->SetTraining(training);
    }
}

void SequentialModel::Summary() const {
    spdlog::info("SequentialModel Summary:");
    spdlog::info("========================");
    for (size_t i = 0; i < modules_.size(); ++i) {
        std::string frozen_marker = modules_[i]->IsTrainable() ? "" : " [FROZEN]";
        spdlog::info("  [{}] {}{}", i, modules_[i]->GetName(), frozen_marker);
    }
    spdlog::info("========================");
}

// ============================================================================
// Transfer Learning Methods
// ============================================================================

void SequentialModel::FreezeLayer(size_t layer_idx) {
    if (layer_idx < modules_.size()) {
        modules_[layer_idx]->Freeze();
        spdlog::debug("SequentialModel: Froze layer {} ({})", layer_idx, modules_[layer_idx]->GetName());
    }
}

void SequentialModel::FreezeUpTo(size_t layer_idx) {
    size_t limit = layer_idx < modules_.size() ? layer_idx : modules_.size();
    for (size_t i = 0; i < limit; ++i) {
        modules_[i]->Freeze();
    }
    if (layer_idx > 0) {
        spdlog::debug("SequentialModel: Froze layers 0 to {}", layer_idx - 1);
    }
}

void SequentialModel::FreezeExceptLast(size_t n) {
    if (modules_.size() > n) {
        FreezeUpTo(modules_.size() - n);
        spdlog::debug("SequentialModel: Froze all except last {} layers", n);
    }
}

void SequentialModel::UnfreezeAll() {
    for (auto& module : modules_) {
        module->Unfreeze();
    }
    spdlog::debug("SequentialModel: Unfroze all layers");
}

bool SequentialModel::IsLayerTrainable(size_t layer_idx) const {
    if (layer_idx < modules_.size()) {
        return modules_[layer_idx]->IsTrainable();
    }
    return false;
}

// ============================================================================
// Factory Function
// ============================================================================

std::unique_ptr<Module> CreateModule(
    ModuleType type,
    const std::map<std::string, std::string>& params)
{
    switch (type) {
        case ModuleType::Linear: {
            size_t in_features = 0;
            size_t out_features = 0;
            bool use_bias = true;

            if (params.count("in_features")) {
                in_features = std::stoul(params.at("in_features"));
            }
            if (params.count("out_features")) {
                out_features = std::stoul(params.at("out_features"));
            }
            if (params.count("units")) {
                out_features = std::stoul(params.at("units"));
            }
            if (params.count("use_bias")) {
                use_bias = params.at("use_bias") == "true";
            }

            if (in_features == 0 || out_features == 0) {
                spdlog::error("CreateModule: Linear requires in_features and out_features");
                return nullptr;
            }

            return std::make_unique<LinearModule>(in_features, out_features, use_bias);
        }

        case ModuleType::ReLU:
            return std::make_unique<ReLUModule>();

        case ModuleType::Sigmoid:
            return std::make_unique<SigmoidModule>();

        case ModuleType::Tanh:
            return std::make_unique<TanhModule>();

        case ModuleType::Softmax: {
            int dim = -1;
            if (params.count("dim")) {
                dim = std::stoi(params.at("dim"));
            }
            return std::make_unique<SoftmaxModule>(dim);
        }

        case ModuleType::Dropout: {
            float p = 0.5f;
            if (params.count("p")) {
                p = std::stof(params.at("p"));
            }
            if (params.count("rate")) {
                p = std::stof(params.at("rate"));
            }
            return std::make_unique<DropoutModule>(p);
        }

        case ModuleType::Flatten: {
            int start_dim = 1;
            if (params.count("start_dim")) {
                start_dim = std::stoi(params.at("start_dim"));
            }
            return std::make_unique<FlattenModule>(start_dim);
        }

        case ModuleType::BatchNorm: {
            size_t num_features = 0;
            float eps = 1e-5f;
            float momentum = 0.1f;

            if (params.count("num_features")) {
                num_features = std::stoul(params.at("num_features"));
            }
            if (params.count("features")) {
                num_features = std::stoul(params.at("features"));
            }
            if (params.count("eps")) {
                eps = std::stof(params.at("eps"));
            }
            if (params.count("momentum")) {
                momentum = std::stof(params.at("momentum"));
            }

            if (num_features == 0) {
                spdlog::error("CreateModule: BatchNorm requires num_features");
                return nullptr;
            }

            return std::make_unique<BatchNormModule>(num_features, eps, momentum);
        }

        case ModuleType::LeakyReLU: {
            float negative_slope = 0.01f;
            if (params.count("negative_slope")) {
                negative_slope = std::stof(params.at("negative_slope"));
            }
            return std::make_unique<LeakyReLUModule>(negative_slope);
        }

        case ModuleType::ELU: {
            float alpha = 1.0f;
            if (params.count("alpha")) {
                alpha = std::stof(params.at("alpha"));
            }
            return std::make_unique<ELUModule>(alpha);
        }

        case ModuleType::GELU:
            return std::make_unique<GELUModule>();

        case ModuleType::Swish:
            return std::make_unique<SwishModule>();

        case ModuleType::Mish:
            return std::make_unique<MishModule>();

        default:
            spdlog::error("CreateModule: Unknown module type {}", static_cast<int>(type));
            return nullptr;
    }
}

} // namespace cyxwiz






