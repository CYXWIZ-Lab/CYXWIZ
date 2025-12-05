#include <cyxwiz/sequential.h>
#include <cyxwiz/layers/linear.h>
#include <cyxwiz/activations/relu.h>
#include <cyxwiz/activations/sigmoid.h>
#include <cyxwiz/activations/tanh.h>
#include <cyxwiz/activation.h>  // For LeakyReLUActivation, ELUActivation, GELUActivation, etc.
#include <spdlog/spdlog.h>
#include <cmath>
#include <random>
#include <algorithm>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {

// ============================================================================
// LinearModule Implementation
// ============================================================================

LinearModule::LinearModule(size_t in_features, size_t out_features, bool use_bias)
    : in_features_(in_features)
    , out_features_(out_features)
{
    layer_ = std::make_unique<LinearLayer>(in_features, out_features, use_bias);
}

Tensor LinearModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    return layer_->Forward(input);
}

Tensor LinearModule::Backward(const Tensor& grad_output) {
    return layer_->Backward(grad_output);
}

std::map<std::string, Tensor> LinearModule::GetParameters() {
    return layer_->GetParameters();
}

void LinearModule::SetParameters(const std::map<std::string, Tensor>& params) {
    layer_->SetParameters(params);
}

std::map<std::string, Tensor> LinearModule::GetGradients() {
    return layer_->GetGradients();
}

std::string LinearModule::GetName() const {
    return "Linear(" + std::to_string(in_features_) + " -> " + std::to_string(out_features_) + ")";
}

// ============================================================================
// ReLUModule Implementation
// ============================================================================

ReLUModule::ReLUModule() {
    activation_ = std::make_unique<ReLU>();
}

Tensor ReLUModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    return activation_->Forward(input);
}

Tensor ReLUModule::Backward(const Tensor& grad_output) {
    return activation_->Backward(grad_output, input_cache_);
}

// ============================================================================
// SigmoidModule Implementation
// ============================================================================

SigmoidModule::SigmoidModule() {
    activation_ = std::make_unique<Sigmoid>();
}

Tensor SigmoidModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    return activation_->Forward(input);
}

Tensor SigmoidModule::Backward(const Tensor& grad_output) {
    return activation_->Backward(grad_output, input_cache_);
}

// ============================================================================
// TanhModule Implementation
// ============================================================================

TanhModule::TanhModule() {
    activation_ = std::make_unique<Tanh>();
}

Tensor TanhModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    return activation_->Forward(input);
}

Tensor TanhModule::Backward(const Tensor& grad_output) {
    return activation_->Backward(grad_output, input_cache_);
}

// ============================================================================
// LeakyReLUModule Implementation
// ============================================================================

LeakyReLUModule::LeakyReLUModule(float negative_slope)
    : negative_slope_(negative_slope)
{
    activation_ = std::make_unique<LeakyReLUActivation>(negative_slope);
}

Tensor LeakyReLUModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    return activation_->Forward(input);
}

Tensor LeakyReLUModule::Backward(const Tensor& grad_output) {
    return activation_->Backward(grad_output, input_cache_);
}

std::string LeakyReLUModule::GetName() const {
    return "LeakyReLU(slope=" + std::to_string(negative_slope_) + ")";
}

// ============================================================================
// ELUModule Implementation
// ============================================================================

ELUModule::ELUModule(float alpha)
    : alpha_(alpha)
{
    activation_ = std::make_unique<ELUActivation>(alpha);
}

Tensor ELUModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    return activation_->Forward(input);
}

Tensor ELUModule::Backward(const Tensor& grad_output) {
    return activation_->Backward(grad_output, input_cache_);
}

std::string ELUModule::GetName() const {
    return "ELU(alpha=" + std::to_string(alpha_) + ")";
}

// ============================================================================
// GELUModule Implementation
// ============================================================================

GELUModule::GELUModule() {
    activation_ = std::make_unique<GELUActivation>();
}

Tensor GELUModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    return activation_->Forward(input);
}

Tensor GELUModule::Backward(const Tensor& grad_output) {
    return activation_->Backward(grad_output, input_cache_);
}

// ============================================================================
// SwishModule Implementation
// ============================================================================

SwishModule::SwishModule() {
    activation_ = std::make_unique<SwishActivation>();
}

Tensor SwishModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    return activation_->Forward(input);
}

Tensor SwishModule::Backward(const Tensor& grad_output) {
    return activation_->Backward(grad_output, input_cache_);
}

// ============================================================================
// MishModule Implementation
// ============================================================================

MishModule::MishModule() {
    activation_ = std::make_unique<MishActivation>();
}

Tensor MishModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    return activation_->Forward(input);
}

Tensor MishModule::Backward(const Tensor& grad_output) {
    return activation_->Backward(grad_output, input_cache_);
}

// ============================================================================
// SoftmaxModule Implementation (ArrayFire)
// ============================================================================

SoftmaxModule::SoftmaxModule(int dim) : dim_(dim) {}

Tensor SoftmaxModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();

#ifdef CYXWIZ_HAS_ARRAYFIRE
    // ArrayFire implementation
    af::array x = input.GetArray();

    // Softmax: exp(x - max) / sum(exp(x - max))
    // Compute along dim 1 (classes dimension) for [batch, classes] input
    // Note: Use (af::max) to prevent Windows macro conflict
    af::array max_vals = (af::max)(x, 1);  // [batch, 1]
    af::array x_shifted = x - af::tile(max_vals, 1, static_cast<unsigned>(x.dims(1)));  // Subtract max for stability
    af::array exp_x = af::exp(x_shifted);
    af::array sum_exp = af::sum(exp_x, 1);  // [batch, 1]
    af::array softmax = exp_x / af::tile(sum_exp, 1, static_cast<unsigned>(x.dims(1)));

    Tensor output(softmax);
    output_cache_ = output.Clone();
    return output;
#else
    // CPU fallback
    const auto& shape = input.Shape();
    size_t batch_size = shape[0];
    size_t num_classes = shape.size() > 1 ? shape[1] : shape[0];

    Tensor output({batch_size, num_classes}, DataType::Float32);
    const float* in_data = input.Data<float>();
    float* out_data = output.Data<float>();

    for (size_t b = 0; b < batch_size; ++b) {
        float max_val = in_data[b * num_classes];
        for (size_t c = 1; c < num_classes; ++c) {
            max_val = std::max(max_val, in_data[b * num_classes + c]);
        }
        float sum = 0.0f;
        for (size_t c = 0; c < num_classes; ++c) {
            out_data[b * num_classes + c] = std::exp(in_data[b * num_classes + c] - max_val);
            sum += out_data[b * num_classes + c];
        }
        for (size_t c = 0; c < num_classes; ++c) {
            out_data[b * num_classes + c] /= sum;
        }
    }
    output_cache_ = output.Clone();
    return output;
#endif
}

Tensor SoftmaxModule::Backward(const Tensor& grad_output) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    // ArrayFire implementation
    // Softmax backward: grad_input = softmax * (grad_output - sum(grad_output * softmax))
    af::array grad = grad_output.GetArray();
    af::array soft = output_cache_.GetArray();

    // Compute dot product per sample: sum(grad * softmax) along classes dimension
    af::array dot = af::sum(grad * soft, 1);  // [batch, 1]

    // grad_input = softmax * (grad - dot)
    af::array grad_input = soft * (grad - af::tile(dot, 1, static_cast<unsigned>(grad.dims(1))));

    return Tensor(grad_input);
#else
    // CPU fallback
    const auto& shape = grad_output.Shape();
    size_t batch_size = shape[0];
    size_t num_classes = shape.size() > 1 ? shape[1] : shape[0];

    Tensor grad_input({batch_size, num_classes}, DataType::Float32);
    const float* grad_data = grad_output.Data<float>();
    const float* soft_data = output_cache_.Data<float>();
    float* out_data = grad_input.Data<float>();

    for (size_t b = 0; b < batch_size; ++b) {
        float dot = 0.0f;
        for (size_t c = 0; c < num_classes; ++c) {
            dot += grad_data[b * num_classes + c] * soft_data[b * num_classes + c];
        }
        for (size_t c = 0; c < num_classes; ++c) {
            out_data[b * num_classes + c] = soft_data[b * num_classes + c] *
                (grad_data[b * num_classes + c] - dot);
        }
    }
    return grad_input;
#endif
}

// ============================================================================
// DropoutModule Implementation (ArrayFire)
// ============================================================================

DropoutModule::DropoutModule(float p) : p_(p) {
    if (p < 0.0f || p > 1.0f) {
        spdlog::warn("DropoutModule: p={} out of range [0,1], clamping", p);
        p_ = std::clamp(p, 0.0f, 1.0f);
    }
}

Tensor DropoutModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();

    // During eval, just return input
    if (!is_training_) {
        return input.Clone();
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    // ArrayFire implementation
    af::array x = input.GetArray();
    float scale = 1.0f / (1.0f - p_);

    // Generate random mask: values > p are kept (scaled), values <= p are dropped
    af::array rand_vals = af::randu(x.dims());
    af::array keep_mask = (rand_vals > p_).as(af::dtype::f32);  // 1 for keep, 0 for drop
    af::array scaled_mask = keep_mask * scale;

    // Store mask for backward pass
    mask_ = Tensor(scaled_mask);

    // Apply dropout
    af::array output = x * scaled_mask;
    return Tensor(output);
#else
    // CPU fallback
    const auto& shape = input.Shape();
    size_t total = input.NumElements();

    Tensor output(shape, input.GetDataType());
    mask_ = Tensor(shape, DataType::Float32);

    const float* in_data = input.Data<float>();
    float* out_data = output.Data<float>();
    float* mask_data = mask_.Data<float>();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float scale = 1.0f / (1.0f - p_);

    for (size_t i = 0; i < total; ++i) {
        if (dist(gen) > p_) {
            mask_data[i] = scale;
            out_data[i] = in_data[i] * scale;
        } else {
            mask_data[i] = 0.0f;
            out_data[i] = 0.0f;
        }
    }
    return output;
#endif
}

Tensor DropoutModule::Backward(const Tensor& grad_output) {
    if (!is_training_) {
        return grad_output.Clone();
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    // ArrayFire implementation
    af::array grad = grad_output.GetArray();
    af::array mask = mask_.GetArray();

    // grad_input = grad * mask (mask already has scaling applied)
    af::array grad_input = grad * mask;
    return Tensor(grad_input);
#else
    // CPU fallback
    const auto& shape = grad_output.Shape();
    Tensor grad_input(shape, DataType::Float32);

    const float* grad_data = grad_output.Data<float>();
    const float* mask_data = mask_.Data<float>();
    float* out_data = grad_input.Data<float>();

    size_t total = grad_output.NumElements();
    for (size_t i = 0; i < total; ++i) {
        out_data[i] = grad_data[i] * mask_data[i];
    }
    return grad_input;
#endif
}

std::string DropoutModule::GetName() const {
    return "Dropout(p=" + std::to_string(p_) + ")";
}

// ============================================================================
// FlattenModule Implementation (ArrayFire)
// ============================================================================

FlattenModule::FlattenModule(int start_dim) : start_dim_(start_dim) {}

Tensor FlattenModule::Forward(const Tensor& input) {
    original_shape_ = input.Shape();

    // Calculate flattened size from start_dim onwards
    size_t batch_size = 1;
    size_t flat_size = 1;

    for (size_t i = 0; i < original_shape_.size(); ++i) {
        if (static_cast<int>(i) < start_dim_) {
            batch_size *= original_shape_[i];
        } else {
            flat_size *= original_shape_[i];
        }
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    // ArrayFire implementation - use moddims for zero-copy reshape
    af::array x = input.GetArray();

    // Reshape to [flat_size, batch_size] (ArrayFire is column-major)
    // Then we return as [batch_size, flat_size] for row-major semantics
    af::array flattened = af::moddims(x, flat_size, batch_size);

    return Tensor(flattened);
#else
    // CPU fallback
    Tensor output({batch_size, flat_size}, input.GetDataType());

    const float* in_data = input.Data<float>();
    float* out_data = output.Data<float>();
    std::copy(in_data, in_data + input.NumElements(), out_data);

    return output;
#endif
}

Tensor FlattenModule::Backward(const Tensor& grad_output) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    // ArrayFire implementation - reshape gradient back to original shape
    af::array grad = grad_output.GetArray();

    // Convert original_shape_ to af::dim4
    af::dim4 dims(1, 1, 1, 1);
    for (size_t i = 0; i < original_shape_.size() && i < 4; ++i) {
        dims[i] = static_cast<dim_t>(original_shape_[i]);
    }

    // Reshape back to original dimensions
    af::array grad_input = af::moddims(grad, dims);

    return Tensor(grad_input);
#else
    // CPU fallback
    Tensor grad_input(original_shape_, grad_output.GetDataType());

    const float* grad_data = grad_output.Data<float>();
    float* out_data = grad_input.Data<float>();
    std::copy(grad_data, grad_data + grad_output.NumElements(), out_data);

    return grad_input;
#endif
}

// ============================================================================
// SequentialModel Implementation
// ============================================================================

Tensor SequentialModel::Forward(const Tensor& input) {
    intermediate_outputs_.clear();
    intermediate_outputs_.reserve(modules_.size() + 1);

    Tensor current = input.Clone();
    intermediate_outputs_.push_back(input.Clone());  // Store input

    for (auto& module : modules_) {
        current = module->Forward(current);
        intermediate_outputs_.push_back(current.Clone());
    }

    return current;
}

Tensor SequentialModel::Backward(const Tensor& grad_output) {
    Tensor grad = grad_output.Clone();

    // Backward through modules in reverse order
    for (int i = static_cast<int>(modules_.size()) - 1; i >= 0; --i) {
        grad = modules_[i]->Backward(grad);
    }

    return grad;
}

std::map<std::string, Tensor> SequentialModel::GetParameters() {
    std::map<std::string, Tensor> all_params;

    for (size_t i = 0; i < modules_.size(); ++i) {
        if (modules_[i]->HasParameters()) {
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
    for (auto& [layer_idx, params] : layer_params) {
        if (layer_idx < modules_.size()) {
            modules_[layer_idx]->SetParameters(params);
        }
    }
}

std::map<std::string, Tensor> SequentialModel::GetGradients() {
    std::map<std::string, Tensor> all_grads;

    for (size_t i = 0; i < modules_.size(); ++i) {
        if (modules_[i]->HasParameters()) {
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
        spdlog::info("  [{}] {}", i, modules_[i]->GetName());
    }
    spdlog::info("========================");
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

        default:
            spdlog::error("CreateModule: Unknown module type {}", static_cast<int>(type));
            return nullptr;
    }
}

} // namespace cyxwiz
