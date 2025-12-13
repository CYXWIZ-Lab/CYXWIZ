#include <cyxwiz/sequential.h>
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
#include <chrono>
#include <ctime>
#include <filesystem>

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
    for (auto& [layer_idx, params] : layer_params) {
        if (layer_idx < modules_.size()) {
            modules_[layer_idx]->SetParameters(params);
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
// Serialization Implementation
// ============================================================================

using json = nlohmann::json;
namespace fs = std::filesystem;

// Helper: Get current timestamp as string
static std::string GetCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

// Helper: Convert DataType to string
static std::string DataTypeToString(DataType dtype) {
    switch (dtype) {
        case DataType::Float32: return "float32";
        case DataType::Float64: return "float64";
        case DataType::Int32: return "int32";
        case DataType::Int64: return "int64";
        case DataType::UInt8: return "uint8";
        default: return "float32";
    }
}

// Helper: Write tensor to binary stream
static void WriteTensor(std::ostream& os, const Tensor& tensor) {
    // Write shape
    auto shape = tensor.Shape();
    size_t ndims = shape.size();
    os.write(reinterpret_cast<const char*>(&ndims), sizeof(ndims));
    os.write(reinterpret_cast<const char*>(shape.data()), ndims * sizeof(size_t));

    // Write dtype
    DataType dtype = tensor.GetDataType();
    os.write(reinterpret_cast<const char*>(&dtype), sizeof(dtype));

    // Write data
    size_t num_bytes = tensor.NumBytes();
    os.write(reinterpret_cast<const char*>(&num_bytes), sizeof(num_bytes));
    os.write(reinterpret_cast<const char*>(tensor.Data()), num_bytes);
}

// Helper: Read tensor from binary stream
static Tensor ReadTensor(std::istream& is) {
    // Read shape
    size_t ndims;
    is.read(reinterpret_cast<char*>(&ndims), sizeof(ndims));
    std::vector<size_t> shape(ndims);
    is.read(reinterpret_cast<char*>(shape.data()), ndims * sizeof(size_t));

    // Read dtype
    DataType dtype;
    is.read(reinterpret_cast<char*>(&dtype), sizeof(dtype));

    // Read data
    size_t num_bytes;
    is.read(reinterpret_cast<char*>(&num_bytes), sizeof(num_bytes));

    Tensor tensor(shape, dtype);
    is.read(reinterpret_cast<char*>(tensor.Data()), num_bytes);

    return tensor;
}

bool SequentialModel::Save(const std::string& path) const {
    try {
        // Ensure path has .cyxmodel extension
        std::string file_path = path;
        if (file_path.size() < 9 || file_path.substr(file_path.size() - 9) != ".cyxmodel") {
            file_path += ".cyxmodel";
        }

        fs::path model_path(file_path);

        // Create directory if needed
        if (model_path.has_parent_path()) {
            fs::create_directories(model_path.parent_path());
        }

        // Prepare metadata JSON
        json meta;
        meta["metadata"]["name"] = model_name_;
        meta["metadata"]["description"] = model_description_;
        meta["metadata"]["created_at"] = GetCurrentTimestamp();
        meta["metadata"]["framework"] = "CyxWiz";
        meta["metadata"]["format_version"] = "2.0";

        // Save module info
        meta["modules"] = json::array();
        for (size_t i = 0; i < modules_.size(); ++i) {
            json module_info;
            module_info["index"] = i;
            module_info["name"] = modules_[i]->GetName();
            module_info["has_parameters"] = modules_[i]->HasParameters();
            module_info["trainable"] = modules_[i]->IsTrainable();

            if (modules_[i]->HasParameters()) {
                auto params = modules_[i]->GetParameters();
                json param_names = json::array();
                for (const auto& [name, tensor] : params) {
                    json param_info;
                    param_info["name"] = name;
                    param_info["shape"] = tensor.Shape();
                    param_info["dtype"] = DataTypeToString(tensor.GetDataType());
                    param_names.push_back(param_info);
                }
                module_info["parameters"] = param_names;
            }

            meta["modules"].push_back(module_info);
        }

        // Serialize JSON to string
        std::string json_str = meta.dump();

        // Open single .cyxmodel file
        std::ofstream file(file_path, std::ios::binary);
        if (!file) {
            spdlog::error("SequentialModel::Save: Failed to create file: {}", file_path);
            return false;
        }

        // Write header
        // Magic number: "CYXW" (4 bytes)
        const uint32_t magic = 0x43595857;
        file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));

        // Version: 2 for single-file format (4 bytes)
        const uint32_t version = 2;
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));

        // JSON length (8 bytes)
        uint64_t json_len = json_str.size();
        file.write(reinterpret_cast<const char*>(&json_len), sizeof(json_len));

        // JSON data
        file.write(json_str.c_str(), json_len);

        // Number of modules (8 bytes)
        size_t num_modules = modules_.size();
        file.write(reinterpret_cast<const char*>(&num_modules), sizeof(num_modules));

        // Write each module's parameters
        for (const auto& module : modules_) {
            auto params = module->GetParameters();
            size_t num_params = params.size();
            file.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));

            for (const auto& [name, tensor] : params) {
                // Write parameter name length and name
                size_t name_len = name.size();
                file.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
                file.write(name.c_str(), name_len);

                // Write tensor
                WriteTensor(file, tensor);
            }
        }

        file.close();
        spdlog::info("SequentialModel::Save: Saved model to {}", file_path);
        return true;

    } catch (const std::exception& e) {
        spdlog::error("SequentialModel::Save: Exception: {}", e.what());
        return false;
    }
}

bool SequentialModel::Load(const std::string& path) {
    try {
        // Determine file path - add .cyxmodel if not present
        std::string file_path = path;
        if (file_path.size() < 9 || file_path.substr(file_path.size() - 9) != ".cyxmodel") {
            file_path += ".cyxmodel";
        }

        // Open the .cyxmodel file
        std::ifstream file(file_path, std::ios::binary);
        if (!file) {
            spdlog::error("SequentialModel::Load: Failed to open file: {}", file_path);
            return false;
        }

        // Read and verify magic number
        uint32_t magic;
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (magic != 0x43595857) {
            spdlog::error("SequentialModel::Load: Invalid magic number (not a CyxWiz model file)");
            return false;
        }

        // Read version
        uint32_t version;
        file.read(reinterpret_cast<char*>(&version), sizeof(version));

        if (version != 2) {
            spdlog::error("SequentialModel::Load: Unsupported format version: {} (expected 2)", version);
            return false;
        }

        // Read JSON length and data
        uint64_t json_len;
        file.read(reinterpret_cast<char*>(&json_len), sizeof(json_len));

        std::string json_str(json_len, '\0');
        file.read(json_str.data(), json_len);

        // Parse JSON metadata
        json meta = json::parse(json_str);
        if (meta.contains("metadata")) {
            model_name_ = meta["metadata"].value("name", "");
            model_description_ = meta["metadata"].value("description", "");
        }

        // Read number of modules
        size_t num_modules;
        file.read(reinterpret_cast<char*>(&num_modules), sizeof(num_modules));

        if (num_modules != modules_.size()) {
            spdlog::error("SequentialModel::Load: Module count mismatch. Expected {}, got {}",
                         modules_.size(), num_modules);
            return false;
        }

        // Load each module's parameters
        for (size_t i = 0; i < num_modules; ++i) {
            size_t num_params;
            file.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));

            std::map<std::string, Tensor> params;
            for (size_t j = 0; j < num_params; ++j) {
                // Read parameter name
                size_t name_len;
                file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
                std::string name(name_len, '\0');
                file.read(name.data(), name_len);

                // Read tensor
                Tensor tensor = ReadTensor(file);
                params[name] = std::move(tensor);
            }

            modules_[i]->SetParameters(params);
        }

        file.close();
        spdlog::info("SequentialModel::Load: Loaded model from {} ({} modules)", file_path, num_modules);
        return true;

    } catch (const std::exception& e) {
        spdlog::error("SequentialModel::Load: Exception: {}", e.what());
        return false;
    }
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
