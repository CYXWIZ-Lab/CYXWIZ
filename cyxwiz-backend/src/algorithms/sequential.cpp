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

        case ModuleType::LayerNorm: {
            std::vector<int> normalized_shape;
            float eps = 1e-5f;
            bool elementwise_affine = true;

            if (params.count("normalized_shape")) {
                normalized_shape.push_back(std::stoi(params.at("normalized_shape")));
            }
            if (params.count("features")) {
                normalized_shape = {std::stoi(params.at("features"))};
            }
            if (params.count("eps")) {
                eps = std::stof(params.at("eps"));
            }
            if (params.count("epsilon")) {
                eps = std::stof(params.at("epsilon"));
            }
            if (params.count("elementwise_affine")) {
                elementwise_affine = params.at("elementwise_affine") != "false" &&
                                     params.at("elementwise_affine") != "0";
            }

            if (normalized_shape.empty()) {
                spdlog::error("CreateModule: LayerNorm requires normalized_shape");
                return nullptr;
            }

            return std::make_unique<LayerNormModule>(
                normalized_shape, eps, elementwise_affine);
        }

        case ModuleType::MultiHeadAttention: {
            size_t embed_dim = 0;
            size_t num_heads = 1;
            float dropout = 0.0f;
            bool use_bias = true;

            if (params.count("embed_dim")) {
                embed_dim = std::stoul(params.at("embed_dim"));
            }
            if (params.count("d_model")) {
                embed_dim = std::stoul(params.at("d_model"));
            }
            if (params.count("num_heads")) {
                num_heads = std::stoul(params.at("num_heads"));
            }
            if (params.count("heads")) {
                num_heads = std::stoul(params.at("heads"));
            }
            if (params.count("dropout")) {
                dropout = std::stof(params.at("dropout"));
            }
            if (params.count("use_bias")) {
                use_bias = params.at("use_bias") != "false" &&
                           params.at("use_bias") != "0";
            }

            if (embed_dim == 0) {
                spdlog::error("CreateModule: MultiHeadAttention requires embed_dim");
                return nullptr;
            }

            return std::make_unique<MultiHeadAttentionModule>(
                embed_dim, num_heads, dropout, use_bias);
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







