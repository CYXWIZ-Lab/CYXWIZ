#include <cyxwiz/sequential.h>
#include <spdlog/spdlog.h>
#include <string>
#include <utility>

namespace cyxwiz {
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

} // namespace cyxwiz

