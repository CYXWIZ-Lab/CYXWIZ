#include <cyxwiz/sequential.h>
#include "attention_configuration.h"
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
    : TransformerEncoderModule(d_model, num_heads, dim_feedforward, dropout, norm_first, 0.0f) {}

TransformerEncoderModule::TransformerEncoderModule(size_t d_model, size_t num_heads,
    size_t dim_feedforward, float dropout, bool norm_first, float ffn_dropout)
    : d_model_(d_model)
    , num_heads_(num_heads)
    , dim_feedforward_(dim_feedforward)
    , dropout_(dropout)
    , norm_first_(norm_first)
{
    layer_ = std::make_unique<TransformerEncoderLayer>(
        attention_configuration_detail::CheckedAttentionDimension(d_model_, "d_model"),
        attention_configuration_detail::CheckedAttentionDimension(num_heads_, "num_heads"),
        attention_configuration_detail::CheckedAttentionDimension(dim_feedforward_, "dim_feedforward"),
        dropout_,
        norm_first_, ffn_dropout);
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
    : TransformerDecoderModule(d_model, num_heads, dim_feedforward, dropout, norm_first, 0.0f) {}

TransformerDecoderModule::TransformerDecoderModule(size_t d_model, size_t num_heads,
    size_t dim_feedforward, float dropout, bool norm_first, float ffn_dropout)
    : d_model_(d_model)
    , num_heads_(num_heads)
    , dim_feedforward_(dim_feedforward)
    , dropout_(dropout)
    , norm_first_(norm_first)
{
    layer_ = std::make_unique<TransformerDecoderLayer>(
        attention_configuration_detail::CheckedAttentionDimension(d_model_, "d_model"),
        attention_configuration_detail::CheckedAttentionDimension(num_heads_, "num_heads"),
        attention_configuration_detail::CheckedAttentionDimension(dim_feedforward_, "dim_feedforward"),
        dropout_,
        norm_first_, ffn_dropout);
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

