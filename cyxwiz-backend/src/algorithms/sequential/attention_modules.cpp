#include <cyxwiz/sequential.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <string>

namespace cyxwiz {

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

MultiHeadAttentionModule::MultiHeadAttentionModule(size_t embed_dim,
                                                   size_t num_heads,
                                                   float dropout,
                                                   bool use_bias)
    : embed_dim_(embed_dim)
    , num_heads_(num_heads)
    , dropout_(dropout)
    , use_bias_(use_bias)
{
    if (embed_dim_ < 1) embed_dim_ = 1;
    if (num_heads_ < 1) num_heads_ = 1;
    if (embed_dim_ % num_heads_ != 0) {
        spdlog::warn("MultiHeadAttentionModule: embed_dim={} is not divisible "
                     "by num_heads={}; falling back to one head",
                     embed_dim_, num_heads_);
        num_heads_ = 1;
    }
    dropout_ = std::clamp(dropout_, 0.0f, 0.999f);

    layer_ = std::make_unique<MultiHeadAttentionLayer>(
        static_cast<int>(embed_dim_),
        static_cast<int>(num_heads_),
        dropout_,
        use_bias_);
}

Tensor MultiHeadAttentionModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    return layer_->Forward(input);
}

Tensor MultiHeadAttentionModule::Backward(const Tensor& grad_output) {
    return layer_->Backward(grad_output);
}

void MultiHeadAttentionModule::SetTraining(bool training) {
    Module::SetTraining(training);
    layer_->SetTraining(training);
}

std::map<std::string, Tensor> MultiHeadAttentionModule::GetParameters() {
    std::map<std::string, Tensor> params;
    for (const auto& [key, value] : layer_->GetParameters()) {
        if (!IsGradientParameterKey(key)) {
            params[key] = value;
        }
    }
    return params;
}

void MultiHeadAttentionModule::SetParameters(
    const std::map<std::string, Tensor>& params) {
    layer_->SetParameters(params);
}

std::map<std::string, Tensor> MultiHeadAttentionModule::GetGradients() {
    std::map<std::string, Tensor> grads;
    for (const auto& [key, value] : layer_->GetParameters()) {
        if (IsGradientParameterKey(key)) {
            grads[NormalizeGradientParameterKey(key)] = value;
        }
    }
    return grads;
}

std::string MultiHeadAttentionModule::GetName() const {
    return "MultiHeadAttention(embed_dim=" + std::to_string(embed_dim_) +
           ", heads=" + std::to_string(num_heads_) + ")";
}

} // namespace cyxwiz
