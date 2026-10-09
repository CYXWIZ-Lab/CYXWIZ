#include <cyxwiz/sequential.h>
#include <stdexcept>
#include <cyxwiz/layers/transformer.h>
#include "attention_configuration.h"

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
    : MultiHeadAttentionModule(embed_dim, num_heads, dropout, use_bias, MultiHeadAttentionOptions{}) {}

MultiHeadAttentionModule::MultiHeadAttentionModule(size_t embed_dim, size_t num_heads, float dropout,
                                                   bool use_bias, const MultiHeadAttentionOptions& options)
    : embed_dim_(embed_dim)
    , num_heads_(num_heads)
    , dropout_(dropout)
    , use_bias_(use_bias)
    , options_(options)
{
    if ((options_.alibi || options_.sliding_window > 0) && !options_.causal) {
        throw std::invalid_argument("MultiHeadAttention alibi and sliding_window need causal=true");
    }
    if (options_.rope && options_.alibi) {
        throw std::invalid_argument("MultiHeadAttention cannot use rope and alibi together");
    }
    if (options_.sliding_window < 0) {
        throw std::invalid_argument("MultiHeadAttention sliding_window must be >= 0");
    }
    const int heads = attention_configuration_detail::CheckedAttentionDimension(num_heads_, "num_heads");
    layer_ = std::make_unique<MultiHeadAttentionLayer>(
        attention_configuration_detail::CheckedAttentionDimension(embed_dim_, "embed_dim"),
        heads,
        dropout_,
        use_bias_,
        options_.num_kv_heads > 0 ? options_.num_kv_heads : heads);
    if (options_.rope) layer_->SetRotaryEmbedding(true, options_.rope_base, options_.rope_fraction);
    if (options_.alibi) layer_->SetAlibi(true);
    if (options_.qk_norm) layer_->SetQKNorm(true, options_.qk_norm_eps);
    layer_->SetLogitSoftcap(options_.logit_softcap);
    // Forward passes the standard causal mask, or no mask when not causal.
    layer_->DeclareStandardMask(options_.causal, options_.sliding_window);
}

Tensor MultiHeadAttentionModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    if (!options_.causal) {
        return layer_->Forward(input);
    }
    const auto& shape = input.Shape();
    if (shape.size() != 3) {
        throw std::invalid_argument("MultiHeadAttention expects [batch, seq_len, embed_dim] input");
    }
    const Tensor mask = TransformerDecoderLayer::GenerateCausalMask(static_cast<int>(shape[1]),
                                                                     options_.sliding_window);
    return layer_->Forward(input, input, input, &mask);
}

Tensor MultiHeadAttentionModule::ForwardIncremental(const Tensor& input, size_t position_offset) {
    if (!options_.causal) {
        throw std::runtime_error("MultiHeadAttention incremental decoding needs causal=true");
    }
    layer_->BeginIncremental(position_offset, options_.sliding_window);
    try {
        Tensor out = layer_->Forward(input, input, input, nullptr);
        layer_->EndIncremental();
        return out;
    } catch (...) {
        layer_->EndIncremental();
        throw;
    }
}

void MultiHeadAttentionModule::ResetIncrementalState() {
    layer_->ResetKvCache();
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

CrossAttentionModule::CrossAttentionModule(size_t embed_dim, size_t num_heads, float dropout, bool use_bias)
    : embed_dim_(embed_dim), num_heads_(num_heads) {
    layer_ = std::make_unique<MultiHeadAttentionLayer>(
        attention_configuration_detail::CheckedAttentionDimension(embed_dim_, "embed_dim"),
        attention_configuration_detail::CheckedAttentionDimension(num_heads_, "num_heads"), dropout, use_bias);
}

Tensor CrossAttentionModule::Forward(const Tensor& /*input*/) {
    throw std::invalid_argument("CrossAttention needs its Query, Key and Value inputs (ForwardInputs)");
}

Tensor CrossAttentionModule::Backward(const Tensor& /*grad_output*/) {
    throw std::invalid_argument("CrossAttention returns one gradient per input (BackwardInputs)");
}

Tensor CrossAttentionModule::ForwardInputs(const std::vector<Tensor>& inputs) {
    if (inputs.size() != 3) {
        throw std::invalid_argument("CrossAttention needs 3 inputs (query, key, value), got " +
                                    std::to_string(inputs.size()));
    }
    // Distinct tensor objects: the layer takes the cross-attention path and
    // keeps dK and dV apart (it only folds them into dQ for self-attention).
    return layer_->Forward(inputs[0], inputs[1], inputs[2], nullptr);
}

std::vector<Tensor> CrossAttentionModule::BackwardInputs(const Tensor& grad_output) {
    Tensor query_gradient = layer_->Backward(grad_output);
    return {std::move(query_gradient), layer_->GetLastKeyGradient(), layer_->GetLastValueGradient()};
}

void CrossAttentionModule::SetTraining(bool training) {
    Module::SetTraining(training);
    layer_->SetTraining(training);
}

std::map<std::string, Tensor> CrossAttentionModule::GetParameters() {
    std::map<std::string, Tensor> params;
    for (const auto& [key, value] : layer_->GetParameters()) {
        if (!IsGradientParameterKey(key)) params[key] = value;
    }
    return params;
}

void CrossAttentionModule::SetParameters(const std::map<std::string, Tensor>& params) {
    layer_->SetParameters(params);
}

std::map<std::string, Tensor> CrossAttentionModule::GetGradients() {
    std::map<std::string, Tensor> grads;
    for (const auto& [key, value] : layer_->GetParameters()) {
        if (IsGradientParameterKey(key)) grads[NormalizeGradientParameterKey(key)] = value;
    }
    return grads;
}

std::string CrossAttentionModule::GetName() const {
    return "CrossAttention(embed_dim=" + std::to_string(embed_dim_) + ", heads=" + std::to_string(num_heads_) + ")";
}

std::string MultiHeadAttentionModule::GetName() const {
    return "MultiHeadAttention(embed_dim=" + std::to_string(embed_dim_) +
           ", heads=" + std::to_string(num_heads_) + ")";
}

} // namespace cyxwiz
