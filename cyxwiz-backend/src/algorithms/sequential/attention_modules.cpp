#include <cyxwiz/sequential.h>
#include <stdexcept>
#include <cyxwiz/layers/transformer.h>
#include "attention_configuration.h"

#include <cmath>
#include <limits>
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

namespace {

constexpr float kLinearAttentionUnbounded = std::numeric_limits<float>::max();

}  // namespace

LinearAttentionModule::LinearAttentionModule(size_t embed_dim, size_t num_heads, FeatureMap feature_map, float eps,
                                             bool causal, bool use_bias)
    : embed_dim_(embed_dim), num_heads_(num_heads), head_dim_(0), feature_map_(feature_map), eps_(eps),
      causal_(causal), use_bias_(use_bias) {
    attention_configuration_detail::CheckedAttentionDimension(embed_dim_, "embed_dim");
    attention_configuration_detail::CheckedAttentionDimension(num_heads_, "num_heads");
    if (embed_dim_ % num_heads_ != 0) {
        throw std::invalid_argument("LinearAttention embed_dim must be divisible by num_heads");
    }
    if (!std::isfinite(eps_) || eps_ <= 0.0f) {
        throw std::invalid_argument("LinearAttention eps must be a positive finite number");
    }
    head_dim_ = embed_dim_ / num_heads_;
    // Xavier uniform weights, zero biases (as MultiHeadAttentionLayer).
    const float limit = std::sqrt(6.0f / static_cast<float>(embed_dim_ + embed_dim_));
    for (const char* name : {"q", "k", "v", "o"}) {
        params_[std::string("W_") + name] = Tensor::Random({embed_dim_, embed_dim_}) * (2.0f * limit) - limit;
        if (use_bias_) params_[std::string("b_") + name] = Tensor::Zeros({embed_dim_});
    }
}

// rows [N, E] x W^T [E, E] (+ b).
Tensor LinearAttentionModule::Project(const Tensor& rows, const std::string& name) const {
    const size_t n = rows.Shape()[0];
    Tensor out = rows.Reshape({1, n, embed_dim_})
                     .BatchMatMul(params_.at("W_" + name).Transpose().Reshape({1, embed_dim_, embed_dim_}))
                     .Reshape({n, embed_dim_});
    return use_bias_ ? out + params_.at("b_" + name) : out;
}

// [B * T, E] -> [B * H, T, d].
Tensor LinearAttentionModule::SplitHeads(const Tensor& rows) const {
    const size_t batch = input_shape_[0], seq = input_shape_[1];
    return rows.Reshape({batch, seq, num_heads_, head_dim_})
        .Permute({0, 2, 1, 3})
        .Reshape({batch * num_heads_, seq, head_dim_});
}

// [B * H, T, d] -> [B * T, E].
Tensor LinearAttentionModule::JoinHeads(const Tensor& heads) const {
    const size_t batch = input_shape_[0], seq = input_shape_[1];
    return heads.Reshape({batch, num_heads_, seq, head_dim_})
        .Permute({0, 2, 1, 3})
        .Reshape({batch * seq, embed_dim_});
}

Tensor LinearAttentionModule::Forward(const Tensor& input) {
    const auto& shape = input.Shape();
    if (input.GetDataType() != DataType::Float32 || shape.size() != 3 || shape[2] != embed_dim_ || shape[0] == 0 ||
        shape[1] == 0) {
        throw std::invalid_argument("LinearAttention expects Float32 [batch, seq_len, " +
                                    std::to_string(embed_dim_) + "] input");
    }
    input_shape_ = shape;
    const size_t seq = shape[1];
    rows_ = input.Reshape({shape[0] * seq, embed_dim_});
    q_pre_ = SplitHeads(Project(rows_, "q"));
    k_pre_ = SplitHeads(Project(rows_, "k"));
    v_ = SplitHeads(Project(rows_, "v"));
    const auto phi = [this](const Tensor& x) {
        Tensor positive = x.Clip(0.0f, kLinearAttentionUnbounded);
        return feature_map_ == FeatureMap::Relu ? positive
                                                : positive + x.Clip(-kLinearAttentionUnbounded, 0.0f).Exp();
    };
    phi_q_ = phi(q_pre_);
    phi_k_ = phi(k_pre_);
    Tensor numerator;
    if (causal_) {
        // A = (phi(Q) phi(K)^T) masked to j <= i; out = A V / (A 1 + eps).
        if (mask_.Shape() != std::vector<size_t>{seq, seq}) {
            mask_ = Tensor::Zeros({seq, seq});
            for (size_t i = 0; i < seq; ++i) {
                for (size_t j = 0; j <= i; ++j) mask_.Set(i, j, 1.0f);
            }
        }
        weights_ = phi_q_.BatchMatMul(phi_k_.Transpose(1, 2)) * mask_;
        numerator = weights_.BatchMatMul(v_);
        den_ = weights_.Sum(2, true) + eps_;
    } else {
        // S = phi(K)^T V [d, d] and z = sum_j phi(k_j) once per head.
        summary_ = phi_k_.Transpose(1, 2).BatchMatMul(v_);
        key_sum_ = phi_k_.Sum(1, true);
        numerator = phi_q_.BatchMatMul(summary_);
        den_ = (phi_q_ * key_sum_).Sum(2, true) + eps_;
    }
    out_heads_ = numerator / den_;
    context_ = JoinHeads(out_heads_);
    return Project(context_, "o").Reshape(shape);
}

Tensor LinearAttentionModule::Backward(const Tensor& grad_output) {
    if (grad_output.Shape() != input_shape_ || rows_.Shape().empty()) {
        throw std::invalid_argument("LinearAttention backward needs a forward pass with the same shape first");
    }
    const size_t rows = input_shape_[0] * input_shape_[1];
    const auto matmul = [](const Tensor& a, const Tensor& b) {
        return a.Reshape({1, a.Shape()[0], a.Shape()[1]})
            .BatchMatMul(b.Reshape({1, b.Shape()[0], b.Shape()[1]}))
            .Reshape({a.Shape()[0], b.Shape()[1]});
    };
    // d(rows W^T + b): dW = dY^T X, db = sum dY, dX = dY W.
    const auto project_backward = [&](const Tensor& dy, const Tensor& x, const std::string& name) {
        grads_["W_" + name] = matmul(dy.Transpose(), x);
        if (use_bias_) grads_["b_" + name] = dy.Sum(0);
        return matmul(dy, params_.at("W_" + name));
    };
    const Tensor dy = grad_output.Reshape({rows, embed_dim_});
    const Tensor d_out = SplitHeads(project_backward(dy, context_, "o"));
    // out = num / den: d num = G / den, d den = -sum(G * out) / den.
    const Tensor d_num = d_out / den_;
    const Tensor d_den = -((d_out * out_heads_).Sum(2, true) / den_);
    Tensor d_phi_q, d_phi_k, d_v;
    if (causal_) {
        const Tensor d_weights = (d_num.BatchMatMul(v_.Transpose(1, 2)) + d_den) * mask_;
        d_v = weights_.Transpose(1, 2).BatchMatMul(d_num);
        d_phi_q = d_weights.BatchMatMul(phi_k_);
        d_phi_k = d_weights.Transpose(1, 2).BatchMatMul(phi_q_);
    } else {
        const Tensor d_summary = phi_q_.Transpose(1, 2).BatchMatMul(d_num);
        const Tensor d_key_sum = (phi_q_ * d_den).Sum(1, true);
        d_phi_q = d_num.BatchMatMul(summary_.Transpose(1, 2)) + d_den * key_sum_;
        d_phi_k = v_.BatchMatMul(d_summary.Transpose(1, 2)) + d_key_sum;
        d_v = phi_k_.BatchMatMul(d_summary);
    }
    // phi'(x): 1 for x > 0; exp(x) (elu + 1) or 0 (relu) otherwise.
    const auto phi_grad = [this](const Tensor& x) {
        return feature_map_ == FeatureMap::Relu ? x.Clip(0.0f, kLinearAttentionUnbounded).Sign()
                                                : x.Clip(-kLinearAttentionUnbounded, 0.0f).Exp();
    };
    const Tensor dq = JoinHeads(d_phi_q * phi_grad(q_pre_));
    const Tensor dk = JoinHeads(d_phi_k * phi_grad(k_pre_));
    const Tensor dx = project_backward(dq, rows_, "q") + project_backward(dk, rows_, "k") +
                      project_backward(JoinHeads(d_v), rows_, "v");
    return dx.Reshape(input_shape_);
}

std::map<std::string, Tensor> LinearAttentionModule::GetParameters() {
    return params_;
}

void LinearAttentionModule::SetParameters(const std::map<std::string, Tensor>& params) {
    for (const auto& [key, value] : params) {
        const auto found = params_.find(key);
        if (found == params_.end()) continue;
        if (value.Shape() != found->second.Shape()) {
            throw std::invalid_argument("LinearAttention parameter " + key + " has the wrong shape");
        }
        found->second = value;
    }
}

std::map<std::string, Tensor> LinearAttentionModule::GetGradients() {
    return grads_;
}

std::string LinearAttentionModule::GetName() const {
    return "LinearAttention(embed_dim=" + std::to_string(embed_dim_) + ", heads=" + std::to_string(num_heads_) +
           (causal_ ? ", causal" : "") + ")";
}

std::string MultiHeadAttentionModule::GetName() const {
    return "MultiHeadAttention(embed_dim=" + std::to_string(embed_dim_) +
           ", heads=" + std::to_string(num_heads_) + ")";
}

} // namespace cyxwiz
