#include "cyxwiz/layers/transformer.h"
#include "cyxwiz/activations/relu.h"
#include "../arrayfire_backend_utils.h"

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

#include <cmath>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

namespace cyxwiz {

namespace {

void ValidateTransformerSequence(const Tensor& input, int width) {
    const auto& shape = input.Shape();
    if (input.GetDataType() != DataType::Float32 || shape.size() != 3 ||
        shape[0] == 0 || shape[1] == 0 || shape[2] != static_cast<size_t>(width)) {
        throw std::invalid_argument("Transformer expects nonempty Float32 [batch, seq_len, d_model]");
    }
}

Tensor FlattenTransformerSequenceForDense(const Tensor& input) {
    const auto& shape = input.Shape();
    if (shape.size() != 3) {
        return input;
    }
    return input.Reshape({input.NumElements() / shape[2], shape[2]});
}

Tensor RestoreTransformerSequenceFromDense(const Tensor& input,
                                           size_t batch,
                                           size_t seq_len) {
    const auto& shape = input.Shape();
    if (shape.size() != 2) {
        return input;
    }
    return input.Reshape({batch, seq_len, shape[1]});
}

Tensor AddSameShape(const Tensor& lhs, const Tensor& rhs) {
    if (lhs.GetDataType() != DataType::Float32 ||
        rhs.GetDataType() != DataType::Float32 ||
        lhs.Shape() != rhs.Shape()) {
        throw std::runtime_error("Transformer residual backward shape mismatch");
    }

    return lhs + rhs;
}

} // namespace

TransformerEncoderLayer::TransformerEncoderLayer(int d_model, int nhead,
                                                   int dim_feedforward, float dropout,
                                                   bool norm_first)
    : d_model_(d_model), nhead_(nhead), dim_feedforward_(dim_feedforward),
      dropout_(dropout), norm_first_(norm_first) {

    self_attn_ = std::make_unique<MultiHeadAttentionLayer>(d_model, nhead, dropout);
    norm1_ = std::make_unique<LayerNormLayer>(std::vector<int>{d_model});
    norm2_ = std::make_unique<LayerNormLayer>(std::vector<int>{d_model});
    linear1_ = std::make_unique<DenseLayer>(d_model, dim_feedforward);
    linear2_ = std::make_unique<DenseLayer>(dim_feedforward, d_model);
    dropout1_ = std::make_unique<DropoutLayer>(dropout);
    dropout2_ = std::make_unique<DropoutLayer>(dropout);
}

Tensor TransformerEncoderLayer::Forward(const Tensor& input) {
    return Forward(input, nullptr);
}

Tensor TransformerEncoderLayer::Forward(const Tensor& input, const Tensor* src_mask) {
    ValidateTransformerSequence(input, d_model_);
    cached_input_ = input;

    if (norm_first_) {
        // Pre-LN: x + attn(norm(x))
        Tensor normed = norm1_->Forward(input);
        Tensor attn_out = self_attn_->Forward(normed, normed, normed, src_mask);
        attn_out = dropout1_->Forward(attn_out);
        cached_residual1_ = input;

        // Add residual
        const auto& shape = input.Shape();
        Tensor x = AddSameShape(input, attn_out);
        cached_attn_output_ = x;

        // FFN
        Tensor normed2 = norm2_->Forward(x);
        Tensor ffn_out = linear1_->Forward(
            FlattenTransformerSequenceForDense(normed2));

        // ReLU activation
        ffn_out = ReLU().Forward(ffn_out);
        cached_ffn_mid_ = ffn_out;

        ffn_out = linear2_->Forward(ffn_out);
        ffn_out = RestoreTransformerSequenceFromDense(ffn_out, shape[0], shape[1]);
        ffn_out = dropout2_->Forward(ffn_out);
        cached_residual2_ = x;

        // Add residual
        Tensor result = AddSameShape(x, ffn_out);

        return result;
    } else {
        // Post-LN: norm(x + attn(x))
        Tensor attn_out = self_attn_->Forward(input, input, input, src_mask);
        attn_out = dropout1_->Forward(attn_out);
        cached_residual1_ = input;

        // Add residual and norm
        const auto& shape = input.Shape();
        Tensor x = AddSameShape(input, attn_out);

        x = norm1_->Forward(x);
        cached_attn_output_ = x;

        // FFN
        Tensor ffn_out = linear1_->Forward(
            FlattenTransformerSequenceForDense(x));

        // ReLU activation
        ffn_out = ReLU().Forward(ffn_out);
        cached_ffn_mid_ = ffn_out;

        ffn_out = linear2_->Forward(ffn_out);
        ffn_out = RestoreTransformerSequenceFromDense(ffn_out, shape[0], shape[1]);
        ffn_out = dropout2_->Forward(ffn_out);
        cached_residual2_ = x;

        // Add residual and norm
        Tensor result = AddSameShape(x, ffn_out);

        return norm2_->Forward(result);
    }
}

Tensor TransformerEncoderLayer::Backward(const Tensor& grad_output) {
    Tensor grad = grad_output;

    if (!norm_first_) {
        grad = norm2_->Backward(grad);
    }

    Tensor grad_ffn = dropout2_->Backward(grad);
    const auto& grad_shape = grad_output.Shape();
    if (grad_shape.size() == 3) {
        grad_ffn = FlattenTransformerSequenceForDense(grad_ffn);
    }
    grad_ffn = linear2_->Backward(grad_ffn);

    grad_ffn = ReLU().Backward(grad_ffn, cached_ffn_mid_);

    grad_ffn = linear1_->Backward(grad_ffn);
    if (grad_shape.size() == 3) {
        grad_ffn = RestoreTransformerSequenceFromDense(
            grad_ffn, grad_shape[0], grad_shape[1]);
    }

    if (norm_first_) {
        grad_ffn = norm2_->Backward(grad_ffn);
    }

    Tensor grad_after_ffn_residual = AddSameShape(grad, grad_ffn);
    Tensor grad_before_attn_residual = grad_after_ffn_residual;
    if (!norm_first_) {
        grad_before_attn_residual = norm1_->Backward(grad_after_ffn_residual);
    }

    Tensor grad_attn = dropout1_->Backward(grad_before_attn_residual);
    grad_attn = self_attn_->Backward(grad_attn);
    if (norm_first_) {
        grad_attn = norm1_->Backward(grad_attn);
    }

    return AddSameShape(grad_before_attn_residual, grad_attn);
}

std::map<std::string, Tensor> TransformerEncoderLayer::GetParameters() {
    std::map<std::string, Tensor> params;

    auto attn_params = self_attn_->GetParameters();
    for (const auto& [key, val] : attn_params) {
        params["self_attn." + key] = val;
    }

    auto norm1_params = norm1_->GetParameters();
    for (const auto& [key, val] : norm1_params) {
        params["norm1." + key] = val;
    }

    auto norm2_params = norm2_->GetParameters();
    for (const auto& [key, val] : norm2_params) {
        params["norm2." + key] = val;
    }

    auto linear1_params = linear1_->GetParameters();
    for (const auto& [key, val] : linear1_params) {
        params["linear1." + key] = val;
    }

    auto linear2_params = linear2_->GetParameters();
    for (const auto& [key, val] : linear2_params) {
        params["linear2." + key] = val;
    }

    return params;
}

void TransformerEncoderLayer::SetParameters(const std::map<std::string, Tensor>& params) {
    std::map<std::string, Tensor> attn_params, norm1_params, norm2_params;
    std::map<std::string, Tensor> linear1_params, linear2_params;

    for (const auto& [key, val] : params) {
        if (key.find("self_attn.") == 0) {
            attn_params[key.substr(10)] = val;
        } else if (key.find("norm1.") == 0) {
            norm1_params[key.substr(6)] = val;
        } else if (key.find("norm2.") == 0) {
            norm2_params[key.substr(6)] = val;
        } else if (key.find("linear1.") == 0) {
            linear1_params[key.substr(8)] = val;
        } else if (key.find("linear2.") == 0) {
            linear2_params[key.substr(8)] = val;
        }
    }

    self_attn_->SetParameters(attn_params);
    norm1_->SetParameters(norm1_params);
    norm2_->SetParameters(norm2_params);
    linear1_->SetParameters(linear1_params);
    linear2_->SetParameters(linear2_params);
}

void TransformerEncoderLayer::SetTraining(bool training) {
    training_ = training;
    self_attn_->SetTraining(training);
    norm1_->SetTraining(training);
    norm2_->SetTraining(training);
    linear1_->SetTraining(training);
    linear2_->SetTraining(training);
    dropout1_->SetTraining(training);
    dropout2_->SetTraining(training);
}

// ============================================================================
// TransformerDecoderLayer Implementation
// ============================================================================

TransformerDecoderLayer::TransformerDecoderLayer(int d_model, int nhead,
                                                   int dim_feedforward, float dropout,
                                                   bool norm_first)
    : d_model_(d_model), nhead_(nhead), dim_feedforward_(dim_feedforward),
      dropout_(dropout), norm_first_(norm_first) {

    self_attn_ = std::make_unique<MultiHeadAttentionLayer>(d_model, nhead, dropout);
    cross_attn_ = std::make_unique<MultiHeadAttentionLayer>(d_model, nhead, dropout);
    norm1_ = std::make_unique<LayerNormLayer>(std::vector<int>{d_model});
    norm2_ = std::make_unique<LayerNormLayer>(std::vector<int>{d_model});
    norm3_ = std::make_unique<LayerNormLayer>(std::vector<int>{d_model});
    linear1_ = std::make_unique<DenseLayer>(d_model, dim_feedforward);
    linear2_ = std::make_unique<DenseLayer>(dim_feedforward, d_model);
    dropout1_ = std::make_unique<DropoutLayer>(dropout);
    dropout2_ = std::make_unique<DropoutLayer>(dropout);
    dropout3_ = std::make_unique<DropoutLayer>(dropout);
}

Tensor TransformerDecoderLayer::Forward(const Tensor& input) {
    // Decoder-only mode: causal self-attention + feed-forward, no encoder memory.
    const auto& shape = input.Shape();
    if (shape.size() != 3) {
        throw std::invalid_argument("TransformerDecoderLayer expects [batch, seq_len, d_model] input");
    }
    ValidateTransformerSequence(input, d_model_);
    cached_input_ = input;
    cached_memory_ = Tensor();
    cached_has_cross_attention_ = false;

    if (shape[1] > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("TransformerDecoderLayer sequence exceeds causal-mask size limit");
    }
    Tensor causal_mask = GenerateCausalMask(static_cast<int>(shape[1]));

    if (norm_first_) {
        Tensor normed = norm1_->Forward(input);
        Tensor self_attn_out = self_attn_->Forward(normed, normed, normed, &causal_mask);
        self_attn_out = dropout1_->Forward(self_attn_out);

        Tensor x = AddSameShape(input, self_attn_out);
        cached_self_attn_output_ = x;
        cached_cross_attn_output_ = x;

        Tensor normed2 = norm2_->Forward(x);
        Tensor ffn_out = linear1_->Forward(
            FlattenTransformerSequenceForDense(normed2));

        ffn_out = ReLU().Forward(ffn_out);
        cached_ffn_mid_ = ffn_out;

        ffn_out = linear2_->Forward(ffn_out);
        ffn_out = RestoreTransformerSequenceFromDense(ffn_out, shape[0], shape[1]);
        ffn_out = dropout3_->Forward(ffn_out);

        Tensor result = AddSameShape(x, ffn_out);
        return result;
    }

    Tensor self_attn_out = self_attn_->Forward(input, input, input, &causal_mask);
    self_attn_out = dropout1_->Forward(self_attn_out);

    Tensor x = AddSameShape(input, self_attn_out);
    x = norm1_->Forward(x);
    cached_self_attn_output_ = x;
    cached_cross_attn_output_ = x;

    Tensor ffn_out = linear1_->Forward(
        FlattenTransformerSequenceForDense(x));

    ffn_out = ReLU().Forward(ffn_out);
    cached_ffn_mid_ = ffn_out;

    ffn_out = linear2_->Forward(ffn_out);
    ffn_out = RestoreTransformerSequenceFromDense(ffn_out, shape[0], shape[1]);
    ffn_out = dropout3_->Forward(ffn_out);

    Tensor result = AddSameShape(x, ffn_out);

    return norm2_->Forward(result);
}

Tensor TransformerDecoderLayer::Forward(const Tensor& tgt, const Tensor& memory,
                                         const Tensor* tgt_mask, const Tensor* memory_mask) {
    ValidateTransformerSequence(tgt, d_model_);
    ValidateTransformerSequence(memory, d_model_);
    cached_input_ = tgt;
    cached_memory_ = memory;
    cached_has_cross_attention_ = true;

    const auto& shape = tgt.Shape();

    if (norm_first_) {
        // Pre-LN decoder

        // Self-attention
        Tensor normed = norm1_->Forward(tgt);
        Tensor self_attn_out = self_attn_->Forward(normed, normed, normed, tgt_mask);
        self_attn_out = dropout1_->Forward(self_attn_out);

        // Residual
        Tensor x = AddSameShape(tgt, self_attn_out);
        cached_self_attn_output_ = x;

        // Cross-attention
        Tensor normed2 = norm2_->Forward(x);
        Tensor cross_attn_out = cross_attn_->Forward(normed2, memory, memory, memory_mask);
        cross_attn_out = dropout2_->Forward(cross_attn_out);

        // Residual
        Tensor x2 = AddSameShape(x, cross_attn_out);
        cached_cross_attn_output_ = x2;

        // FFN
        Tensor normed3 = norm3_->Forward(x2);
        Tensor ffn_out = linear1_->Forward(
            FlattenTransformerSequenceForDense(normed3));

        // ReLU
        ffn_out = ReLU().Forward(ffn_out);
        cached_ffn_mid_ = ffn_out;

        ffn_out = linear2_->Forward(ffn_out);
        ffn_out = RestoreTransformerSequenceFromDense(ffn_out, shape[0], shape[1]);
        ffn_out = dropout3_->Forward(ffn_out);

        // Residual
        Tensor result = AddSameShape(x2, ffn_out);

        return result;
    } else {
        // Post-LN decoder

        // Self-attention
        Tensor self_attn_out = self_attn_->Forward(tgt, tgt, tgt, tgt_mask);
        self_attn_out = dropout1_->Forward(self_attn_out);

        // Residual + norm
        Tensor x = AddSameShape(tgt, self_attn_out);
        x = norm1_->Forward(x);
        cached_self_attn_output_ = x;

        // Cross-attention
        Tensor cross_attn_out = cross_attn_->Forward(x, memory, memory, memory_mask);
        cross_attn_out = dropout2_->Forward(cross_attn_out);

        // Residual + norm
        Tensor x2 = AddSameShape(x, cross_attn_out);
        x2 = norm2_->Forward(x2);
        cached_cross_attn_output_ = x2;

        // FFN
        Tensor ffn_out = linear1_->Forward(
            FlattenTransformerSequenceForDense(x2));

        // ReLU
        ffn_out = ReLU().Forward(ffn_out);
        cached_ffn_mid_ = ffn_out;

        ffn_out = linear2_->Forward(ffn_out);
        ffn_out = RestoreTransformerSequenceFromDense(ffn_out, shape[0], shape[1]);
        ffn_out = dropout3_->Forward(ffn_out);

        // Residual + norm
        Tensor result = AddSameShape(x2, ffn_out);

        return norm3_->Forward(result);
    }
}

Tensor TransformerDecoderLayer::Backward(const Tensor& grad_output) {
    Tensor grad = grad_output;

    if (!norm_first_) {
        grad = (cached_has_cross_attention_ ? norm3_ : norm2_)->Backward(grad);
    }

    Tensor grad_ffn = dropout3_->Backward(grad);
    const auto& grad_shape = grad_output.Shape();
    if (grad_shape.size() == 3) {
        grad_ffn = FlattenTransformerSequenceForDense(grad_ffn);
    }
    grad_ffn = linear2_->Backward(grad_ffn);

    grad_ffn = ReLU().Backward(grad_ffn, cached_ffn_mid_);

    grad_ffn = linear1_->Backward(grad_ffn);
    if (grad_shape.size() == 3) {
        grad_ffn = RestoreTransformerSequenceFromDense(
            grad_ffn, grad_shape[0], grad_shape[1]);
    }

    if (norm_first_) {
        grad_ffn = (cached_has_cross_attention_ ? norm3_ : norm2_)->Backward(grad_ffn);
    }

    Tensor grad_after_ffn_residual = AddSameShape(grad, grad_ffn);
    Tensor grad_before_self_residual = grad_after_ffn_residual;

    if (cached_has_cross_attention_) {
        Tensor grad_cross_norm_input = grad_after_ffn_residual;
        if (!norm_first_) {
            grad_cross_norm_input = norm2_->Backward(grad_after_ffn_residual);
        }

        Tensor grad_cross = dropout2_->Backward(grad_cross_norm_input);
        grad_cross = cross_attn_->Backward(grad_cross);
        if (norm_first_) {
            grad_cross = norm2_->Backward(grad_cross);
        }

        grad_before_self_residual = AddSameShape(grad_cross_norm_input, grad_cross);
    }

    if (!norm_first_) {
        grad_before_self_residual = norm1_->Backward(grad_before_self_residual);
    }

    Tensor grad_self = dropout1_->Backward(grad_before_self_residual);
    grad_self = self_attn_->Backward(grad_self);
    if (norm_first_) {
        grad_self = norm1_->Backward(grad_self);
    }

    return AddSameShape(grad_before_self_residual, grad_self);
}

Tensor TransformerDecoderLayer::GetLastMemoryGradient() const {
    if (!cached_has_cross_attention_) {
        throw std::runtime_error(
            "TransformerDecoderLayer memory gradient requires a forward pass with encoder memory");
    }

    const Tensor grad_key = cross_attn_->GetLastKeyGradient();
    const Tensor grad_value = cross_attn_->GetLastValueGradient();
    if (grad_key.Shape().empty() || grad_value.Shape().empty()) {
        throw std::runtime_error(
            "TransformerDecoderLayer memory gradient requires backward after forward_with_memory");
    }

    return AddSameShape(grad_key, grad_value);
}
std::map<std::string, Tensor> TransformerDecoderLayer::GetParameters() {
    std::map<std::string, Tensor> params;

    auto self_attn_params = self_attn_->GetParameters();
    for (const auto& [key, val] : self_attn_params) {
        params["self_attn." + key] = val;
    }

    auto cross_attn_params = cross_attn_->GetParameters();
    for (const auto& [key, val] : cross_attn_params) {
        params["cross_attn." + key] = val;
    }

    auto norm1_params = norm1_->GetParameters();
    for (const auto& [key, val] : norm1_params) {
        params["norm1." + key] = val;
    }

    auto norm2_params = norm2_->GetParameters();
    for (const auto& [key, val] : norm2_params) {
        params["norm2." + key] = val;
    }

    auto norm3_params = norm3_->GetParameters();
    for (const auto& [key, val] : norm3_params) {
        params["norm3." + key] = val;
    }

    auto linear1_params = linear1_->GetParameters();
    for (const auto& [key, val] : linear1_params) {
        params["linear1." + key] = val;
    }

    auto linear2_params = linear2_->GetParameters();
    for (const auto& [key, val] : linear2_params) {
        params["linear2." + key] = val;
    }

    return params;
}

void TransformerDecoderLayer::SetParameters(const std::map<std::string, Tensor>& params) {
    std::map<std::string, Tensor> self_attn_params, cross_attn_params;
    std::map<std::string, Tensor> norm1_params, norm2_params, norm3_params;
    std::map<std::string, Tensor> linear1_params, linear2_params;

    for (const auto& [key, val] : params) {
        if (key.find("self_attn.") == 0) {
            self_attn_params[key.substr(10)] = val;
        } else if (key.find("cross_attn.") == 0) {
            cross_attn_params[key.substr(11)] = val;
        } else if (key.find("norm1.") == 0) {
            norm1_params[key.substr(6)] = val;
        } else if (key.find("norm2.") == 0) {
            norm2_params[key.substr(6)] = val;
        } else if (key.find("norm3.") == 0) {
            norm3_params[key.substr(6)] = val;
        } else if (key.find("linear1.") == 0) {
            linear1_params[key.substr(8)] = val;
        } else if (key.find("linear2.") == 0) {
            linear2_params[key.substr(8)] = val;
        }
    }

    self_attn_->SetParameters(self_attn_params);
    cross_attn_->SetParameters(cross_attn_params);
    norm1_->SetParameters(norm1_params);
    norm2_->SetParameters(norm2_params);
    norm3_->SetParameters(norm3_params);
    linear1_->SetParameters(linear1_params);
    linear2_->SetParameters(linear2_params);
}

void TransformerDecoderLayer::SetTraining(bool training) {
    training_ = training;
    self_attn_->SetTraining(training);
    cross_attn_->SetTraining(training);
    norm1_->SetTraining(training);
    norm2_->SetTraining(training);
    norm3_->SetTraining(training);
    linear1_->SetTraining(training);
    linear2_->SetTraining(training);
    dropout1_->SetTraining(training);
    dropout2_->SetTraining(training);
    dropout3_->SetTraining(training);
}

Tensor TransformerDecoderLayer::GenerateCausalMask(int size) {
    if (size <= 0) {
        throw std::invalid_argument("TransformerDecoderLayer causal mask size must be positive");
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const af::dim4 dims(size, size);
        const af::array mask = (af::range(dims, 1, u32) > af::range(dims, 0, u32)).as(f32) * -1e9f;
        mask.eval(); // Complete mask generation inside its fallback boundary.
        return Tensor::FromSemanticArray(mask, {static_cast<size_t>(size), static_cast<size_t>(size)});
    } catch (const af::exception& e) {
        ThrowIfArrayFireNativeCpuFallbackForbidden(
            "TransformerDecoderLayer::GenerateCausalMask",
            ClassifyArrayFireBackendFallbackReason(e.what()), e.what(),
            BuildArrayFireBackendFallbackContext("size=" + std::to_string(size)));
    }
#endif
    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LayerCpuPath, "TransformerDecoderLayer::GenerateCausalMask");
    // Preserve the existing finite additive mask contract above the diagonal.
    std::vector<size_t> shape = {static_cast<size_t>(size), static_cast<size_t>(size)};
    Tensor mask(shape, DataType::Float32);
    float* data = mask.MutableData<float>();

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (j > i) {
                data[i * size + j] = -1e9f;  // Large negative for softmax
            } else {
                data[i * size + j] = 0.0f;
            }
        }
    }

    return mask;
}

} // namespace cyxwiz
