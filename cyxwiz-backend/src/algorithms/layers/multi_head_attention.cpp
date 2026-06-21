#include "cyxwiz/layer.h"
#include "layer_arrayfire_utils.h"

#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

#include <spdlog/spdlog.h>

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

namespace cyxwiz {

MultiHeadAttentionLayer::MultiHeadAttentionLayer(int embed_dim, int num_heads,
                                                   float dropout, bool use_bias)
    : embed_dim_(embed_dim), num_heads_(num_heads), dropout_(dropout), use_bias_(use_bias) {

    if (embed_dim_ <= 0 || num_heads_ <= 0 || dropout_ < 0.0f || dropout_ >= 1.0f ||
        embed_dim_ % num_heads_ != 0) {
        throw std::invalid_argument("MultiHeadAttention requires positive divisible dims and dropout in [0, 1)");
    }

    head_dim_ = embed_dim / num_heads;
    scale_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    InitializeWeights();
}

void MultiHeadAttentionLayer::InitializeWeights() {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        // Xavier initialization for projection weights
        float limit = std::sqrt(6.0f / (embed_dim_ + embed_dim_));

        af::array w_q = af::randu(af::dim4(embed_dim_, embed_dim_)) * 2.0f * limit - limit;
        af::array w_k = af::randu(af::dim4(embed_dim_, embed_dim_)) * 2.0f * limit - limit;
        af::array w_v = af::randu(af::dim4(embed_dim_, embed_dim_)) * 2.0f * limit - limit;
        af::array w_o = af::randu(af::dim4(embed_dim_, embed_dim_)) * 2.0f * limit - limit;

        W_q_ = AfToTensor(w_q);
        W_k_ = AfToTensor(w_k);
        W_v_ = AfToTensor(w_v);
        W_o_ = AfToTensor(w_o);

        if (use_bias_) {
            b_q_ = Tensor({static_cast<size_t>(embed_dim_)}, DataType::Float32);
            b_k_ = Tensor({static_cast<size_t>(embed_dim_)}, DataType::Float32);
            b_v_ = Tensor({static_cast<size_t>(embed_dim_)}, DataType::Float32);
            b_o_ = Tensor({static_cast<size_t>(embed_dim_)}, DataType::Float32);
            std::memset(b_q_.Data(), 0, embed_dim_ * sizeof(float));
            std::memset(b_k_.Data(), 0, embed_dim_ * sizeof(float));
            std::memset(b_v_.Data(), 0, embed_dim_ * sizeof(float));
            std::memset(b_o_.Data(), 0, embed_dim_ * sizeof(float));
        }

        // Initialize gradient tensors
        grad_W_q_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_), static_cast<size_t>(embed_dim_)});
        grad_W_k_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_), static_cast<size_t>(embed_dim_)});
        grad_W_v_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_), static_cast<size_t>(embed_dim_)});
        grad_W_o_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_), static_cast<size_t>(embed_dim_)});

        if (use_bias_) {
            grad_b_q_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_)});
            grad_b_k_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_)});
            grad_b_v_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_)});
            grad_b_o_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_)});
        }

        return;
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire init failed: {}, using CPU", e.what());
    }
#endif

    // CPU fallback
    std::random_device rd;
    std::mt19937 gen(rd());
    float limit = std::sqrt(6.0f / (embed_dim_ + embed_dim_));
    std::uniform_real_distribution<float> dist(-limit, limit);

    W_q_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_), static_cast<size_t>(embed_dim_)});
    W_k_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_), static_cast<size_t>(embed_dim_)});
    W_v_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_), static_cast<size_t>(embed_dim_)});
    W_o_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_), static_cast<size_t>(embed_dim_)});

    float* wq = W_q_.Data<float>();
    float* wk = W_k_.Data<float>();
    float* wv = W_v_.Data<float>();
    float* wo = W_o_.Data<float>();

    for (int i = 0; i < embed_dim_ * embed_dim_; i++) {
        wq[i] = dist(gen);
        wk[i] = dist(gen);
        wv[i] = dist(gen);
        wo[i] = dist(gen);
    }

    if (use_bias_) {
        b_q_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_)});
        b_k_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_)});
        b_v_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_)});
        b_o_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_)});
        std::memset(b_q_.Data(), 0, embed_dim_ * sizeof(float));
        std::memset(b_k_.Data(), 0, embed_dim_ * sizeof(float));
        std::memset(b_v_.Data(), 0, embed_dim_ * sizeof(float));
        std::memset(b_o_.Data(), 0, embed_dim_ * sizeof(float));
    }

    grad_W_q_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_), static_cast<size_t>(embed_dim_)});
    grad_W_k_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_), static_cast<size_t>(embed_dim_)});
    grad_W_v_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_), static_cast<size_t>(embed_dim_)});
    grad_W_o_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_), static_cast<size_t>(embed_dim_)});

    if (use_bias_) {
        grad_b_q_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_)});
        grad_b_k_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_)});
        grad_b_v_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_)});
        grad_b_o_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_)});
    }
}

Tensor MultiHeadAttentionLayer::Forward(const Tensor& input) {
    // Self-attention: Q = K = V = input
    return Forward(input, input, input, nullptr);
}

Tensor MultiHeadAttentionLayer::Forward(const Tensor& query, const Tensor& key,
                                         const Tensor& value, const Tensor* attn_mask) {
    // Cache inputs for backward
    cached_query_ = query;
    cached_key_ = key;
    cached_value_ = value;
    cached_self_attention_ = (&query == &key && &key == &value);
    cached_grad_key_ = Tensor();
    cached_grad_value_ = Tensor();

    const auto& q_shape = query.Shape();
    const auto& k_shape = key.Shape();
    const auto& v_shape = value.Shape();
    if (query.GetDataType() != DataType::Float32 || key.GetDataType() != DataType::Float32 ||
        value.GetDataType() != DataType::Float32) {
        throw std::runtime_error("MultiHeadAttention forward CPU fallback requires Float32 inputs");
    }
    if (q_shape.size() != 3 || k_shape.size() != 3 || v_shape.size() != 3) {
        throw std::invalid_argument("MultiHeadAttention expects [batch, seq_len, embed_dim] tensors");
    }
    if (q_shape[0] != k_shape[0] || k_shape[0] != v_shape[0] ||
        k_shape[1] != v_shape[1] ||
        q_shape[2] != static_cast<size_t>(embed_dim_) ||
        k_shape[2] != static_cast<size_t>(embed_dim_) ||
        v_shape[2] != static_cast<size_t>(embed_dim_)) {
        throw std::runtime_error("MultiHeadAttention forward shape mismatch");
    }
    if (attn_mask != nullptr &&
        (attn_mask->GetDataType() != DataType::Float32 ||
         attn_mask->Shape() != std::vector<size_t>{q_shape[1], k_shape[1]})) {
        throw std::runtime_error("MultiHeadAttention mask must be Float32 [seq_len_q, seq_len_kv]");
    }

    const std::vector<size_t> weight_shape{static_cast<size_t>(embed_dim_), static_cast<size_t>(embed_dim_)};
    const std::vector<size_t> bias_shape{static_cast<size_t>(embed_dim_)};
    if (W_q_.GetDataType() != DataType::Float32 || W_k_.GetDataType() != DataType::Float32 ||
        W_v_.GetDataType() != DataType::Float32 || W_o_.GetDataType() != DataType::Float32 ||
        W_q_.Shape() != weight_shape || W_k_.Shape() != weight_shape ||
        W_v_.Shape() != weight_shape || W_o_.Shape() != weight_shape) {
        throw std::runtime_error("MultiHeadAttention forward projection weight mismatch");
    }
    if (use_bias_ &&
        (b_q_.GetDataType() != DataType::Float32 || b_k_.GetDataType() != DataType::Float32 ||
         b_v_.GetDataType() != DataType::Float32 || b_o_.GetDataType() != DataType::Float32 ||
         b_q_.Shape() != bias_shape || b_k_.Shape() != bias_shape ||
         b_v_.Shape() != bias_shape || b_o_.Shape() != bias_shape)) {
        throw std::runtime_error("MultiHeadAttention forward bias mismatch");
    }

    const size_t batch_size = q_shape[0];
    const size_t seq_len_q = q_shape[1];
    const size_t seq_len_kv = k_shape[1];
    const size_t embed_dim = static_cast<size_t>(embed_dim_);
    const size_t num_heads = static_cast<size_t>(num_heads_);
    const size_t head_dim = static_cast<size_t>(head_dim_);

    cached_Q_ = Tensor({batch_size, seq_len_q, embed_dim}, DataType::Float32);
    cached_K_ = Tensor({batch_size, seq_len_kv, embed_dim}, DataType::Float32);
    cached_V_ = Tensor({batch_size, seq_len_kv, embed_dim}, DataType::Float32);
    cached_attn_weights_ = Tensor({seq_len_q, seq_len_kv, batch_size, num_heads}, DataType::Float32);
    cached_context_ = Tensor({batch_size, seq_len_q, embed_dim}, DataType::Float32);
    cached_attention_dropout_ = training_ && dropout_ > 0.0f;
    if (cached_attention_dropout_) {
        dropout_mask_ = Tensor({seq_len_q, seq_len_kv, batch_size, num_heads}, DataType::Float32);
    } else {
        dropout_mask_ = Tensor();
    }
    Tensor output({batch_size, seq_len_q, embed_dim}, DataType::Float32);

    const auto seq_index = [embed_dim](size_t b, size_t s, size_t e, size_t seq_len) {
        return (b * seq_len + s) * embed_dim + e;
    };
    const auto attn_index = [seq_len_kv, batch_size, num_heads](size_t q, size_t k, size_t b, size_t h) {
        return ((q * seq_len_kv + k) * batch_size + b) * num_heads + h;
    };
    const auto project = [&](const Tensor& src, const Tensor& weights, const Tensor* bias,
                             Tensor& dst, size_t seq_len) {
        const float* src_data = src.Data<float>();
        const float* weight_data = weights.Data<float>();
        const float* bias_data = bias != nullptr ? bias->Data<float>() : nullptr;
        float* dst_data = dst.Data<float>();
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t s = 0; s < seq_len; ++s) {
                for (size_t out = 0; out < embed_dim; ++out) {
                    float sum = bias_data != nullptr ? bias_data[out] : 0.0f;
                    for (size_t in = 0; in < embed_dim; ++in) {
                        sum += weight_data[out * embed_dim + in] * src_data[seq_index(b, s, in, seq_len)];
                    }
                    dst_data[seq_index(b, s, out, seq_len)] = sum;
                }
            }
        }
    };

    project(query, W_q_, use_bias_ ? &b_q_ : nullptr, cached_Q_, seq_len_q);
    project(key, W_k_, use_bias_ ? &b_k_ : nullptr, cached_K_, seq_len_kv);
    project(value, W_v_, use_bias_ ? &b_v_ : nullptr, cached_V_, seq_len_kv);

    const float* Q = cached_Q_.Data<float>();
    const float* K = cached_K_.Data<float>();
    const float* V = cached_V_.Data<float>();
    const float* mask_data = attn_mask != nullptr ? attn_mask->Data<float>() : nullptr;
    float* attn_data = cached_attn_weights_.Data<float>();
    float* dropout_mask_data = cached_attention_dropout_ ? dropout_mask_.Data<float>() : nullptr;
    float* context_data = cached_context_.Data<float>();
    static thread_local std::mt19937 dropout_rng(std::random_device{}());
    std::bernoulli_distribution keep_dist(1.0f - dropout_);
    const float dropout_scale = cached_attention_dropout_ ? 1.0f / (1.0f - dropout_) : 1.0f;

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            const size_t head_offset = h * head_dim;
            for (size_t q = 0; q < seq_len_q; ++q) {
                float max_score = -std::numeric_limits<float>::infinity();
                for (size_t k = 0; k < seq_len_kv; ++k) {
                    float score = mask_data != nullptr ? mask_data[q * seq_len_kv + k] : 0.0f;
                    for (size_t d = 0; d < head_dim; ++d) {
                        score += Q[seq_index(b, q, head_offset + d, seq_len_q)] *
                                 K[seq_index(b, k, head_offset + d, seq_len_kv)] * scale_;
                    }
                    attn_data[attn_index(q, k, b, h)] = score;
                    max_score = std::max(max_score, score);
                }

                float sum_exp = 0.0f;
                for (size_t k = 0; k < seq_len_kv; ++k) {
                    const size_t index = attn_index(q, k, b, h);
                    attn_data[index] = std::exp(attn_data[index] - max_score);
                    sum_exp += attn_data[index];
                }
                for (size_t k = 0; k < seq_len_kv; ++k) {
                    attn_data[attn_index(q, k, b, h)] /= sum_exp;
                    if (cached_attention_dropout_) {
                        dropout_mask_data[attn_index(q, k, b, h)] =
                            keep_dist(dropout_rng) ? 1.0f : 0.0f;
                    }
                }

                for (size_t d = 0; d < head_dim; ++d) {
                    float value_sum = 0.0f;
                    for (size_t k = 0; k < seq_len_kv; ++k) {
                        const size_t attention_index = attn_index(q, k, b, h);
                        const float dropped_attention = cached_attention_dropout_
                                                            ? attn_data[attention_index] *
                                                                  dropout_mask_data[attention_index] *
                                                                  dropout_scale
                                                            : attn_data[attention_index];
                        value_sum += dropped_attention *
                                     V[seq_index(b, k, head_offset + d, seq_len_kv)];
                    }
                    context_data[seq_index(b, q, head_offset + d, seq_len_q)] = value_sum;
                }
            }
        }
    }

    const float* context = cached_context_.Data<float>();
    const float* wo = W_o_.Data<float>();
    const float* bo = use_bias_ ? b_o_.Data<float>() : nullptr;
    float* output_data = output.Data<float>();
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t q = 0; q < seq_len_q; ++q) {
            for (size_t out = 0; out < embed_dim; ++out) {
                float sum = bo != nullptr ? bo[out] : 0.0f;
                for (size_t in = 0; in < embed_dim; ++in) {
                    sum += wo[out * embed_dim + in] * context[seq_index(b, q, in, seq_len_q)];
                }
                output_data[seq_index(b, q, out, seq_len_q)] = sum;
            }
        }
    }

    return output;
}

Tensor MultiHeadAttentionLayer::Backward(const Tensor& grad_output) {
    const auto& shape = grad_output.Shape();
    if (grad_output.GetDataType() != DataType::Float32 || shape.size() != 3 ||
        cached_query_.Shape().size() != 3 || cached_key_.Shape().size() != 3 ||
        cached_value_.Shape().size() != 3) {
        throw std::runtime_error("MultiHeadAttention backward CPU fallback requires cached 3D Float32 tensors");
    }

    const size_t batch_size = shape[0];
    const size_t seq_len_q = shape[1];
    const size_t seq_len_kv = cached_key_.Shape()[1];
    const size_t embed_dim = static_cast<size_t>(embed_dim_);
    const size_t num_heads = static_cast<size_t>(num_heads_);
    const size_t head_dim = static_cast<size_t>(head_dim_);
    const std::vector<size_t> q_shape{batch_size, seq_len_q, embed_dim};
    const std::vector<size_t> kv_shape{batch_size, seq_len_kv, embed_dim};
    const std::vector<size_t> weight_shape{embed_dim, embed_dim};
    if (shape != q_shape || cached_query_.Shape() != q_shape ||
        cached_key_.Shape() != kv_shape || cached_value_.Shape() != kv_shape ||
        cached_Q_.Shape() != q_shape || cached_K_.Shape() != kv_shape ||
        cached_V_.Shape() != kv_shape || cached_context_.Shape() != q_shape ||
        cached_attn_weights_.Shape() != std::vector<size_t>{seq_len_q, seq_len_kv, batch_size, num_heads} ||
        W_q_.Shape() != weight_shape || W_k_.Shape() != weight_shape ||
        W_v_.Shape() != weight_shape || W_o_.Shape() != weight_shape) {
        throw std::runtime_error("MultiHeadAttention backward cache/parameter shape mismatch");
    }
    if (cached_attention_dropout_ &&
        (dropout_mask_.GetDataType() != DataType::Float32 ||
         dropout_mask_.Shape() != std::vector<size_t>{seq_len_q, seq_len_kv, batch_size, num_heads})) {
        throw std::runtime_error("MultiHeadAttention backward dropout mask shape mismatch");
    }

    const auto seq_index = [embed_dim](size_t b, size_t s, size_t e, size_t seq_len) {
        return (b * seq_len + s) * embed_dim + e;
    };
    const auto attn_index = [seq_len_kv, batch_size, num_heads](size_t q, size_t k, size_t b, size_t h) {
        return ((q * seq_len_kv + k) * batch_size + b) * num_heads + h;
    };

    Tensor grad_context(q_shape, DataType::Float32);
    Tensor grad_Q(q_shape, DataType::Float32);
    Tensor grad_K(kv_shape, DataType::Float32);
    Tensor grad_V(kv_shape, DataType::Float32);
    Tensor grad_query(q_shape, DataType::Float32);
    Tensor grad_key(kv_shape, DataType::Float32);
    Tensor grad_value(kv_shape, DataType::Float32);
    grad_W_q_ = Tensor(weight_shape, DataType::Float32);
    grad_W_k_ = Tensor(weight_shape, DataType::Float32);
    grad_W_v_ = Tensor(weight_shape, DataType::Float32);
    grad_W_o_ = Tensor(weight_shape, DataType::Float32);
    if (use_bias_) {
        grad_b_q_ = Tensor({embed_dim}, DataType::Float32);
        grad_b_k_ = Tensor({embed_dim}, DataType::Float32);
        grad_b_v_ = Tensor({embed_dim}, DataType::Float32);
        grad_b_o_ = Tensor({embed_dim}, DataType::Float32);
    }

    const float* grad_out = grad_output.Data<float>();
    const float* context = cached_context_.Data<float>();
    const float* wo = W_o_.Data<float>();
    float* grad_context_data = grad_context.Data<float>();
    float* grad_W_o = grad_W_o_.Data<float>();
    float* grad_b_o = use_bias_ ? grad_b_o_.Data<float>() : nullptr;

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t q = 0; q < seq_len_q; ++q) {
            for (size_t out = 0; out < embed_dim; ++out) {
                const float grad = grad_out[seq_index(b, q, out, seq_len_q)];
                if (use_bias_) {
                    grad_b_o[out] += grad;
                }
                for (size_t in = 0; in < embed_dim; ++in) {
                    grad_context_data[seq_index(b, q, in, seq_len_q)] += wo[out * embed_dim + in] * grad;
                    grad_W_o[out * embed_dim + in] +=
                        grad * context[seq_index(b, q, in, seq_len_q)];
                }
            }
        }
    }

    const float* Q = cached_Q_.Data<float>();
    const float* K = cached_K_.Data<float>();
    const float* V = cached_V_.Data<float>();
    const float* attn = cached_attn_weights_.Data<float>();
    const float* dropout_mask_data = cached_attention_dropout_ ? dropout_mask_.Data<float>() : nullptr;
    float* grad_Q_data = grad_Q.Data<float>();
    float* grad_K_data = grad_K.Data<float>();
    float* grad_V_data = grad_V.Data<float>();
    std::vector<float> grad_attn(seq_len_kv, 0.0f);
    const float dropout_scale = cached_attention_dropout_ ? 1.0f / (1.0f - dropout_) : 1.0f;

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            const size_t head_offset = h * head_dim;
            for (size_t q = 0; q < seq_len_q; ++q) {
                std::fill(grad_attn.begin(), grad_attn.end(), 0.0f);
                for (size_t k = 0; k < seq_len_kv; ++k) {
                    const size_t attention_index = attn_index(q, k, b, h);
                    const float attention_multiplier = cached_attention_dropout_
                                                           ? dropout_mask_data[attention_index] * dropout_scale
                                                           : 1.0f;
                    const float dropped_attention = attn[attention_index] * attention_multiplier;
                    for (size_t d = 0; d < head_dim; ++d) {
                        grad_V_data[seq_index(b, k, head_offset + d, seq_len_kv)] +=
                            dropped_attention *
                            grad_context_data[seq_index(b, q, head_offset + d, seq_len_q)];
                        grad_attn[k] +=
                            grad_context_data[seq_index(b, q, head_offset + d, seq_len_q)] *
                            V[seq_index(b, k, head_offset + d, seq_len_kv)] *
                            attention_multiplier;
                    }
                }

                float softmax_dot = 0.0f;
                for (size_t k = 0; k < seq_len_kv; ++k) {
                    softmax_dot += grad_attn[k] * attn[attn_index(q, k, b, h)];
                }
                for (size_t k = 0; k < seq_len_kv; ++k) {
                    const float grad_score =
                        attn[attn_index(q, k, b, h)] * (grad_attn[k] - softmax_dot) * scale_;
                    for (size_t d = 0; d < head_dim; ++d) {
                        grad_Q_data[seq_index(b, q, head_offset + d, seq_len_q)] +=
                            grad_score * K[seq_index(b, k, head_offset + d, seq_len_kv)];
                        grad_K_data[seq_index(b, k, head_offset + d, seq_len_kv)] +=
                            grad_score * Q[seq_index(b, q, head_offset + d, seq_len_q)];
                    }
                }
            }
        }
    }

    const auto projection_backward = [&](const Tensor& input, const Tensor& weight,
                                         const Tensor& grad_projected,
                                         Tensor& grad_input, Tensor& grad_weight,
                                         Tensor* grad_bias, size_t seq_len) {
        const float* input_data = input.Data<float>();
        const float* weight_data = weight.Data<float>();
        const float* grad_proj_data = grad_projected.Data<float>();
        float* grad_input_data = grad_input.Data<float>();
        float* grad_weight_data = grad_weight.Data<float>();
        float* grad_bias_data = grad_bias != nullptr ? grad_bias->Data<float>() : nullptr;

        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t s = 0; s < seq_len; ++s) {
                for (size_t out = 0; out < embed_dim; ++out) {
                    const float grad = grad_proj_data[seq_index(b, s, out, seq_len)];
                    if (grad_bias_data != nullptr) {
                        grad_bias_data[out] += grad;
                    }
                    for (size_t in = 0; in < embed_dim; ++in) {
                        grad_weight_data[out * embed_dim + in] +=
                            grad * input_data[seq_index(b, s, in, seq_len)];
                        grad_input_data[seq_index(b, s, in, seq_len)] +=
                            weight_data[out * embed_dim + in] * grad;
                    }
                }
            }
        }
    };

    projection_backward(cached_query_, W_q_, grad_Q, grad_query, grad_W_q_,
                        use_bias_ ? &grad_b_q_ : nullptr, seq_len_q);
    projection_backward(cached_key_, W_k_, grad_K, grad_key, grad_W_k_,
                        use_bias_ ? &grad_b_k_ : nullptr, seq_len_kv);
    projection_backward(cached_value_, W_v_, grad_V, grad_value, grad_W_v_,
                        use_bias_ ? &grad_b_v_ : nullptr, seq_len_kv);

    cached_grad_key_ = grad_key;
    cached_grad_value_ = grad_value;

    if (cached_self_attention_) {
        float* grad_query_data = grad_query.Data<float>();
        const float* grad_key_data = grad_key.Data<float>();
        const float* grad_value_data = grad_value.Data<float>();
        for (size_t i = 0; i < grad_query.NumElements(); ++i) {
            grad_query_data[i] += grad_key_data[i] + grad_value_data[i];
        }
    }

    return grad_query;
}

std::map<std::string, Tensor> MultiHeadAttentionLayer::GetParameters() {
    std::map<std::string, Tensor> params;
    params["W_q"] = W_q_;
    params["W_k"] = W_k_;
    params["W_v"] = W_v_;
    params["W_o"] = W_o_;
    params["grad_W_q"] = grad_W_q_;
    params["grad_W_k"] = grad_W_k_;
    params["grad_W_v"] = grad_W_v_;
    params["grad_W_o"] = grad_W_o_;

    if (use_bias_) {
        params["b_q"] = b_q_;
        params["b_k"] = b_k_;
        params["b_v"] = b_v_;
        params["b_o"] = b_o_;
        params["grad_b_q"] = grad_b_q_;
        params["grad_b_k"] = grad_b_k_;
        params["grad_b_v"] = grad_b_v_;
        params["grad_b_o"] = grad_b_o_;
    }

    return params;
}

void MultiHeadAttentionLayer::SetParameters(const std::map<std::string, Tensor>& params) {
    if (params.count("W_q")) W_q_ = params.at("W_q");
    if (params.count("W_k")) W_k_ = params.at("W_k");
    if (params.count("W_v")) W_v_ = params.at("W_v");
    if (params.count("W_o")) W_o_ = params.at("W_o");

    if (use_bias_) {
        if (params.count("b_q")) b_q_ = params.at("b_q");
        if (params.count("b_k")) b_k_ = params.at("b_k");
        if (params.count("b_v")) b_v_ = params.at("b_v");
        if (params.count("b_o")) b_o_ = params.at("b_o");
    }
}

} // namespace cyxwiz