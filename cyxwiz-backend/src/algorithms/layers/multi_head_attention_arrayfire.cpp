#include "cyxwiz/layers/attention.h"
#include <limits>
#include <cmath>
#include <vector>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#ifdef max
#undef max
#endif

namespace cyxwiz {
namespace {

// Semantic [B,S,E] -> [S,D,B,H]. Only head factorization is owned here;
// Tensor owns the host/device layout and residency contract.
af::array SplitHeads(const af::array& sequence, int head_dim, int heads) {
    return af::reorder(af::moddims(sequence,
        af::dim4(sequence.dims(0), sequence.dims(1), head_dim, heads)), 1, 2, 0, 3);
}

// Rotate the first rot_dim head features of [S, D, B, H] by position (dim 0)
// using the half-split convention within those features:
// x*cos + rotate_half(x)*sin, rotate_half(x) = [-x2, x1]; features past rot_dim
// pass through (partial RoPE, GPT-NeoX rotary_pct). direction = +1 applies the
// rotation, -1 its inverse (the transpose), which backward needs because the
// rotation is orthogonal.
af::array ApplyRotary(const af::array& x, float base, float direction, dim_t rot_dim, dim_t offset = 0) {
    const dim_t seq = x.dims(0);
    const dim_t dim = x.dims(1);
    const dim_t half = rot_dim / 2;
    const af::array positions = af::range(af::dim4(seq, half), 0, f32) + static_cast<float>(offset);
    const af::array index = af::range(af::dim4(seq, half), 1, f32);
    const af::array inv_freq = af::exp(index * (-2.0f * std::log(base) / static_cast<float>(rot_dim)));
    const af::array angle = positions * inv_freq;
    const af::array cos_half = af::cos(angle);
    const af::array sin_half = af::sin(angle) * direction;
    const af::dim4 spread(1, 1, x.dims(2), x.dims(3));
    const af::array cos_full = af::tile(af::join(1, cos_half, cos_half), spread);
    const af::array sin_full = af::tile(af::join(1, sin_half, sin_half), spread);
    const af::array x1 = x(af::span, af::seq(0, static_cast<double>(half - 1)), af::span, af::span);
    const af::array x2 = x(af::span, af::seq(static_cast<double>(half), static_cast<double>(rot_dim - 1)), af::span, af::span);
    const af::array rotated = af::join(1, x1, x2) * cos_full + af::join(1, -x2, x1) * sin_full;
    if (rot_dim == dim) {
        return rotated;
    }
    return af::join(1, rotated, x(af::span, af::seq(static_cast<double>(rot_dim), static_cast<double>(dim - 1)),
                                  af::span, af::span));
}

// Per-head RMS normalization over head features (dim 1 of [S, D, B, H]):
// n = x / sqrt(mean(x^2) + eps). Returns n; inv_rms receives [S, 1, B, H].
af::array RmsNormalizeHeads(const af::array& x, float eps, af::array& inv_rms) {
    inv_rms = 1.0f / af::sqrt(af::mean(x * x, 1) + eps);
    return x * af::tile(inv_rms, af::dim4(1, x.dims(1)));
}

af::array TileHeadGamma(const Tensor& gamma, const af::array& like) {
    return af::tile(af::moddims(gamma.GetSemanticArray(), 1, like.dims(1)),
                    af::dim4(like.dims(0), 1, like.dims(2), like.dims(3)));
}

// Backward of y = n * gamma, n = RmsNormalizeHeads(x): returns dx and
// sum(dy * n) over positions, batch and heads into grad_gamma ([D]).
af::array RmsNormalizeHeadsBackward(const af::array& dy, const af::array& n,
                                    const af::array& inv_rms, const Tensor& gamma,
                                    af::array& grad_gamma) {
    grad_gamma = af::flat(af::sum(af::sum(af::sum(dy * n, 0), 2), 3));
    const af::array dn = dy * TileHeadGamma(gamma, dy);
    const af::array proj = af::tile(af::mean(dn * n, 1), af::dim4(1, n.dims(1)));
    return af::tile(inv_rms, af::dim4(1, n.dims(1))) * (dn - n * proj);
}

// Grouped-query attention: key/value head j serves query heads
// j*groups .. (j+1)*groups-1 (LLaMA repeat_kv / torch repeat_interleave).
af::array ExpandKvHeads(const af::array& kv, int groups) {
    if (groups == 1) return kv;
    const af::dim4 d = kv.dims();
    const af::array flat = af::moddims(kv, af::dim4(d[0] * d[1] * d[2], 1, d[3]));
    return af::moddims(af::tile(flat, af::dim4(1, groups, 1)), af::dim4(d[0], d[1], d[2], d[3] * groups));
}

// Inverse of ExpandKvHeads for gradients: sum each group of query heads.
af::array ReduceKvHeads(const af::array& heads, int groups) {
    if (groups == 1) return heads;
    const af::dim4 d = heads.dims();
    const dim_t kv = d[3] / groups;
    const af::array grouped = af::moddims(heads, af::dim4(d[0] * d[1] * d[2], groups, kv));
    return af::moddims(af::sum(grouped, 1), af::dim4(d[0], d[1], d[2], kv));
}

af::array JoinHeads(const af::array& heads) {
    return af::moddims(af::reorder(heads, 2, 0, 1, 3),
        af::dim4(heads.dims(2), heads.dims(0), heads.dims(1) * heads.dims(3)));
}

// ALiBi bias slope(h) * (j - i) on [Sq, Sk, 1, H] scores (Press et al. 2022);
// query i sits at position q_offset + i.
af::array AlibiBias(const std::vector<float>& slopes, dim_t sq, dim_t sk, dim_t q_offset = 0) {
    const af::array distance = af::range(af::dim4(sq, sk), 1, f32) -
        (af::range(af::dim4(sq, sk), 0, f32) + static_cast<float>(q_offset));
    const af::array slope = af::moddims(af::array(static_cast<dim_t>(slopes.size()), slopes.data()),
                                        af::dim4(1, 1, 1, static_cast<dim_t>(slopes.size())));
    return af::tile(distance, af::dim4(1, 1, 1, static_cast<dim_t>(slopes.size()))) *
           af::tile(slope, af::dim4(sq, sk, 1, 1));
}

} // namespace

Tensor MultiHeadAttentionLayer::ForwardArrayFire(
    const Tensor& query, const Tensor& key, const Tensor& value, const Tensor* mask) {
    const auto project = [this](const Tensor& input, const Tensor& weight,
                                const Tensor& bias) {
        const af::array x = input.GetSemanticArray();
        const dim_t rows = x.elements() / embed_dim_;
        const dim_t out = static_cast<dim_t>(weight.Shape()[0]);
        af::array result = af::matmul(af::moddims(x, rows, embed_dim_),
                                     weight.GetSemanticArray(), AF_MAT_NONE, AF_MAT_TRANS);
        if (use_bias_) {
            result = result + af::tile(af::moddims(bias.GetSemanticArray(), 1, out), af::dim4(rows));
        }
        return Tensor::FromSemanticArray(af::moddims(result, af::dim4(x.dims(0), x.dims(1), out)),
            {input.Shape()[0], input.Shape()[1], static_cast<size_t>(out)});
    };
    const int groups = num_heads_ / num_kv_heads_;
    // Commit caches only after all ArrayFire work succeeds. Compatibility may
    // restart a failed forward; no parameters have been updated at this boundary.
    Tensor q = project(query, W_q_, b_q_);
    Tensor k = project(key, W_k_, b_k_);
    Tensor v = project(value, W_v_, b_v_);
    af::array qh = SplitHeads(q.GetSemanticArray(), head_dim_, num_heads_);
    af::array kh = SplitHeads(k.GetSemanticArray(), head_dim_, num_kv_heads_);
    af::array vh = SplitHeads(v.GetSemanticArray(), head_dim_, num_kv_heads_);
    if (qk_norm_) {
        af::array inv_rms;
        qh = RmsNormalizeHeads(qh, qk_norm_eps_, inv_rms) * TileHeadGamma(q_norm_gamma_, qh);
        kh = RmsNormalizeHeads(kh, qk_norm_eps_, inv_rms) * TileHeadGamma(k_norm_gamma_, kh);
    }
    if (rope_) {
        qh = ApplyRotary(qh, rope_base_, 1.0f, RotaryDims());
        kh = ApplyRotary(kh, rope_base_, 1.0f, RotaryDims());
    }
    kh = ExpandKvHeads(kh, groups);
    vh = ExpandKvHeads(vh, groups);
    af::array scores = af::matmul(qh, kh, AF_MAT_NONE, AF_MAT_TRANS) * scale_;
    if (logit_softcap_ > 0.0f) {
        scores = logit_softcap_ * af::tanh(scores / logit_softcap_);
    }
    if (alibi_) {
        scores = scores + af::tile(AlibiBias(AlibiSlopes(), qh.dims(0), kh.dims(0)), af::dim4(1, 1, qh.dims(2), 1));
    }
    if (mask) {
        scores = scores + af::tile(mask->GetSemanticArray(), af::dim4(1, 1, qh.dims(2), num_heads_));
    }
    af::array row_max = af::max(scores, 1);
    af::array fully_blocked;
    if (mask) {
        // Only an explicit all-negative-infinity mask row means no visible key.
        // Keep this decision on device; do not hide nonfinite unmasked logits.
        fully_blocked = af::tile(af::allTrue(
            mask->GetSemanticArray() == -std::numeric_limits<float>::infinity(), 1),
            af::dim4(1, 1, qh.dims(2), num_heads_));
        row_max = af::select(fully_blocked, 0.0, row_max);
    }
    af::array exps = af::exp(scores - af::tile(row_max, af::dim4(1, kh.dims(0))));
    if (mask) {
        exps = af::select(af::tile(fully_blocked, af::dim4(1, kh.dims(0))), 0.0, exps);
    }
    af::array denominator = af::sum(exps, 1);
    if (mask) {
        denominator = af::select(fully_blocked, 1.0, denominator);
    }
    const af::array attention = exps / af::tile(denominator, af::dim4(1, kh.dims(0)));
    const bool use_dropout = training_ && dropout_ > 0.0f;
    af::array used_attention = attention;
    Tensor dropout_mask;
    if (use_dropout) {
        // Same selected-backend random stream as DropoutLayer. Cache the mask
        // before use so backward never samples a second mask.
        af::array keep = (af::randu(attention.dims(), f32) > dropout_).as(f32);
        keep.eval();
        dropout_mask = Tensor::FromSemanticArray(keep,
            {query.Shape()[1], key.Shape()[1], query.Shape()[0], static_cast<size_t>(num_heads_)});
        used_attention = attention * keep * (1.0f / (1.0f - dropout_));
    }
    Tensor context = Tensor::FromSemanticArray(JoinHeads(af::matmul(used_attention, vh)), query.Shape());
    Tensor output = project(context, W_o_, b_o_);
    // Realize the returned projection at the exception boundary; this is also
    // the last projection consumed by subsequent layers, not a host readback.
    output.GetSemanticArray().eval();
    cached_query_ = query;
    cached_key_ = key;
    cached_value_ = value;
    cached_Q_ = std::move(q);
    cached_K_ = std::move(k);
    cached_V_ = std::move(v);
    cached_context_ = std::move(context);
    cached_attn_weights_ = Tensor::FromSemanticArray(attention,
        {query.Shape()[1], key.Shape()[1], query.Shape()[0], static_cast<size_t>(num_heads_)});
    cached_self_attention_ = (&query == &key && &key == &value);
    cached_attention_dropout_ = use_dropout;
    dropout_mask_ = std::move(dropout_mask);
    cached_grad_key_ = Tensor();
    cached_grad_value_ = Tensor();
    return output;
}

Tensor MultiHeadAttentionLayer::BackwardArrayFire(const Tensor& grad_output) {
    // Projection rows flatten B and S in the same semantic device order in
    // both directions. Weights remain [out,in], as in the native/checkpoint API.
    const auto backward_project = [this](const Tensor& input, const af::array& grad,
                                         const Tensor& weight, Tensor& grad_weight,
                                         Tensor& grad_bias) {
        const dim_t out = static_cast<dim_t>(weight.Shape()[0]);
        const dim_t rows = grad.elements() / out;
        const af::array dy = af::moddims(grad, rows, out);
        const af::array x = af::moddims(input.GetSemanticArray(), rows, embed_dim_);
        grad_weight = Tensor::FromSemanticArray(
            af::matmul(dy, x, AF_MAT_TRANS), weight.Shape());
        if (use_bias_) {
            grad_bias = Tensor::FromSemanticArray(af::flat(af::sum(dy, 0)),
                                                 {static_cast<size_t>(out)});
        }
        return Tensor::FromSemanticArray(
            af::moddims(af::matmul(dy, weight.GetSemanticArray()), input.GetSemanticArray().dims()),
            input.Shape());
    };
    const int groups = num_heads_ / num_kv_heads_;
    Tensor gwo, gbo, gwq, gbq, gwk, gbk, gwv, gbv;
    const Tensor dc = backward_project(cached_context_, grad_output.GetSemanticArray(), W_o_, gwo, gbo);
    const af::array dch = SplitHeads(dc.GetSemanticArray(), head_dim_, num_heads_);
    // Scores used the normalized, rotated and head-expanded Q and K; cached
    // projections are raw, so those steps are recomputed here.
    af::array q = SplitHeads(cached_Q_.GetSemanticArray(), head_dim_, num_heads_);
    af::array k = SplitHeads(cached_K_.GetSemanticArray(), head_dim_, num_kv_heads_);
    af::array q_n, k_n, q_inv_rms, k_inv_rms;
    if (qk_norm_) {
        q_n = RmsNormalizeHeads(q, qk_norm_eps_, q_inv_rms);
        k_n = RmsNormalizeHeads(k, qk_norm_eps_, k_inv_rms);
        q = q_n * TileHeadGamma(q_norm_gamma_, q_n);
        k = k_n * TileHeadGamma(k_norm_gamma_, k_n);
    }
    if (rope_) {
        q = ApplyRotary(q, rope_base_, 1.0f, RotaryDims());
        k = ApplyRotary(k, rope_base_, 1.0f, RotaryDims());
    }
    k = ExpandKvHeads(k, groups);
    const af::array v = ExpandKvHeads(SplitHeads(cached_V_.GetSemanticArray(), head_dim_, num_kv_heads_), groups);
    const af::array a = cached_attn_weights_.GetSemanticArray();
    af::array da = af::matmul(dch, v, AF_MAT_NONE, AF_MAT_TRANS);
    af::array used_attention = a;
    if (cached_attention_dropout_) {
        const af::array multiplier = dropout_mask_.GetSemanticArray() * (1.0f / (1.0f - dropout_));
        da = da * multiplier;
        used_attention = a * multiplier;
    }
    af::array ds = a * (da - af::tile(af::sum(a * da, 1), af::dim4(1, k.dims(0)))) * scale_;
    if (logit_softcap_ > 0.0f) {
        // scores = cap * tanh(raw / cap): d raw = d scores * (1 - tanh^2).
        const af::array t = af::tanh(af::matmul(q, k, AF_MAT_NONE, AF_MAT_TRANS) * (scale_ / logit_softcap_));
        ds = ds * (1.0f - t * t);
    }
    af::array dq_heads = af::matmul(ds, k);
    af::array dk_heads = ReduceKvHeads(af::matmul(ds, q, AF_MAT_TRANS), groups);
    if (rope_) {
        dq_heads = ApplyRotary(dq_heads, rope_base_, -1.0f, RotaryDims());
        dk_heads = ApplyRotary(dk_heads, rope_base_, -1.0f, RotaryDims());
    }
    af::array grad_q_gamma, grad_k_gamma;
    if (qk_norm_) {
        dq_heads = RmsNormalizeHeadsBackward(dq_heads, q_n, q_inv_rms, q_norm_gamma_, grad_q_gamma);
        dk_heads = RmsNormalizeHeadsBackward(dk_heads, k_n, k_inv_rms, k_norm_gamma_, grad_k_gamma);
        grad_q_gamma.eval();
        grad_k_gamma.eval();
    }
    const af::array dv_heads = ReduceKvHeads(af::matmul(used_attention, dch, AF_MAT_TRANS), groups);
    Tensor dq = backward_project(cached_query_, JoinHeads(dq_heads), W_q_, gwq, gbq);
    Tensor dk = backward_project(cached_key_, JoinHeads(dk_heads), W_k_, gwk, gbk);
    Tensor dv = backward_project(cached_value_, JoinHeads(dv_heads), W_v_, gwv, gbv);
    Tensor result = dq;
    if (cached_self_attention_) {
        result = Tensor::FromSemanticArray(
            dq.GetSemanticArray() + dk.GetSemanticArray() + dv.GetSemanticArray(), dq.Shape());
    }
    // Bias reductions and the returned residual sum are lazy. Realize them
    // before publishing gradients, so fallback cannot expose partial results.
    result.GetSemanticArray().eval();
    if (use_bias_) {
        gbo.GetSemanticArray().eval();
        gbq.GetSemanticArray().eval();
        gbk.GetSemanticArray().eval();
        gbv.GetSemanticArray().eval();
    }
    grad_W_o_ = std::move(gwo); grad_b_o_ = std::move(gbo);
    grad_W_q_ = std::move(gwq); grad_b_q_ = std::move(gbq);
    grad_W_k_ = std::move(gwk); grad_b_k_ = std::move(gbk);
    grad_W_v_ = std::move(gwv); grad_b_v_ = std::move(gbv);
    if (qk_norm_) {
        const std::vector<size_t> gamma_shape{static_cast<size_t>(head_dim_)};
        grad_q_norm_gamma_ = Tensor::FromSemanticArray(grad_q_gamma, gamma_shape);
        grad_k_norm_gamma_ = Tensor::FromSemanticArray(grad_k_gamma, gamma_shape);
    }
    cached_grad_key_ = std::move(dk);
    cached_grad_value_ = std::move(dv);
    return result;
}

Tensor MultiHeadAttentionLayer::ForwardIncrementalArrayFire(const Tensor& input) {
    const auto project = [this](const Tensor& x_tensor, const Tensor& weight, const Tensor& bias) {
        const af::array x = x_tensor.GetSemanticArray();
        const dim_t rows = x.elements() / embed_dim_;
        const dim_t out = static_cast<dim_t>(weight.Shape()[0]);
        af::array result = af::matmul(af::moddims(x, rows, embed_dim_), weight.GetSemanticArray(),
                                      AF_MAT_NONE, AF_MAT_TRANS);
        if (use_bias_) {
            result = result + af::tile(af::moddims(bias.GetSemanticArray(), 1, out), af::dim4(rows));
        }
        return af::moddims(result, af::dim4(x.dims(0), x.dims(1), out));
    };
    const auto& shape = input.Shape();
    if (input.GetDataType() != DataType::Float32 || shape.size() != 3 || shape[2] != static_cast<size_t>(embed_dim_)) {
        throw std::runtime_error("MultiHeadAttention incremental input must be Float32 [batch, seq_len, embed_dim]");
    }
    const dim_t offset = static_cast<dim_t>(incremental_offset_);
    const dim_t new_positions = static_cast<dim_t>(shape[1]);
    const int groups = num_heads_ / num_kv_heads_;
    af::array qh = SplitHeads(project(input, W_q_, b_q_), head_dim_, num_heads_);
    af::array kh = SplitHeads(project(input, W_k_, b_k_), head_dim_, num_kv_heads_);
    af::array vh = SplitHeads(project(input, W_v_, b_v_), head_dim_, num_kv_heads_);
    if (qk_norm_) {
        af::array inv_rms;
        qh = RmsNormalizeHeads(qh, qk_norm_eps_, inv_rms) * TileHeadGamma(q_norm_gamma_, qh);
        kh = RmsNormalizeHeads(kh, qk_norm_eps_, inv_rms) * TileHeadGamma(k_norm_gamma_, kh);
    }
    if (rope_) {
        qh = ApplyRotary(qh, rope_base_, 1.0f, RotaryDims(), offset);
        kh = ApplyRotary(kh, rope_base_, 1.0f, RotaryDims(), offset);
    }
    af::array keys = kh, values = vh;
    if (offset > 0) {
        if (cache_k_.GetSemanticArray().dims(2) != kh.dims(2)) {
            throw std::runtime_error("MultiHeadAttention incremental batch size changed within a sequence");
        }
        keys = af::join(0, cache_k_.GetSemanticArray(), kh);
        values = af::join(0, cache_v_.GetSemanticArray(), vh);
    }
    const dim_t total = keys.dims(0);
    const af::array k_all = ExpandKvHeads(keys, groups);
    const af::array v_all = ExpandKvHeads(values, groups);
    af::array scores = af::matmul(qh, k_all, AF_MAT_NONE, AF_MAT_TRANS) * scale_;
    if (logit_softcap_ > 0.0f) {
        scores = logit_softcap_ * af::tanh(scores / logit_softcap_);
    }
    if (alibi_) {
        scores = scores + af::tile(AlibiBias(AlibiSlopes(), new_positions, total, offset), af::dim4(1, 1, qh.dims(2), 1));
    }
    const af::dim4 grid(new_positions, total);
    const af::array query_pos = af::range(grid, 0, s32) + static_cast<int>(offset);
    const af::array key_pos = af::range(grid, 1, s32);
    af::array hidden = key_pos > query_pos;
    if (incremental_window_ > 0) {
        hidden = hidden || (query_pos - key_pos >= incremental_window_);
    }
    scores = scores + af::tile(hidden.as(f32) * -1e9f, af::dim4(1, 1, qh.dims(2), num_heads_));
    const af::array row_max = af::max(scores, 1);
    const af::array exps = af::exp(scores - af::tile(row_max, af::dim4(1, total)));
    const af::array attention = exps / af::tile(af::sum(exps, 1), af::dim4(1, total));
    const af::array context = JoinHeads(af::matmul(attention, v_all));
    const af::array output = project(Tensor::FromSemanticArray(context, shape), W_o_, b_o_);
    output.eval();
    keys.eval();
    values.eval();
    cache_k_ = Tensor::FromSemanticArray(keys, {static_cast<size_t>(total), static_cast<size_t>(head_dim_),
                                                shape[0], static_cast<size_t>(num_kv_heads_)});
    cache_v_ = Tensor::FromSemanticArray(values, {static_cast<size_t>(total), static_cast<size_t>(head_dim_),
                                                  shape[0], static_cast<size_t>(num_kv_heads_)});
    cache_positions_ = static_cast<size_t>(total);
    return Tensor::FromSemanticArray(output, shape);
}

} // namespace cyxwiz
#endif
