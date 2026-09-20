#include "cyxwiz/layers/attention.h"
#include <limits>

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

af::array JoinHeads(const af::array& heads) {
    return af::moddims(af::reorder(heads, 2, 0, 1, 3),
        af::dim4(heads.dims(2), heads.dims(0), heads.dims(1) * heads.dims(3)));
}

} // namespace

Tensor MultiHeadAttentionLayer::ForwardArrayFire(
    const Tensor& query, const Tensor& key, const Tensor& value, const Tensor* mask) {
    const auto project = [this](const Tensor& input, const Tensor& weight,
                                const Tensor& bias) {
        const af::array x = input.GetSemanticArray();
        const dim_t rows = x.elements() / embed_dim_;
        af::array result = af::matmul(af::moddims(x, rows, embed_dim_),
                                     weight.GetSemanticArray(), AF_MAT_NONE, AF_MAT_TRANS);
        if (use_bias_) {
            result = result + af::tile(af::moddims(bias.GetSemanticArray(), 1, embed_dim_), af::dim4(rows));
        }
        return Tensor::FromSemanticArray(af::moddims(result, x.dims()), input.Shape());
    };
    // Commit caches only after all ArrayFire work succeeds. Compatibility may
    // restart a failed forward; no parameters have been updated at this boundary.
    Tensor q = project(query, W_q_, b_q_);
    Tensor k = project(key, W_k_, b_k_);
    Tensor v = project(value, W_v_, b_v_);
    const af::array qh = SplitHeads(q.GetSemanticArray(), head_dim_, num_heads_);
    const af::array kh = SplitHeads(k.GetSemanticArray(), head_dim_, num_heads_);
    const af::array vh = SplitHeads(v.GetSemanticArray(), head_dim_, num_heads_);
    af::array scores = af::matmul(qh, kh, AF_MAT_NONE, AF_MAT_TRANS) * scale_;
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
        const dim_t rows = grad.elements() / embed_dim_;
        const af::array dy = af::moddims(grad, rows, embed_dim_);
        const af::array x = af::moddims(input.GetSemanticArray(), rows, embed_dim_);
        grad_weight = Tensor::FromSemanticArray(
            af::matmul(dy, x, AF_MAT_TRANS), weight.Shape());
        if (use_bias_) {
            grad_bias = Tensor::FromSemanticArray(af::flat(af::sum(dy, 0)),
                                                 {static_cast<size_t>(embed_dim_)});
        }
        return Tensor::FromSemanticArray(
            af::moddims(af::matmul(dy, weight.GetSemanticArray()), grad.dims()), input.Shape());
    };
    Tensor gwo, gbo, gwq, gbq, gwk, gbk, gwv, gbv;
    const Tensor dc = backward_project(cached_context_, grad_output.GetSemanticArray(), W_o_, gwo, gbo);
    const af::array dch = SplitHeads(dc.GetSemanticArray(), head_dim_, num_heads_);
    const af::array q = SplitHeads(cached_Q_.GetSemanticArray(), head_dim_, num_heads_);
    const af::array k = SplitHeads(cached_K_.GetSemanticArray(), head_dim_, num_heads_);
    const af::array v = SplitHeads(cached_V_.GetSemanticArray(), head_dim_, num_heads_);
    const af::array a = cached_attn_weights_.GetSemanticArray();
    af::array da = af::matmul(dch, v, AF_MAT_NONE, AF_MAT_TRANS);
    af::array used_attention = a;
    if (cached_attention_dropout_) {
        const af::array multiplier = dropout_mask_.GetSemanticArray() * (1.0f / (1.0f - dropout_));
        da = da * multiplier;
        used_attention = a * multiplier;
    }
    const af::array ds = a * (da - af::tile(af::sum(a * da, 1), af::dim4(1, k.dims(0)))) * scale_;
    Tensor dq = backward_project(cached_query_, JoinHeads(af::matmul(ds, k)), W_q_, gwq, gbq);
    Tensor dk = backward_project(cached_key_, JoinHeads(af::matmul(ds, q, AF_MAT_TRANS)), W_k_, gwk, gbk);
    Tensor dv = backward_project(cached_value_, JoinHeads(af::matmul(used_attention, dch, AF_MAT_TRANS)), W_v_, gwv, gbv);
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
    cached_grad_key_ = std::move(dk);
    cached_grad_value_ = std::move(dv);
    return result;
}

} // namespace cyxwiz
#endif
