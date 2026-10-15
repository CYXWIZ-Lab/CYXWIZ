#pragma once

// What word2vec and fastText share (TOFIX152): gensim's down-sampling, the
// examples of an epoch (each position's window drawn from 1..window), the
// negatives from a unigram^0.75 table, and the output side of a negative
// sampling step. All on the ArrayFire device.

#include "arrayfire_indexed.h"

#include <arrayfire.h>

#include <vector>

namespace cyxwiz {

constexpr dim_t kNoiseTableSize = 1000000;

// W[key] -= alpha * (sum of the rows with that key).
inline void ScatterSubtract(af::array& w, const af::array& keys, const af::array& rows, double alpha) {
    if (keys.elements() == 0) return;
    if (keys.elements() == 1) {
        w(keys, af::span) -= static_cast<float>(alpha) * rows;
        return;
    }
    af::array sorted, order;
    SortWithIndex(sorted, order, keys, 0, true);
    af::array unique, sums;
    af::sumByKey(unique, sums, sorted, af::lookup(rows, order, 0), 0);
    w(unique, af::span) -= static_cast<float>(alpha) * sums;
}

inline af::array Softplus(const af::array& x) {
    return af::max(x, 0.0f) + af::log1p(af::exp(-af::abs(x)));
}

// gensim's keep probability per word (1 everywhere when `sample` is 0).
inline af::array KeepProbability(const af::array& count_f, double sample) {
    const dim_t v = count_f.elements();
    if (!(sample > 0.0)) return af::constant(1.0f, v, f32);
    const float threshold = static_cast<float>(sample * static_cast<double>(af::sum<double>(count_f)));
    return af::min((af::sqrt(count_f / threshold) + 1.0f) * (threshold / count_f), 1.0f);
}

// A table where each word fills a share of the slots proportional to
// count^0.75.
inline af::array NoiseTable(const af::array& count_f) {
    const dim_t v = count_f.elements();
    const af::array power = af::pow(count_f, 0.75f);
    const af::array slots = af::floor(af::scan(power / af::sum<float>(power), 0) * static_cast<float>(kNoiseTableSize));
    const af::array starts = af::join(0, af::constant(0.0f, 1), slots(af::seq(0, static_cast<double>(v) - 2)));
    const af::array start_words = af::where(slots > starts);
    // Slot t belongs to the last word that starts at or before it: mark each
    // start with its word and carry the running maximum.
    af::array word_at_start = af::constant(-1, kNoiseTableSize, s32);
    word_at_start(starts(start_words).as(s32)) = start_words.as(s32);
    return af::scan(word_at_start, 0, AF_BINARY_MAX);
}

// The examples of one epoch, in corpus order. CBOW: one per centre, `inputs`
// [E, 2 * window] context words (-1 = none) predicting `outputs` [E] centres.
// Skip-gram: one per (centre, context); word2vec's input is the context word
// predicting the centre (gensim), fastText's the centre predicting the
// context (`centre_input`, as fastText and gensim's FastText do).
struct EpochExamples {
    af::array inputs;
    af::array outputs;
};

inline EpochExamples MakeEpochExamples(const af::array& corpus, const af::array& corpus_sentence,
                                       const af::array& keep_probability, int window, bool cbow, bool centre_input,
                                       af::randomEngine& engine) {
    EpochExamples out;
    const dim_t n = corpus.elements();
    // Kept words of this epoch, in corpus order.
    const af::array kept = af::where(af::randu(af::dim4(n), f32, engine) < keep_probability(corpus));
    const dim_t m = kept.elements();
    if (m < 2) return out;
    const af::array words = corpus(kept).as(s32);
    const af::array sentence = corpus_sentence(kept).as(s32);
    // gensim: reduced_window = random % window, so the reach is 1..window.
    const af::array reach = af::min(af::floor(af::randu(af::dim4(m), f32, engine) * static_cast<float>(window)),
                                    static_cast<float>(window - 1));
    const af::array position = af::range(af::dim4(m), 0, s32);

    // Context slot (o, side) of each centre: valid when inside the sentence and
    // within the centre's reach.
    std::vector<af::array> context_cols;
    for (int o = 1; o <= window; ++o) {
        for (const int side : {-1, 1}) {
            const af::array other = position + side * o;
            const af::array inside = other >= 0 && other < static_cast<int>(m);
            const af::array clamped = af::select(inside, other, 0.0);
            const af::array ok = inside && (sentence(clamped) == sentence) && (static_cast<float>(window - o) >= reach);
            context_cols.push_back(af::select(ok, words(clamped), -1.0));
        }
    }
    af::array contexts = context_cols.front();
    for (size_t i = 1; i < context_cols.size(); ++i) contexts = af::join(1, contexts, context_cols[i]);  // [m, 2W]
    contexts.eval();

    if (cbow) {
        const af::array centres = af::where(af::anyTrue(contexts >= 0, 1));
        out.inputs = contexts(centres, af::span);
        out.outputs = words(centres);
    } else {
        const af::array flat_by_centre = af::flat(contexts.T());  // centre-major
        const af::array valid = af::where(flat_by_centre >= 0);
        const af::array context = flat_by_centre(valid).as(s32);
        const af::array centre = words(valid / static_cast<unsigned>(2 * window)).as(s32);
        out.inputs = centre_input ? centre : context;
        out.outputs = centre_input ? context : centre;
    }
    out.inputs.eval();
    out.outputs.eval();
    return out;
}

// `k` negatives per example from the noise table; one equal to the example's
// output is skipped (-1), as gensim does.
inline af::array DrawNegatives(const af::array& noise_table, const af::array& outputs, int k,
                               af::randomEngine& engine) {
    const dim_t b = outputs.elements();
    const af::array slot =
        af::min(af::floor(af::randu(af::dim4(b * k), f32, engine) * static_cast<float>(kNoiseTableSize)),
                static_cast<float>(kNoiseTableSize - 1))
            .as(s32);
    const af::array negatives = af::moddims(af::lookup(noise_table, slot), b, k);
    return af::select(negatives == af::tile(outputs, 1, static_cast<unsigned>(k)), -1.0, negatives).as(s32);
}

// The output side of a negative sampling step for hidden vectors `h` [B, d]:
// the true `outputs` [B] (label 1) and `negatives` [B, K] (label 0, -1 =
// none). Updates `w_out` by SGD on the summed loss and returns the loss (a
// device scalar); `grad_h` gets dLoss / dh [B, d], computed from the old rows.
inline af::array NegativeSamplingStep(af::array& w_out, const af::array& h, const af::array& outputs,
                                      const af::array& negatives, double alpha, af::array& grad_h) {
    const dim_t b = h.dims(0), d = h.dims(1);
    const dim_t k = negatives.elements() == 0 ? 0 : negatives.dims(1);
    const af::array targets = k > 0 ? af::join(1, outputs, negatives) : outputs;  // [B, K+1]
    const af::array flat_targets = af::flat(targets);
    const af::array target_valid = (flat_targets >= 0).as(f32);
    const af::array label = af::join(0, af::constant(1.0f, b), af::constant(0.0f, b * k));
    const af::array u = af::lookup(w_out, af::select(flat_targets >= 0, flat_targets, 0.0), 0);  // [B(K+1), d]
    const af::array h_rep = af::tile(h, static_cast<unsigned>(k + 1), 1);
    const af::array score = af::sum(u * h_rep, 1);
    const af::array loss = af::sum(af::select(label > 0.5f, Softplus(-score), Softplus(score)) * target_valid);
    const af::array g = (af::sigmoid(score) - label) * target_valid;  // dLoss / dscore
    const af::array g_rep = af::tile(g, 1, static_cast<unsigned>(d));
    af::array grad_out = g_rep * h_rep;
    grad_h = af::moddims(af::sum(af::moddims(g_rep * u, b, k + 1, d), 1), b, d);
    grad_out.eval();
    grad_h.eval();
    // Empty slots (-1) update row 0 by a zero gradient instead of being
    // filtered out: af::where would make the host wait for its count.
    ScatterSubtract(w_out, af::select(flat_targets >= 0, flat_targets, 0.0).as(s32), grad_out, alpha);
    return loss;
}

}  // namespace cyxwiz
