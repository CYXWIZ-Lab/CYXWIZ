// word2vec training on ArrayFire (TOFIX152): skip-gram and CBOW with negative
// sampling, gensim's settings (Word2Vec), run in mini-batches on the device.
// - Step: SGD on the sum of a batch's losses; a row used by several examples
//   gets the sum of their gradients (sort-by-key + sum-by-key, as the Embedding
//   backward). PyTorch with the same matrices and batches is the oracle.
// - Trainer: the vocabulary (min_count, most frequent first) is counted on the
//   host from the strings; everything numeric runs on the device: down-sampling
//   (gensim's keep probability), each position's window drawn from 1..window,
//   the (context, centre) examples, negatives from a unigram^0.75 table, and the
//   learning rate falling linearly from alpha to min_alpha.
// gensim trains one example at a time on its own generator, so its vectors and
// ours differ; the parity check compares their neighbours, not their values.

#ifdef _WIN32
#define NOMINMAX
#endif

#include "cyxwiz/word_embeddings.h"

#include "arrayfire_host_materialization.h"
#include "arrayfire_indexed.h"

#include <arrayfire.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cyxwiz {

namespace {

constexpr dim_t kSyncSteps = 64;

// W[key] -= alpha * (sum of the rows with that key).
void ScatterSubtract(af::array& w, const af::array& keys, const af::array& rows, double alpha) {
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

af::array Softplus(const af::array& x) {
    return af::max(x, 0.0f) + af::log1p(af::exp(-af::abs(x)));
}

// One step. `inputs` [B, C] (-1 = no word), `outputs` [B], `negatives` [B, K]
// (-1 = none), all s32. Returns the summed loss (a device scalar).
af::array Step(af::array& w_in, af::array& w_out, const af::array& inputs, const af::array& outputs,
               const af::array& negatives, double alpha) {
    const dim_t b = inputs.dims(0), c = inputs.dims(1), d = w_in.dims(1);
    const dim_t k = negatives.elements() == 0 ? 0 : negatives.dims(1);

    // h: the input word (skip-gram) or the mean of the context words (CBOW).
    const af::array valid = inputs >= 0;
    const af::array weight = valid.as(f32);
    const af::array count = af::max(af::sum(weight, 1), 1.0f);  // [B]
    const af::array rows_in = af::lookup(w_in, af::flat(af::select(valid, inputs, 0.0)), 0);  // [B*C, d]
    const af::array masked = af::moddims(rows_in, b, c, d) * af::tile(af::moddims(weight, b, c, 1), 1, 1, static_cast<unsigned>(d));
    const af::array h = af::moddims(af::sum(masked, 1), b, d) / af::tile(count, 1, static_cast<unsigned>(d));

    // Targets: the true output (label 1) then the negatives (label 0).
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
    const af::array grad_h = af::moddims(af::sum(af::moddims(g_rep * u, b, k + 1, d), 1), b, d);
    // Each context word gets its share of the mean's gradient; an empty slot
    // gets none.
    af::array grad_in = af::moddims(
        af::tile(af::moddims(grad_h / af::tile(count, 1, static_cast<unsigned>(d)), b, 1, d), 1, static_cast<unsigned>(c), 1),
        b * c, d) * af::tile(af::flat(weight), 1, static_cast<unsigned>(d));
    grad_out.eval();
    grad_in.eval();

    // Empty slots (-1) update row 0 by a zero gradient instead of being
    // filtered out: af::where would make the host wait for its count.
    ScatterSubtract(w_out, af::select(flat_targets >= 0, flat_targets, 0.0).as(s32), grad_out, alpha);
    ScatterSubtract(w_in, af::flat(af::select(valid, inputs, 0.0)).as(s32), grad_in, alpha);
    w_in.eval();
    w_out.eval();
    return loss;
}

// Row-major host [rows, cols] <-> device [rows, cols].
af::array RowsToDevice(const std::vector<float>& values, dim_t rows, dim_t cols) {
    return af::array(cols, rows, values.data()).T();
}

std::vector<float> RowsToHost(const af::array& matrix, const char* operation) {
    std::vector<float> values(static_cast<size_t>(matrix.elements()));
    af::array transposed = matrix.T();
    transposed.eval();
    MaterializeArrayFireToHost(transposed, values.data(), ArrayFireHostSyncCategory::OutputMaterialization, operation,
                               "row_major_words_by_dim");
    return values;
}

af::array IntsToDevice(const std::vector<int>& values, dim_t rows, dim_t cols) {
    // Host row-major [rows, cols] -> device [rows, cols].
    return af::array(cols, rows, values.data()).T();
}

}  // namespace

bool WordEmbeddings::Word2VecSteps(std::vector<float>& input_vectors, std::vector<float>& output_vectors,
                                   int vocab_size, int dim, const std::vector<Word2VecBatch>& batches,
                                   std::vector<double>* losses, std::string* error) {
    const auto fail = [&](const std::string& message) {
        if (error) *error = message;
        return false;
    };
    const size_t table = static_cast<size_t>(vocab_size) * static_cast<size_t>(dim);
    if (vocab_size <= 0 || dim <= 0 || input_vectors.size() != table || output_vectors.size() != table)
        return fail("The vectors must be [vocab_size, dim]");
    for (const auto& batch : batches) {
        const size_t b = batch.outputs.size();
        if (b == 0 || batch.context_width < 1 || batch.negative < 0 ||
            batch.inputs.size() != b * static_cast<size_t>(batch.context_width) ||
            batch.negatives.size() != b * static_cast<size_t>(batch.negative))
            return fail("A batch's inputs, outputs and negatives do not agree in size");
        const auto in_range = [&](int id, bool allow_none) { return (allow_none && id == -1) || (id >= 0 && id < vocab_size); };
        for (const int id : batch.inputs)
            if (!in_range(id, batch.context_width > 1)) return fail("A batch word is outside the vocabulary");
        for (const int id : batch.outputs)
            if (!in_range(id, false)) return fail("A batch word is outside the vocabulary");
        for (const int id : batch.negatives)
            if (!in_range(id, true)) return fail("A batch word is outside the vocabulary");
    }
    try {
        af::array w_in = RowsToDevice(input_vectors, vocab_size, dim);
        af::array w_out = RowsToDevice(output_vectors, vocab_size, dim);
        if (losses) losses->clear();
        for (const auto& batch : batches) {
            const dim_t b = static_cast<dim_t>(batch.outputs.size());
            const af::array loss =
                Step(w_in, w_out, IntsToDevice(batch.inputs, b, batch.context_width),
                     af::array(b, batch.outputs.data()),
                     batch.negative > 0 ? IntsToDevice(batch.negatives, b, batch.negative) : af::array(), batch.alpha);
            if (losses) losses->push_back(af::sum<double>(loss));
        }
        input_vectors = RowsToHost(w_in, "WordEmbeddings::Word2VecSteps");
        output_vectors = RowsToHost(w_out, "WordEmbeddings::Word2VecSteps");
        return true;
    } catch (const af::exception& e) {
        return fail(std::string("WordEmbeddings::Word2VecSteps failed on the ArrayFire device: ") + e.what());
    }
}

Word2VecModel WordEmbeddings::TrainWord2Vec(const std::vector<std::vector<std::string>>& sentences,
                                            const Word2VecOptions& options) {
    Word2VecModel model;
    const auto fail = [&](const std::string& message) {
        model = Word2VecModel{};
        model.error_message = message;
        return model;
    };
    if (options.vector_size < 1 || options.window < 1 || options.min_count < 1 || options.negative < 1 ||
        options.epochs < 1 || options.batch_size < 1 || !(options.alpha > 0.0) || options.min_alpha < 0.0 ||
        options.sample < 0.0)
        return fail("vector_size, window, min_count, negative, epochs and batch_size must be at least 1, alpha "
                    "positive, min_alpha and sample not negative");

    // Vocabulary: counts from the strings (host), most frequent first; ties keep
    // the order words were first seen (gensim's stable sort).
    std::unordered_map<std::string, size_t> first_seen;
    std::vector<std::string> seen_words;
    std::vector<long long> seen_counts;
    for (const auto& sentence : sentences) {
        for (const auto& word : sentence) {
            const auto [it, inserted] = first_seen.emplace(word, seen_words.size());
            if (inserted) {
                seen_words.push_back(word);
                seen_counts.push_back(0);
            }
            ++seen_counts[it->second];
        }
    }
    std::vector<size_t> order;
    for (size_t i = 0; i < seen_words.size(); ++i)
        if (seen_counts[i] >= options.min_count) order.push_back(i);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return seen_counts[a] > seen_counts[b]; });
    if (options.max_vocab > 0 && order.size() > options.max_vocab) order.resize(options.max_vocab);
    if (order.size() < 2) return fail("Fewer than two words appear at least min_count times");
    std::unordered_map<std::string, int> id_of;
    for (size_t r = 0; r < order.size(); ++r) {
        id_of.emplace(seen_words[order[r]], static_cast<int>(r));
        model.vectors.words.push_back(seen_words[order[r]]);
        model.counts.push_back(seen_counts[order[r]]);
    }
    // The corpus as ids, words outside the vocabulary dropped (gensim drops them
    // before the windows), with each word's sentence.
    std::vector<int> tokens, sentence_of;
    for (size_t s = 0; s < sentences.size(); ++s) {
        for (const auto& word : sentences[s]) {
            if (const auto it = id_of.find(word); it != id_of.end()) {
                tokens.push_back(it->second);
                sentence_of.push_back(static_cast<int>(s));
            }
        }
    }
    const dim_t v = static_cast<dim_t>(model.vectors.words.size()), d = options.vector_size;
    const dim_t n = static_cast<dim_t>(tokens.size());
    const int window = options.window, k = options.negative;

    try {
        af::randomEngine engine(AF_RANDOM_ENGINE_DEFAULT, options.seed);
        // gensim: input vectors uniform in +-0.5 / vector_size, output vectors 0.
        af::array w_in = (af::randu(af::dim4(v, d), f32, engine) - 0.5f) / static_cast<float>(d);
        af::array w_out = af::constant(0.0f, v, d, f32);
        const af::array corpus(n, tokens.data());
        const af::array corpus_sentence(n, sentence_of.data());
        std::vector<float> counts_float(model.counts.size());
        std::transform(model.counts.begin(), model.counts.end(), counts_float.begin(),
                       [](long long c) { return static_cast<float>(c); });
        const af::array count_f(v, counts_float.data());

        // Down-sampling: gensim's keep probability per word.
        af::array keep_probability = af::constant(1.0f, v, f32);
        if (options.sample > 0.0) {
            const float threshold = static_cast<float>(options.sample * static_cast<double>(af::sum<double>(count_f)));
            keep_probability = af::min((af::sqrt(count_f / threshold) + 1.0f) * (threshold / count_f), 1.0f);
        }
        // Negatives: a table where each word fills a share of slots
        // proportional to count^0.75.
        constexpr dim_t kTableSize = 1000000;
        const af::array power = af::pow(count_f, 0.75f);
        const af::array slots = af::floor(af::scan(power / af::sum<float>(power), 0) * static_cast<float>(kTableSize));
        const af::array starts = af::join(0, af::constant(0.0f, 1), slots(af::seq(0, static_cast<double>(v) - 2)));
        const af::array has_slots = slots > starts;
        const af::array start_words = af::where(has_slots);
        // Slot t belongs to the last word that starts at or before it: mark
        // each start with its word and carry the running maximum.
        af::array word_at_start = af::constant(-1, kTableSize, s32);
        word_at_start(starts(start_words).as(s32)) = start_words.as(s32);
        const af::array noise_table = af::scan(word_at_start, 0, AF_BINARY_MAX);

        const double total_estimate = static_cast<double>(options.epochs) * static_cast<double>(n) * 2.0 * window;
        double done = 0.0;
        for (int epoch = 0; epoch < options.epochs; ++epoch) {
            // Kept words of this epoch, in corpus order.
            const af::array kept = af::where(af::randu(af::dim4(n), f32, engine) < keep_probability(corpus));
            const dim_t m = kept.elements();
            if (m < 2) continue;
            const af::array words = corpus(kept).as(s32);
            const af::array sentence = corpus_sentence(kept).as(s32);
            // gensim: reduced_window = random % window, so the reach is 1..window.
            const af::array reach =
                af::min(af::floor(af::randu(af::dim4(m), f32, engine) * static_cast<float>(window)), static_cast<float>(window - 1));
            const af::array position = af::range(af::dim4(m), 0, s32);

            // Context slot (o, side) of each centre: valid when inside the
            // sentence and within the centre's reach.
            std::vector<af::array> context_cols;
            for (int o = 1; o <= window; ++o) {
                for (const int side : {-1, 1}) {
                    const af::array other = position + side * o;
                    const af::array inside = other >= 0 && other < static_cast<int>(m);
                    const af::array clamped = af::select(inside, other, 0.0);
                    const af::array ok = inside && (sentence(clamped) == sentence) &&
                                         (static_cast<float>(window - o) >= reach);
                    context_cols.push_back(af::select(ok, words(clamped), -1.0));
                }
            }
            af::array contexts = context_cols.front();
            for (size_t i = 1; i < context_cols.size(); ++i) contexts = af::join(1, contexts, context_cols[i]);  // [m, 2W]
            contexts.eval();

            // Examples in corpus order: CBOW one per centre; skip-gram one per
            // (centre, context) with the context word as input (gensim).
            af::array ex_inputs, ex_outputs;
            if (options.cbow) {
                const af::array any = af::anyTrue(contexts >= 0, 1);
                const af::array centres = af::where(any);
                ex_inputs = contexts(centres, af::span);
                ex_outputs = words(centres);
            } else {
                const af::array flat_by_centre = af::flat(contexts.T());  // centre-major
                const af::array valid = af::where(flat_by_centre >= 0);
                ex_inputs = flat_by_centre(valid);
                ex_outputs = words(valid / static_cast<unsigned>(2 * window)).as(s32);
            }
            ex_inputs.eval();
            ex_outputs.eval();
            const dim_t examples = ex_outputs.elements();
            // The loss stays on the device; reading it each step would stall
            // the queue. The host waits every kSyncSteps steps, so a stop is
            // seen soon and the queue stays short.
            af::array epoch_loss = af::constant(0.0f, 1, f32);
            dim_t steps = 0;
            for (dim_t start = 0; start < examples; start += options.batch_size) {
                const dim_t end = std::min(examples, start + static_cast<dim_t>(options.batch_size));
                const af::seq rows(static_cast<double>(start), static_cast<double>(end - 1));
                const af::array batch_out = ex_outputs(rows);
                const af::array batch_in = options.cbow ? ex_inputs(rows, af::span) : af::array(ex_inputs(rows));
                const af::array slot =
                    af::min(af::floor(af::randu(af::dim4((end - start) * k), f32, engine) * static_cast<float>(kTableSize)),
                            static_cast<float>(kTableSize - 1))
                        .as(s32);
                af::array negatives = af::moddims(af::lookup(noise_table, slot), end - start, k);
                // gensim skips a negative that is the target word.
                negatives = af::select(negatives == af::tile(batch_out, 1, static_cast<unsigned>(k)), -1.0, negatives);
                const double progress = std::min(1.0, done / total_estimate);
                const double alpha = options.alpha - (options.alpha - options.min_alpha) * progress;
                epoch_loss += af::sum(Step(w_in, w_out, batch_in, batch_out, negatives, alpha));
                epoch_loss.eval();
                if (++steps % kSyncSteps == 0) af::sync();
                done += static_cast<double>(end - start);
                if (options.on_progress &&
                    !options.on_progress(std::min(1.0, (static_cast<double>(epoch) + static_cast<double>(end) /
                                                        static_cast<double>(examples)) / options.epochs), epoch + 1))
                    return fail("Training was stopped");
            }
            model.examples += examples;
            model.epoch_loss.push_back(examples > 0 ? epoch_loss.scalar<float>() / static_cast<double>(examples) : 0.0);
        }
        if (model.examples == 0) return fail("The corpus gave no training examples (sentences too short?)");
        model.vectors.values = RowsToHost(w_in, "WordEmbeddings::TrainWord2Vec");
        model.output_vectors = RowsToHost(w_out, "WordEmbeddings::TrainWord2Vec");
        model.vectors.dim = static_cast<int>(d);
        model.vectors.success = true;
        model.success = true;
    } catch (const af::exception& e) {
        return fail(std::string("WordEmbeddings::TrainWord2Vec failed on the ArrayFire device: ") + e.what());
    }
    return model;
}

}  // namespace cyxwiz
