// GloVe training on ArrayFire (TOFIX152 P5): the algorithm of the Stanford
// reference (cooccur.c + glove.c), run on the selected device.
// - Counting: each chunk of the corpus makes its (word, context) pairs on the
//   device (a pair d positions apart in one sentence adds 1/d both ways); keys
//   row * V + col (64-bit) are sorted with the counts so far and summed by run.
//   ArrayFire sums by 32-bit keys only, so the runs of equal keys are numbered
//   first and summed by those numbers.
// - Training: weighted least squares f(X)(w.c + b + b~ - log X)^2 with
//   f = min(1, X/x_max)^alpha and AdaGrad, in minibatches of the shuffled pairs.
//   As in glove.c the step is scaled by the rate before AdaGrad: a row moves by
//   rate * g / sqrt(history) and its history grows by (rate * g)^2. Rows used by
//   several pairs of a batch take the sum of their steps, every row read before
//   the step. scripts/glove_reference.py (PyTorch) is the oracle.
// - Output: word + context vectors (glove.c's default model).

#ifdef _WIN32
#define NOMINMAX
#endif

#include "cyxwiz/word_embeddings.h"

#include "arrayfire_host_materialization.h"
#include "arrayfire_indexed.h"
#include "word_training_vocabulary.h"
#include "word_vector_device.h"

#include <arrayfire.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

namespace cyxwiz {

namespace {

// Pairs made per counting chunk (word, context) in both directions: 2^24
// keys, 128 MB of 64-bit keys plus their counts.
constexpr dim_t kPairsPerChunk = dim_t{1} << 24;
// The host waits for the device every so many training steps, so a stop is
// seen soon and the queue stays short.
constexpr int kSyncSteps = 64;

struct DeviceCounts {
    af::array keys;    // u64, sorted, distinct: row * V + col
    af::array values;  // f32
};

// Sums the values of equal keys; `keys` / `values` any order.
DeviceCounts ReduceByKey(const af::array& keys, const af::array& values) {
    DeviceCounts out;
    af::array sorted_keys, sorted_values;
    af::sort(sorted_keys, sorted_values, keys, values, 0, true);
    const dim_t m = sorted_keys.elements();
    if (m <= 1) {
        out.keys = sorted_keys;
        out.values = sorted_values;
        return out;
    }
    const af::array starts = af::join(0, af::constant(1, 1, b8),
                                      sorted_keys(af::seq(1, static_cast<double>(m) - 1)) !=
                                          sorted_keys(af::seq(0, static_cast<double>(m) - 2)));
    const af::array run = (af::accum(starts.as(s32)) - 1).as(s32);
    af::array run_ids, sums;
    af::sumByKey(run_ids, sums, run, sorted_values, 0);
    out.keys = af::lookup(sorted_keys, af::where(starts));
    out.values = sums;
    out.keys.eval();
    out.values.eval();
    return out;
}

// Co-occurrence counts of the corpus on the device. `progress` (may be empty)
// gets the share counted; returning false stops (counts then empty).
bool CountOnDevice(const std::vector<int>& tokens, const std::vector<int>& sentence_of, int vocab_size, int window,
                   const std::function<bool(double)>& progress, DeviceCounts& out) {
    const dim_t n = static_cast<dim_t>(tokens.size());
    const af::array ids(n, tokens.data());
    const af::array sentence(n, sentence_of.data());
    const af::array v = af::constant(static_cast<unsigned long long>(vocab_size), 1, u64);
    const dim_t chunk = std::max<dim_t>(1, kPairsPerChunk / (2 * window));
    out = DeviceCounts{};
    for (dim_t start = 0; start < n; start += chunk) {
        const dim_t end = std::min(n, start + chunk);
        af::array keys, values;
        for (int d = 1; d <= window; ++d) {
            const dim_t last = std::min(end, n - d);  // left words start .. last-1
            if (last <= start) continue;
            const af::seq left(static_cast<double>(start), static_cast<double>(last) - 1);
            const af::seq right(static_cast<double>(start + d), static_cast<double>(last + d) - 1);
            const af::array same = af::where(sentence(left) == sentence(right));
            if (same.elements() == 0) continue;
            const af::array a = af::lookup(ids(left), same).as(u64);
            const af::array b = af::lookup(ids(right), same).as(u64);
            const af::array forward = a * af::tile(v, static_cast<unsigned>(a.elements())) + b;
            const af::array backward = b * af::tile(v, static_cast<unsigned>(b.elements())) + a;
            const af::array weight = af::constant(1.0f / static_cast<float>(d), 2 * a.elements(), f32);
            const af::array pair_keys = af::join(0, forward, backward);
            keys = keys.isempty() ? pair_keys : af::join(0, keys, pair_keys);
            values = values.isempty() ? weight : af::join(0, values, weight);
        }
        if (!keys.isempty()) {
            if (out.keys.isempty()) {
                out = ReduceByKey(keys, values);
            } else {
                out = ReduceByKey(af::join(0, out.keys, keys), af::join(0, out.values, values));
            }
        }
        if (progress && !progress(static_cast<double>(end) / static_cast<double>(n))) return false;
    }
    return true;
}

struct DeviceState {
    af::array word, context, word_bias, context_bias;
    af::array word_sq, context_sq, word_bias_sq, context_bias_sq;
};

// P[row] -= sum of its steps / sqrt(H[row]); H[row] += sum of their squares.
void AdaGradRows(af::array& p, af::array& history, const af::array& rows, const af::array& steps) {
    if (rows.elements() == 1) {
        p(rows, af::span) -= steps / af::sqrt(history(rows, af::span));
        history(rows, af::span) += steps * steps;
        return;
    }
    af::array sorted, order;
    SortWithIndex(sorted, order, rows, 0, true);
    const af::array ordered = af::lookup(steps, order, 0);
    af::array unique, sums, unique_again, squares;
    af::sumByKey(unique, sums, sorted, ordered, 0);
    af::sumByKey(unique_again, squares, sorted, ordered * ordered, 0);
    p(unique, af::span) -= sums / af::sqrt(history(unique, af::span));
    history(unique, af::span) += squares;
}

// One step over pairs (i, j) with log counts and weights; returns the summed
// cost (a device scalar).
af::array Step(DeviceState& s, const af::array& i, const af::array& j, const af::array& log_x,
               const af::array& weight, double learning_rate) {
    const dim_t d = s.word.dims(1);
    const af::array wi = af::lookup(s.word, i, 0);
    const af::array cj = af::lookup(s.context, j, 0);
    const af::array diff = af::sum(wi * cj, 1) + af::lookup(s.word_bias, i) + af::lookup(s.context_bias, j) - log_x;
    const af::array fdiff = weight * diff;
    af::array cost = af::sum(0.5f * fdiff * diff);
    af::array step = static_cast<float>(learning_rate) * fdiff;
    af::array word_steps = cj * af::tile(step, 1, static_cast<unsigned>(d));
    af::array context_steps = wi * af::tile(step, 1, static_cast<unsigned>(d));
    // Every row is read before any is written.
    cost.eval();
    step.eval();
    word_steps.eval();
    context_steps.eval();
    AdaGradRows(s.word, s.word_sq, i, word_steps);
    AdaGradRows(s.context, s.context_sq, j, context_steps);
    AdaGradRows(s.word_bias, s.word_bias_sq, i, step);
    AdaGradRows(s.context_bias, s.context_bias_sq, j, step);
    s.word.eval();
    s.context.eval();
    s.word_bias.eval();
    s.context_bias.eval();
    s.word_sq.eval();
    s.context_sq.eval();
    s.word_bias_sq.eval();
    s.context_bias_sq.eval();
    return cost;
}

af::array Weights(const af::array& counts, double x_max, double alpha) {
    return af::pow(af::min(counts / static_cast<float>(x_max), 1.0f), static_cast<float>(alpha));
}

}  // namespace

GloVeCounts WordEmbeddings::CountCooccurrences(const std::vector<int>& tokens, const std::vector<int>& sentence_of,
                                               int vocab_size, int window, std::string* error) {
    GloVeCounts out;
    const auto fail = [&](const std::string& message) {
        out = GloVeCounts{};
        out.error_message = message;
        if (error) *error = message;
        return out;
    };
    if (vocab_size <= 0 || window < 1) return fail("The vocabulary must be non-empty and the window at least 1");
    if (tokens.size() != sentence_of.size()) return fail("Every token needs its sentence");
    for (const int id : tokens)
        if (id < 0 || id >= vocab_size) return fail("A token is outside the vocabulary");
    if (tokens.size() < 2) {
        out.success = true;
        return out;
    }
    try {
        DeviceCounts counts;
        CountOnDevice(tokens, sentence_of, vocab_size, window, {}, counts);
        if (counts.keys.isempty()) {
            out.success = true;
            return out;
        }
        const af::array v = af::constant(static_cast<unsigned long long>(vocab_size), counts.keys.elements(), u64);
        af::array rows = (counts.keys / v).as(s32);
        af::array cols = (counts.keys % v).as(s32);
        rows.eval();
        cols.eval();
        const size_t m = static_cast<size_t>(counts.keys.elements());
        out.rows.resize(m);
        out.cols.resize(m);
        out.values.resize(m);
        MaterializeArrayFireToHost(rows, out.rows.data(), ArrayFireHostSyncCategory::OutputMaterialization,
                                   "WordEmbeddings::CountCooccurrences", "rows");
        MaterializeArrayFireToHost(cols, out.cols.data(), ArrayFireHostSyncCategory::OutputMaterialization,
                                   "WordEmbeddings::CountCooccurrences", "cols");
        MaterializeArrayFireToHost(counts.values, out.values.data(), ArrayFireHostSyncCategory::OutputMaterialization,
                                   "WordEmbeddings::CountCooccurrences", "values");
        out.success = true;
        return out;
    } catch (const af::exception& e) {
        return fail(std::string("WordEmbeddings::CountCooccurrences failed on the ArrayFire device: ") + e.what());
    }
}

bool WordEmbeddings::GloVeSteps(GloVeState& state, int vocab_size, int dim, const std::vector<GloVeBatch>& batches,
                                double x_max, double alpha, double learning_rate, std::vector<double>* costs,
                                std::string* error) {
    const auto fail = [&](const std::string& message) {
        if (error) *error = message;
        return false;
    };
    const size_t table = static_cast<size_t>(vocab_size) * static_cast<size_t>(dim);
    const size_t rows = static_cast<size_t>(vocab_size);
    if (vocab_size <= 0 || dim <= 0 || state.word.size() != table || state.context.size() != table ||
        state.word_sq.size() != table || state.context_sq.size() != table || state.word_bias.size() != rows ||
        state.context_bias.size() != rows || state.word_bias_sq.size() != rows || state.context_bias_sq.size() != rows)
        return fail("The GloVe state must be [vocab_size, dim] vectors and [vocab_size] biases");
    if (!(x_max > 0.0) || !(learning_rate > 0.0)) return fail("x_max and the learning rate must be positive");
    for (const auto& batch : batches) {
        if (batch.rows.empty() || batch.rows.size() != batch.cols.size() || batch.rows.size() != batch.counts.size())
            return fail("A batch's rows, columns and counts do not agree in size");
        for (size_t k = 0; k < batch.rows.size(); ++k) {
            if (batch.rows[k] < 0 || batch.rows[k] >= vocab_size || batch.cols[k] < 0 || batch.cols[k] >= vocab_size)
                return fail("A batch word is outside the vocabulary");
            if (!(batch.counts[k] > 0.0f)) return fail("A co-occurrence count must be positive");
        }
    }
    try {
        DeviceState s;
        s.word = RowsToDevice(state.word, vocab_size, dim);
        s.context = RowsToDevice(state.context, vocab_size, dim);
        s.word_sq = RowsToDevice(state.word_sq, vocab_size, dim);
        s.context_sq = RowsToDevice(state.context_sq, vocab_size, dim);
        s.word_bias = af::array(vocab_size, state.word_bias.data());
        s.context_bias = af::array(vocab_size, state.context_bias.data());
        s.word_bias_sq = af::array(vocab_size, state.word_bias_sq.data());
        s.context_bias_sq = af::array(vocab_size, state.context_bias_sq.data());
        if (costs) costs->clear();
        for (const auto& batch : batches) {
            const dim_t b = static_cast<dim_t>(batch.rows.size());
            const af::array counts(b, batch.counts.data());
            const af::array cost = Step(s, af::array(b, batch.rows.data()), af::array(b, batch.cols.data()),
                                        af::log(counts), Weights(counts, x_max, alpha), learning_rate);
            if (costs) costs->push_back(af::sum<double>(cost));
        }
        state.word = RowsToHost(s.word, "WordEmbeddings::GloVeSteps");
        state.context = RowsToHost(s.context, "WordEmbeddings::GloVeSteps");
        state.word_sq = RowsToHost(s.word_sq, "WordEmbeddings::GloVeSteps");
        state.context_sq = RowsToHost(s.context_sq, "WordEmbeddings::GloVeSteps");
        state.word_bias = RowsToHost(s.word_bias, "WordEmbeddings::GloVeSteps");
        state.context_bias = RowsToHost(s.context_bias, "WordEmbeddings::GloVeSteps");
        state.word_bias_sq = RowsToHost(s.word_bias_sq, "WordEmbeddings::GloVeSteps");
        state.context_bias_sq = RowsToHost(s.context_bias_sq, "WordEmbeddings::GloVeSteps");
        return true;
    } catch (const af::exception& e) {
        return fail(std::string("WordEmbeddings::GloVeSteps failed on the ArrayFire device: ") + e.what());
    }
}

GloVeModel WordEmbeddings::TrainGloVe(const std::vector<std::vector<std::string>>& sentences,
                                      const GloVeOptions& options) {
    GloVeModel model;
    const auto fail = [&](const std::string& message) {
        model = GloVeModel{};
        model.error_message = message;
        return model;
    };
    if (options.vector_size < 1 || options.window < 1 || options.min_count < 1 || options.epochs < 1 ||
        options.batch_size < 1 || !(options.x_max > 0.0) || !(options.learning_rate > 0.0) || options.alpha < 0.0)
        return fail("vector_size, window, min_count, epochs and batch_size must be at least 1, x_max and the learning "
                    "rate positive, alpha not negative");

    TrainingVocabulary vocabulary;
    if (!BuildTrainingVocabulary(sentences, options.min_count, options.max_vocab, vocabulary))
        return fail("Fewer than two words appear at least min_count times");
    model.vectors.words = vocabulary.words;
    model.counts = vocabulary.counts;
    const int v = static_cast<int>(vocabulary.words.size());
    const dim_t d = options.vector_size;

    try {
        DeviceCounts counts;
        const auto counting = [&](double done) { return !options.on_progress || options.on_progress(done, 0); };
        if (!CountOnDevice(vocabulary.tokens, vocabulary.sentence_of, v, options.window, counting, counts))
            return fail("Training was stopped");
        // Counting leaves its sorted chunks in ArrayFire's memory cache; give
        // them back before training (on a 4 GB card the cache otherwise fills it).
        af::deviceGC();
        const dim_t n = counts.keys.elements();
        if (n == 0) return fail("The corpus gave no co-occurrences (sentences too short?)");
        model.pairs = static_cast<long long>(n);
        const af::array vv = af::constant(static_cast<unsigned long long>(v), n, u64);
        const af::array rows = (counts.keys / vv).as(s32);
        const af::array cols = (counts.keys % vv).as(s32);
        const af::array log_x = af::log(counts.values);
        const af::array weight = Weights(counts.values, options.x_max, options.alpha);
        rows.eval();
        cols.eval();
        log_x.eval();
        weight.eval();

        // glove.c: every vector and bias uniform in [-0.5, 0.5) / vector_size,
        // AdaGrad sums start at 1.
        af::randomEngine engine(AF_RANDOM_ENGINE_DEFAULT, options.seed);
        const float scale = 1.0f / static_cast<float>(d);
        DeviceState s;
        s.word = (af::randu(af::dim4(v, d), f32, engine) - 0.5f) * scale;
        s.context = (af::randu(af::dim4(v, d), f32, engine) - 0.5f) * scale;
        s.word_bias = (af::randu(af::dim4(v), f32, engine) - 0.5f) * scale;
        s.context_bias = (af::randu(af::dim4(v), f32, engine) - 0.5f) * scale;
        s.word_sq = af::constant(1.0f, v, d, f32);
        s.context_sq = af::constant(1.0f, v, d, f32);
        s.word_bias_sq = af::constant(1.0f, v, f32);
        s.context_bias_sq = af::constant(1.0f, v, f32);

        const dim_t batch = options.batch_size;
        const dim_t steps_per_epoch = (n + batch - 1) / batch;
        for (int epoch = 0; epoch < options.epochs; ++epoch) {
            af::array shuffled, order;
            SortWithIndex(shuffled, order, af::randu(af::dim4(n), f32, engine), 0, true);
            af::array epoch_cost = af::constant(0.0f, 1, f32);
            dim_t step_no = 0;
            for (dim_t start = 0; start < n; start += batch, ++step_no) {
                const dim_t end = std::min(n, start + batch);
                // af::lookup, not x(index array): see the oneAPI note in word2vec.cpp.
                const af::array pick = order(af::seq(static_cast<double>(start), static_cast<double>(end) - 1));
                epoch_cost += Step(s, af::lookup(rows, pick), af::lookup(cols, pick), af::lookup(log_x, pick),
                                   af::lookup(weight, pick), options.learning_rate);
                epoch_cost.eval();
                if ((step_no + 1) % kSyncSteps == 0) af::sync();
                if (options.on_progress &&
                    !options.on_progress((static_cast<double>(epoch) +
                                          static_cast<double>(step_no + 1) / static_cast<double>(steps_per_epoch)) /
                                             options.epochs,
                                         epoch + 1))
                    return fail("Training was stopped");
            }
            model.epoch_cost.push_back(epoch_cost.scalar<float>() / static_cast<double>(n));
            if (options.on_epoch_cost) options.on_epoch_cost(epoch + 1, model.epoch_cost.back());
        }
        model.vectors.values = RowsToHost(s.word + s.context, "WordEmbeddings::TrainGloVe");
        model.vectors.dim = static_cast<int>(d);
        model.vectors.success = true;
        model.success = true;
    } catch (const af::exception& e) {
        return fail(std::string("WordEmbeddings::TrainGloVe failed on the ArrayFire device: ") + e.what());
    }
    return model;
}

}  // namespace cyxwiz
