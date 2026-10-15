// fastText training on ArrayFire (TOFIX152 P6): word2vec with character
// n-grams (Bojanowski et al. 2017), gensim's settings (FastText).
// - Pieces: the n-grams of "<word>" (min_n..max_n characters, UTF-8 aware),
//   hashed with fastText's FNV-1a into `bucket` rows; computed on the host from
//   the strings, as the vocabulary is.
// - Input table [words + bucket, dim]: a word's input vector is the mean of its
//   row and its n-grams' rows (fastText averages over all of them). Every one of
//   those rows takes the whole gradient of the mean, as fastText and gensim
//   update; rows used several times in a batch get the sum (sort-by-key +
//   sum-by-key). It trains stably because each epoch's examples are shuffled
//   (word2vec_sampling.h); in corpus order it diverged. Giving each row only its
//   share instead left the neighbours far from gensim's (overlap 0.42 against
//   0.70, gensim's seed-to-seed 0.74, on 20,000 sentiment posts).
// - Skip-gram: the centre word (with its pieces) predicts each context word;
//   CBOW: the mean over the context words' rows and pieces predicts the centre.
// - Output side, sampling and negatives: word2vec's (word2vec_sampling.h).

#ifdef _WIN32
#define NOMINMAX
#endif

#include "cyxwiz/word_embeddings.h"

#include "arrayfire_host_materialization.h"
#include "arrayfire_indexed.h"
#include "word2vec_sampling.h"
#include "word_training_vocabulary.h"
#include "word_vector_device.h"

#include <arrayfire.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cyxwiz {

namespace {

constexpr dim_t kSyncSteps = 64;

// The rows of each word in the input table: its own row, then its n-gram
// buckets offset by the number of words; `start` [words + 1].
struct PieceRows {
    std::vector<int> start;
    std::vector<int> rows;
    int widest = 0;  // the most rows of one word
};

PieceRows BuildPieceRows(const std::vector<std::string>& words, int min_n, int max_n, int bucket) {
    PieceRows out;
    out.start.reserve(words.size() + 1);
    out.start.push_back(0);
    const int v = static_cast<int>(words.size());
    for (int w = 0; w < v; ++w) {
        out.rows.push_back(w);
        for (const int b : WordEmbeddings::FastTextBuckets(words[static_cast<size_t>(w)], min_n, max_n, bucket))
            out.rows.push_back(v + b);
        out.start.push_back(static_cast<int>(out.rows.size()));
        out.widest = std::max(out.widest, out.start.back() - out.start[out.start.size() - 2]);
    }
    return out;
}

// Skip-gram reads each word's rows from a [V, widest] table padded with -1,
// one gather per batch with no host wait, when that table stays small.
constexpr size_t kPaddedPiecesLimit = size_t{1} << 26;

af::array IntsToDevice(const std::vector<int>& values, dim_t rows, dim_t cols) {
    // Host row-major [rows, cols] -> device [rows, cols].
    return af::array(cols, rows, values.data()).T();
}

// The device side of PieceRows: start [V+1], rows [R], count [V] (all s32),
// and `padded` [V, widest] when it fits (empty otherwise). With a narrower
// width, `narrow` [V, width] holds the first rows of each word (whole for the
// words that fit) and `long_word` [V] (s32 1 / 0) marks the others.
struct DevicePieces {
    af::array start;
    af::array rows;
    af::array count;
    af::array padded;
    af::array narrow;
    af::array long_word;
};

DevicePieces ToDevice(const PieceRows& pieces, int narrow_width = 0) {
    DevicePieces d;
    d.start = af::array(static_cast<dim_t>(pieces.start.size()), pieces.start.data());
    d.rows = af::array(static_cast<dim_t>(pieces.rows.size()), pieces.rows.data());
    const dim_t v = static_cast<dim_t>(pieces.start.size()) - 1;
    d.count = d.start(af::seq(1, static_cast<double>(v))) - d.start(af::seq(0, static_cast<double>(v) - 1));
    const size_t width = static_cast<size_t>(pieces.widest);
    if (width > 0 && static_cast<size_t>(v) * width <= kPaddedPiecesLimit) {
        std::vector<int> padded(static_cast<size_t>(v) * width, -1);
        for (size_t w = 0; w < static_cast<size_t>(v); ++w)
            for (int r = pieces.start[w]; r < pieces.start[w + 1]; ++r)
                padded[w * width + static_cast<size_t>(r - pieces.start[w])] = pieces.rows[static_cast<size_t>(r)];
        d.padded = IntsToDevice(padded, v, static_cast<dim_t>(width));
        if (narrow_width > 0 && static_cast<size_t>(narrow_width) < width) {
            d.narrow = d.padded(af::span, af::seq(0, narrow_width - 1)).copy();
            d.long_word = (d.count > narrow_width).as(s32);
        }
    }
    return d;
}

// The fewest rows that hold all the rows of the words making up `share` of
// the text (skip-gram's narrow table).
int NarrowWidth(const PieceRows& pieces, const std::vector<long long>& counts, double share) {
    std::vector<std::pair<int, long long>> by_rows;
    long long total = 0;
    for (size_t w = 0; w < counts.size(); ++w) {
        by_rows.emplace_back(pieces.start[w + 1] - pieces.start[w], counts[w]);
        total += counts[w];
    }
    std::sort(by_rows.begin(), by_rows.end());
    long long covered = 0;
    for (const auto& [rows, count] : by_rows) {
        covered += count;
        if (static_cast<double>(covered) >= share * static_cast<double>(total)) return rows;
    }
    return pieces.widest;
}

// The input-table rows of a batch, one per (example, input word, piece):
// `inputs` [B, C] (-1 = none) and the total `total` (known on the host).
// Returns the rows [T] and each row's example [T].
void ExpandRows(const DevicePieces& pieces, const af::array& inputs, dim_t total, af::array& rows, af::array& example) {
    const dim_t b = inputs.dims(0), c = inputs.dims(1);
    // Items example-major: item i = e * C + slot.
    const af::array items = af::flat(inputs.T());
    const af::array valid = items >= 0;
    const af::array safe = af::select(valid, items, 0.0).as(s32);
    const af::array count = af::select(valid, af::lookup(pieces.count, safe), 0.0).as(s32);
    const af::array ends = af::scan(count, 0);  // inclusive; the last is `total`
    const af::array begins = ends - count;
    // The item of row p: write each item's index at its first row (an item
    // without rows writes to a spare slot past the end) and carry the running
    // maximum; the first rows are distinct, so no two writes meet in range.
    const af::array slot = af::select(count > 0, begins, static_cast<double>(total)).as(s32);
    af::array marks = af::constant(-1, total + 1, s32);
    marks(slot) = af::range(af::dim4(b * c), 0, s32);
    const af::array item = af::scan(marks(af::seq(0, static_cast<double>(total) - 1)), 0, AF_BINARY_MAX);
    const af::array p = af::range(af::dim4(total), 0, s32);
    const af::array within = p - af::lookup(begins, item);
    const af::array word = af::lookup(safe, item);
    rows = af::lookup(pieces.rows, af::lookup(pieces.start, word) + within).as(s32);
    example = (item / static_cast<int>(c)).as(s32);
    rows.eval();
    example.eval();
}

// The number of input-table rows of each example: `inputs` [B, C].
af::array RowsPerExample(const DevicePieces& pieces, const af::array& inputs) {
    const af::array valid = inputs >= 0;
    const af::array safe = af::select(valid, inputs, 0.0).as(s32);
    const af::array count =
        af::moddims(af::select(af::flat(valid), af::lookup(pieces.count, af::flat(safe)), 0.0), inputs.dims(0),
                    inputs.dims(1));
    return af::sum(count, 1).as(s32);
}

// One skip-gram step from the padded table: `inputs` [B, 1].
af::array StepPadded(af::array& w_in, af::array& w_out, const af::array& table, const af::array& inputs,
                     const af::array& outputs, const af::array& negatives, double alpha) {
    const dim_t b = inputs.dims(0), width = table.dims(1), d = w_in.dims(1);
    const af::array ids = af::lookup(table, af::flat(inputs), 0);  // [B, width], -1 = none
    const af::array valid = af::flat(ids >= 0).as(f32);                    // entry b + B * slot
    const af::array keys = af::select(valid > 0.5f, af::flat(ids), 0.0).as(s32);
    keys.eval();
    // h in its own scope: nothing refers to w_in when it is updated, so the
    // update writes in place instead of copying the table.
    af::array h;
    const af::array count = af::sum(af::moddims(valid, b, width), 1);  // [B]
    {
        const af::array rows = af::lookup(w_in, keys, 0) * af::tile(valid, 1, static_cast<unsigned>(d));
        h = af::moddims(af::sum(af::moddims(rows, b, width, d), 1), b, d) /
            af::tile(count, 1, static_cast<unsigned>(d));
        h.eval();
    }
    af::array grad_h;
    const af::array loss = NegativeSamplingStep(w_out, h, outputs, negatives, alpha, grad_h);
    // Every row of the mean takes the whole gradient (an empty slot updates
    // row 0 by zero).
    af::array grad_rows =
        af::tile(grad_h, static_cast<unsigned>(width), 1) * af::tile(valid, 1, static_cast<unsigned>(d));
    grad_rows.eval();
    ScatterSubtract(w_in, keys, grad_rows, alpha);
    w_in.eval();
    w_out.eval();
    return loss;
}

// One step on `w_in` [V + bucket, d] and `w_out` [V, d]; `total` rows in the
// batch (needed only when the padded table is not used). Returns the summed
// loss (a device scalar).
af::array Step(af::array& w_in, af::array& w_out, const DevicePieces& pieces, const af::array& inputs,
               const af::array& outputs, const af::array& negatives, dim_t total, double alpha) {
    const dim_t b = inputs.dims(0), c = inputs.dims(1), d = w_in.dims(1);
    if (c == 1 && !pieces.padded.isempty())
        return StepPadded(w_in, w_out, pieces.padded, inputs, outputs, negatives, alpha);
    af::array rows, example;
    ExpandRows(pieces, inputs, total, rows, example);
    // h: the mean of the example's rows (every example has one); the rows per
    // example from its input words, so one sum-by-key.
    af::array keys, sums;
    af::sumByKey(keys, sums, example, af::lookup(w_in, rows, 0), 0);
    const af::array per_item = af::moddims(RowsPerExample(pieces, inputs), b, 1);
    const af::array counts = per_item.as(f32);
    const af::array h = sums / af::tile(counts, 1, static_cast<unsigned>(d));
    h.eval();

    af::array grad_h;
    const af::array loss = NegativeSamplingStep(w_out, h, outputs, negatives, alpha, grad_h);
    // Every row of the mean takes the whole gradient.
    af::array grad_rows = af::lookup(grad_h, example, 0);
    grad_rows.eval();
    ScatterSubtract(w_in, rows, grad_rows, alpha);
    w_in.eval();
    w_out.eval();
    return loss;
}

}  // namespace

uint32_t WordEmbeddings::FastTextHash(const std::string& bytes) {
    uint32_t h = 2166136261u;
    for (const char c : bytes) {
        h = h ^ static_cast<uint32_t>(static_cast<int8_t>(c));
        h = h * 16777619u;
    }
    return h;
}

std::vector<std::string> WordEmbeddings::FastTextNgrams(const std::string& word, int min_n, int max_n) {
    // fastText's computeSubwords: start at each character (skipping UTF-8
    // continuation bytes) and extend one character at a time.
    std::vector<std::string> out;
    if (min_n < 1 || max_n < min_n) return out;
    const std::string s = "<" + word + ">";
    const auto continuation = [&](size_t i) { return (static_cast<unsigned char>(s[i]) & 0xC0) == 0x80; };
    for (size_t i = 0; i < s.size(); ++i) {
        if (continuation(i)) continue;
        size_t j = i;
        for (int n = 1; j < s.size() && n <= max_n; ++n) {
            ++j;
            while (j < s.size() && continuation(j)) ++j;
            if (n >= min_n && !(n == 1 && (i == 0 || j == s.size()))) out.push_back(s.substr(i, j - i));
        }
    }
    return out;
}

std::vector<int> WordEmbeddings::FastTextBuckets(const std::string& word, int min_n, int max_n, int bucket) {
    std::vector<int> out;
    if (bucket <= 0) return out;
    for (const auto& gram : FastTextNgrams(word, min_n, max_n))
        out.push_back(static_cast<int>(FastTextHash(gram) % static_cast<uint32_t>(bucket)));
    return out;
}

bool WordEmbeddings::FastTextSteps(FastTextModel& model, const std::vector<Word2VecBatch>& batches,
                                   std::vector<double>* losses, std::string* error) {
    const auto fail = [&](const std::string& message) {
        if (error) *error = message;
        return false;
    };
    const int v = static_cast<int>(model.words.size()), dim = model.dim;
    const size_t table = static_cast<size_t>(v) * static_cast<size_t>(dim);
    if (v <= 0 || dim <= 0 || model.bucket < 0 || model.word_rows.size() != table ||
        model.output_vectors.size() != table ||
        model.ngram_rows.size() != static_cast<size_t>(model.bucket) * static_cast<size_t>(dim))
        return fail("The model's rows must be [words, dim], [bucket, dim] and [words, dim]");
    for (const auto& batch : batches) {
        const size_t b = batch.outputs.size();
        if (b == 0 || batch.context_width < 1 || batch.negative < 0 ||
            batch.inputs.size() != b * static_cast<size_t>(batch.context_width) ||
            batch.negatives.size() != b * static_cast<size_t>(batch.negative))
            return fail("A batch's inputs, outputs and negatives do not agree in size");
        const auto in_range = [&](int id, bool allow_none) { return (allow_none && id == -1) || (id >= 0 && id < v); };
        for (size_t e = 0; e < b; ++e) {
            bool any = false;
            for (int c = 0; c < batch.context_width; ++c) {
                const int id = batch.inputs[e * static_cast<size_t>(batch.context_width) + static_cast<size_t>(c)];
                if (!in_range(id, batch.context_width > 1)) return fail("A batch word is outside the vocabulary");
                any = any || id >= 0;
            }
            if (!any) return fail("A batch example has no input word");
        }
        for (const int id : batch.outputs)
            if (!in_range(id, false)) return fail("A batch word is outside the vocabulary");
        for (const int id : batch.negatives)
            if (!in_range(id, true)) return fail("A batch word is outside the vocabulary");
    }
    try {
        const PieceRows host_pieces = BuildPieceRows(model.words, model.min_n, model.max_n, model.bucket);
        const DevicePieces pieces = ToDevice(host_pieces);
        af::array w_in = RowsToDevice(model.word_rows, v, dim);
        if (model.bucket > 0) w_in = af::join(0, w_in, RowsToDevice(model.ngram_rows, model.bucket, dim));
        af::array w_out = RowsToDevice(model.output_vectors, v, dim);
        if (losses) losses->clear();
        for (const auto& batch : batches) {
            const dim_t b = static_cast<dim_t>(batch.outputs.size());
            const af::array inputs = IntsToDevice(batch.inputs, b, batch.context_width);
            dim_t total = 0;
            for (const int id : batch.inputs) {
                if (id < 0) continue;
                const size_t w = static_cast<size_t>(id);
                total += host_pieces.start[w + 1] - host_pieces.start[w];
            }
            const af::array loss =
                Step(w_in, w_out, pieces, inputs, af::array(b, batch.outputs.data()),
                     batch.negative > 0 ? IntsToDevice(batch.negatives, b, batch.negative) : af::array(), total,
                     batch.alpha);
            if (losses) losses->push_back(af::sum<double>(loss));
        }
        const std::vector<float> all = RowsToHost(w_in, "WordEmbeddings::FastTextSteps");
        model.word_rows.assign(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(table));
        model.ngram_rows.assign(all.begin() + static_cast<std::ptrdiff_t>(table), all.end());
        model.output_vectors = RowsToHost(w_out, "WordEmbeddings::FastTextSteps");
        return true;
    } catch (const af::exception& e) {
        return fail(std::string("WordEmbeddings::FastTextSteps failed on the ArrayFire device: ") + e.what());
    }
}

FastTextModel WordEmbeddings::TrainFastText(const std::vector<std::vector<std::string>>& sentences,
                                            const FastTextOptions& options) {
    FastTextModel model;
    const auto fail = [&](const std::string& message) {
        model = FastTextModel{};
        model.error_message = message;
        return model;
    };
    if (options.vector_size < 1 || options.window < 1 || options.min_count < 1 || options.negative < 1 ||
        options.epochs < 1 || options.batch_size < 1 || !(options.alpha > 0.0) || options.min_alpha < 0.0 ||
        options.sample < 0.0 || options.min_n < 1 || options.max_n < options.min_n || options.bucket < 0)
        return fail("vector_size, window, min_count, negative, epochs, batch_size and min_n must be at least 1, "
                    "max_n at least min_n, bucket not negative, alpha positive, min_alpha and sample not negative");

    TrainingVocabulary vocabulary;
    if (!BuildTrainingVocabulary(sentences, options.min_count, options.max_vocab, vocabulary))
        return fail("Fewer than two words appear at least min_count times");
    model.words = vocabulary.words;
    model.counts = vocabulary.counts;
    model.dim = options.vector_size;
    model.min_n = options.min_n;
    model.max_n = options.max_n;
    model.bucket = options.bucket;
    if (options.on_vocabulary) options.on_vocabulary(vocabulary.words.size());
    const PieceRows host_pieces = BuildPieceRows(model.words, options.min_n, options.max_n, options.bucket);
    const dim_t v = static_cast<dim_t>(model.words.size()), d = options.vector_size;
    const dim_t n = static_cast<dim_t>(vocabulary.tokens.size());
    const int window = options.window, k = options.negative;

    try {
        // Skip-gram: a narrow table for the words of 98% of the text, the
        // others in batches of their own at full width.
        const DevicePieces pieces =
            ToDevice(host_pieces, options.cbow ? 0 : NarrowWidth(host_pieces, model.counts, 0.98));
        af::randomEngine engine(AF_RANDOM_ENGINE_DEFAULT, options.seed);
        // fastText and gensim: input rows (words and n-grams) uniform in
        // +-1 / vector_size, output rows 0.
        af::array w_in =
            (af::randu(af::dim4(v + options.bucket, d), f32, engine) * 2.0f - 1.0f) / static_cast<float>(d);
        af::array w_out = af::constant(0.0f, v, d, f32);
        const af::array corpus(n, vocabulary.tokens.data());
        const af::array corpus_sentence(n, vocabulary.sentence_of.data());
        std::vector<float> counts_float(model.counts.size());
        std::transform(model.counts.begin(), model.counts.end(), counts_float.begin(),
                       [](long long c) { return static_cast<float>(c); });
        const af::array count_f(v, counts_float.data());
        const af::array keep_probability = KeepProbability(count_f, options.sample);
        const af::array noise_table = NoiseTable(count_f);

        for (int epoch = 0; epoch < options.epochs; ++epoch) {
            const EpochExamples examples_of_epoch =
                MakeEpochExamples(corpus, corpus_sentence, keep_probability, window, options.cbow, true, engine);
            if (examples_of_epoch.outputs.isempty()) continue;
            const af::array ex_inputs =
                options.cbow ? examples_of_epoch.inputs
                             : af::moddims(examples_of_epoch.inputs, examples_of_epoch.inputs.elements(), 1);
            const af::array& ex_outputs = examples_of_epoch.outputs;
            const dim_t examples = ex_outputs.elements();
            af::array epoch_loss = af::constant(0.0f, 1, f32);
            dim_t steps = 0, processed = 0;
            // One batch; false when the run is stopped.
            const auto run = [&](const af::array& batch_in, const af::array& batch_out, const af::array* table,
                                 dim_t total) {
                const af::array negatives = DrawNegatives(noise_table, batch_out, k, engine);
                // gensim: the rate falls linearly with the share of the run done.
                const double progress = std::min(
                    1.0, (static_cast<double>(epoch) + static_cast<double>(processed) / static_cast<double>(examples)) /
                             options.epochs);
                const double alpha = options.alpha - (options.alpha - options.min_alpha) * progress;
                epoch_loss += af::sum(table ? StepPadded(w_in, w_out, *table, batch_in, batch_out, negatives, alpha)
                                            : Step(w_in, w_out, pieces, batch_in, batch_out, negatives, total, alpha));
                epoch_loss.eval();
                if (++steps % kSyncSteps == 0) af::sync();
                const dim_t size = batch_out.elements();
                processed += size;
                return !options.on_progress ||
                       options.on_progress(std::min(1.0, (static_cast<double>(epoch) + static_cast<double>(processed) /
                                                          static_cast<double>(examples)) / options.epochs),
                                           epoch + 1);
            };
            const dim_t batch = StepBatch(options.batch_size, v);
            if (!pieces.narrow.isempty()) {
                // Skip-gram: short and long words apart, the long batches
                // spread among the short ones.
                const af::array is_long = af::lookup(pieces.long_word, af::flat(ex_inputs)) > 0;
                const af::array part_idx[2] = {af::where(!is_long), af::where(is_long)};
                const af::array* tables[2] = {&pieces.narrow, &pieces.padded};
                af::array part_in[2], part_out[2];
                dim_t part_batches[2] = {0, 0}, next[2] = {0, 0};
                for (int p = 0; p < 2; ++p) {
                    const dim_t count = part_idx[p].elements();
                    if (count == 0) continue;
                    part_in[p] = ex_inputs(part_idx[p], af::span);
                    part_out[p] = ex_outputs(part_idx[p]);
                    part_batches[p] = (count + batch - 1) / batch;
                }
                ReleaseEpochScratch();
                while (next[0] < part_batches[0] || next[1] < part_batches[1]) {
                    const bool long_turn = next[0] >= part_batches[0] ||
                                           (next[1] < part_batches[1] &&
                                            next[1] * part_batches[0] < next[0] * part_batches[1]);
                    const int p = long_turn ? 1 : 0;
                    const dim_t start = next[p] * batch;
                    const dim_t end = std::min(part_out[p].elements(), start + batch);
                    const af::seq rows(static_cast<double>(start), static_cast<double>(end - 1));
                    ++next[p];
                    if (!run(part_in[p](rows, af::span), part_out[p](rows), tables[p], 0))
                        return fail("Training was stopped");
                }
            } else {
                // Rows per batch, read once per epoch: each batch's rows summed
                // on the device (pad to whole batches, one column per batch).
                const dim_t batches = (examples + batch - 1) / batch;
                af::array per_example = RowsPerExample(pieces, ex_inputs);
                if (batches * batch > examples)
                    per_example = af::join(0, per_example, af::constant(0, batches * batch - examples, s32));
                ReleaseEpochScratch();
                std::vector<int> rows_in_batch(static_cast<size_t>(batches));
                af::array batch_rows = af::sum(af::moddims(per_example, batch, batches), 0).as(s32);
                batch_rows.eval();
                MaterializeArrayFireToHost(batch_rows, rows_in_batch.data(),
                                           ArrayFireHostSyncCategory::OutputMaterialization,
                                           "WordEmbeddings::TrainFastText", "rows_per_batch");
                const af::array* full = options.cbow || pieces.padded.isempty() ? nullptr : &pieces.padded;
                for (dim_t i = 0; i < batches; ++i) {
                    const dim_t start = i * batch, end = std::min(examples, start + batch);
                    const af::seq rows(static_cast<double>(start), static_cast<double>(end - 1));
                    if (!run(ex_inputs(rows, af::span), ex_outputs(rows), full, rows_in_batch[static_cast<size_t>(i)]))
                        return fail("Training was stopped");
                }
            }
            model.examples += examples;
            model.epoch_loss.push_back(examples > 0 ? epoch_loss.scalar<float>() / static_cast<double>(examples) : 0.0);
            if (options.on_epoch_loss) options.on_epoch_loss(epoch + 1, model.epoch_loss.back(), examples);
        }
        if (model.examples == 0) return fail("The corpus gave no training examples (sentences too short?)");
        const std::vector<float> all = RowsToHost(w_in, "WordEmbeddings::TrainFastText");
        const size_t words_table = static_cast<size_t>(v) * static_cast<size_t>(d);
        model.word_rows.assign(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(words_table));
        model.ngram_rows.assign(all.begin() + static_cast<std::ptrdiff_t>(words_table), all.end());
        model.output_vectors = RowsToHost(w_out, "WordEmbeddings::TrainFastText");
        model.success = true;
    } catch (const af::exception& e) {
        return fail(std::string("WordEmbeddings::TrainFastText failed on the ArrayFire device: ") + e.what());
    }
    return model;
}

WordVectors WordEmbeddings::FastTextVectors(const FastTextModel& model, const std::vector<std::string>* words,
                                            std::vector<FastTextSource>* sources) {
    WordVectors out;
    const std::vector<std::string>& wanted = words ? *words : model.words;
    const int v = static_cast<int>(model.words.size()), dim = model.dim;
    if (dim <= 0 || model.word_rows.size() != static_cast<size_t>(v) * static_cast<size_t>(dim) ||
        model.ngram_rows.size() != static_cast<size_t>(model.bucket) * static_cast<size_t>(dim)) {
        out.error_message = "The fastText model has no rows";
        return out;
    }
    std::unordered_map<std::string, int> id_of;
    id_of.reserve(model.words.size());
    for (int w = 0; w < v; ++w) id_of.emplace(model.words[static_cast<size_t>(w)], w);

    // Each wanted word's rows among the ones taken from the model (a word row
    // or an n-gram row, each taken once).
    std::unordered_map<long long, int> taken;  // model row (word: id, n-gram: v + bucket) -> taken row
    std::vector<long long> taken_rows;
    std::vector<int> rows, group;
    if (sources) sources->assign(wanted.size(), FastTextSource::None);
    const auto take = [&](long long model_row, int g) {
        const auto [it, added] = taken.emplace(model_row, static_cast<int>(taken_rows.size()));
        if (added) taken_rows.push_back(model_row);
        rows.push_back(it->second);
        group.push_back(g);
    };
    for (size_t g = 0; g < wanted.size(); ++g) {
        const auto known = id_of.find(wanted[g]);
        const std::vector<int> buckets = FastTextBuckets(wanted[g], model.min_n, model.max_n, model.bucket);
        if (known != id_of.end()) {
            take(known->second, static_cast<int>(g));
            if (sources) (*sources)[g] = FastTextSource::Word;
        } else if (!buckets.empty() && sources) {
            (*sources)[g] = FastTextSource::Pieces;
        }
        if (known != id_of.end() || !buckets.empty())
            for (const int b : buckets) take(static_cast<long long>(v) + b, static_cast<int>(g));
    }
    // The taken rows, copied from the host tables; the means on the device.
    std::vector<float> table(taken_rows.size() * static_cast<size_t>(dim));
    for (size_t r = 0; r < taken_rows.size(); ++r) {
        const long long row = taken_rows[r];
        const float* from = row < v ? model.word_rows.data() + static_cast<size_t>(row) * dim
                                    : model.ngram_rows.data() + static_cast<size_t>(row - v) * dim;
        std::copy(from, from + dim, table.begin() + static_cast<std::ptrdiff_t>(r * static_cast<size_t>(dim)));
    }
    try {
        out.values = GroupMeans(table, static_cast<dim_t>(taken_rows.size()), dim, rows, group,
                                static_cast<dim_t>(wanted.size()), "WordEmbeddings::FastTextVectors");
    } catch (const af::exception& e) {
        out.error_message = std::string("WordEmbeddings::FastTextVectors failed on the ArrayFire device: ") + e.what();
        return out;
    }
    out.words = wanted;
    out.dim = dim;
    out.success = true;
    return out;
}

}  // namespace cyxwiz
