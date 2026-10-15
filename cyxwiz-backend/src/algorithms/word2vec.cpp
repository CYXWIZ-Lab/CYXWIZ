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
#include "word_training_vocabulary.h"
#include "word_vector_device.h"
#include "word2vec_sampling.h"

#include <arrayfire.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

namespace {

constexpr dim_t kSyncSteps = 64;

// One step. `inputs` [B, C] (-1 = no word), `outputs` [B], `negatives` [B, K]
// (-1 = none), all s32. Returns the summed loss (a device scalar).
af::array Step(af::array& w_in, af::array& w_out, const af::array& inputs, const af::array& outputs,
               const af::array& negatives, double alpha) {
    const dim_t b = inputs.dims(0), c = inputs.dims(1), d = w_in.dims(1);

    // h: the input word (skip-gram) or the mean of the context words (CBOW).
    const af::array valid = inputs >= 0;
    const af::array weight = valid.as(f32);
    const af::array count = af::max(af::sum(weight, 1), 1.0f);  // [B]
    const af::array rows_in = af::lookup(w_in, af::flat(af::select(valid, inputs, 0.0)), 0);  // [B*C, d]
    const af::array masked = af::moddims(rows_in, b, c, d) * af::tile(af::moddims(weight, b, c, 1), 1, 1, static_cast<unsigned>(d));
    const af::array h = af::moddims(af::sum(masked, 1), b, d) / af::tile(count, 1, static_cast<unsigned>(d));

    // The outputs and negatives, then the input side from dLoss / dh.
    af::array grad_h;
    const af::array loss = NegativeSamplingStep(w_out, h, outputs, negatives, alpha, grad_h);
    // Each context word gets its share of the mean's gradient; an empty slot
    // gets none.
    af::array grad_in = af::moddims(
        af::tile(af::moddims(grad_h / af::tile(count, 1, static_cast<unsigned>(d)), b, 1, d), 1, static_cast<unsigned>(c), 1),
        b * c, d) * af::tile(af::flat(weight), 1, static_cast<unsigned>(d));
    grad_in.eval();
    // Empty slots (-1) update row 0 by a zero gradient.
    ScatterSubtract(w_in, af::flat(af::select(valid, inputs, 0.0)).as(s32), grad_in, alpha);
    w_in.eval();
    w_out.eval();
    return loss;
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

    TrainingVocabulary vocabulary;
    if (!BuildTrainingVocabulary(sentences, options.min_count, options.max_vocab, vocabulary))
        return fail("Fewer than two words appear at least min_count times");
    model.vectors.words = vocabulary.words;
    model.counts = vocabulary.counts;
    if (options.on_vocabulary) options.on_vocabulary(vocabulary.words.size());
    const std::vector<int>& tokens = vocabulary.tokens;
    const std::vector<int>& sentence_of = vocabulary.sentence_of;
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

        const af::array keep_probability = KeepProbability(count_f, options.sample);
        const af::array noise_table = NoiseTable(count_f);

        const double total_estimate = static_cast<double>(options.epochs) * static_cast<double>(n) * 2.0 * window;
        double done = 0.0;
        for (int epoch = 0; epoch < options.epochs; ++epoch) {
            const EpochExamples epoch_examples =
                MakeEpochExamples(corpus, corpus_sentence, keep_probability, window, options.cbow, false, engine);
            if (epoch_examples.outputs.isempty()) continue;
            const af::array& ex_inputs = epoch_examples.inputs;
            const af::array& ex_outputs = epoch_examples.outputs;
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
                const af::array negatives = DrawNegatives(noise_table, batch_out, k, engine);
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
            if (options.on_epoch_loss) options.on_epoch_loss(epoch + 1, model.epoch_loss.back(), examples);
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
