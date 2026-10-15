#pragma once

// Word vectors (TOFIX152): pretrained files read as gensim reads them
// (KeyedVectors.load_word2vec_format), and a table lined up with a token
// vocabulary for the Embedding layer. Files are read on the host; the matrix
// work runs on the selected ArrayFire device.

#include "cyxwiz/api_export.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

// Words and their vectors, row-major [words.size(), dim], in file order.
struct CYXWIZ_API WordVectors {
    std::vector<std::string> words;
    std::vector<float> values;
    int dim = 0;

    bool success = false;
    std::string error_message;
};

// How AlignWordVectors fills the rows of a vocabulary.
struct CYXWIZ_API WordAlignmentOptions {
    bool lowercase_fallback = true;   // a word not found as written is looked up in lowercase
    std::string missing = "zeros";    // rows without a vector: "zeros" or "random" (normal, the vectors' std)
    std::string unknown = "mean";     // the unknown-token row: "mean" of the vectors, or as a missing word
    int pad_id = 0;                   // row of zeros (-1: none)
    int unknown_id = 1;               // -1: none
    unsigned seed = 0;                // for "random"
    // fastText: vectors built from character n-grams for tokens found neither
    // as written nor in lowercase that have a letter or digit (empty: none).
    // Called once with those tokens; they count as `pieces`, not in the mean.
    std::function<WordVectors(const std::vector<std::string>& tokens)> pieces;
};

// A vocabulary's rows, row-major [vocab.size(), dim], and how they were found.
struct CYXWIZ_API WordAlignment {
    std::vector<float> values;
    int dim = 0;
    size_t exact = 0;      // found as written
    size_t lowercase = 0;  // found in lowercase
    size_t pieces = 0;     // built from character n-grams (fastText)
    size_t missing = 0;    // no vector (PAD and UNK are not counted)
    std::vector<std::string> missing_words;  // in vocabulary order
    std::vector<std::string> piece_words;    // in vocabulary order

    bool success = false;
    std::string error_message;
};

// word2vec training settings, gensim's names and defaults (Word2Vec).
struct CYXWIZ_API Word2VecOptions {
    bool cbow = false;          // false: skip-gram (sg=1); true: CBOW with the mean of the context
    int vector_size = 100;
    int window = 5;             // largest distance; each position draws its own 1..window (gensim)
    int min_count = 5;          // words seen fewer times are dropped before training
    int negative = 5;           // noise words per example, drawn from unigram^0.75
    double sample = 1e-3;       // frequent-word down-sampling threshold (0: off)
    int epochs = 5;
    double alpha = 0.025;       // learning rate, falling linearly to min_alpha
    double min_alpha = 1e-4;
    int batch_size = 512;       // examples per device step
    unsigned seed = 1;
    size_t max_vocab = 0;       // keep the most frequent words only (0: all)
    // Called after each device step with the share of training done (0..1) and
    // the epoch; returning false stops the run (the model is then refused).
    std::function<bool(double done, int epoch)> on_progress;
    // Called once the vocabulary is built, with its size.
    std::function<void(size_t words)> on_vocabulary;
    // Called after each epoch with its mean loss per example and its examples.
    std::function<void(int epoch, double loss, long long examples)> on_epoch_loss;
};

// One device step of word2vec with negative sampling: SGD on the sum of the
// examples' losses, all rows updated from the same old values.
// - skip-gram: `inputs` [B] centre words, `outputs` [B] context words;
// - CBOW: `inputs` [B * context_width] context words (row-major, -1 = none),
//   whose mean predicts `outputs` [B] centre words.
// `negatives` [B * negative] row-major.
struct CYXWIZ_API Word2VecBatch {
    std::vector<int> inputs;
    int context_width = 1;
    std::vector<int> outputs;
    std::vector<int> negatives;
    int negative = 0;
    double alpha = 0.025;
};

// A trained model: the vocabulary (most frequent first), its counts, and the
// input vectors (the word vectors) and output vectors.
struct CYXWIZ_API Word2VecModel {
    WordVectors vectors;
    std::vector<long long> counts;
    std::vector<float> output_vectors;
    std::vector<double> epoch_loss;  // mean loss per example in each epoch
    long long examples = 0;          // training examples seen in all epochs

    bool success = false;
    std::string error_message;
};

// GloVe training settings (Pennington et al. 2014), the Stanford reference's
// names and defaults (glove.c / cooccur.c).
struct CYXWIZ_API GloVeOptions {
    int vector_size = 100;
    int window = 10;               // symmetric; a pair `d` words apart counts 1/d
    int min_count = 5;
    double x_max = 100.0;          // counts above it all weigh 1
    double alpha = 0.75;           // weighting power below x_max
    double learning_rate = 0.05;   // AdaGrad
    int epochs = 25;
    int batch_size = 8192;         // non-zero pairs per device step
    unsigned seed = 1;
    size_t max_vocab = 0;          // keep the most frequent words only (0: all)
    // Called while counting (epoch 0, `done` the share of the text counted) and
    // after each device step (epoch 1.., `done` the share of training done);
    // returning false stops the run (the model is then refused).
    std::function<bool(double done, int epoch)> on_progress;
    // Called after each epoch with its mean cost per pair.
    std::function<void(int epoch, double cost)> on_epoch_cost;
    // Called once the vocabulary is built, with its size.
    std::function<void(size_t words)> on_vocabulary;
    // Called once counting ends, with the vocabulary size and the distinct pairs.
    std::function<void(size_t words, long long pairs)> on_counted;
};

// Co-occurrence counts: one entry per distinct (row, column) pair, ordered by
// row then column.
struct CYXWIZ_API GloVeCounts {
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<float> values;

    bool success = false;
    std::string error_message;
};

// The trainable state, row-major [vocab_size, dim] (biases [vocab_size]) with
// AdaGrad's sums of squared steps (glove.c starts them at 1).
struct CYXWIZ_API GloVeState {
    std::vector<float> word, context, word_bias, context_bias;
    std::vector<float> word_sq, context_sq, word_bias_sq, context_bias_sq;
};

// One device step: the pairs (rows[k], cols[k]) with counts[k].
struct CYXWIZ_API GloVeBatch {
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<float> counts;
};

// A trained GloVe model: the vocabulary (most frequent first), its counts, the
// word vectors (word + context vectors, the reference's default output).
struct CYXWIZ_API GloVeModel {
    WordVectors vectors;
    std::vector<long long> counts;
    std::vector<double> epoch_cost;  // mean cost per pair in each epoch
    long long pairs = 0;             // non-zero co-occurrences

    bool success = false;
    std::string error_message;
};

// fastText training settings (Bojanowski et al. 2017): word2vec's, with
// gensim's names and defaults (FastText), plus the character n-grams ("pieces")
// hashed into `bucket` rows.
struct CYXWIZ_API FastTextOptions {
    bool cbow = false;      // false: skip-gram (the centre word predicts each context word)
    int vector_size = 100;
    int window = 5;
    int min_count = 5;
    int negative = 5;
    double sample = 1e-3;
    int epochs = 5;
    double alpha = 0.025;
    double min_alpha = 1e-4;
    int batch_size = 512;
    unsigned seed = 1;
    size_t max_vocab = 0;
    int min_n = 3;          // n-gram lengths in characters, of "<word>"
    int max_n = 6;
    int bucket = 2000000;   // n-gram rows (0: no n-grams, plain word2vec rows)
    std::function<bool(double done, int epoch)> on_progress;
    std::function<void(size_t words)> on_vocabulary;
    std::function<void(int epoch, double loss, long long examples)> on_epoch_loss;
};

// A fastText model: the vocabulary (most frequent first), its counts, the
// input rows of the words and of the n-gram buckets, and the output rows.
// A word's vector is the mean of its own row and its n-grams' rows; a word
// outside the vocabulary gets the mean of its n-grams' rows (fastText, gensim).
struct CYXWIZ_API FastTextModel {
    std::vector<std::string> words;
    std::vector<long long> counts;
    int dim = 0;
    int min_n = 3;
    int max_n = 6;
    int bucket = 0;
    std::vector<float> word_rows;       // [words, dim]
    std::vector<float> ngram_rows;      // [bucket, dim]
    std::vector<float> output_vectors;  // [words, dim]
    std::vector<double> epoch_loss;     // mean loss per example in each epoch
    long long examples = 0;

    bool success = false;
    std::string error_message;
};

// How a word got its fastText vector.
enum class FastTextSource : int { Word = 0, Pieces = 1, None = 2 };

// What a fastText .bin holds (fastText's own format, as fastText and gensim's
// save_facebook_model write it).
struct CYXWIZ_API FastTextFileInfo {
    int version = 0;
    int dim = 0;
    int min_n = 0;
    int max_n = 0;
    int bucket = 0;
    std::string model;               // "skip-gram", "cbow"
    std::vector<std::string> words;  // in file order
    long long tokens = 0;            // words of the training text
    unsigned long long file_bytes = 0;

    bool success = false;
    std::string error_message;
};

class CYXWIZ_API WordEmbeddings {
public:
    // Text format: one word per line followed by its values, separated by
    // spaces. A first line of two integers ("count dim", word2vec / fastText
    // .vec) is a header; without it (GloVe .txt) the width comes from the first
    // row. A repeated word keeps its first vector. `max_words` > 0 reads at most
    // that many words.
    static WordVectors LoadText(const std::string& path, size_t max_words = 0);
    // word2vec binary: "count dim\n", then per word the word, a space and dim
    // little-endian float32 values.
    static WordVectors LoadBinary(const std::string& path, size_t max_words = 0);
    // .bin is a fastText model (LoadFastTextBin) when it starts with
    // fastText's magic number, word2vec binary otherwise; anything else is text.
    static WordVectors Load(const std::string& path, size_t max_words = 0);

    // True when the file starts with fastText's magic number (793712314).
    static bool IsFastTextBin(const std::string& path);
    // The header and words of a fastText .bin. Quantized (.ftz) and
    // supervised (label) models are refused.
    static FastTextFileInfo ReadFastTextInfo(const std::string& path);
    // Vectors of `words` from a fastText .bin (its first `max_words` words, or
    // all, when null), reading only the rows they need: a word of the model
    // from its row and its n-grams, another word from its n-grams (zeros
    // without any). `sources` gets how each was made; `info` the file's facts.
    static WordVectors LoadFastTextBin(const std::string& path, const std::vector<std::string>* words = nullptr,
                                       std::vector<FastTextSource>* sources = nullptr,
                                       FastTextFileInfo* info = nullptr, size_t max_words = 0);
    // The vectors of the model words a vocabulary takes: each token as
    // written, else in lowercase (`lowercase_fallback`), for Align; with
    // options.pieces = LoadFastTextBin of the rest, a vocabulary lines up
    // without reading the whole model.
    static WordVectors LoadFastTextBinFor(const std::string& path, const std::vector<std::string>& vocabulary,
                                          bool lowercase_fallback, FastTextFileInfo* info = nullptr);

    // The vectors in the order of `vocabulary` (token id = position); with
    // options.pieces, tokens with a letter or digit that are not found take
    // vectors from it.
    static WordAlignment Align(const WordVectors& vectors, const std::vector<std::string>& vocabulary,
                               const WordAlignmentOptions& options = {});

    // Writes word2vec text (.vec): "count dim", then one word and its values per
    // line (gensim's save_word2vec_format(binary=False)); float32 values written
    // so they read back exactly.
    static bool SaveText(const WordVectors& vectors, const std::string& path, std::string* error = nullptr);

    // The `k` words most similar to `word` by cosine similarity, most similar
    // first (the word itself left out), on the device. Empty when the word has
    // no vector, or when the device fails (then `error` says why).
    static std::vector<std::pair<std::string, double>> Nearest(const WordVectors& vectors, const std::string& word,
                                                                int k = 10, std::string* error = nullptr);
    // The `k` words most similar to `vector` (a word the vectors lack, such as
    // a fastText word built from its n-grams), most similar first.
    static std::vector<std::pair<std::string, double>> NearestTo(const WordVectors& vectors,
                                                                  const std::vector<float>& vector, int k = 10,
                                                                  std::string* error = nullptr);

    // One vector per text row: the mean of its tokens' rows of `table`
    // (row-major [table_rows, dim]); `tokens` holds row-major token ids per text
    // row (-1 = none), `tokens_per_row` per row. A row without a token is zeros.
    static std::vector<float> RowMeans(const std::vector<float>& table, int dim, const std::vector<int>& tokens,
                                       int tokens_per_row, std::string* error = nullptr);

    // Runs `batches` in order on row-major [vocab_size, dim] input / output
    // vectors (updated in place); `losses` gets each batch's summed loss.
    static bool Word2VecSteps(std::vector<float>& input_vectors, std::vector<float>& output_vectors, int vocab_size,
                              int dim, const std::vector<Word2VecBatch>& batches, std::vector<double>* losses,
                              std::string* error = nullptr);

    // Trains word2vec on tokenised sentences on the device: vocabulary from
    // min_count, down-sampling, dynamic windows and negatives drawn on the
    // device from `seed`, so a run repeats exactly on one device.
    static Word2VecModel TrainWord2Vec(const std::vector<std::vector<std::string>>& sentences,
                                       const Word2VecOptions& options = {});

    // GloVe co-occurrences of `tokens` (word ids) within `window`, on the
    // device: a pair `d` positions apart in the same sentence adds 1/d to both
    // (i, j) and (j, i).
    static GloVeCounts CountCooccurrences(const std::vector<int>& tokens, const std::vector<int>& sentence_of,
                                          int vocab_size, int window, std::string* error = nullptr);
    // Runs `batches` in order on `state` (updated in place): the weighted least
    // squares cost and an AdaGrad step per row, the gradients of a row summed
    // over the batch, every row read before the step. `costs` gets each
    // batch's summed cost.
    static bool GloVeSteps(GloVeState& state, int vocab_size, int dim, const std::vector<GloVeBatch>& batches,
                           double x_max, double alpha, double learning_rate, std::vector<double>* costs,
                           std::string* error = nullptr);
    // Trains GloVe on tokenised sentences on the device: vocabulary from
    // min_count, counts, then epochs over the shuffled pairs; a run repeats
    // exactly for one seed on one device.
    static GloVeModel TrainGloVe(const std::vector<std::vector<std::string>>& sentences,
                                 const GloVeOptions& options = {});

    // fastText's hash of UTF-8 bytes (FNV-1a, each byte taken as a signed char,
    // as fastText's dictionary.cc and gensim's ft_hash_bytes).
    static uint32_t FastTextHash(const std::string& bytes);
    // The character n-grams of "<word>", min_n..max_n characters (UTF-8 aware),
    // in fastText's order.
    static std::vector<std::string> FastTextNgrams(const std::string& word, int min_n, int max_n);
    // The n-grams' bucket rows (hash % bucket); empty when bucket is 0.
    static std::vector<int> FastTextBuckets(const std::string& word, int min_n, int max_n, int bucket);
    // Runs `batches` in order on `model` (word, n-gram and output rows updated
    // in place); a batch input is a vocabulary word whose input vector is the
    // mean of its row and its n-grams' rows (CBOW: of all its input words'
    // rows), and each of those rows takes its share of the gradient of that
    // mean. `losses` gets each batch's summed loss.
    static bool FastTextSteps(FastTextModel& model, const std::vector<Word2VecBatch>& batches,
                              std::vector<double>* losses, std::string* error = nullptr);
    // Trains fastText on tokenised sentences on the device, as word2vec
    // (vocabulary, down-sampling, windows, negatives) with n-gram rows; a run
    // repeats exactly for one seed on one device.
    static FastTextModel TrainFastText(const std::vector<std::vector<std::string>>& sentences,
                                       const FastTextOptions& options = {});
    // Vectors of `words` (the model's own words when null), on the device: a
    // vocabulary word from its row and its n-grams, another word from its
    // n-grams (zeros when it has none, or the model has no buckets). `sources`
    // gets how each was made.
    static WordVectors FastTextVectors(const FastTextModel& model, const std::vector<std::string>* words = nullptr,
                                       std::vector<FastTextSource>* sources = nullptr);
};

}  // namespace cyxwiz
