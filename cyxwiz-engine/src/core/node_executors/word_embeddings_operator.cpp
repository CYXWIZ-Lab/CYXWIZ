#include "word_embeddings_operator.h"

#include "text_column_utils.h"
#include "../fasttext_model_store.h"
#include "../sha256_digest.h"
#include "../word_vectors_training_record.h"
#include "../word_embeddings_presentation.h"
#include "tree_classification_utils.h"

#include <cyxwiz/text_processing.h>
#include <cyxwiz/tokenizer.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>

namespace cyxwiz {

namespace {

// Values in one RowMeans block ([rows, longest row, dim] gathered on the
// device): 64M floats, 256 MB.
constexpr size_t kRowMeansElements = size_t{1} << 26;
constexpr const char* kLossPerEpoch = "loss per epoch ";
constexpr const char* kCostPerEpoch = "cost per epoch ";

bool ParseBoolParam(const std::map<std::string, std::string>& params, const std::string& key, bool& out,
                    const std::string& op_name, std::string& error) {
    const auto it = params.find(key);
    if (it == params.end()) return true;
    const std::string value = TrimAscii(it->second);
    if (value.empty()) return true;
    if (value == "true" || value == "1") out = true;
    else if (value == "false" || value == "0") out = false;
    else {
        error = op_name + ": '" + key + "' must be true or false";
        return false;
    }
    return true;
}

bool ParseChoice(const std::map<std::string, std::string>& params, const std::string& key,
                 const std::vector<std::string>& choices, std::string& out, const std::string& op_name,
                 std::string& error) {
    const auto it = params.find(key);
    if (it == params.end() || TrimAscii(it->second).empty()) return true;
    const std::string value = TrimAscii(it->second);
    if (std::find(choices.begin(), choices.end(), value) == choices.end()) {
        std::string list;
        for (const auto& c : choices) list += (list.empty() ? "" : ", ") + c;
        error = op_name + ": '" + key + "' must be one of " + list;
        return false;
    }
    out = value;
    return true;
}

std::string Param(const std::map<std::string, std::string>& params, const std::string& key) {
    const auto it = params.find(key);
    return it == params.end() ? std::string() : TrimAscii(it->second);
}

std::string LocalTimeNow() {
    const std::time_t now = std::time(nullptr);
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    char text[32];
    std::strftime(text, sizeof(text), "%Y-%m-%d %H:%M", &local);
    return text;
}

}  // namespace

std::vector<std::string> WordEmbeddingsOperator::SplitWords(const std::string& text) {
    // Tokenizer::Split, Word mode: lowercase, then runs of letters and digits.
    return TextProcessing::Tokenize(TextProcessing::ToLowercase(text), "word", 2, false, true).tokens;
}

std::string WordEmbeddingsOperator::DefaultVectorsFile(int node_id, const std::string& node_name) {
    std::string slug = node_name;
    std::transform(slug.begin(), slug.end(), slug.begin(),
                   [](unsigned char c) { return std::isalnum(c) ? static_cast<char>(std::tolower(c)) : '_'; });
    while (!slug.empty() && slug.back() == '_') slug.pop_back();
    return "word_embeddings/" + std::to_string(node_id) + "_" + (slug.empty() ? "vectors" : slug) + ".vec";
}

std::string WordEmbeddingsOperator::TrainingSettings() const {
    if (source_ == "pretrained") return {};
    std::string settings;
    const auto add = [&](const char* key, const auto& value) {
        std::ostringstream text;
        text.precision(17);
        text << value;
        settings += std::string(key) + '=' + text.str() + '\n';
    };
    add("version", kWordVectorsTrainingVersion);
    add("source", source_);
    add("text_col", text_col_);
    if (source_ == "glove") {
        add("vector_size", glove_.vector_size);
        add("min_count", glove_.min_count);
        add("seed", glove_.seed);
        add("window", glove_.window);
        add("epochs", glove_.epochs);
        add("batch_size", glove_.batch_size);
        add("x_max", glove_.x_max);
        add("weighting_power", glove_.alpha);
        add("learning_rate", glove_.learning_rate);
    } else {
        add("cbow", word2vec_.cbow);
        add("vector_size", word2vec_.vector_size);
        add("window", word2vec_.window);
        add("min_count", word2vec_.min_count);
        add("negative", word2vec_.negative);
        add("sample", word2vec_.sample);
        add("epochs", word2vec_.epochs);
        add("alpha", word2vec_.alpha);
        add("min_alpha", word2vec_.min_alpha);
        add("batch_size", word2vec_.batch_size);
        add("seed", word2vec_.seed);
        if (source_ == "fasttext") {
            add("min_n", fasttext_.min_n);
            add("max_n", fasttext_.max_n);
            add("bucket", fasttext_.bucket);
            add("unseen_from_pieces", unseen_from_pieces_);
        }
    }
    if (!vocab_file_.empty()) {
        add("lowercase_fallback", alignment_.lowercase_fallback);
        add("missing", alignment_.missing);
        add("unknown", alignment_.unknown);
        add("missing_seed", alignment_.seed);
    }
    return settings;
}

bool WordEmbeddingsOperator::Configure(const std::map<std::string, std::string>& params, std::string& error) {
    source_ = "pretrained";
    max_words_ = 0;
    alignment_ = WordAlignmentOptions{};
    word2vec_ = Word2VecOptions{};
    glove_ = GloVeOptions{};
    row_vectors_ = false;
    const std::string op = GetName();
    if (!ParseChoice(params, "source", {"pretrained", "word2vec", "glove", "fasttext"}, source_, op, error))
        return false;
    vectors_file_ = Param(params, "vectors_file");
    vocab_file_ = Param(params, "vocab_file");
    text_col_ = Param(params, "text_col");
    vectors_path_ = Param(params, "vectors_path");

    int max_words = 0;
    if (!ParseIntParam(params, "max_words", max_words, op, error)) return false;
    if (max_words < 0) {
        error = op + ": max_words must be >= 0";
        return false;
    }
    max_words_ = static_cast<size_t>(max_words);
    if (!ParseBoolParam(params, "lowercase_fallback", alignment_.lowercase_fallback, op, error)) return false;
    if (!ParseChoice(params, "missing", {"zeros", "random"}, alignment_.missing, op, error)) return false;
    if (!ParseChoice(params, "unknown", {"mean", "missing"}, alignment_.unknown, op, error)) return false;
    int seed = 0;
    if (!ParseIntParam(params, "seed", seed, op, error)) return false;
    if (seed < 0) {
        error = op + ": seed must be >= 0";
        return false;
    }
    alignment_.seed = static_cast<unsigned>(seed);

    std::string architecture = "skipgram";
    if (!ParseChoice(params, "architecture", {"skipgram", "cbow"}, architecture, op, error)) return false;
    word2vec_.cbow = architecture == "cbow";
    if (!ParseIntParam(params, "vector_size", word2vec_.vector_size, op, error) ||
        !ParseIntParam(params, "window", word2vec_.window, op, error) ||
        !ParseIntParam(params, "min_count", word2vec_.min_count, op, error) ||
        !ParseIntParam(params, "negative", word2vec_.negative, op, error) ||
        !ParseIntParam(params, "epochs", word2vec_.epochs, op, error) ||
        !ParseIntParam(params, "batch_size", word2vec_.batch_size, op, error) ||
        !ParseDoubleParam(params, "sample", word2vec_.sample, op, error) ||
        !ParseDoubleParam(params, "alpha", word2vec_.alpha, op, error) ||
        !ParseDoubleParam(params, "min_alpha", word2vec_.min_alpha, op, error))
        return false;
    int train_seed = 1;
    if (!ParseIntParam(params, "train_seed", train_seed, op, error)) return false;
    if (train_seed < 0) {
        error = op + ": train_seed must be >= 0";
        return false;
    }
    word2vec_.seed = static_cast<unsigned>(train_seed);
    // GloVe: the shared vector size, minimum count and seed; its own window,
    // epochs, weighting, learning rate and batch (the Stanford defaults).
    glove_.vector_size = word2vec_.vector_size;
    glove_.min_count = word2vec_.min_count;
    glove_.seed = word2vec_.seed;
    if (!ParseIntParam(params, "glove_window", glove_.window, op, error) ||
        !ParseIntParam(params, "glove_epochs", glove_.epochs, op, error) ||
        !ParseIntParam(params, "glove_batch", glove_.batch_size, op, error) ||
        !ParseDoubleParam(params, "x_max", glove_.x_max, op, error) ||
        !ParseDoubleParam(params, "weighting_power", glove_.alpha, op, error) ||
        !ParseDoubleParam(params, "glove_learning_rate", glove_.learning_rate, op, error))
        return false;
    // fastText: word2vec's settings, plus the n-gram lengths and buckets.
    fasttext_.cbow = word2vec_.cbow;
    fasttext_.vector_size = word2vec_.vector_size;
    fasttext_.window = word2vec_.window;
    fasttext_.min_count = word2vec_.min_count;
    fasttext_.negative = word2vec_.negative;
    fasttext_.epochs = word2vec_.epochs;
    fasttext_.batch_size = word2vec_.batch_size;
    fasttext_.sample = word2vec_.sample;
    fasttext_.alpha = word2vec_.alpha;
    fasttext_.min_alpha = word2vec_.min_alpha;
    fasttext_.seed = word2vec_.seed;
    if (!ParseIntParam(params, "min_n", fasttext_.min_n, op, error) ||
        !ParseIntParam(params, "max_n", fasttext_.max_n, op, error) ||
        !ParseIntParam(params, "bucket", fasttext_.bucket, op, error))
        return false;
    std::string unseen = "pieces";
    if (!ParseChoice(params, "unseen_words", {"pieces", "missing"}, unseen, op, error)) return false;
    unseen_from_pieces_ = unseen == "pieces";
    if (!ParseBoolParam(params, "row_vectors", row_vectors_, op, error)) return false;

    if (source_ != "pretrained") {
        if (text_col_.empty()) {
            error = op + ": training needs the text column ('text_col')";
            return false;
        }
        if (word2vec_.vector_size < 1 || word2vec_.min_count < 1) {
            error = op + ": vector_size and min_count must be at least 1";
            return false;
        }
        if ((source_ == "word2vec" || source_ == "fasttext") &&
            (word2vec_.window < 1 || word2vec_.negative < 1 || word2vec_.epochs < 1 || word2vec_.batch_size < 1 ||
             !(word2vec_.alpha > 0.0) || word2vec_.min_alpha < 0.0 || word2vec_.sample < 0.0)) {
            error = op + ": window, negative, epochs and batch_size must be at least 1, alpha positive, min_alpha "
                         "and sample not negative";
            return false;
        }
        if (source_ == "glove" && (glove_.window < 1 || glove_.epochs < 1 || glove_.batch_size < 1 ||
                                   !(glove_.x_max > 0.0) || glove_.alpha < 0.0 || !(glove_.learning_rate > 0.0))) {
            error = op + ": glove_window, glove_epochs and glove_batch must be at least 1, x_max and "
                         "glove_learning_rate positive, weighting_power not negative";
            return false;
        }
        if (source_ == "fasttext" &&
            (fasttext_.min_n < 1 || fasttext_.max_n < fasttext_.min_n || fasttext_.bucket < 0)) {
            error = op + ": min_n must be at least 1, max_n at least min_n and bucket not negative";
            return false;
        }
    } else if (vectors_file_.empty()) {
        error = op + ": choose a vector file ('vectors_file': .txt, .vec or .bin) or train word2vec, GloVe or "
                     "fastText";
        return false;
    }
    if (row_vectors_ && text_col_.empty()) {
        error = op + ": row vectors need the text column ('text_col')";
        return false;
    }
    return true;
}

bool WordEmbeddingsOperator::CollectCacheDependencies(std::vector<PipelineOperatorCacheDependency>& dependencies,
                                                      std::string& error) const {
    (void)error;
    if (source_ == "pretrained") dependencies.push_back({"word_vectors", vectors_file_});
    if (!vocab_file_.empty()) dependencies.push_back({"vocabulary", vocab_file_});
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>> WordEmbeddingsOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    const std::string op = GetName();
    if (!input) return arrow::Status::Invalid(op + ": input table is null");
    rows_.reset();
    outcome_.clear();
    const auto report = [&](const std::string& stage, const std::string& message, double progress) {
        if (!progress_callback_) return;
        PipelineOperatorProgress event;
        event.stage = stage;
        event.message = message;
        event.status = "running";
        event.progress = static_cast<float>(progress);
        progress_callback_(event);
    };
    // A training run's figures, for the Training Dashboard (OperatorFigures):
    // its settings at the start, the counts, each epoch's cost or loss, the
    // line-up and the end; every event carries the seconds since the start.
    const bool trained = source_ != "pretrained";
    const auto started = std::chrono::steady_clock::now();
    const auto publish = [&](const std::string& stage, const std::string& message, double progress,
                             std::vector<std::pair<std::string, double>> values,
                             std::vector<std::pair<std::string, std::string>> notes = {}) {
        if (!progress_callback_ || !trained) return;
        PipelineOperatorProgress event;
        event.stage = stage;
        event.message = message;
        event.status = "running";
        event.progress = static_cast<float>(progress);
        values.emplace_back("elapsed",
                            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
        event.figures = std::move(values);
        event.notes = std::move(notes);
        progress_callback_(event);
    };
    const auto publish_start = [&](const std::string& stage, const std::string& message, const std::string& method,
                                   int dims, int window, int epochs, int min_count) {
        publish(stage, message, 0.05,
                {{"dims", dims}, {"window", window}, {"epochs", epochs}, {"min_count", min_count}},
                {{"method", method}, {"saved", vectors_path_}, {"vocabulary_file", vocab_file_}});
    };

    // The text, and its fingerprint when the trained vectors are saved.
    std::vector<std::string> texts;
    std::string text_sha256;
    if (!text_col_.empty()) {
        const auto column = input->GetColumnByName(text_col_);
        if (!column) return arrow::Status::KeyError(op + ": text column '" + text_col_ + "' not found");
        std::string bad_type;
        bool cancelled = false;
        if (!ReadColumnAsStrings(column, texts, bad_type, GetCancellationQuery(), &cancelled)) {
            if (cancelled) return arrow::Status::Cancelled(op + ": cancelled while reading the text");
            return arrow::Status::TypeError(op + ": text column '" + text_col_ + "' must be text (got '" + bad_type + "')");
        }
        if (trained && !vectors_path_.empty()) {
            Sha256Hasher hasher;
            std::string error;
            for (const auto& text : texts) {
                if (!hasher.Update(text, error) || !hasher.Update(std::string_view("\0", 1), error)) break;
            }
            if (error.empty()) hasher.Finish(text_sha256, error);
            if (!error.empty()) return arrow::Status::IOError(op + ": cannot fingerprint the text: " + error);
        }
    }
    ARROW_RETURN_NOT_OK(CheckCancellation(op));

    Vocabulary vocabulary;
    if (!vocab_file_.empty() && !vocabulary.LoadFromFile(vocab_file_))
        return arrow::Status::IOError(op + ": cannot read the vocabulary " + vocab_file_);

    WordVectors vectors;
    std::string source_summary;
    // fastText: where the vectors of tokens the vectors lack come from (the
    // trained model, or the .bin read again for those tokens).
    std::shared_ptr<const FastTextModel> fasttext_model;
    std::string fasttext_file;
    const auto figures = [](const std::vector<double>& values) {
        std::string text;
        for (const double value : values) {
            char number[32];
            std::snprintf(number, sizeof(number), "%.4g", value);
            text += (text.empty() ? "" : ", ") + std::string(number);
        }
        return text;
    };
    // Vectors trained before from the same text, settings and vocabulary: the
    // saved file is read instead of training again.
    WordVectorsTrainingRecord record;
    std::optional<WordVectorsTrainingRecord> reused;
    if (trained && !vectors_path_.empty()) {
        record.settings = TrainingSettings();
        record.text_sha256 = text_sha256;
        std::string error;
        if (!vocab_file_.empty() && !Sha256File(vocab_file_, record.vocab_sha256, error))
            return arrow::Status::IOError(op + ": cannot read the vocabulary " + vocab_file_ + ": " + error);
        const auto saved = ReadWordVectorsTrainingRecord(vectors_path_);
        if (saved && saved->settings == record.settings && saved->text_sha256 == record.text_sha256 &&
            saved->vocab_sha256 == record.vocab_sha256) {
            report("Reading", "Reading the saved vectors", 0.05);
            vectors = WordEmbeddings::Load(vectors_path_);
            if (vectors.success) reused = saved;
            else
                spdlog::warn("WordEmbeddings: training again, the saved vectors are unreadable: {}",
                             vectors.error_message);
        }
    }
    // The text split into words, when training or averaging rows needs it.
    std::vector<std::vector<std::string>> sentences;
    if ((trained && !reused) || row_vectors_) {
        sentences.reserve(texts.size());
        for (const auto& text : texts) sentences.push_back(SplitWords(text));
    }
    texts = {};
    if (reused) {
        source_summary = std::string(kReusedVectorsOutcome) + " (text and settings unchanged since " +
                         reused->trained_at + "); " + reused->summary;
    } else if (source_ == "word2vec") {
        publish_start("Training", "Training word2vec", word2vec_.cbow ? "word2vec CBOW" : "word2vec skip-gram",
                      word2vec_.vector_size, word2vec_.window, word2vec_.epochs, word2vec_.min_count);
        Word2VecOptions options = word2vec_;
        options.on_vocabulary = [&](size_t words) {
            publish("Training", "Training word2vec", 0.05, {{"words", static_cast<double>(words)}});
        };
        options.on_epoch_loss = [&](int epoch, double loss, long long examples) {
            publish("Training", "Training word2vec, epoch " + std::to_string(epoch),
                    0.05 + 0.85 * epoch / std::max(1, word2vec_.epochs),
                    {{"epoch", epoch}, {"loss", loss}, {"examples", static_cast<double>(examples)}});
        };
        options.on_progress = [&](double done, int epoch) {
            report("Training", "Training word2vec, epoch " + std::to_string(epoch), 0.05 + 0.85 * done);
            return !IsCancellationRequested();
        };
        const auto model = WordEmbeddings::TrainWord2Vec(sentences, options);
        if (!model.success) {
            if (IsCancellationRequested()) return arrow::Status::Cancelled(op + ": training cancelled");
            return arrow::Status::Invalid(op + ": " + model.error_message);
        }
        vectors = model.vectors;
        source_summary = "trained " + std::to_string(vectors.words.size()) + " words x " +
                         std::to_string(vectors.dim) + " (" + (word2vec_.cbow ? "CBOW" : "skip-gram") + ", " +
                         std::to_string(word2vec_.epochs) + " epochs, " + kLossPerEpoch +
                         figures(model.epoch_loss) + ")";
    } else if (source_ == "glove") {
        // Two stages: counting the word pairs (to 15%), then the epochs.
        publish_start("Counting", "Counting word pairs", "GloVe", glove_.vector_size, glove_.window, glove_.epochs,
                      glove_.min_count);
        GloVeOptions options = glove_;
        std::string last_cost;
        options.on_vocabulary = [&](size_t words) {
            publish("Counting", "Counting word pairs", 0.05, {{"words", static_cast<double>(words)}});
        };
        options.on_counted = [&](size_t words, long long pairs) {
            publish("Counting", "Counted word pairs", 0.15,
                    {{"words", static_cast<double>(words)}, {"pairs", static_cast<double>(pairs)}});
        };
        options.on_epoch_cost = [&](int epoch, double cost) {
            char number[32];
            std::snprintf(number, sizeof(number), "%.4g", cost);
            last_cost = number;
            publish("Training",
                    "Training GloVe, epoch " + std::to_string(epoch) + " of " + std::to_string(glove_.epochs) +
                        ", cost " + last_cost,
                    0.15 + 0.75 * epoch / std::max(1, glove_.epochs), {{"epoch", epoch}, {"cost", cost}});
        };
        options.on_progress = [&](double done, int epoch) {
            if (epoch == 0) {
                report("Counting", "Counting word pairs", 0.05 + 0.10 * done);
            } else {
                report("Training", "Training GloVe, epoch " + std::to_string(epoch) + " of " +
                                       std::to_string(glove_.epochs) +
                                       (last_cost.empty() ? std::string() : ", cost " + last_cost),
                       0.15 + 0.75 * done);
            }
            return !IsCancellationRequested();
        };
        const auto model = WordEmbeddings::TrainGloVe(sentences, options);
        if (!model.success) {
            if (IsCancellationRequested()) return arrow::Status::Cancelled(op + ": training cancelled");
            return arrow::Status::Invalid(op + ": " + model.error_message);
        }
        vectors = model.vectors;
        source_summary = "trained " + std::to_string(vectors.words.size()) + " words x " +
                         std::to_string(vectors.dim) + " (GloVe, " +
                         GroupedCount(static_cast<size_t>(model.pairs)) + " word pairs, " +
                         std::to_string(glove_.epochs) + " epochs, " + kCostPerEpoch + figures(model.epoch_cost) +
                         ")";
    } else if (source_ == "fasttext") {
        const std::string method = fasttext_.cbow ? "fastText CBOW" : "fastText skip-gram";
        publish("Training", "Training fastText", 0.05,
                {{"dims", fasttext_.vector_size}, {"window", fasttext_.window}, {"epochs", fasttext_.epochs},
                 {"min_count", fasttext_.min_count}, {"bucket", fasttext_.bucket}, {"min_n", fasttext_.min_n},
                 {"max_n", fasttext_.max_n}},
                {{"method", method}, {"saved", vectors_path_}, {"vocabulary_file", vocab_file_}});
        FastTextOptions options = fasttext_;
        options.on_vocabulary = [&](size_t words) {
            publish("Training", "Training fastText", 0.05, {{"words", static_cast<double>(words)}});
        };
        options.on_epoch_loss = [&](int epoch, double loss, long long examples) {
            publish("Training", "Training fastText, epoch " + std::to_string(epoch),
                    0.05 + 0.85 * epoch / std::max(1, fasttext_.epochs),
                    {{"epoch", epoch}, {"loss", loss}, {"examples", static_cast<double>(examples)}});
        };
        options.on_progress = [&](double done, int epoch) {
            report("Training", "Training fastText, epoch " + std::to_string(epoch), 0.05 + 0.85 * done);
            return !IsCancellationRequested();
        };
        auto model = std::make_shared<FastTextModel>(WordEmbeddings::TrainFastText(sentences, options));
        if (!model->success) {
            if (IsCancellationRequested()) return arrow::Status::Cancelled(op + ": training cancelled");
            return arrow::Status::Invalid(op + ": " + model->error_message);
        }
        vectors = WordEmbeddings::FastTextVectors(*model);
        if (!vectors.success) return arrow::Status::Invalid(op + ": " + vectors.error_message);
        size_t pieces = 0;
        for (const auto& word : model->words)
            pieces += WordEmbeddings::FastTextBuckets(word, model->min_n, model->max_n, model->bucket).size();
        publish("Training", "Trained fastText", 0.9,
                {{"pieces_per_word", static_cast<double>(pieces) / static_cast<double>(model->words.size())}});
        source_summary = "trained " + std::to_string(vectors.words.size()) + " words x " +
                         std::to_string(vectors.dim) + " (fastText " + (fasttext_.cbow ? "CBOW" : "skip-gram") +
                         ", n-grams " + std::to_string(fasttext_.min_n) + "-" + std::to_string(fasttext_.max_n) +
                         ", " + GroupedCount(static_cast<size_t>(fasttext_.bucket)) + " buckets, " +
                         std::to_string(fasttext_.epochs) + " epochs, " + kLossPerEpoch +
                         figures(model->epoch_loss) + ")";
        fasttext_model = std::move(model);
    } else if (WordEmbeddings::IsFastTextBin(vectors_file_)) {
        // A fastText model: with a vocabulary only the rows it needs.
        report("Reading", "Reading " + vectors_file_, 0.05);
        FastTextFileInfo info;
        vectors = vocab_file_.empty()
                      ? WordEmbeddings::LoadFastTextBin(vectors_file_, nullptr, nullptr, &info, max_words_)
                      : WordEmbeddings::LoadFastTextBinFor(vectors_file_, vocabulary.GetWords(),
                                                           alignment_.lowercase_fallback, &info);
        if (!vectors.success) return arrow::Status::IOError(op + ": " + vectors.error_message);
        source_summary = "read " + std::to_string(vectors.words.size()) + " of " +
                         std::to_string(info.words.size()) + " words x " + std::to_string(vectors.dim) +
                         " (fastText model, " + GroupedCount(static_cast<size_t>(info.bucket)) + " buckets)";
        fasttext_file = vectors_file_;
    } else {
        report("Reading", "Reading " + vectors_file_, 0.05);
        vectors = WordEmbeddings::Load(vectors_file_, max_words_);
        if (!vectors.success) return arrow::Status::IOError(op + ": " + vectors.error_message);
        source_summary = "read " + std::to_string(vectors.words.size()) + " words x " + std::to_string(vectors.dim);
    }
    ARROW_RETURN_NOT_OK(CheckCancellation(op));

    // Line the vectors up with the vocabulary when one is given.
    std::string coverage = reused ? reused->coverage : std::string();
    if (!vocab_file_.empty() && !reused) {
        report("Aligning", "Lining up with " + vocab_file_, 0.92);
        WordAlignmentOptions options = alignment_;
        options.pad_id = vocabulary.PadIndex();
        options.unknown_id = vocabulary.UnkIndex();
        const bool fasttext = fasttext_model || !fasttext_file.empty();
        if (fasttext && unseen_from_pieces_) {
            options.pieces = [&](const std::vector<std::string>& tokens) {
                return fasttext_model ? WordEmbeddings::FastTextVectors(*fasttext_model, &tokens)
                                      : WordEmbeddings::LoadFastTextBin(fasttext_file, &tokens);
            };
        }
        const auto aligned = WordEmbeddings::Align(vectors, vocabulary.GetWords(), options);
        if (!aligned.success) return arrow::Status::Invalid(op + ": " + aligned.error_message);
        vectors.words = vocabulary.GetWords();
        vectors.values = aligned.values;
        const auto token = [&](int id) {
            return id >= 0 && static_cast<size_t>(id) < vectors.words.size() ? vectors.words[static_cast<size_t>(id)]
                                                                              : std::string();
        };
        publish("Aligning", "Lined up with " + vocab_file_, 0.92,
                {{"tokens", static_cast<double>(vectors.words.size())},
                 {"found", static_cast<double>(aligned.exact)},
                 {"lowercase", static_cast<double>(aligned.lowercase)},
                 {"pieces", static_cast<double>(aligned.pieces)},
                 {"missing", static_cast<double>(aligned.missing)}},
                {{"special", BuildWordCoverageView(aligned.exact, aligned.lowercase, aligned.missing, {},
                                                   token(options.pad_id), token(options.unknown_id),
                                                   alignment_.missing, alignment_.unknown)
                                 .special}});
        coverage = "; vocabulary " + std::to_string(vectors.words.size()) + " tokens: " + std::to_string(aligned.exact) +
                   " found, " + std::to_string(aligned.lowercase) + " in lowercase, " +
                   (fasttext ? std::to_string(aligned.pieces) + " from pieces, " : std::string()) +
                   std::to_string(aligned.missing) + " missing";
    }

    if (!vectors_path_.empty() && !reused) {
        // The default path is a new folder of this run's artifacts.
        const auto folder = std::filesystem::path(vectors_path_).parent_path();
        std::error_code folder_error;
        if (!folder.empty()) std::filesystem::create_directories(folder, folder_error);
        std::string error;
        if (!WordEmbeddings::SaveText(vectors, vectors_path_, &error))
            return arrow::Status::IOError(op + ": cannot save the vectors: " + error);
        if (trained) {
            record.trained_at = LocalTimeNow();
            record.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            record.summary = source_summary;
            record.coverage = coverage;
            if (!WriteWordVectorsTrainingRecord(vectors_path_, record, &error))
                spdlog::warn("WordEmbeddings: the next run trains again: {}", error);
        }
    }
    // The dialog builds unseen words from the model kept for the session.
    if (fasttext_model) KeepFastTextModel(vectors_path_, fasttext_model);
    if (!reused) publish("Done", vectors_path_.empty() ? "Trained the vectors" : "Saved the vectors", 0.94, {});

    // Pin 0: word, v_0 .. v_{d-1}.
    const int64_t n = static_cast<int64_t>(vectors.words.size());
    const int dim = vectors.dim;
    std::vector<std::shared_ptr<arrow::Field>> fields{arrow::field("word", arrow::utf8())};
    std::vector<std::shared_ptr<arrow::Array>> columns;
    arrow::StringBuilder word_builder;
    ARROW_RETURN_NOT_OK(word_builder.AppendValues(vectors.words));
    std::shared_ptr<arrow::Array> word_array;
    ARROW_RETURN_NOT_OK(word_builder.Finish(&word_array));
    columns.push_back(word_array);
    for (int d = 0; d < dim; ++d) {
        arrow::FloatBuilder builder;
        ARROW_RETURN_NOT_OK(builder.Reserve(n));
        for (int64_t r = 0; r < n; ++r) builder.UnsafeAppend(vectors.values[static_cast<size_t>(r) * dim + d]);
        std::shared_ptr<arrow::Array> array;
        ARROW_RETURN_NOT_OK(builder.Finish(&array));
        fields.push_back(arrow::field("v_" + std::to_string(d), arrow::float32()));
        columns.push_back(array);
    }
    auto table = arrow::Table::Make(arrow::schema(fields), columns, n);

    // Pin 1: the mean vector of each row's words.
    if (row_vectors_) {
        report("Row vectors", "Averaging the words of each row", 0.95);
        std::unordered_map<std::string, int> row_of;
        row_of.reserve(vectors.words.size());
        for (size_t r = 0; r < vectors.words.size(); ++r) row_of.emplace(vectors.words[r], static_cast<int>(r));
        // Each row's words that have a vector, as rows of a table holding only
        // the words the text uses (a big pretrained file is not sent whole).
        std::unordered_map<int, int> used_row;
        std::vector<float> used_values;
        std::vector<std::vector<int>> row_ids(sentences.size());
        for (size_t r = 0; r < sentences.size(); ++r) {
            for (const auto& word : sentences[r]) {
                const auto it = row_of.find(word);
                if (it == row_of.end()) continue;
                const auto [used, added] = used_row.emplace(it->second, static_cast<int>(used_row.size()));
                if (added) {
                    const auto from = vectors.values.begin() + static_cast<std::ptrdiff_t>(it->second) * dim;
                    used_values.insert(used_values.end(), from, from + dim);
                }
                row_ids[r].push_back(used->second);
            }
        }
        if (used_values.empty()) used_values.assign(static_cast<size_t>(dim), 0.0f);  // no word has a vector
        // Rows of similar length share a block, so few are padded: one long
        // post no longer widens the block of its short neighbours.
        std::vector<size_t> order(sentences.size());
        for (size_t r = 0; r < order.size(); ++r) order[r] = r;
        std::stable_sort(order.begin(), order.end(),
                         [&](size_t a, size_t b) { return row_ids[a].size() < row_ids[b].size(); });
        std::vector<float> means(sentences.size() * static_cast<size_t>(dim));
        for (size_t start = 0; start < order.size();) {
            ARROW_RETURN_NOT_OK(CheckCancellation(op));
            // Rows while the padded [rows, longest row, dim] block fits the budget.
            size_t end = start, width = 1;
            while (end < order.size()) {
                const size_t w = std::max(width, row_ids[order[end]].size());
                if (end > start && (end - start + 1) * w * static_cast<size_t>(dim) > kRowMeansElements) break;
                width = w;
                ++end;
            }
            std::vector<int> ids((end - start) * width, -1);
            for (size_t i = start; i < end; ++i) {
                const auto& row = row_ids[order[i]];
                std::copy(row.begin(), row.end(), ids.begin() + static_cast<std::ptrdiff_t>((i - start) * width));
            }
            std::string error;
            const auto chunk = WordEmbeddings::RowMeans(used_values, dim, ids, static_cast<int>(width), &error);
            if (chunk.empty()) return arrow::Status::Invalid(op + ": row vectors: " + error);
            // Each mean goes back to its row's place.
            for (size_t i = start; i < end; ++i)
                std::copy_n(chunk.begin() + static_cast<std::ptrdiff_t>((i - start) * dim), dim,
                            means.begin() + static_cast<std::ptrdiff_t>(order[i] * dim));
            start = end;
        }
        std::shared_ptr<arrow::Table> rows = input;
        for (int d = 0; d < dim; ++d) {
            const std::string name = "emb_" + std::to_string(d);
            if (rows->schema()->GetFieldIndex(name) >= 0)
                return arrow::Status::Invalid(op + ": column '" + name + "' already exists");
            arrow::FloatBuilder builder;
            ARROW_RETURN_NOT_OK(builder.Reserve(input->num_rows()));
            for (size_t r = 0; r < sentences.size(); ++r) builder.UnsafeAppend(means[r * static_cast<size_t>(dim) + d]);
            std::shared_ptr<arrow::Array> array;
            ARROW_RETURN_NOT_OK(builder.Finish(&array));
            ARROW_ASSIGN_OR_RAISE(rows, rows->AddColumn(rows->num_columns(), arrow::field(name, arrow::float32()),
                                                        std::make_shared<arrow::ChunkedArray>(array)));
        }
        rows_ = rows;
    }

    outcome_ = source_summary + coverage;
    spdlog::info("WordEmbeddings: {}", outcome_);
    report("Complete", outcome_, 1.0);
    return table;
}

}  // namespace cyxwiz
