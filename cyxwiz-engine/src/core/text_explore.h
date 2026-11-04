#pragma once

// The answers behind Data Studio's Explore tab Text (TOFIX129 D8, approved
// mockup d8_text_mockup.html): one text column, optionally split by a group
// column (a category or the label), in five looks: Words (the commonest
// words and pairs, per group, and words per row), Terms (TF-IDF: what sets
// each group apart, a row's terms and the rows that read most like it),
// Sentiment (VADER on every row, by group), Tokens (what a tokenizer makes of
// the rows and what a length limit cuts) and Embeddings (word vectors learned
// from the column, the nearest words and a map).
//
// The text functions are the backend's, checked against Python / sklearn /
// NLTK / vaderSentiment in test_computation_truth_text; Terms counts with the
// TF-IDF node's own rule (text_vectorizer_contract::Terms and Idf) so the tab
// and the node cannot disagree; Tokens trains the Tokenizer node's tokenizer;
// Embeddings trains with the Word Embeddings node's trainers on the node's
// own split of the text. What is new here is checked in test_text_explore.

#include "data_explore.h"

#include <cyxwiz/text_processing.h>
#include <cyxwiz/word_embeddings.h>

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace cyxwiz::text {

// Returns true when the work should stop (the job's Stop).
using StopQuery = std::function<bool()>;

constexpr size_t kMaxGroups = 12;  // the largest groups; the rest are counted under "other"

struct TextRequest {
    std::string text;
    std::string group;  // "": none
};

struct TextData {
    std::vector<std::string> texts;   // the rows with text, in table order
    std::vector<int64_t> rows;        // each one's row in the table (0-based)
    std::vector<int> group_of;        // each one's group (index into groups); empty without a group column
    std::vector<std::string> groups;  // the largest first; "(missing)" for rows without a group, "other" last when folded
    std::vector<int64_t> group_rows;  // rows with text per group
    int64_t table_rows = 0;
    int64_t missing = 0;              // NULL or a text the project marks as missing
    int64_t empty = 0;                // only spaces
    size_t groups_folded = 0;         // groups beyond kMaxGroups, counted under "other"
};

struct TextAnswer {
    std::string error;
    bool ok() const { return error.empty(); }
    TextData data;
};

// Reads the text column (and the group column) in table order; the rows with
// no text are counted, not kept.
TextAnswer ReadText(const std::string& table, const TextRequest& request, const QueryRunner& run, const explore::MissingTexts& texts);

// Words / tokens per row: the figures and a histogram of `bins` bins up to a
// round cap near the 95th percentile (the longer rows counted in `over`).
struct LengthFigures {
    double mean = 0.0, median = 0.0, p95 = 0.0;
    int64_t max = 0;
    int64_t cap = 0;
    std::vector<int64_t> bins;
    int64_t over = 0;  // rows longer than cap
};
LengthFigures Lengths(const std::vector<int64_t>& lengths, int bins = 30);

struct Count {
    std::string term;
    int64_t count = 0;
};

// ---- Words

struct WordsRequest {
    TextRequest base;
    bool pairs = false;       // count word pairs instead of words
    bool stop_words = true;   // leave out the common English words
    bool stems = false;       // count Porter stems
    int top = 20;
};

struct WordsAnswer {
    std::string error;
    bool ok() const { return error.empty(); }
    WordsRequest request;
    TextData data;
    int64_t words = 0;           // all the words (TextProcessing::Words)
    int64_t kept = 0;            // without the common ones (as asked)
    int64_t distinct = 0;        // distinct terms counted
    std::vector<Count> top;      // words, or pairs when asked
    std::vector<Count> pairs;    // the commonest pairs (always, 10)
    std::vector<std::vector<Count>> by_group;  // per group, its 8 commonest
    LengthFigures per_row;       // words per row
};
WordsAnswer RunWords(const std::string& table, const WordsRequest& request, const QueryRunner& run, const explore::MissingTexts& texts,
                     const StopQuery& stop = {});

// ---- Terms (TF-IDF)

struct TermsRequest {
    TextRequest base;
    int ngram_max = 1;          // 1: words; 2: words and pairs
    int min_df = 5;             // a term in at least this many rows
    std::string norm = "l2";    // "l2", "l1", "none"
    bool stop_words = true;
    int64_t row = -1;           // the row looked at (table row); -1: the first row with text
    int top = 8;
};

struct TermScore {
    std::string term;
    double score = 0.0;   // TF-IDF (a row) or its mean (a group)
    double lift = 0.0;    // a group's mean minus the table's
};

struct NearRow {
    int64_t row = 0;
    double similarity = 0.0;
    std::string group;
    std::string text;
};

struct TermsAnswer {
    std::string error;
    bool ok() const { return error.empty(); }
    TermsRequest request;      // with the row resolved
    TextData data;
    size_t terms = 0;          // terms in at least min_df rows
    std::vector<std::vector<TermScore>> distinctive;  // per group: what sets it apart (the highest lift)
    std::vector<TermScore> overall;                   // the highest mean TF-IDF of the table
    std::string row_text;
    std::string row_group;
    std::vector<TermScore> row_terms;
    std::vector<NearRow> nearest;                     // 5, most alike first (cosine)
    int64_t html_entities = 0, links = 0, repeated = 0;  // text quality the terms show
    int64_t broken = 0;        // rows with broken characters (UTF-8 read as Latin-1 and saved again, or U+FFFD)
};
TermsAnswer RunTerms(const std::string& table, const TermsRequest& request, const QueryRunner& run, const explore::MissingTexts& texts,
                     const StopQuery& stop = {});

// ---- Sentiment (VADER)

struct SentimentRequest {
    TextRequest base;
    double positive_from = 0.05;   // VADER's own thresholds
    double negative_below = -0.05;
    int64_t row = -1;              // the row shown word by word; -1: the first row with text
};

struct GroupSentiment {
    std::string group;
    int64_t rows = 0;
    double mean = 0.0, median = 0.0;
    double negative_share = 0.0, positive_share = 0.0;
};

struct SentimentAnswer {
    std::string error;
    bool ok() const { return error.empty(); }
    SentimentRequest request;   // with the row resolved
    TextData data;
    std::vector<double> compound;  // per text
    int64_t negative = 0, neutral = 0, positive = 0;
    double mean = 0.0;
    std::vector<int64_t> histogram;  // 20 bins over -1 .. 1
    std::vector<GroupSentiment> by_group;
    std::string row_text;
    VaderScores row_scores;
};
SentimentAnswer RunSentiment(const std::string& table, const SentimentRequest& request, const QueryRunner& run,
                             const explore::MissingTexts& texts, const StopQuery& stop = {});

// ---- Tokens

struct TokensRequest {
    TextRequest base;
    int tokenizer = 1;          // cyxwiz::TokenizerType (the Tokenizer node's tokenizer_type; 1: Word)
    int vocabulary = 10000;     // max_vocab_size
    int min_frequency = 2;      // min_word_freq
    int max_length = 256;
    int64_t row = -1;
};

struct TokenShown {
    std::string token;
    int id = 0;
    bool unknown = false;
};

struct TokensAnswer {
    std::string error;
    bool ok() const { return error.empty(); }
    TokensRequest request;       // with the row resolved
    TextData data;
    LengthFigures per_row;       // tokens per row
    double cut_share = 0.0;      // rows longer than max_length
    double cut_share_double = 0.0;  // ... than twice it
    int64_t cut_rows = 0;
    double coverage = 0.0;       // the share of tokens the vocabulary holds (not the unknown token)
    size_t vocabulary_size = 0;  // with the special tokens
    std::string row_text;
    std::vector<TokenShown> row_tokens;  // the first 64
    size_t row_length = 0;
};
TokensAnswer RunTokens(const std::string& table, const TokensRequest& request, const QueryRunner& run, const explore::MissingTexts& texts,
                       const StopQuery& stop = {});

// ---- Embeddings

struct EmbeddingsRequest {
    TextRequest base;
    std::string method = "word2vec";  // "word2vec", "fasttext", "glove"
    bool cbow = false;     // word2vec / fastText: CBOW (a word from its context) instead of skip-gram
    int size = 100;
    int window = 5;        // glove: its own default 10 when 0
    int min_count = 5;
    int epochs = 5;        // glove: 25 when 0
    unsigned seed = 1;
    int64_t sample_words = 500000;  // learn from rows drawn at random (seed) until this many words; 0: every row
};

struct EmbeddingsAnswer {
    std::string error;
    bool ok() const { return error.empty(); }
    EmbeddingsRequest request;
    TextData data;
    WordVectors vectors;
    int64_t words_read = 0;    // words in the sentences trained on
    int64_t rows_used = 0;     // rows trained on (all of data.texts unless sampled)
    bool sampled = false;
    // Per learned word, the rows of each group it is in (the groups of
    // data.groups; empty without a group column), and the rows per group,
    // both over the rows trained on: what "leans to a group" reads.
    std::map<std::string, std::vector<int64_t>> word_group_rows;
    std::vector<int64_t> group_rows;
};
EmbeddingsAnswer RunEmbeddings(const std::string& table, const EmbeddingsRequest& request, const QueryRunner& run,
                               const explore::MissingTexts& texts, const StopQuery& stop = {},
                               const std::function<void(double)>& progress = {});

// The `k` nearest words to `word` (lower-cased) by cosine, on the device.
std::vector<std::pair<std::string, double>> NearestWords(const WordVectors& vectors, const std::string& word, int k, std::string* error);

// ---- The nodes the tab adds (Add as a step / Keep as a node)

std::map<std::string, std::string> NodeParameters(const TermsRequest& request);
std::map<std::string, std::string> NodeParameters(const SentimentRequest& request);
std::map<std::string, std::string> NodeParameters(const TokensRequest& request);
std::map<std::string, std::string> NodeParameters(const EmbeddingsRequest& request);

}  // namespace cyxwiz::text
