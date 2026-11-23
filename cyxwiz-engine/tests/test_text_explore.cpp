// Data Studio's Explore tab Text (TOFIX129 D8): reading a text column and a
// group column through the session query service (missing and blank rows
// counted, groups folded past the largest), the five answers on a small
// corpus whose figures come from scikit-learn's TfidfVectorizer (with the
// Words rule and the stop list) and vaderSentiment (D:/tmp/gui129/d8/
// d8_test_oracle.py), the lengths' histogram, the tokens, word vectors
// learned from the column with their nearest words and map, Stop, the node
// parameters and the words. The text functions themselves are checked in
// test_computation_truth_text.

#include "../src/core/text_explore.h"
#include "../src/core/text_presentation.h"
#include "../src/core/session_query_engine.h"

#include <arrow/api.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using namespace cyxwiz;
namespace fs = std::filesystem;

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void Near(double a, double b, double tol, const std::string& what) {
    Check(std::fabs(a - b) <= tol, what + ": " + std::to_string(a) + " vs " + std::to_string(b));
}

void Contains(const std::string& text, const std::string& part, const std::string& what) {
    Check(text.find(part) != std::string::npos, what + ": \"" + part + "\" not in \"" + text + "\"");
}

const std::vector<std::pair<const char*, const char*>> kRows = {
    {"I feel anxious and my heart races before every exam.", "Anxiety"},
    {"The exam made me anxious, my heart would not slow down.", "Anxiety"},
    {"Anxious again today; the panic and the heart pounding.", "Anxiety"},
    {"I love my new job and my team is great!", "Normal"},
    {"Great day at work, the team is happy and so am I.", "Normal"},
    {"Work was fine. Lunch with the team was great.", "Normal"},
    {"I feel empty and sad, nothing helps and I can't sleep.", "Depression"},
    {"Sad and tired all the time, I don't want to get up.", "Depression"},
    {"Empty days, sad nights, I don't feel anything.", "Depression"},
};

// The nine rows, then a NULL, a blank and a text the project marks missing.
std::shared_ptr<arrow::Table> Corpus() {
    arrow::StringBuilder text, group;
    for (const auto& [t, g] : kRows) {
        (void)text.Append(t);
        (void)group.Append(g);
    }
    (void)text.AppendNull();
    (void)group.Append("Normal");
    (void)text.Append("   ");
    (void)group.Append("Normal");
    (void)text.Append("NA");
    (void)group.Append("Normal");
    std::shared_ptr<arrow::Array> ta, ga;
    (void)text.Finish(&ta);
    (void)group.Finish(&ga);
    return arrow::Table::Make(arrow::schema({arrow::field("statement", arrow::utf8()), arrow::field("status", arrow::utf8())}), {ta, ga});
}

// 30 rows in 15 groups (two rows each): the groups fold past the 11 largest.
std::shared_ptr<arrow::Table> ManyGroups() {
    arrow::StringBuilder text, group;
    for (int i = 0; i < 30; ++i) {
        (void)text.Append("row number " + std::to_string(i) + " says hello");
        (void)group.Append("g" + std::to_string(i % 15 + 10));
    }
    std::shared_ptr<arrow::Array> ta, ga;
    (void)text.Finish(&ta);
    (void)group.Finish(&ga);
    return arrow::Table::Make(arrow::schema({arrow::field("t", arrow::utf8()), arrow::field("g", arrow::utf8())}), {ta, ga});
}

}  // namespace

int main() {
    const fs::path root = fs::temp_directory_path() / "cyxwiz_test_text_explore";
    fs::remove_all(root);
    fs::create_directories(root);
    SessionQueryEngine engine(root.string());
    std::string error;
    Check(engine.AttachArrow("corpus", 1, Corpus(), &error), "attach corpus: " + error);
    Check(engine.AttachArrow("many", 2, ManyGroups(), &error), "attach many: " + error);
    const QueryRunner run = [&](const QueryRequest& r) { return engine.Run(r); };
    const explore::MissingTexts missing = {{"statement", {"NA"}}};
    const text::TextRequest base{"statement", "status"};

    // ---- reading
    const text::TextAnswer read = text::ReadText("corpus", base, run, missing);
    Check(read.ok(), "read: " + read.error);
    Check(read.data.texts.size() == 9 && read.data.table_rows == 12, "read: 9 of 12 rows");
    Check(read.data.missing == 2 && read.data.empty == 1, "read: NULL and NA missing, one blank");
    Check(read.data.groups == std::vector<std::string>({"Anxiety", "Depression", "Normal"}), "groups: the largest first, ties by name");
    Check(read.data.group_of[3] == 2 && read.data.rows[8] == 8, "group of each row, table rows");
    Contains(text_view::RowsWords(read.data), "9 rows with text (3 empty or missing left out)", "rows words");
    Check(!text::ReadText("corpus", {"", ""}, run, missing).ok(), "read: no column");
    Check(!text::ReadText("corpus", {"statement", "statement"}, run, missing).ok(), "read: same column twice");
    const text::TextAnswer many = text::ReadText("many", {"t", "g"}, run, {});
    Check(many.ok() && many.data.groups.size() == text::kMaxGroups && many.data.groups.back() == "other" && many.data.groups_folded == 4 &&
              many.data.group_rows.back() == 8,
          "fold: 11 groups and other (4 groups, 8 rows)");
    Contains(text_view::GroupsWords(many.data, "g"), "15 groups in g (the 11 largest, 4 more as other)", "groups words");

    // ---- lengths
    std::vector<int64_t> lengths;
    for (int64_t i = 1; i <= 1000; ++i) lengths.push_back(i);
    const text::LengthFigures f = text::Lengths(lengths, 30);
    Near(f.median, 500.5, 1e-9, "median");
    Near(f.p95, 950.05, 1e-9, "p95 (numpy linear)");
    Check(f.cap == 1020 && f.bins.size() == 30 && f.over == 0, "cap: 1,000 rounded, a multiple of 30 bins");
    int64_t binned = 0;
    for (const int64_t b : f.bins) binned += b;
    Check(binned == 1000, "every row in a bin");

    // ---- Words
    text::WordsRequest wr;
    wr.base = base;
    const text::WordsAnswer w = text::RunWords("corpus", wr, run, missing);
    Check(w.ok(), "words: " + w.error);
    Check(w.words == 85 && w.kept == 41, "words: 85, 41 without the common ones");
    Check(w.top.size() >= 6 && w.top[0].term == "anxious" && w.top[0].count == 3 && w.top[5].term == "team", "words: the top, ties by the word");
    Check(w.pairs.size() >= 2 && w.pairs[0].term == "anxious heart" && w.pairs[0].count == 2 && w.pairs[1].term == "team great",
          "pairs after the common words go (sklearn's order)");
    Check(w.by_group.size() == 3 && w.by_group[0][0].term == "anxious", "words per group");
    const std::string ws = text_view::WordsSentence(w);
    Contains(ws, "The column talks most about anxious, feel, great, heart, sad and team", "words sentence");
    Contains(ws, "Rows run to 9 words on the median", "words sentence: lengths");
    wr.stems = true;
    const text::WordsAnswer stems = text::RunWords("corpus", wr, run, missing);
    bool feel = false;
    for (const auto& c : stems.top) feel = feel || (c.term == "feel" && c.count == 3);
    Check(stems.ok() && feel, "stems: Porter");
    wr.stems = false;
    wr.pairs = true;
    const text::WordsAnswer pairs = text::RunWords("corpus", wr, run, missing);
    Check(pairs.ok() && pairs.top[0].term == "anxious heart", "words: pairs counted");
    Contains(text_view::WordsCsv(w), "1,anxious,3,", "words csv");

    // ---- Terms (sklearn TfidfVectorizer, min_df 2, l2, smooth)
    text::TermsRequest tr;
    tr.base = base;
    tr.min_df = 2;
    const text::TermsAnswer t = text::RunTerms("corpus", tr, run, missing);
    Check(t.ok(), "terms: " + t.error);
    Check(t.terms == 8, "terms: 8 in two rows or more");
    Check(t.distinctive.size() == 3, "terms: per group");
    // groups: Anxiety, Depression, Normal
    Check(t.distinctive[0][0].term == "anxious" && t.distinctive[0][1].term == "heart" && t.distinctive[0][2].term == "exam", "Anxiety's terms");
    Near(t.distinctive[0][0].score, 0.5788893599379961, 1e-6, "Anxiety: anxious mean");
    Near(t.distinctive[0][0].lift, 0.38592623995866404, 1e-6, "Anxiety: anxious lift");
    Check(t.distinctive[1][0].term == "sad" && t.distinctive[1][1].term == "feel", "Depression's terms");
    Near(t.distinctive[1][0].score, 0.8047378541243649, 1e-6, "Depression: sad mean");
    Check(t.distinctive[2][0].term == "great" && t.distinctive[2][2].term == "work", "Normal's terms");
    Check(t.request.row == 0 && t.row_terms.size() == 4 && t.row_terms[0].term == "exam" && t.row_terms[1].term == "anxious", "row 0's terms");
    Near(t.row_terms[0].score, 0.5531755197949497, 1e-6, "row 0: exam");
    Check(t.nearest.size() == 5 && t.nearest[0].row == 1 && t.nearest[1].row == 2 && t.nearest[2].row == 6, "nearest rows (cosine)");
    Near(t.nearest[0].similarity, 0.8767369722823399, 1e-6, "nearest: similarity");
    Contains(text_view::TermsSentence(t), "Anxiety by anxious, heart, exam", "terms sentence");
    Contains(text_view::TermsCsv(t), "Anxiety,anxious,", "terms csv");
    tr.min_df = 50;
    Check(!text::RunTerms("corpus", tr, run, missing).ok(), "terms: none in 50 rows");

    // ---- Sentiment (vaderSentiment, unrounded)
    text::SentimentRequest sr;
    sr.base = base;
    const text::SentimentAnswer s = text::RunSentiment("corpus", sr, run, missing);
    Check(s.ok(), "sentiment: " + s.error);
    const double expected[] = {0.49391458057363097, 0.49391458057363097, -0.025811286645983277, 0.8622006727673295, 0.8316320352807864,
                               0.7095601949854891, -0.725603915509538, -0.7369092073198141, -0.5993731596731062};
    for (size_t i = 0; i < 9; ++i) Near(s.compound[i], expected[i], 1e-12, "compound " + std::to_string(i));
    Check(s.positive == 5 && s.negative == 3 && s.neutral == 1, "sentiment: by VADER's thresholds");
    Near(s.by_group[1].median, -0.725603915509538, 1e-12, "Depression median");
    Near(s.by_group[2].median, 0.8316320352807864, 1e-12, "Normal median");
    int64_t histogram = 0;
    for (const int64_t b : s.histogram) histogram += b;
    Check(histogram == 9 && s.histogram[19] == 0 && s.histogram[18] == 2, "histogram over -1 .. 1");
    Check(!s.row_scores.words.empty(), "the row word by word");
    const std::string ss = text_view::SentimentSentence(s);
    Contains(ss, "Most rows read positive (56%)", "sentiment sentence");
    Contains(ss, "Depression (-0.73), Anxiety (0.49) and Normal (0.83)", "sentiment by group");
    Contains(text_view::SentimentCsv(s), "7,Depression,", "sentiment csv");
    Check(text::NodeParameters(sr).count("method") == 0, "sentiment node: VADER only, no method");

    // ---- Tokens
    text::TokensRequest kr;
    kr.base = base;
    kr.min_frequency = 2;
    kr.max_length = 10;
    const text::TokensAnswer k = text::RunTokens("corpus", kr, run, missing);
    Check(k.ok(), "tokens: " + k.error);
    Check(k.cut_rows > 0 && k.cut_rows < 9 && k.cut_share > 0.0, "tokens: some rows past 10 tokens");
    Check(k.coverage > 0.3 && k.coverage < 1.0, "tokens: words seen once are unknown");
    Check(!k.row_tokens.empty() && k.row_length >= k.row_tokens.size(), "tokens: the row");
    Contains(text_view::TokensSentence(k), "At 10 tokens, ", "tokens sentence");
    kr.max_length = 0;
    Check(!text::RunTokens("corpus", kr, run, missing).ok(), "tokens: a limit of 0");

    // ---- Embeddings
    text::EmbeddingsRequest er;
    er.base = base;
    er.size = 8;
    er.min_count = 1;
    er.epochs = 3;
    const text::EmbeddingsAnswer e = text::RunEmbeddings("corpus", er, run, missing);
    Check(e.ok(), "embeddings: " + e.error);
    Check(e.vectors.dim == 8 && !e.vectors.words.empty() && e.words_read > 0, "embeddings: vectors learned");
    Check(!e.sampled && e.rows_used == 9, "embeddings: a small column is learned whole");
    Check(e.group_rows.size() == 3 && e.group_rows[0] == 3 && e.word_group_rows.count("sad") && e.word_group_rows.at("sad")[1] == 3,
          "embeddings: rows per group for the word map (\"sad\" in the 3 Depression rows)");
    std::string nearest_error;
    const auto nearest = text::NearestWords(e.vectors, "Sad", 3, &nearest_error);
    Check(nearest.size() == 3, "nearest words: " + nearest_error);
    Contains(text_view::EmbeddingsSentence(e, "sad", nearest), "\"sad\" sits nearest to", "embeddings sentence");
    Contains(text_view::EmbeddingsSentence(e, "sad", nearest), "Learned from this column alone", "embeddings sentence: whole column");
    // A sample: rows drawn until the words asked for.
    text::EmbeddingsRequest sample = er;
    sample.sample_words = 20;
    const text::EmbeddingsAnswer es = text::RunEmbeddings("corpus", sample, run, missing);
    Check(es.ok() && es.sampled && es.rows_used >= 2 && es.rows_used < 9 && es.words_read >= 20,
          "embeddings: a sample of the rows (" + es.error + ", rows " + std::to_string(es.rows_used) + ", words " +
              std::to_string(es.words_read) + ")");
    Contains(text_view::EmbeddingsSentence(es, "", {}), "Learned from " + std::to_string(es.rows_used) + " of the column's 9 rows", "sample sentence");
    // CBOW trains too, and the node gets the architecture.
    text::EmbeddingsRequest cbow = er;
    cbow.cbow = true;
    const text::EmbeddingsAnswer ec = text::RunEmbeddings("corpus", cbow, run, missing);
    Check(ec.ok() && ec.vectors.dim == 8, "embeddings: CBOW: " + ec.error);
    Check(text::NodeParameters(cbow).at("architecture") == "cbow" && text::NodeParameters(er).at("architecture") == "skipgram",
          "Word Embeddings node architecture");
    er.method = "nope";
    Check(!text::RunEmbeddings("corpus", er, run, missing).ok(), "embeddings: unknown method");

    // ---- Stop
    const text::StopQuery stop_now = [] { return true; };
    Check(text::RunWords("corpus", wr, run, missing, stop_now).error == "Cancelled.", "stop: words");
    Check(text::RunSentiment("corpus", sr, run, missing, stop_now).error == "Cancelled.", "stop: sentiment");

    // ---- The nodes
    tr.min_df = 5;
    tr.ngram_max = 2;
    const auto tp = text::NodeParameters(tr);
    Check(tp.at("text_col") == "statement" && tp.at("label_col") == "status" && tp.at("min_df") == "5" && tp.at("ngram_range") == "1,2" &&
              tp.at("stop_words") == "english" && tp.at("norm") == "l2",
          "TF-IDF node parameters");
    const auto kp = text::NodeParameters(k.request);
    Check(kp.at("tokenizer_type") == "1" && kp.at("max_length") == "10" && kp.at("min_word_freq") == "2" && kp.at("max_vocab_size") == "10000",
          "Tokenizer node parameters");
    text::TokensRequest no_cap = k.request;
    no_cap.vocabulary = 0;
    Check(text::NodeParameters(no_cap).at("max_vocab_size") == "1", "Tokenizer node parameters: a vocabulary of 1 or more, as the node takes it");
    er.method = "glove";
    er.window = 0;
    er.epochs = 0;
    const auto ep = text::NodeParameters(er);
    Check(ep.at("source") == "glove" && ep.count("glove_window") == 0 && ep.at("vector_size") == "8", "Word Embeddings node parameters");

    fs::remove_all(root);
    std::cout << "text explore: reading, groups, lengths, words, terms (sklearn), sentiment (VADER), tokens, embeddings, stop, nodes, words. OK\n";
    return 0;
}
