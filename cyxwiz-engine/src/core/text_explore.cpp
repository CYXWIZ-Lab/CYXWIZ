#include "text_explore.h"

#include "explore_read.h"
#include "node_executors/text_vectorizer_contract.h"
#include "node_executors/word_embeddings_operator.h"

#include <cyxwiz/dimensionality_reduction.h>
#include <cyxwiz/tokenizer.h>

#include <arrayfire.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace cyxwiz::text {
namespace {

namespace contract = text_vectorizer_contract;

std::string Ident(const std::string& name) {
    std::string out = "\"";
    for (const char c : name) out += c == '"' ? "\"\"" : std::string(1, c);
    return out + "\"";
}

bool Blank(const std::string& s) {
    for (const char c : s)
        if (c != ' ' && c != '\t' && c != '\r' && c != '\n' && c != '\f' && c != '\v') return false;
    return true;
}

bool Stopped(const StopQuery& stop) { return stop && stop(); }

// The commonest first, ties by the term (a stable order on every run).
std::vector<Count> Top(const std::unordered_map<std::string, int64_t>& counts, size_t n) {
    std::vector<Count> all;
    all.reserve(counts.size());
    for (const auto& [term, count] : counts) all.push_back({term, count});
    const size_t k = std::min(n, all.size());
    std::partial_sort(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(k), all.end(), [](const Count& a, const Count& b) {
        return a.count != b.count ? a.count > b.count : a.term < b.term;
    });
    all.resize(k);
    return all;
}

double Median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    const size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid), v.end());
    if (v.size() % 2 == 1) return v[mid];
    const double hi = v[mid];
    return (*std::max_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid)) + hi) / 2.0;
}

// numpy.percentile's linear rule.
double Percentile(std::vector<double> sorted, double q) {
    if (sorted.empty()) return 0.0;
    std::sort(sorted.begin(), sorted.end());
    const double pos = q / 100.0 * static_cast<double>(sorted.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = std::min(lo + 1, sorted.size() - 1);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - static_cast<double>(lo));
}

// A round number at or above v: 1, 2, 2.5 or 5 times a power of ten.
int64_t RoundUp(double v) {
    if (v <= 1.0) return 1;
    const double p = std::pow(10.0, std::floor(std::log10(v)));
    for (const double m : {1.0, 2.0, 2.5, 5.0, 10.0})
        if (m * p >= v) return static_cast<int64_t>(std::llround(m * p));
    return static_cast<int64_t>(std::ceil(v));
}

// The row looked at: the asked table row when it has text, else the first.
size_t RowIndex(const TextData& d, int64_t row) {
    if (row >= 0) {
        const auto it = std::lower_bound(d.rows.begin(), d.rows.end(), row);
        if (it != d.rows.end() && *it == row) return static_cast<size_t>(it - d.rows.begin());
    }
    return 0;
}

std::string GroupOf(const TextData& d, size_t i) { return d.group_of.empty() ? std::string() : d.groups[static_cast<size_t>(d.group_of[i])]; }

// Reads and checks the column: at least two rows with text.
template <typename Answer>
bool Read(Answer& a, const std::string& table, const TextRequest& request, const QueryRunner& run, const explore::MissingTexts& texts) {
    TextAnswer read = ReadText(table, request, run, texts);
    if (!read.ok()) {
        a.error = read.error;
        return false;
    }
    a.data = std::move(read.data);
    return true;
}

}  // namespace

TextAnswer ReadText(const std::string& table, const TextRequest& request, const QueryRunner& run, const explore::MissingTexts& texts) {
    TextAnswer a;
    if (request.text.empty()) return a.error = "Pick the text column.", a;
    if (request.text == request.group) return a.error = "The text and the group are the same column.", a;
    std::vector<QueryParam> params;
    const auto missing_texts = [&](const std::string& column) {
        const auto it = texts.find(column);
        return it == texts.end() ? std::vector<std::string>{} : it->second;
    };
    std::string sql = "SELECT CASE WHEN " + explore::MissingCondition(request.text, missing_texts(request.text), params) +
                      " THEN NULL ELSE CAST(" + Ident(request.text) + " AS VARCHAR) END AS x";
    if (!request.group.empty())
        sql += ", CASE WHEN " + explore::MissingCondition(request.group, missing_texts(request.group), params) + " THEN NULL ELSE CAST(" +
               Ident(request.group) + " AS VARCHAR) END AS g";
    sql += " FROM " + Ident(table);
    QueryRequest query;
    query.sql = sql;
    query.params = std::move(params);
    query.label = "Text: " + request.text;
    const QueryResult result = run(query);
    if (!result.ok) return a.error = result.error.empty() ? "Cancelled." : result.error, a;
    if (!result.table) return a.error = "The query returned nothing.", a;
    TextData& d = a.data;
    d.table_rows = result.table->num_rows();
    auto x = explore_read::Labels(*result.table, "x");
    const auto g = request.group.empty() ? std::vector<std::optional<std::string>>{} : explore_read::Labels(*result.table, "g");
    std::vector<std::string> raw_groups;
    for (int64_t i = 0; i < d.table_rows; ++i) {
        const size_t k = static_cast<size_t>(i);
        if (k >= x.size() || !x[k]) {
            ++d.missing;
            continue;
        }
        if (Blank(*x[k])) {
            ++d.empty;
            continue;
        }
        d.texts.push_back(std::move(*x[k]));
        d.rows.push_back(i);
        if (!request.group.empty()) raw_groups.push_back(k < g.size() && g[k] ? *g[k] : std::string("(missing)"));
    }
    if (d.texts.size() < 2)
        return a.error = d.texts.empty() ? "\"" + request.text + "\" has no text." : "\"" + request.text + "\" has text in one row only.", a;
    if (!request.group.empty()) {
        std::unordered_map<std::string, int64_t> sizes;
        for (const std::string& v : raw_groups) ++sizes[v];
        std::vector<Count> order = Top(sizes, sizes.size());
        const size_t keep = order.size() > kMaxGroups ? kMaxGroups - 1 : order.size();
        std::unordered_map<std::string, int> index;
        for (size_t i = 0; i < keep; ++i) {
            index[order[i].term] = static_cast<int>(i);
            d.groups.push_back(order[i].term);
            d.group_rows.push_back(order[i].count);
        }
        if (keep < order.size()) {
            d.groups_folded = order.size() - keep;
            d.groups.push_back("other");
            int64_t rest = 0;
            for (size_t i = keep; i < order.size(); ++i) rest += order[i].count;
            d.group_rows.push_back(rest);
        }
        d.group_of.reserve(raw_groups.size());
        for (const std::string& v : raw_groups) {
            const auto it = index.find(v);
            d.group_of.push_back(it != index.end() ? it->second : static_cast<int>(keep));
        }
    }
    return a;
}

LengthFigures Lengths(const std::vector<int64_t>& lengths, int bins) {
    LengthFigures f;
    if (lengths.empty()) return f;
    std::vector<double> v;
    v.reserve(lengths.size());
    for (const int64_t n : lengths) v.push_back(static_cast<double>(n));
    f.mean = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
    f.median = Median(v);
    f.p95 = Percentile(v, 95.0);
    f.max = *std::max_element(lengths.begin(), lengths.end());
    f.cap = std::max<int64_t>(RoundUp(Percentile(v, 99.0)), bins);
    const int64_t width = (f.cap + bins - 1) / bins;
    f.cap = width * bins;
    f.bins.assign(static_cast<size_t>(bins), 0);
    for (const int64_t n : lengths) {
        if (n > f.cap) {
            ++f.over;
            ++f.bins.back();
        } else {
            ++f.bins[static_cast<size_t>(std::min<int64_t>(n / width, bins - 1))];
        }
    }
    return f;
}

// ---- Words

WordsAnswer RunWords(const std::string& table, const WordsRequest& request, const QueryRunner& run, const explore::MissingTexts& texts,
                     const StopQuery& stop) {
    WordsAnswer a;
    a.request = request;
    if (!Read(a, table, request.base, run, texts)) return a;
    const TextData& d = a.data;
    std::unordered_map<std::string, int64_t> words, pairs;
    std::vector<std::unordered_map<std::string, int64_t>> per_group(d.groups.size());
    std::vector<int64_t> lengths;
    lengths.reserve(d.texts.size());
    for (size_t i = 0; i < d.texts.size(); ++i) {
        if ((i & 1023) == 0 && Stopped(stop)) return a.error = "Cancelled.", a;
        std::vector<std::string> w = TextProcessing::Words(d.texts[i]);
        lengths.push_back(static_cast<int64_t>(w.size()));
        a.words += static_cast<int64_t>(w.size());
        if (request.stop_words) w = TextProcessing::RemoveStopwords(w, "english");
        if (request.stems) w = TextProcessing::StemWords(w);
        a.kept += static_cast<int64_t>(w.size());
        for (const std::string& t : w) ++words[t];
        for (size_t k = 1; k < w.size(); ++k) ++pairs[w[k - 1] + " " + w[k]];
        if (!d.group_of.empty()) {
            auto& counts = per_group[static_cast<size_t>(d.group_of[i])];
            if (request.pairs)
                for (size_t k = 1; k < w.size(); ++k) ++counts[w[k - 1] + " " + w[k]];
            else
                for (const std::string& t : w) ++counts[t];
        }
    }
    const auto& counted = request.pairs ? pairs : words;
    a.distinct = static_cast<int64_t>(counted.size());
    a.top = Top(counted, static_cast<size_t>(std::max(1, request.top)));
    a.pairs = Top(pairs, 10);
    for (const auto& counts : per_group) a.by_group.push_back(Top(counts, 8));
    a.per_row = Lengths(lengths);
    return a;
}

// ---- Terms

TermsAnswer RunTerms(const std::string& table, const TermsRequest& request, const QueryRunner& run, const explore::MissingTexts& texts,
                     const StopQuery& stop) {
    TermsAnswer a;
    a.request = request;
    if (request.norm != "l2" && request.norm != "l1" && request.norm != "none") return a.error = "Normalise by l2, l1 or none.", a;
    if (!Read(a, table, request.base, run, texts)) return a;
    const TextData& d = a.data;
    const int ngram_max = std::clamp(request.ngram_max, 1, 2);
    const int min_df = std::max(1, request.min_df);

    // The counts per row and the rows each term is in, as the TF-IDF node counts them.
    std::vector<std::unordered_map<std::string, int>> counts(d.texts.size());
    std::unordered_map<std::string, int64_t> df;
    std::unordered_set<std::string> seen;
    for (size_t i = 0; i < d.texts.size(); ++i) {
        if ((i & 1023) == 0 && Stopped(stop)) return a.error = "Cancelled.", a;
        const std::string& t = d.texts[i];
        if (t.find("&quot;") != std::string::npos || t.find("&amp;") != std::string::npos || t.find("&#") != std::string::npos) ++a.html_entities;
        if (t.find("http://") != std::string::npos || t.find("https://") != std::string::npos) ++a.links;
        if (!seen.insert(t).second) ++a.repeated;
        for (const std::string& term : contract::Terms(t, request.stop_words, 1, ngram_max)) ++counts[i][term];
        for (const auto& [term, n] : counts[i]) ++df[term];
    }
    // The vocabulary: terms in at least min_df rows, in term order.
    std::vector<std::string> vocab;
    for (const auto& [term, n] : df)
        if (n >= min_df) vocab.push_back(term);
    std::sort(vocab.begin(), vocab.end());
    a.terms = vocab.size();
    if (vocab.empty()) return a.error = "No term is in " + std::to_string(min_df) + " rows or more; lower \"in at least\".", a;
    std::unordered_map<std::string, int> column;
    column.reserve(vocab.size());
    for (size_t j = 0; j < vocab.size(); ++j) column.emplace(vocab[j], static_cast<int>(j));
    std::vector<double> idf(vocab.size());
    for (size_t j = 0; j < vocab.size(); ++j)
        idf[j] = contract::Idf(static_cast<double>(d.texts.size()), static_cast<double>(df[vocab[j]]), true);

    // Each row's TF-IDF (count x idf, normalised), as CSR.
    std::vector<int> row_offsets{0}, columns;
    std::vector<float> values;
    std::vector<double> norms;
    std::vector<double> mean(vocab.size(), 0.0);
    std::vector<std::vector<double>> group_sum(d.groups.size(), std::vector<double>(vocab.size(), 0.0));
    for (size_t i = 0; i < d.texts.size(); ++i) {
        if ((i & 1023) == 0 && Stopped(stop)) return a.error = "Cancelled.", a;
        std::vector<std::pair<int, double>> row;
        for (const auto& [term, n] : counts[i]) {
            const auto it = column.find(term);
            if (it != column.end()) row.emplace_back(it->second, static_cast<double>(n) * idf[static_cast<size_t>(it->second)]);
        }
        std::sort(row.begin(), row.end());
        double norm = 0.0;
        for (const auto& [j, v] : row) norm += request.norm == "l1" ? std::fabs(v) : v * v;
        if (request.norm == "l2") norm = std::sqrt(norm);
        if (request.norm != "none" && norm > 0)
            for (auto& [j, v] : row) v /= norm;
        double length = 0.0;
        for (const auto& [j, v] : row) {
            length += v * v;
            mean[static_cast<size_t>(j)] += v;
            if (!d.group_of.empty()) group_sum[static_cast<size_t>(d.group_of[i])][static_cast<size_t>(j)] += v;
            columns.push_back(j);
            values.push_back(static_cast<float>(v));
        }
        norms.push_back(std::sqrt(length));
        row_offsets.push_back(static_cast<int>(columns.size()));
        counts[i].clear();
    }
    const double n_rows = static_cast<double>(d.texts.size());
    for (double& m : mean) m /= n_rows;
    const auto best = [&](const std::vector<double>& score, const std::vector<double>* base, size_t k) {
        std::vector<int> order(score.size());
        std::iota(order.begin(), order.end(), 0);
        const auto key = [&](int j) { return base ? score[static_cast<size_t>(j)] - (*base)[static_cast<size_t>(j)] : score[static_cast<size_t>(j)]; };
        const size_t take = std::min(k, order.size());
        std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(take), order.end(), [&](int x, int y) {
            return key(x) != key(y) ? key(x) > key(y) : vocab[static_cast<size_t>(x)] < vocab[static_cast<size_t>(y)];
        });
        std::vector<TermScore> out;
        for (size_t i = 0; i < take; ++i) {
            const size_t j = static_cast<size_t>(order[i]);
            out.push_back({vocab[j], score[j], base ? score[j] - (*base)[j] : 0.0});
        }
        return out;
    };
    const size_t top = static_cast<size_t>(std::max(1, request.top));
    a.overall = best(mean, nullptr, top);
    for (size_t g = 0; g < d.groups.size(); ++g) {
        std::vector<double> m = group_sum[g];
        for (double& v : m) v /= static_cast<double>(std::max<int64_t>(1, d.group_rows[g]));
        a.distinctive.push_back(best(m, &mean, top));
    }

    // The row and the rows most like it: one sparse product on the device.
    const size_t r = RowIndex(d, request.row);
    a.request.row = d.rows[r];
    a.row_text = d.texts[r];
    a.row_group = GroupOf(d, r);
    std::vector<double> row_dense(vocab.size(), 0.0);
    for (int k = row_offsets[r]; k < row_offsets[r + 1]; ++k) row_dense[static_cast<size_t>(columns[static_cast<size_t>(k)])] = values[static_cast<size_t>(k)];
    a.row_terms = best(row_dense, nullptr, top);
    a.row_terms.erase(std::remove_if(a.row_terms.begin(), a.row_terms.end(), [](const TermScore& t) { return t.score <= 0.0; }),
                      a.row_terms.end());
    if (!columns.empty() && norms[r] > 0) {
        std::vector<float> similarity;
        try {
            const af::array csr = af::sparse(static_cast<dim_t>(d.texts.size()), static_cast<dim_t>(vocab.size()),
                                             af::array(static_cast<dim_t>(values.size()), values.data()),
                                             af::array(static_cast<dim_t>(row_offsets.size()), row_offsets.data()),
                                             af::array(static_cast<dim_t>(columns.size()), columns.data()), AF_STORAGE_CSR);
            std::vector<float> dense;
            dense.reserve(row_dense.size());
            for (const double x : row_dense) dense.push_back(static_cast<float>(x));
            const af::array product = af::matmul(csr, af::array(static_cast<dim_t>(dense.size()), dense.data()));
            similarity.resize(d.texts.size());
            product.host(similarity.data());
        } catch (const af::exception& e) {
            return a.error = std::string("The device could not compare the rows: ") + e.what(), a;
        }
        std::vector<std::pair<double, size_t>> alike;
        for (size_t i = 0; i < d.texts.size(); ++i)
            if (i != r && norms[i] > 0) alike.emplace_back(similarity[i] / (norms[i] * norms[r]), i);
        const size_t take = std::min<size_t>(5, alike.size());
        std::partial_sort(alike.begin(), alike.begin() + static_cast<std::ptrdiff_t>(take), alike.end(),
                          [](const auto& x, const auto& y) { return x.first != y.first ? x.first > y.first : x.second < y.second; });
        for (size_t k = 0; k < take; ++k)
            a.nearest.push_back({d.rows[alike[k].second], alike[k].first, GroupOf(d, alike[k].second), d.texts[alike[k].second]});
    }
    return a;
}

// ---- Sentiment

SentimentAnswer RunSentiment(const std::string& table, const SentimentRequest& request, const QueryRunner& run,
                             const explore::MissingTexts& texts, const StopQuery& stop) {
    SentimentAnswer a;
    a.request = request;
    if (!(request.negative_below <= request.positive_from)) return a.error = "\"Negative below\" must not be above \"positive from\".", a;
    if (!Read(a, table, request.base, run, texts)) return a;
    const TextData& d = a.data;
    a.compound.reserve(d.texts.size());
    a.histogram.assign(20, 0);
    std::vector<std::vector<double>> per_group(d.groups.size());
    double sum = 0.0;
    for (size_t i = 0; i < d.texts.size(); ++i) {
        if ((i & 255) == 0 && Stopped(stop)) return a.error = "Cancelled.", a;
        const double c = TextProcessing::Vader(d.texts[i]).compound;
        a.compound.push_back(c);
        sum += c;
        if (c >= request.positive_from) ++a.positive;
        else if (c <= request.negative_below) ++a.negative;
        else ++a.neutral;
        ++a.histogram[static_cast<size_t>(std::clamp(static_cast<int>(std::floor((c + 1.0) * 10.0)), 0, 19))];
        if (!d.group_of.empty()) per_group[static_cast<size_t>(d.group_of[i])].push_back(c);
    }
    a.mean = sum / static_cast<double>(d.texts.size());
    for (size_t g = 0; g < d.groups.size(); ++g) {
        const auto& v = per_group[g];
        GroupSentiment s;
        s.group = d.groups[g];
        s.rows = static_cast<int64_t>(v.size());
        if (!v.empty()) {
            s.mean = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
            s.median = Median(v);
            int64_t neg = 0, pos = 0;
            for (const double c : v) {
                if (c >= request.positive_from) ++pos;
                else if (c <= request.negative_below) ++neg;
            }
            s.negative_share = static_cast<double>(neg) / static_cast<double>(v.size());
            s.positive_share = static_cast<double>(pos) / static_cast<double>(v.size());
        }
        a.by_group.push_back(s);
    }
    const size_t r = RowIndex(d, request.row);
    a.request.row = d.rows[r];
    a.row_text = d.texts[r];
    a.row_scores = TextProcessing::Vader(d.texts[r]);
    return a;
}

// ---- Tokens

TokensAnswer RunTokens(const std::string& table, const TokensRequest& request, const QueryRunner& run, const explore::MissingTexts& texts,
                       const StopQuery& stop) {
    TokensAnswer a;
    a.request = request;
    if (request.tokenizer < 0 || request.tokenizer > static_cast<int>(TokenizerType::SentencePieceUnigram))
        return a.error = "Pick a tokenizer.", a;
    const auto type = static_cast<TokenizerType>(request.tokenizer);
    if (IsSentencePieceTokenizerType(type) && !IsSentencePieceTokenizerAvailable()) return a.error = SentencePieceTokenizerUnavailableMessage(), a;
    if (request.max_length < 1) return a.error = "The length limit must be 1 or more tokens.", a;
    if (!Read(a, table, request.base, run, texts)) return a;
    const TextData& d = a.data;
    Tokenizer tokenizer(type);
    tokenizer.SetCancellationQuery([&stop] { return Stopped(stop); });
    tokenizer.SetPadding(false);
    tokenizer.SetTruncation(false);
    try {
        tokenizer.Train(d.texts, std::max(1, request.min_frequency), request.vocabulary > 0 ? request.vocabulary : -1);
    } catch (const std::exception& e) {
        if (Stopped(stop)) return a.error = "Cancelled.", a;
        return a.error = std::string("The tokenizer could not learn this column: ") + e.what(), a;
    }
    a.vocabulary_size = tokenizer.GetVocabulary().Size();
    const int unk = tokenizer.GetVocabulary().UnkIndex();
    std::vector<int64_t> lengths;
    lengths.reserve(d.texts.size());
    int64_t tokens = 0, known = 0, double_cut = 0;
    for (size_t i = 0; i < d.texts.size(); ++i) {
        if ((i & 1023) == 0 && Stopped(stop)) return a.error = "Cancelled.", a;
        const std::vector<int> ids = tokenizer.Encode(d.texts[i]);
        lengths.push_back(static_cast<int64_t>(ids.size()));
        tokens += static_cast<int64_t>(ids.size());
        for (const int id : ids) known += id != unk ? 1 : 0;
        if (static_cast<int64_t>(ids.size()) > request.max_length) ++a.cut_rows;
        if (static_cast<int64_t>(ids.size()) > 2LL * request.max_length) ++double_cut;
    }
    a.per_row = Lengths(lengths, 32);
    a.cut_share = static_cast<double>(a.cut_rows) / static_cast<double>(d.texts.size());
    a.cut_share_double = static_cast<double>(double_cut) / static_cast<double>(d.texts.size());
    a.coverage = tokens > 0 ? static_cast<double>(known) / static_cast<double>(tokens) : 0.0;
    const size_t r = RowIndex(d, request.row);
    a.request.row = d.rows[r];
    a.row_text = d.texts[r];
    const TokenizedText shown = tokenizer.Tokenize(d.texts[r]);
    a.row_length = shown.token_ids.size();
    for (size_t k = 0; k < shown.token_ids.size() && k < 64; ++k)
        a.row_tokens.push_back({k < shown.tokens.size() ? shown.tokens[k] : tokenizer.GetVocabulary().IndexToWord(shown.token_ids[k]),
                                shown.token_ids[k], shown.token_ids[k] == unk});
    return a;
}

// ---- Embeddings

EmbeddingsAnswer RunEmbeddings(const std::string& table, const EmbeddingsRequest& request, const QueryRunner& run,
                               const explore::MissingTexts& texts, const StopQuery& stop, const std::function<void(double)>& progress) {
    EmbeddingsAnswer a;
    a.request = request;
    if (request.method != "word2vec" && request.method != "fasttext" && request.method != "glove") return a.error = "Learn with word2vec, fastText or GloVe.", a;
    if (request.size < 2) return a.error = "The size must be 2 or more.", a;
    if (!Read(a, table, request.base, run, texts)) return a;
    // The Word Embeddings node's own split of the text, so its vectors are these.
    std::vector<std::vector<std::string>> sentences;
    sentences.reserve(a.data.texts.size());
    for (const std::string& t : a.data.texts) {
        sentences.push_back(WordEmbeddingsOperator::SplitWords(t));
        a.words_read += static_cast<int64_t>(sentences.back().size());
    }
    const auto on_progress = [&](double done, int) {
        if (progress) progress(done);
        return !Stopped(stop);
    };
    if (request.method == "glove") {
        GloVeOptions o;
        o.vector_size = request.size;
        if (request.window > 0) o.window = request.window;
        o.min_count = std::max(1, request.min_count);
        if (request.epochs > 0) o.epochs = request.epochs;
        o.seed = request.seed;
        o.on_progress = on_progress;
        GloVeModel m = WordEmbeddings::TrainGloVe(sentences, o);
        if (!m.success) return a.error = Stopped(stop) ? "Cancelled." : m.error_message, a;
        a.vectors = std::move(m.vectors);
    } else if (request.method == "fasttext") {
        FastTextOptions o;
        o.vector_size = request.size;
        o.window = std::max(1, request.window);
        o.min_count = std::max(1, request.min_count);
        o.epochs = std::max(1, request.epochs);
        o.seed = request.seed;
        o.on_progress = on_progress;
        const FastTextModel m = WordEmbeddings::TrainFastText(sentences, o);
        if (!m.success) return a.error = Stopped(stop) ? "Cancelled." : m.error_message, a;
        a.vectors = WordEmbeddings::FastTextVectors(m);
    } else {
        Word2VecOptions o;
        o.vector_size = request.size;
        o.window = std::max(1, request.window);
        o.min_count = std::max(1, request.min_count);
        o.epochs = std::max(1, request.epochs);
        o.seed = request.seed;
        o.on_progress = on_progress;
        Word2VecModel m = WordEmbeddings::TrainWord2Vec(sentences, o);
        if (!m.success) return a.error = Stopped(stop) ? "Cancelled." : m.error_message, a;
        a.vectors = std::move(m.vectors);
    }
    if (a.vectors.words.empty()) return a.error = "No word is seen " + std::to_string(request.min_count) + " times or more; lower it.", a;
    // The trainers keep the most frequent first: the map's words come from there.
    const std::set<std::string>& stop_words = TextProcessing::GetStopwords();
    for (const std::string& w : a.vectors.words) {
        if (a.common.size() >= 40) break;
        if (w.size() >= 3 && stop_words.count(w) == 0) a.common.push_back(w);
    }
    return a;
}

std::vector<std::pair<std::string, double>> NearestWords(const WordVectors& vectors, const std::string& word, int k, std::string* error) {
    std::string lower;
    for (const std::string& w : TextProcessing::Words(word)) lower += (lower.empty() ? "" : " ") + w;
    if (lower.empty()) lower = word;
    return WordEmbeddings::Nearest(vectors, lower, std::max(1, k), error);
}

WordMap MapWords(const WordVectors& vectors, const std::vector<std::string>& words) {
    WordMap map;
    std::unordered_map<std::string, size_t> row;
    for (size_t i = 0; i < vectors.words.size(); ++i) row.emplace(vectors.words[i], i);
    std::vector<std::vector<double>> data;
    std::vector<std::string> kept;
    for (const std::string& w : words) {
        const auto it = row.find(w);
        if (it == row.end() || vectors.dim <= 0) continue;
        const float* v = vectors.values.data() + it->second * static_cast<size_t>(vectors.dim);
        data.emplace_back(v, v + vectors.dim);
        kept.push_back(w);
    }
    if (data.size() < 3) return map;
    const PCAResult pca = DimensionalityReduction::ComputePCA(data, 2);
    if (!pca.success || pca.transformed.size() != data.size()) return map;
    for (size_t i = 0; i < kept.size(); ++i)
        map.points.push_back({kept[i], pca.transformed[i][0], pca.transformed[i].size() > 1 ? pca.transformed[i][1] : 0.0});
    if (!pca.explained_variance_ratio.empty()) map.share_x = pca.explained_variance_ratio[0];
    if (pca.explained_variance_ratio.size() > 1) map.share_y = pca.explained_variance_ratio[1];
    return map;
}

// ---- The nodes

std::map<std::string, std::string> NodeParameters(const TermsRequest& request) {
    return {{"text_col", request.base.text},
            {"label_col", request.base.group},
            {"min_df", std::to_string(std::max(1, request.min_df))},
            {"max_features", "2000"},
            {"use_idf", "true"},
            {"smooth_idf", "true"},
            {"norm", request.norm},
            {"ngram_range", request.ngram_max >= 2 ? "1,2" : "1,1"},
            {"stop_words", request.stop_words ? "english" : "none"}};
}

std::map<std::string, std::string> NodeParameters(const SentimentRequest& request) {
    return {{"text_col", request.base.text}, {"label_col", request.base.group}};
}

std::map<std::string, std::string> NodeParameters(const TokensRequest& request) {
    return {{"tokenizer_type", std::to_string(request.tokenizer)},
            {"text_col", request.base.text},
            {"label_col", request.base.group},
            {"max_length", std::to_string(request.max_length)},
            {"min_word_freq", std::to_string(std::max(1, request.min_frequency))},
            {"max_vocab_size", std::to_string(request.vocabulary)}};
}

std::map<std::string, std::string> NodeParameters(const EmbeddingsRequest& request) {
    std::map<std::string, std::string> p = {{"source", request.method},
                                            {"text_col", request.base.text},
                                            {"vector_size", std::to_string(request.size)},
                                            {"min_count", std::to_string(std::max(1, request.min_count))},
                                            {"train_seed", std::to_string(request.seed)}};
    if (request.method == "glove") {
        if (request.window > 0) p["glove_window"] = std::to_string(request.window);
        if (request.epochs > 0) p["glove_epochs"] = std::to_string(request.epochs);
    } else {
        p["window"] = std::to_string(std::max(1, request.window));
        p["epochs"] = std::to_string(std::max(1, request.epochs));
    }
    return p;
}

}  // namespace cyxwiz::text
