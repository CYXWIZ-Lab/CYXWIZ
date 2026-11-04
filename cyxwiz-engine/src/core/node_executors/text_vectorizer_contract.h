#pragma once

#include <cyxwiz/text_processing.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <vector>

namespace cyxwiz::text_vectorizer_contract {

inline bool IsAsciiWhitespace(char value) {
    return value == ' ' || value == '\t' || value == '\r' || value == '\n';
}

inline std::string_view TrimAsciiWhitespace(std::string_view value) {
    while (!value.empty() && IsAsciiWhitespace(value.front())) {
        value.remove_prefix(1);
    }
    while (!value.empty() && IsAsciiWhitespace(value.back())) {
        value.remove_suffix(1);
    }
    return value;
}

inline bool ParsePositiveInteger(std::string_view raw_value,
                                 const std::string& operator_name,
                                 const std::string& field_name,
                                 int& output,
                                 std::string& error) {
    const std::string_view value = TrimAsciiWhitespace(raw_value);
    int parsed = 0;
    const auto result = std::from_chars(
        value.data(), value.data() + value.size(), parsed);
    if (value.empty() || result.ec != std::errc{} ||
        result.ptr != value.data() + value.size()) {
        error = operator_name + ": '" + field_name +
                "' is not a valid integer: " + std::string(raw_value);
        return false;
    }
    if (parsed < 1) {
        error = operator_name + ": '" + field_name +
                "' must be >= 1 (got " + std::to_string(parsed) + ")";
        return false;
    }
    output = parsed;
    return true;
}

inline bool ParseNGramRange(
    const std::map<std::string, std::string>& parameters,
    const std::string& operator_name,
    int& ngram_min,
    int& ngram_max,
    std::string& error) {
    ngram_min = 1;
    ngram_max = 1;

    const auto range = parameters.find("ngram_range");
    const bool has_range =
        range != parameters.end() && !range->second.empty();
    if (has_range) {
        const std::string_view value = range->second;
        const size_t delimiter = value.find_first_of(",;");
        if (delimiter == std::string_view::npos ||
            value.find_first_of(",;", delimiter + 1) !=
                std::string_view::npos) {
            error = operator_name +
                    ": ngram_range must be formatted as 'min,max' (got '" +
                    range->second + "')";
            return false;
        }
        if (!ParsePositiveInteger(value.substr(0, delimiter), operator_name,
                                  "ngram_range minimum", ngram_min, error) ||
            !ParsePositiveInteger(value.substr(delimiter + 1), operator_name,
                                  "ngram_range maximum", ngram_max, error)) {
            return false;
        }
    } else {
        const auto parse_legacy = [&](const char* key, int& output) {
            const auto value = parameters.find(key);
            return value == parameters.end() || value->second.empty() ||
                   ParsePositiveInteger(
                       value->second, operator_name, key, output, error);
        };
        if (!parse_legacy("ngram_min", ngram_min) ||
            !parse_legacy("ngram_max", ngram_max)) {
            return false;
        }
    }

    if (ngram_min > ngram_max) {
        error = operator_name + ": ngram_min must be <= ngram_max (got " +
                std::to_string(ngram_min) + "," +
                std::to_string(ngram_max) + ")";
        return false;
    }
    if (ngram_max > 3) {
        error = operator_name + ": ngram_max > 3 is not supported yet (got " +
                std::to_string(ngram_max) + ")";
        return false;
    }
    return true;
}

// scikit-learn's idf of a term in `df` of `rows` texts: ln(n / df) + 1, or
// smoothed ln((1 + n) / (1 + df)) + 1.
inline double Idf(double rows, double df, bool smooth) {
    return smooth ? std::log((rows + 1.0) / (df + 1.0)) + 1.0
                  : std::log(rows / df) + 1.0;
}

// The terms of a column (TextProcessing::Terms), each kept once as an id (in
// the order first read), and each row as its (id, count) pairs, ids
// ascending, in flat arrays: the memory grows with the terms read, not with a
// table of strings per row.
struct CountedTerms {
    std::vector<std::string> terms;      // by id
    std::vector<int64_t> doc_freq;       // the counted rows each term is in
    std::vector<int64_t> corpus_count;   // its count over those rows
    std::vector<int64_t> starts{0};      // row r's pairs: [starts[r], starts[r + 1])
    std::vector<int32_t> ids, counts;
};

// Counts every text; `counted(r)` says whether row r adds to doc_freq and
// corpus_count (the fit rows). `keep_going(rows_done)` is asked every 1,024
// rows and at the end; false stops the count, which then returns false.
template <class Counted, class KeepGoing>
bool CountTerms(const std::vector<std::string>& texts, bool english_stop_words,
                int ngram_min, int ngram_max, Counted counted,
                KeepGoing keep_going, CountedTerms& out) {
    out = {};
    out.starts.reserve(texts.size() + 1);
    std::unordered_map<std::string, int32_t> id_of;
    std::vector<int32_t> row;
    for (size_t r = 0; r < texts.size(); ++r) {
        if ((r & 1023) == 0 && !keep_going(r)) return false;
        row.clear();
        for (std::string& term : TextProcessing::Terms(
                 texts[r], english_stop_words, ngram_min, ngram_max)) {
            const auto [it, added] = id_of.try_emplace(
                std::move(term), static_cast<int32_t>(out.doc_freq.size()));
            if (added) {
                out.doc_freq.push_back(0);
                out.corpus_count.push_back(0);
            }
            row.push_back(it->second);
        }
        std::sort(row.begin(), row.end());
        const bool adds = counted(r);
        for (size_t k = 0; k < row.size(); ++k) {
            const size_t id = static_cast<size_t>(row[k]);
            if (adds) ++out.corpus_count[id];
            if (k > 0 && row[k] == row[k - 1]) {
                ++out.counts.back();
                continue;
            }
            out.ids.push_back(row[k]);
            out.counts.push_back(1);
            if (adds) ++out.doc_freq[id];
        }
        out.starts.push_back(static_cast<int64_t>(out.ids.size()));
    }
    out.terms.resize(out.doc_freq.size());
    for (auto it = id_of.begin(); it != id_of.end();) {
        auto node = id_of.extract(it++);
        out.terms[static_cast<size_t>(node.mapped())] = std::move(node.key());
    }
    return keep_going(texts.size());
}

// Each id's column among `kept` (the terms kept, in column order), or -1.
inline std::vector<int32_t> ColumnsOf(const CountedTerms& counted,
                                      const std::vector<std::string>& kept) {
    std::unordered_map<std::string_view, int32_t> column;
    column.reserve(kept.size());
    for (size_t j = 0; j < kept.size(); ++j)
        column.emplace(kept[j], static_cast<int32_t>(j));
    std::vector<int32_t> out(counted.terms.size(), -1);
    for (size_t id = 0; id < counted.terms.size(); ++id) {
        const auto it = column.find(counted.terms[id]);
        if (it != column.end()) out[id] = it->second;
    }
    return out;
}

}  // namespace cyxwiz::text_vectorizer_contract
