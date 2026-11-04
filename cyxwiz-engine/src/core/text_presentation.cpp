#include "text_presentation.h"

#include "explore_presentation.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace cyxwiz::text_view {
namespace {

using explore_view::Figure;
using explore_view::Number;

std::string N(int64_t n) { return Number(static_cast<size_t>(std::max<int64_t>(0, n))); }

std::string Percent(double share) {
    char buf[32];
    std::snprintf(buf, sizeof buf, share > 0.9 || share < 0.1 ? "%.1f%%" : "%.0f%%", share * 100.0);
    return buf;
}

std::string Fixed(double v, int digits) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.*f", digits, v);
    return std::string(buf) == "-0.00" ? "0.00" : buf;
}

std::string Quoted(const std::string& s) { return "\"" + s + "\""; }

// "a, b and c".
std::string List(const std::vector<std::string>& items) {
    std::string out;
    for (size_t i = 0; i < items.size(); ++i) {
        if (i > 0) out += i + 1 == items.size() ? " and " : ", ";
        out += items[i];
    }
    return out;
}

}  // namespace

std::string RowsWords(const text::TextData& d) {
    const int64_t used = static_cast<int64_t>(d.texts.size());
    const int64_t left = d.missing + d.empty;
    if (left == 0) return N(used) + " rows, all with text";
    return N(used) + " rows with text (" + N(left) + (d.empty == 0 ? " missing" : d.missing == 0 ? " empty" : " empty or missing") + " left out)";
}

std::string GroupsWords(const text::TextData& d, const std::string& group_column) {
    if (d.groups.empty()) return "";
    if (d.groups_folded == 0) return explore_view::Count(d.groups.size(), "group", "groups") + " in " + group_column;
    const size_t named = d.groups.size() - 1;
    return explore_view::Count(named + d.groups_folded, "group", "groups") + " in " + group_column + " (the " + Number(named) +
           " largest, " + Number(d.groups_folded) + " more as other)";
}

std::string LengthWords(const text::LengthFigures& f, const char* unit) {
    return "median " + Figure(f.median) + " " + unit + " per row (95% within " + Figure(std::round(f.p95)) + ", longest " + N(f.max) + ")";
}

std::string WordsSentence(const text::WordsAnswer& a) {
    if (a.top.empty()) return "No word is left to count.";
    const bool pairs = a.request.pairs;
    std::vector<std::string> named;
    for (size_t i = 0; i < a.top.size() && i < 6; ++i) named.push_back(a.top[i].term);
    std::string s = std::string(pairs ? "The commonest pairs are " : "The column talks most about ") + List(named) + ": " +
                    Quoted(a.top[0].term) + " " + N(a.top[0].count) + " times";
    if (a.top.size() > 1) s += ", " + Quoted(a.top[1].term) + " " + N(a.top[1].count);
    s += ".";
    if (!pairs && !a.pairs.empty()) s += " The commonest pair is " + Quoted(a.pairs[0].term) + " (" + N(a.pairs[0].count) + ").";
    // Each group's own word: its commonest that is not among the table's five commonest.
    if (!a.by_group.empty()) {
        std::vector<std::string> own;
        for (size_t g = 0; g < a.by_group.size() && g < a.data.groups.size(); ++g) {
            if (a.data.groups[g] == "other") continue;
            for (const text::Count& c : a.by_group[g]) {
                bool common = false;
                for (size_t k = 0; k < a.top.size() && k < 5; ++k) common = common || a.top[k].term == c.term;
                if (!common) {
                    own.push_back(a.data.groups[g] + " " + Quoted(c.term));
                    break;
                }
            }
        }
        if (!own.empty()) s += " Each group's own " + std::string(pairs ? "pair" : "word") + ": " + List(own) + ".";
    }
    const text::LengthFigures& f = a.per_row;
    s += " Rows run to " + Figure(f.median) + " words on the median (95% within " + Figure(std::round(f.p95)) + ", the longest " + N(f.max) + ")";
    if (f.over > 0) s += "; " + N(f.over) + " rows run past " + N(f.cap) + " words";
    return s + ".";
}

std::string TermsSentence(const text::TermsAnswer& a) {
    std::string s;
    if (!a.distinctive.empty()) {
        std::vector<std::string> parts;
        for (size_t g = 0; g < a.distinctive.size() && g < a.data.groups.size(); ++g) {
            if (a.data.groups[g] == "other") continue;
            std::vector<std::string> terms;
            for (size_t k = 0; k < a.distinctive[g].size() && k < 4; ++k) terms.push_back(a.distinctive[g][k].term);
            if (!terms.empty()) parts.push_back(a.data.groups[g] + " by " + List(terms));
        }
        s = "The terms that set each group apart (its mean TF-IDF above the table's): ";
        for (size_t i = 0; i < parts.size(); ++i) s += (i > 0 ? "; " : "") + parts[i];
        s += ".";
    } else if (!a.overall.empty()) {
        std::vector<std::string> terms;
        for (size_t k = 0; k < a.overall.size() && k < 6; ++k) terms.push_back(a.overall[k].term);
        s = "The terms that weigh most across the rows (mean TF-IDF): " + List(terms) + ".";
    }
    if (!a.nearest.empty())
        s += " Row " + N(a.request.row + 1) + " reads most like row " + N(a.nearest[0].row + 1) + " (" + Fixed(a.nearest[0].similarity, 2) + ").";
    std::vector<std::string> quality;
    if (a.html_entities > 0) quality.push_back(N(a.html_entities) + " rows carry HTML entities (&quot;)");
    if (a.links > 0) quality.push_back(N(a.links) + " hold links");
    if (a.repeated > 0) quality.push_back(N(a.repeated) + " repeat an earlier row");
    if (a.broken > 0) quality.push_back(N(a.broken) + " hold broken characters (text saved in one encoding and read in another)");
    if (!quality.empty()) s += " " + List(quality) + (a.html_entities > 0 ? ": Text Clean with remove HTML takes the entities out first." : ".");
    return s;
}

const char* SentimentLabel(double compound, const text::SentimentRequest& request) {
    if (compound >= request.positive_from) return "positive";
    if (compound <= request.negative_below) return "negative";
    return "neutral";
}

std::string SentimentSentence(const text::SentimentAnswer& a) {
    const double total = static_cast<double>(a.data.texts.size());
    const double neg = static_cast<double>(a.negative) / total, pos = static_cast<double>(a.positive) / total;
    std::string s;
    if (neg > 0.5) s = "Most rows read negative (" + Percent(neg) + ")";
    else if (pos > 0.5) s = "Most rows read positive (" + Percent(pos) + ")";
    else s = "No side holds most rows: " + Percent(neg) + " negative, " + Percent(pos) + " positive";
    s += "; the mean compound score is " + Fixed(a.mean, 2) + ".";
    if (a.by_group.size() >= 2) {
        std::vector<text::GroupSentiment> order;
        for (const auto& g : a.by_group)
            if (g.group != "other" && g.rows > 0) order.push_back(g);
        std::sort(order.begin(), order.end(), [](const auto& x, const auto& y) { return x.median < y.median; });
        if (order.size() >= 2) {
            std::vector<std::string> named;
            for (const auto& g : order) named.push_back(g.group + " (" + Fixed(g.median, 2) + ")");
            s += " By group, from the most negative median: " + List(named) + ".";
        }
    }
    return s;
}

std::string TokensSentence(const text::TokensAnswer& a) {
    const int limit = a.request.max_length;
    std::string s;
    if (a.cut_rows == 0) s = "Every row fits in " + N(limit) + " tokens (the longest has " + N(a.per_row.max) + ").";
    else
        s = "At " + N(limit) + " tokens, " + N(a.cut_rows) + " rows (" + Percent(a.cut_share) + ") lose their end; " + N(2LL * limit) +
            " would cut " + Percent(a.cut_share_double) + ".";
    s += " A vocabulary of " + Number(a.vocabulary_size) + " tokens covers " + Percent(a.coverage) + " of the text";
    if (a.coverage < 1.0) s += ": the other " + Percent(1.0 - a.coverage) + " become the unknown token";
    return s + ".";
}

std::string EmbeddingsSentence(const text::EmbeddingsAnswer& a, const std::string& word,
                               const std::vector<std::pair<std::string, double>>& nearest) {
    std::string s = "Learned from " + (a.sampled ? N(a.rows_used) + " of the column's " + Number(a.data.texts.size()) + " rows" : std::string("this column alone")) + " (" +
                    Number(a.vectors.words.size()) + " words seen " + std::to_string(a.request.min_count) + " times or more)";
    if (nearest.empty()) return s + (word.empty() ? "." : ", " + Quoted(word) + " has no vector: it is not seen often enough.");
    std::vector<std::string> alike;
    for (size_t i = 0; i < nearest.size() && i < 3; ++i) alike.push_back(nearest[i].first + " (" + Fixed(nearest[i].second, 2) + ")");
    return s + ", " + Quoted(word) + " sits nearest to " + List(alike) + ".";
}

std::string WordsCsv(const text::WordsAnswer& a) {
    std::vector<std::vector<std::string>> rows = {{"rank", a.request.pairs ? "pair" : "word", "count", "share"}};
    for (size_t i = 0; i < a.top.size(); ++i)
        rows.push_back({std::to_string(i + 1), a.top[i].term, std::to_string(a.top[i].count),
                        Figure(a.kept > 0 ? static_cast<double>(a.top[i].count) / static_cast<double>(a.kept) : 0.0)});
    return explore_view::Csv(rows);
}

std::string TermsCsv(const text::TermsAnswer& a) {
    std::vector<std::vector<std::string>> rows = {{"group", "term", "mean_tfidf", "above_table"}};
    if (a.distinctive.empty())
        for (const auto& t : a.overall) rows.push_back({"", t.term, Figure(t.score), ""});
    for (size_t g = 0; g < a.distinctive.size() && g < a.data.groups.size(); ++g)
        for (const auto& t : a.distinctive[g]) rows.push_back({a.data.groups[g], t.term, Figure(t.score), Figure(t.lift)});
    return explore_view::Csv(rows);
}

std::string SentimentCsv(const text::SentimentAnswer& a) {
    std::vector<std::vector<std::string>> rows = {{"row", "group", "compound", "sentiment"}};
    for (size_t i = 0; i < a.compound.size(); ++i)
        rows.push_back({std::to_string(a.data.rows[i] + 1), a.data.group_of.empty() ? "" : a.data.groups[static_cast<size_t>(a.data.group_of[i])],
                        Figure(a.compound[i]), SentimentLabel(a.compound[i], a.request)});
    return explore_view::Csv(rows);
}

}  // namespace cyxwiz::text_view
