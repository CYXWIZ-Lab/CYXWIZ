// The word rule, the English stop words and the Porter stemmer
// (TOFIX129 D8). Checked against Python, sklearn's ENGLISH_STOP_WORDS and
// NLTK's PorterStemmer(ORIGINAL_ALGORITHM) by test_computation_truth_text.

#include "cyxwiz/text_processing.h"
#include "text_unicode.h"

#include <functional>
#include <limits>
#include <stdexcept>

namespace cyxwiz {
namespace {

namespace tu = text_unicode;
using U32 = std::u32string;

// ---- Porter (1980), as NLTK's ORIGINAL_ALGORITHM mode --------------------

bool IsVowel(char32_t c) { return c == U'a' || c == U'e' || c == U'i' || c == U'o' || c == U'u'; }

bool IsConsonant(const U32& w, size_t i) {
    if (IsVowel(w[i])) return false;
    if (w[i] == U'y') {
        bool negate = false;
        while (i > 0 && w[i] == U'y') {
            negate = !negate;
            --i;
        }
        return !IsVowel(w[i]) != negate;
    }
    return true;
}

// A 'y' is a consonant when the letter before it is not one (or it starts
// the word): one pass instead of IsConsonant's walk back over a run of 'y's.
std::vector<bool> ConsonantFlags(const U32& w) {
    std::vector<bool> flags(w.size());
    for (size_t i = 0; i < w.size(); ++i) {
        if (IsVowel(w[i])) flags[i] = false;
        else if (w[i] == U'y') flags[i] = i == 0 ? true : !flags[i - 1];
        else flags[i] = true;
    }
    return flags;
}

int Measure(const U32& stem) {
    const std::vector<bool> flags = ConsonantFlags(stem);
    int m = 0;
    for (size_t i = 1; i < flags.size(); ++i)
        if (!flags[i - 1] && flags[i]) ++m;
    return m;
}

bool ContainsVowel(const U32& stem) {
    for (const bool consonant : ConsonantFlags(stem))
        if (!consonant) return true;
    return false;
}

bool EndsDoubleConsonant(const U32& w) {
    return w.size() >= 2 && w[w.size() - 1] == w[w.size() - 2] && IsConsonant(w, w.size() - 1);
}

bool EndsCvc(const U32& w) {
    if (w.size() < 3) return false;
    const char32_t last = w.back();
    return IsConsonant(w, w.size() - 3) && !IsConsonant(w, w.size() - 2) && IsConsonant(w, w.size() - 1) &&
           last != U'w' && last != U'x' && last != U'y';
}

bool EndsWith(const U32& w, const U32& suffix) {
    return w.size() >= suffix.size() && w.compare(w.size() - suffix.size(), suffix.size(), suffix) == 0;
}

struct Rule {
    U32 suffix;  // U"*d": the word ends in a double consonant
    U32 replacement;
    std::function<bool(const U32&)> condition;  // empty: always
};

// The first rule whose suffix the word ends with decides: its replacement
// when its condition holds, else the word unchanged.
U32 ApplyRules(const U32& w, const std::vector<Rule>& rules) {
    for (const Rule& r : rules) {
        if (r.suffix == U"*d") {
            if (EndsDoubleConsonant(w)) {
                const U32 stem = w.substr(0, w.size() - 2);
                return !r.condition || r.condition(stem) ? stem + r.replacement : w;
            }
            continue;
        }
        if (EndsWith(w, r.suffix)) {
            const U32 stem = w.substr(0, w.size() - r.suffix.size());
            return !r.condition || r.condition(stem) ? stem + r.replacement : w;
        }
    }
    return w;
}

bool PositiveMeasure(const U32& stem) { return Measure(stem) > 0; }
bool MeasureAboveOne(const U32& stem) { return Measure(stem) > 1; }

U32 Step1a(const U32& w) {
    return ApplyRules(w, {{U"sses", U"ss", {}}, {U"ies", U"i", {}}, {U"ss", U"ss", {}}, {U"s", U"", {}}});
}

U32 Step1b(const U32& w) {
    if (EndsWith(w, U"eed")) {
        const U32 stem = w.substr(0, w.size() - 3);
        return Measure(stem) > 0 ? stem + U"ee" : w;
    }
    U32 intermediate;
    bool removed = false;
    for (const U32& suffix : {U32(U"ed"), U32(U"ing")}) {
        if (EndsWith(w, suffix)) {
            intermediate = w.substr(0, w.size() - suffix.size());
            if (ContainsVowel(intermediate)) {
                removed = true;
                break;
            }
        }
    }
    if (!removed) return w;
    const char32_t last = intermediate.back();
    return ApplyRules(intermediate,
                      {{U"at", U"ate", {}},
                       {U"bl", U"ble", {}},
                       {U"iz", U"ize", {}},
                       {U"*d", U32(1, last), [last](const U32&) { return last != U'l' && last != U's' && last != U'z'; }},
                       {U"", U"e", [](const U32& stem) { return Measure(stem) == 1 && EndsCvc(stem); }}});
}

U32 Step1c(const U32& w) { return ApplyRules(w, {{U"y", U"i", ContainsVowel}}); }

U32 Step2(const U32& w) {
    return ApplyRules(w, {{U"ational", U"ate", PositiveMeasure}, {U"tional", U"tion", PositiveMeasure},
                          {U"enci", U"ence", PositiveMeasure},   {U"anci", U"ance", PositiveMeasure},
                          {U"izer", U"ize", PositiveMeasure},    {U"abli", U"able", PositiveMeasure},
                          {U"alli", U"al", PositiveMeasure},     {U"entli", U"ent", PositiveMeasure},
                          {U"eli", U"e", PositiveMeasure},       {U"ousli", U"ous", PositiveMeasure},
                          {U"ization", U"ize", PositiveMeasure}, {U"ation", U"ate", PositiveMeasure},
                          {U"ator", U"ate", PositiveMeasure},    {U"alism", U"al", PositiveMeasure},
                          {U"iveness", U"ive", PositiveMeasure}, {U"fulness", U"ful", PositiveMeasure},
                          {U"ousness", U"ous", PositiveMeasure}, {U"aliti", U"al", PositiveMeasure},
                          {U"iviti", U"ive", PositiveMeasure},   {U"biliti", U"ble", PositiveMeasure}});
}

U32 Step3(const U32& w) {
    return ApplyRules(w, {{U"icate", U"ic", PositiveMeasure}, {U"ative", U"", PositiveMeasure},
                          {U"alize", U"al", PositiveMeasure}, {U"iciti", U"ic", PositiveMeasure},
                          {U"ical", U"ic", PositiveMeasure},  {U"ful", U"", PositiveMeasure},
                          {U"ness", U"", PositiveMeasure}});
}

U32 Step4(const U32& w) {
    return ApplyRules(w, {{U"al", U"", MeasureAboveOne},
                          {U"ance", U"", MeasureAboveOne},
                          {U"ence", U"", MeasureAboveOne},
                          {U"er", U"", MeasureAboveOne},
                          {U"ic", U"", MeasureAboveOne},
                          {U"able", U"", MeasureAboveOne},
                          {U"ible", U"", MeasureAboveOne},
                          {U"ant", U"", MeasureAboveOne},
                          {U"ement", U"", MeasureAboveOne},
                          {U"ment", U"", MeasureAboveOne},
                          {U"ent", U"", MeasureAboveOne},
                          {U"ion", U"", [](const U32& stem) { return Measure(stem) > 1 && (stem.back() == U's' || stem.back() == U't'); }},
                          {U"ou", U"", MeasureAboveOne},
                          {U"ism", U"", MeasureAboveOne},
                          {U"ate", U"", MeasureAboveOne},
                          {U"iti", U"", MeasureAboveOne},
                          {U"ous", U"", MeasureAboveOne},
                          {U"ive", U"", MeasureAboveOne},
                          {U"ize", U"", MeasureAboveOne}});
}

U32 Step5a(const U32& w) {
    if (EndsWith(w, U"e")) {
        const U32 stem = w.substr(0, w.size() - 1);
        if (Measure(stem) > 1) return stem;
        if (Measure(stem) == 1 && !EndsCvc(stem)) return stem;
    }
    return w;
}

U32 Step5b(const U32& w) {
    return ApplyRules(w, {{U"ll", U"l", [&w](const U32&) { return Measure(w.substr(0, w.size() - 1)) > 1; }}});
}

}  // namespace

std::vector<std::string> TextProcessing::Words(const std::string& text) {
    U32 s = tu::Decode(text);
    for (char32_t& cp : s)
        if (cp == U'’') cp = U'\'';
    std::vector<std::string> words;
    size_t i = 0;
    while (i < s.size()) {
        if (!tu::IsAlnum(s[i])) {
            ++i;
            continue;
        }
        const size_t start = i;
        while (i < s.size() && tu::IsAlnum(s[i])) ++i;
        while (i + 1 < s.size() && s[i] == U'\'' && tu::IsAlnum(s[i + 1])) {
            ++i;
            while (i < s.size() && tu::IsAlnum(s[i])) ++i;
        }
        if (i - start >= 2) words.push_back(tu::Encode(tu::LowerString(s.substr(start, i - start))));
    }
    return words;
}

const std::set<std::string>& TextProcessing::GetStopwords() {
    static const std::set<std::string> words = {
#include "text_stop_words.inc"
    };
    return words;
}

std::vector<std::string> TextProcessing::RemoveStopwords(const std::vector<std::string>& tokens,
                                                         const std::string& /*language*/) {
    const std::set<std::string>& stop = GetStopwords();
    std::vector<std::string> kept;
    kept.reserve(tokens.size());
    for (const std::string& token : tokens)
        if (stop.find(tu::Encode(tu::LowerString(tu::Decode(token)))) == stop.end()) kept.push_back(token);
    return kept;
}

std::vector<std::string> TextProcessing::NGrams(const std::vector<std::string>& tokens, int ngram_min, int ngram_max) {
    std::vector<std::string> grams;
    if (tokens.empty() || ngram_min < 1 || ngram_max < ngram_min) return grams;
    size_t total = 0;
    for (int n = ngram_min; n <= ngram_max; ++n) {
        const size_t width = static_cast<size_t>(n);
        if (tokens.size() < width) continue;
        const size_t count = tokens.size() - width + 1;
        if (total > std::numeric_limits<size_t>::max() - count) throw std::length_error("text n-gram count overflow");
        total += count;
    }
    grams.reserve(total);
    for (int n = ngram_min; n <= ngram_max; ++n) {
        const size_t width = static_cast<size_t>(n);
        for (size_t start = 0; start + width <= tokens.size(); ++start) {
            std::string gram = tokens[start];
            for (size_t k = 1; k < width; ++k) gram += ' ' + tokens[start + k];
            grams.push_back(std::move(gram));
        }
    }
    return grams;
}

std::vector<std::string> TextProcessing::Terms(const std::string& text, bool english_stop_words, int ngram_min, int ngram_max) {
    std::vector<std::string> words = Words(text);
    if (english_stop_words) words = RemoveStopwords(words, "english");
    return NGrams(words, ngram_min, ngram_max);
}

std::string TextProcessing::Stem(const std::string& word) {
    U32 w = tu::LowerString(tu::Decode(word));
    w = Step1a(w);
    w = Step1b(w);
    w = Step1c(w);
    w = Step2(w);
    w = Step3(w);
    w = Step4(w);
    w = Step5a(w);
    w = Step5b(w);
    return tu::Encode(w);
}

std::vector<std::string> TextProcessing::StemWords(const std::vector<std::string>& words) {
    std::vector<std::string> stems;
    stems.reserve(words.size());
    for (const std::string& word : words) stems.push_back(Stem(word));
    return stems;
}

}  // namespace cyxwiz
