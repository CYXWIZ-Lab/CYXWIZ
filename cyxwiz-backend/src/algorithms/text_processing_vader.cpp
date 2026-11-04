// VADER sentiment (Hutto and Gilbert, 2014), ported from vaderSentiment
// 3.3.2 rule for rule, its quirks included (the "but" rule finds each score
// by value, as Python's list.index does), so the scores are that package's
// (TOFIX129 D8; test_computation_truth_text checks them). The lexicon and
// the emoji descriptions are vaderSentiment's own (MIT, vader_lexicon.inc).

#include "cyxwiz/text_processing.h"
#include "text_unicode.h"

#include <array>
#include <cmath>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace cyxwiz {
namespace {

namespace tu = text_unicode;
using U32 = std::u32string;

constexpr double kBIncr = 0.293;
constexpr double kBDecr = -0.293;
constexpr double kCIncr = 0.733;
constexpr double kNScalar = -0.74;

struct WordValence {
    const char* word;
    double valence;
};
struct EmojiDescription {
    char32_t emoji;
    const char* description;
};

constexpr WordValence kLexicon[] = {
#define CYXWIZ_VADER_WORDS
#include "vader_lexicon.inc"
#undef CYXWIZ_VADER_WORDS
};
constexpr EmojiDescription kEmojis[] = {
#define CYXWIZ_VADER_EMOJIS
#include "vader_lexicon.inc"
#undef CYXWIZ_VADER_EMOJIS
};

struct Tables {
    std::unordered_map<std::string, double> lexicon;
    std::unordered_map<char32_t, U32> emojis;
    std::unordered_set<std::string> negate;
    std::unordered_map<std::string, double> booster;
    std::unordered_map<std::string, double> special;
};

const Tables& VaderTables() {
    static const Tables tables = [] {
        Tables t;
        t.lexicon.reserve(std::size(kLexicon));
        for (const WordValence& w : kLexicon) t.lexicon.insert_or_assign(w.word, w.valence);  // a later line wins, as in a Python dict
        for (const EmojiDescription& e : kEmojis) t.emojis.insert_or_assign(e.emoji, tu::Decode(e.description));
        t.negate = {"aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt", "ain't", "aren't",
                    "can't", "couldn't", "daren't", "didn't", "doesn't", "dont", "hadnt", "hasnt", "havent", "isnt",
                    "mightnt", "mustnt", "neither", "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't",
                    "mustn't", "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
                    "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent", "oughtn't", "shan't", "shouldn't",
                    "uh-uh", "wasn't", "weren't", "without", "wont", "wouldnt", "won't", "wouldn't", "rarely",
                    "seldom", "despite"};
        for (const char* w : {"absolutely", "amazingly", "awfully", "completely", "considerable", "considerably",
                              "decidedly", "deeply", "effing", "enormous", "enormously", "entirely", "especially",
                              "exceptional", "exceptionally", "extreme", "extremely", "fabulously", "flipping",
                              "flippin", "frackin", "fracking", "fricking", "frickin", "frigging", "friggin", "fully",
                              "fuckin", "fucking", "fuggin", "fugging", "greatly", "hella", "highly", "hugely",
                              "incredible", "incredibly", "intensely", "major", "majorly", "more", "most",
                              "particularly", "purely", "quite", "really", "remarkably", "so", "substantially",
                              "thoroughly", "total", "totally", "tremendous", "tremendously", "uber", "unbelievably",
                              "unusually", "utter", "utterly", "very"})
            t.booster.emplace(w, kBIncr);
        for (const char* w : {"almost", "barely", "hardly", "just enough", "kind of", "kinda", "kindof", "kind-of",
                              "less", "little", "marginal", "marginally", "occasional", "occasionally", "partly",
                              "scarce", "scarcely", "slight", "slightly", "somewhat", "sort of", "sorta", "sortof",
                              "sort-of"})
            t.booster.emplace(w, kBDecr);
        t.special = {{"the shit", 3.0},      {"the bomb", 3.0},       {"bad ass", 1.5},    {"badass", 1.5},
                     {"bus stop", 0.0},      {"yeah right", -2.0},    {"kiss of death", -1.5},
                     {"to die for", 3.0},    {"beating heart", 3.5}};
        return t;
    }();
    return tables;
}

// ---- The words ----------------------------------------------------------

constexpr std::string_view kPunctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

bool IsPunctuation(char32_t c) { return c < 128 && kPunctuation.find(static_cast<char>(c)) != std::string_view::npos; }

// str.strip(string.punctuation); a word left with two characters or fewer
// was an emoticon and keeps its marks.
U32 StripPunctuationIfWord(const U32& token) {
    size_t a = 0, b = token.size();
    while (a < b && IsPunctuation(token[a])) ++a;
    while (b > a && IsPunctuation(token[b - 1])) --b;
    if (b - a <= 2) return token;
    return token.substr(a, b - a);
}

U32 StripSpace(const U32& s) {
    size_t a = 0, b = s.size();
    while (a < b && tu::IsSpace(s[a])) ++a;
    while (b > a && tu::IsSpace(s[b - 1])) --b;
    return s.substr(a, b - a);
}

std::vector<U32> SplitSpace(const U32& s) {
    std::vector<U32> parts;
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && tu::IsSpace(s[i])) ++i;
        const size_t start = i;
        while (i < s.size() && !tu::IsSpace(s[i])) ++i;
        if (i > start) parts.push_back(s.substr(start, i - start));
    }
    return parts;
}

struct Text {
    std::vector<std::string> words;  // each word or emoticon, as written (UTF-8)
    std::vector<std::string> lower;  // the same, lower case
    std::vector<bool> upper;         // str.isupper of each
    bool cap_differential = false;   // some, not all, are in capitals
};

Text ReadText(const U32& text) {
    Text t;
    for (const U32& token : SplitSpace(text)) {
        const U32 w = StripPunctuationIfWord(token);
        t.words.push_back(tu::Encode(w));
        t.lower.push_back(tu::Encode(tu::LowerString(w)));
        t.upper.push_back(tu::IsUpperString(w));
    }
    size_t caps = 0;
    for (const bool u : t.upper) caps += u ? 1 : 0;
    const size_t differential = t.words.size() - caps;
    t.cap_differential = differential > 0 && differential < t.words.size();
    return t;
}

// ---- The rules ----------------------------------------------------------

bool Negated(const std::string& lower_word) {
    return VaderTables().negate.count(lower_word) > 0 || lower_word.find("n't") != std::string::npos;
}

double ScalarIncDec(const Text& t, size_t j, double valence) {
    const auto& booster = VaderTables().booster;
    const auto it = booster.find(t.lower[j]);
    if (it == booster.end()) return 0.0;
    double scalar = it->second;
    if (valence < 0) scalar *= -1;
    if (t.upper[j] && t.cap_differential) {
        if (valence > 0) scalar += kCIncr;
        else scalar -= kCIncr;
    }
    return scalar;
}

double NegationCheck(double valence, const Text& t, int start_i, size_t i) {
    const auto& w = t.lower;
    if (start_i == 0) {
        if (Negated(w[i - 1])) valence *= kNScalar;
    }
    if (start_i == 1) {
        if (w[i - 2] == "never" && (w[i - 1] == "so" || w[i - 1] == "this")) valence *= 1.25;
        else if (w[i - 2] == "without" && w[i - 1] == "doubt") {
        } else if (Negated(w[i - 2])) valence *= kNScalar;
    }
    if (start_i == 2) {
        if ((w[i - 3] == "never" && (w[i - 2] == "so" || w[i - 2] == "this")) || (w[i - 1] == "so" || w[i - 1] == "this"))
            valence *= 1.25;
        else if (w[i - 3] == "without" && (w[i - 2] == "doubt" || w[i - 1] == "doubt")) {
        } else if (Negated(w[i - 3])) valence *= kNScalar;
    }
    return valence;
}

double SpecialIdiomsCheck(double valence, const Text& t, size_t i) {
    const auto& w = t.lower;
    const auto& special = VaderTables().special;
    const std::string onezero = w[i - 1] + " " + w[i];
    const std::string twoonezero = w[i - 2] + " " + w[i - 1] + " " + w[i];
    const std::string twoone = w[i - 2] + " " + w[i - 1];
    const std::string threetwoone = w[i - 3] + " " + w[i - 2] + " " + w[i - 1];
    const std::string threetwo = w[i - 3] + " " + w[i - 2];
    for (const std::string* seq : {&onezero, &twoonezero, &twoone, &threetwoone, &threetwo}) {
        const auto it = special.find(*seq);
        if (it != special.end()) {
            valence = it->second;
            break;
        }
    }
    if (w.size() - 1 > i) {
        const auto it = special.find(w[i] + " " + w[i + 1]);
        if (it != special.end()) valence = it->second;
    }
    if (w.size() - 1 > i + 1) {
        const auto it = special.find(w[i] + " " + w[i + 1] + " " + w[i + 2]);
        if (it != special.end()) valence = it->second;
    }
    const auto& booster = VaderTables().booster;
    for (const std::string* gram : {&threetwoone, &threetwo, &twoone}) {
        const auto it = booster.find(*gram);
        if (it != booster.end()) valence = valence + it->second;
    }
    return valence;
}

double LeastCheck(double valence, const Text& t, size_t i) {
    const auto& w = t.lower;
    const auto& lexicon = VaderTables().lexicon;
    if (i > 1 && lexicon.count(w[i - 1]) == 0 && w[i - 1] == "least") {
        if (w[i - 2] != "at" && w[i - 2] != "very") valence *= kNScalar;
    } else if (i > 0 && lexicon.count(w[i - 1]) == 0 && w[i - 1] == "least") {
        valence *= kNScalar;
    }
    return valence;
}

double SentimentValence(const Text& t, size_t i) {
    const auto& lexicon = VaderTables().lexicon;
    const auto& w = t.lower;
    const auto it = lexicon.find(w[i]);
    if (it == lexicon.end()) return 0.0;
    double valence = it->second;
    if (w[i] == "no" && i != w.size() - 1 && lexicon.count(w[i + 1]) > 0) valence = 0.0;
    if ((i > 0 && w[i - 1] == "no") || (i > 1 && w[i - 2] == "no") ||
        (i > 2 && w[i - 3] == "no" && (w[i - 1] == "or" || w[i - 1] == "nor")))
        valence = it->second * kNScalar;
    if (t.upper[i] && t.cap_differential) {
        if (valence > 0) valence += kCIncr;
        else valence -= kCIncr;
    }
    for (int start_i = 0; start_i < 3; ++start_i) {
        const size_t back = static_cast<size_t>(start_i) + 1;
        if (i > static_cast<size_t>(start_i) && lexicon.count(w[i - back]) == 0) {
            double s = ScalarIncDec(t, i - back, valence);
            if (start_i == 1 && s != 0) s = s * 0.95;
            if (start_i == 2 && s != 0) s = s * 0.9;
            valence = valence + s;
            valence = NegationCheck(valence, t, start_i, i);
            if (start_i == 2) valence = SpecialIdiomsCheck(valence, t, i);
        }
    }
    return LeastCheck(valence, t, i);
}

// "but": the scores before it count half and those after it half again.
// vaderSentiment finds each score by value (list.index), so of two equal
// scores the first is changed twice; kept, so the scores stay the package's.
void ButCheck(const Text& t, std::vector<double>& sentiments) {
    size_t bi = t.lower.size();
    for (size_t j = 0; j < t.lower.size(); ++j)
        if (t.lower[j] == "but") {
            bi = j;
            break;
        }
    if (bi == t.lower.size()) return;
    for (size_t k = 0; k < sentiments.size(); ++k) {
        const double v = sentiments[k];
        size_t si = 0;
        while (sentiments[si] != v) ++si;
        if (si < bi) sentiments[si] = v * 0.5;
        else if (si > bi) sentiments[si] = v * 1.5;
    }
}

double PunctuationEmphasis(const U32& text) {
    size_t ep = 0, qm = 0;
    for (const char32_t c : text) {
        if (c == U'!') ++ep;
        if (c == U'?') ++qm;
    }
    if (ep > 4) ep = 4;
    const double ep_amplifier = static_cast<double>(ep) * 0.292;
    double qm_amplifier = 0;
    if (qm > 1) qm_amplifier = qm <= 3 ? static_cast<double>(qm) * 0.18 : 0.96;
    return ep_amplifier + qm_amplifier;
}

double Normalize(double score) {
    const double norm = score / std::sqrt(score * score + 15.0);
    if (norm < -1.0) return -1.0;
    if (norm > 1.0) return 1.0;
    return norm;
}

}  // namespace

VaderScores TextProcessing::Vader(const std::string& text) {
    const Tables& tables = VaderTables();
    // Emojis read as their descriptions.
    const U32 in = tu::Decode(text);
    U32 replaced;
    replaced.reserve(in.size());
    bool prev_space = true;
    for (const char32_t c : in) {
        const auto it = tables.emojis.find(c);
        if (it != tables.emojis.end()) {
            if (!prev_space) replaced.push_back(U' ');
            replaced += it->second;
            prev_space = false;
        } else {
            replaced.push_back(c);
            prev_space = c == U' ';
        }
    }
    const U32 stripped = StripSpace(replaced);
    const Text t = ReadText(stripped);

    std::vector<double> sentiments;
    sentiments.reserve(t.words.size());
    for (size_t i = 0; i < t.words.size(); ++i) {
        if (tables.booster.count(t.lower[i]) > 0) {
            sentiments.push_back(0.0);
            continue;
        }
        if (i < t.words.size() - 1 && t.lower[i] == "kind" && t.lower[i + 1] == "of") {
            sentiments.push_back(0.0);
            continue;
        }
        sentiments.push_back(SentimentValence(t, i));
    }
    ButCheck(t, sentiments);

    VaderScores scores;
    scores.words.reserve(t.words.size());
    for (size_t i = 0; i < t.words.size(); ++i) scores.words.emplace_back(t.words[i], sentiments[i]);
    if (sentiments.empty()) return scores;

    double sum = 0.0;
    for (const double s : sentiments) sum += s;
    const double emphasis = PunctuationEmphasis(stripped);
    if (sum > 0) sum += emphasis;
    else if (sum < 0) sum -= emphasis;
    scores.compound = Normalize(sum);

    double pos_sum = 0.0, neg_sum = 0.0;
    size_t neu_count = 0;
    for (const double s : sentiments) {
        if (s > 0) pos_sum += s + 1;
        if (s < 0) neg_sum += s - 1;
        if (s == 0) ++neu_count;
    }
    if (pos_sum > std::fabs(neg_sum)) pos_sum += emphasis;
    else if (pos_sum < std::fabs(neg_sum)) neg_sum -= emphasis;
    const double total = pos_sum + std::fabs(neg_sum) + static_cast<double>(neu_count);
    scores.positive = std::fabs(pos_sum / total);
    scores.negative = std::fabs(neg_sum / total);
    scores.neutral = std::fabs(static_cast<double>(neu_count) / total);
    return scores;
}

}  // namespace cyxwiz
