// cyxwiz::TextProcessing's word rule, stop words, Porter stemmer and VADER
// against Python (TOFIX129 D8): fixtures/text_truth.json from
// reference/generate_text_fixtures.py (Python's str.isalnum and lower,
// scikit-learn's ENGLISH_STOP_WORDS, NLTK's PorterStemmer in
// ORIGINAL_ALGORITHM mode, vaderSentiment 3.3.2 unrounded). When
// CYXWIZ_TEXT_TRUTH_LOCAL names the local fixture of real rows (written with
// --local, not committed), it is checked the same way.

#include <cyxwiz/text_processing.h>

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

using json = nlohmann::json;
using cyxwiz::TextProcessing;

namespace {

int checks = 0;

void Check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

std::string Shown(const std::string& text) { return text.size() > 120 ? text.substr(0, 120) + "..." : text; }

std::string Joined(const std::vector<std::string>& words) {
    std::string out;
    for (const std::string& w : words) out += (out.empty() ? "" : " | ") + w;
    return out;
}

void CheckWords(const json& cases, const std::string& from) {
    for (const auto& c : cases) {
        const std::string text = c.at("text").get<std::string>();
        const std::vector<std::string> expected = c.at("words").get<std::vector<std::string>>();
        const std::vector<std::string> actual = TextProcessing::Words(text);
        Check(actual == expected, from + " words of \"" + Shown(text) + "\": " + Joined(actual) + "  vs  " + Joined(expected));
    }
}

void CheckVader(const json& cases, const std::string& from) {
    for (const auto& c : cases) {
        const std::string text = c.at("text").get<std::string>();
        const cyxwiz::VaderScores s = TextProcessing::Vader(text);
        const auto near = [&](double actual, const char* key) {
            const double expected = c.at(key).get<double>();
            Check(std::fabs(actual - expected) <= 1e-12, from + " vader " + key + " of \"" + Shown(text) + "\": " +
                                                             std::to_string(actual) + " vs " + std::to_string(expected));
        };
        near(s.compound, "compound");
        near(s.negative, "neg");
        near(s.neutral, "neu");
        near(s.positive, "pos");
    }
}

void CheckPorter(const json& cases, const std::string& from) {
    for (const auto& c : cases) {
        const std::string word = c.at("word").get<std::string>();
        const std::string expected = c.at("stem").get<std::string>();
        const std::string actual = TextProcessing::Stem(word);
        Check(actual == expected, from + " stem of " + word + ": " + actual + " vs " + expected);
    }
}

json Load(const char* path) {
    std::ifstream in(path, std::ios::binary);
    Check(in.good(), std::string("cannot open ") + path);
    return json::parse(in);
}

}  // namespace

int main() {
    const json fixture = Load(CYXWIZ_TEXT_FIXTURE);

    // The stop words are sklearn's list plus the contractions, exactly.
    const std::vector<std::string> stop = fixture.at("stop_words").get<std::vector<std::string>>();
    const std::set<std::string>& ours = TextProcessing::GetStopwords();
    Check(ours.size() == stop.size(), "stop words: " + std::to_string(ours.size()) + " vs " + std::to_string(stop.size()));
    for (const std::string& w : stop) Check(ours.count(w) == 1, "stop word missing: " + w);
    const std::vector<std::string> kept = TextProcessing::RemoveStopwords({"My", "anxiety", "don't", "I've", "life"});
    Check(kept == std::vector<std::string>({"anxiety", "life"}), "RemoveStopwords: " + Joined(kept));

    CheckWords(fixture.at("words"), "fixture");
    CheckVader(fixture.at("vader"), "fixture");
    CheckPorter(fixture.at("porter"), "fixture");

    // The word valences VADER reports line up with its words.
    const cyxwiz::VaderScores not_good = TextProcessing::Vader("The food was not good");
    Check(not_good.words.size() == 5 && not_good.words[4].first == "good" && not_good.words[4].second < 0,
          "vader words: 'good' after 'not' reads negative");
    Check(TextProcessing::Vader("").words.empty() && TextProcessing::Vader("").compound == 0.0, "vader of nothing");

    // TF-IDF: the raw count times the smoothed idf, as sklearn's TfidfVectorizer.
    const cyxwiz::TFIDFResult tfidf = TextProcessing::ComputeTFIDF({"good good movie", "bad movie"}, true, true, "none");
    Check(tfidf.success && tfidf.vocabulary == std::vector<std::string>({"bad", "good", "movie"}), "tfidf vocabulary");
    const double idf_once = std::log(3.0 / 2.0) + 1.0;
    Check(std::fabs(tfidf.tfidf_matrix[0][1] - 2.0 * idf_once) < 1e-12, "tfidf: count x idf");
    Check(std::fabs(tfidf.tfidf_matrix[0][2] - 1.0) < 1e-12, "tfidf: a term in every row has idf 1");

    size_t local_rows = 0;
    if (const char* local = std::getenv("CYXWIZ_TEXT_TRUTH_LOCAL"); local != nullptr && *local != '\0') {
        const json real = Load(local);
        CheckWords(real.at("words"), "local");
        CheckVader(real.at("vader"), "local");
        CheckPorter(real.at("porter"), "local");
        local_rows = real.at("words").size() + real.at("vader").size() + real.at("porter").size();
    }

    std::cout << "text matches Python / sklearn / NLTK Porter / vaderSentiment: " << fixture.at("words").size() << " word cases, "
              << fixture.at("vader").size() << " VADER cases, " << fixture.at("porter").size() << " stems";
    if (local_rows > 0) std::cout << " (and " << local_rows << " local cases from real rows)";
    std::cout << ", " << checks << " checks\n";
    return 0;
}
