#include "cyxwiz/text_processing.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <cctype>
#include <regex>
#include <numeric>
#include <iomanip>

namespace cyxwiz {

// ============================================================================
// Static Data Initialization
// ============================================================================

static std::set<std::string> g_stopwords;
static bool g_stopwords_initialized = false;

void TextProcessing::InitStopwords() {
    if (g_stopwords_initialized) return;

    // Common English stopwords
    g_stopwords = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare", "ought",
        "used", "this", "that", "these", "those", "i", "you", "he", "she", "it",
        "we", "they", "what", "which", "who", "whom", "this", "that", "am", "is",
        "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
        "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or",
        "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
        "against", "between", "into", "through", "during", "before", "after", "above",
        "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
        "again", "further", "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "each", "few", "more", "most", "other", "some", "such", "no",
        "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
        "just", "don", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren",
        "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn",
        "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn"
    };

    g_stopwords_initialized = true;
}

// ============================================================================
// Tokenization
// ============================================================================

TokenizationResult TextProcessing::Tokenize(
    const std::string& text,
    const std::string& method,
    int ngram_n,
    bool lowercase,
    bool remove_punctuation
) {
    TokenizationResult result;

    if (text.empty()) {
        result.success = true;
        result.method = method;
        return result;
    }

    try {
        if (method == "whitespace") {
            result.tokens = TokenizeWhitespace(text);
        } else if (method == "word") {
            result.tokens = TokenizeWord(text, lowercase, remove_punctuation);
        } else if (method == "sentence") {
            result.tokens = TokenizeSentence(text);
        } else if (method == "ngram") {
            result.tokens = TokenizeNgram(text, ngram_n, lowercase);
        } else {
            result.error_message = "Unknown tokenization method: " + method;
            return result;
        }

        result.method = method;
        result.token_count = static_cast<int>(result.tokens.size());

        // Count unique tokens
        std::set<std::string> unique_set(result.tokens.begin(), result.tokens.end());
        result.unique_count = static_cast<int>(unique_set.size());

        // Calculate average token length
        if (!result.tokens.empty()) {
            double total_length = 0;
            for (const auto& token : result.tokens) {
                total_length += token.length();
            }
            result.avg_token_length = total_length / result.tokens.size();
        }

        // Calculate spans (simplified - just track positions)
        int pos = 0;
        for (const auto& token : result.tokens) {
            size_t found = text.find(token, pos);
            if (found != std::string::npos) {
                result.spans.emplace_back(static_cast<int>(found),
                                          static_cast<int>(found + token.length()));
                pos = static_cast<int>(found + 1);
            }
        }

        result.success = true;
    } catch (const std::exception& e) {
        result.error_message = std::string("Tokenization error: ") + e.what();
    }

    return result;
}

std::vector<std::string> TextProcessing::TokenizeWhitespace(const std::string& text) {
    std::vector<std::string> tokens;
    std::istringstream iss(text);
    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

std::vector<std::string> TextProcessing::TokenizeWord(
    const std::string& text,
    bool lowercase,
    bool remove_punct
) {
    std::vector<std::string> tokens;
    std::string current_token;

    for (char c : text) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            if (lowercase) {
                current_token += static_cast<char>(
                    std::tolower(static_cast<unsigned char>(c)));
            } else {
                current_token += c;
            }
        } else if (std::ispunct(static_cast<unsigned char>(c))) {
            if (!current_token.empty()) {
                tokens.push_back(current_token);
                current_token.clear();
            }
            if (!remove_punct) {
                tokens.push_back(std::string(1, c));
            }
        } else if (std::isspace(static_cast<unsigned char>(c))) {
            if (!current_token.empty()) {
                tokens.push_back(current_token);
                current_token.clear();
            }
        }
    }

    if (!current_token.empty()) {
        tokens.push_back(current_token);
    }

    return tokens;
}

std::vector<std::string> TextProcessing::TokenizeSentence(const std::string& text) {
    std::vector<std::string> sentences;
    std::string current;

    for (size_t i = 0; i < text.length(); ++i) {
        current += text[i];

        // Check for sentence-ending punctuation
        if (text[i] == '.' || text[i] == '!' || text[i] == '?') {
            // Check if it's actually end of sentence (not abbreviation)
            bool is_end = true;
            if (text[i] == '.') {
                // Simple check: if next char is uppercase or space followed by uppercase
                if (i + 1 < text.length()) {
                    char next = text[i + 1];
                    if (std::isalpha(static_cast<unsigned char>(next)) &&
                        !std::isupper(static_cast<unsigned char>(next))) {
                        is_end = false;
                    }
                }
            }

            if (is_end) {
                // Trim whitespace
                size_t start = current.find_first_not_of(" \t\n\r");
                size_t end = current.find_last_not_of(" \t\n\r");
                if (start != std::string::npos) {
                    sentences.push_back(current.substr(start, end - start + 1));
                }
                current.clear();
            }
        }
    }

    // Add remaining text as last sentence
    if (!current.empty()) {
        size_t start = current.find_first_not_of(" \t\n\r");
        size_t end = current.find_last_not_of(" \t\n\r");
        if (start != std::string::npos) {
            sentences.push_back(current.substr(start, end - start + 1));
        }
    }

    return sentences;
}

std::vector<std::string> TextProcessing::TokenizeNgram(
    const std::string& text,
    int n,
    bool lowercase
) {
    // First tokenize into words
    auto words = TokenizeWord(text, lowercase, true);

    std::vector<std::string> ngrams;
    if (words.size() < static_cast<size_t>(n)) {
        return ngrams;
    }

    for (size_t i = 0; i <= words.size() - n; ++i) {
        std::string ngram;
        for (int j = 0; j < n; ++j) {
            if (j > 0) ngram += " ";
            ngram += words[i + j];
        }
        ngrams.push_back(ngram);
    }

    return ngrams;
}

std::vector<std::string> TextProcessing::RemoveStopwords(
    const std::vector<std::string>& tokens,
    const std::string& /*language*/
) {
    InitStopwords();

    std::vector<std::string> filtered;
    for (const auto& token : tokens) {
        std::string lower_token = ToLowercase(token);
        if (g_stopwords.find(lower_token) == g_stopwords.end()) {
            filtered.push_back(token);
        }
    }
    return filtered;
}

std::string TextProcessing::Stem(const std::string& word) {
    // Simple Porter-like stemmer (simplified version)
    if (word.length() < 3) return word;

    std::string result = ToLowercase(word);

    // Remove common suffixes
    const std::vector<std::pair<std::string, std::string>> rules = {
        {"ational", "ate"}, {"tional", "tion"}, {"enci", "ence"},
        {"anci", "ance"}, {"izer", "ize"}, {"isation", "ize"},
        {"ization", "ize"}, {"ation", "ate"}, {"ator", "ate"},
        {"alism", "al"}, {"iveness", "ive"}, {"fulness", "ful"},
        {"ousness", "ous"}, {"aliti", "al"}, {"iviti", "ive"},
        {"biliti", "ble"}, {"alli", "al"}, {"entli", "ent"},
        {"eli", "e"}, {"ousli", "ous"}, {"ation", "ate"},
        {"ness", ""}, {"ment", ""}, {"ing", ""}, {"ings", ""},
        {"ed", ""}, {"es", ""}, {"ly", ""}, {"s", ""}
    };

    for (const auto& rule : rules) {
        if (result.length() > rule.first.length() + 2) {
            if (result.substr(result.length() - rule.first.length()) == rule.first) {
                result = result.substr(0, result.length() - rule.first.length()) + rule.second;
                break;
            }
        }
    }

    return result;
}

std::vector<std::string> TextProcessing::StemWords(const std::vector<std::string>& words) {
    std::vector<std::string> stemmed;
    stemmed.reserve(words.size());
    for (const auto& word : words) {
        stemmed.push_back(Stem(word));
    }
    return stemmed;
}

const std::set<std::string>& TextProcessing::GetStopwords() {
    InitStopwords();
    return g_stopwords;
}

// ============================================================================
// Helper Functions
// ============================================================================

double TextProcessing::L2Norm(const std::vector<double>& vec) {
    double sum = 0.0;
    for (double v : vec) {
        sum += v * v;
    }
    return std::sqrt(sum);
}

double TextProcessing::L1Norm(const std::vector<double>& vec) {
    double sum = 0.0;
    for (double v : vec) {
        sum += std::abs(v);
    }
    return sum;
}

void TextProcessing::NormalizeVector(std::vector<double>& vec, const std::string& norm) {
    if (norm == "none") return;

    double n = (norm == "l1") ? L1Norm(vec) : L2Norm(vec);
    if (n > 0) {
        for (double& v : vec) {
            v /= n;
        }
    }
}

} // namespace cyxwiz
