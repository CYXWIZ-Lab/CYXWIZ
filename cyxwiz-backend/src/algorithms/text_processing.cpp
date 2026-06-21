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
                current_token += std::tolower(static_cast<unsigned char>(c));
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
// Word Frequency
// ============================================================================

WordFrequencyResult TextProcessing::ComputeWordFrequency(
    const std::string& text,
    int top_n,
    bool remove_stopwords,
    int min_word_length
) {
    WordFrequencyResult result;

    if (text.empty()) {
        result.success = true;
        return result;
    }

    try {
        // Tokenize
        auto tokens = TokenizeWord(text, true, true);

        // Remove stopwords if requested
        if (remove_stopwords) {
            tokens = RemoveStopwords(tokens, "english");
        }

        // Filter by minimum length
        std::vector<std::string> filtered;
        for (const auto& token : tokens) {
            if (static_cast<int>(token.length()) >= min_word_length) {
                filtered.push_back(token);
            }
        }

        result = ComputeWordFrequencyFromTokens(filtered, top_n);

    } catch (const std::exception& e) {
        result.error_message = std::string("Word frequency error: ") + e.what();
    }

    return result;
}

WordFrequencyResult TextProcessing::ComputeWordFrequencyFromTokens(
    const std::vector<std::string>& tokens,
    int top_n
) {
    WordFrequencyResult result;

    if (tokens.empty()) {
        result.success = true;
        return result;
    }

    try {
        // Count frequencies
        std::map<std::string, int> freq_map;
        for (const auto& token : tokens) {
            freq_map[token]++;
            result.length_distribution[static_cast<int>(token.length())]++;
        }

        // Convert to sorted vector
        result.frequencies.reserve(freq_map.size());
        for (const auto& pair : freq_map) {
            result.frequencies.emplace_back(pair.first, pair.second);
        }

        // Sort by frequency (descending)
        std::sort(result.frequencies.begin(), result.frequencies.end(),
            [](const auto& a, const auto& b) {
                return a.second > b.second;
            });

        // Compute statistics
        result.total_words = static_cast<int>(tokens.size());
        result.unique_words = static_cast<int>(freq_map.size());
        result.type_token_ratio = result.total_words > 0 ?
            static_cast<double>(result.unique_words) / result.total_words : 0.0;

        if (!result.frequencies.empty()) {
            result.max_frequency = result.frequencies[0].second;
            result.most_common_word = result.frequencies[0].first;
        }

        // Calculate average word length
        double total_length = 0;
        for (const auto& token : tokens) {
            total_length += token.length();
        }
        result.avg_word_length = result.total_words > 0 ?
            total_length / result.total_words : 0.0;

        // Trim to top_n if specified
        if (top_n > 0 && static_cast<int>(result.frequencies.size()) > top_n) {
            result.frequencies.resize(top_n);
        }

        result.success = true;
    } catch (const std::exception& e) {
        result.error_message = std::string("Frequency computation error: ") + e.what();
    }

    return result;
}

std::map<std::string, int> TextProcessing::BuildVocabulary(
    const std::vector<std::string>& documents,
    int min_freq,
    int max_vocab_size
) {
    // Count word frequencies across all documents
    std::map<std::string, int> word_counts;

    for (const auto& doc : documents) {
        auto tokens = TokenizeWord(doc, true, true);
        for (const auto& token : tokens) {
            word_counts[token]++;
        }
    }

    // Filter by minimum frequency and sort
    std::vector<std::pair<std::string, int>> sorted_words;
    for (const auto& pair : word_counts) {
        if (pair.second >= min_freq) {
            sorted_words.emplace_back(pair);
        }
    }

    // Sort by frequency (descending)
    std::sort(sorted_words.begin(), sorted_words.end(),
        [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

    // Build vocabulary with indices
    std::map<std::string, int> vocabulary;
    int index = 0;
    for (const auto& pair : sorted_words) {
        if (max_vocab_size > 0 && index >= max_vocab_size) break;
        vocabulary[pair.first] = index++;
    }

    return vocabulary;
}

// ============================================================================
// Text Utilities
// ============================================================================

std::string TextProcessing::CleanText(
    const std::string& text,
    bool remove_urls,
    bool remove_emails,
    bool remove_numbers,
    bool remove_special
) {
    std::string result = text;

    // Remove URLs
    if (remove_urls) {
        std::regex url_pattern(R"(https?://\S+|www\.\S+)");
        result = std::regex_replace(result, url_pattern, " ");
    }

    // Remove emails
    if (remove_emails) {
        std::regex email_pattern(R"(\S+@\S+\.\S+)");
        result = std::regex_replace(result, email_pattern, " ");
    }

    // Remove numbers
    if (remove_numbers) {
        std::regex num_pattern(R"(\d+)");
        result = std::regex_replace(result, num_pattern, " ");
    }

    // Remove special characters (keep alphanumeric and basic punctuation)
    if (remove_special) {
        std::string cleaned;
        for (char c : result) {
            if (std::isalnum(static_cast<unsigned char>(c)) ||
                std::isspace(static_cast<unsigned char>(c)) ||
                c == '.' || c == ',' || c == '!' || c == '?' || c == '\'' || c == '-') {
                cleaned += c;
            } else {
                cleaned += ' ';
            }
        }
        result = cleaned;
    }

    return NormalizeWhitespace(result);
}

std::vector<std::string> TextProcessing::SplitSentences(const std::string& text) {
    return TokenizeSentence(text);
}

std::string TextProcessing::ToLowercase(const std::string& text) {
    std::string result;
    result.reserve(text.size());
    for (char c : text) {
        result += std::tolower(static_cast<unsigned char>(c));
    }
    return result;
}

std::string TextProcessing::RemovePunctuation(const std::string& text) {
    std::string result;
    for (char c : text) {
        if (!std::ispunct(static_cast<unsigned char>(c))) {
            result += c;
        }
    }
    return result;
}

std::string TextProcessing::NormalizeWhitespace(const std::string& text) {
    std::string result;
    bool prev_space = true;  // Start true to trim leading

    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!prev_space) {
                result += ' ';
                prev_space = true;
            }
        } else {
            result += c;
            prev_space = false;
        }
    }

    // Trim trailing space
    if (!result.empty() && result.back() == ' ') {
        result.pop_back();
    }

    return result;
}

// ============================================================================
// Sample Text Generation
// ============================================================================

std::string TextProcessing::GenerateSampleText(const std::string& type) {
    if (type == "lorem") {
        return "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor "
               "incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud "
               "exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute "
               "irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla "
               "pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia "
               "deserunt mollit anim id est laborum.";
    } else if (type == "news") {
        return "Scientists at the research institute announced a breakthrough discovery today "
               "that could revolutionize our understanding of the universe. The team, led by "
               "Dr. Smith, has been working on this project for over five years. The findings "
               "were published in a leading scientific journal and have already attracted "
               "attention from researchers worldwide. This discovery builds on previous work "
               "and opens new possibilities for future research in the field.";
    } else if (type == "review_positive") {
        return "This product is absolutely amazing! I've never been happier with a purchase. "
               "The quality is outstanding and it exceeded all my expectations. The customer "
               "service was excellent and shipping was incredibly fast. I would highly recommend "
               "this to anyone looking for a great product. Five stars without hesitation! "
               "Best purchase I've made this year. Will definitely buy again.";
    } else if (type == "review_negative") {
        return "Terrible experience with this product. Complete waste of money. It broke within "
               "a week of normal use and customer support was unhelpful. The quality is extremely "
               "poor and nothing like what was advertised. I deeply regret this purchase and "
               "would never recommend it to anyone. Avoid at all costs. Worst product I have "
               "ever bought. Very disappointed and frustrated.";
    } else if (type == "technical") {
        return "Machine learning algorithms process data to identify patterns and make predictions. "
               "Neural networks are a subset of machine learning inspired by the human brain. "
               "Deep learning uses multiple layers of neural networks to learn representations "
               "of data with multiple levels of abstraction. Convolutional neural networks are "
               "particularly effective for image recognition tasks while recurrent neural networks "
               "excel at processing sequential data like text and time series.";
    }

    return "Sample text for analysis. This is a default placeholder text that can be "
           "used for testing various text processing features and algorithms.";
}

std::vector<std::string> TextProcessing::GenerateSampleDocuments(
    int num_docs,
    const std::string& type
) {
    std::vector<std::string> documents;

    if (type == "news") {
        documents = {
            "The stock market experienced significant volatility today as investors reacted to "
            "new economic data. Technology stocks led the decline while energy companies showed gains.",

            "Scientists have discovered a new species of deep-sea fish in the Pacific Ocean. "
            "The creature lives at depths previously thought too extreme for complex life forms.",

            "The government announced new climate change policies aimed at reducing carbon emissions "
            "by fifty percent over the next decade through investments in renewable energy.",

            "A major cybersecurity breach has affected millions of users worldwide. Experts advise "
            "changing passwords immediately and enabling two-factor authentication.",

            "The annual technology conference unveiled several innovative products including "
            "advanced artificial intelligence systems and quantum computing developments."
        };
    } else if (type == "reviews") {
        documents = {
            "Great product, excellent quality. Fast shipping and good customer service. "
            "Would definitely recommend to friends and family.",

            "Disappointing purchase. The item arrived damaged and did not match the description. "
            "Very poor experience overall. Will not buy again.",

            "Average product for the price. Does what it's supposed to do but nothing special. "
            "Might consider other options next time.",

            "Absolutely love this! Best thing I've bought all year. Works perfectly and "
            "the design is beautiful. Five stars!",

            "Terrible quality, broke after one week. Customer service was unhelpful. "
            "Complete waste of money. Avoid this seller."
        };
    } else {
        documents = {
            "Machine learning is transforming industries across the globe.",
            "Data science combines statistics and programming skills.",
            "Deep learning enables computers to learn from large datasets.",
            "Neural networks are inspired by biological brain structure.",
            "Artificial intelligence is revolutionizing healthcare and medicine."
        };
    }

    // Return requested number of documents (with cycling if needed)
    std::vector<std::string> result;
    for (int i = 0; i < num_docs; ++i) {
        result.push_back(documents[i % documents.size()]);
    }

    return result;
}

std::vector<std::string> TextProcessing::GenerateSampleVocabulary(
    int size,
    const std::string& domain
) {
    std::vector<std::string> words;

    if (domain == "tech") {
        words = {"algorithm", "computer", "data", "network", "software", "hardware",
                 "database", "server", "cloud", "security", "encryption", "protocol",
                 "interface", "programming", "code", "debug", "compile", "deploy",
                 "system", "architecture", "framework", "library", "function", "class",
                 "object", "variable", "loop", "condition", "array", "string"};
    } else if (domain == "science") {
        words = {"experiment", "hypothesis", "theory", "research", "analysis", "data",
                 "observation", "measurement", "result", "conclusion", "method", "study",
                 "sample", "control", "variable", "statistics", "significance", "model",
                 "predict", "test", "validate", "replicate", "review", "publish",
                 "discover", "evidence", "proof", "equation", "formula", "constant"};
    } else {
        words = {"the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
                 "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
                 "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
                 "an", "will", "my", "one", "all", "would", "there", "their", "what",
                 "so", "up", "out", "if", "about", "who", "get", "which", "go", "me"};
    }

    // Extend or trim to requested size
    std::vector<std::string> result;
    for (int i = 0; i < size; ++i) {
        if (i < static_cast<int>(words.size())) {
            result.push_back(words[i]);
        } else {
            result.push_back("word" + std::to_string(i));
        }
    }

    return result;
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
