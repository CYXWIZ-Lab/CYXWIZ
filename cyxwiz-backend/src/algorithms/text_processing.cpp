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
static std::map<std::string, double> g_simple_lexicon;
static std::map<std::string, double> g_afinn_lexicon;
static bool g_stopwords_initialized = false;
static bool g_lexicons_initialized = false;

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

void TextProcessing::InitSentimentLexicons() {
    if (g_lexicons_initialized) return;

    // Simple sentiment lexicon (common words)
    g_simple_lexicon = {
        // Positive words
        {"good", 0.7}, {"great", 0.9}, {"excellent", 1.0}, {"amazing", 0.95},
        {"wonderful", 0.9}, {"fantastic", 0.95}, {"awesome", 0.9}, {"best", 0.85},
        {"love", 0.8}, {"like", 0.5}, {"happy", 0.8}, {"joy", 0.85}, {"beautiful", 0.8},
        {"perfect", 0.95}, {"nice", 0.6}, {"pleasant", 0.65}, {"positive", 0.7},
        {"brilliant", 0.9}, {"superb", 0.9}, {"outstanding", 0.9}, {"exceptional", 0.9},
        {"impressive", 0.75}, {"remarkable", 0.8}, {"incredible", 0.85}, {"delightful", 0.8},
        {"enjoyable", 0.7}, {"satisfying", 0.7}, {"recommend", 0.6}, {"fun", 0.65},
        {"exciting", 0.75}, {"pleased", 0.7}, {"glad", 0.65}, {"thankful", 0.7},
        {"grateful", 0.75}, {"appreciate", 0.6}, {"friendly", 0.6}, {"helpful", 0.65},

        // Negative words
        {"bad", -0.7}, {"terrible", -0.95}, {"awful", -0.9}, {"horrible", -0.95},
        {"worst", -1.0}, {"hate", -0.9}, {"dislike", -0.6}, {"sad", -0.7},
        {"angry", -0.8}, {"disappointed", -0.75}, {"poor", -0.6}, {"negative", -0.7},
        {"ugly", -0.7}, {"boring", -0.6}, {"annoying", -0.7}, {"frustrating", -0.75},
        {"useless", -0.8}, {"waste", -0.7}, {"fail", -0.75}, {"failed", -0.75},
        {"failure", -0.8}, {"problem", -0.5}, {"issue", -0.4}, {"broken", -0.7},
        {"wrong", -0.6}, {"error", -0.5}, {"mistake", -0.55}, {"difficult", -0.4},
        {"hard", -0.3}, {"pain", -0.6}, {"painful", -0.7}, {"unfortunately", -0.5},
        {"regret", -0.65}, {"sorry", -0.4}, {"disappoint", -0.7}, {"unhappy", -0.75},
        {"upset", -0.65}, {"worried", -0.5}, {"concern", -0.4}, {"fear", -0.6}
    };

    // AFINN-like lexicon (more comprehensive)
    g_afinn_lexicon = g_simple_lexicon;  // Start with simple lexicon
    // Add more words with nuanced scores
    g_afinn_lexicon["abandon"] = -0.4;
    g_afinn_lexicon["ability"] = 0.2;
    g_afinn_lexicon["able"] = 0.2;
    g_afinn_lexicon["abuse"] = -0.6;
    g_afinn_lexicon["accept"] = 0.3;
    g_afinn_lexicon["accident"] = -0.4;
    g_afinn_lexicon["accomplish"] = 0.5;
    g_afinn_lexicon["achieve"] = 0.5;
    g_afinn_lexicon["advantage"] = 0.4;
    g_afinn_lexicon["adventure"] = 0.4;
    g_afinn_lexicon["afraid"] = -0.4;
    g_afinn_lexicon["agree"] = 0.3;
    g_afinn_lexicon["alert"] = 0.1;
    g_afinn_lexicon["alone"] = -0.2;
    g_afinn_lexicon["amaze"] = 0.6;
    g_afinn_lexicon["anger"] = -0.6;
    g_afinn_lexicon["annoy"] = -0.5;
    g_afinn_lexicon["anxiety"] = -0.5;
    g_afinn_lexicon["anxious"] = -0.4;
    g_afinn_lexicon["apologize"] = -0.2;
    g_afinn_lexicon["approve"] = 0.4;
    g_afinn_lexicon["attack"] = -0.5;
    g_afinn_lexicon["attractive"] = 0.5;
    g_afinn_lexicon["avoid"] = -0.2;
    g_afinn_lexicon["award"] = 0.5;

    g_lexicons_initialized = true;
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
// TF-IDF
// ============================================================================

TFIDFResult TextProcessing::ComputeTFIDF(
    const std::vector<std::string>& documents,
    bool use_idf,
    bool smooth_idf,
    const std::string& norm
) {
    TFIDFResult result;

    if (documents.empty()) {
        result.success = true;
        return result;
    }

    try {
        result.num_documents = static_cast<int>(documents.size());
        result.normalization = norm;

        // Build vocabulary
        std::set<std::string> vocab_set;
        std::vector<std::vector<std::string>> tokenized_docs;

        for (const auto& doc : documents) {
            auto tokens = TokenizeWord(doc, true, true);
            tokens = RemoveStopwords(tokens, "english");
            tokenized_docs.push_back(tokens);

            for (const auto& token : tokens) {
                vocab_set.insert(token);
            }
        }

        // Convert to vector
        result.vocabulary.assign(vocab_set.begin(), vocab_set.end());
        std::sort(result.vocabulary.begin(), result.vocabulary.end());
        result.vocab_size = static_cast<int>(result.vocabulary.size());

        // Create word to index map
        std::map<std::string, int> word_to_idx;
        for (size_t i = 0; i < result.vocabulary.size(); ++i) {
            word_to_idx[result.vocabulary[i]] = static_cast<int>(i);
        }

        // Compute document frequency for each term
        std::vector<int> doc_freq(result.vocab_size, 0);
        for (const auto& tokens : tokenized_docs) {
            std::set<std::string> unique_tokens(tokens.begin(), tokens.end());
            for (const auto& token : unique_tokens) {
                auto it = word_to_idx.find(token);
                if (it != word_to_idx.end()) {
                    doc_freq[it->second]++;
                }
            }
        }

        // Compute IDF
        result.idf_scores.resize(result.vocab_size);
        for (int i = 0; i < result.vocab_size; ++i) {
            if (use_idf) {
                double df = smooth_idf ? doc_freq[i] + 1 : doc_freq[i];
                double n = smooth_idf ? result.num_documents + 1 : result.num_documents;
                result.idf_scores[i] = std::log(n / df) + 1;
            } else {
                result.idf_scores[i] = 1.0;
            }
        }

        // Compute TF-IDF matrix
        result.tfidf_matrix.resize(result.num_documents);
        result.doc_top_terms.resize(result.num_documents);

        for (size_t doc_idx = 0; doc_idx < tokenized_docs.size(); ++doc_idx) {
            const auto& tokens = tokenized_docs[doc_idx];

            // Count term frequencies
            std::map<std::string, int> tf_map;
            for (const auto& token : tokens) {
                tf_map[token]++;
            }

            // Compute TF-IDF
            std::vector<double>& tfidf_vec = result.tfidf_matrix[doc_idx];
            tfidf_vec.resize(result.vocab_size, 0.0);

            for (const auto& pair : tf_map) {
                auto it = word_to_idx.find(pair.first);
                if (it != word_to_idx.end()) {
                    double tf = static_cast<double>(pair.second) / tokens.size();
                    tfidf_vec[it->second] = tf * result.idf_scores[it->second];
                }
            }

            // Normalize
            NormalizeVector(tfidf_vec, norm);

            // Get top terms for this document
            std::vector<std::pair<std::string, double>> term_scores;
            for (size_t i = 0; i < tfidf_vec.size(); ++i) {
                if (tfidf_vec[i] > 0) {
                    term_scores.emplace_back(result.vocabulary[i], tfidf_vec[i]);
                }
            }
            std::sort(term_scores.begin(), term_scores.end(),
                [](const auto& a, const auto& b) {
                    return a.second > b.second;
                });

            // Keep top 10 terms
            if (term_scores.size() > 10) {
                term_scores.resize(10);
            }
            result.doc_top_terms[doc_idx] = term_scores;
        }

        // Compute similarity matrix
        result.similarity_matrix = ComputeSimilarityMatrix(result.tfidf_matrix);

        result.success = true;
    } catch (const std::exception& e) {
        result.error_message = std::string("TF-IDF error: ") + e.what();
    }

    return result;
}

std::vector<double> TextProcessing::ComputeTF(
    const std::string& document,
    const std::vector<std::string>& vocabulary
) {
    auto tokens = TokenizeWord(document, true, true);

    std::map<std::string, int> word_counts;
    for (const auto& token : tokens) {
        word_counts[token]++;
    }

    std::vector<double> tf(vocabulary.size(), 0.0);
    for (size_t i = 0; i < vocabulary.size(); ++i) {
        auto it = word_counts.find(vocabulary[i]);
        if (it != word_counts.end()) {
            tf[i] = static_cast<double>(it->second) / tokens.size();
        }
    }

    return tf;
}

std::vector<double> TextProcessing::ComputeIDF(
    const std::vector<std::string>& documents,
    const std::vector<std::string>& vocabulary,
    bool smooth
) {
    int n = static_cast<int>(documents.size());
    std::vector<int> doc_freq(vocabulary.size(), 0);

    // Create word to index map
    std::map<std::string, int> word_to_idx;
    for (size_t i = 0; i < vocabulary.size(); ++i) {
        word_to_idx[vocabulary[i]] = static_cast<int>(i);
    }

    // Count document frequency
    for (const auto& doc : documents) {
        auto tokens = TokenizeWord(doc, true, true);
        std::set<std::string> unique_tokens(tokens.begin(), tokens.end());
        for (const auto& token : unique_tokens) {
            auto it = word_to_idx.find(token);
            if (it != word_to_idx.end()) {
                doc_freq[it->second]++;
            }
        }
    }

    // Compute IDF
    std::vector<double> idf(vocabulary.size());
    for (size_t i = 0; i < vocabulary.size(); ++i) {
        double df = smooth ? doc_freq[i] + 1 : std::max(1, doc_freq[i]);
        double total = smooth ? n + 1 : n;
        idf[i] = std::log(total / df) + 1;
    }

    return idf;
}

double TextProcessing::CosineSimilarity(
    const std::vector<double>& vec1,
    const std::vector<double>& vec2
) {
    if (vec1.size() != vec2.size() || vec1.empty()) {
        return 0.0;
    }

    double dot = 0.0, norm1 = 0.0, norm2 = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i) {
        dot += vec1[i] * vec2[i];
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
    }

    double denom = std::sqrt(norm1) * std::sqrt(norm2);
    return denom > 0 ? dot / denom : 0.0;
}

std::vector<std::vector<double>> TextProcessing::ComputeSimilarityMatrix(
    const std::vector<std::vector<double>>& tfidf_matrix
) {
    int n = static_cast<int>(tfidf_matrix.size());
    std::vector<std::vector<double>> sim_matrix(n, std::vector<double>(n, 0.0));

    for (int i = 0; i < n; ++i) {
        sim_matrix[i][i] = 1.0;
        for (int j = i + 1; j < n; ++j) {
            double sim = CosineSimilarity(tfidf_matrix[i], tfidf_matrix[j]);
            sim_matrix[i][j] = sim;
            sim_matrix[j][i] = sim;
        }
    }

    return sim_matrix;
}

// ============================================================================
// Word Embeddings
// ============================================================================

EmbeddingResult TextProcessing::CreateOneHotEmbeddings(
    const std::vector<std::string>& vocabulary
) {
    EmbeddingResult result;

    if (vocabulary.empty()) {
        result.success = true;
        result.method = "onehot";
        return result;
    }

    try {
        result.words = vocabulary;
        result.embedding_dim = static_cast<int>(vocabulary.size());
        result.method = "onehot";

        result.embeddings.resize(vocabulary.size());
        for (size_t i = 0; i < vocabulary.size(); ++i) {
            result.embeddings[i].resize(vocabulary.size(), 0.0);
            result.embeddings[i][i] = 1.0;
        }

        result.success = true;
    } catch (const std::exception& e) {
        result.error_message = std::string("One-hot error: ") + e.what();
    }

    return result;
}

EmbeddingResult TextProcessing::CreateRandomEmbeddings(
    const std::vector<std::string>& vocabulary,
    int embedding_dim,
    int seed
) {
    EmbeddingResult result;

    if (vocabulary.empty()) {
        result.success = true;
        result.method = "random";
        return result;
    }

    try {
        result.words = vocabulary;
        result.embedding_dim = embedding_dim;
        result.method = "random";

        std::mt19937 gen(seed >= 0 ? seed : std::random_device{}());
        std::normal_distribution<> dist(0.0, 1.0 / std::sqrt(embedding_dim));

        result.embeddings.resize(vocabulary.size());
        for (size_t i = 0; i < vocabulary.size(); ++i) {
            result.embeddings[i].resize(embedding_dim);
            for (int j = 0; j < embedding_dim; ++j) {
                result.embeddings[i][j] = dist(gen);
            }
        }

        result.success = true;
    } catch (const std::exception& e) {
        result.error_message = std::string("Random embedding error: ") + e.what();
    }

    return result;
}

EmbeddingResult TextProcessing::FindSimilarWords(
    const std::string& word,
    const EmbeddingResult& embeddings,
    int top_n
) {
    EmbeddingResult result;
    result.words = embeddings.words;
    result.embeddings = embeddings.embeddings;
    result.embedding_dim = embeddings.embedding_dim;
    result.method = embeddings.method;

    // Find the word in vocabulary
    int word_idx = -1;
    for (size_t i = 0; i < embeddings.words.size(); ++i) {
        if (embeddings.words[i] == word) {
            word_idx = static_cast<int>(i);
            break;
        }
    }

    if (word_idx < 0) {
        result.error_message = "Word not found in vocabulary: " + word;
        return result;
    }

    // Compute similarities
    std::vector<std::pair<std::string, double>> similarities;
    for (size_t i = 0; i < embeddings.words.size(); ++i) {
        if (static_cast<int>(i) != word_idx) {
            double sim = CosineSimilarity(embeddings.embeddings[word_idx],
                                          embeddings.embeddings[i]);
            similarities.emplace_back(embeddings.words[i], sim);
        }
    }

    // Sort by similarity (descending)
    std::sort(similarities.begin(), similarities.end(),
        [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

    // Keep top_n
    if (static_cast<int>(similarities.size()) > top_n) {
        similarities.resize(top_n);
    }

    result.similar_words = similarities;
    result.success = true;

    return result;
}

double TextProcessing::WordSimilarity(
    const std::string& word1,
    const std::string& word2,
    const EmbeddingResult& embeddings
) {
    int idx1 = -1, idx2 = -1;
    for (size_t i = 0; i < embeddings.words.size(); ++i) {
        if (embeddings.words[i] == word1) idx1 = static_cast<int>(i);
        if (embeddings.words[i] == word2) idx2 = static_cast<int>(i);
    }

    if (idx1 < 0 || idx2 < 0) {
        return 0.0;
    }

    return CosineSimilarity(embeddings.embeddings[idx1], embeddings.embeddings[idx2]);
}

// ============================================================================
// Sentiment Analysis
// ============================================================================

SentimentResult TextProcessing::AnalyzeSentiment(
    const std::string& text,
    const std::string& method
) {
    SentimentResult result;

    if (text.empty()) {
        result.label = "neutral";
        result.success = true;
        return result;
    }

    try {
        InitSentimentLexicons();

        // Get lexicon
        const auto& lexicon = (method == "afinn") ? g_afinn_lexicon : g_simple_lexicon;

        // Tokenize
        auto tokens = TokenizeWord(text, true, true);

        // Compute sentiment
        double total_score = 0.0;
        int scored_words = 0;

        for (const auto& token : tokens) {
            auto it = lexicon.find(token);
            if (it != lexicon.end()) {
                double score = it->second;
                result.word_scores.emplace_back(token, score);
                total_score += score;
                scored_words++;

                if (score > 0.1) result.positive_count++;
                else if (score < -0.1) result.negative_count++;
                else result.neutral_count++;
            }
        }

        // Compute polarity (-1 to 1)
        if (scored_words > 0) {
            result.polarity = total_score / scored_words;
            // Clamp to [-1, 1]
            result.polarity = std::max(-1.0, std::min(1.0, result.polarity));
        }

        // Compute subjectivity (0 to 1) - based on proportion of sentiment words
        result.subjectivity = tokens.empty() ? 0.0 :
            static_cast<double>(scored_words) / tokens.size();

        // Determine label
        if (result.polarity > 0.1) {
            result.label = "positive";
        } else if (result.polarity < -0.1) {
            result.label = "negative";
        } else {
            result.label = "neutral";
        }

        // Compute confidence based on consistency
        if (result.positive_count + result.negative_count > 0) {
            int dominant = std::max(result.positive_count, result.negative_count);
            int total = result.positive_count + result.negative_count;
            result.confidence = static_cast<double>(dominant) / total;
        } else {
            result.confidence = 0.5;  // Neutral confidence
        }

        // Generate analysis text
        std::ostringstream analysis;
        analysis << "Analyzed " << tokens.size() << " tokens. ";
        analysis << "Found " << scored_words << " sentiment words: ";
        analysis << result.positive_count << " positive, ";
        analysis << result.negative_count << " negative, ";
        analysis << result.neutral_count << " neutral. ";
        analysis << "Overall sentiment: " << result.label << " ";
        analysis << "(polarity: " << std::fixed << std::setprecision(2) << result.polarity << ", ";
        analysis << "confidence: " << std::fixed << std::setprecision(0) << (result.confidence * 100) << "%).";
        result.analysis = analysis.str();

        result.success = true;
    } catch (const std::exception& e) {
        result.error_message = std::string("Sentiment error: ") + e.what();
    }

    return result;
}

const std::map<std::string, double>& TextProcessing::GetSentimentLexicon(
    const std::string& name
) {
    InitSentimentLexicons();
    return (name == "afinn") ? g_afinn_lexicon : g_simple_lexicon;
}

double TextProcessing::ComputePolarityScore(
    const std::vector<std::string>& tokens,
    const std::map<std::string, double>& lexicon
) {
    double total = 0.0;
    int count = 0;

    for (const auto& token : tokens) {
        std::string lower = ToLowercase(token);
        auto it = lexicon.find(lower);
        if (it != lexicon.end()) {
            total += it->second;
            count++;
        }
    }

    return count > 0 ? total / count : 0.0;
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
