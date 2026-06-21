#include "cyxwiz/text_processing.h"

#include <algorithm>

namespace cyxwiz {

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

} // namespace cyxwiz