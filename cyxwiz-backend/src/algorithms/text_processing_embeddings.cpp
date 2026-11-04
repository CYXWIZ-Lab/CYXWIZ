#include "cyxwiz/text_processing.h"

#include <algorithm>
#include <cmath>

namespace cyxwiz {

// ============================================================================
// Word Embeddings
// ============================================================================

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

} // namespace cyxwiz