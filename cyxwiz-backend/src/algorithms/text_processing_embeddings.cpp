#include "cyxwiz/text_processing.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace cyxwiz {

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

} // namespace cyxwiz