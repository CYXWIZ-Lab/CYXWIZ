#include "cyxwiz/text_processing.h"

#include <algorithm>
#include <cmath>
#include <set>

namespace cyxwiz {

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

} // namespace cyxwiz