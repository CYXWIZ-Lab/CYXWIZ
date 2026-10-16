#pragma once

// The vocabulary a word-vector trainer works on (word2vec, GloVe; TOFIX152):
// counted from the strings on the host, words seen at least `min_count` times,
// most frequent first with ties in the order they were first seen (gensim's
// stable sort), and the corpus as ids with each word's sentence, words outside
// the vocabulary dropped before any window is taken.

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

namespace cyxwiz {

struct TrainingVocabulary {
    std::vector<std::string> words;
    std::vector<long long> counts;
    std::vector<int> tokens;       // the corpus as word ids
    std::vector<int> sentence_of;  // each token's sentence
};

// False when fewer than two words remain.
inline bool BuildTrainingVocabulary(const std::vector<std::vector<std::string>>& sentences, int min_count,
                                    size_t max_vocab, TrainingVocabulary& out) {
    out = TrainingVocabulary{};
    std::unordered_map<std::string, size_t> first_seen;
    std::vector<std::string> seen_words;
    std::vector<long long> seen_counts;
    for (const auto& sentence : sentences) {
        for (const auto& word : sentence) {
            const auto [it, inserted] = first_seen.emplace(word, seen_words.size());
            if (inserted) {
                seen_words.push_back(word);
                seen_counts.push_back(0);
            }
            ++seen_counts[it->second];
        }
    }
    std::vector<size_t> order;
    for (size_t i = 0; i < seen_words.size(); ++i)
        if (seen_counts[i] >= min_count) order.push_back(i);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return seen_counts[a] > seen_counts[b]; });
    if (max_vocab > 0 && order.size() > max_vocab) order.resize(max_vocab);
    if (order.size() < 2) return false;
    std::unordered_map<std::string, int> id_of;
    for (size_t r = 0; r < order.size(); ++r) {
        id_of.emplace(seen_words[order[r]], static_cast<int>(r));
        out.words.push_back(seen_words[order[r]]);
        out.counts.push_back(seen_counts[order[r]]);
    }
    for (size_t s = 0; s < sentences.size(); ++s) {
        for (const auto& word : sentences[s]) {
            if (const auto it = id_of.find(word); it != id_of.end()) {
                out.tokens.push_back(it->second);
                out.sentence_of.push_back(static_cast<int>(s));
            }
        }
    }
    return true;
}

}  // namespace cyxwiz
