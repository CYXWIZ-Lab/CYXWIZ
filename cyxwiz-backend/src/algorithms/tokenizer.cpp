#include "cyxwiz/tokenizer.h"
#include "cyxwiz/text_processing.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <spdlog/spdlog.h>
#include <unordered_map>

namespace cyxwiz {

// ============================================================================
// Vocabulary
// ============================================================================

Vocabulary::Vocabulary() {
    AddSpecialTokens();
}

void Vocabulary::AddSpecialTokens() {
    word_to_idx_.clear();
    idx_to_word_.clear();

    idx_to_word_ = {"[PAD]", "[UNK]", "[BOS]", "[EOS]"};
    word_to_idx_["[PAD]"] = 0;
    word_to_idx_["[UNK]"] = 1;
    word_to_idx_["[BOS]"] = 2;
    word_to_idx_["[EOS]"] = 3;
}

void Vocabulary::BuildFromDocuments(const std::vector<std::string>& documents,
                                     int min_freq, int max_vocab_size,
                                     bool lowercase) {
    // Count word frequencies using existing TextProcessing
    std::unordered_map<std::string, int> freq;
    for (const auto& doc : documents) {
        auto result = TextProcessing::Tokenize(doc, "word", 2, lowercase, true);
        for (const auto& token : result.tokens) {
            freq[token]++;
        }
    }

    // Sort by frequency descending
    std::vector<std::pair<std::string, int>> sorted;
    sorted.reserve(freq.size());
    for (const auto& [word, count] : freq) {
        if (count >= min_freq) {
            sorted.push_back({word, count});
        }
    }
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Rebuild with special tokens first
    AddSpecialTokens();
    const int token_capacity = max_vocab_size > 0
        ? std::max(0, max_vocab_size - static_cast<int>(idx_to_word_.size()))
        : static_cast<int>(sorted.size());
    if (static_cast<int>(sorted.size()) > token_capacity) {
        sorted.resize(static_cast<size_t>(token_capacity));
    }
    for (const auto& [word, count] : sorted) {
        if (word_to_idx_.find(word) == word_to_idx_.end()) {
            int idx = static_cast<int>(idx_to_word_.size());
            word_to_idx_[word] = idx;
            idx_to_word_.push_back(word);
        }
    }

    spdlog::info("Vocabulary built: {} words (from {} documents, min_freq={})",
                 idx_to_word_.size(), documents.size(), min_freq);
}

void Vocabulary::SetVocabulary(const std::vector<std::string>& words) {
    AddSpecialTokens();
    for (const auto& word : words) {
        if (word_to_idx_.find(word) == word_to_idx_.end()) {
            int idx = static_cast<int>(idx_to_word_.size());
            word_to_idx_[word] = idx;
            idx_to_word_.push_back(word);
        }
    }
}

int Vocabulary::AddWord(const std::string& word) {
    auto it = word_to_idx_.find(word);
    if (it != word_to_idx_.end()) {
        return it->second;
    }
    int idx = static_cast<int>(idx_to_word_.size());
    word_to_idx_[word] = idx;
    idx_to_word_.push_back(word);
    return idx;
}

int Vocabulary::WordToIndex(const std::string& word) const {
    auto it = word_to_idx_.find(word);
    return (it != word_to_idx_.end()) ? it->second : unk_idx_;
}

std::string Vocabulary::IndexToWord(int index) const {
    if (index >= 0 && index < static_cast<int>(idx_to_word_.size())) {
        return idx_to_word_[index];
    }
    return "[UNK]";
}

bool Vocabulary::HasWord(const std::string& word) const {
    return word_to_idx_.find(word) != word_to_idx_.end();
}

// ============================================================================

Tokenizer::Tokenizer(TokenizerType type) : type_(type) {}

std::vector<std::string> Tokenizer::Split(const std::string& text) const {
    std::string processed = text;
    if (lowercase_) {
        processed = TextProcessing::ToLowercase(processed);
    }

    switch (type_) {
        case TokenizerType::Whitespace: {
            auto result = TextProcessing::Tokenize(processed, "whitespace", 2, false, false);
            return result.tokens;
        }
        case TokenizerType::Word: {
            auto result = TextProcessing::Tokenize(processed, "word", 2, false, true);
            return result.tokens;
        }
        case TokenizerType::Character: {
            std::vector<std::string> chars;
            for (char c : processed) {
                chars.push_back(std::string(1, c));
            }
            return chars;
        }
    }
    return {};
}

std::vector<int> Tokenizer::Encode(const std::string& text) const {
    auto tokens = Split(text);

    std::vector<int> ids;
    ids.reserve(tokens.size() + 2); // +2 for BOS/EOS

    if (add_bos_) {
        ids.push_back(vocab_.BosIndex());
    }

    for (const auto& token : tokens) {
        ids.push_back(vocab_.WordToIndex(token));
    }

    if (add_eos_) {
        ids.push_back(vocab_.EosIndex());
    }

    // Truncation
    if (do_truncation_ && max_length_ > 0 && static_cast<int>(ids.size()) > max_length_) {
        ids.resize(max_length_);
    }

    // Padding
    if (do_padding_ && max_length_ > 0 && static_cast<int>(ids.size()) < max_length_) {
        ids.resize(max_length_, vocab_.PadIndex());
    }

    return ids;
}

std::string Tokenizer::Decode(const std::vector<int>& token_ids) const {
    std::string result;
    for (size_t i = 0; i < token_ids.size(); i++) {
        int id = token_ids[i];
        // Skip special tokens
        if (id == vocab_.PadIndex() || id == vocab_.BosIndex() || id == vocab_.EosIndex()) {
            continue;
        }
        if (!result.empty()) result += " ";
        result += vocab_.IndexToWord(id);
    }
    return result;
}

std::vector<std::vector<int>> Tokenizer::EncodeBatch(const std::vector<std::string>& texts) const {
    std::vector<std::vector<int>> batch;
    batch.reserve(texts.size());
    for (const auto& text : texts) {
        batch.push_back(Encode(text));
    }
    return batch;
}

std::vector<std::string> Tokenizer::DecodeBatch(const std::vector<std::vector<int>>& batch) const {
    std::vector<std::string> texts;
    texts.reserve(batch.size());
    for (const auto& ids : batch) {
        texts.push_back(Decode(ids));
    }
    return texts;
}

TokenizedText Tokenizer::Tokenize(const std::string& text) const {
    TokenizedText result;
    result.original_length = text.length();

    result.tokens = Split(text);
    result.token_ids = Encode(text);
    result.truncated = do_truncation_ && max_length_ > 0 &&
                       static_cast<int>(result.tokens.size()) > max_length_;
    result.padded = do_padding_ && max_length_ > 0 &&
                    static_cast<int>(result.tokens.size()) < max_length_;

    return result;
}

std::vector<std::vector<int>> Tokenizer::PadBatch(const std::vector<std::vector<int>>& batch,
                                                    int max_length) const {
    if (batch.empty()) return {};

    // Determine target length
    int target = max_length;
    if (target <= 0) {
        target = 0;
        for (const auto& seq : batch) {
            target = std::max(target, static_cast<int>(seq.size()));
        }
    }

    // Pad each sequence
    std::vector<std::vector<int>> padded;
    padded.reserve(batch.size());
    for (const auto& seq : batch) {
        auto s = seq;
        if (static_cast<int>(s.size()) > target) {
            s.resize(target);
        } else if (static_cast<int>(s.size()) < target) {
            s.resize(target, vocab_.PadIndex());
        }
        padded.push_back(std::move(s));
    }
    return padded;
}

void Tokenizer::Train(const std::vector<std::string>& documents,
                       int min_freq, int max_vocab_size) {
    std::unordered_map<std::string, int> freq;
    for (const auto& doc : documents) {
        for (const auto& token : Split(doc)) {
            ++freq[token];
        }
    }

    std::vector<std::pair<std::string, int>> sorted;
    sorted.reserve(freq.size());
    for (const auto& [token, count] : freq) {
        if (count >= min_freq) {
            sorted.push_back({token, count});
        }
    }
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) {
                  if (a.second != b.second) return a.second > b.second;
                  return a.first < b.first;
              });

    const int special_count = 4;
    const int token_capacity = max_vocab_size > 0
        ? std::max(0, max_vocab_size - special_count)
        : static_cast<int>(sorted.size());
    if (static_cast<int>(sorted.size()) > token_capacity) {
        sorted.resize(static_cast<size_t>(token_capacity));
    }

    std::vector<std::string> words;
    words.reserve(sorted.size());
    for (const auto& [token, count] : sorted) {
        (void)count;
        words.push_back(token);
    }
    vocab_.SetVocabulary(words);
    spdlog::info("Tokenizer trained: vocab_size={}, type={}",
                 vocab_.Size(),
                 type_ == TokenizerType::Word ? "word" :
                 type_ == TokenizerType::Whitespace ? "whitespace" : "character");
}

} // namespace cyxwiz

