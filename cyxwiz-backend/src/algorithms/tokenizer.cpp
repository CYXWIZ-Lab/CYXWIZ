#include "cyxwiz/tokenizer.h"
#include "cyxwiz/text_processing.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <spdlog/spdlog.h>
#include <unordered_map>

#ifndef CYXWIZ_HAS_SENTENCEPIECE
#define CYXWIZ_HAS_SENTENCEPIECE 0
#endif

#if CYXWIZ_HAS_SENTENCEPIECE
#include <sentencepiece_processor.h>
#endif

namespace cyxwiz {

struct SentencePieceTokenizerState {
#if CYXWIZ_HAS_SENTENCEPIECE
    sentencepiece::SentencePieceProcessor processor;
#endif
    bool loaded = false;
    std::string model_bytes;
};

// ============================================================================
// Vocabulary
// ============================================================================

Vocabulary::Vocabulary() {
    AddSpecialTokens();
}

void Vocabulary::AddSpecialTokens() {
    byte_bpe_ = false;
    bpe_merges_.clear();
    bpe_ranks_.clear();
    pad_idx_ = 0; unk_idx_ = 1; bos_idx_ = 2; eos_idx_ = 3;
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
    if (byte_bpe_) throw std::logic_error("BPE vocabulary is immutable; retrain or load an artifact");
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

bool IsSentencePieceTokenizerType(TokenizerType type) {
    return type == TokenizerType::SentencePieceBPE ||
           type == TokenizerType::SentencePieceUnigram;
}

bool IsSentencePieceTokenizerAvailable() {
#if CYXWIZ_HAS_SENTENCEPIECE
    return true;
#else
    return false;
#endif
}

std::string SentencePieceTokenizerUnavailableMessage() {
#if CYXWIZ_HAS_SENTENCEPIECE
    return "SentencePiece tokenizer provider is enabled";
#else
    return "SentencePiece tokenizer support is not enabled in this build; install/build the optional provider or choose a native tokenizer family";
#endif
}

Tokenizer::Tokenizer(TokenizerType type) : type_(type) {
    if (type_ == TokenizerType::ByteBPE) lowercase_ = false;
    if (IsSentencePieceTokenizerType(type_)) lowercase_ = false;
}

void Tokenizer::CheckCancelled() const {
    if (cancellation_query_ && cancellation_query_())
        throw std::runtime_error("Tokenizer cancelled");
}

void Tokenizer::ValidateVocabulary() const {
    if (IsSentencePieceTokenizerType(type_)) {
#if CYXWIZ_HAS_SENTENCEPIECE
        if (!sentencepiece_ || !sentencepiece_->loaded) {
            throw std::invalid_argument("SentencePiece tokenizer requires a loaded tokenizer/model.spm artifact");
        }
        return;
#else
        throw std::invalid_argument(SentencePieceTokenizerUnavailableMessage());
#endif
    }
    if (type_ == TokenizerType::ByteBPE && !vocab_.IsByteBPE())
        throw std::invalid_argument("Tokenizer strategy and vocabulary artifact do not match");
    if (type_ != TokenizerType::ByteBPE && vocab_.IsByteBPE())
        throw std::invalid_argument("Tokenizer strategy and vocabulary artifact do not match");
    if (type_ == TokenizerType::ByteBPE && lowercase_)
        throw std::invalid_argument("Byte BPE requires lowercase=false to preserve input bytes");
}

std::vector<std::string> Tokenizer::Split(const std::string& text) const {
    CheckCancelled();
    ValidateVocabulary();
    if (type_ == TokenizerType::ByteBPE) return SplitByteBPE(text);
    if (type_ == TokenizerType::WordPiece) return SplitWordPiece(text);
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
        case TokenizerType::ByteBPE: break; // handled above
        case TokenizerType::WordPiece: break; // handled above
        case TokenizerType::SentencePieceBPE: break; // rejected by ValidateVocabulary
        case TokenizerType::SentencePieceUnigram: break; // rejected by ValidateVocabulary
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
    if (IsSentencePieceTokenizerType(type_)) {
        CheckCancelled();
        ValidateVocabulary();
#if CYXWIZ_HAS_SENTENCEPIECE
        std::vector<int> ids;
        const auto status = sentencepiece_->processor.Encode(text, &ids);
        if (!status.ok()) {
            throw std::runtime_error("SentencePiece encode failed: " + status.ToString());
        }
        if (add_bos_) {
            const int bos = sentencepiece_->processor.bos_id();
            if (bos >= 0) ids.insert(ids.begin(), bos);
        }
        if (add_eos_) {
            const int eos = sentencepiece_->processor.eos_id();
            if (eos >= 0) ids.push_back(eos);
        }
        if (do_truncation_ && max_length_ > 0 && static_cast<int>(ids.size()) > max_length_) {
            ids.resize(max_length_);
        }
        const int pad = sentencepiece_->processor.pad_id();
        if (do_padding_ && pad >= 0 && max_length_ > 0 &&
            static_cast<int>(ids.size()) < max_length_) {
            ids.resize(max_length_, pad);
        }
        return ids;
#else
        throw std::invalid_argument(SentencePieceTokenizerUnavailableMessage());
#endif
    }

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
    ValidateVocabulary();
    if (IsSentencePieceTokenizerType(type_)) {
#if CYXWIZ_HAS_SENTENCEPIECE
        std::vector<int> decode_ids;
        decode_ids.reserve(token_ids.size());
        const int pad = sentencepiece_->processor.pad_id();
        const int bos = sentencepiece_->processor.bos_id();
        const int eos = sentencepiece_->processor.eos_id();
        for (int id : token_ids) {
            if (id == pad || id == bos || id == eos) {
                continue;
            }
            decode_ids.push_back(id);
        }
        std::string text;
        const auto status = sentencepiece_->processor.Decode(decode_ids, &text);
        if (!status.ok()) {
            throw std::runtime_error("SentencePiece decode failed: " + status.ToString());
        }
        return text;
#else
        throw std::invalid_argument(SentencePieceTokenizerUnavailableMessage());
#endif
    }

    std::string result;
    for (size_t i = 0; i < token_ids.size(); i++) {
        int id = token_ids[i];
        // Skip special tokens
        if (id == vocab_.PadIndex() || id == vocab_.BosIndex() || id == vocab_.EosIndex()) {
            continue;
        }
        const std::string token = vocab_.IndexToWord(id);
        if (type_ == TokenizerType::WordPiece) {
            if (token.rfind("##", 0) == 0) {
                result += token.substr(2);
            } else {
                if (!result.empty()) result += " ";
                result += token;
            }
        } else {
            if (type_ != TokenizerType::Character && type_ != TokenizerType::ByteBPE && !result.empty()) result += " ";
            result += token;
        }
    }
    return result;
}

void Tokenizer::LoadSentencePieceModelFromSerialized(const std::string& model_bytes) {
    if (!IsSentencePieceTokenizerType(type_)) {
        throw std::invalid_argument("SentencePiece model artifacts require a SentencePiece tokenizer type");
    }
#if CYXWIZ_HAS_SENTENCEPIECE
    auto state = std::make_shared<SentencePieceTokenizerState>();
    const auto status = state->processor.LoadFromSerializedProto(model_bytes);
    if (!status.ok()) {
        throw std::invalid_argument("Invalid SentencePiece model artifact: " + status.ToString());
    }
    state->loaded = true;
    state->model_bytes = model_bytes;
    sentencepiece_ = std::move(state);
#else
    (void)model_bytes;
    throw std::invalid_argument(SentencePieceTokenizerUnavailableMessage());
#endif
}

bool Tokenizer::HasSentencePieceModel() const {
    return sentencepiece_ && sentencepiece_->loaded;
}

size_t Tokenizer::GetVocabularySize() const {
    if (IsSentencePieceTokenizerType(type_)) {
        ValidateVocabulary();
#if CYXWIZ_HAS_SENTENCEPIECE
        return static_cast<size_t>(std::max(0, sentencepiece_->processor.GetPieceSize()));
#else
        return 0;
#endif
    }
    return vocab_.Size();
}

int Tokenizer::GetPadId() const {
    if (IsSentencePieceTokenizerType(type_)) {
        ValidateVocabulary();
#if CYXWIZ_HAS_SENTENCEPIECE
        return sentencepiece_->processor.pad_id();
#else
        return -1;
#endif
    }
    return vocab_.PadIndex();
}

int Tokenizer::GetUnkId() const {
    if (IsSentencePieceTokenizerType(type_)) {
        ValidateVocabulary();
#if CYXWIZ_HAS_SENTENCEPIECE
        return sentencepiece_->processor.unk_id();
#else
        return -1;
#endif
    }
    return vocab_.UnkIndex();
}

int Tokenizer::GetBosId() const {
    if (IsSentencePieceTokenizerType(type_)) {
        ValidateVocabulary();
#if CYXWIZ_HAS_SENTENCEPIECE
        return sentencepiece_->processor.bos_id();
#else
        return -1;
#endif
    }
    return vocab_.BosIndex();
}

int Tokenizer::GetEosId() const {
    if (IsSentencePieceTokenizerType(type_)) {
        ValidateVocabulary();
#if CYXWIZ_HAS_SENTENCEPIECE
        return sentencepiece_->processor.eos_id();
#else
        return -1;
#endif
    }
    return vocab_.EosIndex();
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
    if (IsSentencePieceTokenizerType(type_)) {
        throw std::invalid_argument("SentencePiece training is not implemented yet; load an official tokenizer/model.spm artifact");
    }
    if (type_ == TokenizerType::ByteBPE) {
        TrainByteBPE(documents, min_freq, max_vocab_size);
        return;
    }
    if (type_ == TokenizerType::WordPiece) {
        TrainWordPiece(documents, min_freq, max_vocab_size);
        return;
    }
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
                 type_ == TokenizerType::Whitespace ? "whitespace" :
                 type_ == TokenizerType::WordPiece ? "wordpiece" :
                 type_ == TokenizerType::SentencePieceBPE ? "sentencepiece_bpe" :
                 type_ == TokenizerType::SentencePieceUnigram ? "sentencepiece_unigram" : "character");
}

void Tokenizer::TrainWordPiece(const std::vector<std::string>& documents,
                               int min_freq,
                               int max_vocab_size) {
    if (min_freq < 1) {
        throw std::invalid_argument("WordPiece requires min_freq>=1");
    }

    std::unordered_map<std::string, int> counts;
    Tokenizer word_tokenizer(TokenizerType::Word);
    word_tokenizer.SetLowercase(lowercase_);
    word_tokenizer.SetPadding(false);
    word_tokenizer.SetTruncation(false);

    for (const auto& doc : documents) {
        CheckCancelled();
        for (const auto& word : word_tokenizer.Split(doc)) {
            CheckCancelled();
            if (word.empty()) continue;
            ++counts[word];
            for (size_t end = 1; end <= word.size(); ++end) {
                ++counts[word.substr(0, end)];
            }
            for (size_t start = 1; start < word.size(); ++start) {
                for (size_t end = start + 1; end <= word.size(); ++end) {
                    ++counts["##" + word.substr(start, end - start)];
                }
            }
        }
    }

    std::vector<std::pair<std::string, int>> candidates;
    candidates.reserve(counts.size());
    for (const auto& [piece, count] : counts) {
        if (count >= min_freq) candidates.push_back({piece, count});
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) return a.second > b.second;
        if (a.first.size() != b.first.size()) return a.first.size() > b.first.size();
        return a.first < b.first;
    });

    const int special_count = 4;
    const int token_capacity = max_vocab_size > 0
        ? std::max(0, max_vocab_size - special_count)
        : static_cast<int>(candidates.size());
    if (static_cast<int>(candidates.size()) > token_capacity) {
        candidates.resize(static_cast<size_t>(token_capacity));
    }

    std::vector<std::string> pieces;
    pieces.reserve(candidates.size());
    for (const auto& [piece, count] : candidates) {
        (void)count;
        pieces.push_back(piece);
    }
    vocab_.SetVocabulary(pieces);
    spdlog::info("Tokenizer trained: vocab_size={}, type=wordpiece", vocab_.Size());
}

std::vector<std::string> Tokenizer::SplitWordPiece(const std::string& text) const {
    std::string processed = text;
    if (lowercase_) {
        processed = TextProcessing::ToLowercase(processed);
    }

    Tokenizer word_tokenizer(TokenizerType::Word);
    word_tokenizer.SetLowercase(false);
    word_tokenizer.SetPadding(false);
    word_tokenizer.SetTruncation(false);

    std::vector<std::string> output;
    for (const auto& word : word_tokenizer.Split(processed)) {
        CheckCancelled();
        if (word.empty()) continue;
        if (vocab_.HasWord(word)) {
            output.push_back(word);
            continue;
        }
        std::vector<std::string> pieces;
        size_t start = 0;
        bool failed = false;
        while (start < word.size()) {
            size_t end = word.size();
            std::string best;
            while (end > start) {
                std::string candidate = word.substr(start, end - start);
                if (start > 0) candidate = "##" + candidate;
                if (vocab_.HasWord(candidate)) {
                    best = std::move(candidate);
                    break;
                }
                --end;
            }
            if (best.empty()) {
                failed = true;
                break;
            }
            pieces.push_back(std::move(best));
            start = end;
        }
        if (failed) output.push_back("[UNK]");
        else output.insert(output.end(), pieces.begin(), pieces.end());
    }
    return output;
}

} // namespace cyxwiz

