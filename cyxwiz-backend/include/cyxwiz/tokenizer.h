#pragma once

#include "api_export.h"
#include <string>
#include <vector>
#include <map>
#include <unordered_map>

namespace cyxwiz {

// Tokenizer strategies
enum class TokenizerType {
    Whitespace,     // Split on whitespace
    Word,           // Word-level (handles punctuation)
    Character       // Character-level
};

// Result of tokenizing a single text
struct CYXWIZ_API TokenizedText {
    std::vector<int> token_ids;
    std::vector<std::string> tokens;
    size_t original_length = 0;
    bool truncated = false;
    bool padded = false;
};

// ============================================================================
// Vocabulary - word <-> index mapping with special tokens
// ============================================================================

class CYXWIZ_API Vocabulary {
public:
    Vocabulary();

    // Build vocabulary from a corpus
    void BuildFromDocuments(const std::vector<std::string>& documents,
                            int min_freq = 1,
                            int max_vocab_size = -1,
                            bool lowercase = true);

    // Manual vocabulary control
    void SetVocabulary(const std::vector<std::string>& words);
    int AddWord(const std::string& word);

    // Lookup
    int WordToIndex(const std::string& word) const;
    std::string IndexToWord(int index) const;
    bool HasWord(const std::string& word) const;

    // Special token indices
    int PadIndex() const { return pad_idx_; }
    int UnkIndex() const { return unk_idx_; }
    int BosIndex() const { return bos_idx_; }
    int EosIndex() const { return eos_idx_; }

    size_t Size() const { return idx_to_word_.size(); }

    // Persistence
    bool SaveToFile(const std::string& filepath) const;
    bool LoadFromFile(const std::string& filepath);

    // Get all words sorted by index
    const std::vector<std::string>& GetWords() const { return idx_to_word_; }

private:
    std::unordered_map<std::string, int> word_to_idx_;
    std::vector<std::string> idx_to_word_;

    int pad_idx_ = 0;   // [PAD]
    int unk_idx_ = 1;   // [UNK]
    int bos_idx_ = 2;   // [BOS]
    int eos_idx_ = 3;   // [EOS]

    void AddSpecialTokens();
};

// ============================================================================
// Tokenizer - text to integer sequences
// ============================================================================

class CYXWIZ_API Tokenizer {
public:
    explicit Tokenizer(TokenizerType type = TokenizerType::Word);

    // Encode text -> token ids
    std::vector<int> Encode(const std::string& text) const;

    // Decode token ids -> text
    std::string Decode(const std::vector<int>& token_ids) const;

    // Batch operations
    std::vector<std::vector<int>> EncodeBatch(const std::vector<std::string>& texts) const;
    std::vector<std::string> DecodeBatch(const std::vector<std::vector<int>>& batch) const;

    // Full tokenization with metadata
    TokenizedText Tokenize(const std::string& text) const;

    // Pad a batch to uniform length
    // Returns padded batch. If max_length <= 0, uses longest sequence in batch.
    std::vector<std::vector<int>> PadBatch(const std::vector<std::vector<int>>& batch,
                                            int max_length = -1) const;

    // Train tokenizer (builds vocabulary from documents)
    void Train(const std::vector<std::string>& documents,
               int min_freq = 1,
               int max_vocab_size = -1);

    // Vocabulary access
    void SetVocabulary(const Vocabulary& vocab) { vocab_ = vocab; }
    Vocabulary& GetVocabulary() { return vocab_; }
    const Vocabulary& GetVocabulary() const { return vocab_; }

    // Configuration
    void SetLowercase(bool v) { lowercase_ = v; }
    void SetMaxLength(int v) { max_length_ = v; }
    void SetPadding(bool v) { do_padding_ = v; }
    void SetTruncation(bool v) { do_truncation_ = v; }
    void SetAddBos(bool v) { add_bos_ = v; }
    void SetAddEos(bool v) { add_eos_ = v; }

    bool GetLowercase() const { return lowercase_; }
    int GetMaxLength() const { return max_length_; }
    bool GetPadding() const { return do_padding_; }
    bool GetTruncation() const { return do_truncation_; }
    TokenizerType GetType() const { return type_; }

private:
    TokenizerType type_;
    Vocabulary vocab_;

    bool lowercase_ = true;
    int max_length_ = 512;
    bool do_padding_ = true;
    bool do_truncation_ = true;
    bool add_bos_ = false;
    bool add_eos_ = false;

    // Split text into string tokens
    std::vector<std::string> Split(const std::string& text) const;
};

} // namespace cyxwiz
