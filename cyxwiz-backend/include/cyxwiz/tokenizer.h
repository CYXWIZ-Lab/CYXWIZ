#pragma once

#include "api_export.h"
#include <iosfwd>
#include <array>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>

namespace cyxwiz {

struct SentencePieceTokenizerState;

// Tokenizer strategies
enum class TokenizerType {
    Whitespace,     // Split on whitespace
    Word,           // Word-level (handles punctuation)
    Character,          // Byte-level
    ByteBPE,            // Reversible byte BPE, persisted merge ranks (strategy 3)
    WordPiece,          // Greedy WordPiece with ## continuation pieces (strategy 4)
    SentencePieceBPE,   // Optional official SentencePiece BPE provider (strategy 5)
    SentencePieceUnigram // Optional official SentencePiece Unigram provider (strategy 6)
};

CYXWIZ_API bool IsSentencePieceTokenizerType(TokenizerType type);
CYXWIZ_API bool IsSentencePieceTokenizerAvailable();
CYXWIZ_API std::string SentencePieceTokenizerUnavailableMessage();

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

    // Persistence. Printable legacy word files remain supported. Tokens that
    // contain control/non-ASCII bytes use a versioned, byte-safe encoding.
    // BPE v1 uses four specials, all 256 bytes, then learned tokens. Merge
    // triples are {left, right, result}, ordered by rank. Validation is atomic.
    bool SetByteBPE(const std::vector<std::string>& words,
                    const std::vector<std::array<int, 3>>& merges);
    bool IsByteBPE() const { return byte_bpe_; }
    const std::vector<std::array<int, 3>>& GetBPEMerges() const { return bpe_merges_; }
    const std::map<std::pair<int, int>, std::pair<int, int>>& GetBPERanks() const {
        return bpe_ranks_;
    }
    bool LoadFromStream(std::istream& input);
    bool SaveToStream(std::ostream& output) const;
    static bool ReadTokens(std::istream& input, std::vector<std::string>& tokens);
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

    bool byte_bpe_ = false;
    std::vector<std::array<int, 3>> bpe_merges_;
    std::map<std::pair<int, int>, std::pair<int, int>> bpe_ranks_;
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

    // Optional provider-backed tokenizer model. Currently used for
    // SentencePiece .model bytes when CYXWIZ_HAS_SENTENCEPIECE is enabled.
    void LoadSentencePieceModelFromSerialized(const std::string& model_bytes);
    bool HasSentencePieceModel() const;

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

    // Host text-ingress cancellation. Training commits vocabulary only on success.
    void SetCancellationQuery(std::function<bool()> query) { cancellation_query_ = std::move(query); }
    void ValidateVocabulary() const;

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
    size_t GetVocabularySize() const;
    int GetPadId() const;
    int GetUnkId() const;
    int GetBosId() const;
    int GetEosId() const;

private:
    TokenizerType type_;
    Vocabulary vocab_;
    std::shared_ptr<SentencePieceTokenizerState> sentencepiece_;

    bool lowercase_ = true;
    int max_length_ = 512;
    bool do_padding_ = true;
    bool do_truncation_ = true;
    bool add_bos_ = false;
    bool add_eos_ = false;

    std::function<bool()> cancellation_query_;
    void CheckCancelled() const;
    void TrainByteBPE(const std::vector<std::string>& documents, int min_freq, int max_vocab_size);
    std::vector<std::string> SplitByteBPE(const std::string& text) const;
    void TrainWordPiece(const std::vector<std::string>& documents, int min_freq, int max_vocab_size);
    std::vector<std::string> SplitWordPiece(const std::string& text) const;

    // Split text into string tokens
    std::vector<std::string> Split(const std::string& text) const;
};

} // namespace cyxwiz
