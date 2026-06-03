#pragma once

#include "../data_registry.h"
#include <cyxwiz/tokenizer.h>
#include <string>
#include <vector>

namespace cyxwiz {

/**
 * Configuration for text dataset loading
 */
struct TextDatasetConfig {
    std::string text_column = "text";      // CSV column for text data
    std::string label_column = "label";    // CSV column for labels
    bool has_labels = false;

    // Tokenization settings
    TokenizerType tokenizer_type = TokenizerType::Word;
    int max_length = 512;
    bool do_padding = true;
    bool do_truncation = true;
    bool lowercase = true;

    // Vocabulary settings
    int min_word_freq = 1;
    int max_vocab_size = -1;       // -1 = unlimited
    std::string vocab_file;        // Pre-built vocabulary path (optional)
};

enum class TextDatasetLoadMode {
    Tokenized,
    RawOnly,
};

/**
 * TextDataset - Loads text data and tokenizes on-the-fly.
 *
 * Supports:
 * - Plain text files (one sample per line)
 * - CSV with text column
 * - JSON with text field
 * - Directory of text files (one file = one sample)
 *
 * GetItem() returns tokenized integer sequences as floats (for Dataset interface compatibility).
 */
class TextDataset : public Dataset {
public:
    TextDataset(const std::string& path,
                const TextDatasetConfig& config,
                TextDatasetLoadMode mode = TextDatasetLoadMode::Tokenized);
    ~TextDataset() override = default;

    // Dataset interface
    size_t Size() const override { return texts_.size(); }
    std::pair<std::vector<float>, int> GetItem(size_t index) const override;
    DatasetInfo GetInfo() const override;

    // Text-specific access
    const std::string& GetText(size_t index) const;
    int GetLabel(size_t index) const;
    std::vector<int> GetTokenIds(size_t index) const;

    // Tokenizer access
    const Tokenizer& GetTokenizer() const { return tokenizer_; }
    Tokenizer& GetTokenizer() { return tokenizer_; }

    size_t GetVocabSize() const {
        return tokenizer_initialized_ ? tokenizer_.GetVocabulary().Size() : 0;
    }
    bool IsTokenizerInitialized() const { return tokenizer_initialized_; }

private:
    std::vector<std::string> texts_;
    std::vector<int> labels_;
    // Cached unique class names in the order they were first seen during
    // CSV loading. Populated by LoadCSV when a label column is present;
    // consumed by GetInfo() so the caller gets real class names (e.g.
    // "Anxiety", "Depression") instead of synthetic "class_0".
    // Empty for unlabeled datasets.
    std::vector<std::string> class_names_cache_;
    TextDatasetConfig config_;
    Tokenizer tokenizer_;
    std::string source_path_;
    bool tokenizer_initialized_ = false;

    void LoadTextFile(const std::string& path);
    void LoadCSV(const std::string& path);
    void LoadJSON(const std::string& path);
    void LoadCorpus(const std::string& directory);
    void InitTokenizer();
};

} // namespace cyxwiz
