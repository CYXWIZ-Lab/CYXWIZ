#pragma once

#include "../data_registry.h"
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
};

/**
 * TextDataset - Loads raw text samples and their labels.
 *
 * Supports:
 * - Plain text files (one sample per line)
 * - CSV with text column
 * - JSON with text field
 * - Directory of text files (one file = one sample)
 *
 * It does not tokenize: the Text Tokenizer node does. GetItem() returns no
 * features, only the label.
 */
class TextDataset : public Dataset {
public:
    TextDataset(const std::string& path, const TextDatasetConfig& config);
    ~TextDataset() override = default;

    // Dataset interface
    size_t Size() const override { return texts_.size(); }
    std::pair<std::vector<float>, int> GetItem(size_t index) const override;
    DatasetInfo GetInfo() const override;

    // Text-specific access
    const std::string& GetText(size_t index) const;
    int GetLabel(size_t index) const;

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
    std::string source_path_;

    void LoadTextFile(const std::string& path);
    void LoadCSV(const std::string& path);
    void LoadJSON(const std::string& path);
    void LoadCorpus(const std::string& directory);
};

} // namespace cyxwiz
