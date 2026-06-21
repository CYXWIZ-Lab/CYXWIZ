#include "cyxwiz/tokenizer.h"

#include <spdlog/spdlog.h>

#include <fstream>
#include <string>

namespace cyxwiz {
bool Vocabulary::SaveToFile(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        spdlog::error("Failed to open vocabulary file for writing: {}", filepath);
        return false;
    }

    // Format: one word per line, index is line number
    for (const auto& word : idx_to_word_) {
        file << word << "\n";
    }

    spdlog::info("Vocabulary saved: {} words to {}", idx_to_word_.size(), filepath);
    return true;
}

bool Vocabulary::LoadFromFile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        spdlog::error("Failed to open vocabulary file: {}", filepath);
        return false;
    }

    word_to_idx_.clear();
    idx_to_word_.clear();

    std::string line;
    int idx = 0;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            word_to_idx_[line] = idx;
            idx_to_word_.push_back(line);
            idx++;
        }
    }

    // Update special token indices based on loaded data
    auto find_idx = [this](const std::string& token, int default_val) {
        auto it = word_to_idx_.find(token);
        return (it != word_to_idx_.end()) ? it->second : default_val;
    };
    pad_idx_ = find_idx("[PAD]", 0);
    unk_idx_ = find_idx("[UNK]", 1);
    bos_idx_ = find_idx("[BOS]", 2);
    eos_idx_ = find_idx("[EOS]", 3);

    spdlog::info("Vocabulary loaded: {} words from {}", idx_to_word_.size(), filepath);
    return true;
}

// ============================================================================
// Tokenizer
}  // namespace cyxwiz
