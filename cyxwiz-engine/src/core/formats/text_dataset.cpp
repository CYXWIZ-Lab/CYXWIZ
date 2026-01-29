#include "text_dataset.h"
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace fs = std::filesystem;

namespace cyxwiz {

TextDataset::TextDataset(const std::string& path, const TextDatasetConfig& config)
    : config_(config), tokenizer_(config.tokenizer_type), source_path_(path) {

    // Detect format and load
    fs::path p(path);
    if (fs::is_directory(p)) {
        LoadCorpus(path);
    } else {
        auto ext = p.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".csv" || ext == ".tsv") {
            LoadCSV(path);
        } else if (ext == ".json" || ext == ".jsonl") {
            LoadJSON(path);
        } else {
            LoadTextFile(path);
        }
    }

    if (texts_.empty()) {
        spdlog::warn("TextDataset: No text samples loaded from {}", path);
        return;
    }

    // Initialize tokenizer
    InitTokenizer();

    spdlog::info("TextDataset loaded: {} samples, vocab_size={}, max_length={}",
                 texts_.size(), tokenizer_.GetVocabulary().Size(), config_.max_length);
}

void TextDataset::InitTokenizer() {
    tokenizer_.SetLowercase(config_.lowercase);
    tokenizer_.SetMaxLength(config_.max_length);
    tokenizer_.SetPadding(config_.do_padding);
    tokenizer_.SetTruncation(config_.do_truncation);

    if (!config_.vocab_file.empty() && fs::exists(config_.vocab_file)) {
        tokenizer_.GetVocabulary().LoadFromFile(config_.vocab_file);
    } else {
        tokenizer_.Train(texts_, config_.min_word_freq, config_.max_vocab_size);
    }
}

std::pair<std::vector<float>, int> TextDataset::GetItem(size_t index) const {
    if (index >= texts_.size()) {
        return {{}, -1};
    }

    auto token_ids = tokenizer_.Encode(texts_[index]);

    // Convert int ids to float for Dataset interface compatibility
    std::vector<float> data(token_ids.begin(), token_ids.end());

    int label = config_.has_labels && index < labels_.size() ? labels_[index] : -1;
    return {data, label};
}

DatasetInfo TextDataset::GetInfo() const {
    DatasetInfo info;
    info.name = fs::path(source_path_).stem().string();
    info.path = source_path_;
    info.type = DatasetType::TXT;
    info.num_samples = texts_.size();
    info.shape = {static_cast<size_t>(config_.max_length)};
    info.is_loaded = true;

    if (config_.has_labels && !labels_.empty()) {
        int max_label = *std::max_element(labels_.begin(), labels_.end());
        info.num_classes = static_cast<size_t>(max_label + 1);
    }

    // Estimate memory: texts + token cache
    size_t mem = 0;
    for (const auto& t : texts_) {
        mem += t.size();
    }
    mem += texts_.size() * config_.max_length * sizeof(int); // tokenized cache estimate
    info.memory_usage = mem;

    return info;
}

const std::string& TextDataset::GetText(size_t index) const {
    static const std::string empty;
    return (index < texts_.size()) ? texts_[index] : empty;
}

std::vector<int> TextDataset::GetTokenIds(size_t index) const {
    if (index >= texts_.size()) return {};
    return tokenizer_.Encode(texts_[index]);
}

// ============================================================================
// Loading implementations
// ============================================================================

void TextDataset::LoadTextFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        spdlog::error("TextDataset: Cannot open file {}", path);
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines
        if (line.empty() || std::all_of(line.begin(), line.end(), ::isspace)) {
            continue;
        }
        texts_.push_back(line);
        labels_.push_back(-1);
    }
}

void TextDataset::LoadCSV(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        spdlog::error("TextDataset: Cannot open CSV file {}", path);
        return;
    }

    char delimiter = path.ends_with(".tsv") ? '\t' : ',';

    // Read header
    std::string header_line;
    std::getline(file, header_line);

    std::vector<std::string> headers;
    std::istringstream hss(header_line);
    std::string col;
    while (std::getline(hss, col, delimiter)) {
        // Trim whitespace
        col.erase(0, col.find_first_not_of(" \t\r\n"));
        col.erase(col.find_last_not_of(" \t\r\n") + 1);
        headers.push_back(col);
    }

    // Find text and label column indices
    int text_idx = -1, label_idx = -1;
    for (int i = 0; i < static_cast<int>(headers.size()); i++) {
        if (headers[i] == config_.text_column) text_idx = i;
        if (headers[i] == config_.label_column) label_idx = i;
    }

    if (text_idx < 0) {
        // Fallback: use first column as text
        text_idx = 0;
        spdlog::warn("TextDataset: Column '{}' not found, using column 0", config_.text_column);
    }

    // Read rows
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::vector<std::string> fields;
        std::istringstream lss(line);
        std::string field;
        while (std::getline(lss, field, delimiter)) {
            // Trim
            field.erase(0, field.find_first_not_of(" \t\r\n\""));
            field.erase(field.find_last_not_of(" \t\r\n\"") + 1);
            fields.push_back(field);
        }

        if (text_idx < static_cast<int>(fields.size())) {
            texts_.push_back(fields[text_idx]);

            if (label_idx >= 0 && label_idx < static_cast<int>(fields.size())) {
                config_.has_labels = true;
                try {
                    labels_.push_back(std::stoi(fields[label_idx]));
                } catch (...) {
                    labels_.push_back(-1);
                }
            } else {
                labels_.push_back(-1);
            }
        }
    }
}

void TextDataset::LoadJSON(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        spdlog::error("TextDataset: Cannot open JSON file {}", path);
        return;
    }

    // Try JSONL (one JSON object per line) first
    std::string line;
    bool is_jsonl = false;

    // Peek first character
    char first = file.peek();
    if (first == '{') {
        is_jsonl = true;
    }

    file.seekg(0);

    if (is_jsonl) {
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            try {
                auto obj = nlohmann::json::parse(line);
                if (obj.contains(config_.text_column)) {
                    texts_.push_back(obj[config_.text_column].get<std::string>());
                    if (obj.contains(config_.label_column)) {
                        config_.has_labels = true;
                        labels_.push_back(obj[config_.label_column].get<int>());
                    } else {
                        labels_.push_back(-1);
                    }
                }
            } catch (const std::exception& e) {
                spdlog::warn("TextDataset: Skipping malformed JSON line: {}", e.what());
            }
        }
    } else {
        // Array of objects
        try {
            auto data = nlohmann::json::parse(file);
            if (data.is_array()) {
                for (const auto& obj : data) {
                    if (obj.contains(config_.text_column)) {
                        texts_.push_back(obj[config_.text_column].get<std::string>());
                        if (obj.contains(config_.label_column)) {
                            config_.has_labels = true;
                            labels_.push_back(obj[config_.label_column].get<int>());
                        } else {
                            labels_.push_back(-1);
                        }
                    }
                }
            }
        } catch (const std::exception& e) {
            spdlog::error("TextDataset: Failed to parse JSON: {}", e.what());
        }
    }
}

void TextDataset::LoadCorpus(const std::string& directory) {
    // Each subdirectory = class label, each file = one sample
    // Or flat directory: each file = one sample (no labels)

    std::vector<fs::path> subdirs;
    std::vector<fs::path> files;

    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_directory()) {
            subdirs.push_back(entry.path());
        } else if (entry.is_regular_file()) {
            auto ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".txt" || ext == ".text" || ext == ".md") {
                files.push_back(entry.path());
            }
        }
    }

    if (!subdirs.empty()) {
        // Labeled corpus: subdirectory names are labels
        config_.has_labels = true;
        std::map<std::string, int> label_map;
        int label_counter = 0;

        std::sort(subdirs.begin(), subdirs.end());
        for (const auto& subdir : subdirs) {
            std::string label_name = subdir.filename().string();
            label_map[label_name] = label_counter++;

            for (const auto& entry : fs::recursive_directory_iterator(subdir)) {
                if (!entry.is_regular_file()) continue;
                auto ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".txt" || ext == ".text" || ext == ".md") {
                    std::ifstream f(entry.path());
                    std::string content((std::istreambuf_iterator<char>(f)),
                                        std::istreambuf_iterator<char>());
                    if (!content.empty()) {
                        texts_.push_back(content);
                        labels_.push_back(label_map[label_name]);
                    }
                }
            }
        }
    } else {
        // Flat directory: no labels
        std::sort(files.begin(), files.end());
        for (const auto& fp : files) {
            std::ifstream f(fp);
            std::string content((std::istreambuf_iterator<char>(f)),
                                std::istreambuf_iterator<char>());
            if (!content.empty()) {
                texts_.push_back(content);
                labels_.push_back(-1);
            }
        }
    }
}

} // namespace cyxwiz
