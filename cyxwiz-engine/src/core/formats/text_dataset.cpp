#include "text_dataset.h"
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <map>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace fs = std::filesystem;

namespace cyxwiz {

namespace {

// RFC-4180-ish CSV reader that handles quoted fields, embedded delimiters,
// embedded newlines, and escaped `""` quotes. Returns one row as a vector
// of field strings. Reads from the stream until a row terminates at the
// top level (outside any quoted region).
//
// Returns false at EOF with no partial row; true when a row was parsed
// (even an empty one). Written inline because the rest of the codebase
// uses Arrow's CSV reader for the main tabular path — TextDataset is the
// only place that needs a plain std::string-field parser.
bool ReadCSVRow(std::istream& in, char delimiter,
                std::vector<std::string>& out_fields) {
    out_fields.clear();
    std::string field;
    bool in_quotes = false;
    bool any_char_read = false;
    int c;

    while ((c = in.get()) != EOF) {
        any_char_read = true;
        if (in_quotes) {
            if (c == '"') {
                // Escaped quote `""` → emit one literal `"` and stay in quotes.
                if (in.peek() == '"') {
                    field.push_back('"');
                    in.get();
                } else {
                    // Close quote.
                    in_quotes = false;
                }
            } else {
                field.push_back(static_cast<char>(c));
            }
        } else {
            if (c == '"') {
                in_quotes = true;
            } else if (c == delimiter) {
                out_fields.push_back(std::move(field));
                field.clear();
            } else if (c == '\r') {
                // Eat any \n that follows a \r so CRLF counts as one newline.
                if (in.peek() == '\n') in.get();
                out_fields.push_back(std::move(field));
                return true;
            } else if (c == '\n') {
                out_fields.push_back(std::move(field));
                return true;
            } else {
                field.push_back(static_cast<char>(c));
            }
        }
    }

    // EOF: flush the last field if we were in the middle of a row.
    if (any_char_read) {
        out_fields.push_back(std::move(field));
        return true;
    }
    return false;
}

} // namespace

TextDataset::TextDataset(const std::string& path,
                         const TextDatasetConfig& config,
                         TextDatasetLoadMode mode)
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

    if (mode == TextDatasetLoadMode::RawOnly) {
        spdlog::info("TextDataset raw-loaded: {} samples, max_length={}",
                     texts_.size(), config_.max_length);
        return;
    }

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
    tokenizer_initialized_ = true;
}

std::pair<std::vector<float>, int> TextDataset::GetItem(size_t index) const {
    if (index >= texts_.size()) {
        return {{}, -1};
    }

    int label = config_.has_labels && index < labels_.size() ? labels_[index] : -1;
    if (!tokenizer_initialized_) {
        return {{}, label};
    }

    auto token_ids = tokenizer_.Encode(texts_[index]);

    // Convert int ids to float for Dataset interface compatibility.
    std::vector<float> data;
    data.reserve(token_ids.size());
    for (int id : token_ids) {
        data.push_back(static_cast<float>(id));
    }

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

    // Prefer the class name cache populated by LoadCSV/LoadCorpus — it
    // has the real string labels. Fall back to max_label+1 for JSON
    // (which still uses int labels) until it's rewired to string labels.
    if (!class_names_cache_.empty()) {
        info.class_names = class_names_cache_;
        info.num_classes = class_names_cache_.size();
    } else if (config_.has_labels && !labels_.empty()) {
        int max_label = *std::max_element(labels_.begin(), labels_.end());
        if (max_label >= 0) {
            info.num_classes = static_cast<size_t>(max_label + 1);
        }
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

int TextDataset::GetLabel(size_t index) const {
    return (index < labels_.size()) ? labels_[index] : -1;
}

std::vector<int> TextDataset::GetTokenIds(size_t index) const {
    if (index >= texts_.size() || !tokenizer_initialized_) return {};
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

    // Parse the header row with the proper CSV reader so the column count
    // matches the rest of the file. The naive line-based parser we used
    // before miscounted columns whenever the header itself contained a
    // quoted/escaped field.
    std::vector<std::string> headers;
    if (!ReadCSVRow(file, delimiter, headers)) {
        spdlog::error("TextDataset: CSV file '{}' is empty", path);
        return;
    }
    for (auto& h : headers) {
        // Trim whitespace on header names so lookups are forgiving.
        while (!h.empty() && (h.front() == ' ' || h.front() == '\t' ||
                              h.front() == '\r' || h.front() == '\n')) {
            h.erase(h.begin());
        }
        while (!h.empty() && (h.back()  == ' ' || h.back()  == '\t' ||
                              h.back()  == '\r' || h.back()  == '\n')) {
            h.pop_back();
        }
    }

    // Find text and label column indices.
    int text_idx = -1, label_idx = -1;
    for (int i = 0; i < static_cast<int>(headers.size()); i++) {
        if (headers[i] == config_.text_column) text_idx = i;
        if (!config_.label_column.empty() && headers[i] == config_.label_column)
            label_idx = i;
    }

    if (text_idx < 0) {
        // Fallback: look for any column named "text" / "statement" / "sentence",
        // otherwise use column 0. Kaggle datasets sometimes use different
        // conventions so we're lenient here.
        for (int i = 0; i < static_cast<int>(headers.size()); i++) {
            const std::string& h = headers[i];
            if (h == "text" || h == "statement" || h == "sentence" ||
                h == "content" || h == "body") {
                text_idx = i;
                break;
            }
        }
        if (text_idx < 0) text_idx = 0;
        spdlog::warn("TextDataset: Column '{}' not found, using column {} ('{}')",
                     config_.text_column, text_idx,
                     text_idx < static_cast<int>(headers.size())
                         ? headers[text_idx] : "<out of range>");
    }

    // String labels: we map unique label strings to sequential integer IDs
    // as we encounter them. This mirrors how audio_dataset handles string
    // class names in its ClassSubdirs layout. The map is kept local to the
    // loader; the resolved int labels go into labels_ and the class name
    // list is cached for GetInfo() to return.
    //
    // If a label column was configured, we ASSUME string labels regardless
    // of content and always build a mapping — this is much less surprising
    // than the previous "try stoi and fall back to -1" behavior which
    // silently dropped every row when labels happened to be words.
    std::map<std::string, int> label_to_id;
    class_names_cache_.clear();
    const bool has_label_col = (label_idx >= 0);
    if (has_label_col) {
        config_.has_labels = true;
    }

    // Read rows with the proper RFC-4180-ish parser so quoted fields with
    // embedded commas, escaped quotes, and embedded newlines all parse as
    // a single row. This is the load-time fix for the 67,877-row / 8,401-
    // class bug on sentiment_mental_health.csv where the old naive parser
    // split quoted multi-line statements across rows and then tried to
    // interpret random text fragments as integer labels.
    std::vector<std::string> fields;
    size_t skipped_empty = 0;
    size_t skipped_short = 0;
    while (ReadCSVRow(file, delimiter, fields)) {
        // Completely empty trailing row (file ends with \n).
        if (fields.empty()) continue;
        if (fields.size() == 1 && fields[0].empty()) continue;

        if (text_idx >= static_cast<int>(fields.size())) {
            skipped_short++;
            continue;
        }

        const std::string& text = fields[text_idx];
        if (text.empty() ||
            std::all_of(text.begin(), text.end(), ::isspace)) {
            // Skip empty text rows — they'd tokenize to all-PAD and train
            // the model on nothing. Rate of skips gets reported at the end.
            skipped_empty++;
            continue;
        }

        texts_.push_back(text);

        if (has_label_col && label_idx < static_cast<int>(fields.size())) {
            const std::string& raw_label = fields[label_idx];
            auto it = label_to_id.find(raw_label);
            if (it == label_to_id.end()) {
                int new_id = static_cast<int>(class_names_cache_.size());
                label_to_id[raw_label] = new_id;
                class_names_cache_.push_back(raw_label);
                labels_.push_back(new_id);
            } else {
                labels_.push_back(it->second);
            }
        } else {
            labels_.push_back(-1);
        }
    }

    if (skipped_empty > 0 || skipped_short > 0) {
        spdlog::warn("TextDataset: skipped {} empty-text and {} short rows "
                     "while loading '{}'",
                     skipped_empty, skipped_short, path);
    }

    if (has_label_col) {
        spdlog::info("TextDataset: loaded {} rows, {} classes from '{}' "
                     "(text_col='{}' [{}], label_col='{}' [{}])",
                     texts_.size(), class_names_cache_.size(), path,
                     headers[text_idx], text_idx,
                     headers[label_idx], label_idx);
    } else {
        spdlog::info("TextDataset: loaded {} rows (unlabeled) from '{}' "
                     "(text_col='{}' [{}])",
                     texts_.size(), path, headers[text_idx], text_idx);
    }
}

void TextDataset::LoadJSON(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        spdlog::error("TextDataset: Cannot open JSON file {}", path);
        return;
    }

    // Shared label-string-to-int mapping used by both JSONL and JSON-array
    // paths. Labels can be either strings ("positive") or integers (3);
    // we handle both with the same normalization: ints become their
    // direct value as a stringified key, strings map by content. This
    // way "positive" and 0 both end up as proper class_names_cache_
    // entries instead of the old "throw on stoi, emit -1" failure mode.
    class_names_cache_.clear();
    std::map<std::string, int> label_to_id;
    auto resolve_label = [&](const nlohmann::json& lbl) -> int {
        std::string key;
        if (lbl.is_string()) {
            key = lbl.get<std::string>();
        } else if (lbl.is_number_integer()) {
            key = std::to_string(lbl.get<int64_t>());
        } else if (lbl.is_number_float()) {
            key = std::to_string(lbl.get<double>());
        } else if (lbl.is_boolean()) {
            key = lbl.get<bool>() ? "true" : "false";
        } else {
            return -1;
        }
        auto it = label_to_id.find(key);
        if (it == label_to_id.end()) {
            int new_id = static_cast<int>(class_names_cache_.size());
            label_to_id[key] = new_id;
            class_names_cache_.push_back(key);
            return new_id;
        }
        return it->second;
    };

    auto process_obj = [&](const nlohmann::json& obj) {
        if (!obj.contains(config_.text_column)) return;
        if (!obj[config_.text_column].is_string()) return;
        std::string text = obj[config_.text_column].get<std::string>();
        if (text.empty() ||
            std::all_of(text.begin(), text.end(), ::isspace)) {
            return;
        }
        texts_.push_back(std::move(text));
        if (!config_.label_column.empty() &&
            obj.contains(config_.label_column)) {
            config_.has_labels = true;
            labels_.push_back(resolve_label(obj[config_.label_column]));
        } else {
            labels_.push_back(-1);
        }
    };

    // Detect JSONL vs single-document JSON by peeking the first non-whitespace
    // byte. JSONL lines start with '{', a JSON document starts with '[' or '{'
    // at file scope. Treat JSONL as the default when we see '{' because
    // picking the wrong parser there loses 99% of the rows.
    int first = file.peek();
    bool is_jsonl = (first == '{');
    file.seekg(0);

    if (is_jsonl) {
        std::string line;
        size_t bad_lines = 0;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            try {
                auto obj = nlohmann::json::parse(line);
                process_obj(obj);
            } catch (const std::exception& e) {
                bad_lines++;
                if (bad_lines <= 5) {
                    spdlog::warn("TextDataset: Skipping malformed JSONL line: {}", e.what());
                }
            }
        }
        if (bad_lines > 5) {
            spdlog::warn("TextDataset: total {} malformed JSONL lines skipped", bad_lines);
        }
    } else {
        try {
            auto data = nlohmann::json::parse(file);
            if (data.is_array()) {
                for (const auto& obj : data) {
                    process_obj(obj);
                }
            } else if (data.is_object()) {
                process_obj(data);
            }
        } catch (const std::exception& e) {
            spdlog::error("TextDataset: Failed to parse JSON: {}", e.what());
        }
    }

    spdlog::info("TextDataset: loaded {} rows, {} classes from JSON '{}' "
                 "(text_col='{}', label_col='{}')",
                 texts_.size(), class_names_cache_.size(), path,
                 config_.text_column, config_.label_column);
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
        // Labeled corpus: subdirectory names are labels. Analogous to
        // ImageFolder for text — folder/<class>/*.txt. Subdirs are sorted
        // alphabetically so class IDs are stable across runs. Populates
        // class_names_cache_ so GetInfo() can return real class names.
        config_.has_labels = true;
        class_names_cache_.clear();

        std::sort(subdirs.begin(), subdirs.end());
        size_t files_loaded = 0;
        size_t files_empty = 0;
        for (const auto& subdir : subdirs) {
            std::string label_name = subdir.filename().string();
            int label_id = static_cast<int>(class_names_cache_.size());
            class_names_cache_.push_back(label_name);

            for (const auto& entry : fs::recursive_directory_iterator(subdir)) {
                if (!entry.is_regular_file()) continue;
                auto ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".txt" || ext == ".text" || ext == ".md") {
                    std::ifstream f(entry.path());
                    std::string content((std::istreambuf_iterator<char>(f)),
                                        std::istreambuf_iterator<char>());
                    if (content.empty() ||
                        std::all_of(content.begin(), content.end(), ::isspace)) {
                        files_empty++;
                        continue;
                    }
                    texts_.push_back(std::move(content));
                    labels_.push_back(label_id);
                    files_loaded++;
                }
            }
        }
        spdlog::info("TextDataset: corpus-subdirs layout loaded {} files "
                     "across {} classes from '{}'{}",
                     files_loaded, class_names_cache_.size(), directory,
                     files_empty > 0
                         ? " (skipped " + std::to_string(files_empty) + " empty files)"
                         : "");
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
