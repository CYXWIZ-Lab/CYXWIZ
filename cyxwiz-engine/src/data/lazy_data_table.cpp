#include "lazy_data_table.h"
#include <spdlog/spdlog.h>
#include <filesystem>
#include <sstream>
#include <algorithm>

namespace fs = std::filesystem;

namespace cyxwiz {

LazyDataTable::~LazyDataTable() {
    ClearCache();
}

bool LazyDataTable::ShouldUseLazyLoading(const std::string& filepath, size_t threshold_mb) {
    try {
        auto size = fs::file_size(filepath);
        return size > (threshold_mb * 1024 * 1024);
    } catch (...) {
        return false;
    }
}

bool LazyDataTable::IndexFile(const std::string& filepath, char delimiter,
                               std::function<void(float, const std::string&)> progress_callback) {
    filepath_ = filepath;
    delimiter_ = delimiter;
    headers_.clear();
    row_offsets_.clear();
    ClearCache();

    // Get file size
    try {
        file_size_ = fs::file_size(filepath);
    } catch (...) {
        spdlog::error("LazyDataTable: Cannot get file size: {}", filepath);
        return false;
    }

    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        spdlog::error("LazyDataTable: Cannot open file: {}", filepath);
        return false;
    }

    if (progress_callback) {
        progress_callback(0.0f, "Scanning file structure...");
    }

    std::string line;
    bool first_row = true;
    size_t bytes_read = 0;
    float last_progress = 0.0f;

    while (file) {
        // Record position before reading line
        std::streampos line_start = file.tellg();

        if (!std::getline(file, line)) break;
        if (line.empty()) continue;

        // Remove trailing \r if present (Windows line endings)
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        if (first_row) {
            // Parse headers
            std::stringstream ss(line);
            std::string token;
            while (std::getline(ss, token, delimiter)) {
                // Trim whitespace
                token.erase(0, token.find_first_not_of(" \t\r\n"));
                if (!token.empty()) {
                    token.erase(token.find_last_not_of(" \t\r\n") + 1);
                }
                headers_.push_back(token);
            }
            first_row = false;
        } else {
            // Store row offset
            row_offsets_.push_back(line_start);
        }

        // Progress reporting (every 1%)
        bytes_read += line.size() + 1;
        float progress = static_cast<float>(bytes_read) / file_size_;
        if (progress_callback && progress - last_progress >= 0.01f) {
            progress_callback(progress * 0.9f,
                "Indexing... " + std::to_string(row_offsets_.size()) + " rows");
            last_progress = progress;
        }
    }

    if (progress_callback) {
        progress_callback(1.0f, "Index complete");
    }

    double file_size_mb = static_cast<double>(file_size_) / (1024.0 * 1024.0);
    spdlog::info("LazyDataTable: Indexed {} rows, {} columns from {} ({:.1f} MB)",
                 row_offsets_.size(), headers_.size(), filepath, file_size_mb);

    return !row_offsets_.empty();
}

LazyDataTable::Row LazyDataTable::ParseLine(const std::string& line) const {
    Row row;
    row.reserve(headers_.size());

    std::stringstream ss(line);
    std::string token;

    // Handle quoted CSV fields
    if (delimiter_ == ',') {
        bool in_quotes = false;
        std::string field;
        for (size_t i = 0; i < line.size(); i++) {
            char c = line[i];
            if (c == '"') {
                in_quotes = !in_quotes;
            } else if (c == delimiter_ && !in_quotes) {
                // Process field
                field.erase(0, field.find_first_not_of(" \t"));
                if (!field.empty()) {
                    field.erase(field.find_last_not_of(" \t") + 1);
                }

                if (field.empty()) {
                    row.push_back(std::monostate{});
                } else {
                    // Try to parse as number
                    try {
                        if (field.find('.') == std::string::npos &&
                            field.find('e') == std::string::npos &&
                            field.find('E') == std::string::npos) {
                            row.push_back(std::stoll(field));
                        } else {
                            row.push_back(std::stod(field));
                        }
                    } catch (...) {
                        row.push_back(field);
                    }
                }
                field.clear();
            } else {
                field += c;
            }
        }
        // Don't forget the last field
        field.erase(0, field.find_first_not_of(" \t"));
        if (!field.empty()) {
            field.erase(field.find_last_not_of(" \t") + 1);
        }
        if (field.empty()) {
            row.push_back(std::monostate{});
        } else {
            try {
                if (field.find('.') == std::string::npos &&
                    field.find('e') == std::string::npos &&
                    field.find('E') == std::string::npos) {
                    row.push_back(std::stoll(field));
                } else {
                    row.push_back(std::stod(field));
                }
            } catch (...) {
                row.push_back(field);
            }
        }
    } else {
        // Simple delimiter parsing (TSV, etc.)
        while (std::getline(ss, token, delimiter_)) {
            token.erase(0, token.find_first_not_of(" \t\r\n"));
            if (!token.empty()) {
                token.erase(token.find_last_not_of(" \t\r\n") + 1);
            }

            if (token.empty()) {
                row.push_back(std::monostate{});
            } else {
                try {
                    if (token.find('.') == std::string::npos &&
                        token.find('e') == std::string::npos &&
                        token.find('E') == std::string::npos) {
                        row.push_back(std::stoll(token));
                    } else {
                        row.push_back(std::stod(token));
                    }
                } catch (...) {
                    row.push_back(token);
                }
            }
        }
    }

    return row;
}

LazyDataTable::Row LazyDataTable::LoadRowFromDisk(size_t index) {
    if (index >= row_offsets_.size()) {
        return {};
    }

    std::ifstream file(filepath_, std::ios::binary);
    if (!file) {
        spdlog::error("LazyDataTable: Cannot open file for row {}", index);
        return {};
    }

    // Seek to row position
    file.seekg(row_offsets_[index]);

    std::string line;
    if (!std::getline(file, line)) {
        return {};
    }

    // Remove trailing \r
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }

    return ParseLine(line);
}

void LazyDataTable::AddToCache(size_t index, Row&& row) {
    std::lock_guard<std::mutex> lock(cache_mutex_);

    // Check if already cached
    auto it = cache_.find(index);
    if (it != cache_.end()) {
        // Move to front of LRU
        lru_list_.erase(it->second.lru_it);
        lru_list_.push_front(index);
        it->second.lru_it = lru_list_.begin();
        return;
    }

    // Evict if cache is full
    while (cache_.size() >= max_cache_size_ && !cache_.empty()) {
        EvictOldest();
    }

    // Add to cache
    lru_list_.push_front(index);
    cache_[index] = {lru_list_.begin(), std::move(row)};
}

LazyDataTable::Row* LazyDataTable::GetFromCache(size_t index) {
    std::lock_guard<std::mutex> lock(cache_mutex_);

    auto it = cache_.find(index);
    if (it == cache_.end()) {
        return nullptr;
    }

    // Move to front of LRU
    lru_list_.erase(it->second.lru_it);
    lru_list_.push_front(index);
    it->second.lru_it = lru_list_.begin();

    return &it->second.data;
}

void LazyDataTable::EvictOldest() {
    // Must be called with cache_mutex_ held
    if (lru_list_.empty()) return;

    size_t oldest = lru_list_.back();
    lru_list_.pop_back();
    cache_.erase(oldest);
}

void LazyDataTable::ClearCache() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cache_.clear();
    lru_list_.clear();
}

LazyDataTable::Row LazyDataTable::GetRow(size_t index) {
    if (index >= row_offsets_.size()) {
        return {};
    }

    // Check cache first
    Row* cached = GetFromCache(index);
    if (cached) {
        return *cached;  // Return copy
    }

    // Load from disk
    Row row = LoadRowFromDisk(index);
    if (!row.empty()) {
        Row copy = row;  // Make copy before moving
        AddToCache(index, std::move(row));
        return copy;
    }

    return {};
}

LazyDataTable::CellValue LazyDataTable::GetCell(size_t row, size_t col) {
    Row r = GetRow(row);
    if (col >= r.size()) {
        return std::monostate{};
    }
    return r[col];
}

std::string LazyDataTable::GetCellAsString(size_t row, size_t col) {
    auto cell = GetCell(row, col);

    return std::visit([](auto&& arg) -> std::string {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::string>) {
            return arg;
        } else if constexpr (std::is_same_v<T, double>) {
            char buf[32];
            snprintf(buf, sizeof(buf), "%.6g", arg);
            return buf;
        } else if constexpr (std::is_same_v<T, int64_t>) {
            return std::to_string(arg);
        } else {
            return "";
        }
    }, cell);
}

void LazyDataTable::PreloadRange(size_t start_row, size_t end_row) {
    start_row = std::min(start_row, row_offsets_.size());
    end_row = std::min(end_row, row_offsets_.size());

    for (size_t i = start_row; i < end_row; i++) {
        GetRow(i);  // This will cache the row
    }
}

} // namespace cyxwiz
