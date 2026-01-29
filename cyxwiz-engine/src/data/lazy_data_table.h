#pragma once

#include "data_table.h"
#include <fstream>
#include <unordered_map>
#include <list>
#include <mutex>
#include <functional>

namespace cyxwiz {

/**
 * LazyDataTable - Memory-efficient table for large files
 *
 * Instead of loading all data into memory, this class:
 * 1. Scans the file to build a row offset index
 * 2. Loads rows on-demand when accessed
 * 3. Uses an LRU cache to keep recently accessed rows in memory
 *
 * Use for files > 10MB or > 100K rows
 */
class LazyDataTable {
public:
    using CellValue = DataTable::CellValue;
    using Row = DataTable::Row;

    LazyDataTable() = default;
    ~LazyDataTable();

    // Initialize from file (builds row index, doesn't load data)
    bool IndexFile(const std::string& filepath, char delimiter = ',',
                   std::function<void(float, const std::string&)> progress_callback = nullptr);

    // Access
    const std::vector<std::string>& GetHeaders() const { return headers_; }
    size_t GetRowCount() const { return row_offsets_.size(); }
    size_t GetColumnCount() const { return headers_.size(); }

    // Get row (loads from disk if not cached)
    Row GetRow(size_t index);
    CellValue GetCell(size_t row, size_t col);
    std::string GetCellAsString(size_t row, size_t col);

    // Preload rows (for smooth scrolling)
    void PreloadRange(size_t start_row, size_t end_row);

    // Cache management
    void SetCacheSize(size_t max_rows) { max_cache_size_ = max_rows; }
    size_t GetCacheSize() const { return max_cache_size_; }
    size_t GetCachedRowCount() const { return cache_.size(); }
    void ClearCache();

    // Metadata
    void SetName(const std::string& name) { name_ = name; }
    const std::string& GetName() const { return name_; }
    const std::string& GetFilePath() const { return filepath_; }
    size_t GetFileSize() const { return file_size_; }

    // Check if lazy loading is appropriate for a file
    static bool ShouldUseLazyLoading(const std::string& filepath, size_t threshold_mb = 10);

private:
    // Parse a single line into cells
    Row ParseLine(const std::string& line) const;

    // Load a row from disk
    Row LoadRowFromDisk(size_t index);

    // LRU cache operations
    void AddToCache(size_t index, Row&& row);
    Row* GetFromCache(size_t index);
    void EvictOldest();

    // File info
    std::string name_;
    std::string filepath_;
    char delimiter_ = ',';
    size_t file_size_ = 0;

    // Column headers
    std::vector<std::string> headers_;

    // Row offset index (byte position of each row in file)
    std::vector<std::streampos> row_offsets_;

    // LRU cache: maps row index to (iterator in lru_list_, row data)
    struct CacheEntry {
        std::list<size_t>::iterator lru_it;
        Row data;
    };
    std::unordered_map<size_t, CacheEntry> cache_;
    std::list<size_t> lru_list_;  // Most recently used at front
    size_t max_cache_size_ = 1000;  // Default: cache 1000 rows

    // Thread safety for cache
    mutable std::mutex cache_mutex_;
};

} // namespace cyxwiz
