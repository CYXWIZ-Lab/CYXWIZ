#pragma once

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <memory>
#include <string>
#include <vector>
#include "csv_ingestion_options.h"

namespace cyxwiz {

/**
 * ParquetBackedDataset - Disk-backed tabular dataset for larger-than-RAM use.
 *
 * Wraps a Parquet file on disk and reads it lazily via memory-mapped I/O.
 * Row groups are fetched on demand (not materialized up front), so the
 * in-memory footprint is bounded by the OS page cache, not the file size.
 * Used by ArrowDatasetBatcher's streaming training path for datasets that
 * won't fit in RAM.
 *
 * Intentionally a separate class from ArrowDataset so the two loading
 * strategies stay explicit at call sites. Both types are registered in
 * DataRegistry and the batcher picks which one it needs to read from.
 *
 * Typical flow:
 *   1. Caller picks a Parquet file path (either a user-chosen file or the
 *      output of CsvToParquet when a CSV is too big to fit in RAM).
 *   2. Open() creates a MemoryMappedFile on it and builds a FileReader.
 *   3. Training time: batcher calls ReadRowGroup(i) for each row group
 *      in a shuffled order, iterating within each one.
 */
class ParquetBackedDataset {
public:
    /**
     * Open an existing Parquet file.
     * Returns nullptr on failure (bad path, corrupt file, missing module).
     */
    static std::shared_ptr<ParquetBackedDataset> Open(const std::string& path,
                                                        const std::string& name);

    /**
     * Compute the canonical cache path for a given CSV file. The result is
     * stable per-CSV-path: the same absolute CSV path always maps to the
     * same cache file, different CSVs map to different files. Creates any
     * missing parent directories.
     *
     * Layout: <system_temp>/cyxwiz/cache/<basename>_<pathhash>.parquet
     */
    static std::string GetCacheFilePath(
        const std::string& csv_path,
        const std::string& parser_signature = {});

    /**
     * Return the cache directory path used by GetCacheFilePath. Exposed so
     * cache hygiene code (PruneCache) and diagnostics can locate the dir
     * without duplicating the layout logic.
     */
    static std::string GetCacheDir();

    /**
     * Walk the cache directory and remove cache files that:
     *   - have mtime older than max_age_days (mtime-based expiry), OR
     *   - exceed the total size cap (LRU eviction by mtime, oldest first).
     *
     * Files that fail to delete (e.g. Windows mmap-locked) are silently
     * skipped — the next prune will retry. Safe to call from a worker
     * thread; uses only filesystem operations.
     *
     * Defaults: 10 GB total cap, 30-day mtime expiry. The defaults are
     * deliberately generous for ML datasets and conservative on age, so
     * users don't get surprise cache wipes between sessions.
     */
    static void PruneCache(size_t max_total_bytes = 10ULL * 1024 * 1024 * 1024,
                            int max_age_days = 30);

    /**
     * Check whether an existing cache file is still valid for a given CSV
     * source. Returns true if the cache exists and its mtime is >= the
     * CSV's mtime. Returns false if the cache is missing or stale.
     */
    static bool IsCacheFresh(const std::string& csv_path, const std::string& cache_path);

    /**
     * Stream a CSV file through to a Parquet cache. Memory usage stays
     * bounded to one CSV batch at a time (via arrow::csv::StreamingReader),
     * so this can convert files much larger than available RAM.
     *
     * Uses Snappy compression by default — fast to encode/decode and
     * typically 3-5x smaller than the source CSV on numeric data.
     *
     * Returns true on success. On failure, the output file is left in an
     * indeterminate state and the caller should delete it if they need
     * a clean re-run.
     */
    static bool ConvertCsvToParquet(const std::string& csv_path,
                                     const std::string& parquet_path,
                                     bool has_header = true,
                                     char delimiter = ',',
                                     int skip_rows = 0,
                                     int64_t max_rows = 0,
                                     const std::vector<std::string>& missing_value_tokens =
                                         DefaultTabularMissingValueTokens(),
                                     const std::vector<std::string>& selected_columns = {});

    /**
     * Read a single row group as an in-memory Arrow Table.
     * The batcher calls this lazily and discards the result after draining.
     */
    std::shared_ptr<arrow::Table> ReadRowGroup(int row_group_idx) const;

    /**
     * Read a subset of row groups and concatenate. Useful for batch-level
     * prefetch where one logical batch straddles row group boundaries.
     */
    std::shared_ptr<arrow::Table> ReadRowGroups(const std::vector<int>& indices) const;

    // Metadata accessors - mirror ArrowDataset's shape where it makes sense,
    // so call sites that only need row/col/schema info can treat the two
    // types interchangeably via templates or small wrappers.
    const std::string& GetName() const { return name_; }
    const std::string& GetFilePath() const { return file_path_; }
    int64_t GetNumRows() const { return num_rows_; }
    int64_t GetNumColumns() const { return num_columns_; }
    std::vector<std::string> GetColumnNames() const;
    std::shared_ptr<arrow::Schema> GetSchema() const { return schema_; }

    // Row group metadata used by the batcher for shuffling.
    int GetNumRowGroups() const { return static_cast<int>(row_group_sizes_.size()); }
    int64_t GetRowGroupSize(int row_group_idx) const;

    // Approximate memory usage - returns the file size on disk. Actual RAM
    // depends on which pages the OS has cached at any given moment, which
    // is the whole point of the mmap approach. This is the worst-case.
    size_t GetMemoryUsage() const { return file_size_; }
    size_t GetFileSizeBytes() const { return file_size_; }

private:
    ParquetBackedDataset(const std::string& name, const std::string& path);

    std::string name_;
    std::string file_path_;

    // The memory-mapped file handle keeps the mapping alive for the lifetime
    // of the dataset. Arrow's FileReader reads through this mapping.
    std::shared_ptr<arrow::io::MemoryMappedFile> mmap_file_;
    std::unique_ptr<parquet::arrow::FileReader> reader_;

    std::shared_ptr<arrow::Schema> schema_;
    int64_t num_rows_ = 0;
    int64_t num_columns_ = 0;
    size_t file_size_ = 0;

    // Row count per row group, pre-computed in Open() so the batcher can
    // partition work without re-reading metadata every epoch.
    std::vector<int64_t> row_group_sizes_;
};

} // namespace cyxwiz
