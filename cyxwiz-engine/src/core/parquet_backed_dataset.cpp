#include "parquet_backed_dataset.h"
#include "data_budget.h"

#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/metadata.h>
#include <parquet/properties.h>
#include <arrow/csv/api.h>
#include <arrow/io/file.h>
#include <arrow/io/memory.h>
#include <arrow/record_batch.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <thread>
#include <vector>

namespace cyxwiz {

namespace fs = std::filesystem;

namespace {

uint64_t StableCacheHash(const std::string& value) {
    uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char byte : value) {
        hash ^= static_cast<uint64_t>(byte);
        hash *= 1099511628211ULL;
    }
    return hash;
}

}  // namespace

ParquetBackedDataset::ParquetBackedDataset(const std::string& name, const std::string& path)
    : name_(name), file_path_(path) {}

std::shared_ptr<ParquetBackedDataset> ParquetBackedDataset::Open(
    const std::string& path, const std::string& name) {

    if (!fs::exists(path)) {
        spdlog::error("ParquetBackedDataset::Open: file not found: {}", path);
        return nullptr;
    }

    // Memory-map the file. The OS page cache handles eviction and prefetch
    // automatically - training reads that touch a page fault it in, reads
    // that repeat stay hot, cold pages get evicted under memory pressure.
    auto maybe_mmap = arrow::io::MemoryMappedFile::Open(path, arrow::io::FileMode::READ);
    if (!maybe_mmap.ok()) {
        spdlog::error("ParquetBackedDataset::Open: MemoryMappedFile::Open failed for {}: {}",
                      path, maybe_mmap.status().ToString());
        return nullptr;
    }
    auto mmap_file = maybe_mmap.ValueOrDie();

    // Build the Parquet reader on top of the mapped file. Same API we
    // already use in ArrowDataset::FromParquet, just with an mmap source
    // instead of ReadableFile::Open.
    auto maybe_reader = parquet::arrow::OpenFile(mmap_file, arrow::default_memory_pool());
    if (!maybe_reader.ok()) {
        spdlog::error("ParquetBackedDataset::Open: OpenFile failed for {}: {}",
                      path, maybe_reader.status().ToString());
        return nullptr;
    }
    auto reader = std::move(maybe_reader).ValueOrDie();

    // Read schema up front so downstream code doesn't need to touch the
    // file for simple metadata queries.
    std::shared_ptr<arrow::Schema> schema;
    auto status = reader->GetSchema(&schema);
    if (!status.ok() || !schema) {
        spdlog::error("ParquetBackedDataset::Open: GetSchema failed for {}: {}",
                      path, status.ToString());
        return nullptr;
    }

    // Grab file-level metadata (row count, row group sizes) so the batcher
    // can shuffle and partition without re-parsing on every epoch.
    auto parquet_reader = reader->parquet_reader();
    if (!parquet_reader) {
        spdlog::error("ParquetBackedDataset::Open: parquet_reader() returned null for {}", path);
        return nullptr;
    }
    auto metadata = parquet_reader->metadata();
    if (!metadata) {
        spdlog::error("ParquetBackedDataset::Open: metadata() returned null for {}", path);
        return nullptr;
    }

    const int num_row_groups = metadata->num_row_groups();
    std::vector<int64_t> row_group_sizes;
    row_group_sizes.reserve(num_row_groups);
    int64_t total_rows = 0;
    uint64_t uncompressed_bytes = 0;
    for (int i = 0; i < num_row_groups; ++i) {
        auto rg_metadata = metadata->RowGroup(i);
        int64_t rg_rows = rg_metadata ? rg_metadata->num_rows() : 0;
        row_group_sizes.push_back(rg_rows);
        total_rows += rg_rows;
        if (rg_metadata && rg_metadata->total_byte_size() > 0)
            uncompressed_bytes += static_cast<uint64_t>(rg_metadata->total_byte_size());
    }

    size_t file_size = 0;
    try {
        file_size = static_cast<size_t>(fs::file_size(path));
    } catch (...) {
        file_size = 0;
    }

    // Private constructor then fill in the rest - using shared_ptr::make
    // isn't an option because of the private ctor, so we new+wrap.
    auto ds = std::shared_ptr<ParquetBackedDataset>(new ParquetBackedDataset(name, path));
    ds->mmap_file_ = mmap_file;
    ds->reader_ = std::move(reader);
    ds->schema_ = schema;
    ds->num_rows_ = total_rows;
    ds->num_columns_ = metadata->num_columns();
    ds->row_group_sizes_ = std::move(row_group_sizes);
    ds->file_size_ = file_size;
    ds->uncompressed_bytes_ = uncompressed_bytes;

    spdlog::info("ParquetBackedDataset: opened '{}' ({} rows, {} cols, {} row groups, {:.1f} MB on disk)",
                 path, ds->num_rows_, ds->num_columns_,
                 num_row_groups,
                 file_size / (1024.0 * 1024.0));

    return ds;
}

std::shared_ptr<arrow::Table> ParquetBackedDataset::ReadRowGroup(int row_group_idx) const {
    if (!reader_) return nullptr;
    if (row_group_idx < 0 || row_group_idx >= static_cast<int>(row_group_sizes_.size())) {
        spdlog::warn("ParquetBackedDataset::ReadRowGroup: out of range idx={}, num={}",
                     row_group_idx, row_group_sizes_.size());
        return nullptr;
    }

    std::shared_ptr<arrow::Table> table;
    std::lock_guard<std::mutex> lock(reader_mutex_);
    auto status = reader_->ReadRowGroup(row_group_idx, &table);
    if (!status.ok()) {
        spdlog::error("ParquetBackedDataset::ReadRowGroup({}): {}", row_group_idx, status.ToString());
        return nullptr;
    }
    return table;
}

std::shared_ptr<arrow::Table> ParquetBackedDataset::ReadRowGroups(
    const std::vector<int>& indices) const {
    if (!reader_ || indices.empty()) return nullptr;

    std::shared_ptr<arrow::Table> table;
    std::lock_guard<std::mutex> lock(reader_mutex_);
    auto status = reader_->ReadRowGroups(indices, &table);
    if (!status.ok()) {
        spdlog::error("ParquetBackedDataset::ReadRowGroups: {}", status.ToString());
        return nullptr;
    }
    return table;
}

std::shared_ptr<arrow::Table> ParquetBackedDataset::ReadAll() const {
    std::vector<int> all(row_group_sizes_.size());
    for (size_t i = 0; i < all.size(); ++i) all[i] = static_cast<int>(i);
    if (all.empty()) return arrow::Table::MakeEmpty(schema_).ValueOr(nullptr);
    return ReadRowGroups(all);
}

std::shared_ptr<arrow::Table> ParquetBackedDataset::ReadSpreadRowGroups(int64_t max_rows) const {
    const int groups = GetNumRowGroups();
    if (groups == 0 || num_rows_ <= max_rows) return ReadAll();
    const int64_t average = std::max<int64_t>(1, num_rows_ / groups);
    const int wanted = static_cast<int>(std::clamp<int64_t>((max_rows + average - 1) / average, 1, groups));
    std::vector<int> indices;
    for (int i = 0; i < wanted; ++i) {
        const int index = static_cast<int>(static_cast<int64_t>(i) * groups / wanted);
        if (indices.empty() || indices.back() != index) indices.push_back(index);
    }
    return ReadRowGroups(indices);
}

bool ParquetBackedDataset::WriteColumnProjection(const std::string& source_path,
                                                 const std::string& output_path,
                                                 const std::vector<std::string>& columns,
                                                 std::string& error) {
    // One row group at a time, only the selected columns read: memory stays at
    // one group whatever the file size.
    auto source = Open(source_path, "projection_source");
    if (!source) {
        error = "cannot open '" + source_path + "'";
        return false;
    }
    std::vector<int> indices;
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (const auto& column : columns) {
        const int index = source->schema_->GetFieldIndex(column);
        if (index < 0) {
            error = "column '" + column + "' is not in '" + source_path + "'";
            return false;
        }
        indices.push_back(index);
        fields.push_back(source->schema_->field(index));
    }
    const std::string tmp_path = UniqueTempPath(output_path);
    std::error_code ignored;
    fs::create_directories(fs::path(output_path).parent_path(), ignored);
    auto out = arrow::io::FileOutputStream::Open(tmp_path);
    if (!out.ok()) {
        error = out.status().ToString();
        return false;
    }
    auto props = parquet::WriterProperties::Builder().compression(parquet::Compression::SNAPPY)->build();
    auto writer = parquet::arrow::FileWriter::Open(*arrow::schema(fields), arrow::default_memory_pool(),
                                                   *out, props);
    auto fail = [&](const std::string& message) {
        error = message;
        (void)(*out)->Close();
        fs::remove(tmp_path, ignored);
        return false;
    };
    if (!writer.ok()) return fail(writer.status().ToString());
    for (int group = 0; group < source->GetNumRowGroups(); ++group) {
        std::shared_ptr<arrow::Table> table;
        auto status = source->reader_->ReadRowGroup(group, indices, &table);
        if (status.ok()) status = (*writer)->WriteTable(*table, std::max<int64_t>(1, table->num_rows()));
        if (!status.ok()) return fail(status.ToString());
    }
    auto status = (*writer)->Close();
    if (status.ok()) status = (*out)->Close();
    if (!status.ok()) return fail(status.ToString());
    source.reset();
    std::error_code moved;
    fs::rename(tmp_path, output_path, moved);
    if (moved && IsCacheFresh(source_path, output_path)) {  // another load put the same file in place
        fs::remove(tmp_path, ignored);
        return true;
    }
    if (moved) return fail("cannot move '" + tmp_path + "' into place: " + moved.message());
    return true;
}

std::shared_ptr<arrow::ChunkedArray> ParquetBackedDataset::ReadColumn(const std::string& name) const {
    if (!reader_ || !schema_) return nullptr;
    const int index = schema_->GetFieldIndex(name);
    if (index < 0) return nullptr;
    std::shared_ptr<arrow::ChunkedArray> column;
    std::lock_guard<std::mutex> lock(reader_mutex_);
    auto status = reader_->ReadColumn(index, &column);
    if (!status.ok()) {
        spdlog::error("ParquetBackedDataset::ReadColumn({}): {}", name, status.ToString());
        return nullptr;
    }
    return column;
}

std::shared_ptr<arrow::ChunkedArray> ParquetBackedDataset::ReadRowGroupColumn(int row_group_idx,
                                                                             const std::string& name) const {
    if (!reader_ || !schema_) return nullptr;
    if (row_group_idx < 0 || row_group_idx >= static_cast<int>(row_group_sizes_.size())) return nullptr;
    const int index = schema_->GetFieldIndex(name);
    if (index < 0) return nullptr;
    std::shared_ptr<arrow::ChunkedArray> column;
    std::lock_guard<std::mutex> lock(reader_mutex_);
    auto status = reader_->RowGroup(row_group_idx)->Column(index)->Read(&column);
    if (!status.ok()) {
        spdlog::error("ParquetBackedDataset::ReadRowGroupColumn({}, {}): {}", row_group_idx, name, status.ToString());
        return nullptr;
    }
    return column;
}

void AppendDistinctValues(const arrow::ChunkedArray& column, std::vector<std::string>& seen, size_t max_values) {
    for (int64_t row = 0; row < column.length() && seen.size() < max_values; ++row) {
        auto scalar = column.GetScalar(row);
        if (!scalar.ok() || !(*scalar)->is_valid) continue;
        std::string value = (*scalar)->ToString();
        if (std::find(seen.begin(), seen.end(), value) == seen.end()) seen.push_back(std::move(value));
    }
}

std::vector<std::string> ParquetBackedDataset::FirstDistinctValues(const std::string& name, size_t max_values) const {
    std::vector<std::string> seen;
    for (int group = 0; group < GetNumRowGroups() && seen.size() < max_values; ++group) {
        const auto column = ReadRowGroupColumn(group, name);
        if (!column) break;
        AppendDistinctValues(*column, seen, max_values);
    }
    return seen;
}

std::vector<std::string> ParquetBackedDataset::GetColumnNames() const {
    std::vector<std::string> names;
    if (!schema_) return names;
    names.reserve(schema_->num_fields());
    for (int i = 0; i < schema_->num_fields(); ++i) {
        names.push_back(schema_->field(i)->name());
    }
    return names;
}

int64_t ParquetBackedDataset::GetRowGroupSize(int row_group_idx) const {
    if (row_group_idx < 0 || row_group_idx >= static_cast<int>(row_group_sizes_.size())) {
        return 0;
    }
    return row_group_sizes_[row_group_idx];
}

// -----------------------------------------------------------------------------
// Cache location helpers
// -----------------------------------------------------------------------------

std::string ParquetBackedDataset::GetCacheDir() {
    // Use system temp as the cache root — works across platforms, survives
    // across sessions (user doesn't lose the cache when they close the
    // engine), and doesn't require the project to be saved to disk.
    fs::path cache_dir = fs::temp_directory_path() / "cyxwiz" / "cache";
    try {
        fs::create_directories(cache_dir);
    } catch (const std::exception& e) {
        spdlog::warn("ParquetBackedDataset::GetCacheDir: could not create cache dir {}: {}",
                     cache_dir.string(), e.what());
        // Return the path anyway — the eventual write will fail loudly.
    }
    return cache_dir.string();
}

std::string ParquetBackedDataset::GetCacheFilePath(
    const std::string& csv_path,
    const std::string& parser_signature,
    const std::string& cache_directory) {
    // Layout: <cache-root>/<basename>_<stable-contract-hash>.parquet.
    // The cache root is project-scoped when supplied and otherwise falls back
    // to <temp>/cyxwiz/cache for headless/test callers.
    //
    // Hashing the full absolute path means the same CSV at the same location
    // maps to the same cache file, but two CSVs with the same basename in
    // different directories stay disambiguated.
    fs::path csv_abs;
    try {
        csv_abs = fs::absolute(csv_path);
    } catch (...) {
        csv_abs = csv_path;  // best effort
    }

    const std::string abs_str = csv_abs.string();
    const uint64_t hash = StableCacheHash(abs_str + "|" + parser_signature);

    const std::string stem = csv_abs.stem().string();  // filename without extension
    char hash_hex[32];
    snprintf(hash_hex, sizeof(hash_hex), "%016llx",
             static_cast<unsigned long long>(hash));

    const fs::path cache_root = cache_directory.empty()
        ? fs::path(GetCacheDir())
        : fs::path(cache_directory);
    try {
        fs::create_directories(cache_root);
    } catch (const std::exception& e) {
        spdlog::warn("GetCacheFilePath: could not create cache directory '{}': {}",
                     cache_root.string(), e.what());
    }
    fs::path cache_file = cache_root / (stem + "_" + hash_hex + ".parquet");
    return cache_file.string();
}

std::string ParquetBackedDataset::UniqueTempPath(const std::string& path) {
    const auto thread = std::hash<std::thread::id>{}(std::this_thread::get_id());
    const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
    return path + ".tmp." + std::to_string(thread) + "." + std::to_string(ticks);
}

void ParquetBackedDataset::PruneCache(size_t max_total_bytes, int max_age_days) {
    // Loads on other threads prune too; one walk at a time.
    static std::mutex prune_mutex;
    std::lock_guard<std::mutex> prune_lock(prune_mutex);
    fs::path cache_dir = GetCacheDir();
    if (!fs::exists(cache_dir)) return;

    // Collect every .parquet file with its size and mtime. We don't touch
    // anything else in the directory — only files matching our suffix and
    // naming pattern. Anything else stays.
    struct CacheEntry {
        fs::path path;
        uintmax_t size = 0;
        fs::file_time_type mtime{};
    };
    std::vector<CacheEntry> entries;

    try {
        for (const auto& dir_entry : fs::directory_iterator(cache_dir)) {
            if (!dir_entry.is_regular_file()) continue;
            const auto& p = dir_entry.path();
            // A temp file (UniqueTempPath) a day old was left by a crash.
            if (p.filename().string().find(".tmp.") != std::string::npos) {
                std::error_code stale;
                const auto written = fs::last_write_time(p, stale);
                if (!stale && fs::file_time_type::clock::now() - written > std::chrono::hours(24))
                    fs::remove(p, stale);
                continue;
            }
            if (p.extension() != ".parquet") continue;
            CacheEntry e;
            e.path = p;
            try { e.size = fs::file_size(p); } catch (...) { e.size = 0; }
            try { e.mtime = fs::last_write_time(p); } catch (...) { continue; }
            entries.push_back(std::move(e));
        }
    } catch (const std::exception& ex) {
        spdlog::warn("ParquetBackedDataset::PruneCache: directory_iterator failed on {}: {}",
                     cache_dir.string(), ex.what());
        return;
    }

    if (entries.empty()) return;

    // Helper: try to delete a file, swallow errors. Windows can't delete a
    // file that another process (or our own mmap) has open — that's fine,
    // the next prune will retry.
    auto try_remove = [](const fs::path& p) -> bool {
        try {
            return fs::remove(p);
        } catch (const std::exception& e) {
            spdlog::debug("PruneCache: could not delete {}: {}", p.string(), e.what());
            return false;
        }
    };

    // Pass 1 — mtime expiry. Delete any cache file older than max_age_days.
    // file_time_type uses an unspecified clock per spec, so we go via the
    // system_clock cast available in C++20 (file_clock::to_sys) where
    // available, falling back to a wall-clock approximation otherwise.
    size_t expired_count = 0;
    uintmax_t expired_bytes = 0;
    {
        const auto cutoff = std::chrono::hours(24) * max_age_days;
        const auto now = fs::file_time_type::clock::now();
        for (auto it = entries.begin(); it != entries.end(); ) {
            if (now - it->mtime > cutoff) {
                if (try_remove(it->path)) {
                    expired_count++;
                    expired_bytes += it->size;
                }
                it = entries.erase(it);
            } else {
                ++it;
            }
        }
    }

    // Pass 2 — size cap. If the surviving entries exceed max_total_bytes,
    // delete oldest first (LRU by mtime) until we're under the cap.
    uintmax_t total_size = 0;
    for (const auto& e : entries) total_size += e.size;

    size_t evicted_count = 0;
    uintmax_t evicted_bytes = 0;
    if (total_size > max_total_bytes) {
        std::sort(entries.begin(), entries.end(),
                  [](const CacheEntry& a, const CacheEntry& b) {
                      return a.mtime < b.mtime;  // oldest first
                  });
        for (auto& e : entries) {
            if (total_size <= max_total_bytes) break;
            if (try_remove(e.path)) {
                evicted_count++;
                evicted_bytes += e.size;
                total_size -= e.size;
            }
        }
    }

    if (expired_count > 0 || evicted_count > 0) {
        spdlog::info("ParquetBackedDataset::PruneCache: expired {} files ({:.1f} MB), "
                     "evicted {} files ({:.1f} MB), {:.1f} MB remaining",
                     expired_count, expired_bytes / (1024.0 * 1024.0),
                     evicted_count, evicted_bytes / (1024.0 * 1024.0),
                     total_size / (1024.0 * 1024.0));
    } else {
        spdlog::debug("ParquetBackedDataset::PruneCache: nothing to prune ({} files, {:.1f} MB)",
                      entries.size(), total_size / (1024.0 * 1024.0));
    }
}

bool ParquetBackedDataset::IsCacheFresh(const std::string& csv_path,
                                         const std::string& cache_path) {
    try {
        if (!fs::exists(cache_path)) return false;
        if (!fs::exists(csv_path))   return false;
        auto csv_time   = fs::last_write_time(csv_path);
        auto cache_time = fs::last_write_time(cache_path);
        return cache_time >= csv_time;
    } catch (...) {
        return false;
    }
}

// -----------------------------------------------------------------------------
// Streaming CSV -> Parquet conversion
// -----------------------------------------------------------------------------

bool ParquetBackedDataset::ConvertCsvToParquet(const std::string& csv_path,
                                                 const std::string& parquet_path,
                                                 bool has_header,
                                                 char delimiter,
                                                 int skip_rows,
                                                 int64_t max_rows,
                                                 const std::vector<std::string>& missing_value_tokens,
                                                 const std::vector<std::string>& selected_columns,
                                                 char decimal_point,
                                                 bool use_threads,
                                                 const CsvProgressCallback& progress_callback,
                                                 char quote_char,
                                                 const CsvSourceObserver* source_observer) {
    spdlog::info(
        "ParquetBackedDataset::ConvertCsvToParquet: {} -> {} (max_rows={})",
        csv_path,
        parquet_path,
        max_rows);

    // Atomic write: stream to a .tmp path first, then rename on success.
    // This prevents three failure modes:
    //   1. Partial file on disk after a mid-conversion crash/error (old
    //      behavior would leave a truncated Parquet file that IsCacheFresh
    //      would then accept as valid on next load).
    //   2. Two concurrent loads of the same CSV racing to write the same
    //      file: each writes its own temp file (UniqueTempPath) and the
    //      rename is atomic, so one wins and the other uses its file.
    //   3. Power loss mid-write (the temp file has no effect on the
    //      final cache path; next load just rebuilds).
    const std::string tmp_path = UniqueTempPath(parquet_path);
    int64_t csv_source_size = 0;
    try {
        csv_source_size = static_cast<int64_t>(fs::file_size(csv_path));
    } catch (...) {
        csv_source_size = 0;
    }

    // Helper: on any failure, wipe the partial .tmp so the next load
    // cleanly rebuilds. Declared outside the try so both the try and the
    // catch blocks can use it.
    auto cleanup_tmp = [&]() {
        try {
            if (fs::exists(tmp_path)) fs::remove(tmp_path);
        } catch (...) {}
    };

    try {
        // 1) Open the CSV as a streaming reader. StreamingReader yields
        //    record batches one at a time so memory stays bounded to a
        //    single batch rather than the full file.
        auto read_options = arrow::csv::ReadOptions::Defaults();
        read_options.skip_rows = skip_rows;
        read_options.use_threads = use_threads;
        if (!has_header) {
            read_options.autogenerate_column_names = true;
        }
        // Peak memory is about 30 read blocks per thread whatever the file
        // size (measured on 1 GB: 4 MB -> 225 MB, 16 -> 660, 64 -> 1940; the
        // time is the same), so the block follows the data budget (TOFIX154):
        // the conversion stays under half of it. 64 MB at most, 1 MB at least.
        const uint64_t threads = use_threads ? std::max(1u, std::thread::hardware_concurrency()) : 1u;
        read_options.block_size = static_cast<int32_t>(std::clamp<uint64_t>(
            DataBudgetBytes() / (64 * threads), 1ULL << 20, 64ULL << 20));

        auto parse_options = arrow::csv::ParseOptions::Defaults();
        // Quoted values may span lines (free text, reviews): read them as one value.
        parse_options.newlines_in_values = true;
        parse_options.delimiter = delimiter;
        ApplyCsvQuoteChar(parse_options, quote_char);

        auto convert_options = MakeTabularCsvConvertOptions(
            missing_value_tokens, decimal_point);
        convert_options.include_columns = selected_columns;

        const bool preflight_widen_integers =
            csv_source_size > read_options.block_size &&
            CsvSourceSamplesContainDecimalValues(
                csv_path, decimal_point, read_options.block_size);
        if (preflight_widen_integers) {
            spdlog::info(
                "ConvertCsvToParquet: schema preflight found later decimal "
                "values; widening inferred integer columns before conversion");
        }

        int64_t total_rows = 0;
        int batches_written = 0;
        bool converted = false;
        const int max_attempts = preflight_widen_integers ? 1 : 2;
        for (int attempt = 0; attempt < max_attempts && !converted; ++attempt) {
          const bool is_retry = !preflight_widen_integers && attempt > 0;
          const bool promote_inferred_integers =
              preflight_widen_integers || is_retry;
            size_t promoted_integer_columns = 0;
            std::shared_ptr<arrow::io::ReadableFile> input_source;
            auto maybe_reader = MakeStableCsvStreamingReader(
                csv_path, read_options, parse_options, convert_options,
                &promoted_integer_columns, promote_inferred_integers,
                &input_source, source_observer);
          if (!maybe_reader.ok()) {
            spdlog::error(
                "ConvertCsvToParquet: stable StreamingReader failed for {}: {}",
                csv_path, maybe_reader.status().ToString());
            cleanup_tmp();
            return false;
          }
          if (promoted_integer_columns > 0) {
            spdlog::info("ConvertCsvToParquet: promoted {} inferred integer "
                         "columns to float64 for stable full-source parsing",
                         promoted_integer_columns);
          }
          auto reader = maybe_reader.ValueOrDie();

          // 2) Open the Parquet output file and create a FileWriter using the
          //    schema we just got from the CSV reader. Writes go to tmp_path,
          //    not parquet_path, so the final cache file only appears after a
          //    successful rename at the end.
          auto schema = reader->schema();
          if (!schema) {
            spdlog::error(
                "ConvertCsvToParquet: CSV reader returned null schema");
            cleanup_tmp();
            return false;
          }

          // Ensure the output parent directory exists
          try {
            auto parent = fs::path(tmp_path).parent_path();
            if (!parent.empty())
              fs::create_directories(parent);
          } catch (...) {
            // Not fatal; the FileOutputStream::Open will fail loudly if so.
          }

          auto maybe_pq_output = arrow::io::FileOutputStream::Open(tmp_path);
          if (!maybe_pq_output.ok()) {
            spdlog::error(
                "ConvertCsvToParquet: FileOutputStream::Open failed for {}: {}",
                tmp_path, maybe_pq_output.status().ToString());
            cleanup_tmp();
            return false;
          }
          auto pq_output = maybe_pq_output.ValueOrDie();

          // Snappy compression: fast encode/decode, good ratio on numeric data
          // (typically 3-5x smaller than raw CSV for ML datasets).
          // Row groups of 128K rows (like SessionQueryEngine) whatever the
          // read block size: WriteRecordBatch buffers batches into them.
          auto writer_props = parquet::WriterProperties::Builder()
                                  .compression(parquet::Compression::SNAPPY)
                                  ->max_row_group_length(128 * 1024)
                                  ->build();
          auto arrow_props = parquet::ArrowWriterProperties::Builder().build();

          // Arrow 21+: FileWriter::Open returns Result<unique_ptr<FileWriter>>
          // (earlier versions took an out-param unique_ptr).
          auto maybe_writer = parquet::arrow::FileWriter::Open(
              *schema, arrow::default_memory_pool(), pq_output, writer_props,
              arrow_props);
          if (!maybe_writer.ok()) {
            spdlog::error("ConvertCsvToParquet: FileWriter::Open failed: {}",
                          maybe_writer.status().ToString());
            cleanup_tmp();
            return false;
          }
          std::unique_ptr<parquet::arrow::FileWriter> writer =
              std::move(maybe_writer).ValueOrDie();

          // 3) Streaming loop: read a record batch from CSV, write to Parquet,
          //    repeat until the CSV reader is exhausted.
          total_rows = 0;
          batches_written = 0;
          bool read_failed = false;
          while (true) {
            if (max_rows > 0 && total_rows >= max_rows) {
              break;
            }

            std::shared_ptr<arrow::RecordBatch> batch;
            auto read_status = reader->ReadNext(&batch);
            if (!read_status.ok()) {
              spdlog::warn("ConvertCsvToParquet: ReadNext failed at batch {} "
                           "on attempt {}: {}",
                           batches_written, attempt + 1,
                           read_status.ToString());
              read_failed = true;
              break;
            }
            if (!batch)
              break; // end of input

            if (max_rows > 0) {
              const int64_t remaining = max_rows - total_rows;
              if (batch->num_rows() > remaining) {
                batch = batch->Slice(0, remaining);
              }
            }

            auto write_status = writer->WriteRecordBatch(*batch);
            if (!write_status.ok()) {
              spdlog::error(
                  "ConvertCsvToParquet: WriteRecordBatch failed at batch {}: {}",
                  batches_written, write_status.ToString());
              cleanup_tmp();
              return false;
            }

                total_rows += batch->num_rows();
                batches_written++;

                if (progress_callback && csv_source_size > 0 && input_source) {
                    auto position = input_source->Tell();
                    if (position.ok()) {
                        const float source_fraction = std::clamp(
                            static_cast<float>(position.ValueOrDie()) /
                                static_cast<float>(csv_source_size),
                            0.0f, 1.0f);
                        const float base = is_retry ? 0.8f : 0.0f;
                        const float span = is_retry ? 0.19f : 0.99f;
                        if (!progress_callback(
                                base + span * source_fraction,
                                is_retry
                                    ? "Caching CSV with widened numeric schema"
                                    : (promote_inferred_integers
                                        ? "Caching CSV with preflight schema"
                                        : "Caching CSV as Parquet"))) {
                            (void)writer->Close();
                            writer.reset();
                            (void)pq_output->Close();
                            pq_output.reset();
                            cleanup_tmp();
                            return false;
                        }
                    }
                }
            }

          if (read_failed) {
            (void)writer->Close();
            writer.reset();
            (void)pq_output->Close();
                pq_output.reset();
                cleanup_tmp();
                if (!preflight_widen_integers && attempt == 0 &&
                    progress_callback &&
                    !progress_callback(0.8f,
                                       "Retrying cache with widened numeric schema")) {
                    return false;
                }
                continue;
          }

          // 4) Close the writer (flushes footer, etc.)
          auto close_status = writer->Close();
          if (!close_status.ok()) {
            spdlog::error("ConvertCsvToParquet: writer->Close() failed: {}",
                          close_status.ToString());
            cleanup_tmp();
            return false;
          }

          // 4b) Release the writer and the underlying output stream before
          //     attempting the rename. Windows won't rename a file that any
          //     process still has a handle on, and Arrow's FileOutputStream
          //     holds the OS handle open until it's explicitly Close()d OR
          //     its last shared_ptr reference is dropped. writer->Close()
          //     only closes the writer's own state, not the underlying sink.
          //
          //     Step one: drop the writer. It holds an internal reference
          //     to pq_output; dropping it releases that reference.
          writer.reset();

          //     Step two: explicitly Close() the output stream and drop
          //     its shared_ptr. Either of these alone might be enough on
          //     some platforms, but doing both is defensive and matches
          //     Arrow's documented lifecycle.
          auto pq_close_status = pq_output->Close();
          if (!pq_close_status.ok()) {
            spdlog::warn(
                "ConvertCsvToParquet: pq_output->Close() returned non-OK: {}",
                pq_close_status.ToString());
            // Non-fatal — the handle is still dropped below.
          }
          pq_output.reset();
          converted = true;
        }

        if (!converted) {
          spdlog::error("ConvertCsvToParquet: CSV read failed with both "
                        "inferred and numeric fallback schemas");
          cleanup_tmp();
          return false;
        }

        // 5) Atomic rename .tmp -> final cache path. This is the point where
        //    the cache becomes visible to IsCacheFresh on subsequent loads.
        //    Any failure before this point leaves no cache file behind.
        try {
            // If a previous cache exists (stale mtime), remove it so the
            // rename can succeed on Windows (which won't overwrite).
            if (fs::exists(parquet_path)) {
                fs::remove(parquet_path);
            }
            fs::rename(tmp_path, parquet_path);
        } catch (const std::exception& e) {
            cleanup_tmp();
            // Another load of the same CSV put its file in place first.
            if (!IsCacheFresh(csv_path, parquet_path)) {
                spdlog::error("ConvertCsvToParquet: final rename {} -> {} failed: {}",
                              tmp_path, parquet_path, e.what());
                return false;
            }
        }

        // Log the end result including compression ratio for visibility.
        size_t csv_size = 0;
        size_t parquet_size = 0;
        try {
            csv_size = static_cast<size_t>(fs::file_size(csv_path));
            parquet_size = static_cast<size_t>(fs::file_size(parquet_path));
        } catch (...) {}

        double ratio = (parquet_size > 0)
            ? (static_cast<double>(csv_size) / parquet_size)
            : 0.0;
        spdlog::info("ConvertCsvToParquet: {} rows in {} batches, "
                     "{:.1f} MB CSV -> {:.1f} MB Parquet ({:.1f}x compression)",
                     total_rows, batches_written,
                     csv_size / (1024.0 * 1024.0),
                     parquet_size / (1024.0 * 1024.0),
                     ratio);
        if (progress_callback) {
            (void)progress_callback(1.0f, "Parquet cache ready");
        }

        // Run cache hygiene right after a fresh write so the directory
        // doesn't grow unbounded across many CSV-> Parquet conversions
        // in a single session. The just-written file has the newest mtime
        // so it's safe from both the age expiry and the LRU pass.
        PruneCache();
        return true;

    } catch (const std::exception& e) {
        spdlog::error("ConvertCsvToParquet exception: {}", e.what());
        cleanup_tmp();
        return false;
    }
}

} // namespace cyxwiz
