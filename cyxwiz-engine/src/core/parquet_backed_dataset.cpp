#include "parquet_backed_dataset.h"

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
#include <filesystem>
#include <functional>
#include <vector>

namespace cyxwiz {

namespace fs = std::filesystem;

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
    for (int i = 0; i < num_row_groups; ++i) {
        auto rg_metadata = metadata->RowGroup(i);
        int64_t rg_rows = rg_metadata ? rg_metadata->num_rows() : 0;
        row_group_sizes.push_back(rg_rows);
        total_rows += rg_rows;
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
    auto status = reader_->ReadRowGroups(indices, &table);
    if (!status.ok()) {
        spdlog::error("ParquetBackedDataset::ReadRowGroups: {}", status.ToString());
        return nullptr;
    }
    return table;
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
    const std::string& parser_signature) {
    // Layout:  <temp>/cyxwiz/cache/<basename>_<pathhash>.parquet
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
    const size_t hash = std::hash<std::string>{}(
        abs_str + "|" + parser_signature);

    const std::string stem = csv_abs.stem().string();  // filename without extension
    char hash_hex[32];
    snprintf(hash_hex, sizeof(hash_hex), "%016zx", hash);

    fs::path cache_file = fs::path(GetCacheDir()) / (stem + "_" + hash_hex + ".parquet");
    return cache_file.string();
}

void ParquetBackedDataset::PruneCache(size_t max_total_bytes, int max_age_days) {
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
                                                 const std::vector<std::string>& missing_value_tokens) {
    spdlog::info("ParquetBackedDataset::ConvertCsvToParquet: {} -> {}", csv_path, parquet_path);

    // Atomic write: stream to a .tmp path first, then rename on success.
    // This prevents three failure modes:
    //   1. Partial file on disk after a mid-conversion crash/error (old
    //      behavior would leave a truncated Parquet file that IsCacheFresh
    //      would then accept as valid on next load).
    //   2. Two concurrent loads of the same CSV racing to write the same
    //      file (the rename() is atomic on the filesystem, so one wins).
    //   3. Power loss mid-write (the temp file has no effect on the
    //      final cache path; next load just rebuilds).
    const std::string tmp_path = parquet_path + ".tmp";

    // Clean up any stale .tmp from a previous aborted run.
    try {
        if (fs::exists(tmp_path)) {
            fs::remove(tmp_path);
        }
    } catch (...) {
        // Non-fatal; FileOutputStream::Open will fail loudly below if so.
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
        auto maybe_csv_input = arrow::io::ReadableFile::Open(csv_path);
        if (!maybe_csv_input.ok()) {
            spdlog::error("ConvertCsvToParquet: cannot open CSV {}: {}",
                          csv_path, maybe_csv_input.status().ToString());
            cleanup_tmp();
            return false;
        }
        auto csv_input = maybe_csv_input.ValueOrDie();

        auto read_options = arrow::csv::ReadOptions::Defaults();
        read_options.skip_rows = skip_rows;
        if (!has_header) {
            read_options.autogenerate_column_names = true;
        }
        // Bound per-batch memory — 64 MB block is more than enough headroom
        // for streaming conversion without blowing out RAM on large CSVs.
        read_options.block_size = 64 * 1024 * 1024;

        auto parse_options = arrow::csv::ParseOptions::Defaults();
        parse_options.delimiter = delimiter;

        auto convert_options = MakeTabularCsvConvertOptions(missing_value_tokens);

        auto maybe_reader = arrow::csv::StreamingReader::Make(
            arrow::io::default_io_context(), csv_input,
            read_options, parse_options, convert_options);
        if (!maybe_reader.ok()) {
            spdlog::error("ConvertCsvToParquet: StreamingReader::Make failed for {}: {}",
                          csv_path, maybe_reader.status().ToString());
            cleanup_tmp();
            return false;
        }
        auto reader = maybe_reader.ValueOrDie();

        // 2) Open the Parquet output file and create a FileWriter using the
        //    schema we just got from the CSV reader. Writes go to tmp_path,
        //    not parquet_path, so the final cache file only appears after a
        //    successful rename at the end.
        auto schema = reader->schema();
        if (!schema) {
            spdlog::error("ConvertCsvToParquet: CSV reader returned null schema");
            cleanup_tmp();
            return false;
        }

        // Ensure the output parent directory exists
        try {
            auto parent = fs::path(tmp_path).parent_path();
            if (!parent.empty()) fs::create_directories(parent);
        } catch (...) {
            // Not fatal; the FileOutputStream::Open will fail loudly if so.
        }

        auto maybe_pq_output = arrow::io::FileOutputStream::Open(tmp_path);
        if (!maybe_pq_output.ok()) {
            spdlog::error("ConvertCsvToParquet: FileOutputStream::Open failed for {}: {}",
                          tmp_path, maybe_pq_output.status().ToString());
            cleanup_tmp();
            return false;
        }
        auto pq_output = maybe_pq_output.ValueOrDie();

        // Snappy compression: fast encode/decode, good ratio on numeric data
        // (typically 3-5x smaller than raw CSV for ML datasets).
        auto writer_props = parquet::WriterProperties::Builder()
            .compression(parquet::Compression::SNAPPY)
            ->build();
        auto arrow_props = parquet::ArrowWriterProperties::Builder().build();

        // Arrow 21+: FileWriter::Open returns Result<unique_ptr<FileWriter>>
        // (earlier versions took an out-param unique_ptr).
        auto maybe_writer = parquet::arrow::FileWriter::Open(
            *schema, arrow::default_memory_pool(), pq_output,
            writer_props, arrow_props);
        if (!maybe_writer.ok()) {
            spdlog::error("ConvertCsvToParquet: FileWriter::Open failed: {}",
                          maybe_writer.status().ToString());
            cleanup_tmp();
            return false;
        }
        std::unique_ptr<parquet::arrow::FileWriter> writer = std::move(maybe_writer).ValueOrDie();

        // 3) Streaming loop: read a record batch from CSV, write to Parquet,
        //    repeat until the CSV reader is exhausted.
        int64_t total_rows = 0;
        int batches_written = 0;
        while (true) {
            std::shared_ptr<arrow::RecordBatch> batch;
            auto read_status = reader->ReadNext(&batch);
            if (!read_status.ok()) {
                spdlog::error("ConvertCsvToParquet: ReadNext failed at batch {}: {}",
                              batches_written, read_status.ToString());
                cleanup_tmp();
                return false;
            }
            if (!batch) break;  // end of input

            // WriteRecordBatch would be nicer but requires wrapping in a Table
            // because FileWriter's direct batch API varies across Arrow versions.
            // Table::Make is cheap — it's just a metadata wrapper, no data copy.
            auto table = arrow::Table::FromRecordBatches(schema, {batch});
            if (!table.ok()) {
                spdlog::error("ConvertCsvToParquet: Table::FromRecordBatches failed: {}",
                              table.status().ToString());
                cleanup_tmp();
                return false;
            }
            auto write_status = writer->WriteTable(*table.ValueOrDie(), batch->num_rows());
            if (!write_status.ok()) {
                spdlog::error("ConvertCsvToParquet: WriteTable failed at batch {}: {}",
                              batches_written, write_status.ToString());
                cleanup_tmp();
                return false;
            }

            total_rows += batch->num_rows();
            batches_written++;
        }

        // 4) Close the writer (flushes footer, etc.)
        auto close_status = writer->Close();
        if (!close_status.ok()) {
            spdlog::error("ConvertCsvToParquet: writer->Close() failed: {}", close_status.ToString());
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
            spdlog::warn("ConvertCsvToParquet: pq_output->Close() returned non-OK: {}",
                         pq_close_status.ToString());
            // Non-fatal — the handle is still dropped below.
        }
        pq_output.reset();

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
            spdlog::error("ConvertCsvToParquet: final rename {} -> {} failed: {}",
                          tmp_path, parquet_path, e.what());
            cleanup_tmp();
            return false;
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
