#pragma once

// A text column's words, split once (TOFIX134 P3 text, owner 2026-10-05:
// "only once and next should load"). The table's rows with two more columns
// (cyxwiz_words: the words joined by spaces - a word has none, and query
// results carry text, not lists; cyxwiz_word_count) are saved as Parquet
// in the project's cache, keyed by the source file (path, size, time) and the
// column, so every text widget and KPI reads them instead of splitting the
// text again, and a reopened dashboard loads them.

#include "../dataset_profiler.h"

#include <filesystem>
#include <string>

namespace cyxwiz::dashboard {

struct WordsTable {
    std::string name;   // the side table's name in the session's queries
    std::string path;   // its Parquet file
    size_t rows = 0;
    bool loaded = false;  // read from the cache (not split now)
    std::string error;
};

// The SELECT that adds the words to every row of `table` (one scan).
QueryRequest WordsTableQuery(const std::string& table, const std::string& text_field);

// The cache key: the source file (path, size, write time) when there is one,
// else the dataset's name and version (then only this session reuses it).
std::string WordsCacheKey(const std::string& source_path, const std::string& dataset, uint64_t generation, const std::string& text_field);

// Loads the words from `cache_dir` when saved for this key, else splits them
// with `run` and saves them (a temporary file, then renamed).
WordsTable EnsureWordsTable(const std::string& table, const std::string& text_field, const std::string& key,
                            const std::filesystem::path& cache_dir, const QueryRunner& run);

}  // namespace cyxwiz::dashboard
