#include "text_words.h"

#include "dashboard_model.h"

#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>
#include <parquet/file_reader.h>
#include <parquet/properties.h>

#include <cstdio>
#include <functional>
#include <system_error>

namespace cyxwiz::dashboard {

namespace fs = std::filesystem;

namespace {

std::string Quote(const std::string& name) {
    std::string q = "\"";
    for (char c : name) q += c == '"' ? std::string("\"\"") : std::string(1, c);
    return q + "\"";
}

std::string Hex(uint64_t v) {
    char buf[24];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(v));
    return buf;
}

}  // namespace

QueryRequest WordsTableQuery(const std::string& table, const std::string& text_field) {
    QueryRequest r;
    r.inputs = {table};
    r.label = "Dashboard: words of " + text_field;
    const std::string tokens = TokensSql(Quote(text_field));
    r.sql = "SELECT *, array_to_string(" + tokens + ", ' ') AS cyxwiz_words, CAST(len(" + tokens + ") AS INTEGER) AS cyxwiz_word_count FROM " +
            Quote(table);
    return r;
}

std::string WordsCacheKey(const std::string& source_path, const std::string& dataset, uint64_t generation, const std::string& text_field) {
    // v2: the tokenizer (TokensSql) and the saved layout (words joined by spaces);
    // a change to either must change the key.
    std::string key = "v2|" + text_field + "|";
    std::error_code ec;
    if (!source_path.empty() && fs::is_regular_file(source_path, ec)) {
        const auto size = fs::file_size(source_path, ec);
        const auto time = fs::last_write_time(source_path, ec).time_since_epoch().count();
        key += fs::absolute(source_path, ec).lexically_normal().string() + "|" + std::to_string(size) + "|" + std::to_string(time);
    } else {
        key += "session|" + dataset + "|" + std::to_string(generation);
    }
    return Hex(std::hash<std::string>{}(key));
}

WordsTable EnsureWordsTable(const std::string& table, const std::string& text_field, const std::string& key, const fs::path& cache_dir,
                            const QueryRunner& run) {
    WordsTable out;
    out.name = "cyxwiz_words_" + key;
    const fs::path path = cache_dir / (key + ".parquet");
    out.path = path.string();
    std::error_code ec;
    // Saved before: its row count from the file's footer.
    if (fs::is_regular_file(path, ec)) {
        try {
            auto reader = parquet::ParquetFileReader::OpenFile(out.path, false);
            out.rows = static_cast<size_t>(reader->metadata()->num_rows());
            out.loaded = true;
            return out;
        } catch (const std::exception&) {
            fs::remove(path, ec);  // a broken file is made again
        }
    }
    const QueryResult r = run(WordsTableQuery(table, text_field));
    if (!r.ok || !r.table) {
        out.error = r.error.empty() ? std::string("the words could not be split") : r.error;
        return out;
    }
    fs::create_directories(cache_dir, ec);
    const fs::path tmp = path.string() + ".tmp";
    {
        auto file = arrow::io::FileOutputStream::Open(tmp.string());
        if (!file.ok()) {
            out.error = "could not write " + tmp.string() + ": " + file.status().ToString();
            return out;
        }
        auto props = parquet::WriterProperties::Builder().compression(parquet::Compression::SNAPPY)->build();
        const auto status = parquet::arrow::WriteTable(*r.table, arrow::default_memory_pool(), *file, 64 * 1024, props);
        const auto closed = (*file)->Close();
        if (!status.ok() || !closed.ok()) {
            fs::remove(tmp, ec);
            out.error = "could not write the words: " + (status.ok() ? closed : status).ToString();
            return out;
        }
    }
    fs::rename(tmp, path, ec);
    if (ec) {
        fs::remove(tmp, ec);
        out.error = "could not save the words in " + path.string();
        return out;
    }
    out.rows = static_cast<size_t>(r.table->num_rows());
    return out;
}

}  // namespace cyxwiz::dashboard
