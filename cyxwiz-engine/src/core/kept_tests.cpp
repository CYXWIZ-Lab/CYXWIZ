#include "kept_tests.h"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/util/key_value_metadata.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <system_error>

namespace cyxwiz {

namespace fs = std::filesystem;

fs::path KeptTestsFolder(const fs::path& project_root) { return project_root / "results" / "tests"; }

std::string KeptTestFileName(const std::string& name, std::time_t when) {
    std::string slug;
    for (unsigned char c : name) {
        if (std::isalnum(c)) slug += static_cast<char>(std::tolower(c));
        else if (!slug.empty() && slug.back() != '_') slug += '_';
    }
    while (!slug.empty() && slug.back() == '_') slug.pop_back();
    if (slug.empty()) slug = "test";
    if (slug.size() > 60) slug.resize(60);
    std::tm local{};
#if defined(_WIN32)
    localtime_s(&local, &when);
#else
    localtime_r(&when, &local);
#endif
    char stamp[32];
    std::snprintf(stamp, sizeof(stamp), "_%04d%02d%02d_%02d%02d", local.tm_year + 1900, local.tm_mon + 1, local.tm_mday,
                  local.tm_hour, local.tm_min);
    return slug + stamp + ".parquet";
}

bool WriteKeptTest(const arrow::Table& rows, const std::string& name, const fs::path& path, std::string& error) {
    std::error_code ec;
    if (fs::exists(path, ec)) {
        error = "'" + path.string() + "' already exists";
        return false;
    }
    fs::create_directories(path.parent_path(), ec);
    if (ec) {
        error = "cannot create '" + path.parent_path().string() + "': " + ec.message();
        return false;
    }
    std::shared_ptr<arrow::KeyValueMetadata> metadata =
        rows.schema()->metadata() ? rows.schema()->metadata()->Copy() : std::make_shared<arrow::KeyValueMetadata>();
    (void)metadata->Delete(kKeptTestNameKey);  // a kept test kept again takes the new name
    metadata->Append(kKeptTestNameKey, name);
    const std::shared_ptr<arrow::Table> named = rows.ReplaceSchemaMetadata(metadata);
    auto file = arrow::io::FileOutputStream::Open(path.string());
    if (!file.ok()) {
        error = file.status().ToString();
        return false;
    }
    parquet::WriterProperties::Builder properties;
    properties.compression(parquet::Compression::SNAPPY);
    arrow::Status status = parquet::arrow::WriteTable(*named, arrow::default_memory_pool(), *file, 1024 * 1024,
                                                      properties.build(),
                                                      parquet::ArrowWriterProperties::Builder().store_schema()->build());
    const arrow::Status closed = (*file)->Close();
    if (status.ok()) status = closed;
    if (!status.ok()) {
        error = status.ToString();
        fs::remove(path, ec);
        return false;
    }
    return true;
}

namespace {

std::unique_ptr<parquet::arrow::FileReader> OpenReader(const fs::path& path, std::string& error) {
    auto input = arrow::io::ReadableFile::Open(path.string());
    if (!input.ok()) {
        error = input.status().ToString();
        return nullptr;
    }
    auto reader = parquet::arrow::OpenFile(*input, arrow::default_memory_pool());
    if (!reader.ok()) {
        error = reader.status().ToString();
        return nullptr;
    }
    return std::move(*reader);
}

}  // namespace

std::shared_ptr<arrow::Table> ReadKeptTest(const fs::path& path, std::string& error) {
    const auto reader = OpenReader(path, error);
    if (!reader) return nullptr;
    std::shared_ptr<arrow::Table> table;
    if (const arrow::Status status = reader->ReadTable(&table); !status.ok()) {
        error = status.ToString();
        return nullptr;
    }
    return table;
}

std::vector<KeptTest> ListKeptTests(const fs::path& folder) {
    std::vector<KeptTest> out;
    std::error_code ec;
    if (!fs::is_directory(folder, ec)) return out;
    for (const auto& entry : fs::directory_iterator(folder, ec)) {
        if (!entry.is_regular_file(ec) || entry.path().extension() != ".parquet") continue;
        KeptTest kept;
        kept.path = entry.path();
        kept.name = entry.path().stem().string();
        kept.time = entry.last_write_time(ec);
        std::string error;
        if (const auto reader = OpenReader(entry.path(), error)) {
            std::shared_ptr<arrow::Schema> schema;
            if (reader->GetSchema(&schema).ok() && schema && schema->metadata()) {
                const int at = schema->metadata()->FindKey(kKeptTestNameKey);
                if (at >= 0 && !schema->metadata()->value(at).empty()) kept.name = schema->metadata()->value(at);
            }
        }
        out.push_back(std::move(kept));
    }
    std::sort(out.begin(), out.end(), [](const KeptTest& a, const KeptTest& b) { return a.time > b.time; });
    return out;
}

}  // namespace cyxwiz
