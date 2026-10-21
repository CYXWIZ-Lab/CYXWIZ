#pragma once

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/json/api.h>
#include <memory>
#include <string>

namespace cyxwiz {

// Newline-delimited JSON objects -> Arrow table (types inferred, nested
// objects become structs, arrays become lists). Shared by DataInput and
// Data Convert. Plain JSON (one array or object) is not JSON Lines.
inline std::shared_ptr<arrow::Table> ReadJsonLinesTable(const std::string& path,
                                                        bool newlines_in_values,
                                                        std::string& error) {
    auto input = arrow::io::ReadableFile::Open(path);
    if (!input.ok()) {
        error = "Could not open JSON Lines input '" + path + "': " + input.status().ToString();
        return nullptr;
    }
    auto parse_options = arrow::json::ParseOptions::Defaults();
    parse_options.newlines_in_values = newlines_in_values;
    auto reader = arrow::json::TableReader::Make(arrow::default_memory_pool(), *input,
                                                 arrow::json::ReadOptions::Defaults(), parse_options);
    if (!reader.ok()) {
        error = "JSON Lines reader creation failed for '" + path + "': " + reader.status().ToString();
        return nullptr;
    }
    auto table = (*reader)->Read();
    if (!table.ok()) {
        error = "JSON Lines input read failed for '" + path +
                "'. Expected newline-delimited JSON objects. Error: " + table.status().ToString();
        return nullptr;
    }
    return *table;
}

} // namespace cyxwiz
