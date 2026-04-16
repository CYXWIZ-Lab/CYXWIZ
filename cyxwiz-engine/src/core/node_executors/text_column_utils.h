#pragma once

#include <arrow/api.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

/// Read all string values from an Arrow column into std::vector<std::string>.
/// Null values become empty strings. Returns false (with the offending Arrow
/// type name in `error_type_name`) if the column isn't a string/large_string.
/// Shared between TextTokenizerOperator, TFIDFVectorizerOperator, and
/// future text-input Cat-1 operators.
inline bool ReadColumnAsStrings(
    const std::shared_ptr<arrow::ChunkedArray>& column,
    std::vector<std::string>& out,
    std::string& error_type_name) {

    out.clear();
    out.reserve(static_cast<size_t>(column->length()));

    for (int c = 0; c < column->num_chunks(); ++c) {
        auto chunk = column->chunk(c);
        const int64_t chunk_len = chunk->length();
        if (chunk->type_id() != arrow::Type::STRING &&
            chunk->type_id() != arrow::Type::LARGE_STRING) {
            error_type_name = chunk->type()->ToString();
            return false;
        }
        if (chunk->type_id() == arrow::Type::STRING) {
            auto arr = std::static_pointer_cast<arrow::StringArray>(chunk);
            for (int64_t i = 0; i < chunk_len; ++i) {
                if (chunk->IsNull(i)) {
                    out.emplace_back();
                } else {
                    out.emplace_back(arr->GetString(i));
                }
            }
        } else {
            auto arr = std::static_pointer_cast<arrow::LargeStringArray>(chunk);
            for (int64_t i = 0; i < chunk_len; ++i) {
                if (chunk->IsNull(i)) {
                    out.emplace_back();
                } else {
                    out.emplace_back(arr->GetString(i));
                }
            }
        }
    }
    return true;
}

/// Read a label column. If the column is numeric, copies as int. If
/// the column is string, builds a first-seen string→int mapping
/// (returned via `class_names`) and emits the per-row class index.
/// Matches what the legacy `TextDataset` does for class_names ordering.
inline bool ReadLabelColumnAsInt(
    const std::shared_ptr<arrow::ChunkedArray>& column,
    std::vector<int>& out,
    std::vector<std::string>& class_names,
    std::string& error_type_name) {

    out.clear();
    out.reserve(static_cast<size_t>(column->length()));
    class_names.clear();

    if (column->num_chunks() == 0) return true;
    auto first_type = column->chunk(0)->type_id();

    if (first_type == arrow::Type::STRING ||
        first_type == arrow::Type::LARGE_STRING) {
        std::map<std::string, int> name_to_idx;
        for (int c = 0; c < column->num_chunks(); ++c) {
            auto chunk = column->chunk(c);
            const int64_t chunk_len = chunk->length();
            for (int64_t i = 0; i < chunk_len; ++i) {
                std::string s;
                if (!chunk->IsNull(i)) {
                    if (chunk->type_id() == arrow::Type::STRING) {
                        s = std::static_pointer_cast<arrow::StringArray>(chunk)->GetString(i);
                    } else {
                        s = std::static_pointer_cast<arrow::LargeStringArray>(chunk)->GetString(i);
                    }
                }
                auto it = name_to_idx.find(s);
                int idx;
                if (it == name_to_idx.end()) {
                    idx = static_cast<int>(class_names.size());
                    class_names.push_back(s);
                    name_to_idx[s] = idx;
                } else {
                    idx = it->second;
                }
                out.push_back(idx);
            }
        }
        return true;
    }

    for (int c = 0; c < column->num_chunks(); ++c) {
        auto chunk = column->chunk(c);
        const int64_t chunk_len = chunk->length();
        for (int64_t i = 0; i < chunk_len; ++i) {
            if (chunk->IsNull(i)) { out.push_back(0); continue; }
            switch (chunk->type_id()) {
                case arrow::Type::INT64:
                    out.push_back(static_cast<int>(
                        std::static_pointer_cast<arrow::Int64Array>(chunk)->Value(i)));
                    break;
                case arrow::Type::INT32:
                    out.push_back(std::static_pointer_cast<arrow::Int32Array>(chunk)->Value(i));
                    break;
                case arrow::Type::INT16:
                    out.push_back(std::static_pointer_cast<arrow::Int16Array>(chunk)->Value(i));
                    break;
                case arrow::Type::INT8:
                    out.push_back(std::static_pointer_cast<arrow::Int8Array>(chunk)->Value(i));
                    break;
                case arrow::Type::UINT8:
                    out.push_back(std::static_pointer_cast<arrow::UInt8Array>(chunk)->Value(i));
                    break;
                case arrow::Type::UINT16:
                    out.push_back(std::static_pointer_cast<arrow::UInt16Array>(chunk)->Value(i));
                    break;
                case arrow::Type::UINT32:
                    out.push_back(static_cast<int>(
                        std::static_pointer_cast<arrow::UInt32Array>(chunk)->Value(i)));
                    break;
                case arrow::Type::FLOAT:
                    out.push_back(static_cast<int>(
                        std::static_pointer_cast<arrow::FloatArray>(chunk)->Value(i)));
                    break;
                case arrow::Type::DOUBLE:
                    out.push_back(static_cast<int>(
                        std::static_pointer_cast<arrow::DoubleArray>(chunk)->Value(i)));
                    break;
                default:
                    error_type_name = chunk->type()->ToString();
                    return false;
            }
        }
    }
    return true;
}

} // namespace cyxwiz
