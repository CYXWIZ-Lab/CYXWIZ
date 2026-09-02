#include "sparse_feature_dataset_cache.h"
#include "sparse_feature_dataset.h"

#include <arrow/api.h>
#include <arrow/array/concatenate.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/util/key_value_metadata.h>

#include <charconv>
#include <chrono>
#include <filesystem>
#include <limits>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace cyxwiz {
namespace {

namespace fs = std::filesystem;

constexpr std::string_view kFormatKey = "cyxwiz_sparse_cache_format";
constexpr std::string_view kFormatValue = "SparseFeatureDataset";
constexpr std::string_view kVersionKey = "cyxwiz_sparse_cache_version";
constexpr std::string_view kNameKey = "name";
constexpr std::string_view kRowsKey = "num_rows";
constexpr std::string_view kFeaturesKey = "num_features";
constexpr std::string_view kLabelNameKey = "label_name";

arrow::Status Invalid(const std::string& detail) {
    return arrow::Status::Invalid("SparseFeatureDatasetCache: " + detail);
}

class TemporaryFileGuard {
public:
    explicit TemporaryFileGuard(fs::path path) : path_(std::move(path)) {}
    ~TemporaryFileGuard() {
        if (!released_) {
            std::error_code ignored;
            fs::remove(path_, ignored);
        }
    }
    void Release() noexcept { released_ = true; }

private:
    fs::path path_;
    bool released_ = false;
};

std::string TemporaryPathFor(const fs::path& destination) {
    const auto thread_hash = std::hash<std::thread::id>{}(
        std::this_thread::get_id());
    const auto nonce = std::chrono::steady_clock::now()
        .time_since_epoch().count();
    return destination.string() + ".tmp." + std::to_string(thread_hash) +
        "." + std::to_string(nonce);
}

arrow::Status ReplaceFileAtomically(const fs::path& source,
                                    const fs::path& destination) {
#ifdef _WIN32
    if (!MoveFileExW(source.c_str(), destination.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        return arrow::Status::IOError(
            "SparseFeatureDatasetCache: atomic replace failed with Windows "
            "error ",
            static_cast<unsigned long>(GetLastError()));
    }
    return arrow::Status::OK();
#else
    std::error_code error;
    fs::rename(source, destination, error);
    if (error) {
        return arrow::Status::IOError(
            "SparseFeatureDatasetCache: atomic replace failed: ",
            error.message());
    }
    return arrow::Status::OK();
#endif
}

template <typename Builder, typename Value>
arrow::Result<std::shared_ptr<arrow::Array>> BuildPrimitiveArray(
    const std::vector<Value>& values) {
    Builder builder;
    ARROW_RETURN_NOT_OK(builder.AppendValues(values));
    return builder.Finish();
}

arrow::Result<std::shared_ptr<arrow::Array>> BuildStringArray(
    const std::vector<std::string>& values) {
    arrow::StringBuilder builder;
    ARROW_RETURN_NOT_OK(builder.AppendValues(values));
    return builder.Finish();
}

arrow::Result<std::shared_ptr<arrow::Array>> WrapInSingleList(
    const std::shared_ptr<arrow::Array>& values) {
    if (!values ||
        values->length() > (std::numeric_limits<int32_t>::max)()) {
        return Invalid("list payload exceeds the int32 Arrow list contract");
    }

    arrow::Int32Builder offsets_builder;
    ARROW_RETURN_NOT_OK(offsets_builder.Append(0));
    ARROW_RETURN_NOT_OK(
        offsets_builder.Append(static_cast<int32_t>(values->length())));
    ARROW_ASSIGN_OR_RAISE(auto offsets, offsets_builder.Finish());
    return arrow::ListArray::FromArrays(*offsets, *values);
}

arrow::Result<std::shared_ptr<arrow::Array>> CombineLabels(
    const std::shared_ptr<const arrow::ChunkedArray>& labels) {
    if (!labels) {
        return Invalid("cannot combine absent labels");
    }
    if (labels->num_chunks() == 0) {
        return arrow::MakeArrayOfNull(labels->type(), 0);
    }
    if (labels->num_chunks() == 1) {
        return labels->chunk(0);
    }
    return arrow::Concatenate(labels->chunks());
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> MakeCacheBatch(
    const SparseFeatureDataset& dataset) {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> columns;

    ARROW_ASSIGN_OR_RAISE(
        auto row_offsets,
        (BuildPrimitiveArray<arrow::Int32Builder>(dataset.GetRowOffsets())));
    ARROW_ASSIGN_OR_RAISE(auto row_offset_list,
                          WrapInSingleList(row_offsets));
    fields.push_back(arrow::field("row_offsets", row_offset_list->type()));
    columns.push_back(std::move(row_offset_list));

    ARROW_ASSIGN_OR_RAISE(
        auto column_indices,
        (BuildPrimitiveArray<arrow::Int32Builder>(
            dataset.GetColumnIndices())));
    ARROW_ASSIGN_OR_RAISE(auto column_index_list,
                          WrapInSingleList(column_indices));
    fields.push_back(
        arrow::field("column_indices", column_index_list->type()));
    columns.push_back(std::move(column_index_list));

    ARROW_ASSIGN_OR_RAISE(
        auto values,
        (BuildPrimitiveArray<arrow::FloatBuilder>(dataset.GetValues())));
    ARROW_ASSIGN_OR_RAISE(auto value_list, WrapInSingleList(values));
    fields.push_back(arrow::field("values", value_list->type()));
    columns.push_back(std::move(value_list));

    if (!dataset.GetFeatureNames().empty()) {
        ARROW_ASSIGN_OR_RAISE(auto feature_names,
                              BuildStringArray(dataset.GetFeatureNames()));
        ARROW_ASSIGN_OR_RAISE(auto feature_name_list,
                              WrapInSingleList(feature_names));
        fields.push_back(
            arrow::field("feature_names", feature_name_list->type()));
        columns.push_back(std::move(feature_name_list));
    }

    if (dataset.GetLabels()) {
        ARROW_ASSIGN_OR_RAISE(auto labels,
                              CombineLabels(dataset.GetLabels()));
        ARROW_ASSIGN_OR_RAISE(auto label_list, WrapInSingleList(labels));
        fields.push_back(arrow::field("labels", label_list->type()));
        columns.push_back(std::move(label_list));
    }

    auto metadata = arrow::KeyValueMetadata::Make(
        {std::string(kFormatKey), std::string(kVersionKey),
         std::string(kNameKey), std::string(kRowsKey),
         std::string(kFeaturesKey), std::string(kLabelNameKey)},
        {std::string(kFormatValue),
         std::to_string(SparseFeatureDatasetCache::kFormatVersion),
         dataset.GetName(), std::to_string(dataset.GetNumRows()),
         std::to_string(dataset.GetNumFeatures()), dataset.GetLabelName()});
    const auto schema = arrow::schema(std::move(fields), std::move(metadata));
    return arrow::RecordBatch::Make(schema, 1, std::move(columns));
}

arrow::Result<std::string> RequiredMetadata(
    const std::shared_ptr<const arrow::KeyValueMetadata>& metadata,
    std::string_view key) {
    if (!metadata) {
        return Invalid("schema metadata is missing");
    }
    auto value = metadata->Get(key);
    if (!value.ok()) {
        return Invalid("required metadata key is missing: " +
                       std::string(key));
    }
    return value;
}

arrow::Result<int64_t> ParseNonNegativeInt64(const std::string& text,
                                             std::string_view field) {
    int64_t value = 0;
    const char* begin = text.data();
    const char* end = begin + text.size();
    const auto [parsed_end, error] = std::from_chars(begin, end, value);
    if (error != std::errc() || parsed_end != end || value < 0) {
        return Invalid("invalid non-negative integer metadata: " +
                       std::string(field));
    }
    return value;
}

arrow::Result<std::shared_ptr<arrow::Array>> ReadListValues(
    const arrow::RecordBatch& batch,
    const std::string& name,
    bool required) {
    auto column = batch.GetColumnByName(name);
    if (!column) {
        if (!required) return std::shared_ptr<arrow::Array>();
        return Invalid("required cache column is missing: " + name);
    }
    if (column->type_id() != arrow::Type::LIST || column->length() != 1) {
        return Invalid("cache column must be a single-row list: " + name);
    }
    const auto list = std::static_pointer_cast<arrow::ListArray>(column);
    if (list->IsNull(0)) {
        return Invalid("cache list must not be null: " + name);
    }
    return list->value_slice(0);
}

template <typename ArrayType, typename Value>
arrow::Result<std::vector<Value>> ReadPrimitiveValues(
    const std::shared_ptr<arrow::Array>& values,
    arrow::Type::type expected_type,
    const std::string& name) {
    if (!values || values->type_id() != expected_type) {
        return Invalid("cache value type mismatch: " + name);
    }
    if (values->null_count() != 0) {
        return Invalid("cache values must not contain nulls: " + name);
    }

    const auto typed = std::static_pointer_cast<ArrayType>(values);
    std::vector<Value> result;
    result.reserve(static_cast<size_t>(typed->length()));
    for (int64_t index = 0; index < typed->length(); ++index) {
        result.push_back(typed->Value(index));
    }
    return result;
}

arrow::Result<std::vector<std::string>> ReadFeatureNames(
    const std::shared_ptr<arrow::Array>& values) {
    if (!values) return std::vector<std::string>();
    if (values->type_id() != arrow::Type::STRING) {
        return Invalid("cache value type mismatch: feature_names");
    }
    if (values->null_count() != 0) {
        return Invalid("feature_names must not contain nulls");
    }
    const auto strings = std::static_pointer_cast<arrow::StringArray>(values);
    std::vector<std::string> result;
    result.reserve(static_cast<size_t>(strings->length()));
    for (int64_t index = 0; index < strings->length(); ++index) {
        result.push_back(strings->GetString(index));
    }
    return result;
}

} // namespace

arrow::Status SparseFeatureDatasetCache::SaveAtomically(
    const SparseFeatureDataset& dataset,
    const std::string& path) {
    if (path.empty()) {
        return Invalid("cache path must not be empty");
    }

    ARROW_ASSIGN_OR_RAISE(auto batch, MakeCacheBatch(dataset));
    const fs::path destination(path);
    const fs::path temporary(TemporaryPathFor(destination));
    TemporaryFileGuard temporary_guard(temporary);

    std::error_code directory_error;
    const fs::path parent = destination.parent_path();
    if (!parent.empty()) {
        fs::create_directories(parent, directory_error);
        if (directory_error) {
            return arrow::Status::IOError(
                "SparseFeatureDatasetCache: could not create cache directory: ",
                directory_error.message());
        }
    }

    ARROW_ASSIGN_OR_RAISE(auto output,
                          arrow::io::FileOutputStream::Open(
                              temporary.string()));
    ARROW_ASSIGN_OR_RAISE(auto writer,
                          arrow::ipc::MakeFileWriter(output, batch->schema()));
    ARROW_RETURN_NOT_OK(writer->WriteRecordBatch(*batch));
    ARROW_RETURN_NOT_OK(writer->Close());
    ARROW_RETURN_NOT_OK(output->Close());
    ARROW_RETURN_NOT_OK(ReplaceFileAtomically(temporary, destination));
    temporary_guard.Release();
    return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<SparseFeatureDataset>>
SparseFeatureDatasetCache::Load(const std::string& path) {
    if (path.empty()) {
        return Invalid("cache path must not be empty");
    }

    ARROW_ASSIGN_OR_RAISE(auto input,
                          arrow::io::ReadableFile::Open(path));
    ARROW_ASSIGN_OR_RAISE(
        auto reader,
        arrow::ipc::RecordBatchFileReader::Open(std::move(input)));
    if (reader->num_record_batches() != 1) {
        return Invalid("cache must contain exactly one record batch");
    }
    ARROW_ASSIGN_OR_RAISE(auto batch, reader->ReadRecordBatch(0));
    if (!batch || batch->num_rows() != 1) {
        return Invalid("cache record batch must contain exactly one row");
    }

    const auto metadata = batch->schema()->metadata();
    ARROW_ASSIGN_OR_RAISE(auto format,
                          RequiredMetadata(metadata, kFormatKey));
    if (format != kFormatValue) {
        return Invalid("cache format identifier is not supported");
    }
    ARROW_ASSIGN_OR_RAISE(auto version_text,
                          RequiredMetadata(metadata, kVersionKey));
    ARROW_ASSIGN_OR_RAISE(
        auto version,
        ParseNonNegativeInt64(version_text, kVersionKey));
    if (version != kFormatVersion) {
        return Invalid("cache version is not supported: " + version_text);
    }

    SparseFeatureDataset::Contents contents;
    ARROW_ASSIGN_OR_RAISE(contents.name,
                          RequiredMetadata(metadata, kNameKey));
    ARROW_ASSIGN_OR_RAISE(auto rows_text,
                          RequiredMetadata(metadata, kRowsKey));
    ARROW_ASSIGN_OR_RAISE(
        contents.num_rows,
        ParseNonNegativeInt64(rows_text, kRowsKey));
    ARROW_ASSIGN_OR_RAISE(auto features_text,
                          RequiredMetadata(metadata, kFeaturesKey));
    ARROW_ASSIGN_OR_RAISE(
        contents.num_features,
        ParseNonNegativeInt64(features_text, kFeaturesKey));
    ARROW_ASSIGN_OR_RAISE(contents.label_name,
                          RequiredMetadata(metadata, kLabelNameKey));

    ARROW_ASSIGN_OR_RAISE(auto row_offsets,
                          ReadListValues(*batch, "row_offsets", true));
    ARROW_ASSIGN_OR_RAISE(
        contents.row_offsets,
        (ReadPrimitiveValues<arrow::Int32Array, int32_t>(
            row_offsets, arrow::Type::INT32, "row_offsets")));
    ARROW_ASSIGN_OR_RAISE(auto column_indices,
                          ReadListValues(*batch, "column_indices", true));
    ARROW_ASSIGN_OR_RAISE(
        contents.column_indices,
        (ReadPrimitiveValues<arrow::Int32Array, int32_t>(
            column_indices, arrow::Type::INT32, "column_indices")));
    ARROW_ASSIGN_OR_RAISE(auto values,
                          ReadListValues(*batch, "values", true));
    ARROW_ASSIGN_OR_RAISE(
        contents.values,
        (ReadPrimitiveValues<arrow::FloatArray, float>(
            values, arrow::Type::FLOAT, "values")));

    ARROW_ASSIGN_OR_RAISE(
        auto feature_names,
        ReadListValues(*batch, "feature_names", false));
    ARROW_ASSIGN_OR_RAISE(contents.feature_names,
                          ReadFeatureNames(feature_names));

    ARROW_ASSIGN_OR_RAISE(auto labels,
                          ReadListValues(*batch, "labels", false));
    if (labels) {
        contents.labels = std::make_shared<arrow::ChunkedArray>(labels);
    }
    return SparseFeatureDataset::Create(std::move(contents));
}

} // namespace cyxwiz
