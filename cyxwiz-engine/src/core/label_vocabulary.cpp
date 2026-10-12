#include "label_vocabulary.h"

#include <algorithm>
#include <set>
#include <string_view>

namespace cyxwiz {

namespace {

template <typename ArrayType>
void CollectValues(const arrow::Array& chunk, std::set<std::string>& out) {
    const auto& strings = static_cast<const ArrayType&>(chunk);
    for (int64_t i = 0; i < strings.length(); ++i)
        if (!strings.IsNull(i)) out.emplace(strings.GetView(i));
}

template <typename ArrayType>
arrow::Status EncodeChunk(const arrow::Array& chunk, const LabelVocabulary& vocabulary, arrow::Int32Builder& builder) {
    const auto& strings = static_cast<const ArrayType&>(chunk);
    ARROW_RETURN_NOT_OK(builder.Reserve(strings.length()));
    for (int64_t i = 0; i < strings.length(); ++i) {
        if (strings.IsNull(i)) {
            builder.UnsafeAppendNull();
            continue;
        }
        const std::string value(strings.GetView(i));
        const int32_t index = vocabulary.IndexOf(value);
        if (index < 0) return arrow::Status::Invalid("label '", value, "' is not one of the ", vocabulary.names.size(),
                                                     " known classes");
        builder.UnsafeAppend(index);
    }
    return arrow::Status::OK();
}

}  // namespace

int32_t LabelVocabulary::IndexOf(const std::string& name) const {
    const auto it = index.find(name);
    return it == index.end() ? -1 : it->second;
}

bool IsTextLabelType(const arrow::DataType& type) {
    return type.id() == arrow::Type::STRING || type.id() == arrow::Type::LARGE_STRING;
}

LabelVocabulary BuildLabelVocabulary(const std::vector<std::shared_ptr<arrow::ChunkedArray>>& columns) {
    std::set<std::string> distinct;  // sorted
    for (const auto& column : columns) {
        if (!column) continue;
        for (const auto& chunk : column->chunks()) {
            if (chunk->type_id() == arrow::Type::STRING) CollectValues<arrow::StringArray>(*chunk, distinct);
            else if (chunk->type_id() == arrow::Type::LARGE_STRING) CollectValues<arrow::LargeStringArray>(*chunk, distinct);
        }
    }
    LabelVocabulary vocabulary;
    vocabulary.names.assign(distinct.begin(), distinct.end());
    for (size_t i = 0; i < vocabulary.names.size(); ++i)
        vocabulary.index.emplace(vocabulary.names[i], static_cast<int32_t>(i));
    return vocabulary;
}

LabelVocabulary BuildLabelVocabulary(const std::vector<std::string>& values) {
    std::set<std::string> distinct;
    for (const auto& value : values)
        if (!value.empty()) distinct.insert(value);
    LabelVocabulary vocabulary;
    vocabulary.names.assign(distinct.begin(), distinct.end());
    for (size_t i = 0; i < vocabulary.names.size(); ++i)
        vocabulary.index.emplace(vocabulary.names[i], static_cast<int32_t>(i));
    return vocabulary;
}

arrow::Result<std::shared_ptr<arrow::ChunkedArray>> EncodeLabels(const arrow::ChunkedArray& column,
                                                                 const LabelVocabulary& vocabulary) {
    arrow::ArrayVector encoded;
    for (const auto& chunk : column.chunks()) {
        arrow::Int32Builder builder;
        if (chunk->type_id() == arrow::Type::STRING) {
            ARROW_RETURN_NOT_OK(EncodeChunk<arrow::StringArray>(*chunk, vocabulary, builder));
        } else if (chunk->type_id() == arrow::Type::LARGE_STRING) {
            ARROW_RETURN_NOT_OK(EncodeChunk<arrow::LargeStringArray>(*chunk, vocabulary, builder));
        } else {
            return arrow::Status::TypeError("label column holds ", chunk->type()->ToString(), ", not text");
        }
        std::shared_ptr<arrow::Array> array;
        ARROW_RETURN_NOT_OK(builder.Finish(&array));
        encoded.push_back(std::move(array));
    }
    return std::make_shared<arrow::ChunkedArray>(std::move(encoded), arrow::int32());
}

arrow::Result<std::shared_ptr<arrow::Table>> EncodeLabelColumn(const std::shared_ptr<arrow::Table>& table,
                                                                const std::string& label_column,
                                                                const LabelVocabulary& vocabulary) {
    if (!table) return arrow::Status::Invalid("no table");
    const int position = table->schema()->GetFieldIndex(label_column);
    if (position < 0) return arrow::Status::KeyError("label column '", label_column, "' not found");
    ARROW_ASSIGN_OR_RAISE(auto encoded, EncodeLabels(*table->column(position), vocabulary));
    return table->SetColumn(position, arrow::field(label_column, arrow::int32()), std::move(encoded));
}

std::shared_ptr<arrow::ChunkedArray> TextLabelColumn(const std::shared_ptr<arrow::Table>& table,
                                                     const std::string& label_column) {
    if (!table || label_column.empty()) return nullptr;
    const int position = table->schema()->GetFieldIndex(label_column);
    if (position < 0) return nullptr;
    auto column = table->column(position);
    return IsTextLabelType(*column->type()) ? column : nullptr;
}

}  // namespace cyxwiz
