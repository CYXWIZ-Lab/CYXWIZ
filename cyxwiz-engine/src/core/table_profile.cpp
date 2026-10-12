#include "table_profile.h"

#include <arrow/api.h>

#include <algorithm>
#include <unordered_map>

namespace cyxwiz {

namespace {
bool IsNumeric(arrow::Type::type id) {
    switch (id) {
        case arrow::Type::INT8:
        case arrow::Type::INT16:
        case arrow::Type::INT32:
        case arrow::Type::INT64:
        case arrow::Type::UINT8:
        case arrow::Type::UINT16:
        case arrow::Type::UINT32:
        case arrow::Type::UINT64:
        case arrow::Type::FLOAT:
        case arrow::Type::DOUBLE:
            return true;
        default:
            return false;
    }
}

double NumberAt(const arrow::Array& array, int64_t i) {
    switch (array.type_id()) {
        case arrow::Type::INT8: return static_cast<const arrow::Int8Array&>(array).Value(i);
        case arrow::Type::INT16: return static_cast<const arrow::Int16Array&>(array).Value(i);
        case arrow::Type::INT32: return static_cast<const arrow::Int32Array&>(array).Value(i);
        case arrow::Type::INT64: return static_cast<double>(static_cast<const arrow::Int64Array&>(array).Value(i));
        case arrow::Type::UINT8: return static_cast<const arrow::UInt8Array&>(array).Value(i);
        case arrow::Type::UINT16: return static_cast<const arrow::UInt16Array&>(array).Value(i);
        case arrow::Type::UINT32: return static_cast<const arrow::UInt32Array&>(array).Value(i);
        case arrow::Type::UINT64: return static_cast<double>(static_cast<const arrow::UInt64Array&>(array).Value(i));
        case arrow::Type::FLOAT: return static_cast<const arrow::FloatArray&>(array).Value(i);
        case arrow::Type::DOUBLE: return static_cast<const arrow::DoubleArray&>(array).Value(i);
        default: return 0.0;
    }
}

std::string TextAt(const arrow::Array& array, int64_t i) {
    if (array.type_id() == arrow::Type::STRING) {
        return std::string(static_cast<const arrow::StringArray&>(array).GetView(i));
    }
    if (array.type_id() == arrow::Type::LARGE_STRING) {
        return std::string(static_cast<const arrow::LargeStringArray&>(array).GetView(i));
    }
    auto scalar = array.GetScalar(i);
    return scalar.ok() ? (*scalar)->ToString() : std::string();
}
}  // namespace

std::vector<ArrowColumnProfile> ProfileArrowTable(const arrow::Table& table, size_t top_values) {
    std::vector<ArrowColumnProfile> profiles;
    for (int c = 0; c < table.num_columns(); ++c) {
        const auto& field = table.schema()->field(c);
        ArrowColumnProfile profile;
        profile.name = field->name();
        profile.type = field->type()->ToString();
        profile.rows = table.num_rows();
        profile.numeric = IsNumeric(field->type()->id());

        std::unordered_map<std::string, int64_t> counts;
        std::vector<std::string> first_seen;  // order values first appeared
        std::vector<double> numbers;
        double sum = 0.0;
        for (const auto& chunk : table.column(c)->chunks()) {
            for (int64_t i = 0; i < chunk->length(); ++i) {
                if (chunk->IsNull(i)) {
                    ++profile.nulls;
                    continue;
                }
                std::string text = TextAt(*chunk, i);
                if (text.empty()) ++profile.empty;
                auto [it, inserted] = counts.try_emplace(std::move(text), 0);
                if (inserted) first_seen.push_back(it->first);
                else ++profile.repeated;
                ++it->second;
                if (profile.numeric) {
                    const double value = NumberAt(*chunk, i);
                    numbers.push_back(value);
                    sum += value;
                }
            }
        }
        profile.distinct = static_cast<int64_t>(counts.size());
        if (profile.numeric && !numbers.empty()) {
            std::sort(numbers.begin(), numbers.end());
            profile.min = numbers.front();
            profile.max = numbers.back();
            profile.mean = sum / static_cast<double>(numbers.size());
            const size_t mid = numbers.size() / 2;
            profile.median = numbers.size() % 2 ? numbers[mid] : (numbers[mid - 1] + numbers[mid]) / 2.0;
        }
        // Most common values; a stable sort keeps first-seen order on ties.
        std::vector<std::pair<std::string, int64_t>> ordered;
        ordered.reserve(first_seen.size());
        for (const auto& value : first_seen) ordered.emplace_back(value, counts[value]);
        std::stable_sort(ordered.begin(), ordered.end(),
                         [](const auto& a, const auto& b) { return a.second > b.second; });
        if (ordered.size() > top_values) ordered.resize(top_values);
        profile.top = std::move(ordered);
        profiles.push_back(std::move(profile));
    }
    return profiles;
}

}  // namespace cyxwiz
