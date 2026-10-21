#pragma once

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace cyxwiz {

enum class SequenceVocabularyKind {
    Token,
    PartOfSpeech,
    Tag,
};

struct SequenceVocabularyConfig {
    SequenceVocabularyKind kind = SequenceVocabularyKind::Token;
    size_t min_frequency = 1;
    size_t max_size = 0;  // 0 = unlimited, includes special tokens
    bool lowercase = false;
    bool add_pad = true;
    bool add_unk = true;
    std::string pad_token = "[PAD]";
    std::string unk_token = "[UNK]";
};

class SequenceVocabulary {
public:
    SequenceVocabulary() = default;

    SequenceVocabulary(std::vector<std::string> values,
                       int64_t pad_id,
                       int64_t unk_id)
        : id_to_value_(std::move(values)),
          pad_id_(pad_id),
          unk_id_(unk_id) {
        for (size_t i = 0; i < id_to_value_.size(); ++i) {
            value_to_id_[id_to_value_[i]] = static_cast<int64_t>(i);
        }
    }

    size_t Size() const { return id_to_value_.size(); }
    bool Empty() const { return id_to_value_.empty(); }
    int64_t PadId() const { return pad_id_; }
    int64_t UnkId() const { return unk_id_; }
    bool HasPad() const { return pad_id_ >= 0; }
    bool HasUnk() const { return unk_id_ >= 0; }

    const std::vector<std::string>& Values() const { return id_to_value_; }

    bool Contains(const std::string& value) const {
        return value_to_id_.find(value) != value_to_id_.end();
    }

    int64_t IdFor(const std::string& value) const {
        auto it = value_to_id_.find(value);
        if (it != value_to_id_.end()) {
            return it->second;
        }
        if (unk_id_ >= 0) {
            return unk_id_;
        }
        throw std::runtime_error("sequence vocabulary value is unknown: " + value);
    }

    const std::string& ValueFor(int64_t id) const {
        if (id < 0 || static_cast<size_t>(id) >= id_to_value_.size()) {
            throw std::runtime_error("sequence vocabulary id is out of range");
        }
        return id_to_value_[static_cast<size_t>(id)];
    }

    std::vector<int64_t> Encode(const std::vector<std::string>& values) const {
        std::vector<int64_t> ids;
        ids.reserve(values.size());
        for (const auto& value : values) {
            ids.push_back(IdFor(value));
        }
        return ids;
    }

private:
    std::vector<std::string> id_to_value_;
    std::unordered_map<std::string, int64_t> value_to_id_;
    int64_t pad_id_ = -1;
    int64_t unk_id_ = -1;
};

inline std::string NormalizeSequenceVocabValue(std::string value,
                                               bool lowercase) {
    if (lowercase) {
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) {
                           return static_cast<char>(std::tolower(c));
                       });
    }
    return value;
}

inline SequenceVocabulary BuildSequenceVocabulary(
    const std::vector<std::vector<std::string>>& sequences,
    SequenceVocabularyConfig config = {}) {
    if (config.min_frequency == 0) {
        config.min_frequency = 1;
    }
    if (config.kind == SequenceVocabularyKind::Tag) {
        config.lowercase = false;
        config.add_pad = false;
        config.add_unk = false;
    }

    std::unordered_map<std::string, size_t> counts;
    for (const auto& sequence : sequences) {
        for (const auto& value : sequence) {
            if (!value.empty()) {
                ++counts[NormalizeSequenceVocabValue(value, config.lowercase)];
            }
        }
    }

    std::vector<std::pair<std::string, size_t>> candidates;
    candidates.reserve(counts.size());
    for (const auto& [value, count] : counts) {
        if (count >= config.min_frequency &&
            value != config.pad_token &&
            value != config.unk_token) {
            candidates.push_back({value, count});
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) {
                  if (a.second != b.second) {
                      return a.second > b.second;
                  }
                  return a.first < b.first;
              });

    std::vector<std::string> values;
    int64_t pad_id = -1;
    int64_t unk_id = -1;

    const auto can_add = [&]() {
        return config.max_size == 0 || values.size() < config.max_size;
    };
    const auto add_value = [&](const std::string& value) {
        if (!can_add()) {
            return false;
        }
        values.push_back(value);
        return true;
    };

    if (config.add_pad && add_value(config.pad_token)) {
        pad_id = static_cast<int64_t>(values.size() - 1);
    }
    if (config.add_unk && add_value(config.unk_token)) {
        unk_id = static_cast<int64_t>(values.size() - 1);
    }

    if (config.kind == SequenceVocabularyKind::Tag) {
        auto it = std::find_if(candidates.begin(), candidates.end(),
                               [](const auto& candidate) {
                                   return candidate.first == "O";
                               });
        if (it != candidates.end() && add_value(it->first)) {
            candidates.erase(it);
        }
    }

    for (const auto& [value, _] : candidates) {
        if (!add_value(value)) {
            break;
        }
    }

    return SequenceVocabulary(std::move(values), pad_id, unk_id);
}

} // namespace cyxwiz
