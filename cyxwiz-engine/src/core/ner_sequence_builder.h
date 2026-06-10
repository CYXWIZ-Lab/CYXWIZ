#pragma once

#include "sequence_batcher.h"
#include "sequence_vocabulary.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

struct NERSequenceRow {
    std::vector<std::string> tokens;
    std::vector<std::string> pos_tags;
    std::vector<std::string> ner_tags;
};

struct NERSequenceBuilderConfig {
    SequenceVocabularyConfig token_vocabulary;
    SequenceVocabularyConfig pos_vocabulary;
    SequenceVocabularyConfig tag_vocabulary;
    SequenceBatcherConfig batcher;
    bool use_pos_tags = true;
    bool require_tags = true;

    NERSequenceBuilderConfig() {
        token_vocabulary.kind = SequenceVocabularyKind::Token;
        pos_vocabulary.kind = SequenceVocabularyKind::PartOfSpeech;
        tag_vocabulary.kind = SequenceVocabularyKind::Tag;
        batcher.tag_ignore_index = -100;
    }
};

struct NERSequenceBuildResult {
    SequenceVocabulary token_vocabulary;
    SequenceVocabulary pos_vocabulary;
    SequenceVocabulary tag_vocabulary;
    std::vector<SequenceSample> samples;
    SequenceBatcherConfig batcher_config;
    bool has_pos_tags = false;
    bool has_tags = false;

    SequenceBatcher CreateBatcher() const {
        return SequenceBatcher(samples, batcher_config);
    }
};

class NERSequenceBuilder {
public:
    explicit NERSequenceBuilder(NERSequenceBuilderConfig config = {})
        : config_(std::move(config)) {
        config_.token_vocabulary.kind = SequenceVocabularyKind::Token;
        config_.pos_vocabulary.kind = SequenceVocabularyKind::PartOfSpeech;
        config_.tag_vocabulary.kind = SequenceVocabularyKind::Tag;
    }

    NERSequenceBuildResult Build(const std::vector<NERSequenceRow>& rows) const {
        ValidateRows(rows);

        const bool has_pos = HasPosTags(rows);
        const bool has_tags = HasTags(rows);

        std::vector<std::vector<std::string>> token_sequences;
        std::vector<std::vector<std::string>> pos_sequences;
        std::vector<std::vector<std::string>> tag_sequences;
        token_sequences.reserve(rows.size());
        pos_sequences.reserve(rows.size());
        tag_sequences.reserve(rows.size());

        for (const auto& row : rows) {
            token_sequences.push_back(row.tokens);
            if (has_pos) {
                pos_sequences.push_back(row.pos_tags);
            }
            if (has_tags) {
                tag_sequences.push_back(row.ner_tags);
            }
        }

        NERSequenceBuildResult result;
        result.token_vocabulary =
            BuildSequenceVocabulary(token_sequences, config_.token_vocabulary);
        if (has_pos) {
            result.pos_vocabulary =
                BuildSequenceVocabulary(pos_sequences, config_.pos_vocabulary);
        }
        if (has_tags) {
            result.tag_vocabulary =
                BuildSequenceVocabulary(tag_sequences, config_.tag_vocabulary);
        }

        result.samples.reserve(rows.size());
        for (const auto& row : rows) {
            SequenceSample sample;
            sample.word_ids = EncodeWithConfig(
                result.token_vocabulary,
                row.tokens,
                config_.token_vocabulary.lowercase);
            if (has_pos) {
                sample.pos_ids = EncodeWithConfig(
                    result.pos_vocabulary,
                    row.pos_tags,
                    config_.pos_vocabulary.lowercase);
            }
            if (has_tags) {
                sample.tag_ids = EncodeWithConfig(
                    result.tag_vocabulary,
                    row.ner_tags,
                    false);
            }
            result.samples.push_back(std::move(sample));
        }

        result.batcher_config = config_.batcher;
        result.batcher_config.word_pad_id = result.token_vocabulary.HasPad()
            ? result.token_vocabulary.PadId()
            : config_.batcher.word_pad_id;
        if (has_pos && result.pos_vocabulary.HasPad()) {
            result.batcher_config.pos_pad_id = result.pos_vocabulary.PadId();
        }
        result.has_pos_tags = has_pos;
        result.has_tags = has_tags;
        return result;
    }

private:
    static std::vector<int64_t> EncodeWithConfig(
        const SequenceVocabulary& vocabulary,
        const std::vector<std::string>& values,
        bool lowercase) {
        std::vector<int64_t> ids;
        ids.reserve(values.size());
        for (const auto& value : values) {
            ids.push_back(
                vocabulary.IdFor(NormalizeSequenceVocabValue(value, lowercase)));
        }
        return ids;
    }

    bool HasPosTags(const std::vector<NERSequenceRow>& rows) const {
        if (!config_.use_pos_tags) {
            return false;
        }
        for (const auto& row : rows) {
            if (!row.pos_tags.empty()) {
                return true;
            }
        }
        return false;
    }

    bool HasTags(const std::vector<NERSequenceRow>& rows) const {
        for (const auto& row : rows) {
            if (!row.ner_tags.empty()) {
                return true;
            }
        }
        return false;
    }

    void ValidateRows(const std::vector<NERSequenceRow>& rows) const {
        if (rows.empty()) {
            throw std::runtime_error("NERSequenceBuilder requires at least one row");
        }

        const bool has_pos = HasPosTags(rows);
        const bool has_tags = HasTags(rows);
        if (config_.require_tags && !has_tags) {
            throw std::runtime_error("NERSequenceBuilder requires tag labels");
        }

        for (size_t i = 0; i < rows.size(); ++i) {
            const auto& row = rows[i];
            const std::string prefix =
                "NERSequenceBuilder row " + std::to_string(i) + ": ";
            if (row.tokens.empty()) {
                throw std::runtime_error(prefix + "tokens are empty");
            }
            if (has_pos && row.pos_tags.size() != row.tokens.size()) {
                throw std::runtime_error(
                    prefix + "POS tag count must match token count");
            }
            if (has_tags && row.ner_tags.size() != row.tokens.size()) {
                throw std::runtime_error(
                    prefix + "NER tag count must match token count");
            }
            if (config_.require_tags && row.ner_tags.empty()) {
                throw std::runtime_error(prefix + "NER tags are empty");
            }
        }
    }

    NERSequenceBuilderConfig config_;
};

inline NERSequenceBuildResult BuildNERSequenceData(
    const std::vector<NERSequenceRow>& rows,
    NERSequenceBuilderConfig config = {}) {
    return NERSequenceBuilder(std::move(config)).Build(rows);
}

} // namespace cyxwiz
