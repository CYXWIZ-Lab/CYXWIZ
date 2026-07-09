#include "text_preprocessing_tracer.h"

#include "data_registry.h"
#include "error_codes.h"
#include "formats/text_dataset.h"
#include <algorithm>
#include <spdlog/spdlog.h>

namespace cyxwiz {

namespace {

constexpr size_t kTextPreviewLimit = 220;
constexpr size_t kTokenPreviewLimit = 24;
constexpr size_t kIdPreviewLimit = 48;

std::string NodeName(gui::NodeType type) {
    switch (type) {
        case gui::NodeType::TextTokenizer: return "TextTokenizer";
        case gui::NodeType::TextVocabulary: return "TextVocabulary";
        case gui::NodeType::TextPadding: return "TextPadding";
        default: return "TextPreprocessing";
    }
}

int FindNodeId(const std::vector<gui::MLNode>& nodes, gui::NodeType type) {
    for (const auto& node : nodes) {
        if (node.type == type) {
            return node.id;
        }
    }
    return -1;
}

TextDatasetConfig BuildTextDatasetConfig(
    const DataRegistry::TextDatasetEntry& entry,
    const TextPreprocessingConfig& preprocess) {
    TextDatasetConfig cfg;
    cfg.text_column = entry.text_column;
    cfg.label_column = entry.label_column;
    cfg.has_labels = entry.has_labels;

    switch (entry.tokenizer_type) {
        case 0: cfg.tokenizer_type = TokenizerType::Whitespace; break;
        case 2: cfg.tokenizer_type = TokenizerType::Character; break;
        case 1:
        default: cfg.tokenizer_type = TokenizerType::Word; break;
    }
    cfg.max_length = entry.max_length;
    cfg.lowercase = entry.lowercase;
    cfg.do_padding = entry.do_padding;
    cfg.do_truncation = entry.do_truncation;
    cfg.min_word_freq = entry.min_word_freq;
    cfg.max_vocab_size = entry.max_vocab_size;
    cfg.vocab_file = entry.vocab_file;

    if (preprocess.has_tokenizer_node) {
        switch (preprocess.tokenizer_type) {
            case 0: cfg.tokenizer_type = TokenizerType::Whitespace; break;
            case 2: cfg.tokenizer_type = TokenizerType::Character; break;
            case 1:
            default: cfg.tokenizer_type = TokenizerType::Word; break;
        }
        cfg.lowercase = preprocess.lowercase;
        cfg.do_padding = preprocess.do_padding;
        cfg.do_truncation = preprocess.do_truncation;
        cfg.max_length = preprocess.max_length;
        cfg.min_word_freq = preprocess.min_word_freq;
        cfg.max_vocab_size = preprocess.max_vocab_size;
    }

    if (preprocess.has_vocabulary_node) {
        cfg.min_word_freq = preprocess.min_word_freq;
        cfg.max_vocab_size = preprocess.max_vocab_size;
        cfg.vocab_file = preprocess.vocab_file;
    }

    if (preprocess.has_padding_node) {
        cfg.do_padding = true;
        cfg.max_length = preprocess.max_length;
    }

    return cfg;
}

template <typename T>
std::vector<T> PreviewVector(const std::vector<T>& values, size_t limit) {
    const size_t n = std::min(values.size(), limit);
    return std::vector<T>(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(n));
}

std::string PreviewText(const std::string& text) {
    if (text.size() <= kTextPreviewLimit) {
        return text;
    }
    return text.substr(0, kTextPreviewLimit) + "...";
}

void AttachTextPreprocessingContext(DebugTraceRecord& record,
                                    const std::string& diagnostic_phase,
                                    const std::string& component) {
    DebugNodeTraceContract::AttachDiagnosticContext(
        record,
        diagnostic_phase,
        component,
        "cyxwiz-engine/src/core/text_preprocessing_tracer.cpp",
        "cyxwiz::TextPreprocessingTracer::TraceSample");
}

DebugTraceRecord MakeRecord(const std::string& run_id,
                            gui::NodeType type,
                            int node_id,
                            const std::string& phase) {
    DebugTraceRecord record;
    record.run_id = run_id;
    record.node_id = node_id;
    record.node_name = NodeName(type);
    record.node_type = NodeName(type);
    record.phase = phase;
    record.role = DebugTraceRole::PreprocessingOutput;
    record.status = "ok";
    AttachTextPreprocessingContext(record, phase, record.node_type);
    return record;
}

} // namespace

std::vector<DebugTraceRecord> TextPreprocessingTracer::TraceFirstSample(
    const TrainingConfiguration& config,
    const std::vector<gui::MLNode>& nodes,
    const std::string& run_id) const {
    return TraceSample(config, nodes, run_id, 0);
}

std::vector<DebugTraceRecord> TextPreprocessingTracer::TraceSample(
    const TrainingConfiguration& config,
    const std::vector<gui::MLNode>& nodes,
    const std::string& run_id,
    size_t sample_index) const {
    std::vector<DebugTraceRecord> traces;

    if (config.preprocessing_domain != PreprocessingDomain::Text ||
        config.dataset_name.empty()) {
        return traces;
    }

    const auto* entry = DataRegistry::Instance().GetTextDatasetEntry(config.dataset_name);
    if (!entry) {
        DebugTraceRecord record;
        record.run_id = run_id;
        record.phase = "Preprocessing";
        record.role = DebugTraceRole::Error;
        record.status = "failed";
        record.node_name = "TextDataset";
        record.node_type = "TextDataset";
        AttachTextPreprocessingContext(record, "data_source", "TextDataset");
        record.payload["message"] =
            "Text dataset is not registered: " + config.dataset_name;
        record.payload["error_code"] = errors::Runtime::InputDatasetMissing;
        record.issues.push_back({
            IssueLevel::Error,
            -1,
            "TextDataset",
            record.payload["message"].get<std::string>(),
            errors::Runtime::InputDatasetMissing
        });
        DebugNodeTraceContract::AttachIssueSummary(record, record.issues);
        traces.push_back(std::move(record));
        return traces;
    }

    try {
        const TextDatasetConfig cfg = BuildTextDatasetConfig(*entry, config.text_preprocessing);
        TextDataset dataset(entry->source_path, cfg);
        if (dataset.Size() == 0) {
            DebugTraceRecord record;
            record.run_id = run_id;
            record.phase = "Preprocessing";
            record.role = DebugTraceRole::Error;
            record.status = "failed";
            record.node_name = "TextDataset";
            record.node_type = "TextDataset";
            AttachTextPreprocessingContext(record, "data_source", "TextDataset");
            record.payload["message"] = "Text dataset has no samples.";
            record.payload["error_code"] = errors::Data::RowCountMismatch;
            record.issues.push_back({
                IssueLevel::Error,
                -1,
                "TextDataset",
                "Text dataset has no samples.",
                errors::Data::RowCountMismatch
            });
            DebugNodeTraceContract::AttachIssueSummary(record, record.issues);
            traces.push_back(std::move(record));
            return traces;
        }
        if (sample_index >= dataset.Size()) {
            DebugTraceRecord record;
            record.run_id = run_id;
            record.phase = "Preprocessing";
            record.role = DebugTraceRole::Error;
            record.status = "failed";
            record.node_name = "TextDataset";
            record.node_type = "TextDataset";
            AttachTextPreprocessingContext(record, "sample_selection", "TextDataset");
            record.payload["message"] =
                "Selected sample index is outside the text dataset.";
            record.payload["error_code"] = errors::Runtime::InvalidParameter;
            record.payload["sample_index"] = sample_index;
            record.payload["dataset_size"] = dataset.Size();
            record.issues.push_back({
                IssueLevel::Error,
                -1,
                "TextDataset",
                "Selected sample index is outside the text dataset.",
                errors::Runtime::InvalidParameter
            });
            DebugNodeTraceContract::AttachIssueSummary(record, record.issues);
            traces.push_back(std::move(record));
            return traces;
        }

        const std::string& raw_text = dataset.GetText(sample_index);
        const Tokenizer& tokenizer = dataset.GetTokenizer();
        const TokenizedText tokenized = tokenizer.Tokenize(raw_text);
        const Vocabulary& vocab = tokenizer.GetVocabulary();

        int unk_count = 0;
        int pad_count = 0;
        int vocab_hits = 0;
        int vocab_misses = 0;
        std::vector<std::string> missing_preview;

        for (const auto& token : tokenized.tokens) {
            if (vocab.HasWord(token)) {
                vocab_hits++;
            } else {
                vocab_misses++;
                if (missing_preview.size() < kTokenPreviewLimit) {
                    missing_preview.push_back(token);
                }
            }
        }

        for (int id : tokenized.token_ids) {
            if (id == vocab.UnkIndex()) {
                unk_count++;
            }
            if (id == vocab.PadIndex()) {
                pad_count++;
            }
        }

        const float unknown_ratio = tokenized.tokens.empty()
            ? 0.0f
            : static_cast<float>(vocab_misses) / static_cast<float>(tokenized.tokens.size());
        const float pad_ratio = tokenized.token_ids.empty()
            ? 0.0f
            : static_cast<float>(pad_count) / static_cast<float>(tokenized.token_ids.size());

        DebugTraceRecord tokenizer_record = MakeRecord(
            run_id, gui::NodeType::TextTokenizer,
            FindNodeId(nodes, gui::NodeType::TextTokenizer),
            "TextTokenizer");
        tokenizer_record.output_shape = {tokenized.tokens.size()};
        tokenizer_record.payload["dataset"] = config.dataset_name;
        tokenizer_record.payload["sample_index"] = sample_index;
        tokenizer_record.payload["dataset_size"] = dataset.Size();
        tokenizer_record.payload["raw_text_preview"] = PreviewText(raw_text);
        tokenizer_record.payload["normalized_preview"] = tokenizer.Decode(tokenized.token_ids);
        tokenizer_record.payload["token_count"] = tokenized.tokens.size();
        tokenizer_record.payload["tokens_preview"] = PreviewVector(tokenized.tokens, kTokenPreviewLimit);
        traces.push_back(std::move(tokenizer_record));

        DebugTraceRecord vocab_record = MakeRecord(
            run_id, gui::NodeType::TextVocabulary,
            FindNodeId(nodes, gui::NodeType::TextVocabulary),
            "TextVocabulary");
        vocab_record.output_shape = {tokenized.token_ids.size()};
        vocab_record.payload["vocab_size"] = vocab.Size();
        vocab_record.payload["vocab_file"] = cfg.vocab_file;
        vocab_record.payload["vocab_hits"] = vocab_hits;
        vocab_record.payload["vocab_misses"] = vocab_misses;
        vocab_record.payload["unknown_token_count"] = unk_count;
        vocab_record.payload["unknown_token_ratio"] = unknown_ratio;
        vocab_record.payload["missing_tokens_preview"] = missing_preview;
        vocab_record.payload["token_ids_preview"] = PreviewVector(tokenized.token_ids, kIdPreviewLimit);
        if (unknown_ratio > 0.2f) {
            vocab_record.status = "warning";
            vocab_record.issues.push_back({
                IssueLevel::Warning,
                vocab_record.node_id,
                vocab_record.node_name,
                "High unknown-token ratio in selected text sample.",
                errors::Data::VocabularyCoverageWarning
            });
            DebugNodeTraceContract::AttachIssueSummary(vocab_record, vocab_record.issues);
        }
        traces.push_back(std::move(vocab_record));

        DebugTraceRecord padding_record = MakeRecord(
            run_id, gui::NodeType::TextPadding,
            FindNodeId(nodes, gui::NodeType::TextPadding),
            "TextPadding");
        padding_record.output_shape = {tokenized.token_ids.size()};
        padding_record.payload["max_length"] = cfg.max_length;
        padding_record.payload["final_sequence_length"] = tokenized.token_ids.size();
        padding_record.payload["pad_count"] = pad_count;
        padding_record.payload["pad_ratio"] = pad_ratio;
        padding_record.payload["truncated"] = tokenized.truncated;
        padding_record.payload["padded"] = tokenized.padded;
        if (tokenized.truncated) {
            padding_record.status = "warning";
            padding_record.issues.push_back({
                IssueLevel::Warning,
                padding_record.node_id,
                padding_record.node_name,
                "Selected text sample was truncated.",
                errors::Compiler::InvalidParameter
            });
            DebugNodeTraceContract::AttachIssueSummary(padding_record, padding_record.issues);
        } else if (pad_ratio > 0.8f) {
            padding_record.payload["note"] =
                "Selected sample is mostly padding. This is informational for a single sample; aggregate smoke statistics should decide whether max_length is too large.";
        }
        traces.push_back(std::move(padding_record));
    } catch (const std::exception& e) {
        spdlog::warn("TextPreprocessingTracer: failed: {}", e.what());
        DebugTraceRecord record;
        record.run_id = run_id;
        record.phase = "Preprocessing";
        record.role = DebugTraceRole::Error;
        record.status = "failed";
        record.node_name = "TextPreprocessing";
        record.node_type = "TextPreprocessing";
        AttachTextPreprocessingContext(record, "materialization", "TextPreprocessing");
        record.payload["message"] = e.what();
        record.payload["error_code"] = errors::Data::MaterializationFailed;
        record.issues.push_back({
            IssueLevel::Error,
            -1,
            "TextPreprocessing",
            e.what(),
            errors::Data::MaterializationFailed
        });
        DebugNodeTraceContract::AttachIssueSummary(record, record.issues);
        traces.push_back(std::move(record));
    }

    return traces;
}

} // namespace cyxwiz
