#pragma once

#include "pipeline_operator.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cyxwiz {

/**
 * TextTokenizerOperator — Cat-1 Band 1 pipeline operator.
 *
 * Transforms an Arrow table containing raw text + label columns into a
 * wide tokenized table that ArrowDatasetBatcher can consume directly:
 *   `tok_0, tok_1, ..., tok_{max_length-1}, y`
 *
 * One row per sample. Token columns are float32 (the EmbeddingLayer
 * casts back to int at lookup time, matching how the existing text
 * path works). Label column `y` is the row's class index, written as
 * int32 — built either by copying a numeric label column or by
 * mapping a string label column to integer class IDs (string→int
 * vocab built in this Apply pass).
 *
 * v1 combines tokenize + vocab build + padding into one operator; its
 * own settings hold the vocabulary and padding (the legacy TextVocabulary
 * and TextPadding nodes are not applied; the compiler says so).
 *
 * Closes Fix B from tofix.md "TextTokenizer is a config extractor,
 * not a pipeline operation" — this operator IS the real pipeline
 * operation. The legacy config-extractor path in graph_compiler.cpp
 * still works for graphs that haven't been migrated; both can coexist
 * because the pipeline run applies it only to Arrow tables
 * (text-via-RegisterTextDataset graphs are skipped).
 *
 * Params:
 *   text_col          (required)        — string column with raw text
 *   label_col         (optional)        — column to use as label
 *   max_length        (default 256)     — pad/truncate to this length
 *   tokenizer_type    (default 1)       — 0=Whitespace, 1=Word, 2=Character, 3=ByteBPE
 *   lowercase         (default true)    — lowercase before tokenizing
 *   min_word_freq     (default 2)       — vocabulary frequency floor
 *   max_vocab_size    (default 10000)   — vocabulary cap
 *   vocab_file        (optional)        — load vocabulary instead of training
 *   vocab_build_if_missing (default false) — train and save vocab_file if absent
 *   output_mode       (default wide)     — wide, causal_windows, decode, roundtrip
 *   target_text_col   (optional)         — causal_windows only: response text per row. Each row
 *                                          becomes one window (text_col, then this column plus
 *                                          EOS) whose loss covers only the response targets
 *                                          (__target_mask); rows longer than one window fail.
 *   token_ids_col     (default token_ids)— decode input column containing id text
 */
class Tokenizer;

class TextTokenizerOperator : public IPipelineOperator {
public:
    TextTokenizerOperator();
    ~TextTokenizerOperator() override;

    std::string GetName() const override { return "TextTokenizer"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(
        const std::map<std::string, std::string>& params,
        std::string& error) override;

    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;

    void SetProgressCallback(PipelineOperatorProgressCallback callback) override {
        progress_callback_ = std::move(callback);
    }

    size_t GetLastVocabSize() const { return last_vocab_size_; }

    // Wide rows stream a row group at a time (TOFIX157 A4), so a table kept
    // on disk tokenizes without being read whole: the fit pass counts the
    // tokens (Whitespace, Word, Character) and the string labels' classes in
    // file order; EndFit builds the vocabulary Apply would (saving vocab_file
    // when asked); ApplyBatch writes each row group's tok_i and y. A saved
    // vocab_file streams with any tokenizer.
    bool SupportsStreaming() const override;
    bool NeedsFitPass() const override { return true; }
    arrow::Status BeginStream(const std::shared_ptr<arrow::Schema>& schema) override;
    arrow::Status FitBatch(const std::shared_ptr<arrow::Table>& batch) override;
    arrow::Status EndFit() override;
    arrow::Result<std::shared_ptr<arrow::Table>> ApplyBatch(const std::shared_ptr<arrow::Table>& batch) override;
    arrow::Status EndStream() override;

private:
    Tokenizer MakeTokenizer() const;
    // A loaded or built vocabulary against the node's BPE settings (throws).
    void CheckLoadedVocabulary(const Tokenizer& tokenizer) const;
    arrow::Status SaveBuiltVocabulary(const Tokenizer& tokenizer) const;
    // Rows tok_0..tok_{max_length-1} (float32) and y (int32, class names in
    // its metadata) from texts and their class ids (none: no y).
    using WideProgress = std::function<void(const std::string& stage, const std::string& message, double progress,
                                            uint64_t processed, uint64_t total)>;
    arrow::Result<std::shared_ptr<arrow::Table>> BuildWideTable(
        const std::vector<std::string>& texts, const Tokenizer& tokenizer, const std::vector<int>& labels,
        const std::vector<std::string>& class_names, const WideProgress& report);
    // A row group's class ids: string labels by the fit pass's classes.
    arrow::Status StreamLabels(const std::shared_ptr<arrow::Table>& batch, std::vector<int>& labels);
    arrow::Status ValidateWindowInput(const std::shared_ptr<arrow::Table>& input) const;
    arrow::Result<std::shared_ptr<arrow::Table>> BuildTokenWindows(
        const std::shared_ptr<arrow::Table>& input, const std::vector<std::string>& texts, Tokenizer& tokenizer,
        const std::vector<std::string>* targets = nullptr);
    arrow::Result<std::shared_ptr<arrow::Table>> DecodeTokenRows(
        const std::shared_ptr<arrow::Table>& input, Tokenizer& tokenizer);
    arrow::Result<std::shared_ptr<arrow::Table>> RoundTripTextRows(
        const std::vector<std::string>& texts, Tokenizer& tokenizer);
    std::string output_mode_ = "wide";
    std::string document_id_col_;
    std::string split_col_ = "split";
    std::string text_col_;
    std::string target_text_col_;
    std::string token_ids_col_ = "token_ids";
    std::string label_col_;
    int max_length_ = 256;
    int tokenizer_type_ = 1;
    bool lowercase_ = true;
    int min_word_freq_ = 2;
    int max_vocab_size_ = 10000;
    int pad_value_ = 0;
    std::string vocab_file_;
    std::string bpe_initial_unit_; // omitted: artifact units on load, byte on fit
    std::string bpe_piece_policy_; // omitted: artifact policy on load, whitespace_v1 on fit
    bool vocab_build_if_missing_ = false;
    size_t last_vocab_size_ = 0;
    PipelineOperatorProgressCallback progress_callback_;

    // Streaming state (BeginStream to EndStream).
    std::unique_ptr<Tokenizer> stream_tokenizer_;
    bool stream_vocab_ready_ = false;
    std::unordered_map<std::string, int64_t> stream_counts_;
    bool stream_string_labels_ = false;
    std::map<std::string, int> stream_label_ids_;
    std::vector<std::string> stream_class_names_;
};

} // namespace cyxwiz
