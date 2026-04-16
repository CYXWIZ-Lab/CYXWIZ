#pragma once

#include "pipeline_operator.h"

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
 * v1 combines tokenize + vocab build + padding into one operator,
 * mirroring how TimeSeriesWindow combined windowing + label
 * extraction. Splitting into separate TextTokenizer / TextVocabulary
 * / TextPadding nodes is a future refinement once a real use case
 * needs the intermediate stages addressable separately.
 *
 * Closes Fix B from tofix.md "TextTokenizer is a config extractor,
 * not a pipeline operation" — this operator IS the real pipeline
 * operation. The legacy config-extractor path in graph_compiler.cpp
 * still works for graphs that haven't been migrated; both can coexist
 * because PipelineMaterializer only fires when the source dataset is
 * an Arrow table (text-via-RegisterTextDataset graphs are skipped).
 *
 * Params:
 *   text_col          (required)        — string column with raw text
 *   label_col         (optional)        — column to use as label
 *   max_length        (default 256)     — pad/truncate to this length
 *   tokenizer_type    (default 1)       — 0=Whitespace, 1=Word, 2=Character
 *   lowercase         (default true)    — lowercase before tokenizing
 *   min_word_freq     (default 2)       — vocabulary frequency floor
 *   max_vocab_size    (default 10000)   — vocabulary cap
 */
class TextTokenizerOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "TextTokenizer"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(
        const std::map<std::string, std::string>& params,
        std::string& error) override;

    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;

private:
    std::string text_col_;
    std::string label_col_;
    int max_length_ = 256;
    int tokenizer_type_ = 1;
    bool lowercase_ = true;
    int min_word_freq_ = 2;
    int max_vocab_size_ = 10000;
};

} // namespace cyxwiz
