#pragma once

// Word Embeddings node (TOFIX152): word vectors from a pretrained file
// (GloVe / fastText / word2vec .txt .vec .bin) or trained with word2vec on the
// input's text column, on the selected device (cyxwiz/word_embeddings.h).
// - Pin 0, Vectors: a table word, v_0 .. v_{d-1}; in the token-id order of
//   `vocab_file` when one is given (lined up, with a coverage report), else in
//   the file's / the trained vocabulary's order. Saved as word2vec text to
//   `vectors_path` (the file an Embedding layer loads).
// - Pin 1, Rows (with `row_vectors`): the input plus emb_0 .. emb_{d-1}, the
//   mean vector of each row's words.
// Text is split as the Text Tokenizer's Word mode splits it (lowercase, runs of
// letters and digits).

#include "pipeline_operator.h"

#include <cyxwiz/word_embeddings.h>

#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

class WordEmbeddingsOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "WordEmbeddings"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(const std::map<std::string, std::string>& params, std::string& error) override;
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(const std::shared_ptr<arrow::Table>& input) override;
    std::vector<std::shared_ptr<arrow::Table>> ExtraOutputs() const override { return {rows_}; }
    std::string OutcomeSummary() const override { return outcome_; }
    bool CollectCacheDependencies(std::vector<PipelineOperatorCacheDependency>& dependencies,
                                  std::string& error) const override;
    void SetProgressCallback(PipelineOperatorProgressCallback callback) override {
        progress_callback_ = std::move(callback);
    }

    // The words of `text` as the Text Tokenizer's Word mode gives them.
    static std::vector<std::string> SplitWords(const std::string& text);
    // The mean loss per example of each epoch, read back from a training
    // outcome ("... loss per epoch 2.31, 2.05)"); empty for a pretrained one.
    static std::vector<double> EpochLosses(const std::string& outcome);
    // Where a node saves its vectors when no path is set, relative to the
    // project's artifacts: word_embeddings/<id>_<name>.vec.
    static std::string DefaultVectorsFile(int node_id, const std::string& node_name);

private:
    bool train_ = false;
    std::string vectors_file_;
    size_t max_words_ = 0;
    std::string vocab_file_;
    WordAlignmentOptions alignment_;
    std::string text_col_;
    Word2VecOptions word2vec_;
    std::string vectors_path_;
    bool row_vectors_ = false;

    std::shared_ptr<arrow::Table> rows_;
    std::string outcome_;
    PipelineOperatorProgressCallback progress_callback_;
};

}  // namespace cyxwiz
