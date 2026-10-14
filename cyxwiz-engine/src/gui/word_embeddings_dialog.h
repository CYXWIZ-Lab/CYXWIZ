#pragma once

#include "node_config_dialog.h"

#include "../core/word_embeddings_presentation.h"

#include <cyxwiz/word_embeddings.h>

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace gui {

// Word Embeddings (TOFIX152, mockup approved 2026-10-14): tabs Source,
// Vocabulary, Nearest words and Outputs. A pretrained file is read here on a
// task worker (its format, size and how the vocabulary lines up); training
// runs the step through the graph (Task View), and the dialog shows its
// progress and the result.
class WordEmbeddingsDialog : public NodeConfigDialog {
public:
    explicit WordEmbeddingsDialog(MLNode* node);

    void Apply() override;
    void Reset() override;
    ImVec2 GetDefaultSize() const override { return ImVec2(900, 720); }

protected:
    void RenderContent() override;
    float FooterHeight() const override;
    void RenderFooter(bool& should_close) override;

private:
    // A pretrained file read on a task worker; shared with it, so closing the
    // dialog never leaves the worker writing into freed memory.
    struct ReadJob {
        std::atomic<bool> finished{false};
        std::mutex mutex;
        std::string error;
        std::string summary;  // "GloVe text (no header) · 400,000 words · ..."
        std::shared_ptr<const cyxwiz::WordVectors> vectors;
        std::optional<cyxwiz::WordCoverageView> coverage;
        size_t vocab_size = 0;
        std::vector<std::string> missing_words;
    };

    std::string Get(const std::string& key) const;
    void Set(const std::string& key, const std::string& value);
    bool Training() const { return Get("source") == "train"; }
    std::string Resolve(const std::string& path) const;

    void RefreshInputs();
    void FindTokenizerVocabulary();
    std::string ReadKey() const;
    void StartRead();
    void PollRead();
    void StartTraining();
    void PollRun();
    void LoadRunVectors();

    void RenderSource();
    void RenderPretrained();
    void RenderTraining();
    void RenderCoverage(bool full_list);
    void RenderVocabulary();
    void RenderNearest();
    void RenderOutputs();
    void RenderRunCard();

    // Rows: a label column, then the field.
    void Label(const char* text);
    bool FileField(const char* id, const char* key, const char* title, bool save, const char* hint = nullptr);
    bool IntField(const char* id, const char* key, float width);
    bool FloatField(const char* id, const char* key, float width);
    bool Choice(const char* id, const char* key, const char* const* labels, const char* const* values, int count);

    std::map<std::string, std::string> params_;
    int tab_request_ = -1;  // a tab to select on the next frame

    // The step's input, when it has run (columns for the text column).
    bool inputs_resolved_ = false;
    std::string input_name_;
    std::vector<std::string> input_columns_;
    int64_t input_rows_ = -1;
    std::string tokenizer_vocab_;  // a Text Tokenizer's vocabulary in the graph

    std::shared_ptr<ReadJob> read_;
    std::string read_key_;  // what the last read was for
    std::string read_summary_;
    std::string read_error_;
    std::shared_ptr<const cyxwiz::WordVectors> read_vectors_;
    std::optional<cyxwiz::WordCoverageView> coverage_;
    size_t vocab_size_ = 0;
    std::vector<std::string> missing_words_;

    bool run_started_ = false;  // this dialog started the graph run
    uint64_t run_task_ = 0;
    std::string run_error_;
    std::string run_outcome_;
    double run_seconds_ = 0.0;
    std::string run_dataset_;
    std::string run_finished_at_;  // the run whose result is shown (time and error)
    std::shared_ptr<const cyxwiz::WordVectors> run_vectors_;

    char query_[128] = "";
    std::string query_word_;
    std::vector<std::pair<std::string, double>> nearest_;
    std::string nearest_note_;
};

}  // namespace gui
