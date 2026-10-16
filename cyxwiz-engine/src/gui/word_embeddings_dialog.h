#pragma once

#include "node_config_dialog.h"

#include "../core/word_embeddings_presentation.h"
#include "../core/word_vectors_training_record.h"

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

// Word Embeddings (TOFIX152, mockups approved 2026-10-14; GloVe the same day): tabs Source,
// Vocabulary, Nearest words and Outputs. A pretrained file is read here on a
// task worker (its format, size and how the vocabulary lines up); training
// runs the step through the graph (Task View), and the dialog shows its
// progress and the result. Without a run in this session it shows the saved
// training (the record beside the vectors file) and looks words up in it.
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
        bool fasttext = false;  // a fastText model (.bin)
    };
    // The saved vectors of a training, read on a task worker.
    struct SavedJob {
        std::atomic<bool> finished{false};
        std::shared_ptr<const cyxwiz::WordVectors> vectors;  // null when unreadable
    };

    std::string Get(const std::string& key) const;
    void Set(const std::string& key, const std::string& value);
    bool Training() const { return Get("source") != "pretrained"; }  // word2vec or GloVe
    bool GloVe() const { return Get("source") == "glove"; }
    bool FastText() const { return Get("source") == "fasttext"; }
    // The file the run saves its vectors to (the default when empty).
    std::string VectorsPath() const;
    // fastText: a word the vectors lack, built from its n-grams (the trained
    // model of this session or the pretrained .bin); false when there is none.
    bool UnseenWordVector(const std::string& word, std::vector<float>& vector, size_t& pieces, bool& trained) const;
    void RenderUnseenWordsChoice();
    std::string Resolve(const std::string& path) const;

    void RefreshInputs();
    void FindTokenizerVocabulary();
    std::string ReadKey() const;
    void StartRead();
    void PollRead();
    void StartTraining();
    void PollRun();
    void LoadRunVectors();
    void PollSaved();
    // The training settings of the dialog's values (as the step compares them).
    std::string TrainingSettings() const;

    void RenderSource();
    void RenderPretrained();
    void RenderTraining();
    void RenderWord2VecSettings();
    void RenderGloVeSettings();
    void RenderCoverage(bool full_list);
    void RenderVocabulary();
    void RenderNearest();
    void RenderOutputs();
    void RenderRunCard();
    void RenderSavedCard();

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
    bool read_fasttext_ = false;  // the pretrained file is a fastText model

    bool run_started_ = false;  // this dialog started the graph run
    uint64_t run_task_ = 0;
    std::string run_error_;
    std::string run_outcome_;
    double run_seconds_ = 0.0;
    std::string run_dataset_;
    std::string run_finished_at_;  // the run whose result is shown (time and error)
    std::shared_ptr<const cyxwiz::WordVectors> run_vectors_;

    // The saved training of the vectors file, checked every 2 s.
    std::string saved_path_;
    double saved_polled_at_ = -1.0e9;
    std::optional<cyxwiz::WordVectorsTrainingRecord> saved_;
    std::shared_ptr<SavedJob> saved_job_;
    std::shared_ptr<const cyxwiz::WordVectors> saved_vectors_;

    char query_[128] = "";
    std::string query_word_;
    std::vector<std::pair<std::string, double>> nearest_;
    std::string nearest_note_;
    std::string nearest_badge_;  // fastText: "not in the training words · from 26 pieces"
};

}  // namespace gui
