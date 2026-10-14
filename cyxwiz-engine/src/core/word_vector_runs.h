#pragma once

#include "operator_figures.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

// Word-vector training runs (word2vec, GloVe) of the Word Embeddings node,
// as the Training Dashboard shows them (TOFIX152, mockup approved
// 2026-10-14). Fed by the operator's figures (OperatorFigures); kept apart
// from the model run so a word-vector run never touches its charts.
struct WordVectorRun {
    enum class State { Running, Completed, Failed, Cancelled };

    int number = 0;  // 1, 2, ... in this session
    int node_id = -1;
    std::string node_name;
    std::string method;  // "GloVe", "word2vec skip-gram", "word2vec CBOW"
    std::string saved;   // the .vec path ("" when not saved)
    std::string vocabulary_file;
    int dims = 0;
    int window = 0;
    int epochs = 0;
    int min_count = 0;

    State state = State::Running;
    std::string stage;  // the operator's: Counting, Training, Aligning, Done
    std::string message;
    float step_share = 0.0f;  // the step's own progress, 0..1
    std::string error;

    double started_at = 0.0;       // seconds (operator clock) when training began
    double elapsed = 0.0;          // seconds at the latest figure
    double clock_at = 0.0;         // the caller's clock when `elapsed` arrived
    double counted_at = -1.0;      // seconds when counting ended (GloVe)
    double training_from = -1.0;   // seconds when the epochs started
    long long words = -1;
    long long pairs = -1;          // distinct word pairs (GloVe)
    std::vector<double> figures;   // cost (GloVe) or loss (word2vec) per epoch
    std::vector<double> epoch_end; // seconds at the end of each epoch
    std::vector<long long> examples;  // per epoch (word2vec)
    long long tokens = -1;         // vocabulary file tokens, once lined up
    long long found = 0;
    long long lowercase = 0;
    long long missing = 0;
    std::string special;           // "[PAD] zeros · [UNK] mean"
    std::string used_by;           // "Embedding (5,000 x 100)", set by the GUI

    bool GloVe() const { return method == "GloVe"; }
    bool Live() const { return state == State::Running; }
};

// The session's runs, oldest first.
class WordVectorRunBook {
public:
    // A figure event of a Word Embeddings training step. An event with a
    // "method" note starts a new run; the others update that node's live run.
    // Returns the run it changed (nullptr when no run matches). `now` is the
    // caller's clock in seconds, so a view can run the time on between events.
    const WordVectorRun* Record(const OperatorFigures& event, double now);
    // The step's live progress (Task View's status), between figure events.
    void Progress(int node_id, float step_share, const std::string& message);
    void SetUsedBy(int number, const std::string& used_by);
    // The run's task ended before its last figure: failed or cancelled.
    void EndLive(int node_id, bool cancelled, const std::string& error);
    void EndAllLive(bool cancelled, const std::string& error);

    const std::vector<WordVectorRun>& Runs() const { return runs_; }
    const WordVectorRun* Find(int number) const;
    size_t Version() const { return version_; }

private:
    WordVectorRun* LiveRun(int node_id);
    std::vector<WordVectorRun> runs_;
    size_t version_ = 0;
};

// What the dashboard draws for one run.
struct WordVectorRunView {
    struct Step {
        std::string name;
        std::string detail;
        enum class State { Done, Now, Todo } state = State::Todo;
    };
    struct Card {
        std::string label;
        std::string value;
        std::string line1;
        int trend = 0;  // -1 the cost or loss fell (good), +1 it rose
        std::string line2;
    };
    std::string switcher_label;  // "Word vectors · GloVe"
    std::string switcher_sub;    // "epoch 7 of 25"
    std::string pill;            // Preparing, Training, Completed, Failed, Cancelled
    std::string title;           // "Word vectors (word2vec) · GloVe"
    std::string subtitle;        // "Epoch 7 / 25 · 18 remaining"
    float progress = 0.0f;       // the run, 0..1
    std::vector<Step> steps;
    std::vector<Card> cards;
    std::string chart_title;     // "Cost per word pair"
    std::string chart_empty;     // shown before the first epoch ends
    std::string figure_name;     // "cost" or "loss"
    std::vector<std::pair<std::string, std::string>> result;  // label, value (finished runs)
    std::string error;
};

// `now`: the caller's clock (as given to Record), for the time remaining.
WordVectorRunView BuildWordVectorRunView(const WordVectorRun& run, double now);

// One row of the "Word vector runs" table and its CSV.
struct WordVectorRunRow {
    std::string run;
    std::string method;
    std::string settings;   // "100 dims · window 10 · 25 epochs"
    std::string words;
    std::string per_epoch;  // "8,024,345 pairs", "24.6 M examples"
    std::string figures;    // "cost 0.0341 → 0.00614"
    std::string time;
    std::string saved;
};

// The node dialog's line under a live run: "Counted 8,024,345 word pairs ·
// about 3 min 11 s left" (empty when nothing is known yet).
std::string WordVectorRunLine(const WordVectorRun& run, double now);

WordVectorRunRow BuildWordVectorRunRow(const WordVectorRun& run);
std::string WordVectorRunsCsv(const std::vector<WordVectorRun>& runs);
// One run's curve: epoch, cost or loss, seconds at the epoch's end.
std::string WordVectorCurveCsv(const WordVectorRun& run);

// "3 min 04 s", "269.2 s", "1 h 02 min".
std::string WordVectorDuration(double seconds);
// "24.6 M", "8,024,345" below ten million.
std::string WordVectorCount(long long count);
// The word the quick check starts with: the vocabulary's most frequent word
// (first in the list) of at least 7 letters, skipping special tokens.
std::string WordVectorQuickWord(const std::vector<std::string>& words);

}  // namespace cyxwiz
