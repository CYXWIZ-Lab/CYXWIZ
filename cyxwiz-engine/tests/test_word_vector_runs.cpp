// Word-vector runs on the Training Dashboard (TOFIX152). The main scenario is
// the real GloVe run of 2026-10-14 on the sentiment set (GTX 1050 Ti): 16,321
// words, 8,024,345 pairs counted in 10.6 s, 25 epochs of about 10.2 s, 269.2 s
// in all; then a word2vec run, a cancelled run and the CSV exports.

#include "../src/core/word_vector_runs.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

const std::string kDot = " \xC2\xB7 ";

const std::vector<double> kCosts = {0.03414,  0.01822,  0.01603,  0.01473,  0.01346,  0.01227,  0.0112,
                                    0.01034,  0.009629, 0.009045, 0.008566, 0.008173, 0.007848, 0.007576,
                                    0.007349, 0.007151, 0.006983, 0.00683,  0.0067,   0.006584, 0.006478,
                                    0.006381, 0.006294, 0.006214, 0.006136};

cyxwiz::OperatorFigures Event(const std::string& stage, const std::string& message, float progress,
                              std::vector<std::pair<std::string, double>> figures,
                              std::vector<std::pair<std::string, std::string>> notes = {}, int node = 7) {
    cyxwiz::OperatorFigures event;
    event.node_id = node;
    event.node_name = "Word vectors (word2vec)";
    event.stage = stage;
    event.message = message;
    event.progress = progress;
    event.figures = std::move(figures);
    event.notes = std::move(notes);
    return event;
}

cyxwiz::OperatorFigures Start(const std::string& method, int epochs, int window, int node = 7) {
    return Event(method == "GloVe" ? "Counting" : "Training", "start", 0.05f,
                 {{"dims", 100}, {"window", window}, {"epochs", epochs}, {"min_count", 5}, {"elapsed", 2.0}},
                 {{"method", method},
                  {"saved", "D:/p/artifacts/word_embeddings/sentiment_word_vectors.vec"},
                  {"vocabulary_file", "D:/p/datasets/sentiment_analysis_vocab.txt"}},
                 node);
}

double EpochEnd(int epoch) { return 12.6 + 10.2 * epoch; }

}  // namespace

int main() {
    using cyxwiz::WordVectorRun;
    using Step = cyxwiz::WordVectorRunView::Step;
    cyxwiz::WordVectorRunBook book;

    // Counting.
    const auto* run = book.Record(Start("GloVe", 25, 10), 100.0);
    Check(run && run->number == 1 && run->GloVe() && run->Live() && run->epochs == 25, "a GloVe run starts");
    book.Record(Event("Counting", "Counting word pairs", 0.05f, {{"words", 16321}, {"elapsed", 2.1}}), 100.1);
    book.Progress(7, 0.112f, "Counting word pairs");
    auto view = cyxwiz::BuildWordVectorRunView(*book.Find(1), 105.0);
    Check(view.pill == "Preparing" && view.switcher_label == "Word vectors" + kDot + "GloVe" &&
              view.switcher_sub == "counting word pairs",
          "counting pill and switcher: " + view.switcher_sub);
    Check(view.title == "Word vectors (word2vec)" + kDot + "GloVe", "title " + view.title);
    Check(view.subtitle == "Counting word pairs" + kDot + "62% of the text", "counting share: " + view.subtitle);
    Check(view.steps.size() == 4 && view.steps[0].state == Step::State::Now &&
              view.steps[0].detail == "62% of the text" && view.steps[1].state == Step::State::Todo &&
              view.steps[2].name == "Lining up with the vocabulary" && view.steps[3].name == "Saved",
          "counting steps");
    Check(view.cards.size() == 4 && view.cards[0].value == "-" &&
              view.cards[0].line2 == "appears after epoch 1" && view.cards[1].label == "Time remaining" &&
              view.cards[1].line2 == "estimated once an epoch has run" && view.cards[2].value == "16,321" &&
              view.cards[2].line1 == "words seen 5+ times" && view.cards[2].line2 == "100 values each" &&
              view.cards[3].value == "counting..." &&
              view.cards[3].line2 == "window 10" + kDot + "closer pairs count more",
          "counting cards");
    Check(view.chart_title == "Cost per word pair" && view.figure_name == "cost", "chart title");

    // Counted, then six epochs.
    book.Record(Event("Counting", "Counted word pairs", 0.15f,
                      {{"words", 16321}, {"pairs", 8024345}, {"elapsed", 12.6}}),
                110.6);
    for (int e = 1; e <= 6; ++e) {
        book.Record(Event("Training", "Training GloVe", 0.15f + 0.75f * e / 25.0f,
                          {{"epoch", e}, {"cost", kCosts[e - 1]}, {"elapsed", EpochEnd(e)}}),
                    98.0 + EpochEnd(e));
    }
    view = cyxwiz::BuildWordVectorRunView(*book.Find(1), 98.0 + EpochEnd(6) + 3.0);
    Check(view.pill == "Training" && view.switcher_sub == "epoch 7 of 25" &&
              view.subtitle == "Epoch 7 / 25" + kDot + "18 remaining",
          "training status: " + view.subtitle);
    Check(view.steps[0].state == Step::State::Done && view.steps[0].detail == "8,024,345 pairs" + kDot + "10.6 s" &&
              view.steps[1].state == Step::State::Now && view.steps[1].detail == "epoch 7 of 25",
          "training steps: " + view.steps[0].detail);
    Check(view.cards[0].value == "0.0123" && view.cards[0].trend == -1 &&
              view.cards[0].line1 == "0.0219 since epoch 1" && view.cards[0].line2 == "best 0.0123 at epoch 6",
          "cost card: " + view.cards[0].value + " / " + view.cards[0].line1);
    // 10.2 s per epoch, 19 epochs left, 3 s into epoch 7.
    Check(view.cards[1].label == "Time remaining" && view.cards[1].value == "3 min 11 s" &&
              view.cards[1].line1 == "10.2 s per epoch" && view.cards[1].line2 == "elapsed 1 min 15 s",
          "time card: " + view.cards[1].value + " / " + view.cards[1].line2);
    Check(view.cards[3].value == "8,024,345" && view.cards[3].line1 == "counted in 10.6 s" &&
              view.cards[3].line2 == "window 10",
          "pairs card");
    Check(std::abs(view.progress - (0.15f + 0.75f * 6 / 25.0f)) < 1e-6f, "progress is the step's");
    Check(cyxwiz::WordVectorRunLine(*book.Find(1), 98.0 + EpochEnd(6) + 3.0) ==
              "Counted 8,024,345 word pairs" + kDot + "about 3 min 11 s left",
          "the dialog's line: " + cyxwiz::WordVectorRunLine(*book.Find(1), 98.0 + EpochEnd(6) + 3.0));

    // The rest of the epochs, the line-up and the end.
    for (int e = 7; e <= 25; ++e) {
        book.Record(Event("Training", "Training GloVe", 0.15f + 0.75f * e / 25.0f,
                          {{"epoch", e}, {"cost", kCosts[e - 1]}, {"elapsed", EpochEnd(e)}}),
                    98.0 + EpochEnd(e));
    }
    view = cyxwiz::BuildWordVectorRunView(*book.Find(1), 98.0 + EpochEnd(25) + 0.5);
    Check(view.switcher_sub == "lining up" && view.steps[1].state == Step::State::Done &&
              view.steps[2].state == Step::State::Now,
          "after the last epoch");
    book.Record(Event("Aligning", "Lined up", 0.92f,
                      {{"tokens", 5000}, {"found", 4919}, {"lowercase", 0}, {"missing", 79}, {"elapsed", 268.4}},
                      {{"special", "[PAD] zeros" + kDot + "[UNK] mean"}}),
                366.4);
    view = cyxwiz::BuildWordVectorRunView(*book.Find(1), 366.6);
    Check(view.switcher_sub == "lining up" && view.steps[1].state == Step::State::Done &&
              view.steps[1].detail == "25 epochs" && view.steps[2].state == Step::State::Done &&
              view.steps[2].detail == "4,919 found" + kDot + "79 missing",
          "lined up");
    book.Record(Event("Done", "Saved the vectors", 0.94f, {{"elapsed", 269.2}}), 367.2);
    const WordVectorRun* glove = book.Find(1);
    Check(glove->state == WordVectorRun::State::Completed && glove->figures.size() == 25, "completed");
    book.SetUsedBy(1, "Embedding (5,000 x 100)");
    view = cyxwiz::BuildWordVectorRunView(*book.Find(1), 1000.0);
    Check(cyxwiz::WordVectorRunLine(*book.Find(1), 1000.0) == "Counted 8,024,345 word pairs", "finished line");
    Check(view.pill == "Completed" && view.switcher_sub == "trained in 269.2 s" &&
              view.subtitle == "Executed epochs 25 / 25" + kDot + "269.2 s" && view.progress == 1.0f,
          "completed status: " + view.subtitle);
    Check(view.steps[3].state == Step::State::Done && view.steps[3].detail == "sentiment_word_vectors.vec",
          "saved step");
    Check(view.cards[0].value == "0.00614" && view.cards[0].line1 == "0.028 since epoch 1" &&
              view.cards[0].line2 == "best 0.00614 at epoch 25",
          "final cost: " + view.cards[0].line1);
    Check(view.cards[1].label == "Total time" && view.cards[1].value == "269.2 s" &&
              view.cards[1].line1 == "25 / 25 epochs run" && view.cards[1].line2 == "10.2 s per epoch",
          "total time");
    Check(view.result.size() == 4 && view.result[0].second == "D:/p/artifacts/word_embeddings/sentiment_word_vectors.vec" &&
              view.result[1].second == "sentiment_analysis_vocab.txt" + kDot +
                                           "5,000 tokens: 4,919 found, 0 in lowercase, 79 missing" &&
              view.result[2].second == "[PAD] zeros" + kDot + "[UNK] mean" &&
              view.result[3].first == "Used by",
          "result card: " + view.result[1].second);

    // A word2vec run of another node: its loss and examples.
    book.Record(Start("word2vec skip-gram", 2, 5, 9), 2000.0);
    book.Record(Event("Training", "Training word2vec", 0.05f, {{"words", 16321}, {"elapsed", 2.5}}, {}, 9), 2000.5);
    view = cyxwiz::BuildWordVectorRunView(*book.Find(2), 2001.0);
    Check(view.pill == "Training" && view.switcher_sub == "epoch 1 of 2" && view.steps.size() == 3 &&
              view.steps[0].name == "Training" && view.cards[0].label == "Loss per example" &&
              view.cards[3].label == "Examples" && view.cards[3].line2 == "window 5" + kDot + "counted after epoch 1",
          "word2vec before its first epoch");
    book.Record(Event("Training", "e1", 0.475f, {{"epoch", 1}, {"loss", 2.684}, {"examples", 24612345}, {"elapsed", 105.0}},
                      {}, 9),
                2103.0);
    book.Record(Event("Training", "e2", 0.9f, {{"epoch", 2}, {"loss", 2.507}, {"examples", 24598765}, {"elapsed", 204.0}},
                      {}, 9),
                2202.0);
    book.Record(Event("Done", "Saved", 0.94f, {{"elapsed", 206.5}}, {}, 9), 2204.5);
    const auto word2vec = cyxwiz::BuildWordVectorRunRow(*book.Find(2));
    Check(word2vec.method == "word2vec skip-gram" && word2vec.settings == "100 dims" + kDot + "window 5" + kDot + "2 epochs" &&
              word2vec.per_epoch == "24.6 M examples" && word2vec.figures == "loss 2.68 to 2.51" &&
              word2vec.time == "206.5 s",
          "word2vec row: " + word2vec.per_epoch + " / " + word2vec.figures);
    const auto glove_row = cyxwiz::BuildWordVectorRunRow(*book.Find(1));
    Check(glove_row.run == "1" && glove_row.words == "16,321" && glove_row.per_epoch == "8,024,345 pairs" &&
              glove_row.figures == "cost 0.0341 to 0.00614" && glove_row.time == "269.2 s" &&
              glove_row.saved == "sentiment_word_vectors.vec",
          "GloVe row: " + glove_row.figures);

    // A run cancelled while counting; a new start ends a run that never ended.
    book.Record(Start("GloVe", 25, 10), 3000.0);
    book.EndLive(7, true, "");
    view = cyxwiz::BuildWordVectorRunView(*book.Find(3), 3005.0);
    Check(view.pill == "Cancelled" && view.steps[0].state == Step::State::Now && view.steps[0].detail == "stopped" &&
              view.steps[1].state == Step::State::Todo && cyxwiz::BuildWordVectorRunRow(*book.Find(3)).time == "cancelled",
          "cancelled while counting");
    book.Record(Start("GloVe", 25, 10), 4000.0);
    book.Record(Start("GloVe", 25, 10), 4001.0);
    Check(book.Find(4)->state == WordVectorRun::State::Cancelled && book.Find(5)->Live(), "a fresh start ends the old");
    book.EndAllLive(false, "out of device memory");
    Check(book.Find(5)->state == WordVectorRun::State::Failed && book.Find(5)->error == "out of device memory",
          "failed with its error");
    Check(book.Record(Event("Training", "late", 0.5f, {{"epoch", 1}, {"cost", 0.1}}), 4002.0) == nullptr,
          "a figure without a live run is dropped");

    // Exports.
    const std::string runs_csv = cyxwiz::WordVectorRunsCsv(book.Runs());
    Check(runs_csv.rfind("run,node,method,dims,window,epochs,min_count,words,pairs,last_epoch_examples,first,last,"
                         "seconds,state,saved\n1,Word vectors (word2vec),GloVe,100,10,25,5,16321,8024345,,0.03414,"
                         "0.006136,269.2,completed,D:/p/artifacts/word_embeddings/sentiment_word_vectors.vec\n",
                         0) == 0,
          "runs CSV:\n" + runs_csv);
    const std::string curve = cyxwiz::WordVectorCurveCsv(*book.Find(1));
    Check(curve.rfind("epoch,cost,seconds\n1,0.03414,20.8\n2,0.01822,31.0\n", 0) == 0, "curve CSV:\n" + curve);

    Check(cyxwiz::WordVectorDuration(42.24) == "42.2 s" && cyxwiz::WordVectorDuration(184.0) == "3 min 04 s" &&
              cyxwiz::WordVectorDuration(3725.0) == "1 h 02 min",
          "durations");
    Check(cyxwiz::WordVectorCount(8024345) == "8,024,345" && cyxwiz::WordVectorCount(24612345) == "24.6 M", "counts");
    Check(cyxwiz::WordVectorQuickWord({"[PAD]", "[UNK]", "i", "to", "because", "therapy"}) == "because" &&
              cyxwiz::WordVectorQuickWord({"[PAD]", "can't", "sleep"}) == "can't",
          "quick-check word");
    std::cout << "word vector runs tests passed\n";
    return 0;
}
