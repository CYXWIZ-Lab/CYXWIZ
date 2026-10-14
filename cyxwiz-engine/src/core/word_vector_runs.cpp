#include "word_vector_runs.h"

#include "word_embeddings_presentation.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <sstream>

namespace cyxwiz {

namespace {

const std::string kDot = " \xC2\xB7 ";
// The UI font has no dash or arrow glyphs: a word and a hyphen.
const std::string kArrow = " to ";
const std::string kDash = "-";

std::string Figure(double value) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.3g", value);
    return text;
}

std::string Seconds(double seconds) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.1f s", seconds);
    return text;
}

std::string FileName(const std::string& path) {
    const std::string name = std::filesystem::path(path).filename().string();
    return name.empty() ? path : name;
}

std::string CsvField(const std::string& value) {
    if (value.find_first_of(",\"\n") == std::string::npos) return value;
    std::string quoted = "\"";
    for (const char c : value) {
        if (c == '"') quoted += '"';
        quoted += c;
    }
    return quoted + "\"";
}

int Done(const WordVectorRun& run) { return static_cast<int>(run.figures.size()); }

// Seconds per finished epoch, measured from the start of the epochs.
double EpochSeconds(const WordVectorRun& run) {
    if (run.epoch_end.empty() || run.training_from < 0.0) return -1.0;
    return (run.epoch_end.back() - run.training_from) / static_cast<double>(run.epoch_end.size());
}

bool PastTraining(const WordVectorRun& run) {
    return run.state == WordVectorRun::State::Completed || run.stage == "Aligning" || run.stage == "Done" ||
           (run.epochs > 0 && Done(run) >= run.epochs);
}

}  // namespace

WordVectorRun* WordVectorRunBook::LiveRun(int node_id) {
    for (auto it = runs_.rbegin(); it != runs_.rend(); ++it) {
        if (it->node_id == node_id) return it->Live() ? &*it : nullptr;
    }
    return nullptr;
}

const WordVectorRun* WordVectorRunBook::Record(const OperatorFigures& event, double now) {
    WordVectorRun* run = nullptr;
    const std::string method = event.Note("method");
    if (!method.empty()) {
        // A new run; one of the same node still live never reported its end.
        if (WordVectorRun* stale = LiveRun(event.node_id)) stale->state = WordVectorRun::State::Cancelled;
        WordVectorRun fresh;
        fresh.number = static_cast<int>(runs_.size()) + 1;
        fresh.node_id = event.node_id;
        fresh.node_name = event.node_name;
        fresh.method = method;
        fresh.saved = event.Note("saved");
        fresh.vocabulary_file = event.Note("vocabulary_file");
        fresh.dims = static_cast<int>(event.Figure("dims", 0));
        fresh.window = static_cast<int>(event.Figure("window", 0));
        fresh.epochs = static_cast<int>(event.Figure("epochs", 0));
        fresh.min_count = static_cast<int>(event.Figure("min_count", 0));
        fresh.started_at = event.Figure("elapsed", 0.0);
        // word2vec trains at once; GloVe counts first.
        if (!fresh.GloVe()) fresh.training_from = fresh.started_at;
        runs_.push_back(std::move(fresh));
        run = &runs_.back();
    } else {
        run = LiveRun(event.node_id);
        if (!run) return nullptr;
    }
    if (event.Has("elapsed")) {
        run->elapsed = event.Figure("elapsed");
        run->clock_at = now;
    }
    if (!event.stage.empty()) run->stage = event.stage;
    if (!event.message.empty()) run->message = event.message;
    run->step_share = event.progress;
    if (event.Has("words")) run->words = static_cast<long long>(event.Figure("words"));
    if (event.Has("pairs")) {
        run->pairs = static_cast<long long>(event.Figure("pairs"));
        run->counted_at = run->elapsed;
        run->training_from = run->elapsed;
    }
    if (event.Has("epoch")) {
        const size_t epoch = static_cast<size_t>(std::max(1.0, event.Figure("epoch")));
        const double value = event.Has("cost") ? event.Figure("cost") : event.Figure("loss");
        run->figures.resize(epoch - 1, value);
        run->epoch_end.resize(epoch - 1, run->elapsed);
        run->figures.push_back(value);
        run->epoch_end.push_back(run->elapsed);
        if (event.Has("examples")) {
            run->examples.resize(epoch - 1, 0);
            run->examples.push_back(static_cast<long long>(event.Figure("examples")));
        }
    }
    if (event.Has("tokens")) {
        run->tokens = static_cast<long long>(event.Figure("tokens"));
        run->found = static_cast<long long>(event.Figure("found", 0));
        run->lowercase = static_cast<long long>(event.Figure("lowercase", 0));
        run->missing = static_cast<long long>(event.Figure("missing", 0));
        run->special = event.Note("special");
    }
    if (event.stage == "Done") run->state = WordVectorRun::State::Completed;
    ++version_;
    return run;
}

void WordVectorRunBook::Progress(int node_id, float step_share, const std::string& message) {
    WordVectorRun* run = LiveRun(node_id);
    if (!run || (run->step_share == step_share && (message.empty() || run->message == message))) return;
    run->step_share = std::max(run->step_share, step_share);
    if (!message.empty()) run->message = message;
    ++version_;
}

void WordVectorRunBook::SetUsedBy(int number, const std::string& used_by) {
    for (auto& run : runs_) {
        if (run.number == number && run.used_by != used_by) {
            run.used_by = used_by;
            ++version_;
        }
    }
}

void WordVectorRunBook::EndLive(int node_id, bool cancelled, const std::string& error) {
    WordVectorRun* run = LiveRun(node_id);
    if (!run) return;
    run->state = cancelled ? WordVectorRun::State::Cancelled : WordVectorRun::State::Failed;
    run->error = error;
    ++version_;
}

void WordVectorRunBook::EndAllLive(bool cancelled, const std::string& error) {
    for (auto& run : runs_) {
        if (!run.Live()) continue;
        run.state = cancelled ? WordVectorRun::State::Cancelled : WordVectorRun::State::Failed;
        run.error = error;
        ++version_;
    }
}

const WordVectorRun* WordVectorRunBook::Find(int number) const {
    for (const auto& run : runs_) {
        if (run.number == number) return &run;
    }
    return nullptr;
}

WordVectorRunView BuildWordVectorRunView(const WordVectorRun& run, double now) {
    using State = WordVectorRun::State;
    using Step = WordVectorRunView::Step;
    WordVectorRunView view;
    const bool glove = run.GloVe();
    const bool counting = run.Live() && glove && run.pairs < 0;
    const int done = Done(run);
    const int current = std::clamp(done + 1, 1, std::max(1, run.epochs));
    const double live_elapsed = run.elapsed + (run.Live() ? std::max(0.0, now - run.clock_at) : 0.0);
    const double per_epoch = EpochSeconds(run);
    const std::string epochs_run = std::to_string(done) + " / " + std::to_string(run.epochs);

    view.figure_name = glove ? "cost" : "loss";
    view.switcher_label = "Word vectors \xC2\xB7 " + run.method;
    view.title = (run.node_name.empty() ? std::string("Word Embeddings") : run.node_name) + kDot + run.method;
    view.chart_title = glove ? "Cost per word pair" : "Loss per example";
    view.chart_empty = "The curve starts when the first epoch ends.";
    view.error = run.error;

    // Status: the pill, the line under the title and the run's progress.
    const float text_share = std::clamp((run.step_share - 0.05f) / 0.10f, 0.0f, 1.0f);
    switch (run.state) {
    case State::Running:
        if (counting) {
            view.pill = "Preparing";
            view.switcher_sub = "counting word pairs";
            view.subtitle = "Counting word pairs" + kDot + std::to_string(static_cast<int>(text_share * 100.0f + 0.5f)) +
                            "% of the text";
        } else if (PastTraining(run)) {
            view.pill = "Training";
            view.switcher_sub = "lining up";
            view.subtitle = "Lining up with the vocabulary";
        } else {
            view.pill = "Training";
            view.switcher_sub = "epoch " + std::to_string(current) + " of " + std::to_string(run.epochs);
            view.subtitle = "Epoch " + std::to_string(current) + " / " + std::to_string(run.epochs) + kDot +
                            std::to_string(std::max(0, run.epochs - current)) + " remaining";
        }
        view.progress = std::clamp(run.step_share, 0.0f, 1.0f);
        break;
    case State::Completed:
        view.pill = "Completed";
        view.switcher_sub = "trained in " + Seconds(run.elapsed);
        view.subtitle = "Executed epochs " + epochs_run + kDot + Seconds(run.elapsed);
        view.progress = 1.0f;
        break;
    case State::Failed:
    case State::Cancelled:
        view.pill = run.state == State::Failed ? "Failed" : "Cancelled";
        view.switcher_sub = run.state == State::Failed ? "failed" : "cancelled";
        view.subtitle = "Stopped after epochs " + epochs_run;
        view.progress = std::clamp(run.step_share, 0.0f, 1.0f);
        break;
    }

    // Steps.
    const bool stopped = run.state == State::Failed || run.state == State::Cancelled;
    const auto add_step = [&](const std::string& name, bool is_done, bool is_now, const std::string& done_detail,
                              const std::string& now_detail) {
        Step step;
        step.name = name;
        if (is_done) {
            step.state = Step::State::Done;
            step.detail = done_detail;
        } else if (is_now) {
            step.state = Step::State::Now;
            step.detail = stopped ? std::string("stopped") : now_detail;
        }
        view.steps.push_back(std::move(step));
    };
    if (glove) {
        add_step("Counting word pairs", run.pairs >= 0, counting || (stopped && run.pairs < 0),
                 GroupedCount(static_cast<size_t>(std::max(0LL, run.pairs))) + " pairs" + kDot +
                     Seconds(run.counted_at - run.started_at),
                 std::to_string(static_cast<int>(text_share * 100.0f + 0.5f)) + "% of the text");
    }
    const bool training_done = PastTraining(run);
    add_step("Training", training_done, !training_done && !counting && !(stopped && glove && run.pairs < 0),
             std::to_string(run.epochs) + " epochs",
             "epoch " + std::to_string(current) + " of " + std::to_string(run.epochs));
    if (!run.vocabulary_file.empty()) {
        add_step("Lining up with the vocabulary", run.tokens >= 0, training_done && run.tokens < 0,
                 GroupedCount(static_cast<size_t>(run.found)) + " found" + kDot +
                     GroupedCount(static_cast<size_t>(run.missing)) + " missing",
                 "matching tokens");
    }
    add_step(run.saved.empty() ? "Done" : "Saved", run.state == State::Completed, false,
             run.saved.empty() ? std::string("not saved to a file") : FileName(run.saved), "");

    // Cards: the figure, the time, the vocabulary, the pairs or examples.
    WordVectorRunView::Card figure;
    figure.label = glove ? "Cost per pair" : "Loss per example";
    if (run.figures.empty()) {
        figure.value = kDash;
        figure.line2 = "appears after epoch 1";
    } else {
        figure.value = Figure(run.figures.back());
        if (run.figures.size() >= 2) {
            const double change = run.figures.back() - run.figures.front();
            figure.trend = change < 0.0 ? -1 : (change > 0.0 ? 1 : 0);
            figure.line1 = change == 0.0 ? "no change since epoch 1" : Figure(std::abs(change)) + " since epoch 1";
        } else {
            figure.line1 = "first epoch";
        }
        const auto best = std::min_element(run.figures.begin(), run.figures.end());
        figure.line2 = "best " + Figure(*best) + " at epoch " + std::to_string(best - run.figures.begin() + 1);
    }
    view.cards.push_back(std::move(figure));

    WordVectorRunView::Card time;
    char per_epoch_text[48] = "";
    if (per_epoch > 0.0) std::snprintf(per_epoch_text, sizeof(per_epoch_text), "%.1f s per epoch", per_epoch);
    if (run.state == State::Completed) {
        time.label = "Total time";
        time.value = Seconds(run.elapsed);
        time.line1 = epochs_run + " epochs run";
        time.line2 = per_epoch_text;
    } else if (stopped) {
        time.label = "Time";
        time.value = WordVectorDuration(run.elapsed);
        time.line1 = epochs_run + " epochs run";
        time.line2 = per_epoch_text;
    } else if (per_epoch > 0.0 && !PastTraining(run)) {
        time.label = "Time remaining";
        const double into_epoch = live_elapsed - run.epoch_end.back();
        time.value = WordVectorDuration(std::max(0.0, per_epoch * (run.epochs - done) - into_epoch));
        time.line1 = per_epoch_text;
        time.line2 = "elapsed " + WordVectorDuration(live_elapsed - run.started_at);
    } else {
        time.label = PastTraining(run) ? "Time" : "Time remaining";
        time.value = PastTraining(run) ? WordVectorDuration(live_elapsed - run.started_at) : kDash;
        time.line2 = PastTraining(run) ? per_epoch_text : "estimated once an epoch has run";
    }
    view.cards.push_back(std::move(time));

    WordVectorRunView::Card vocabulary;
    vocabulary.label = "Vocabulary";
    vocabulary.value = run.words >= 0 ? GroupedCount(static_cast<size_t>(run.words)) : kDash;
    vocabulary.line1 = "words seen " + std::to_string(run.min_count) + "+ times";
    vocabulary.line2 = std::to_string(run.dims) + " values each";
    view.cards.push_back(std::move(vocabulary));

    WordVectorRunView::Card amount;
    const std::string window = "window " + std::to_string(run.window);
    if (glove) {
        amount.label = "Word pairs";
        if (run.pairs >= 0) {
            amount.value = GroupedCount(static_cast<size_t>(run.pairs));
            amount.line1 = "counted in " + Seconds(run.counted_at - run.started_at);
            amount.line2 = window;
        } else {
            amount.value = run.Live() ? "counting..." : kDash;
            amount.line2 = window + kDot + "closer pairs count more";
        }
    } else {
        amount.label = "Examples";
        if (!run.examples.empty()) {
            amount.value = WordVectorCount(run.examples.back());
            amount.line1 = "in the last epoch";
            amount.line2 = window;
        } else {
            amount.value = kDash;
            amount.line2 = window + kDot + "counted after epoch 1";
        }
    }
    view.cards.push_back(std::move(amount));

    // Result, once trained.
    if (run.state == State::Completed) {
        view.result.emplace_back("Saved", run.saved.empty() ? std::string("not saved to a file") : run.saved);
        if (run.tokens >= 0) {
            view.result.emplace_back(
                "Lined up", FileName(run.vocabulary_file) + kDot + GroupedCount(static_cast<size_t>(run.tokens)) +
                                " tokens: " + GroupedCount(static_cast<size_t>(run.found)) + " found, " +
                                GroupedCount(static_cast<size_t>(run.lowercase)) + " in lowercase, " +
                                GroupedCount(static_cast<size_t>(run.missing)) + " missing");
        }
        if (!run.special.empty()) view.result.emplace_back("Special", run.special);
        if (!run.used_by.empty()) view.result.emplace_back("Used by", run.used_by);
    }
    return view;
}

std::string WordVectorRunLine(const WordVectorRun& run, double now) {
    std::string line;
    if (run.GloVe() && run.pairs >= 0) line = "Counted " + GroupedCount(static_cast<size_t>(run.pairs)) + " word pairs";
    if (run.Live()) {
        const WordVectorRunView view = BuildWordVectorRunView(run, now);
        const auto& time = view.cards[1];
        if (time.label == "Time remaining" && time.value != kDash) {
            line += (line.empty() ? "about " : kDot + "about ") + time.value + " left";
        }
    }
    return line;
}

WordVectorRunRow BuildWordVectorRunRow(const WordVectorRun& run) {
    WordVectorRunRow row;
    row.run = std::to_string(run.number);
    row.method = run.method;
    row.settings = std::to_string(run.dims) + " dims" + kDot + "window " + std::to_string(run.window) + kDot +
                   std::to_string(run.epochs) + " epochs";
    row.words = run.words >= 0 ? GroupedCount(static_cast<size_t>(run.words)) : kDash;
    if (run.GloVe()) {
        row.per_epoch = run.pairs >= 0 ? GroupedCount(static_cast<size_t>(run.pairs)) + " pairs" : kDash;
    } else {
        row.per_epoch = run.examples.empty() ? kDash : WordVectorCount(run.examples.back()) + " examples";
    }
    const std::string name = run.GloVe() ? "cost " : "loss ";
    if (run.figures.empty()) {
        row.figures = kDash;
    } else if (run.figures.size() == 1) {
        row.figures = name + Figure(run.figures.front());
    } else {
        row.figures = name + Figure(run.figures.front()) + kArrow + Figure(run.figures.back());
    }
    switch (run.state) {
    case WordVectorRun::State::Running: row.time = "running"; break;
    case WordVectorRun::State::Completed: row.time = Seconds(run.elapsed); break;
    case WordVectorRun::State::Failed: row.time = "failed"; break;
    case WordVectorRun::State::Cancelled: row.time = "cancelled"; break;
    }
    row.saved = run.saved.empty() ? kDash : FileName(run.saved);
    return row;
}

std::string WordVectorRunsCsv(const std::vector<WordVectorRun>& runs) {
    std::ostringstream csv;
    csv << "run,node,method,dims,window,epochs,min_count,words,pairs,last_epoch_examples,first,last,seconds,state,"
           "saved\n";
    for (const auto& run : runs) {
        const char* state = run.state == WordVectorRun::State::Completed   ? "completed"
                            : run.state == WordVectorRun::State::Running   ? "running"
                            : run.state == WordVectorRun::State::Cancelled ? "cancelled"
                                                                           : "failed";
        char first[32] = "", last[32] = "", seconds[32];
        if (!run.figures.empty()) {
            std::snprintf(first, sizeof(first), "%.6g", run.figures.front());
            std::snprintf(last, sizeof(last), "%.6g", run.figures.back());
        }
        std::snprintf(seconds, sizeof(seconds), "%.1f", run.elapsed);
        csv << run.number << ',' << CsvField(run.node_name) << ',' << CsvField(run.method) << ',' << run.dims << ','
            << run.window << ',' << run.epochs << ',' << run.min_count << ','
            << (run.words >= 0 ? std::to_string(run.words) : "") << ','
            << (run.pairs >= 0 ? std::to_string(run.pairs) : "") << ','
            << (run.examples.empty() ? "" : std::to_string(run.examples.back())) << ',' << first << ',' << last << ','
            << seconds << ',' << state << ',' << CsvField(run.saved) << '\n';
    }
    return csv.str();
}

std::string WordVectorCurveCsv(const WordVectorRun& run) {
    std::ostringstream csv;
    csv << "epoch," << (run.GloVe() ? "cost" : "loss") << ",seconds\n";
    for (size_t i = 0; i < run.figures.size(); ++i) {
        char line[96];
        std::snprintf(line, sizeof(line), "%zu,%.6g,%.1f\n", i + 1, run.figures[i],
                      i < run.epoch_end.size() ? run.epoch_end[i] - run.started_at : 0.0);
        csv << line;
    }
    return csv.str();
}

std::string WordVectorDuration(double seconds) {
    seconds = std::max(0.0, seconds);
    char text[48];
    if (seconds < 60.0) {
        std::snprintf(text, sizeof(text), "%.1f s", seconds);
    } else if (seconds < 3600.0) {
        const long long whole = std::llround(seconds);
        std::snprintf(text, sizeof(text), "%lld min %02lld s", whole / 60, whole % 60);
    } else {
        const long long minutes = std::llround(seconds / 60.0);
        std::snprintf(text, sizeof(text), "%lld h %02lld min", minutes / 60, minutes % 60);
    }
    return text;
}

std::string WordVectorCount(long long count) {
    if (count < 10000000) return GroupedCount(static_cast<size_t>(std::max(0LL, count)));
    char text[32];
    std::snprintf(text, sizeof(text), "%.1f M", static_cast<double>(count) / 1e6);
    return text;
}

std::string WordVectorQuickWord(const std::vector<std::string>& words) {
    for (const auto& word : words) {
        if (word.size() < 7) continue;
        if (std::all_of(word.begin(), word.end(), [](unsigned char c) { return std::isalpha(c) != 0; })) return word;
    }
    for (const auto& word : words) {
        if (!word.empty() && word.front() != '[' && word.front() != '<') return word;
    }
    return {};
}

}  // namespace cyxwiz
