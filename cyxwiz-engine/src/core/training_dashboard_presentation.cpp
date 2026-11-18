#include "training_dashboard_presentation.h"

#include "rl_device_presentation.h"
#include "training_progress_estimate.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace cyxwiz::training_dashboard {

namespace {

constexpr double kLargeGap = 0.25;  // validation loss this far above training: overfitting shows

std::string Format(const char* format, double value) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), format, value);
    return buffer;
}

std::string EpochText(double epoch) {
    if (std::abs(epoch - std::round(epoch)) < 1e-6) return std::to_string(static_cast<int>(std::lround(epoch)));
    return Format("%.2f", epoch);
}

// The value of `s` at epoch `x`, if it has one there.
std::optional<double> At(const Series& s, double x) {
    for (size_t i = s.Size(); i-- > 0;)
        if (std::abs(s.x[i] - x) < 1e-6) return s.y[i];
    return std::nullopt;
}

size_t BestIndex(const Series& s, bool lower_is_better) {
    size_t best = 0;
    for (size_t i = 1; i < s.Size(); ++i)
        if (lower_is_better ? s.y[i] < s.y[best] : s.y[i] > s.y[best]) best = i;
    return best;
}

// The change of an epoch series since the epoch before.
void Change(const Series& s, bool lower_is_better, bool percent, Card& card) {
    const size_t n = s.Size();
    if (n < 2) {
        card.change = "the first epoch";
        card.change_tone = Tone::Faint;
        return;
    }
    const double delta = s.y[n - 1] - s.y[n - 2];
    const std::string since = " since epoch " + EpochText(s.x[n - 2]);
    if (std::abs(delta) < 1e-12) {
        card.change = "no change" + since;
        card.change_tone = Tone::Plain;
        return;
    }
    card.change = (percent ? Format("%.2f points", std::abs(delta)) : Format("%.4f", std::abs(delta))) + since;
    card.direction = delta < 0.0 ? -1 : 1;
    card.change_tone = (lower_is_better ? delta < 0.0 : delta > 0.0) ? Tone::Good : Tone::Bad;
}

std::string Best(const Series& s, bool lower_is_better, const char* format) {
    const size_t best = BestIndex(s, lower_is_better);
    return "best " + Format(format, s.y[best]) + " at epoch " + EpochText(s.x[best]);
}

// A training card (loss or accuracy): the finished epoch's value; while the
// next epoch runs, the latest batches as a second figure.
void TrainingCard(std::vector<Card>& cards, const RunFacts& facts, const Series& epochs,
                  const std::optional<Point>& latest, const char* id, const char* name, int series,
                  bool lower_is_better, bool percent) {
    const char* format = percent ? "%.2f%%" : "%.4f";
    Card card;
    card.id = id;
    card.series = series;
    if (!epochs.Empty()) {
        const double epoch = epochs.x[epochs.Size() - 1];
        const double value = epochs.y[epochs.Size() - 1];
        card.label = std::string(name) + " \xC2\xB7 epoch " + EpochText(epoch);
        card.value = Format(format, value);
        card.tooltip = std::string("Training ") + (percent ? "accuracy" : "loss") + " over epoch " + EpochText(epoch) +
                       ": " + Format(percent ? "%.4f%%" : "%.6f", value);
        Change(epochs, lower_is_better, percent, card);
        card.line2 = Best(epochs, lower_is_better, format);
        if (facts.training && latest && latest->x > epoch + 1e-9)
            card.line3 = "latest batches " + Format(format, latest->y);
    } else if (latest) {
        const std::string epoch = std::to_string(static_cast<int>(std::ceil(latest->x - 1e-9)));
        card.label = std::string(name) + " \xC2\xB7 " + (facts.training ? "epoch " + epoch + ", latest batches"
                                                                        : "stopped in epoch " + epoch);
        card.value = Format(format, latest->y);
        card.tooltip = "The mean of the latest batches; the card shows the whole epoch's value once an epoch ends.";
        card.change = facts.training ? "the first epoch is running" : "no epoch finished";
        card.change_tone = Tone::Faint;
    } else {
        return;
    }
    cards.push_back(std::move(card));
}

std::string Plural(int n, const char* one, const char* many) { return std::to_string(n) + " " + (n == 1 ? one : many); }

// Words for the gap between validation and training loss at one epoch.
std::string GapWords(double gap, Tone& tone) {
    if (gap >= 0.0) {
        tone = gap > kLargeGap ? Tone::Bad : Tone::Good;
        return Format("%.4f", gap) + " above the training loss" + (gap > kLargeGap ? ": a large gap" : ": a small gap");
    }
    tone = Tone::Good;
    return Format("%.4f", -gap) + " below the training loss";
}

double Slope(const Series& s, size_t window) {
    const size_t n = s.Size();
    if (n < 2) return 0.0;
    const size_t start = n > window ? n - window : 0;
    const double dx = s.x[n - 1] - s.x[start];
    return dx > 0.0 ? (s.y[n - 1] - s.y[start]) / dx : 0.0;
}

double Roughness(const Series& s, size_t window) {
    const size_t n = s.Size();
    if (n < 3) return 0.0;
    const size_t start = n > window ? n - window : 1;
    double total = 0.0;
    for (size_t i = start; i < n; ++i) total += std::abs(s.y[i] - s.y[i - 1]);
    return total / static_cast<double>(n - start);
}

// A loss with three significant digits (2.21, 0.222, 0.00123).
std::string Loss3(double value) { return Format(std::abs(value) >= 1.0 ? "%.2f" : "%.3g", value); }

// "fell every epoch (2.21 to 0.222)", "falling (...)", "flat", "rising".
std::string Trend(const Series& epochs, bool training, bool lower_is_better) {
    const size_t n = epochs.Size();
    if (n == 0) return {};
    const auto value = [&](double v) { return lower_is_better ? Loss3(v) : Format("%.2f%%", v); };
    if (n == 1) return "epoch " + EpochText(epochs.x[0]) + ": " + value(epochs.y[0]);
    bool every = true;
    for (size_t i = 1; i < n; ++i)
        every = every && (lower_is_better ? epochs.y[i] < epochs.y[i - 1] : epochs.y[i] > epochs.y[i - 1]);
    const std::string range = " (" + value(epochs.y[0]) + " to " + value(epochs.y[n - 1]) + ")";
    if (every) return std::string(lower_is_better ? (training ? "falling every epoch" : "fell every epoch")
                                                  : (training ? "rising every epoch" : "rose every epoch")) + range;
    const double slope = Slope(epochs, 5);
    const double threshold = lower_is_better ? 0.002 : 0.05;
    if (std::abs(slope) < threshold) return "flat lately" + range;
    const bool improving = lower_is_better ? slope < 0.0 : slope > 0.0;
    return std::string(improving ? "improving lately" : "getting worse lately") + range;
}

}  // namespace

std::string Thousands(double value) {
    const long long n = std::llround(std::max(0.0, value));
    std::string digits = std::to_string(n);
    for (int i = static_cast<int>(digits.size()) - 3; i > 0; i -= 3) digits.insert(static_cast<size_t>(i), ",");
    return digits;
}

std::string PerEpoch(double seconds) {
    if (seconds < 1.0) return Format("%.2fs", seconds);
    if (seconds < 10.0) return Format("%.1fs", seconds);
    return FormatTrainingDuration(seconds);
}

std::vector<Card> BuildCards(const RunFacts& facts) {
    std::vector<Card> cards;
    TrainingCard(cards, facts, facts.train_loss_epochs, facts.latest_loss_point, "##kpi_loss", "Loss", 0, true, false);
    TrainingCard(cards, facts, facts.train_accuracy_epochs, facts.latest_accuracy_point, "##kpi_acc", "Accuracy", 2,
                 false, true);

    if (!facts.val_loss.Empty()) {
        const Series& v = facts.val_loss;
        Card card;
        card.id = "##kpi_val_loss";
        card.label = "Validation loss";
        card.series = 1;
        const double epoch = v.x[v.Size() - 1];
        card.value = Format("%.4f", v.y[v.Size() - 1]);
        card.tooltip = "Validation loss after epoch " + EpochText(epoch) + ": " + Format("%.6f", v.y[v.Size() - 1]) +
                       "\n" + Best(v, true, "%.4f");
        Change(v, true, false, card);
        // The gap at the same epoch (the card compared with the last batch window).
        if (const auto train = At(facts.train_loss_epochs, epoch)) {
            card.line2 = GapWords(v.y[v.Size() - 1] - *train, card.line2_tone);
        } else {
            card.line2 = Best(v, true, "%.4f");
        }
        cards.push_back(std::move(card));
    }
    if (!facts.val_accuracy.Empty()) {
        const Series& v = facts.val_accuracy;
        Card card;
        card.id = "##kpi_val_acc";
        card.label = "Validation accuracy";
        card.series = 3;
        card.value = Format("%.2f%%", v.y[v.Size() - 1]);
        card.tooltip = "Validation accuracy after epoch " + EpochText(v.x[v.Size() - 1]) + ": " +
                       Format("%.4f%%", v.y[v.Size() - 1]);
        Change(v, false, true, card);
        card.line2 = Best(v, false, "%.2f%%");
        cards.push_back(std::move(card));
    }

    if (facts.training && (facts.total_batches > 0 || facts.avg_epoch_seconds > 0.0)) {
        Card card;
        card.id = "##kpi_time";
        card.label = "Time left";
        card.value = facts.remaining_seconds ? FormatTrainingDuration(*facts.remaining_seconds) : "Estimating...";
        card.tooltip = "The batches left in every epoch at the recent speed (the last 2 minutes), plus " +
                       (facts.epoch_end_seconds ? "about " + FormatTrainingDuration(*facts.epoch_end_seconds)
                                                : std::string("the time")) +
                       " at each epoch's end (validation, previews, checkpoint" +
                       (facts.epoch_end_seconds ? "" : "; measured after the first epoch") +
                       "). It follows the speed as it changes.";
        if (facts.batches_per_second > 0.0) card.tooltip += "\n" + Format("%.2f batches per second", facts.batches_per_second);
        const int epochs_left = std::max(0, facts.total_epochs - facts.current_epoch);
        const int batches_left = facts.total_batches > 0 ? std::max(0, facts.total_batches - facts.current_batch) : 0;
        if (facts.total_epochs > 0) {
            if (epochs_left > 0 && batches_left > 0)
                card.line2 = Plural(epochs_left, "epoch", "epochs") + " and " + Plural(batches_left, "batch", "batches") + " to go";
            else if (epochs_left > 0)
                card.line2 = Plural(epochs_left, "epoch", "epochs") + " to go";
            else
                card.line2 = Plural(batches_left, "batch", "batches") + " to go";
        }
        if (facts.samples_per_second > 0.0) {
            card.line3 = Thousands(facts.samples_per_second) + " samples/s";
        } else if (facts.batches_per_second > 0.0) {
            card.line3 = Format("%.2f batches/s", facts.batches_per_second);
        } else {
            card.line3 = "measuring the speed";
            card.line3_tone = Tone::Faint;
        }
        cards.push_back(std::move(card));
    } else if (!facts.training && facts.total_seconds > 0.0) {
        Card card;
        card.id = "##kpi_time";
        card.label = "Total time";
        card.value = FormatTrainingDuration(facts.total_seconds);
        if (facts.total_epochs > 0) {
            card.line2 = std::to_string(facts.last_executed_epoch) + " of " + Plural(facts.total_epochs, "epoch", "epochs");
            if (facts.avg_epoch_seconds > 0.0) card.line2 += ", " + PerEpoch(facts.avg_epoch_seconds) + " each";
        } else {
            card.line2 = "the run has finished";
        }
        card.line2_tone = Tone::Faint;
        cards.push_back(std::move(card));
    }
    return cards;
}

Insights BuildInsights(const RunFacts& facts) {
    Insights out;
    const Series& train = facts.train_loss_epochs;
    const Series& val = facts.val_loss;
    if (!train.Empty()) out.rows.push_back({"Training loss", Trend(train, facts.training, true), Tone::Plain});

    double gap = 0.0;
    bool has_gap = false;
    if (!val.Empty()) {
        std::string text = Trend(val, facts.training, true);
        if (const auto t = At(train, val.x[val.Size() - 1])) {
            gap = val.y[val.Size() - 1] - *t;
            has_gap = true;
            Tone tone = Tone::Plain;
            text += "; " + GapWords(gap, tone);
        }
        out.rows.push_back({"Validation loss", text, has_gap && gap > kLargeGap ? Tone::Bad : Tone::Plain});
        const size_t best = BestIndex(val, true);
        std::string best_text = "epoch " + EpochText(val.x[best]) + ": loss " + Format("%.4f", val.y[best]);
        if (const auto acc = At(facts.val_accuracy, val.x[best])) best_text += ", accuracy " + Format("%.2f%%", *acc);
        out.rows.push_back({"Best validation", best_text, Tone::Plain});
    }

    // Validation loss rising twice in a row after its best epoch.
    double rising_since = -1.0;
    if (val.Size() >= 3) {
        int streak = 0;
        for (size_t i = BestIndex(val, true) + 1; i < val.Size(); ++i) {
            streak = val.y[i] > val.y[i - 1] ? streak + 1 : 0;
            if (streak >= 2) {
                rising_since = val.x[i - 1];
                break;
            }
        }
    }
    if (rising_since >= 0.0)
        out.rows.push_back({"Rising", "validation loss has risen since epoch " + EpochText(rising_since), Tone::Bad});

    const bool has_validation = !val.Empty() || !facts.val_accuracy.Empty();
    const bool val_worse = Slope(val, 5) > 0.0 || Slope(facts.val_accuracy, 5) < 0.0;
    const bool val_improving_last = val.Size() >= 2 && val.y[val.Size() - 1] < val.y[val.Size() - 2];
    const bool train_improving_last =
        train.Size() >= 2 && train.y[train.Size() - 1] < train.y[train.Size() - 2];
    std::string short_word, sentence;
    if (has_validation && rising_since >= 0.0 && Roughness(val, 5) > 0.02) {
        short_word = "consider stopping early";
        sentence = "consider stopping early: validation loss keeps rising";
        out.suggestion_tone = Tone::Bad;
    } else if (has_validation && ((has_gap && gap > kLargeGap) || val_worse || rising_since >= 0.0)) {
        short_word = "look at validation";
        sentence = "look at the validation curve: it is not following training";
        out.suggestion_tone = Tone::Bad;
    } else if (facts.training) {
        short_word = "keep going";
        sentence = val_improving_last && train_improving_last ? "keep going: both are still improving" : "keep going";
    } else if (val_improving_last) {
        short_word = "train longer";
        sentence = "train longer: validation was still improving at the last epoch";
    } else {
        short_word = "keep this model";
        sentence = has_validation ? "keep this model: validation stopped improving" : "keep this model";
    }
    out.suggestion = "Suggested: " + short_word;
    if (!out.rows.empty()) out.rows.push_back({"Suggested", sentence, out.suggestion_tone});
    return out;
}

namespace {

constexpr const char* kDot = " \xC2\xB7 ";

RunState StateOf(const RunFacts& run, const HeaderFacts& h) {
    if (run.training && !h.waiting_for_gpu.empty()) return RunState::WaitingForGpu;
    if (run.training) return h.paused ? RunState::Paused : RunState::Training;
    if (h.preparation_failed) return RunState::PreparationFailed;
    if (h.preparing) return RunState::Preparing;
    if (run.total_seconds > 0.0) {
        if (h.terminal_status == "early_stopped") return RunState::EarlyStopped;
        if (h.terminal_status == "cancelled" || h.terminal_status == "stopped") return RunState::Stopped;
        if (h.terminal_status == "failed") return RunState::Failed;
        return RunState::Completed;
    }
    if (h.checkpoint_loaded) return RunState::ModelLoaded;
    return RunState::Idle;
}

const char* PillWord(RunState state) {
    switch (state) {
    case RunState::WaitingForGpu: return "Waiting for the GPU";
    case RunState::Training: return "Training";
    case RunState::Paused: return "Paused";
    case RunState::PreparationFailed: return "Preparation failed";
    case RunState::Preparing: return "Preparing";
    case RunState::EarlyStopped: return "Early stopped";
    case RunState::Stopped: return "Stopped";
    case RunState::Failed: return "Failed";
    case RunState::Completed: return "Completed";
    case RunState::ModelLoaded: return "Model loaded";
    case RunState::Idle: break;
    }
    return "Idle";
}

bool Running(RunState s) { return s == RunState::Training || s == RunState::Paused || s == RunState::WaitingForGpu; }
bool Ended(RunState s) {
    return s == RunState::EarlyStopped || s == RunState::Stopped || s == RunState::Failed || s == RunState::Completed;
}

// A stop code has no spaces ("completed_all_epochs"); a failure's reason is a sentence.
bool IsCode(const std::string& text) { return !text.empty() && text.find(' ') == std::string::npos; }

// "validation_loss_plateau_patience_3" -> 3; 0 when it is not that code.
int PlateauPatience(const std::string& reason) {
    const std::string prefix = "validation_loss_plateau_patience_";
    if (reason.rfind(prefix, 0) != 0) return 0;
    return std::atoi(reason.c_str() + prefix.size());
}

std::string Outcome(RunState state, const HeaderFacts& h, Tone& tone) {
    switch (state) {
    case RunState::Completed:
        tone = Tone::Good;
        return "Finished: every epoch ran.";
    case RunState::EarlyStopped: {
        tone = Tone::Bad;
        if (const int patience = PlateauPatience(h.terminal_reason))
            return "Stopped early: the validation loss did not improve for " + Plural(patience, "epoch.", "epochs.");
        if (h.terminal_reason == "plugin_requested_early_stop") return "Stopped early: a plugin asked the run to stop.";
        return "Stopped early.";
    }
    case RunState::Stopped:
        tone = Tone::Bad;
        return h.terminal_reason == "user_cancelled" ? "Stopped: you stopped the run before its last epoch."
                                                     : "Stopped before the last epoch.";
    case RunState::Failed:
        tone = Tone::Error;
        return h.terminal_reason.empty() ? "Failed." : "Failed: " + h.terminal_reason;
    case RunState::PreparationFailed:
        tone = Tone::Error;
        return h.preparation_error.empty() ? "Training preparation failed." : h.preparation_error;
    default:
        return {};
    }
}

std::string Place(const Route& r) {
    if (r.effective_backend == "arrayfire_cpu") return "the CPU (ArrayFire CPU)";
    const std::string word = rl_device::BackendWord(r.effective_backend);
    return r.device_name.empty() ? word : word + " (" + r.device_name + ")";
}

std::string Rows(const HeaderFacts& h) {
    std::vector<std::string> parts;
    if (h.train_rows > 0) parts.push_back(Thousands(static_cast<double>(h.train_rows)) + " training");
    if (h.validation_rows > 0) parts.push_back(Thousands(static_cast<double>(h.validation_rows)) + " validation");
    if (h.test_rows > 0) parts.push_back(Thousands(static_cast<double>(h.test_rows)) + " test");
    if (parts.empty()) return {};
    std::string text;
    for (size_t i = 0; i < parts.size(); ++i) text += (i == 0 ? "" : ", ") + parts[i];
    return text + " rows";
}

}  // namespace

Header BuildHeader(const RunFacts& run, const HeaderFacts& h) {
    Header out;
    out.state = StateOf(run, h);
    out.pill = PillWord(out.state);
    const bool running = Running(out.state);
    const bool ended = Ended(out.state);

    // The status line and the progress bar.
    switch (out.state) {
    case RunState::WaitingForGpu:
        out.status = "Starts when the other run ends";
        out.status_tooltip = h.waiting_for_gpu + "\nStop cancels the run while it waits.";
        break;
    case RunState::Training:
    case RunState::Paused: {
        const int epoch = std::max(1, run.current_epoch);
        out.status = "Epoch " + std::to_string(epoch);
        if (run.total_epochs > 0) out.status += " of " + std::to_string(run.total_epochs);
        if (run.total_batches > 0 && run.current_batch > 0)
            out.status += kDot + std::string("batch ") + std::to_string(run.current_batch) + " of " +
                          std::to_string(run.total_batches);
        break;
    }
    case RunState::Preparing: out.status = "Getting the data and model ready"; break;
    case RunState::PreparationFailed: out.status = "The data or the model could not be made ready"; break;
    case RunState::Idle: out.status = "No training run yet"; break;
    case RunState::ModelLoaded: break;
    case RunState::EarlyStopped:
    case RunState::Stopped:
    case RunState::Failed:
    case RunState::Completed: {
        const std::string time = FormatTrainingDuration(run.total_seconds);
        if (run.total_epochs <= 0) {
            out.status = "Ran for " + time;
        } else if (out.state == RunState::Completed && run.last_executed_epoch >= run.total_epochs) {
            out.status = "All " + Plural(run.total_epochs, "epoch", "epochs") + " in " + time;
        } else {
            out.status = std::to_string(run.last_executed_epoch) + " of " +
                         Plural(run.total_epochs, "epoch", "epochs") + " in " + time;
            if (run.current_epoch > run.last_executed_epoch)
                out.status += ", stopped during epoch " + std::to_string(run.current_epoch);
        }
        break;
    }
    }
    if (out.state == RunState::Preparing) {
        out.show_progress = true;
        out.progress = std::clamp(h.preparation_progress, 0.0, 1.0);
        out.progress_text = h.preparation_message.empty() ? "Preparing training..." : h.preparation_message;
    } else if ((running || ended) && run.total_epochs > 0) {
        out.show_progress = true;
        if (running) {
            out.progress = run.total_batches > 0
                ? TrainingFractionComplete(run.current_epoch, run.current_batch, run.total_batches, run.total_epochs)
                : static_cast<double>(run.current_epoch) / run.total_epochs;
        } else {
            out.progress = static_cast<double>(run.last_executed_epoch) / run.total_epochs;
        }
        out.progress = std::clamp(out.progress, 0.0, 1.0);
        out.progress_tooltip = Format("%.0f%% of all epochs", out.progress * 100.0);
    }

    // Where it ran and on what data (not while waiting: the route is not known yet).
    if ((running && out.state != RunState::WaitingForGpu) || ended) {
        if (h.route && !h.route->effective_backend.empty()) {
            const Route& r = *h.route;
            out.where = std::string(running ? "Runs on " : "Ran on ") + Place(r);
            if (r.fallback && !r.requested_backend.empty())
                out.where += " (the chosen " + rl_device::BackendWord(r.requested_backend) + " device failed its check)";
            if (r.policy == "compatibility" || r.policy == "strict") out.where += kDot + r.policy + " policy";
            out.where_tooltip = r.effective_backend + " device " + std::to_string(r.effective_device_id) +
                                (r.device_name.empty() ? "" : kDot + r.device_name);
            if (!r.requested_backend.empty() && r.requested_backend != r.effective_backend)
                out.where_tooltip += "\nChosen: " + r.requested_backend;
        }
        std::vector<std::string> data;
        if (!h.dataset.empty()) data.push_back(h.dataset);
        if (const std::string rows = Rows(h); !rows.empty()) data.push_back(rows);
        if (h.samples_per_batch > 0) data.push_back("batches of " + Thousands(h.samples_per_batch));
        for (size_t i = 0; i < data.size(); ++i) out.data += (i == 0 ? "" : kDot) + data[i];
    }

    // While it trains: the latest batches of the epoch that runs.
    if (out.state == RunState::Training || out.state == RunState::Paused) {
        const double finished = run.train_loss_epochs.Empty() ? 0.0 : run.train_loss_epochs.x.back();
        const auto fresh = [&](const std::optional<Point>& p) { return p && p->x > finished + 1e-9; };
        if (fresh(run.latest_loss_point)) {
            out.now = "Latest batches: loss " + Format("%.4f", run.latest_loss_point->y);
            if (fresh(run.latest_accuracy_point))
                out.now += kDot + std::string("accuracy ") + Format("%.2f%%", run.latest_accuracy_point->y);
            if (run.total_batches > 0 && run.current_batch > 0) {
                out.now_detail = "(batch " + std::to_string(run.current_batch) + " of " +
                                 std::to_string(run.total_batches);
                if (run.batches_per_second > 0.0)
                    out.now_detail += ", " + Format(run.batches_per_second >= 10.0 ? "%.0f" : "%.1f",
                                                    run.batches_per_second) + " batches/s";
                out.now_detail += ")";
            }
        }
    }

    out.outcome = Outcome(out.state, h, out.outcome_tone);

    // The active model, once no run trains.
    if (!running && out.state != RunState::Preparing) {
        const bool loaded = out.state == RunState::ModelLoaded || h.active_model_provenance == "loaded_checkpoint_for_testing";
        const bool restored = h.active_model_provenance == "restored_best_checkpoint" ||
                              (h.active_model_provenance.empty() && h.checkpoint_epoch > 0 && !h.checkpoint_path.empty());
        const std::string epoch = h.checkpoint_epoch > 0 ? ", epoch " + std::to_string(h.checkpoint_epoch) : "";
        if (loaded) {
            out.active_model = "Active model: a checkpoint loaded for testing" + epoch;
        } else if (restored) {
            out.active_model = "Active model: the best validation checkpoint" + epoch;
        } else if (ended && (!h.active_model_provenance.empty() || !h.checkpoint_path.empty())) {
            out.active_model = "Active model: the final state of the run";
            if (run.last_executed_epoch > 0) out.active_model += ", epoch " + std::to_string(run.last_executed_epoch);
        }
        if ((loaded || restored) && h.checkpoint_epoch > 0 && h.has_checkpoint_validation) {
            out.active_model_detail = "(validation loss " + Format("%.4f", h.checkpoint_val_loss);
            if (h.classification) out.active_model_detail += ", accuracy " + Format("%.2f%%", h.checkpoint_val_accuracy * 100.0);
            out.active_model_detail += ")";
        }
    }
    if (ended && h.has_test) {
        if (h.classification) {
            out.test = "Held-out test: accuracy " + Format("%.2f%%", h.test_accuracy * 100.0) + ", loss " +
                       Format("%.4f", h.test_loss);
        } else {
            out.test = "Held-out test: loss " + Format("%.4f", h.test_loss);
            if (h.test_mae > 0.0) out.test += ", mean absolute error " + Format("%.4g", h.test_mae);
        }
        if (h.test_rows > 0) out.test_detail = "(" + Thousands(static_cast<double>(h.test_rows)) + " rows the model never saw)";
    }

    // Run details > The run.
    if (ended) {
        const std::string code = IsCode(h.terminal_reason) ? h.terminal_reason : h.terminal_status;
        if (!code.empty()) out.run_rows.push_back({"Stop code", code});
    }
    if (h.checkpoint_epoch > 0 || h.checkpoint_step > 0) {
        std::string checkpoint;
        if (h.checkpoint_epoch > 0) checkpoint = "epoch " + std::to_string(h.checkpoint_epoch);
        if (h.checkpoint_step > 0) checkpoint += (checkpoint.empty() ? "" : ", ") + std::string("step ") + std::to_string(h.checkpoint_step);
        out.run_rows.push_back({"Checkpoint", checkpoint});
    }
    if (!h.checkpoint_path.empty()) out.run_rows.push_back({"Path", h.checkpoint_path});
    if ((running || ended) && h.samples_per_batch > 0) {
        std::string batches = Plural(h.samples_per_batch, "sample", "samples") + " each";
        if (h.batches_per_update > 1) batches += kDot + std::string("an update every ") + std::to_string(h.batches_per_update) + " batches";
        batches += kDot + (h.metric_interval > 0 ? "metrics every " + Plural(h.metric_interval, "batch", "batches")
                                                 : std::string("metrics at the first and last batch"));
        out.run_rows.push_back({"Batches", batches});
    }
    if (running && run.current_batch > 0) out.run_rows.push_back({"Last batch loss", Format("%.4f", h.last_batch_loss)});
    for (DetailRow& row : TaskRows(h.tasks)) out.run_rows.push_back(std::move(row));
    return out;
}

std::vector<DetailRow> TaskRows(const std::vector<EngineTask>& tasks) {
    std::vector<DetailRow> rows;
    for (size_t i = 0; i < tasks.size(); ++i) {
        if (i == 3) {
            rows.push_back({"Engine tasks", "and " + std::to_string(tasks.size() - 3) + " more"});
            break;
        }
        const EngineTask& t = tasks[i];
        rows.push_back({"Engine task", t.name + kDot + Format("%.0f%%", t.progress * 100.0) + kDot +
                                           (t.status.empty() ? std::string("running") : t.status)});
    }
    return rows;
}

}  // namespace cyxwiz::training_dashboard
