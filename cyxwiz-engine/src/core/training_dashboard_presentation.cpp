#include "training_dashboard_presentation.h"

#include "training_progress_estimate.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

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

}  // namespace cyxwiz::training_dashboard
