#pragma once

// What the Training Dashboard says, without ImGui (TOFIX129 B4, approved
// mockup 2026-11-18): the metric cards, the "How the run is going" rows and
// the time words. The panel (gui/panels/training_plot_panel.cpp) draws them;
// test_training_dashboard_presentation checks them on a real run's numbers.

#include <optional>
#include <string>
#include <vector>

namespace cyxwiz::training_dashboard {

// A curve: x is the epoch (a fraction inside an epoch for batch points).
struct Series {
    std::vector<double> x;
    std::vector<double> y;
    bool Empty() const { return x.empty() || y.empty(); }
    size_t Size() const { return x.size() < y.size() ? x.size() : y.size(); }
};

// A chart point: x is the epoch position, y the value.
struct Point {
    double x = 0.0;
    double y = 0.0;
};

// What the panel knows about the run.
struct RunFacts {
    bool training = false;
    // Training: the latest batch point (x = epoch - 1 + batch / batches; its
    // value is the mean of the batches since the point before, not the
    // epoch's running mean) and one value per finished epoch (x = the epoch).
    std::optional<Point> latest_loss_point;
    std::optional<Point> latest_accuracy_point;
    Series train_loss_epochs;
    Series train_accuracy_epochs;
    // Validation: one value per finished epoch.
    Series val_loss;
    Series val_accuracy;
    // Progress and time.
    int current_epoch = 0;
    int total_epochs = 0;
    int last_executed_epoch = 0;
    int current_batch = 0;
    int total_batches = 0;
    std::optional<double> remaining_seconds;  // the ETA, once it has a rate
    double batches_per_second = 0.0;
    double samples_per_second = 0.0;
    double avg_epoch_seconds = 0.0;
    std::optional<double> epoch_end_seconds;  // measured time of an epoch's end (validation, previews, checkpoint)
    double total_seconds = 0.0;
};

enum class Tone { Good, Bad, Plain, Faint };

// One metric card: its name, the value, the change since the epoch before,
// then up to two more lines.
struct Card {
    std::string id;      // a stable ImGui id
    std::string label;   // "Loss · epoch 10"
    int series = -1;     // which curve's colour: 0 train loss, 1 val loss, 2 train acc, 3 val acc, -1 time
    std::string value;   // "0.2223"
    std::string tooltip;
    std::string change;  // "0.0376 since epoch 9" (the caret is drawn by the panel)
    int direction = 0;   // -1 down, +1 up, 0 none
    Tone change_tone = Tone::Faint;
    std::string line2;
    Tone line2_tone = Tone::Faint;
    std::string line3;   // "latest batches 0.5287"
    Tone line3_tone = Tone::Plain;
};
std::vector<Card> BuildCards(const RunFacts& facts);

// "How the run is going": rows in plain words and the suggestion.
struct InsightRow {
    std::string label;
    std::string value;
    Tone tone = Tone::Plain;
};
struct Insights {
    std::string suggestion;  // "Suggested: keep going"
    Tone suggestion_tone = Tone::Good;
    std::vector<InsightRow> rows;
};
Insights BuildInsights(const RunFacts& facts);

// "23,000"
std::string Thousands(double value);
// "0.22s" for a time per epoch.
std::string PerEpoch(double seconds);

}  // namespace cyxwiz::training_dashboard
