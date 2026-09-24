#pragma once

// Training progress and remaining-time estimate shared by the Training
// Dashboard and the Tasks panel.
//
// Work is counted in batches across all epochs (epoch numbers and batch
// numbers are 1-based, as the batch callback reports them): an epoch in
// progress counts as (epoch - 1) complete epochs plus its finished batches.
// The rate is measured over a recent time window (default 120 s), so it follows
// speed changes and forgets start-up warm-up. The pause at each epoch boundary
// (validation, generation previews, checkpointing) is measured when the next
// epoch's first batch arrives and added once per remaining boundary.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>
#include <optional>
#include <string>

namespace cyxwiz {

// Fraction of the whole run completed, in [0, 1].
inline double TrainingFractionComplete(int epoch, int batch, int batches_per_epoch, int total_epochs) {
    if (total_epochs <= 0) return 0.0;
    double done_epochs = std::max(0, epoch - 1);
    if (batches_per_epoch > 0) {
        done_epochs += std::clamp(static_cast<double>(batch) / batches_per_epoch, 0.0, 1.0);
    }
    return std::clamp(done_epochs / total_epochs, 0.0, 1.0);
}

// "1h 04m", "12m 30s", "45s".
inline std::string FormatTrainingDuration(double seconds) {
    const long long total = std::max(0LL, static_cast<long long>(std::llround(seconds)));
    const long long h = total / 3600, m = (total % 3600) / 60, s = total % 60;
    char buffer[48];
    if (h > 0) std::snprintf(buffer, sizeof(buffer), "%lldh %02lldm", h, m);
    else if (m > 0) std::snprintf(buffer, sizeof(buffer), "%lldm %02llds", m, s);
    else std::snprintf(buffer, sizeof(buffer), "%llds", s);
    return buffer;
}

class TrainingEtaEstimator {
public:
    explicit TrainingEtaEstimator(double window_seconds = 120.0, double minimum_span_seconds = 10.0)
        : window_seconds_(window_seconds), minimum_span_seconds_(minimum_span_seconds) {}

    void Reset() { *this = TrainingEtaEstimator(window_seconds_, minimum_span_seconds_); }

    // Record progress observed at `now_seconds` (any monotonic clock).
    void Observe(double now_seconds, int epoch, int batch, int batches_per_epoch, int total_epochs) {
        if (batches_per_epoch <= 0 || total_epochs <= 0 || epoch <= 0) return;
        if (batches_per_epoch != batches_per_epoch_ || total_epochs != total_epochs_) {
            // A different run shape: start over rather than mix rates.
            Reset();
            batches_per_epoch_ = batches_per_epoch;
            total_epochs_ = total_epochs;
        }
        const double units = static_cast<double>(epoch - 1) * batches_per_epoch + batch;
        if (!samples_.empty() && units < samples_.back().units) {
            Reset();  // progress went backwards: a new run reused the estimator
            batches_per_epoch_ = batches_per_epoch;
            total_epochs_ = total_epochs;
        }
        if (!samples_.empty() && units == samples_.back().units) return;  // duplicate poll

        // Epoch boundary: time between the previous epoch's last batch and this
        // epoch's first batch, minus one ordinary batch, is boundary overhead.
        // The pause is then removed from the timeline (samples keep continuity,
        // the batch rate excludes it, and the estimate never disappears).
        if (epoch > last_epoch_ && last_epoch_ > 0 && !samples_.empty()) {
            const double gap = (now_seconds - pause_offset_) - samples_.back().seconds;
            const double rate = BatchesPerSecond();
            const double overhead = gap - (rate > 0.0 ? 1.0 / rate : 0.0);
            if (overhead > 0.0) {
                overhead_total_ += overhead;
                ++overhead_count_;
                pause_offset_ += overhead;
            }
        }
        last_epoch_ = epoch;
        epoch_ = epoch;
        units_ = units;
        const double busy_seconds = now_seconds - pause_offset_;
        samples_.push_back({busy_seconds, units});
        while (samples_.size() > 2 && samples_.front().seconds < busy_seconds - window_seconds_) {
            samples_.pop_front();
        }
    }

    // Batches per second over the recent window; 0 until enough is measured.
    double BatchesPerSecond() const {
        if (samples_.size() < 2) return 0.0;
        const double span = samples_.back().seconds - samples_.front().seconds;
        const double work = samples_.back().units - samples_.front().units;
        if (span < minimum_span_seconds_ || work <= 0.0) return 0.0;
        return work / span;
    }

    // Remaining wall-clock seconds, or nothing while still estimating.
    std::optional<double> RemainingSeconds() const {
        const double rate = BatchesPerSecond();
        if (rate <= 0.0) return std::nullopt;
        const double total_units = static_cast<double>(batches_per_epoch_) * total_epochs_;
        const double remaining_batches = std::max(0.0, total_units - units_);
        const int remaining_boundaries = std::max(0, total_epochs_ - epoch_ + 1);
        const double overhead = overhead_count_ > 0 ? overhead_total_ / overhead_count_ : 0.0;
        return remaining_batches / rate + remaining_boundaries * overhead;
    }

    double FractionComplete() const {
        return batches_per_epoch_ > 0 && total_epochs_ > 0
            ? std::clamp(units_ / (static_cast<double>(batches_per_epoch_) * total_epochs_), 0.0, 1.0)
            : 0.0;
    }
    double MeanEpochOverheadSeconds() const {
        return overhead_count_ > 0 ? overhead_total_ / overhead_count_ : 0.0;
    }
    bool HasMeasuredEpochOverhead() const { return overhead_count_ > 0; }

private:
    struct Sample { double seconds; double units; };
    double window_seconds_;
    double minimum_span_seconds_;
    std::deque<Sample> samples_;
    int batches_per_epoch_ = 0;
    int total_epochs_ = 0;
    int epoch_ = 0;
    int last_epoch_ = 0;
    double units_ = 0.0;
    double overhead_total_ = 0.0;
    int overhead_count_ = 0;
    double pause_offset_ = 0.0;  // measured boundary pauses removed from the timeline
};

}  // namespace cyxwiz
