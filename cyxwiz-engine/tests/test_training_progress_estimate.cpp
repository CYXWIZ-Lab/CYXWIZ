// Training progress fraction and remaining-time estimate (dashboard + Tasks).
#include "../src/core/training_progress_estimate.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {
int failures = 0;
void Check(bool ok, const std::string& what) {
    if (!ok) { std::cerr << "FAIL: " << what << "\n"; ++failures; }
}
bool Near(double a, double b, double tol) { return std::fabs(a - b) <= tol; }
}  // namespace

int main() {
    using cyxwiz::TrainingEtaEstimator;
    using cyxwiz::TrainingFractionComplete;

    // Progress counts the epoch in progress (the old formula said 1/1 = 100%).
    Check(Near(TrainingFractionComplete(1, 17971, 19426, 1), 17971.0 / 19426.0, 1e-9),
          "single-epoch progress is the batch fraction, not 100%");
    Check(Near(TrainingFractionComplete(2, 0, 100, 3), 1.0 / 3.0, 1e-9), "epoch 2 start of 3 is one third");
    Check(Near(TrainingFractionComplete(3, 100, 100, 3), 1.0, 1e-9), "last batch of last epoch is complete");
    Check(TrainingFractionComplete(1, 5, 0, 0) == 0.0, "unknown shape reports zero");

    // Single epoch, steady 1.4 batches/s: estimate appears after 10 s and is exact.
    {
        TrainingEtaEstimator eta;
        double t = 0.0;
        int batch = 0;
        eta.Observe(t, 1, ++batch, 19426, 1);
        Check(!eta.RemainingSeconds(), "no estimate from a single observation");
        for (int i = 0; i < 12; ++i) { t += 1.0 / 1.4; eta.Observe(t, 1, ++batch, 19426, 1); }
        Check(!eta.RemainingSeconds(), "still estimating before 10 s of progress");
        while (t < 90.0) { t += 1.0 / 1.4; eta.Observe(t, 1, ++batch, 19426, 1); }
        const auto remaining = eta.RemainingSeconds();
        Check(remaining.has_value(), "estimate available in a one-epoch run");
        Check(remaining && Near(*remaining, (19426 - batch) / 1.4, 2.0),
              "one-epoch estimate = remaining batches / rate");
        Check(Near(eta.BatchesPerSecond(), 1.4, 1e-6), "measured rate");
        Check(Near(eta.FractionComplete(), batch / 19426.0, 1e-9), "fraction from the estimator");
    }

    // Speed change: the estimate follows the recent rate (dynamic), not the start.
    {
        TrainingEtaEstimator eta;
        double t = 0.0;
        int batch = 0;
        for (; t < 300.0; t += 0.5) eta.Observe(t, 1, ++batch, 100000, 1);  // 2 batches/s
        Check(Near(eta.BatchesPerSecond(), 2.0, 0.02), "initial rate 2/s");
        for (double end = t + 300.0; t < end; t += 1.0) eta.Observe(t, 1, ++batch, 100000, 1);  // 1/s
        Check(Near(eta.BatchesPerSecond(), 1.0, 0.02), "rate follows the slowdown within the window");
        const auto remaining = eta.RemainingSeconds();
        Check(remaining && Near(*remaining, (100000 - batch) / 1.0, 150.0), "estimate uses the new rate");
    }

    // Epoch boundaries: measured pause is added once per remaining boundary and
    // is kept out of the batch rate.
    {
        TrainingEtaEstimator eta;
        double t = 0.0;
        for (int b = 1; b <= 100; ++b) { eta.Observe(t, 1, b, 100, 3); t += 0.5; }  // 2 batches/s
        t += 30.0;  // validation + preview + checkpoint
        eta.Observe(t, 2, 1, 100, 3);
        Check(eta.RemainingSeconds().has_value(), "estimate survives the epoch boundary");
        for (int b = 2; b <= 50; ++b) { t += 0.5; eta.Observe(t, 2, b, 100, 3); }
        Check(eta.HasMeasuredEpochOverhead(), "boundary pause measured");
        Check(Near(eta.MeanEpochOverheadSeconds(), 30.0, 0.05), "boundary pause about 30 s");
        Check(Near(eta.BatchesPerSecond(), 2.0, 0.01), "pause does not dilute the batch rate");
        // 50 + 100 batches left at 2/s = 75 s, plus two boundaries (end of epoch 2
        // and of epoch 3) at ~30 s.
        const auto remaining = eta.RemainingSeconds();
        Check(remaining && Near(*remaining, 75.0 + 2 * 30.0, 1.0), "estimate includes remaining boundaries");
    }

    // Duplicate polls are ignored; a new run (progress backwards) starts over.
    {
        TrainingEtaEstimator eta;
        for (int b = 1; b <= 200; ++b) eta.Observe(b * 0.1, 1, b, 1000, 1);
        const double rate = eta.BatchesPerSecond();
        eta.Observe(25.0, 1, 200, 1000, 1);  // same batch polled later
        Check(Near(eta.BatchesPerSecond(), rate, 1e-9), "duplicate poll ignored");
        eta.Observe(30.0, 1, 3, 1000, 1);    // a restarted run
        Check(!eta.RemainingSeconds(), "restart resets the estimate");
    }

    Check(cyxwiz::FormatTrainingDuration(3725) == "1h 02m", "hours format");
    Check(cyxwiz::FormatTrainingDuration(750) == "12m 30s", "minutes format");
    Check(cyxwiz::FormatTrainingDuration(45.4) == "45s", "seconds format");

    if (failures) return 1;
    std::cout << "Training progress estimate test passed\n";
    return 0;
}
