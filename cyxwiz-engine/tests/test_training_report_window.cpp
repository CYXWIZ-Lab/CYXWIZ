// The Training Dashboard's batch points are the mean since the previous point
// (TOFIX178 1.8i): ReportWindow turns an epoch's running sums into them.
#include "core/training_report_window.h"

#include <cmath>
#include <cstdio>
#include <string>

namespace {
int failures = 0;
void Check(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what.c_str());
        ++failures;
    }
}
bool Near(double a, double b) { return std::abs(a - b) < 1e-5; }
}  // namespace

int main() {
    using cyxwiz::Reduction;
    using cyxwiz::ReportWindow;

    // Batches of 64 with losses 0.9, 0.5, 0.1; points after batch 1 and batch 3.
    ReportWindow w;
    Check(Near(w.Loss(Reduction::Mean, 0.9f, 64.0), 0.9), "the first point is batch 1");
    const float running = (0.9f + 0.5f + 0.1f) / 3.0f;
    Check(Near(w.Loss(Reduction::Mean, running, 192.0), 0.3), "the next point is batches 2-3 only: 0.3");

    // A smaller last batch weighs by its loss weight (32 of 96 samples).
    ReportWindow partial;
    partial.Loss(Reduction::Mean, 0.6f, 64.0);
    const float with_last = static_cast<float>((0.6 * 64 + 0.3 * 32) / 96.0);
    Check(Near(partial.Loss(Reduction::Mean, with_last, 96.0), 0.3), "a partial batch: its own mean");

    // Sum reduction: the running value is a sum; the window is what was added.
    ReportWindow sum;
    sum.Loss(Reduction::Sum, 10.0f, 1.0);
    Check(Near(sum.Loss(Reduction::Sum, 25.0f, 2.0), 15.0), "sum reduction: the added sum");

    // Accuracy from counts: 60 / 64, then 180 / 192 -> 120 / 128 since the point.
    ReportWindow acc;
    Check(Near(acc.Accuracy(60, 64, 60.0f / 64.0f), 60.0 / 64.0), "first accuracy point");
    Check(Near(acc.Accuracy(180, 192, 180.0f / 192.0f), 120.0 / 128.0), "accuracy since the point");
    Check(Near(acc.Accuracy(180, 192, 0.5f), 0.5), "nothing new: the running value");

    // Batch 1 ends a window in epoch 1 only.
    Check(ReportWindow::Ends(1, 1) && !ReportWindow::Ends(2, 1) && ReportWindow::Ends(2, 10),
          "batch 1 is a point in the first epoch only");

    if (failures == 0) std::printf("training report window: OK\n");
    return failures == 0 ? 0 : 1;
}
