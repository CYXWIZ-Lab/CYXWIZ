// MetricAccumulator (TOFIX242 2.2a): a pass's loss and metrics summed on the
// device and read once, against the same sums worked by hand.
#include "../src/core/metric_accumulator.h"

#include <cyxwiz/device.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << "\n";
        std::exit(1);
    }
}

bool Near(double a, double b) { return std::abs(a - b) <= 1e-5 * std::max(1.0, std::abs(b)); }

cyxwiz::Tensor Scalar(float value) { return cyxwiz::Tensor({1}, &value, cyxwiz::DataType::Float32); }

cyxwiz::Tensor Values(const std::vector<float>& values, std::vector<size_t> shape) {
    return cyxwiz::Tensor(shape, values.data(), cyxwiz::DataType::Float32);
}

af::array Counts(std::vector<float> values) {
    return af::array(static_cast<dim_t>(values.size()), values.data());
}

}  // namespace

int main() {
    const auto activation = cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(true);
    Check(activation.success, "the ArrayFire CPU device activates");

    // A Mean loss weighted by its device denominator: (2 * 1 + 4 * 3) / (1 + 3)
    {
        cyxwiz::MetricAccumulator sums;
        const auto w1 = Scalar(1.0f), w3 = Scalar(3.0f);
        sums.AddLoss(Scalar(2.0f), &w1, 0.0f);
        sums.AddLoss(Scalar(4.0f), &w3, 0.0f);
        const auto totals = sums.Read("test", "test");
        Check(Near(totals.Loss(cyxwiz::Reduction::Mean), 3.5) && Near(totals.weight_sum, 4.0) && totals.finite &&
                  totals.batches == 2,
              "a Mean loss is the device-weighted mean (" + std::to_string(totals.Loss(cyxwiz::Reduction::Mean)) + ")");
    }
    // Host weights, and a Sum loss is the losses' sum
    {
        cyxwiz::MetricAccumulator sums;
        sums.AddLoss(Scalar(1.0f), nullptr, 2.0f);
        sums.AddLoss(Scalar(3.0f), nullptr, 6.0f);
        const auto totals = sums.Read("test", "test");
        Check(Near(totals.Loss(cyxwiz::Reduction::Mean), 20.0 / 8.0), "a Mean loss with host weights");
        cyxwiz::MetricAccumulator summed;
        summed.AddLoss(Scalar(1.5f), nullptr, 1.0f);
        summed.AddLoss(Scalar(2.5f), nullptr, 1.0f);
        Check(Near(summed.Read("test", "test").Loss(cyxwiz::Reduction::Sum), 4.0), "a Sum loss is the losses' sum");
        bool refused = false;
        try {
            sums.AddLoss(Scalar(1.0f), nullptr, -1.0f);
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        Check(refused, "a negative loss weight is refused");
    }
    // A NaN or infinite loss shows in the sums
    {
        cyxwiz::MetricAccumulator sums;
        sums.AddLoss(Scalar(1.0f), nullptr, 1.0f);
        sums.AddLoss(Scalar(std::numeric_limits<float>::quiet_NaN()), nullptr, 1.0f);
        sums.AddLoss(Scalar(2.0f), nullptr, 1.0f);
        Check(!sums.Read("test", "test").finite, "a NaN loss makes the pass not finite");
        cyxwiz::MetricAccumulator infinite;
        infinite.AddLoss(Scalar(std::numeric_limits<float>::infinity()), nullptr, 1.0f);
        Check(!infinite.Read("test", "test").finite, "an infinite loss makes the pass not finite");
    }
    // Decision counts, as [correct, total] or with the two refusal counts
    {
        cyxwiz::MetricAccumulator sums;
        sums.AddCounts(Counts({3, 4}));
        sums.AddCounts(Counts({1, 4, 0, 0}));
        sums.AddHostCounts(2, 2);
        const auto totals = sums.Read("test", "test");
        Check(totals.correct == 6 && totals.total == 10 && Near(totals.Accuracy(), 0.6), "counts add up");
        cyxwiz::MetricAccumulator outside;
        outside.AddCounts(Counts({1, 2, 1, 0}));
        bool refused = false;
        try {
            (void)outside.Read("test", "test");
        } catch (const std::runtime_error& e) {
            refused = std::string(e.what()).find("outside the classes") != std::string::npos;
        }
        Check(refused, "a target outside the classes is refused at the read");
        cyxwiz::MetricAccumulator nonfinite;
        nonfinite.AddCounts(Counts({1, 2, 0, 3}));
        refused = false;
        try {
            (void)nonfinite.Read("test", "test");
        } catch (const std::runtime_error& e) {
            refused = std::string(e.what()).find("NaN or infinite") != std::string::npos;
        }
        Check(refused, "non-finite model outputs are refused at the read");
    }
    // Regression: |e| and e^2 over two batches of [2, 2], in original units
    {
        cyxwiz::RegressionTargetTransform none;
        cyxwiz::MetricAccumulator sums;
        sums.AddRegression(Values({1, 2, 3, 4}, {2, 2}), Values({0, 2, 5, 4}, {2, 2}), 2, 2, none);  // e = 1, 0, -2, 0
        sums.AddRegression(Values({0, 0, 0, 0}, {2, 2}), Values({1, 1, 1, 1}, {2, 2}), 2, 2, none);  // e = -1 x 4
        auto totals = sums.Read("test", "test");
        Check(totals.regression_values == 8 && Near(totals.Mae(), 7.0 / 8.0) && Near(totals.Rmse(), std::sqrt(9.0 / 8.0)),
              "MAE and RMSE over every value (" + std::to_string(totals.Mae()) + ", " + std::to_string(totals.Rmse()) +
                  ")");
        cyxwiz::RegressionTargetTransform scaled;
        scaled.enabled = true;
        scaled.resolved = true;
        scaled.scales = {10.0, 1.0};
        cyxwiz::MetricAccumulator original;
        original.AddRegression(Values({1, 2, 3, 4}, {2, 2}), Values({0, 2, 5, 4}, {2, 2}), 2, 2, scaled);  // 10, 0, -20, 0
        totals = original.Read("test", "test");
        Check(Near(totals.Mae(), 30.0 / 4.0) && Near(totals.Rmse(), std::sqrt(500.0 / 4.0)),
              "regression errors in the target's original units");
    }
    // A pass resumed from saved totals continues them
    {
        cyxwiz::MetricAccumulator::Totals saved;
        saved.loss_sum = 6.0;
        saved.weight_sum = 3.0;
        saved.correct = 2;
        saved.total = 3;
        saved.batches = 3;
        cyxwiz::MetricAccumulator sums;
        sums.Restore(saved);
        sums.AddLoss(Scalar(4.0f), nullptr, 1.0f);
        sums.AddCounts(Counts({1, 1}));
        const auto totals = sums.Read("test", "test");
        Check(Near(totals.Loss(cyxwiz::Reduction::Mean), 10.0 / 4.0) && totals.correct == 3 && totals.total == 4 &&
                  totals.batches == 4,
              "a restored pass continues its sums");
    }
    // Nothing added: zeros, finite
    {
        cyxwiz::MetricAccumulator sums;
        const auto totals = sums.Read("test", "test");
        Check(totals.finite && totals.Loss(cyxwiz::Reduction::Mean) == 0.0 && totals.Accuracy() == 0.0,
              "an empty pass reads zeros");
    }
    std::cout << "MetricAccumulator tests passed\n";
    return 0;
}
