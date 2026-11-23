// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
#pragma once

// Calibration of a classifier's probabilities (TOFIX129 D11): how often the
// model is right when it says p (reliability bins, expected calibration
// error), and fixes that keep every prediction's order: one temperature for
// several classes (softmax of log p / T), Platt's sigmoid or isotonic
// regression for a yes / no score. Checked against scikit-learn and scipy
// (the Engine's threshold_calibration truth test); the Engine's Results >
// Test > Calibration and pycyxwiz.model use these.

#include "cyxwiz/api_export.h"

#include <cstddef>
#include <vector>

namespace cyxwiz {

struct CYXWIZ_API ReliabilityBin {
    double low = 0.0, high = 0.0;
    int count = 0;
    double confidence = 0.0;  // mean confidence of its rows
    double accuracy = 0.0;    // share of its rows the model got right
};
struct CYXWIZ_API Reliability {
    std::vector<ReliabilityBin> bins;  // `bins` equal bands of [0, 1]; 1.0 in the last
    double ece = 0.0;                  // sum over bins of count / n x |accuracy - confidence|
    int rows = 0;
};
CYXWIZ_API Reliability ReliabilityOf(const std::vector<double>& confidence, const std::vector<int>& correct, int bins = 10);

// Several classes: T minimizing the mean negative log-likelihood of
// softmax(log p / T) over `rows` (p clipped at 1e-12), on log T.
struct CYXWIZ_API TemperatureFit {
    double temperature = 1.0;
    double nll_before = 0.0, nll_after = 0.0;
};
CYXWIZ_API TemperatureFit FitTemperature(const std::vector<float>& probabilities, int classes, const std::vector<int>& truth,
                              const std::vector<size_t>& rows);
// One row's probabilities after a temperature.
CYXWIZ_API std::vector<double> WithTemperature(const float* probabilities, int classes, double temperature);

// Yes / no: Platt's sigmoid 1 / (1 + exp(a s + b)) of the score s, fitted
// as scikit-learn's _sigmoid_calibration (Platt's smoothed targets, Newton
// steps on the log-loss).
struct CYXWIZ_API PlattFit {
    double a = 0.0, b = 0.0;
};
CYXWIZ_API PlattFit FitPlatt(const std::vector<double>& scores, const std::vector<int>& labels);
CYXWIZ_API double ApplyPlatt(const PlattFit& fit, double score);

// Yes / no: isotonic regression of the label on the score (non-decreasing,
// 0..1), as scikit-learn's IsotonicRegression(out_of_bounds="clip"): ties of
// the score averaged, pool adjacent violators, linear between the fitted
// points, clipped outside them.
struct CYXWIZ_API IsotonicFit {
    std::vector<double> x, y;  // the fitted points, x ascending
};
CYXWIZ_API IsotonicFit FitIsotonic(const std::vector<double>& scores, const std::vector<int>& labels);
CYXWIZ_API double ApplyIsotonic(const IsotonicFit& fit, double score);

}  // namespace cyxwiz
