#pragma once

// Data Studio > Explore > Regression (TOFIX129 D3, approved mockup boards 4-5;
// replaces the Regression Analysis window). Predict a number column from one
// or more others (or a polynomial in one): coefficients with their CIs, the
// equation with the column names, R², AIC, what the fit says in words, and
// the residuals (vs fitted, Q-Q, histogram, actual vs predicted). Fitted in
// the background on the device (core/stats_explore, StatisticalTests).

#include "explore_tab.h"
#include "../../core/stats_explore.h"
#include "../plot/column_picker.h"

#include <functional>
#include <memory>
#include <string>

namespace cyxwiz {

class RegressionView {
public:
    void Render(const ExploreContext& context);
    // Starts predicting this column.
    void Select(const std::string& column) { request_.response = column; }
    bool IsRunning() const { return job_.Running(); }

    // Add to the graph: a Linear or Polynomial Regression node with these columns.
    std::function<void(const stats::RegressionRequest&)> add_to_graph;

private:
    void RenderControls(const DatasetProfile& profile);
    void RenderCoefficients(const stats::RegressionAnswer& a);
    void RenderFit(const stats::RegressionAnswer& a);
    void RenderPlot(const stats::RegressionAnswer& a, float width);
    void Export(const stats::RegressionAnswer& a);

    std::shared_ptr<const DatasetProfile> seen_;
    stats::RegressionRequest request_;
    int plot_ = 0;  // data and fit (one predictor) / vs fitted / Q-Q / histogram / actual vs predicted
    // The residual plots' data, worked out once per answer.
    std::string key_;          // the answer shown
    std::string plotted_key_;  // the answer the plot data is for
    std::vector<double> qq_x_, qq_y_;
    std::vector<int64_t> bins_;
    double bins_lo_ = 0.0, bins_hi_ = 0.0;
    std::string message_;
    bool message_error_ = false;
    plot::ColumnPicker picker_;
    ExploreJob<stats::RegressionAnswer> job_;
};

}  // namespace cyxwiz
