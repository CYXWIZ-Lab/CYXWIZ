#pragma once

// Data Studio > Explore > Forecast (TOFIX129 D6, approved mockup board 4;
// replaces the Forecasting window). One value column over a time column,
// forecast `horizon` steps ahead by simple exponential smoothing, Holt
// (damped or not), Holt-Winters (additive or multiplicative season),
// ARIMA (orders typed or chosen) or the seasonal naive baseline, with the
// model's prediction interval at a typed level, future dates continued from
// the step, and the held-out check: the last H rows held out, every method
// scored against repeating last cycle (D6-6, on by default). Worked out in
// the background (core/series_explore, core/series_presentation).

#include "explore_tab.h"
#include "../../core/series_explore.h"
#include "../plot/column_picker.h"

#include <functional>
#include <map>
#include <memory>
#include <string>

namespace cyxwiz {

class ForecastView {
public:
    void Render(const ExploreContext& context);
    // Starts from these series settings (the Series tab's) unless the user already chose here.
    void StartFrom(const series::SeriesRequest& base);
    bool IsRunning() const { return job_.Running(); }

    // Into the graph: the ARIMA or Exp. Smoothing node (its legacy type name)
    // with these parameters, after the data in use (`step`) or as a branch.
    std::function<std::string(const std::string& type, const std::string& name, std::map<std::string, std::string> params, bool step)> add_to_graph;

private:
    void RenderControls(const ExploreContext& context, const DatasetProfile& profile);
    void RenderMethodSettings();
    void RenderAnswer(const series::ForecastAnswer& a);
    void RenderTable(const series::ForecastAnswer& a);
    void RenderHoldout(const series::ForecastAnswer& a);
    void RenderActions(const series::ForecastAnswer& a);
    void Message(const std::string& text, bool error);

    std::shared_ptr<const DatasetProfile> seen_;
    series::ForecastRequest request_;
    bool chosen_ = false;        // the user picked the series here
    bool estimate_ = true;       // alpha / beta / gamma estimated
    bool holdout_ = true;        // the held-out check
    int holdout_rows_ = 0;       // 0: the model's default (twice the period found, else 12)
    int last_holdout_ = 0;       // what the last answer held out (shown in the field)
    std::string stopped_key_;
    std::string message_;
    bool message_error_ = false;
    plot::ColumnPicker picker_;
    ExploreJob<series::ForecastAnswer> job_;
};

}  // namespace cyxwiz
