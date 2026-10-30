#pragma once

// Data Studio > Explore > Series (TOFIX129 D6, approved mockup boards 1-3;
// replaces the Decomposition, ACF / PACF, Stationarity and Seasonality
// windows). One value column over a time column (the Data Input's time
// column, else a date column, else the row order), its period found or
// typed, in four looks: Parts (STL or classical decomposition), Seasonality
// (the detrended periodogram and the season by position), Stationarity (ADF
// and KPSS on the series and its differences, rolling mean and spread) and
// Correlogram (ACF / PACF with their bands and the Ljung-Box test). Worked
// out in the background (core/series_explore, core/series_presentation).

#include "explore_tab.h"
#include "../../core/series_explore.h"
#include "../plot/column_picker.h"

#include <functional>
#include <map>
#include <memory>
#include <string>

namespace cyxwiz {

// The Time (a date column, a text that reads as dates, numbers, or the row
// order), Value (a number column) and Period pickers shared by the Series and
// Forecast tabs; the defaults are the Data Input's time column, else the one
// date column, else the row order (D6-4). True when a choice changed.
bool RenderSeriesPickers(const char* id, const ExploreContext& context, const DatasetProfile& profile, plot::ColumnPicker& picker,
                         series::SeriesRequest& base);

class SeriesView {
public:
    enum class Look { Parts, Seasonality, Stationarity, Correlogram };

    void Render(const ExploreContext& context);
    // Opens on this look (Tools > Time Series).
    void SelectLook(Look look) { look_ = look; }
    bool IsRunning() const { return parts_job_.Running() || season_job_.Running() || stationarity_job_.Running() || correlogram_job_.Running(); }
    // The series settings in use (time, value, period), for the Forecast tab to start from.
    const series::SeriesRequest& Base() const { return base_; }

    // Into the graph: a node of `type` (its legacy name, "TimeSeriesDecomposition")
    // named `name` with `params`, after the data in use (`step`) or as a
    // branch reading it; returns what happened.
    std::function<std::string(const std::string& type, const std::string& name, std::map<std::string, std::string> params, bool step)> add_to_graph;

private:
    void RenderControls(const ExploreContext& context, const DatasetProfile& profile);
    void RenderLookSettings();
    void RenderParts(const ExploreContext& context);
    void RenderSeasonality(const ExploreContext& context);
    void RenderStationarity(const ExploreContext& context);
    void RenderCorrelogram(const ExploreContext& context);
    void RenderSeriesLine(const series::Series& s);
    void Message(const std::string& text, bool error);
    void Export(const std::string& name, const std::string& csv);

    std::shared_ptr<const DatasetProfile> seen_;
    std::string dataset_;
    Look look_ = Look::Parts;
    series::SeriesRequest base_;
    series::PartsRequest parts_;
    series::StationarityRequest stationarity_;
    series::CorrelogramRequest correlogram_;
    std::string message_;
    bool message_error_ = false;
    plot::ColumnPicker picker_;
    ExploreJob<series::PartsAnswer> parts_job_;
    ExploreJob<series::SeasonalityAnswer> season_job_;
    ExploreJob<series::StationarityAnswer> stationarity_job_;
    ExploreJob<series::CorrelogramAnswer> correlogram_job_;
};

}  // namespace cyxwiz
