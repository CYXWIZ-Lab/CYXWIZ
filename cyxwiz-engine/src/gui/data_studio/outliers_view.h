#pragma once

// Data Studio > Explore > Outliers (TOFIX129 D1, approved mockup board 2;
// replaces the Outlier Detection window): a number column, a rule (IQR,
// Z-score, Modified Z) and its factor; the count re-made at once, the
// profile's histogram with the bounds drawn, the figures, and the rows
// outside the bounds by score, 50 a page. The queries are core/data_explore.

#include "explore_tab.h"
#include "../../core/data_explore.h"
#include "../plot/column_picker.h"

#include <functional>
#include <memory>
#include <string>

namespace cyxwiz {

class OutliersView {
public:
    void Render(const ExploreContext& context);
    // Picks the column (Profile's outlier count, Data > Explore).
    void Select(const std::string& column) { column_ = column; offset_ = 0; }
    bool IsRunning() const { return mad_.Running() || counts_.Running() || rows_.Running() || bins_.Running(); }

    std::function<void(const std::string& sql)> on_open_query;

private:
    bool RenderControls(const DatasetProfile& profile);
    void RenderDistribution(const ExploreContext& context, const ProfiledColumn& column, const explore::OutlierBounds* bounds);
    void RenderRows(const ExploreContext& context, const ProfiledColumn& column, const explore::OutlierBounds& bounds, size_t total);
    void RenderFigures(const DatasetProfile& profile, const ProfiledColumn& column, const explore::OutlierBounds* bounds);
    std::string Key(const std::string& dataset) const;  // dataset, column, rule, factor

    std::shared_ptr<const DatasetProfile> seen_;
    std::string column_;
    explore::OutlierRule rule_ = explore::OutlierRule::Iqr;
    double factor_ = 1.5;       // in use
    double factor_edit_ = 1.5;  // in the field while it is being typed
    size_t offset_ = 0;
    plot::ColumnPicker picker_;
    ExploreJob<explore::Mad> mad_;
    ExploreJob<explore::OutlierCounts> counts_;
    ExploreJob<explore::RowPage> rows_;
    ExploreJob<explore::Bins> bins_;
};

}  // namespace cyxwiz
