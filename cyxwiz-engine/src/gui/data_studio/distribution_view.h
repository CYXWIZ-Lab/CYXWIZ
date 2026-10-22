#pragma once

// Data Studio > Explore > Distribution (TOFIX129 D1, approved mockup board 4;
// replaces the Summary Statistics and Distribution Fitter windows): one
// number column's figures and percentiles, its histogram with the chosen
// fit drawn over it, a box plot, a Q-Q plot against the fit, and Normal,
// Uniform, Exponential and Log-Normal fits ranked by AIC (KS test), with
// Export CSV. The queries and fits are core/data_explore.

#include "explore_tab.h"
#include "../../core/data_explore.h"
#include "../plot/column_picker.h"

#include <memory>
#include <string>

namespace cyxwiz {

class DistributionView {
public:
    void Render(const ExploreContext& context);
    // Picks the column (Profile's "More about this column", Tools > Statistics).
    void Select(const std::string& column) { column_ = column; fit_index_ = 0; }
    bool IsRunning() const { return figures_.Running() || fit_.Running() || bins_.Running(); }

private:
    void RenderControls(const ExploreContext& context, const DatasetProfile& profile);
    void RenderHistogram(const ExploreContext& context, const ProfiledColumn& column, const explore::ColumnFigures* figures,
                         const DistributionFitResult* fit);
    void RenderBoxPlot(const explore::ColumnFigures& figures, const explore::ColumnFit* values);
    void RenderQq(const explore::ColumnFit& values, const DistributionFitResult& fit);
    void RenderSide(const explore::ColumnFigures* figures, const explore::ColumnFit* fit);
    void Export(const ExploreContext& context, const explore::ColumnFigures& figures, const explore::ColumnFit* fit);

    std::shared_ptr<const DatasetProfile> seen_;
    std::string column_;
    int bins_wanted_ = 20;
    size_t fit_index_ = 0;  // the fit drawn (best AIC first)
    std::string message_;
    bool message_error_ = false;
    plot::ColumnPicker picker_;
    ExploreJob<explore::ColumnFigures> figures_;
    ExploreJob<explore::ColumnFit> fit_;
    ExploreJob<explore::Bins> bins_;
};

}  // namespace cyxwiz
