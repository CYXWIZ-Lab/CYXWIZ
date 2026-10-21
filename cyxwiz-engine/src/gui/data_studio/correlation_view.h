#pragma once

// Data Studio > Explore > Correlation (TOFIX129 D1, approved mockup board 3;
// replaces the Correlation Matrix window): Pearson r of every pair of number
// columns over the rows both have, as a heatmap (the profile's matrix of its
// first 30 numbers, or the columns picked), the picked pair in words with the
// rows it used, the strongest pairs, display options and Export CSV.

#include "explore_tab.h"
#include "../plot/column_picker.h"

#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

class CorrelationView {
public:
    void Render(const ExploreContext& context);
    bool IsRunning() const { return matrix_.Running(); }

private:
    void RenderControls(const ExploreContext& context, const DatasetProfile& profile, const ColumnCorrelations* matrix);
    void RenderHeatmap(const ColumnCorrelations& matrix);
    void RenderPair(const ColumnCorrelations& matrix);

    std::shared_ptr<const DatasetProfile> seen_;
    std::vector<std::string> chosen_;  // empty: the profile's columns
    int pick_i_ = 0, pick_j_ = 1;      // the pair shown beside the matrix
    float cell_ = 56.0f;
    bool show_values_ = true;
    bool show_legend_ = true;
    std::string message_;
    bool message_error_ = false;
    plot::ColumnPicker picker_;
    ExploreJob<ColumnCorrelations> matrix_;
};

}  // namespace cyxwiz
