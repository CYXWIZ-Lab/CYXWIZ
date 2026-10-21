#pragma once

// Data Studio > Explore > Tests (TOFIX129 D3, approved mockup boards 1-3;
// replaces the Hypothesis Testing window). Pick a question (two groups,
// paired columns, several groups, one column vs a value, two categories) and
// its columns; the answer comes with the groups, the figures, a sentence that
// says what it means, and the checks behind it. Worked out in the background
// on the device (core/stats_explore, core/stats_presentation).

#include "explore_tab.h"
#include "../../core/stats_explore.h"
#include "../plot/column_picker.h"

#include <functional>
#include <memory>
#include <string>

namespace cyxwiz {

class TestsView {
public:
    void Render(const ExploreContext& context);
    // Starts on this column (Tools > Statistics, a column's links).
    void Select(const std::string& column) { request_.values = column; }
    bool IsRunning() const { return job_.Running(); }

    // Keep as a node: the Hypothesis Test node with these settings, after the data in use.
    std::function<void(const stats::TestRequest&)> keep_as_node;

private:
    void RenderControls(const DatasetProfile& profile);
    void RenderGroups(const stats::TestAnswer& a);
    void RenderAnswer(const stats::TestAnswer& a);
    void RenderBoxes(const stats::TestAnswer& a, float width);
    void Export(const stats::TestAnswer& a);

    std::shared_ptr<const DatasetProfile> seen_;
    stats::TestRequest request_;
    std::string message_;
    bool message_error_ = false;
    plot::ColumnPicker picker_;
    ExploreJob<stats::TestAnswer> job_;
};

}  // namespace cyxwiz
