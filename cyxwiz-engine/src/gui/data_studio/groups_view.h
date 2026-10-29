#pragma once

// Data Studio > Explore > Groups (TOFIX129 D5, approved mockup boards 1-3;
// replaces the K-Means, DBSCAN, Hierarchical, Gaussian Mixture and Cluster
// Evaluation windows). Which rows group together: K-Means, DBSCAN,
// hierarchical or a Gaussian mixture on the chosen number columns, with how
// good the groups are, what each group is, how they compare with a column,
// and how many groups to pick. Worked out in the background on the device
// (core/groups_explore, core/groups_presentation).

#include "explore_tab.h"
#include "../../core/groups_explore.h"
#include "../plot/column_picker.h"

#include <functional>
#include <memory>
#include <string>

namespace cyxwiz {

class GroupsView {
public:
    void Render(const ExploreContext& context);
    // Opens on this method (Tools > Clustering and Projection).
    void SelectMethod(groups::Method method) { request_.method = method; }
    bool IsRunning() const { return job_.Running() || how_many_job_.Running(); }
    // The groups shown, for the dataset (Projection colours by them); null when none.
    const groups::GroupsAnswer* Shown(const std::string& dataset) const;

    // Into the graph: the method's node with these settings after the data in
    // use, in its path (`step`) or as a branch; returns what happened.
    std::function<std::string(const groups::GroupsRequest&, bool step)> add_to_graph;

private:
    void RenderControls(const DatasetProfile& profile);
    void RenderMethodSettings();
    void RenderAnswer(const groups::GroupsAnswer& a, const DatasetProfile& profile);
    void RenderProfile(const groups::GroupsAnswer& a);
    void RenderCompare(const groups::GroupsAnswer& a, const DatasetProfile& profile);
    void RenderActions(const groups::GroupsAnswer& a);
    void RenderHowMany(const ExploreContext& context);
    void Export(const groups::GroupsAnswer& a);

    std::shared_ptr<const DatasetProfile> seen_;
    std::string dataset_;
    groups::GroupsRequest request_;
    std::string key_;  // the question asked
    bool how_many_open_ = false;
    int from_ = 2, to_ = 8;
    std::string message_;
    bool message_error_ = false;
    plot::ColumnPicker picker_;
    ExploreJob<groups::GroupsAnswer> job_;
    ExploreJob<groups::HowManyAnswer> how_many_job_;
};

}  // namespace cyxwiz
