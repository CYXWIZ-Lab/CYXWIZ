#pragma once

// Data Studio > Explore > Projection (TOFIX129 D5, approved mockup board 4;
// replaces the PCA / t-SNE / UMAP window). What the rows look like in two
// dimensions: PCA (with the share of the variation each component keeps and
// what the components are), t-SNE or UMAP, coloured by a column or by the
// groups found in Explore > Groups. t-SNE and UMAP read at most 5,000 rows (a
// seeded sample beyond). Worked out in the background on the device
// (core/groups_explore, core/groups_presentation); t-SNE shows its progress
// and stops on Cancel.

#include "explore_tab.h"
#include "../../core/groups_explore.h"
#include "../plot/column_picker.h"

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

namespace cyxwiz {

class ProjectionView {
public:
    void Render(const ExploreContext& context);
    // Opens on this method (Tools > Clustering and Projection).
    void SelectMethod(groups::ProjectionMethod method) { request_.method = method; }
    bool IsRunning() const { return job_.Running(); }

    // The groups shown in Explore > Groups for a dataset (null: none).
    std::function<const groups::GroupsAnswer*(const std::string& dataset)> groups_shown;
    // Into the graph: the method's node with these settings after the data in
    // use, in its path (`step`) or as a branch; returns what happened.
    std::function<std::string(const groups::ProjectionRequest&, bool step)> add_to_graph;

private:
    struct Progress {
        std::atomic<float> done{0.0f};
        std::mutex mutex;
        std::string text;
    };

    void RenderControls(const DatasetProfile& profile, const groups::GroupsAnswer* shown);
    void RenderRunning();
    void RenderAnswer(const groups::ProjectionAnswer& a, const groups::GroupsAnswer* shown);
    void RenderComponents(const groups::ProjectionAnswer& a);
    void RenderActions(const groups::ProjectionAnswer& a);
    void Export(const groups::ProjectionAnswer& a);

    std::shared_ptr<const DatasetProfile> seen_;
    groups::ProjectionRequest request_;
    std::string asked_key_;    // the question last asked
    std::string stopped_key_;  // the question Cancel stopped (not asked again until a setting changes)
    bool by_groups_ = false;  // colour by the groups (else by request_.colour)
    std::string message_;
    bool message_error_ = false;
    std::shared_ptr<Progress> progress_ = std::make_shared<Progress>();
    plot::ColumnPicker picker_;
    ExploreJob<groups::ProjectionAnswer> job_;
};

}  // namespace cyxwiz
