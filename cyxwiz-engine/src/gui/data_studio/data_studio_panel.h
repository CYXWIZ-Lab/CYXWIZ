#pragma once

// Data Studio (TOFIX148, approved mockup v8): the open graph's data work in
// one place, as seven jobs: Bring in, Explore (Query, Profile, Missing, Outliers, Correlation,
// Distribution, Visualize),
// Prepare, Validate, ML readiness, Publish, Results. A rail lists the data
// (the graph's Data Inputs, node results, published files, other datasets),
// each job shows the graph nodes that do it ("In the graph", Show on
// canvas), and the status bar the last run. Everything here is a node in the
// graph: changes show on the canvas, and runs go through the Data Pipeline
// executor. The model is core/data_studio_presentation.

#include "../../core/data_studio_presentation.h"
#include "query_editor.h"
#include "profile_view.h"
#include "missing_view.h"
#include "outliers_view.h"
#include "correlation_view.h"
#include "distribution_view.h"
#include "prepare_view.h"
#include "readiness_view.h"
#include "validate_view.h"
#include "publish_view.h"
#include "results_view.h"
#include "visualizer.h"
#include <imgui.h>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace gui {
class NodeEditor;
}

namespace cyxwiz {

class DataStudioPanel {
public:
    // Explore's tabs, in the order they show.
    enum class ExploreTab { Query, Profile, Missing, Outliers, Correlation, Distribution, Visualize };

    DataStudioPanel();
    ~DataStudioPanel() = default;

    void Render();

    // The dataset Explore shows (a registry name).
    void SetActiveDataset(const std::string& dataset_name);
    const std::string& GetActiveDataset() const { return active_dataset_; }

    bool IsVisible() const { return visible_; }
    void SetVisible(bool visible) { visible_ = visible; }
    bool* GetVisiblePtr() { return &visible_; }

    // The graph the jobs read (MainWindow's Node Editor).
    void SetNodeEditor(::gui::NodeEditor* editor) { node_editor_ = editor; }

    // Show graph / Show on canvas raise CyxWiz Studio through this.
    void SetOpenNodeEditorCallback(std::function<void()> cb) { open_node_editor_callback_ = std::move(cb); }

    // Opens Explore > Profile on a dataset (a dashboard's "Edit roles").
    void ShowProfile(const std::string& dataset_name);

    // Opens an Explore tab on a dataset, at a column when given (Data >
    // Explore, Profile's missing counts).
    void ShowExplore(const std::string& dataset_name, ExploreTab tab, const std::string& column = {});

    // Opens Explore > Query on a dataset with this SQL in the editor (a
    // dashboard's "Open in Data Studio", the Asset Browser); it runs.
    void ShowQuery(const std::string& dataset_name, const std::string& sql);

    // Results > Test run's "Open Test Results" and a kept test's "Open"
    // (MainWindow shows the panel).
    void SetOpenTestResultsCallback(std::function<void()> cb) {
        if (results_) results_->on_open_test_results = std::move(cb);
    }
    void SetOpenKeptTestCallback(std::function<void(const std::string&)> cb) {
        if (results_) results_->on_open_kept = std::move(cb);
    }

    // Query results open in the Table Viewer through this (MainWindow): the
    // result and its tab's name.
    void SetOpenTableCallback(std::function<void(std::shared_ptr<arrow::Table>, const std::string&)> cb) {
        if (query_editor_) query_editor_->on_open_table = std::move(cb);
    }

private:
    std::unique_ptr<QueryEditor> query_editor_;
    std::unique_ptr<ProfileView> profile_;
    std::unique_ptr<MissingView> missing_;
    std::unique_ptr<OutliersView> outliers_;
    std::unique_ptr<CorrelationView> correlation_;
    std::unique_ptr<DistributionView> distribution_;
    std::unique_ptr<PrepareView> prepare_;
    std::unique_ptr<ValidateView> validate_;
    std::unique_ptr<PublishView> publish_;
    std::unique_ptr<ResultsView> results_;
    std::unique_ptr<ReadinessView> readiness_;
    std::unique_ptr<Visualizer> visualizer_;

    ::gui::NodeEditor* node_editor_ = nullptr;
    std::function<void()> open_node_editor_callback_;

    std::string active_dataset_;
    bool visible_ = true;
    studio::Job job_ = studio::Job::Explore;
    ExploreTab explore_tab_ = ExploreTab::Query;

    // The shell's view, rebuilt from the graph twice a second.
    studio::View view_;
    double view_time_ = -1.0;
    std::map<int, int> canvas_node_;  // a step of a collapsed recipe -> the recipe (Show on canvas)
    std::vector<studio::GraphNode> graph_nodes_;  // the graph as last read (Turn into SQL step)
    float rail_width_ = 250.0f;
    // Narrower than the rail plus this, the rail stacks above the job.
    static constexpr float kMinJobWidth = 420.0f;

    void RefreshView();
    // The node whose last run made `table` (a name a query used), or null.
    const studio::GraphNode* NodeMaking(const std::string& table) const;
    // The node whose table `dataset` is: a node's last result or a Data Input's dataset (null: none).
    const studio::GraphNode* NodeOfDataset(const std::string& dataset) const;
    std::string StepBlocker(const std::vector<std::string>& inputs) const;
    std::string TurnIntoStep(const std::string& sql, const std::vector<std::string>& inputs);
    void ShowOnCanvas(const std::vector<int>& node_ids);
    void RenderHeader();
    // Beside the job (height = the body's), or stacked above it in a narrow
    // panel (height 0: as tall as its lists, at most max_height).
    void RenderRail(float height, float max_height = 0.0f);
    void RenderJobTabs();
    void RenderNodeStrip(const studio::JobView& job);
    void RenderJobBody();
    void RenderExplore();
    ExploreContext MakeExploreContext() const;
    // Missing's "Drop these rows in Prepare": a guided Keep rows step after the dataset's node.
    std::string AddKeepRowsStep(const std::string& column, const std::vector<std::string>& texts);
    void RenderBringIn();
    void RenderStatusBar();
};

}  // namespace cyxwiz
