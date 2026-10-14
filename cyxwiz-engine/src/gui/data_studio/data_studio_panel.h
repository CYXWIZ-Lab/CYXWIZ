#pragma once

// Data Studio (TOFIX148, approved mockup v8): the open graph's data work in
// one place, as seven jobs: Bring in, Explore (Query, Profile, Visualize),
// Prepare, Validate, ML readiness, Publish, Results. A rail lists the data
// (the graph's Data Inputs, node results, published files, other datasets),
// each job shows the graph nodes that do it ("In the graph", Show on
// canvas), and the status bar the last run. Everything here is a node in the
// graph: changes show on the canvas, and runs go through the Data Pipeline
// executor. The model is core/data_studio_presentation.

#include "../../core/data_studio_presentation.h"
#include "query_editor.h"
#include "profile_view.h"
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

    // Opens Explore > Query on a dataset with this SQL in the editor (a
    // dashboard's "Open in Data Studio", the Asset Browser); it runs.
    void ShowQuery(const std::string& dataset_name, const std::string& sql);

    // Query results open in the Table Viewer through this (MainWindow).
    void SetOpenTableCallback(std::function<void(std::shared_ptr<class DataTable>)> cb) {
        if (query_editor_) query_editor_->on_open_table = std::move(cb);
    }

private:
    enum class ExploreTab { Query = 0, Profile = 1, Visualize = 2 };

    std::unique_ptr<QueryEditor> query_editor_;
    std::unique_ptr<ProfileView> profile_;
    std::unique_ptr<Visualizer> visualizer_;

    ::gui::NodeEditor* node_editor_ = nullptr;
    std::function<void()> open_node_editor_callback_;

    std::string active_dataset_;
    bool visible_ = true;
    bool focus_next_ = false;      // opened from elsewhere: bring the window forward
    studio::Job job_ = studio::Job::Explore;
    ExploreTab explore_tab_ = ExploreTab::Query;

    // The shell's view, rebuilt from the graph twice a second.
    studio::View view_;
    double view_time_ = -1.0;
    std::map<int, int> canvas_node_;  // a step of a collapsed recipe -> the recipe (Show on canvas)
    std::vector<studio::GraphNode> graph_nodes_;  // the graph as last read (Turn into SQL step)
    float rail_width_ = 250.0f;

    void RefreshView();
    // The node whose last run made `table` (a name a query used), or null.
    const studio::GraphNode* NodeMaking(const std::string& table) const;
    std::string StepBlocker(const std::vector<std::string>& inputs) const;
    std::string TurnIntoStep(const std::string& sql, const std::vector<std::string>& inputs);
    void ShowOnCanvas(const std::vector<int>& node_ids);
    void RenderHeader();
    void RenderRail(float height);
    void RenderJobTabs();
    void RenderNodeStrip(const studio::JobView& job);
    void RenderJobBody();
    void RenderExplore();
    void RenderBringIn();
    void RenderStatusBar();
};

}  // namespace cyxwiz
