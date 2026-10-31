#pragma once

// Data Studio Profile tab (TOFIX134 P3.5, approved board 14; replaces
// Analyze): one profile of the picked dataset (core/dataset_profiler) and the
// column roles. Roles are edited here only (owner answer 2): contract (the
// graph) > roles set here (saved with the project) > inferred. Profiling runs
// in the background when a dataset is picked or re-loaded.

#include "../../core/dataset_contract.h"
#include "../../core/dataset_profiler.h"

#include <cstdint>
#include <functional>
#include <map>
#include <vector>
#include <memory>
#include <string>

namespace cyxwiz {

class ImageProfileView;
class LiveGraphCompile;

class ProfileView {
public:
    ProfileView();
    ~ProfileView();

    void Render();
    void SetActiveDataset(const std::string& dataset_name);
    // Off where the view is embedded (the Data Input dialog): roles and
    // missing-value texts are set in Data Studio's Profile only.
    void SetRolesEditable(bool editable) { roles_editable_ = editable; }
    // The canvas graph's compile: an image dataset's profile shows how its Resize
    // treats the images and which prepared set it trains from.
    void SetLiveCompile(const LiveGraphCompile* live_compile) { live_compile_ = live_compile; }
    // Passed to the image profile: writes a node's settings with an undo point
    // (Use in Normalize). Unset where the view is embedded.
    using NodeWriter = std::function<bool(int node_id, const std::map<std::string, std::string>& values)>;
    void SetNodeWriter(NodeWriter writer) { node_writer_ = std::move(writer); }
    // Passed to the image profile: selects a node on the canvas.
    void SetShowNode(std::function<void(int node_id)> show) { show_node_ = std::move(show); }
    void Reprofile();
    bool IsRunning() const { return task_id_ != 0; }
    float Progress() const { return progress_; }

    // What the other Explore tabs read (TOFIX129 D1): the profile (null while
    // profiling), the dataset as the graph calls it, the error, and the texts
    // marked as missing per column (project settings).
    std::shared_ptr<const DatasetProfile> Current() const { return profile_; }
    const std::string& Shown() const { return shown_.empty() ? dataset_ : shown_; }
    const std::string& Error() const { return error_; }
    std::map<std::string, std::vector<std::string>> MissingTexts() const;
    // Shows this column's detail (Missing's "set them in Profile").
    void SelectColumn(const std::string& column);

    // Links into the other Explore tabs (TOFIX129 D1): a column's missing
    // and outlier counts, "More about this column", "See the matrix". Unset
    // where the view is embedded (the Data Input dialog): plain figures.
    std::function<void(const std::string& column)> on_missing;
    std::function<void(const std::string& column)> on_outliers;
    std::function<void(const std::string& column)> on_distribution;
    std::function<void()> on_correlation;

    // The graph's Data Input node this table is (TOFIX129 D7a G1), -1 for a
    // table that is not one: the roles set here are written into it
    // (`column_roles`, through the node writer) so training follows them.
    // `node_roles` is what the node holds now; roles saved with the project
    // before the node had them are written as soon as they differ.
    void SetDataInputNode(int node_id, const std::string& node_roles);
    // "Make them empty in Prepare" under Treat as missing: adds a step that
    // turns the column's missing texts into empty cells; returns what happened.
    std::function<std::string(const std::string& column, const std::vector<std::string>& texts)> on_make_empty;

private:
    void RenderOverview();
    void RenderColumns(float width);
    void RenderDetail();
    void RebuildContract();
    std::string SourceKey() const;
    // Writes this table's user roles into its Data Input node (nothing when
    // the table is not one, or no node writer is set).
    void SyncRolesToNode();
    int data_input_node_ = -1;
    std::string node_roles_;        // the node's column_roles as last seen or written
    std::string make_empty_message_;

    std::string dataset_;           // catalog name
    std::string shown_;             // as the graph calls it
    std::string source_path_;
    std::string target_column_;     // the graph's target
    uint64_t generation_ = 0;       // of the data profiled
    std::shared_ptr<DatasetProfile> profile_;
    DatasetContract contract_;
    uint64_t task_id_ = 0;
    float progress_ = 0;
    std::string progress_text_;
    std::string error_;
    std::string profiled_at_;
    int selected_ = -1;
    std::string pending_select_;    // SelectColumn before the profile arrived
    bool roles_editable_ = true;
    char missing_buf_[256] = {};
    std::string missing_for_;
    std::string save_error_;
    // An image dataset has its own profile (no columns to profile).
    std::unique_ptr<ImageProfileView> image_;
    bool images_ = false;
    const LiveGraphCompile* live_compile_ = nullptr;
    NodeWriter node_writer_;
    std::function<void(int)> show_node_;
    std::shared_ptr<int> alive_ = std::make_shared<int>(0);
};

}  // namespace cyxwiz
