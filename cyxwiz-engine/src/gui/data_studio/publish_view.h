#pragma once

// Data Studio > Publish (TOFIX148, approved mockup v8): the graph's outputs
// (Export CSV / JSON / Parquet, Data Output) as a table (the node, what it
// reads, the file it writes, its last write) and, for the selected one, the
// lineage written beside its file (core/dataset_lineage: recipe revision,
// input digests, result digest, checks). Publish... sets an output's name
// and location, or adds a Parquet export after a step, and runs it; the
// executor writes the file and its lineage. The model is
// core/data_studio_publish.

#include "../../core/data_studio_publish.h"
#include "../../core/dataset_lineage.h"

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace gui {
class NodeEditor;
}

namespace cyxwiz {

class PublishView {
public:
    // The graph as Data Studio last read it (outputs carry their parameters,
    // what feeds them and the file their last run wrote).
    void SetGraph(::gui::NodeEditor* editor, const std::vector<studio::GraphNode>& nodes, const std::string& active_dataset);
    void Render();
    // Opens the Publish dialog (the header's Publish...).
    void OpenPublish();
    bool IsRunning() const;

    std::function<void(const std::vector<int>&)> show_on_canvas;

private:
    struct Target {
        int id = -1;
        std::string name;
    };

    void RenderTable();
    void RenderLineage();
    void RenderDialog();
    void Publish();
    void AddDataOutput();
    void LoadLineage(const std::string& path);

    ::gui::NodeEditor* editor_ = nullptr;
    std::vector<studio::GraphNode> nodes_;
    std::vector<studio::OutputRow> rows_;
    std::vector<Target> targets_;      // the steps an output can read (Data Inputs, Prepare steps)
    int default_target_ = -1;
    int selected_ = -1;                // the output whose lineage shows

    // The selected output's lineage, read again when its file changes.
    std::string lineage_path_;
    std::filesystem::file_time_type lineage_time_{};
    bool lineage_ok_ = false;
    DatasetLineage lineage_;
    std::string lineage_error_;

    // The Publish dialog.
    bool open_dialog_ = false;
    int dialog_output_ = -1;           // the output it publishes through (-1: a new Parquet export)
    int dialog_target_ = -1;           // a new export reads this step
    std::string dialog_name_;
    std::string dialog_path_;
    std::string dialog_error_;

    // A new export added by Publish runs once it is on the canvas.
    std::string pending_name_;
    std::vector<int> known_outputs_;

    std::string message_;
    bool message_error_ = false;
};

}  // namespace cyxwiz
