#pragma once

// Data Studio > Prepare (TOFIX148 step 3, approved mockup v8, D1): the
// graph's Preparation Recipe as a step list; an SQL step opens in the SQL
// Workbench's editor pane in place (Save to step writes the node, with an undo
// point, also inside a collapsed recipe); Run step runs the Data Pipeline
// through the step and its checks; its result shows in the table grid with
// the rows each part of the query keeps. A guided (form) step is edited in
// place in Guided mode with the Workbench's guided editor (D3: the guided
// operations stay the form UI; none of them has a canonical operator to
// lower to, checks aside, which D4 moves to Check nodes). Other steps
// show their settings and open in their dialog.

#include "../../core/data_studio_presentation.h"
#include "../guided_editor.h"
#include "../sql_editor_pane.h"
#include "../table_grid.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace gui {
class NodeEditor;
}

namespace cyxwiz {

class PrepareView {
public:
    PrepareView();
    ~PrepareView();

    // The graph as Data Studio last read it (twice a second).
    void SetGraph(::gui::NodeEditor* editor, const std::vector<studio::GraphNode>& nodes, const std::vector<int>& prepare_nodes);
    void Render();
    // Shows this step (a rail double-click, the node strip).
    void Select(int step_id);

    std::function<void()> raise_canvas;              // brings CyxWiz Studio forward
    std::function<void(const std::vector<int>&)> show_on_canvas;

private:
    struct PartCounts;  // the background count of rows per part

    void RenderSteps(float height);
    void RenderSqlStep(const studio::GraphNode& node, float height);
    void RenderGuided(float height);
    void RenderOtherStep(const studio::GraphNode& node);
    void RenderResult(const studio::GraphNode& node);
    void LoadStep(int step_id);
    void SaveDraft();
    void RunStep(int step_id);
    void OpenInDialog(int step_id);
    // Adds a node after a step (on the canvas): an SQL step (guided or not) or another step node.
    void AddStepAfter(int step_id, ::gui::NodeType type, bool guided = false);
    void RenderAddStepMenu(int step_id);  // the "Add a step" menu's items
    void StartPartCounts(const studio::GraphNode& node);
    // The step's inputs by the names its query uses, with their tables when in memory.
    std::vector<::gui::GuidedEditor::Input> StepInputs(int step_id) const;
    void OnGuidedChanged();  // the operations' SQL into the editor (read-only)
    const studio::GraphNode* NodeById(int id) const;
    int RecipeOf(int step_id) const;  // the recipe holding a step (-1: none)

    ::gui::NodeEditor* editor_ = nullptr;
    std::vector<studio::GraphNode> nodes_;
    std::vector<int> recipes_;       // Preparation Recipe nodes
    std::vector<int> loose_;         // Prepare's nodes outside any recipe
    int recipe_ = -1;                // the recipe shown (-1: the loose steps)
    std::vector<studio::StepRow> steps_;
    int selected_ = -1;
    float list_width_ = 270.0f;

    ::gui::SqlEditorPane pane_{"data_studio_prepare"};
    int loaded_ = -1;                // the step in the editor
    std::string saved_query_;        // its query when loaded or saved
    ::gui::GuidedEditor guided_;     // Active(): a guided step, its SQL written from its operations
    int mode_ = 1;                   // 0 Guided, 1 SQL
    bool confirm_start_guided_ = false;
    bool confirm_edit_as_sql_ = false;
    std::string reads_;              // the step's inputs ("Tier 1 books (ZIP text)")
    size_t checks_after_ = 0;        // Check nodes after the step
    std::string message_;
    bool message_error_ = false;

    TableGrid grid_;
    std::string grid_key_;           // dataset + finish time shown
    std::shared_ptr<PartCounts> parts_;
};

}  // namespace cyxwiz
