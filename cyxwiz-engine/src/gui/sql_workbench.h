#pragma once

// SQL Workbench (TOFIX146, mockup v5 board 2): the SQL step's node dialog,
// a dockable window opened from Properties ("Open in SQL Workbench", Open
// Dialog) or by double-clicking the step. It edits a draft of the step's
// query and runs it on the step's inputs already in memory (read-only
// DuckDB, the step's own contract), without touching the graph until "Save
// to step".

#include "../core/sql_plan.h"
#include "../core/sql_step_presentation.h"
#include "../core/table_profile.h"
#include "../data/arrow_data_table.h"
#include "code_editor.h"
#include "node_config_dialog.h"
#include "node_editor.h"

#include <arrow/api.h>

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace gui {

class SqlWorkbenchDialog : public NodeConfigDialog {
public:
    explicit SqlWorkbenchDialog(MLNode* node);
    ~SqlWorkbenchDialog() override;

    void Apply() override;  // save the draft to the step
    void Reset() override;  // back to the step's saved query
    ImVec2 GetDefaultSize() const override { return ImVec2(1600, 1000); }
    bool UsesStandardButtons() const override { return false; }
    bool IsBusy() const override { return run_ != nullptr; }

    // Opens a registered dataset in the Table Viewer (same callback as
    // Properties).
    using OpenTableCallback = std::function<void(const std::string& dataset_name, const std::string& title)>;
    void SetOpenTableCallback(OpenTableCallback callback) { open_table_ = std::move(callback); }

protected:
    void RenderContent() override;

private:
    struct Run;  // one background DuckDB run
    enum class RunKind { Step, Selection, ToPart, Counts, Removed, Check, Explain, Profile };
    enum BottomTab { kResults, kProfile, kRemoved, kChecks, kPlan, kMessages, kBottomTabs };
    struct Message {
        std::string time;
        std::string text;
        bool error = false;
    };
    struct InputTable {
        std::string alias;
        std::string source;  // "Data Input \"Tier 1 books\""
        int node_id = -1;
        std::shared_ptr<arrow::Table> table;  // null when not in memory
    };
    struct HistoryEntry {
        std::string time;
        std::string label;
        std::string outcome;  // "15,349 rows, 2.1 s" or the error
        bool error = false;
        std::string draft;    // the query text it ran
    };
    struct CheckOutcome {
        std::string name;
        std::string rule;
        bool passed = false;
        std::string detail;
    };

    MLNode* CurrentNode();
    void RefreshInputs();
    void StartRun(RunKind kind, size_t part_index = 0);
    void StopRun();
    void PollRun();
    void AddMessage(std::string text, bool error);
    void MarkPart(size_t index);
    void MarkError(const std::string& error);
    void EvaluateChecks();
    void ExportResult(bool parquet);

    void RenderHeader(MLNode& node);
    void RenderToolbar();
    void RenderSidePanel(float height);
    void RenderInputsTab();
    void RenderPartsTab();
    void RenderHistoryTab();
    void RenderSnippetsTab();
    void RenderEditor(float height);
    void RenderBottomPanel(float height);
    void RenderResults();
    void RenderProfile();
    void RenderRemoved();
    void RenderChecks();
    // Plan tab (board 4), gui/sql_workbench_plan.cpp.
    void RenderPlan();
    void RenderPlanSummary();
    void RenderPlanTimeBar();
    void RenderPlanTree();
    void RenderPlanDiagram();
    void RenderPlanText();
    void RenderPlanDetails();
    int PartIndex(const std::string& name) const;
    void RenderMessages();
    void RenderStatusBar();

    int node_id_ = -1;
    std::string step_name_;
    cyxwiz::CodeEditor editor_;
    std::string saved_query_;           // the step's query when last loaded or saved
    uint64_t outline_version_ = ~0ull;  // editor document version the outline was built from
    cyxwiz::sqlstep::SqlOutline outline_;
    std::string draft_;                 // editor text at outline_version_

    NodeEditor::DataStepContext context_;
    std::vector<InputTable> inputs_;
    double inputs_time_ = -1.0;
    bool pipeline_was_running_ = false;

    std::unique_ptr<Run> run_;
    RunKind running_kind_ = RunKind::Step;  // of run_, while it runs
    double running_since_ = 0.0;            // ImGui time run_ started
    std::shared_ptr<arrow::Table> result_;
    cyxwiz::ArrowPreview result_preview_;
    std::vector<cyxwiz::ArrowColumnProfile> result_profile_;
    std::string result_label_;  // "result", "part kept", "rows kept removed"
    bool result_is_step_ = false;  // the whole draft's result (checks apply)
    double result_seconds_ = 0.0;
    std::string result_dataset_;  // registry name of the shown result

    // Rows after each part, for the draft they were counted on.
    std::map<std::string, int64_t> part_rows_;
    std::string part_rows_draft_;
    double part_rows_seconds_ = 0.0;

    std::vector<CheckOutcome> check_outcomes_;
    cyxwiz::sqlplan::Plan plan_;
    std::string plan_text_;   // DuckDB's own text drawing of the plan
    std::string plan_json_;   // the JSON the plan was read from (Export)
    std::string plan_draft_;  // the editor text the plan was made for
    std::string plan_when_;   // clock time of the plan
    int plan_view_ = 0;       // 0 Tree, 1 Diagram, 2 Text
    int plan_picked_ = -1;    // selected operator
    std::string syntax_state_;  // "Syntax OK", "Error on line 57", ...
    bool syntax_error_ = false;

    std::vector<Message> messages_;
    std::vector<HistoryEntry> history_;
    int side_tab_ = 0;    // 0 Inputs, 1 Parts, 2 History, 3 Snippets
    int bottom_tab_ = kResults;
    // Share of the body the editor takes; the splitter below it moves it.
    // A plan makes room for itself once, unless the user has dragged.
    float editor_share_ = 0.55f;
    bool editor_share_set_by_user_ = false;
    int marked_part_ = -1;
    OpenTableCallback open_table_;
};

}  // namespace gui
