#pragma once

// Data Studio > Validate (TOFIX148 D4, approved mockup v8): the graph's
// checks as a table (the check node, what it reads, its rule, its last run)
// with Run checks, and "Add a check" with the five kinds of the one check
// model (row count, allowed values, unique, not empty, range) or a Data
// Validator. A check is a Row Count Check node added after the step it
// checks; Edit changes that node. The model is core/data_studio_checks.

#include "../../core/data_studio_checks.h"

#include <functional>
#include <string>
#include <vector>

namespace gui {
class NodeEditor;
}

namespace cyxwiz {

class ValidateView {
public:
    // The graph as Data Studio last read it (checks carry their parameters
    // and what feeds them); `active_dataset` is the rail's choice;
    // `pipeline_error` why the graph cannot run ("" when it can).
    void SetGraph(::gui::NodeEditor* editor, const std::vector<studio::GraphNode>& nodes, const std::vector<int>& recipes,
                  const std::string& active_dataset, const std::string& pipeline_error);
    void Render();
    bool IsRunning() const;

    std::function<void(const std::vector<int>&)> show_on_canvas;

private:
    struct Target {
        int id = -1;
        std::string name;
    };

    void RenderTable();
    void RenderAddRow();
    void RenderForm();
    void ColumnField(const char* id, std::string& value, float width);
    // Several columns: checkboxes when the columns are known, else typed (a comma list).
    void ColumnSetField(const char* id, std::string& list, const char* hint);
    void RenderValidatorForm();
    void OpenForm(studio::CheckKind kind);
    void OpenValidatorForm();
    void EditCheck(const studio::CheckRow& row);
    void SubmitForm();
    void SubmitValidator();
    // The node a new validator reads: the target, or its recipe's result.
    int ValidatorSource(int target, std::string* what) const;
    void RunChecks();
    void LoadColumns();
    int RecipeOf(int step_id) const;  // the recipe holding a step (-1: none)

    ::gui::NodeEditor* editor_ = nullptr;
    std::vector<studio::GraphNode> nodes_;
    std::vector<int> recipes_;
    std::vector<studio::CheckRow> rows_;
    std::vector<Target> targets_;      // the steps a check can follow (Data Inputs, Prepare steps)
    int default_target_ = -1;
    std::string pipeline_error_;

    bool form_open_ = false;
    int editing_ = -1;                 // the check node being edited (-1: a new check)
    int target_ = -1;                  // the step a new check follows
    studio::CheckForm form_;
    bool validator_form_ = false;      // the form edits a Data Validator
    studio::ValidatorForm vform_;
    std::string form_error_;
    std::vector<std::string> columns_; // the target's columns, from its last run
    std::string columns_key_;

    std::string message_;
    bool message_error_ = false;
};

}  // namespace cyxwiz
