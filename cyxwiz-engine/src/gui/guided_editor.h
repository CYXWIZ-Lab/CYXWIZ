#pragma once

// The guided step editor (TOFIX146 step 5, mockup v5 board 3): the operation
// library, the operation cards and the picked operation's fields, over one
// guided step (core/guided_sql). The SQL Workbench's Guided mode and Data
// Studio > Prepare (TOFIX148 D3) both host it; the host owns the step's
// query, runs and saving, and hears of every change through on_changed.

#include "../core/guided_sql.h"
#include "ui_buttons.h"

#include <arrow/api.h>
#include <imgui.h>

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace gui {

class GuidedEditor {
public:
    struct Input {
        std::string alias;
        std::shared_ptr<arrow::Table> table;  // null when not loaded; columns only for a table on disk
        std::string parquet_path;             // a table kept on disk (TOFIX154)
        int64_t rows = 0;
    };
    // What the cards show beside each operation.
    struct CardFacts {
        const std::map<std::string, int64_t>* rows = nullptr;  // part name -> rows, when counted for this SQL
        int64_t input_rows = -1;                                // rows the first operation receives
        size_t checks_after = 0;                                // Check nodes after the step
    };

    // Reads a step's guided_ops ("": an SQL step). False, with `error`, when
    // the text cannot be read; the step then counts as an SQL step.
    bool Load(const std::string& guided_ops, std::string* error = nullptr);
    // A new guided step with no operations (it replaces an SQL step on save).
    void Start();
    // Edit as SQL: the step stops being a guided step.
    void Stop() { active_ = false; }
    bool Active() const { return active_; }

    // The step's inputs in pin order. Writes the SQL again when their
    // columns change (they load after the step opens); true when it did.
    bool SetInputs(std::vector<Input> inputs);
    const std::vector<Input>& Inputs() const { return inputs_; }

    const cyxwiz::guided::GuidedStep& Step() const { return step_; }
    const cyxwiz::guided::Generated& Generated() const { return generated_; }
    std::string Json() const { return cyxwiz::guided::ToJson(step_); }
    // The operations differ from the ones loaded or last saved.
    bool Edited() const;
    void MarkSaved() { saved_json_ = active_ ? Json() : std::string(); }
    // Saving: the last "Add checks" operation leaves the step and its checks
    // come back as Check node settings (TOFIX148 D4); the SQL is written
    // again without it (on_changed). Undo brings the operation back.
    std::vector<cyxwiz::guided::CheckNode> TakeChecks();
    int Picked() const { return picked_; }
    // Rows after operation `op`, or -1 when `rows` does not have its part.
    int64_t RowsAfter(size_t op, const std::map<std::string, int64_t>& rows) const;

    bool CanUndo() const { return !undo_.empty(); }
    bool CanRedo() const { return !redo_.empty(); }
    void Undo();
    void Redo();

    void RenderTemplateButton(cyxwiz::ui::ButtonSize size = cyxwiz::ui::ButtonSize::Regular);  // and its menu
    // The library as a menu, for a narrow host (Prepare).
    void RenderAddMenu(cyxwiz::ui::ButtonSize size = cyxwiz::ui::ButtonSize::Regular);
    void RenderLibrary(const ImVec2& size);
    void RenderCards(const ImVec2& size, const CardFacts& facts);
    void RenderEditor();  // the picked operation's fields

    // After every change of the operations, with the new SQL written.
    std::function<void()> on_changed;
    // A run is in progress: "Find values" waits for it.
    bool busy = false;

private:
    cyxwiz::guided::Context MakeContext() const;
    void Regenerate();
    void Changed(const std::string& before_json);  // undo point, regenerate, on_changed
    bool RenderFields(cyxwiz::guided::Operation& op, const cyxwiz::guided::Columns& in);
    void AddOp(const cyxwiz::guided::OpInfo& op);  // after the picked operation
    bool Matches(const cyxwiz::guided::OpInfo& op) const;  // the search finds it

    bool active_ = false;
    cyxwiz::guided::GuidedStep step_;
    cyxwiz::guided::Generated generated_;
    std::vector<cyxwiz::guided::Columns> columns_;
    std::string saved_json_;  // guided_ops as last loaded or saved
    std::vector<std::string> undo_;
    std::vector<std::string> redo_;
    int picked_ = -1;
    char search_[64] = {};
    std::vector<Input> inputs_;
    std::string inputs_key_;  // inputs the SQL was last written for
};

}  // namespace gui
