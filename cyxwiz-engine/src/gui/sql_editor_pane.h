#pragma once

// SQL editor pane (TOFIX148 step 2): the SQL Workbench's editor as a
// component, so the Workbench, Data Studio's Query tab and the Data Input
// database query box share one editor. It owns the code editor, the find bar
// (Ctrl+F), completion (typing or Ctrl+Space) and the query's outline.

#include "../core/sql_editing.h"
#include "../core/sql_step_presentation.h"
#include "code_editor.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace gui {

class SqlEditorPane {
public:
    // `id` keeps the editor and its completion window apart from other panes.
    explicit SqlEditorPane(std::string id);

    cyxwiz::CodeEditor& Editor() { return editor_; }
    const cyxwiz::CodeEditor& Editor() const { return editor_; }

    // Re-reads the text and outline when the document changed; call once a
    // frame before Text() or Outline().
    void Refresh();
    const std::string& Text() const { return text_; }
    const cyxwiz::sqlstep::SqlOutline& Outline() const { return outline_; }

    // Column names offered by completion, besides keywords, functions and the
    // query's own part names.
    void SetCompletionColumns(std::vector<std::string> columns) { columns_ = std::move(columns); }

    // The find bar (when open), the editor and the completion list, `height`
    // pixels tall in all.
    void Render(float height);
    void OpenFind();

    // Keywords upper case, trailing spaces removed, as one undoable edit.
    // False when the text is already formatted.
    bool Format();

private:
    void CloseFind();
    void UpdateFind();
    void FindStep(bool forward);
    void RenderFindBar();
    void UpdateCompletion(bool forced);
    void AcceptCompletion();
    void RenderCompletion();

    std::string id_;
    cyxwiz::CodeEditor editor_;
    uint64_t outline_version_ = ~0ull;  // document version text_ and outline_ were read at
    std::string text_;
    cyxwiz::sqlstep::SqlOutline outline_;
    std::vector<std::string> columns_;
    float line_height_ = 18.0f;

    struct FindState {
        bool open = false;
        bool focus = false;
        char query[256] = {};
        bool case_sensitive = false;
        bool whole_word = false;
        bool regex = false;
        uint64_t version = ~0ull;
        std::string key;
        std::string error;
        std::vector<std::pair<cyxwiz::editor::Pos, cyxwiz::editor::Pos>> matches;
        int current = -1;
    } find_;
    bool completion_open_ = false;
    std::vector<cyxwiz::sqledit::Completion> completion_items_;
    int completion_selected_ = 0;
    cyxwiz::editor::Pos completion_start_;
    uint64_t completion_version_ = ~0ull;
};

}  // namespace gui
