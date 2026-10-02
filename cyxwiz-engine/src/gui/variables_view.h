#pragma once

// The Variables view (TOFIX133 P5, approved board 9): the Engine's Variable
// Explorer and a notebook's Variables panel both draw this. Values come from
// python_tools/cyxwiz_vars.py on the variables worker, read when a run
// finishes, when the view opens and on Refresh; never polled.

#include "../core/variables_presentation.h"
#include "../scripting/variables_service.h"

#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace scripting {
class ScriptingEngine;
}

namespace cyxwiz {

// The local time as "12:04:31" (read stamps of the views).
std::string ClockNow();

class VariablesView {
public:
    struct Scope {
        std::string key;     // "" = the Python session, else a notebook key
        std::string label;   // "Python session", "nb_board5.ipynb"
        std::string detail;  // "scripts and Console", "notebook"
    };

    VariablesView();
    ~VariablesView();
    VariablesView(const VariablesView&) = delete;
    VariablesView& operator=(const VariablesView&) = delete;

    void SetEngine(scripting::ScriptingEngine* engine) { engine_ = engine; }
    void SetScope(const Scope& scope);
    const std::string& ScopeKey() const { return scope_.key; }
    // A bold title before the header (a notebook's "Variables"); empty: none.
    void SetTitle(const std::string& title) { title_ = title; }

    // Read again the next time the view draws ("after the run finished").
    void Invalidate(const std::string& reason);
    // The namespace was emptied (Restart): forget values and marks.
    void Forget(const std::string& reason);
    // Top-level names of the last read (for a notebook's problem check).
    std::vector<std::string> Names() const;

    // Draws the view filling `height` (0: the rest of the window).
    void Render(float height = 0.0f);

    // Engine panel: the scopes to pick from; unset, no picker.
    std::function<std::vector<Scope>()> scopes;
    // What the Data Viewer shows: a value of a scope, read with these limits.
    struct OpenRequest {
        std::string name;            // "df", "Out[3]"
        std::string path;            // JSON
        Scope scope;
        bool plot = false;           // open its quick plot too
        std::vector<int> index;      // leading-axis indices of an array
        long long max_rows = 200000; // <= 0: every row
    };
    // A value read for the Data Viewer.
    std::function<void(const OpenRequest&, const scripting::VariablesService::Result&)> on_open_table;
    // Insert a name at the cursor of the Script Editor.
    std::function<void(const std::string& name)> on_insert_name;

    // Routes the worker's results to whoever asked; call once per frame.
    static void PollAll(scripting::ScriptingEngine* engine);
    // A read for someone else (the Data Viewer): `done` gets the result on
    // the UI thread; CancelReads(owner) drops the owner's pending ones.
    static std::uint64_t Read(scripting::ScriptingEngine* engine, scripting::VariablesService::Request request,
                              const void* owner, std::function<void(const scripting::VariablesService::Result&)> done);
    static void CancelReads(const void* owner);

private:
    using Kind = scripting::VariablesService::Kind;
    struct Pending {
        Kind kind = Kind::List;
        std::string path;  // JSON
        std::string name;
        std::string scope;
        bool plot = false;
    };

    // Request id -> who asked and what to do with the result.
    struct Route {
        const void* owner = nullptr;
        std::function<void(const scripting::VariablesService::Result&)> done;
    };
    static std::map<std::uint64_t, Route>& Routes();
    void Submit(scripting::VariablesService::Request request, Pending pending);
    void OnResult(const scripting::VariablesService::Result& result, const Pending& pending);
    void ReadIfNeeded();
    void RenderHeader();
    void RenderChips();
    void RenderTable(float height);
    void RenderFooter();
    void RenderRowMenu(const vars::TreeRow& row);
    void HandleKeys(const std::vector<vars::TreeRow>& rows);
    void ViewData(const std::string& path, const std::string& name, bool plot = false);
    void CopyValue(const std::string& path);
    void SaveCsv(const std::string& path, const std::string& name);
    void Delete(const std::string& path, const std::string& name);
    void ToggleOpen(const std::string& path);
    void Note(const std::string& text);

    scripting::ScriptingEngine* engine_ = nullptr;
    Scope scope_{"", "Python session", "scripts and Console"};
    std::string title_;

    std::vector<vars::Variable> vars_;
    std::map<std::string, std::vector<vars::Variable>> children_;
    std::set<std::string> open_;
    std::set<std::string> changed_;
    std::map<std::string, std::map<std::string, std::string>> digests_;  // per scope
    bool have_read_ = false;

    bool read_wanted_ = true;
    std::string read_reason_ = "on opening";
    std::uint64_t list_request_ = 0;
    bool was_running_ = false;
    int last_frame_ = -10;
    std::string status_;
    std::string note_;
    double note_until_ = 0.0;

    char filter_[128] = {};
    vars::Chip chip_ = vars::Chip::All;
    bool show_all_ = false;
    vars::Column sort_column_ = vars::Column::Name;
    bool sort_ascending_ = true;
    std::string selected_;  // path JSON
    std::string menu_path_;
};

}  // namespace cyxwiz
