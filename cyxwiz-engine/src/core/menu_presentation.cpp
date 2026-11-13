#include "menu_presentation.h"

#include <algorithm>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz::menu {

namespace {

using Kind = MenuItem::Kind;

struct Opts {
    std::string why;        // disabled reason; empty = enabled
    bool checkable = false;
    bool checked = false;
    bool planned = false;
    std::string argument{};
};

MenuItem Action(Context focus, const std::string& id, const std::string& label,
                const std::string& hint, Opts opts = {}) {
    MenuItem item;
    item.kind = Kind::Action;
    item.id = id;
    item.label = label;
    item.hint = hint;
    item.shortcut = ShortcutFor(id, focus);
    item.argument = std::move(opts.argument);
    item.checkable = opts.checkable;
    item.checked = opts.checked;
    item.planned = opts.planned;
    if (item.planned) {
        item.enabled = false;
        item.disabled_reason = "Planned, not available yet.";
    } else if (!opts.why.empty()) {
        item.enabled = false;
        item.disabled_reason = opts.why;
    }
    return item;
}

MenuItem Planned(Context focus, const std::string& id, const std::string& label, const std::string& hint) {
    Opts opts;
    opts.planned = true;
    return Action(focus, id, label, hint, opts);
}

MenuItem Toggle(Context focus, const std::string& id, const std::string& label, const std::string& hint,
                bool checked, const std::string& why = "") {
    Opts opts;
    opts.checkable = true;
    opts.checked = checked;
    opts.why = why;
    return Action(focus, id, label, hint, opts);
}

MenuItem Tool(Context focus, const std::string& id, const std::string& label) {
    return Action(focus, id, label, "Opens the " + label + " tool.");
}

MenuItem Sep() {
    MenuItem item;
    item.kind = Kind::Separator;
    return item;
}

MenuItem Head(const std::string& label) {
    MenuItem item;
    item.kind = Kind::Header;
    item.label = label;
    return item;
}

// The Engine's top menus a plugin command can name
bool IsMenuTitle(const std::string& title) {
    static const char* const kTitles[] = {"File", "Edit", "View", "Nodes", "Train", "Data", "Tools",
                                          "Script", "Deploy", "Simulation", "Help"};
    return std::find(std::begin(kTitles), std::end(kTitles), title) != std::end(kTitles);
}

// The plugin commands of one menu, a separator between groups: each group
// once, in the order its first command registered (a plugin registered
// again does not split its group), its commands in registration order
void AppendPluginCommands(Context focus, const std::vector<PluginCommandEntry>& commands, const std::string& menu,
                          std::vector<MenuItem>& items) {
    // A group is one plugin's (its id, the key's prefix): two plugins' "run" stay apart
    const auto group_of = [](const PluginCommandEntry& c) { return c.key.substr(0, c.key.find(':')) + '\n' + c.group; };
    std::vector<std::string> groups;
    for (const auto& c : commands)
        if (c.menu == menu && std::find(groups.begin(), groups.end(), group_of(c)) == groups.end())
            groups.push_back(group_of(c));
    std::vector<const PluginCommandEntry*> ordered;
    for (const auto& g : groups)
        for (const auto& c : commands)
            if (c.menu == menu && group_of(c) == g) ordered.push_back(&c);
    bool first = true;
    std::string group;
    for (const auto* entry : ordered) {
        const auto& c = *entry;
        if (!first && group_of(c) != group) items.push_back(Sep());
        first = false;
        group = group_of(c);
        Opts opts;
        opts.argument = c.key;
        opts.checkable = c.checkable;
        opts.checked = c.checked;
        if (!c.enabled) opts.why = c.disabled_reason.empty() ? "Not available now." : c.disabled_reason;
        items.push_back(Action(focus, "plugin.command", c.label, c.hint, opts));
    }
}

MenuItem Sub(const std::string& id, const std::string& label, const std::string& hint,
             std::vector<MenuItem> children, const std::string& why = "") {
    MenuItem item;
    item.kind = Kind::Submenu;
    item.id = id;
    item.label = label;
    item.hint = hint;
    item.children = std::move(children);
    if (!why.empty()) {
        item.enabled = false;
        item.disabled_reason = why;
    }
    return item;
}

std::vector<MenuItem> ExportSubmenu(Context focus, bool onnx_available) {
    std::vector<MenuItem> items;
    items.push_back(Action(focus, "export.cyxmodel", "CyxWiz package (.cyxmodel)",
                           "Exports the trained model as a CyxWiz package."));
    items.push_back(Action(focus, "export.safetensors", "Weights only (.safetensors)",
                           "Exports only the trained weights."));
    Opts onnx;
    if (!onnx_available) onnx.why = "ONNX export is not compiled into this build.";
    items.push_back(Action(focus, "export.onnx", "ONNX model (.onnx)",
                           "Exports a model that other runtimes can load.", onnx));
    items.push_back(Planned(focus, "export.gguf", "GGUF LLM package (.gguf)",
                            "Exports a language model for llama.cpp style runtimes."));
    return items;
}

std::vector<MenuItem> RecentProjects(Context focus, const MenuInputs& in, const std::string& id,
                                     const std::string& hint) {
    std::vector<MenuItem> items;
    for (const auto& project : in.recent_projects) {
        Opts opts;
        opts.argument = project.id;
        items.push_back(Action(focus, id, project.label, hint, opts));
    }
    return items;
}

}  // namespace

const char* ContextName(Context context) {
    switch (context) {
        case Context::Any: return "Everywhere";
        case Context::StudioCanvas: return "Studio canvas";
        case Context::ScriptEditor: return "Script Editor";
        case Context::ScriptDebugging: return "Script Editor, debugging";
        case Context::ScriptNotebook: return "Script Editor, notebook";
        case Context::Variables: return "Variable Explorer";
        case Context::TableViewer: return "Table Viewer";
        case Context::DataStudio: return "Data Studio";
    }
    return "Everywhere";
}

const std::vector<ShortcutEntry>& ShortcutTable() {
    static const std::vector<ShortcutEntry> table = [] {
        std::vector<ShortcutEntry> t;
        auto add = [&t](Context c, const char* id, const char* label, const char* chord, const char* menu,
                        bool window_handled = false, bool planned = false) {
            ShortcutEntry e;
            e.context = c;
            e.action_id = id;
            e.action_label = label;
            e.chord = chord;
            e.menu_path = menu;
            e.window_handled = window_handled;
            e.planned = planned;
            t.push_back(e);
        };
        const Context any = Context::Any;
        const Context canvas = Context::StudioCanvas;
        const Context script = Context::ScriptEditor;
        const Context debug = Context::ScriptDebugging;
        const Context notebook = Context::ScriptNotebook;
        const Context variables = Context::Variables;
        const Context viewer = Context::TableViewer;
        const Context studio = Context::DataStudio;

        // Everywhere: handled by the main window.
        add(any, "file.new_project", "New Project...", "Ctrl+Shift+N", "File");
        add(any, "file.open_project", "Open Project...", "Ctrl+Shift+O", "File");
        add(any, "script.new", "New Script...", "Ctrl+N", "File, Script");
        add(any, "script.open", "Open Script...", "Ctrl+O", "File, Script");
        add(any, "file.save", "Save (the script or graph when the Script Editor or canvas is focused)", "Ctrl+S", "File");
        add(any, "file.save_as", "Save As... (the script or graph when the Script Editor or canvas is focused)", "Ctrl+Shift+S", "File");
        add(any, "file.save_all", "Save All", "Ctrl+Alt+S", "File");
        add(any, "file.import_model", "Open Model File...", "Ctrl+I", "File");
        add(any, "export.cyxmodel", "Export Model as CyxWiz package", "Ctrl+E", "File > Export Model");
        add(any, "file.preferences", "Preferences...", "Ctrl+,", "File");
        add(any, "view.command_palette", "Command Palette", "Ctrl+P", "View");
        add(any, "view.patterns", "Patterns panel", "Ctrl+Shift+P", "View > Panels");
        add(any, "edit.find_in_files", "Find in Files...", "Ctrl+Shift+F", "Edit");
        add(any, "edit.replace_in_files", "Replace in Files...", "Ctrl+Shift+H", "Edit");
        add(any, "train.start", "Run: start training (runs the script when the Script Editor is focused)", "F5", "Train");
        add(any, "train.local_debug", "Local Debug", "F6", "Train");
        add(any, "train.compile", "Compile Graph", "F7", "Train");
        add(any, "train.run_test", "Run Test", "F8", "Train > Test");
        add(any, "script.python_console", "Open Python Console (go to definition when the Script Editor is focused)", "F12", "Script");
        add(any, "help.documentation", "Documentation", "F1", "Help");
        add(any, "view.fullscreen", "Full Screen", "F11", "View");
        add(any, "file.exit", "Exit", "Alt+F4", "File", true);

        // Studio canvas: the node editor handles these itself.
        add(canvas, "edit.undo", "Undo", "Ctrl+Z", "Edit", true);
        add(canvas, "edit.redo", "Redo", "Ctrl+Y", "Edit", true);
        add(canvas, "edit.redo", "Redo", "Ctrl+Shift+Z", "Edit", true);
        add(canvas, "edit.cut", "Cut", "Ctrl+X", "Edit", true);
        add(canvas, "edit.copy", "Copy", "Ctrl+C", "Edit", true);
        add(canvas, "edit.paste", "Paste", "Ctrl+V", "Edit", true);
        add(canvas, "nodes.duplicate", "Duplicate", "Ctrl+D", "Nodes", true);
        add(canvas, "edit.select_all", "Select All", "Ctrl+A", "Edit", true);
        add(canvas, "nodes.delete", "Delete Selected", "Delete", "Nodes", true);
        add(canvas, "edit.find", "Find node", "Ctrl+F", "Edit", true);
        add(canvas, "canvas.next_match", "Next match", "F3", "", true);
        add(canvas, "canvas.previous_match", "Previous match", "Shift+F3", "", true);
        add(canvas, "canvas.escape", "Close search, clear selection", "Esc", "", true);
        add(canvas, "nodes.group", "Group Selected", "Ctrl+G", "Nodes", true);
        add(canvas, "nodes.ungroup", "Ungroup", "Ctrl+Shift+G", "Nodes", true);
        add(canvas, "canvas.subgraph", "Create subgraph from selection", "Ctrl+Shift+U", "", true);
        add(canvas, "view.studio_minimap", "Toggle minimap", "M", "View > Minimaps", true);
        add(canvas, "canvas.frame", "Frame selected nodes, or all", "F", "", true);
        add(canvas, "graph.save", "Save the graph to its file (asks for a name the first time)", "Ctrl+S", "", true);
        add(canvas, "graph.save_as", "Save the graph as a new file", "Ctrl+Shift+S", "", true);

        // Script Editor: the editor handles some keys, the main window others.
        add(script, "script.new", "New script tab", "Ctrl+N", "File, Script", true);
        add(script, "script.open", "Open Script...", "Ctrl+O", "File, Script", true);
        add(script, "script.save", "Save script", "Ctrl+S", "File", true);
        add(script, "script.save_as", "Save script as...", "Ctrl+Shift+S", "File", true);
        add(script, "script.close_tab", "Close tab", "Ctrl+W", "", true);
        add(script, "script.cell_mode", "Toggle notebook cell mode", "Ctrl+Shift+M", "", true);
        add(script, "script.run", "Run script", "F5", "Script", true);
        add(script, "script.stop", "Stop script", "Shift+F5", "Script", true);
        add(script, "script.run_selection", "Run selection", "F9", "", true);
        add(script, "script.run_cell", "Run current cell", "Ctrl+Enter", "", true);
        add(script, "script.debug", "Start debugging", "F10", "", true);
        add(script, "script.completion", "Completion (with the list open: show or hide details)", "Ctrl+Space", "", true);
        add(script, "editor.go_to_definition", "Go to definition (also Ctrl+click)", "F12", "", true);
        add(script, "editor.signature_help", "Show the signature of the call (also opens when you type ( or ,)", "Ctrl+Shift+Space", "", true);
        add(script, "editor.go_to_definition_click", "Go to the definition of the name clicked", "Ctrl+Click", "", true);
        add(script, "edit.find", "Find...", "Ctrl+F", "Edit");
        add(script, "edit.replace", "Replace...", "Ctrl+H", "Edit");
        add(script, "edit.go_to_line", "Go to Line...", "Ctrl+G", "Edit");
        add(script, "edit.duplicate_line", "Duplicate Line", "Ctrl+D", "Edit > Lines");
        add(script, "edit.move_line_up", "Move Line Up", "Alt+Up", "Edit > Lines");
        add(script, "edit.move_line_down", "Move Line Down", "Alt+Down", "Edit > Lines");
        add(script, "edit.join_lines", "Join Lines", "Ctrl+J", "Edit > Lines");
        add(script, "edit.indent", "Indent", "Tab", "Edit > Lines", true);
        add(script, "edit.outdent", "Outdent", "Shift+Tab", "Edit > Lines", true);
        add(script, "edit.toggle_line_comment", "Toggle Line Comment", "Ctrl+/", "Edit");
        add(script, "edit.toggle_block_comment", "Toggle Block Comment", "Shift+Alt+A", "Edit");
        add(script, "edit.undo", "Undo", "Ctrl+Z", "Edit", true);
        add(script, "edit.redo", "Redo", "Ctrl+Y", "Edit", true);
        add(script, "edit.cut", "Cut", "Ctrl+X", "Edit", true);
        add(script, "edit.copy", "Copy", "Ctrl+C", "Edit", true);
        add(script, "edit.paste", "Paste", "Ctrl+V", "Edit", true);
        add(script, "edit.delete", "Delete", "Delete", "Edit", true);
        add(script, "edit.select_all", "Select All", "Ctrl+A", "Edit", true);
        // The code view's own keys (TOFIX133 P1).
        add(script, "editor.cursor_above", "Add a cursor on the line above", "Ctrl+Alt+Up", "", true);
        add(script, "editor.cursor_below", "Add a cursor on the line below", "Ctrl+Alt+Down", "", true);
        add(script, "editor.select_occurrences", "Select every occurrence of the word", "Ctrl+Shift+L", "", true);
        add(script, "editor.single_cursor", "Back to one cursor; close the completion list or signature", "Escape", "", true);
        add(script, "editor.fold", "Fold the block at the cursor", "Ctrl+Shift+[", "", true);
        add(script, "editor.unfold", "Unfold the block at the cursor", "Ctrl+Shift+]", "", true);
        add(script, "editor.add_cursor_click", "Add a cursor where you click", "Alt+Click", "", true);
        add(script, "editor.find_next", "Next match (find open)", "F3", "", true);
        add(script, "editor.find_previous", "Previous match (find open)", "Shift+F3", "", true);

        // Script Editor while debugging: the debugger handles these.
        add(debug, "debug.continue", "Continue", "F5", "", true);
        add(debug, "debug.stop", "Stop debugging", "Shift+F5", "", true);
        add(debug, "debug.step_over", "Step over", "F10", "", true);
        add(debug, "debug.step_into", "Step into", "F11", "", true);
        add(debug, "debug.step_out", "Step out", "Shift+F11", "", true);
        add(debug, "debug.toggle_breakpoint", "Toggle breakpoint", "F9", "", true);
        add(script, "editor.breakpoint_settings", "Breakpoint settings: condition, hit count, enabled", "Right-click the gutter", "", true);
        add(debug, "debug.frame_values", "Run Python in the paused frame (Console REPL)", "Enter in the REPL", "", true);

        // Script Editor in notebook mode: the notebook handles these. The
        // letters work in command mode (no cell being edited).
        add(notebook, "notebook.edit_cell", "Edit the selected cell", "Enter", "", true);
        add(notebook, "notebook.command_mode", "Stop editing (command mode)", "Escape", "", true);
        add(notebook, "notebook.run_cell", "Run cell and select the next", "Shift+Enter", "", true);
        add(notebook, "notebook.run_cell_stay", "Run cell and stay on it", "Ctrl+Enter", "", true);
        add(notebook, "notebook.previous_cell", "Select the cell above", "Up", "", true);
        add(notebook, "notebook.next_cell", "Select the cell below", "Down", "", true);
        add(notebook, "notebook.add_above", "Add a code cell above", "A", "", true);
        add(notebook, "notebook.add_below", "Add a code cell below", "B", "", true);
        add(notebook, "notebook.delete_cell", "Delete the cell (press D twice)", "D, D", "", true);
        add(notebook, "notebook.to_markdown", "Make the cell markdown", "M", "", true);
        add(notebook, "notebook.to_code", "Make the cell code", "Y", "", true);
        add(notebook, "notebook.collapse", "Fold or unfold the cell", "C", "", true);
        add(notebook, "notebook.collapse_output", "Fold or unfold the output", "O", "", true);
        add(notebook, "script.cell_mode", "Leave notebook mode", "Ctrl+Shift+M", "", true);

        // Variable Explorer and a notebook's Variables (TOFIX133 P5): the view handles these.
        add(variables, "variables.move", "Select the variable above or below", "Up, Down", "", true);
        add(variables, "variables.expand", "Expand or collapse the selected value", "Right, Left", "", true);
        add(variables, "variables.view_data", "View data (tables, arrays, lists), or expand", "Enter", "", true);
        add(variables, "variables.copy_value", "Copy the value", "Ctrl+C", "", true);
        add(variables, "variables.delete", "Delete the variable", "Delete", "", true);

        // Table Viewer (and the Data Viewer's tabs of Python values): the viewer handles these.
        add(viewer, "table.save", "Save the table's edits", "Ctrl+S", "", true);
        add(viewer, "table.cancel_edit", "Cancel editing a cell", "Escape", "", true);
        add(viewer, "table.edit_cell", "Edit a cell", "Double-click", "", true);
        add(viewer, "table.add_selection", "Add a cell to the selection", "Ctrl+Click", "", true);
        add(viewer, "table.extend_selection", "Extend the selection", "Shift+Click", "", true);
        add(viewer, "table.slice_step", "Step the first index of an array slice", "Up, Down", "", true);

        // Data Studio: the Query tab handles these while its editor has focus.
        add(studio, "datastudio.run_query", "Run the query", "Ctrl+Enter", "", true);
        // While a code cell is being edited, the language keys work as in a script.
        add(notebook, "script.completion", "Completion in the cell (with the list open: details)", "Ctrl+Space", "", true);
        add(notebook, "editor.signature_help", "Show the signature of the call in the cell", "Ctrl+Shift+Space", "", true);
        add(notebook, "editor.go_to_definition", "Go to definition (also Ctrl+click)", "F12", "", true);
        return t;
    }();
    return table;
}

std::string ShortcutFor(const std::string& action_id, Context focus) {
    const ShortcutEntry* any = nullptr;
    const ShortcutEntry* first = nullptr;
    for (const auto& e : ShortcutTable()) {
        if (e.action_id != action_id) continue;
        if (e.context == focus) return e.chord;
        if (e.context == Context::Any && !any) any = &e;
        if (!first) first = &e;
    }
    if (any) return any->chord;
    if (first) return first->chord;
    return {};
}

MenuModel BuildMenuModel(const MenuInputs& in) {
    const Context f = in.focus;
    const bool in_script =
        f == Context::ScriptEditor || f == Context::ScriptDebugging || f == Context::ScriptNotebook;
    const bool in_canvas = f == Context::StudioCanvas;
    const bool has_sel = in_canvas && in.selected_nodes > 0;
    const std::string any_editor =
        (in_script || in_canvas) ? "" : "Click in the Script Editor or on the Studio canvas first.";
    const std::string script_only =
        in_script ? "" : "Click in the Script Editor first. This command edits script text.";
    const std::string need_sel =
        !in_canvas ? "Click on the Studio canvas first."
                   : (has_sel ? "" : "Select one or more nodes on the canvas first.");
    const std::string cut_copy = in_script ? "" : need_sel;
    const std::string need_project = in.has_project ? "" : "Open or create a project first.";
    const bool idle = in.training == TrainingState::Idle;
    const bool paused = in.training == TrainingState::Paused;
    const std::string sign_note = in.signed_in ? "" : " Asks you to sign in first.";

    MenuModel model;

    // ---------------------------------------------------------------- File
    {
        Menu m{"file", "File", {}};
        auto& i = m.items;
        i.push_back(Action(f, "file.new_project", "New Project...", "Creates a project in a folder you choose."));
        i.push_back(Action(f, "file.open_project", "Open Project...", "Opens a project file."));
        {
            auto recent = RecentProjects(f, in, "file.open_recent", "Opens this project.");
            const std::string why = recent.empty() ? "No project has been opened yet." : "";
            if (!recent.empty()) {
                recent.push_back(Sep());
                recent.push_back(Action(f, "file.clear_recent", "Clear Recent Projects", "Empties this list."));
            }
            i.push_back(Sub("file.recent", "Open Recent", "Projects you opened lately.", std::move(recent), why));
        }
        i.push_back(Action(f, "file.close_project", "Close Project", "Saves settings and closes the project.",
                           Opts{need_project}));
        i.push_back(Action(f, "file.show_project_folder", "Show Project Folder",
                           "Opens the project's folder in the file manager.", Opts{need_project}));
        i.push_back(Sep());
        i.push_back(Action(f, "file.new_window", "New Window", "Starts a second Engine window."));
        {
            auto recent = RecentProjects(f, in, "file.open_in_new_window", "Opens this project in its own window.");
            const std::string why = recent.empty() ? "No project has been opened yet." : "";
            i.push_back(Sub("file.new_window_recent", "Open Project in New Window",
                            "Opens a recent project in its own window.", std::move(recent), why));
        }
        i.push_back(Sep());
        i.push_back(Action(f, "script.new", "New Script...", "Asks for a name and folder, then opens the new script."));
        i.push_back(Action(f, "script.open", "Open Script...", "Opens a Python script in the Script Editor."));
        i.push_back(Sep());
        i.push_back(Action(f, "file.save", "Save", "Saves the graph, every open script, the layout and the project settings.",
                           Opts{need_project}));
        i.push_back(Action(f, "file.save_as", "Save As...", "Saves the project under a new name.", Opts{need_project}));
        i.push_back(Action(f, "file.save_all", "Save All", "Saves every open script."));
        i.push_back(Toggle(f, "file.auto_save", "Auto Save", "Saves open scripts every 60 seconds.", in.auto_save));
        i.push_back(Sep());
        i.push_back(Action(f, "file.import_model", "Open Model File...",
                           "Reads a model file and opens its graph, without its weights."));
        i.push_back(Sub("file.export", "Export Model", "Exports the trained model. The same menu is in Deploy.",
                        ExportSubmenu(f, in.onnx_export_available)));
        i.push_back(Sep());
        i.push_back(Action(f, "file.preferences", "Preferences...", "Opens Preferences."));
        i.push_back(Sep());
        i.push_back(Action(f, "file.restart", "Restart", "Saves, then restarts the Engine.", Opts{need_project}));
        i.push_back(Action(f, "file.exit", "Exit", "Closes the Engine. Unsaved work is asked about first."));
        model.menus.push_back(std::move(m));
    }

    // ---------------------------------------------------------------- Edit
    {
        Menu m{"edit", "Edit", {}};
        auto& i = m.items;
        i.push_back(Head(in_script ? "Editing: Script Editor"
                                   : (in_canvas ? "Editing: Studio canvas" : "Editing: nothing focused")));
        i.push_back(Action(f, "edit.undo", "Undo", "Undoes the last change in the focused editor.", Opts{any_editor}));
        i.push_back(Action(f, "edit.redo", "Redo", "Redoes the change you undid.", Opts{any_editor}));
        i.push_back(Sep());
        i.push_back(Action(f, "edit.cut", "Cut", "Cuts the selected text or nodes.", Opts{cut_copy}));
        i.push_back(Action(f, "edit.copy", "Copy", "Copies the selected text or nodes.", Opts{cut_copy}));
        i.push_back(Action(f, "edit.paste", "Paste", "Pastes text or nodes.", Opts{any_editor}));
        i.push_back(Action(f, "edit.delete", "Delete", "Deletes the selected text or nodes.", Opts{cut_copy}));
        i.push_back(Action(f, "edit.select_all", "Select All", "Selects all text or all nodes.", Opts{any_editor}));
        i.push_back(Sep());
        i.push_back(Action(f, "edit.find", "Find...", "Finds text in the script, or a node on the canvas.",
                           Opts{any_editor}));
        i.push_back(Action(f, "edit.replace", "Replace...", "Finds and replaces text in the script.", Opts{script_only}));
        i.push_back(Action(f, "edit.find_in_files", "Find in Files...", "Searches every file in the project."));
        i.push_back(Action(f, "edit.replace_in_files", "Replace in Files...", "Replaces text across the project."));
        i.push_back(Action(f, "edit.go_to_line", "Go to Line...", "Jumps to a line number.", Opts{script_only}));
        i.push_back(Sep());
        i.push_back(Sub("edit.lines", "Lines", "Commands for whole lines.",
                        {Action(f, "edit.duplicate_line", "Duplicate Line", "Copies the current line below itself."),
                         Action(f, "edit.move_line_up", "Move Line Up", "Moves the current line up."),
                         Action(f, "edit.move_line_down", "Move Line Down", "Moves the current line down."),
                         Action(f, "edit.indent", "Indent", "Indents the selected lines."),
                         Action(f, "edit.outdent", "Outdent", "Removes one indent level."),
                         Action(f, "edit.join_lines", "Join Lines", "Joins the selected lines into one."),
                         Sep(),
                         Action(f, "edit.sort_asc", "Sort Ascending (A-Z)", "Sorts the selected lines."),
                         Action(f, "edit.sort_desc", "Sort Descending (Z-A)", "Sorts the selected lines in reverse.")},
                        script_only));
        i.push_back(Sub("edit.transform", "Transform Text", "Changes the case of the selected text.",
                        {Action(f, "edit.uppercase", "UPPERCASE", "Makes the selection upper case."),
                         Action(f, "edit.lowercase", "lowercase", "Makes the selection lower case."),
                         Action(f, "edit.titlecase", "Title Case", "Capitalises each word.")},
                        script_only));
        i.push_back(Action(f, "edit.toggle_line_comment", "Toggle Line Comment",
                           "Comments or uncomments the selected lines.", Opts{script_only}));
        i.push_back(Action(f, "edit.toggle_block_comment", "Toggle Block Comment",
                           "Wraps the selection in a block comment.", Opts{script_only}));
        model.menus.push_back(std::move(m));
    }

    // ---------------------------------------------------------------- View
    {
        Menu m{"view", "View", {}};
        auto& i = m.items;
        i.push_back(Action(f, "view.command_palette", "Command Palette...",
                           "Search every command by name. Works from any panel."));
        i.push_back(Sep());
        {
            std::vector<MenuItem> panels;
            std::string group;
            for (const auto& p : in.panels) {
                if (p.group != group) {
                    group = p.group;
                    panels.push_back(Head(group));
                }
                Opts opts;
                opts.checkable = true;
                opts.checked = p.visible;
                opts.argument = p.name;
                auto item = Action(f, "view.panel", p.name, "Shows or hides the " + p.name + " panel.", opts);
                if (p.name == "Patterns") item.shortcut = ShortcutFor("view.patterns", f);
                if (!p.shortcut.empty()) item.shortcut = p.shortcut;
                panels.push_back(std::move(item));
            }
            i.push_back(Sub("view.panels", "Panels", "Show or hide panels, grouped by area.", std::move(panels)));
        }
        i.push_back(Sub("view.layout", "Layout", "Save or reset the panel arrangement.",
                        {Action(f, "view.save_layout", "Save Layout", "Saves the current panel arrangement."),
                         Action(f, "view.reset_layout", "Reset to Default",
                                "Puts every panel back in its default place.")}));
        {
            // The focused window out of the main window and back (TOFIX129 A8).
            Opts out;
            out.argument = in.focused_window;
            if (in.focused_window.empty()) out.why = "No window has the focus.";
            else if (!in.focused_window_may_leave) out.why = "This window stays in the main window.";
            else if (in.focused_window_outside) out.why = "Already in a window of its own.";
            Opts back;
            back.argument = in.focused_window;
            if (!in.focused_window_outside) back.why = "The window is in the main window.";
            i.push_back(Sub("view.window", "Window", "The focused window out of the main window, or back.",
                            {Action(f, "view.window_out", "Move to its own window",
                                    "The focused window becomes a window of its own on the desktop, on any monitor.", out),
                             Action(f, "view.window_back", "Dock back into the main window",
                                    "The window returns to its place in the main window.", back)}));
        }
        i.push_back(Sep());
        {
            std::vector<MenuItem> themes;
            std::string group;
            for (const auto& t : in.themes) {
                if (!t.group.empty() && t.group != group) {
                    group = t.group;
                    themes.push_back(Head(group));
                }
                Opts opts;
                opts.checkable = true;
                opts.checked = t.checked;
                opts.argument = t.id;
                themes.push_back(Action(f, "view.theme", t.label, "Applies the " + t.label + " theme.", opts));
            }
            i.push_back(Sub("view.themes", "Theme", "Pick a colour theme.", std::move(themes)));
        }
        i.push_back(Action(f, "view.theme_editor", "Theme Editor...", "Edit colours and save your own theme."));
        {
            std::vector<MenuItem> packs;
            for (const auto& p : in.icon_packs) {
                Opts opts;
                opts.checkable = true;
                opts.checked = p.checked;
                opts.argument = p.id;
                packs.push_back(Action(f, "view.icon_pack", p.label, "Draws node icons with " + p.label + ".", opts));
            }
            i.push_back(Sub("view.icon_packs", "Node Icon Pack", "Pick the icon set drawn on nodes.", std::move(packs)));
        }
        i.push_back(Sub("view.minimaps", "Minimaps", "Small overview maps.",
                        {Toggle(f, "view.studio_minimap", "Studio Minimap",
                                "Shows the overview map on the canvas.", in.studio_minimap),
                         Toggle(f, "view.script_minimap", "Script Editor Minimap",
                                "Shows the overview map beside the script.", in.script_minimap)}));
        i.push_back(Sep());
        i.push_back(Toggle(f, "view.fullscreen", "Full Screen",
                           "Fills the screen with the Engine window; F11 again goes back.", in.fullscreen));
        model.menus.push_back(std::move(m));
    }

    // --------------------------------------------------------------- Nodes
    {
        Menu m{"nodes", "Nodes", {}};
        auto& i = m.items;
        i.push_back(Sub("nodes.add_layer", "Add Layer", "Adds a layer node to the canvas.",
                        {Action(f, "nodes.add_dense", "Dense / Linear", "Adds a Dense node."),
                         Action(f, "nodes.add_conv", "Convolutional", "Adds a convolution node."),
                         Action(f, "nodes.add_pooling", "Pooling", "Adds a pooling node."),
                         Action(f, "nodes.add_dropout", "Dropout", "Adds a Dropout node."),
                         Action(f, "nodes.add_batchnorm", "Batch Normalization", "Adds a Batch Normalization node."),
                         Action(f, "nodes.add_attention", "Attention", "Adds an attention node.")}));
        i.push_back(Sep());
        i.push_back(Action(f, "nodes.group", "Group Selected", "Groups the selected nodes.", Opts{need_sel}));
        i.push_back(Action(f, "nodes.ungroup", "Ungroup", "Splits the selected group.", Opts{need_sel}));
        i.push_back(Action(f, "nodes.duplicate", "Duplicate", "Copies the selected nodes beside themselves.",
                           Opts{need_sel}));
        i.push_back(Action(f, "nodes.delete", "Delete Selected", "Deletes the selected nodes.", Opts{need_sel}));
        i.push_back(Sep());
        {
            const std::string on_canvas = in_canvas ? "" : "Click on the Studio canvas first.";
            i.push_back(Sub("nodes.freeze", "Freeze", "Keeps layers' weights as they are while the rest trains.",
                            {Action(f, "nodes.freeze_selected", "Freeze Selected",
                                    "Freezes the selected layers that hold weights.", Opts{need_sel}),
                             Action(f, "nodes.unfreeze_selected", "Unfreeze Selected",
                                    "Lets the selected layers train again.", Opts{need_sel}),
                             Sep(),
                             Action(f, "nodes.freeze_loaded", "Freeze Every Layer Loaded from the File",
                                    "Freezes the layers the Data Loader's weights file fills.", Opts{on_canvas}),
                             Action(f, "nodes.freeze_but_head", "Freeze All but the Head",
                                    "Freezes every layer with weights except the last.", Opts{on_canvas}),
                             Action(f, "nodes.freeze_but_last", "Freeze All but the Last N Layers...",
                                    "Asks how many layers keep training, counted from the output.", Opts{on_canvas}),
                             Sep(),
                             Action(f, "nodes.unfreeze_all", "Unfreeze All", "Lets every layer train.",
                                    Opts{on_canvas})}));
        }
        i.push_back(Sep());
        i.push_back(Action(f, "nodes.custom_node_editor", "Custom Node Editor...", "Create or edit your own node type."));
        model.menus.push_back(std::move(m));
    }

    // --------------------------------------------------------------- Train
    {
        Menu m{"train", "Train", {}};
        auto& i = m.items;
        i.push_back(Action(f, "train.compile", "Compile Graph", "Checks the graph and reports every problem found."));
        i.push_back(Action(f, "train.local_debug", "Local Debug", "Runs a short debug pass on this machine."));
        i.push_back(Sep());
        i.push_back(Action(f, "train.start", "Start Training", "Compiles the graph and starts training.",
                           Opts{idle ? "" : "Training is already running. Stop it first."}));
        if (paused) {
            i.push_back(Action(f, "train.resume", "Resume Training", "Continues the paused run."));
        } else {
            i.push_back(Action(f, "train.pause", "Pause Training", "Pauses the run.",
                               Opts{idle ? "No training is running." : ""}));
        }
        i.push_back(Action(f, "train.stop", "Stop Training", "Stops the run.",
                           Opts{idle ? "No training is running." : ""}));
        i.push_back(Sep());
        i.push_back(Sub("train.test", "Test", "Test a trained model.",
                        {Action(f, "train.run_test", "Run Test", "Tests the trained model on the test split."),
                         Action(f, "train.quick_test", "Run Quick Test", "Tests the first 10 batches of the test split and marks the result as partial."),
                         Action(f, "train.view_test_results", "View Test Results", "Opens Data Studio > Results > Test."),
                         Action(f, "train.compare_test_results", "Compare Test Results...", "Opens Results > Test on Comparison: the latest test beside the one before."),
                         Action(f, "train.export_test_report", "Export Test Report...",
                                "Saves the latest test results as a JSON report."),
                         Sep(),
                         Action(f, "train.load_checkpoint", "Load Checkpoint for Testing...",
                                "Loads saved weights to test them.")}));
        i.push_back(Sub("train.evaluate", "Evaluate", "Train the graph again on folds of its data.",
                        {Action(f, "train.cross_validate", "Cross-validate",
                                "Trains the graph once per fold and scores each fold once (Data Studio > Results > Cross-validation)."),
                         Action(f, "train.learning_curve", "Learning Curve",
                                "Trains the graph on shares of its rows: does more data help? (Data Studio > Results > Learning curve).")}));
        i.push_back(Sep());
        i.push_back(Action(f, "train.dashboard", "Training Dashboard", "Opens the Training panel."));
        i.push_back(Action(f, "train.optimizer_settings", "Optimizer Settings...", "Shows the graph's optimizer node in Properties."));
        i.push_back(Action(f, "train.hyperparam_search", "Hyperparameter Search",
                           "Opens the Hyperparameter Search panel."));
        model.menus.push_back(std::move(m));
    }

    // ---------------------------------------------------------------- Data
    {
        Menu m{"data", "Data", {}};
        auto& i = m.items;
        i.push_back(Action(f, "data.import", "Import Dataset...", "Adds a Data Input node and opens it: data enters the Engine there."));
        i.push_back(Action(f, "data.create_custom", "Create Custom Dataset...", "Adds a Data Input node for your own files (tables, image, audio or text folders)."));
        i.push_back(Action(f, "data.statistics", "Dataset Statistics", "Opens Data Studio's Profile: row counts, columns and value summaries."));
        i.push_back(Sep());
        i.push_back(Sub("data.explore", "Explore", "Look at a table before using it.",
                        {Action(f, "data.missing_values", "Missing Value Analysis", "Opens Data Studio's Missing tab: incomplete rows and the columns that go missing together."),
                         Action(f, "data.outliers", "Outlier Detection", "Opens Data Studio's Outliers tab: values far from the rest, by IQR, Z-score or Modified Z."),
                         Action(f, "data.correlation", "Correlation Matrix", "Opens Data Studio's Correlation tab: how the number columns move together.")}));
        i.push_back(Sub("data.transform", "Transform", "Rescale or reshape columns: each adds a step in Data Studio's Prepare.",
                        {Action(f, "data.normalization", "Normalization (Min-Max)",
                                "Adds a Min-Max Scaler step in Data Studio's Prepare, after the data in use."),
                         Action(f, "data.standardization", "Standardization (Z-Score)",
                                "Adds a Z-score (Standard Scaler) step in Data Studio's Prepare, after the data in use."),
                         Action(f, "data.log_transform", "Log Transform",
                                "Adds a Log Transform step in Data Studio's Prepare, after the data in use."),
                         Action(f, "data.boxcox", "Box-Cox Transform",
                                "Adds a Power Transform step (Box-Cox) in Data Studio's Prepare, after the data in use."),
                         Action(f, "data.feature_scaling", "Feature Scaling (All Methods)",
                                "Adds a scaling step in Data Studio's Prepare and compares every method on it.")}));
        i.push_back(Action(f, "tools.tokenization", "Tokenization",
                           "Opens Data Studio's Text tab on Tokens: a tokenizer trained on a text column, the tokens per row and the length limit."));
        i.push_back(Planned(f, "data.augment", "Augment", "Data augmentation."));
        model.menus.push_back(std::move(m));
    }

    // --------------------------------------------------------------- Tools
    {
        Menu m{"tools", "Tools", {}};
        auto& i = m.items;
        i.push_back(Sub("tools.model", "Model", "Inspect and explain a model.",
                        {Action(f, "tools.model_summary", "Model Summary",
                                "Opens the Model window on Summary: every layer as compiled, its parameters and multiply-adds."),
                         Action(f, "tools.flops", "FLOPs Calculator",
                                "Opens the Model window on Summary: multiply-adds and FLOPs a sample, per layer and in all."),
                         Action(f, "tools.lr_finder", "Learning Rate Finder",
                                "Opens the Model window on Learning rate: a range test on the Train rows and the rate it suggests."),
                         Sep(),
                         Action(f, "tools.gradcam", "Grad-CAM Visualization",
                                "Opens the Model window on Explain: where a convolution found what raised a test row's score."),
                         Action(f, "tools.saliency", "Saliency Maps",
                                "Opens the Model window on Explain: saliency and Integrated Gradients of a test row's score."),
                         Sep(),
                         Action(f, "tools.model_health", "Training Health",
                                "Opens the Model window on Health: weights, gradients and dead units of each layer, epoch by epoch."),
                         Action(f, "tools.projector", "Embedding Projector",
                                "Opens the Model window on Projector: what a layer makes of the test rows, in two dimensions."),
                         Sep(),
                         Action(f, "tools.nas", "Neural Architecture Search",
                                "Changed versions of the graph, scored, and the best ones trained the way Train does.")}));
        i.push_back(Sub("tools.evaluation", "Evaluation", "Measure how well a model performs.",
                        {Action(f, "tools.cross_validation", "Cross-Validation",
                                "Opens Data Studio > Results > Cross-validation: the graph trained once per fold, each fold scored once."),
                         Action(f, "tools.confusion_matrix", "Confusion Matrix",
                                "Opens Data Studio > Results > Test: the latest test, a kept test or any table of predictions."),
                         Action(f, "tools.roc_auc", "ROC Curve / AUC",
                                "Opens Results > Test: Curves and Threshold, the positive class picked by name."),
                         Action(f, "tools.pr_curve", "Precision-Recall Curve",
                                "Opens Results > Test: Curves and Threshold, the positive class picked by name."),
                         Action(f, "tools.learning_curves", "Learning Curves",
                                "Opens Results > Learning curve: the graph trained on shares of its rows, training against validation."),
                         Tool(f, "tools.feature_importance", "Feature Importance")}));
        i.push_back(Sub("tools.clustering", "Clustering and Projection", "Group rows and reduce dimensions.",
                        {Action(f, "tools.kmeans", "K-Means Clustering", "Opens Data Studio's Groups tab on K-Means: groups, scores, what each group is, how many groups."),
                         Action(f, "tools.dbscan", "DBSCAN", "Opens the Groups tab on DBSCAN: dense groups and noise; the k-distance knee suggests eps."),
                         Action(f, "tools.hierarchical", "Hierarchical Clustering", "Opens the Groups tab on Hierarchical: the dendrogram and its cut."),
                         Action(f, "tools.gmm", "Gaussian Mixture Models", "Opens the Groups tab on Gaussian mixture: soft groups, BIC and AIC."),
                         Action(f, "tools.cluster_eval", "Cluster Evaluation", "Opens the Groups tab: silhouette, Davies-Bouldin and Calinski-Harabasz are on every answer."),
                         Sep(),
                         Action(f, "tools.dim_reduction", "PCA / t-SNE / UMAP", "Opens Data Studio's Projection tab: the rows in two dimensions, coloured by a column or the groups.")}));
        i.push_back(Sub("tools.statistics", "Statistics", "Describe and test data.",
                        {Action(f, "tools.descriptive_stats", "Summary Statistics", "Opens Data Studio's Distribution tab: a column's figures, percentiles and box plot."),
                         Action(f, "tools.hypothesis_test", "Hypothesis Testing", "Opens Data Studio's Tests tab: two groups, paired columns, several groups, one value, two categories."),
                         Action(f, "tools.regression", "Linear / Polynomial Regression", "Opens Data Studio's Regression tab: predict a number column from others, or a curve in one."),
                         Action(f, "tools.distribution_fitter", "Distribution Fitter", "Opens Data Studio's Distribution tab: Normal, Uniform, Exponential and Log-Normal fits.")}));
        i.push_back(Sub("tools.linear_algebra", "Linear Algebra", "Matrices: combine, describe, solve and factor them.",
                        {Action(f, "tools.matrix_calculator", "Matrix Calculator",
                                "Opens the Linear Algebra window on A × B: sums, products, the inverse, determinant, rank, A x = b."),
                         Action(f, "tools.eigen", "Eigenvalue Decomposition", "Opens the Linear Algebra window on Eigen: the directions A only stretches."),
                         Action(f, "tools.svd", "SVD (Singular Value Decomposition)", "Opens the Linear Algebra window on SVD: A's directions and their strengths."),
                         Action(f, "tools.qr", "QR Decomposition", "Opens the Linear Algebra window on QR: orthonormal columns times an upper triangle."),
                         Action(f, "tools.cholesky", "Cholesky Decomposition", "Opens the Linear Algebra window on Cholesky: A = L L^T.")}));
        i.push_back(Sub("tools.signal", "Signal Processing", "Frequencies, filters and wavelets of a number column.",
                        {Action(f, "tools.fft", "FFT (Fourier Transform)", "Opens Data Studio's Signal tab on Spectrum: the frequencies of a column and its peaks."),
                         Action(f, "tools.spectrogram", "Spectrogram (STFT)", "Opens the Signal tab on Spectrogram: the frequencies over time."),
                         Action(f, "tools.filter_designer", "Filter Designer", "Opens the Signal tab on Filter: a band kept or removed, its response and the column filtered."),
                         Action(f, "tools.convolution", "Convolution Calculator", "Opens the Signal tab on Filter with a typed kernel: the column convolved with it."),
                         Action(f, "tools.wavelet", "Wavelet Transform", "Opens the Signal tab on Wavelet: the column split by scale, each part's share of the energy.")}));
        i.push_back(Sub("tools.optimization", "Optimization and Calculus", "Typed formulas minimized, integrated and differentiated; linear and quadratic programs.",
                        {Action(f, "tools.gradient_descent", "Gradient Descent Visualizer",
                                "Opens the Optimization and Calculus window on Optimizers: the training optimizers' paths on a formula's map."),
                         Action(f, "tools.convexity", "Convexity Analyzer", "Opens the window on Curvature: how a formula curves at a point and over a range."),
                         Action(f, "tools.lp", "Linear Programming (LP)", "Opens the window on Linear: the best c . x under linear constraints, any number of variables."),
                         Action(f, "tools.qp", "Quadratic Programming (QP)", "Opens the window on Quadratic: the smallest 0.5 x' Q x + c' x under linear constraints, or a portfolio from a table of returns."),
                         Action(f, "tools.differentiation", "Numerical Differentiation", "Opens the window on Derivative: f' and f'' exactly and by differences."),
                         Action(f, "tools.integration", "Numerical Integration", "Opens the window on Integral: adaptive with an error estimate, and the fixed rules.")}));
        i.push_back(Sub("tools.time_series", "Time Series", "A column over time: its parts, its season, its forecast.",
                        {Action(f, "tools.decomposition", "Time Series Decomposition", "Opens Data Studio's Series tab on Parts: trend, season and the rest."),
                         Action(f, "tools.acf_pacf", "ACF / PACF (Correlogram)", "Opens the Series tab on Correlogram: how the series follows its own past."),
                         Action(f, "tools.stationarity", "Stationarity Testing", "Opens the Series tab on Stationarity: the ADF and KPSS tests."),
                         Action(f, "tools.seasonality", "Seasonality Detection", "Opens the Series tab on Seasonality: the period the series repeats on."),
                         Action(f, "tools.forecasting", "Forecasting", "Opens Data Studio's Forecast tab: a forecast with its range, checked on the last rows held out.")}));
        i.push_back(Sub("tools.text", "Text", "A text column: its words, terms, sentiment, tokens and word vectors.",
                        {Action(f, "tools.tokenization", "Tokenization",
                                "Opens Data Studio's Text tab on Tokens: a tokenizer trained on a text column, the tokens per row and the length limit."),
                         Action(f, "tools.word_frequency", "Word Frequency", "Opens the Text tab on Words: the commonest words or pairs, per group too."),
                         Action(f, "tools.tfidf", "TF-IDF Analysis", "Opens the Text tab on Terms: what sets each group apart, and the rows alike."),
                         Action(f, "tools.embeddings", "Word Embeddings", "Opens the Text tab on Embeddings: word vectors learned from the column, nearest words and the word map."),
                         Action(f, "tools.sentiment", "Sentiment Analysis", "Opens the Text tab on Sentiment: VADER on every row, per group too."),
                         Tool(f, "tools.language_model", "Language Model Generation")}));
        i.push_back(Sub("tools.utilities", "Utilities", "Small helper tools.",
                        {Tool(f, "tools.calculator", "Calculator"),
                         Tool(f, "tools.unit_converter", "Unit Converter"),
                         Tool(f, "tools.random_generator", "Random Generator"),
                         Tool(f, "tools.hash_generator", "Hash Generator"),
                         Tool(f, "tools.json_viewer", "JSON Viewer"),
                         Tool(f, "tools.regex_tester", "Regex Tester")}));
        i.push_back(Sep());
        i.push_back(Sub("tools.monitoring", "Monitoring", "Watch time and memory use.",
                        {Action(f, "tools.monitor_system", "System",
                                "The machine, the compute device and this Engine: CPU, memory, threads."),
                         Action(f, "tools.monitor_memory", "Memory", "The Engine's memory over time."),
                         Action(f, "tools.monitor_loaded_data", "Loaded Data", "Every dataset the Engine holds, in memory and on disk."),
                         Action(f, "tools.monitor_frames", "Frames", "Frame times and the windows they go to."),
                         Action(f, "tools.profiler", "Performance Profiler",
                                "Opens the Studio Debugger on its runtime view."),
                         Sep(),
                         Action(f, "tools.clear_cache", "Clear Caches...", "Lists the caches with their sizes and clears the ones you tick."),
                         Action(f, "tools.gc", "Free Unused Memory", "Returns unused device memory to the driver, collects Python's garbage and says what was freed.")}));
        i.push_back(Sub("tools.diagnostics", "Diagnostics", "Extra logging for support.",
                        {Toggle(f, "tools.idle_log", "Log Frame-rate Changes",
                                "One log line when the Engine slows or speeds its frames: idle, not in front, minimised.", in.idle_log),
                         Toggle(f, "tools.verbose_python", "Log Every Python Command",
                                "Logs each command the Engine sends to Python, with how long it took.", in.verbose_python),
                         Sep(),
                         Toggle(f, "tools.frame_overlay", "Frame Time Overlay",
                                "Shows the whole frame time and the Engine's own interface-building time. More in Monitor > Frames.",
                                in.frame_overlay)}));
        model.menus.push_back(std::move(m));
    }

    // -------------------------------------------------------------- Script
    {
        Menu m{"script", "Script", {}};
        auto& i = m.items;
        i.push_back(Action(f, "script.new", "New Script...", "Asks for a name and folder, then opens the new script."));
        i.push_back(Action(f, "script.open", "Open Script...", "Opens a Python script in the Script Editor."));
        i.push_back(Sep());
        i.push_back(Action(f, "script.run", "Run Script", "Runs the open script.",
                           Opts{!in.script_open ? "Open a script in the Script Editor first."
                                                : (in.script_running ? "A script is already running." : "")}));
        i.push_back(Action(f, "script.stop", "Stop Script", "Stops the running script.",
                           Opts{in.script_running ? "" : "No script is running."}));
        i.push_back(Sep());
        i.push_back(Action(f, "script.python_console", "Open Python Console", "Opens the Python console."));
        model.menus.push_back(std::move(m));
    }

    // -------------------------------------------------------------- Deploy
    {
        Menu m{"deploy", "Deploy", {}};
        auto& i = m.items;
        i.push_back(Action(f, "deploy.connect", "Connect to Server...",
                           "Reserves a Server Node and connects to it." + sign_note));
        i.push_back(Action(f, "deploy.deploy", "Deploy to Server Node...",
                           "Sends the trained model to a Server Node." + sign_note));
        i.push_back(Action(f, "deploy.serving", "Model Serving", "Serves the model on this machine."));
        i.push_back(Sep());
        i.push_back(Sub("deploy.export", "Export Model", "Exports the trained model. The same menu is in File.",
                        ExportSubmenu(f, in.onnx_export_available)));
        i.push_back(Sub("deploy.convert", "Convert Model", "Convert between model file layouts.",
                        {Action(f, "deploy.convert_bin_to_dir", "Binary to Directory...",
                                "Unpacks a model file into a folder."),
                         Action(f, "deploy.convert_dir_to_bin", "Directory to Binary...",
                                "Packs a model folder into one file.")}));
        i.push_back(Sub("deploy.quantize", "Quantize", "Make a model smaller.",
                        {Planned(f, "deploy.quantize_int8", "INT8", "8-bit integer weights."),
                         Planned(f, "deploy.quantize_int4", "INT4", "4-bit integer weights."),
                         Planned(f, "deploy.quantize_fp16", "FP16", "16-bit float weights.")}));
        i.push_back(Sep());
        i.push_back(Planned(f, "deploy.marketplace", "Publish to Marketplace...", "Offers the model on the marketplace."));
        model.menus.push_back(std::move(m));
    }

    // ---------------------------------------------------------- Simulation
    {
        // The open graph's run when it has simulation nodes; then each
        // simulation plugin's commands under its name (its own run controls
        // give way to the graph run); then the windows.
        Menu m{"simulation", "Simulation", {}};
        auto& i = m.items;
        if (in.simulation_graph) {
            i.push_back(Head("Graph run: " + in.simulation_graph_name));
            for (const auto& c : in.simulation_commands) {
                Opts opts;
                if (!c.enabled) opts.why = c.disabled_reason.empty() ? "Not available now." : c.disabled_reason;
                i.push_back(Action(f, c.action_id, c.label, c.hint, opts));
            }
        }
        std::vector<PluginCommandEntry> plugin_commands;
        for (const auto& c : in.plugin_commands)
            if (c.menu == "Simulation" && !(in.simulation_graph && c.replaced_by_graph_run)) plugin_commands.push_back(c);
        // Each plugin once, under its name, in the order it first registered
        std::vector<std::string> plugins;
        for (const auto& c : plugin_commands)
            if (std::find(plugins.begin(), plugins.end(), c.plugin_name) == plugins.end()) plugins.push_back(c.plugin_name);
        for (const auto& plugin : plugins) {
            std::vector<PluginCommandEntry> own;
            for (const auto& c : plugin_commands)
                if (c.plugin_name == plugin) own.push_back(c);
            if (!i.empty()) i.push_back(Sep());
            if (!plugin.empty()) i.push_back(Head(plugin));
            AppendPluginCommands(f, own, "Simulation", i);
        }
        const bool from_plugins = !plugin_commands.empty() ||
            std::any_of(in.plugin_panels.begin(), in.plugin_panels.end(),
                        [](const NamedEntry& p) { return p.group == "Simulation"; });
        if (!from_plugins) {
            if (!i.empty()) i.push_back(Sep());
            i.push_back(Head("No simulation plugin loaded. Add one in the Plugin Manager."));
        }
        if (!i.empty()) i.push_back(Sep());
        {
            Opts opts;
            opts.checkable = true;
            opts.checked = in.simulation_monitor_visible;
            i.push_back(Action(f, "simulation.monitor", "Simulation Monitor",
                               "Shows or hides the Simulation Monitor: the graph run and its Scopes.", opts));
        }
        for (const auto& p : in.plugin_panels) {
            if (p.group != "Simulation") continue;
            Opts opts;
            opts.checkable = true;
            opts.checked = p.checked;
            opts.argument = p.id;
            i.push_back(Action(f, "apps.panel", p.label, "Shows or hides this window.", opts));
        }
        model.menus.push_back(std::move(m));
    }

    // ---------------------------------------------------------------- Apps
    {
        Menu m{"apps", "Apps", {}};
        auto& i = m.items;
        if (in.plugin_panels.empty() && in.loaded_plugins.empty() && in.python_bundles.empty()) {
            i.push_back(Head("No plugins loaded"));
        } else {
            std::string group;
            for (const auto& p : in.plugin_panels) {
                if (p.group != group) {
                    group = p.group;
                    i.push_back(Head(group));
                }
                Opts opts;
                opts.checkable = true;
                opts.checked = p.checked;
                opts.argument = p.id;
                i.push_back(Action(f, "apps.panel", p.label, "Panel provided by a plugin.", opts));
            }
        }
        // Every loaded plugin, windows or not: a node-only plugin is found
        // here too and opens its nodes in the Node Browser.
        if (!in.loaded_plugins.empty()) {
            if (!in.plugin_panels.empty()) i.push_back(Sep());
            i.push_back(Head("Loaded plugins"));
            for (const auto& p : in.loaded_plugins) {
                Opts opts;
                opts.argument = p.id;
                MenuItem item = p.has_nodes
                    ? Action(f, "apps.plugin", p.name, "Shows the nodes this plugin adds in the Node Browser.", opts)
                    : Action(f, "apps.plugin_manager", p.name, "Shows this plugin in the Plugin Manager.", opts);
                item.detail = p.detail;
                i.push_back(std::move(item));
            }
        }
        if (!in.python_bundles.empty()) {
            if (in.loaded_plugins.empty() && !in.plugin_panels.empty()) i.push_back(Sep());
            i.push_back(Head("Python node bundles"));
            for (const auto& b : in.python_bundles) {
                Opts opts;
                opts.argument = b.id;
                MenuItem item =
                    Action(f, "apps.plugin", b.name, "Shows the nodes this Python bundle adds in the Node Browser.", opts);
                item.detail = b.detail;
                i.push_back(std::move(item));
            }
        }
        // Plugin commands for a menu the Engine does not have, under its title
        std::vector<std::string> other_menus;
        for (const auto& c : in.plugin_commands)
            if (!IsMenuTitle(c.menu) &&
                std::find(other_menus.begin(), other_menus.end(), c.menu) == other_menus.end())
                other_menus.push_back(c.menu);
        for (const auto& title : other_menus) {
            i.push_back(Sep());
            i.push_back(Head(title));
            AppendPluginCommands(f, in.plugin_commands, title, i);
        }
        i.push_back(Sep());
        i.push_back(Action(f, "apps.plugin_manager", "Plugin Manager", "Install, enable or remove plugins."));
        model.menus.push_back(std::move(m));
    }

    // ---------------------------------------------------------------- Help
    {
        Menu m{"help", "Help", {}};
        auto& i = m.items;
        {
            std::vector<MenuItem> tutorials;
            for (const auto& t : in.tutorials) {
                Opts opts;
                opts.checkable = true;
                opts.checked = t.checked;
                opts.argument = t.id;
                tutorials.push_back(Action(f, "help.tutorial", t.label, "Starts this tutorial.", opts));
            }
            if (!tutorials.empty()) tutorials.push_back(Sep());
            tutorials.push_back(Action(f, "help.browse_tutorials", "Browse All Tutorials...",
                                       "Opens the tutorial browser."));
            i.push_back(Sub("help.tutorials", "Interactive Tutorials", "Guided tours inside the Engine.",
                            std::move(tutorials)));
        }
        i.push_back(Action(f, "help.shortcuts", "Keyboard Shortcuts...", "Opens the list of shortcuts."));
        i.push_back(Action(f, "help.documentation", "Documentation", "Opens the usage guides in your browser."));
        i.push_back(Action(f, "help.api_reference", "API Reference",
                           "Opens the pycyxwiz Python reference in your browser."));
        i.push_back(Action(f, "help.whats_new", "What's New", "Opens the release notes in your browser."));
        {
            Opts opts;
            opts.why = in.known_issues_found ? "" : "KNOWN_ISSUES.md is not beside the Engine.";
            i.push_back(Action(f, "help.known_issues", "Known Issues",
                               "Opens KNOWN_ISSUES.md, the problems known in this release.", opts));
        }
        i.push_back(Sep());
        i.push_back(Action(f, "help.report_issue", "Report Issue...", "Opens the issue page in your browser."));
        i.push_back(Action(f, "help.open_logs", "Open Logs Folder",
                           "Shows engine_log.txt in the file manager, for a bug report."));
        i.push_back(Planned(f, "help.check_updates", "Check for Updates...", "Looks for a newer Engine."));
        i.push_back(Sep());
        i.push_back(Action(f, "help.about", "About CyxWiz...", "Shows the version and licence."));
        model.menus.push_back(std::move(m));
    }

    // ------------------------------------------------------------- Account
    {
        Menu m{"account", "Account", {}};
        auto& i = m.items;
        if (in.signed_in) {
            // Who you are is drawn above these (the chip's initials, name, email, plan).
            model.account_chip = AccountChipFor(in.user_name, in.user_username, in.user_email, in.user_role);
            i.push_back(Action(f, "account.settings", "Account Settings...", "Opens your account settings."));
            i.push_back(Action(f, "account.wallet", "Wallet", "Opens the Wallet panel."));
            i.push_back(Action(f, "account.servers", "Servers...",
                               "Opens Preferences > Servers: the central server and your nodes."));
            i.push_back(Sep());
            i.push_back(Action(f, "account.sign_out", "Sign out", "Signs you out and clears the session."));
        } else {
            i.push_back(Action(f, "account.sign_in", "Sign in...", "Signs in to CyxWiz."));
        }
        model.account = std::move(m);
    }

    // ------------------------------------------------------------- Project
    if (in.has_project) {
        Menu m{"project", "Project", {}};
        auto& i = m.items;
        i.push_back(Action(f, "file.show_project_folder", "Show in Explorer",
                           "Opens the project's folder in the file manager."));
        i.push_back(Action(f, "file.copy_project_path", "Copy Path", "Copies the project file's path."));
        i.push_back(Sep());
        i.push_back(Action(f, "file.close_project", "Close Project", "Saves settings and closes the project."));
        model.project = std::move(m);
    }

    return model;
}

AccountChip AccountChipFor(const std::string& name, const std::string& username, const std::string& email,
                           const std::string& role) {
    AccountChip chip;
    chip.email = email;
    const auto trim = [](std::string t) {
        while (!t.empty() && t.front() == ' ') t.erase(t.begin());
        while (!t.empty() && t.back() == ' ') t.pop_back();
        return t;
    };
    chip.full_name = trim(name);
    if (chip.full_name.empty()) chip.full_name = trim(username);
    if (chip.full_name.empty()) chip.full_name = email.substr(0, email.find('@'));
    if (chip.full_name.empty()) chip.full_name = "Account";
    // Initials: the first letters of the first two words ("Sample User" -> "SU");
    // one word gives its first two letters.
    std::vector<std::string> words;
    std::string word;
    for (char c : chip.full_name) {
        if (c == ' ' || c == '.' || c == '_' || c == '-') {
            if (!word.empty()) words.push_back(word);
            word.clear();
        } else {
            word += c;
        }
    }
    if (!word.empty()) words.push_back(word);
    const auto upper = [](char c) { return (c >= 'a' && c <= 'z') ? static_cast<char>(c - 'a' + 'A') : c; };
    if (words.size() >= 2) {
        chip.initials = {upper(words[0][0]), upper(words[1][0])};
    } else if (!words.empty()) {
        chip.initials = std::string(1, upper(words[0][0]));
        if (words[0].size() > 1) chip.initials += upper(words[0][1]);
    }
    // On the bar: the whole name when short, else the first word.
    chip.short_name = chip.full_name.size() <= 14 || words.empty() ? chip.full_name : words[0];
    if (chip.short_name.size() > 14) chip.short_name = chip.short_name.substr(0, 13) + "\xe2\x80\xa6";  // "...": one character
    if (role == "pro") chip.plan = "Pro";
    else if (role == "admin") chip.plan = "Admin";
    return chip;
}

namespace {

void Flatten(const std::vector<MenuItem>& items, const std::string& path, std::vector<PaletteEntry>& out) {
    for (const auto& item : items) {
        if (item.kind == Kind::Submenu) {
            Flatten(item.children, path + " > " + item.label, out);
        } else if (item.kind == Kind::Action) {
            PaletteEntry e;
            e.id = item.id;
            e.label = item.label;
            e.menu_path = path;
            e.shortcut = item.shortcut;
            e.hint = item.hint;
            e.argument = item.argument;
            e.enabled = item.enabled;
            e.disabled_reason = item.disabled_reason;
            e.planned = item.planned;
            out.push_back(std::move(e));
        }
    }
}

}  // namespace

std::vector<PaletteEntry> BuildPaletteEntries(const MenuModel& model) {
    std::vector<PaletteEntry> out;
    for (const auto& menu : model.menus) Flatten(menu.items, menu.label, out);
    Flatten(model.account.items, model.account.label, out);
    return out;
}

}  // namespace cyxwiz::menu
