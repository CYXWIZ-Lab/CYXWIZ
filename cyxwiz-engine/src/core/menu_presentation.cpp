#include "menu_presentation.h"

#include <utility>

namespace cyxwiz::menu {

namespace {

using Kind = MenuItem::Kind;

struct Opts {
    std::string why;        // disabled reason; empty = enabled
    bool checkable = false;
    bool checked = false;
    bool planned = false;
    std::string argument;
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

        // Everywhere: handled by the main window.
        add(any, "file.new_project", "New Project...", "Ctrl+Shift+N", "File");
        add(any, "file.open_project", "Open Project...", "Ctrl+Shift+O", "File");
        add(any, "script.new", "New Script...", "Ctrl+N", "File, Script");
        add(any, "script.open", "Open Script...", "Ctrl+O", "File, Script");
        add(any, "file.save", "Save (the script when the Script Editor is focused)", "Ctrl+S", "File");
        add(any, "file.save_as", "Save As... (the script when the Script Editor is focused)", "Ctrl+Shift+S", "File");
        add(any, "file.save_all", "Save All", "Ctrl+Alt+S", "File");
        add(any, "file.import_model", "Import Model...", "Ctrl+I", "File");
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
        add(any, "script.python_console", "Open Python Console", "F12", "Script");
        add(any, "help.documentation", "Documentation", "F1", "Help", false, true);
        add(any, "view.fullscreen", "Fullscreen", "F11", "View", false, true);
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
        add(script, "script.completion", "Completion", "Ctrl+Space", "", true);
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

        // Script Editor while debugging: the debugger handles these.
        add(debug, "debug.continue", "Continue", "F5", "", true);
        add(debug, "debug.stop", "Stop debugging", "Shift+F5", "", true);
        add(debug, "debug.step_over", "Step over", "F10", "", true);
        add(debug, "debug.step_into", "Step into", "F11", "", true);
        add(debug, "debug.step_out", "Step out", "Shift+F11", "", true);
        add(debug, "debug.toggle_breakpoint", "Toggle breakpoint", "F9", "", true);
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
    const bool in_script = f == Context::ScriptEditor || f == Context::ScriptDebugging;
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
        i.push_back(Action(f, "file.import_model", "Import Model...", "Loads a model file into the project."));
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
        i.push_back(Planned(f, "view.fullscreen", "Fullscreen", "Fills the screen with the Engine window."));
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
                         Action(f, "train.view_test_results", "View Test Results", "Opens the Test Results panel."),
                         Action(f, "train.compare_test_results", "Compare Test Results...", "Shows the latest test run beside the previous one."),
                         Action(f, "train.export_test_report", "Export Test Report...",
                                "Saves the latest test results as a JSON report."),
                         Sep(),
                         Action(f, "train.load_checkpoint", "Load Checkpoint for Testing...",
                                "Loads saved weights to test them.")}));
        i.push_back(Sep());
        i.push_back(Action(f, "train.dashboard", "Training Dashboard", "Opens the Training panel."));
        i.push_back(Action(f, "train.optimizer_settings", "Optimizer Settings...", "Opens the optimizer settings."));
        i.push_back(Action(f, "train.hyperparam_search", "Hyperparameter Search",
                           "Opens the Hyperparameter Search panel."));
        model.menus.push_back(std::move(m));
    }

    // ---------------------------------------------------------------- Data
    {
        Menu m{"data", "Data", {}};
        auto& i = m.items;
        i.push_back(Action(f, "data.import", "Import Dataset...", "Opens the dataset import."));
        i.push_back(Action(f, "data.create_custom", "Create Custom Dataset...", "Starts a dataset from your own files."));
        i.push_back(Action(f, "data.statistics", "Dataset Statistics", "Shows row counts, columns and value summaries."));
        i.push_back(Sep());
        i.push_back(Sub("data.explore", "Explore", "Look at a table before using it.",
                        {Tool(f, "data.profiler", "Data Profiler"),
                         Tool(f, "data.missing_values", "Missing Value Analysis"),
                         Tool(f, "data.outliers", "Outlier Detection"),
                         Tool(f, "data.correlation", "Correlation Matrix")}));
        i.push_back(Sub("data.transform", "Transform", "Rescale or reshape columns.",
                        {Tool(f, "data.normalization", "Normalization (Min-Max)"),
                         Tool(f, "data.standardization", "Standardization (Z-Score)"),
                         Tool(f, "data.log_transform", "Log Transform"),
                         Tool(f, "data.boxcox", "Box-Cox Transform"),
                         Tool(f, "data.feature_scaling", "Feature Scaling (All Methods)")}));
        i.push_back(Tool(f, "tools.tokenization", "Tokenization"));
        i.push_back(Planned(f, "data.augment", "Augment", "Data augmentation."));
        model.menus.push_back(std::move(m));
    }

    // --------------------------------------------------------------- Tools
    {
        Menu m{"tools", "Tools", {}};
        auto& i = m.items;
        i.push_back(Sub("tools.model", "Model", "Inspect and explain a model.",
                        {Tool(f, "tools.model_summary", "Model Summary"),
                         Tool(f, "tools.flops", "FLOPs Calculator"),
                         Tool(f, "tools.architecture_diagram", "Architecture Diagram"),
                         Tool(f, "tools.lr_finder", "Learning Rate Finder"),
                         Sep(),
                         Tool(f, "tools.gradcam", "Grad-CAM Visualization"),
                         Tool(f, "tools.saliency", "Saliency Maps"),
                         Sep(),
                         Tool(f, "tools.nas", "Neural Architecture Search"),
                         Tool(f, "tools.architecture_suggestions", "Architecture Suggestions"),
                         Sep(),
                         Tool(f, "tools.dnn_inference", "DNN Inference")}));
        i.push_back(Sub("tools.evaluation", "Evaluation", "Measure how well a model performs.",
                        {Tool(f, "tools.cross_validation", "Cross-Validation"),
                         Tool(f, "tools.confusion_matrix", "Confusion Matrix"),
                         Tool(f, "tools.roc_auc", "ROC Curve / AUC"),
                         Tool(f, "tools.pr_curve", "Precision-Recall Curve"),
                         Tool(f, "tools.learning_curves", "Learning Curves"),
                         Tool(f, "tools.feature_importance", "Feature Importance")}));
        i.push_back(Sub("tools.clustering", "Clustering and Projection", "Group rows and reduce dimensions.",
                        {Tool(f, "tools.kmeans", "K-Means Clustering"),
                         Tool(f, "tools.dbscan", "DBSCAN"),
                         Tool(f, "tools.hierarchical", "Hierarchical Clustering"),
                         Tool(f, "tools.gmm", "Gaussian Mixture Models"),
                         Tool(f, "tools.cluster_eval", "Cluster Evaluation"),
                         Sep(),
                         Tool(f, "tools.dim_reduction", "PCA / t-SNE / UMAP")}));
        i.push_back(Sub("tools.statistics", "Statistics", "Describe and test data.",
                        {Tool(f, "tools.descriptive_stats", "Summary Statistics"),
                         Tool(f, "tools.hypothesis_test", "Hypothesis Testing"),
                         Tool(f, "tools.regression", "Linear / Polynomial Regression"),
                         Tool(f, "tools.distribution_fitter", "Distribution Fitter")}));
        i.push_back(Sub("tools.linear_algebra", "Linear Algebra", "Matrix tools.",
                        {Tool(f, "tools.matrix_calculator", "Matrix Calculator"),
                         Tool(f, "tools.eigen", "Eigenvalue Decomposition"),
                         Tool(f, "tools.svd", "SVD (Singular Value Decomposition)"),
                         Tool(f, "tools.qr", "QR Decomposition"),
                         Tool(f, "tools.cholesky", "Cholesky Decomposition")}));
        i.push_back(Sub("tools.signal", "Signal Processing", "Signal tools.",
                        {Tool(f, "tools.fft", "FFT (Fourier Transform)"),
                         Tool(f, "tools.spectrogram", "Spectrogram (STFT)"),
                         Tool(f, "tools.filter_designer", "Filter Designer"),
                         Tool(f, "tools.convolution", "Convolution Calculator"),
                         Tool(f, "tools.wavelet", "Wavelet Transform")}));
        i.push_back(Sub("tools.optimization", "Optimization and Calculus", "Optimisation and calculus tools.",
                        {Tool(f, "tools.gradient_descent", "Gradient Descent Visualizer"),
                         Tool(f, "tools.convexity", "Convexity Analyzer"),
                         Tool(f, "tools.lp", "Linear Programming (LP)"),
                         Tool(f, "tools.qp", "Quadratic Programming (QP)"),
                         Tool(f, "tools.differentiation", "Numerical Differentiation"),
                         Tool(f, "tools.integration", "Numerical Integration")}));
        i.push_back(Sub("tools.time_series", "Time Series", "Time series tools.",
                        {Tool(f, "tools.decomposition", "Time Series Decomposition"),
                         Tool(f, "tools.acf_pacf", "ACF / PACF (Correlogram)"),
                         Tool(f, "tools.stationarity", "Stationarity Testing"),
                         Tool(f, "tools.seasonality", "Seasonality Detection"),
                         Tool(f, "tools.forecasting", "Forecasting")}));
        i.push_back(Sub("tools.text", "Text", "Text tools.",
                        {Tool(f, "tools.tokenization", "Tokenization"),
                         Tool(f, "tools.word_frequency", "Word Frequency"),
                         Tool(f, "tools.tfidf", "TF-IDF Analysis"),
                         Tool(f, "tools.embeddings", "Word Embeddings"),
                         Tool(f, "tools.sentiment", "Sentiment Analysis"),
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
                        {Action(f, "tools.profiler", "Performance Profiler",
                                "Opens the Studio Debugger on its runtime view."),
                         Action(f, "tools.memory_panel", "Memory Visualization", "Opens the Memory panel."),
                         Action(f, "tools.system_monitor", "System Monitor", "Opens the system monitor."),
                         Sep(),
                         Action(f, "tools.clear_cache", "Clear Cache", "Removes the prepared-data cache of this project, after asking."),
                         Action(f, "tools.gc", "Garbage Collection", "Returns unused device memory to the driver and runs a Python garbage collection.")}));
        i.push_back(Sub("tools.diagnostics", "Diagnostics", "Extra logging for support.",
                        {Toggle(f, "tools.idle_log", "Log Idle Mode Transitions",
                                "Logs when the Engine slows its frame rate to save power.", in.idle_log),
                         Toggle(f, "tools.verbose_python", "Verbose Python Logging",
                                "Logs more detail from the Python runtime.", in.verbose_python)}));
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
        Menu m{"simulation", "Simulation", {}};
        auto& i = m.items;
        const std::string need_mujoco = in.mujoco_loaded ? "" : "The MuJoCo plugin is not loaded.";
        i.push_back(Planned(f, "sim.run", "Run Simulation", "Runs the simulation."));
        i.push_back(Planned(f, "sim.pause", "Pause", "Pauses the simulation."));
        i.push_back(Planned(f, "sim.stop", "Stop", "Stops the simulation."));
        i.push_back(Planned(f, "sim.step", "Step Once", "Advances one step."));
        i.push_back(Sep());
        i.push_back(Toggle(f, "sim.viewport", "Open Viewport", "Shows the MuJoCo viewport.",
                           in.mujoco_viewport_visible, need_mujoco));
        i.push_back(Toggle(f, "sim.env_library", "Environment Library", "Shows the environment list.",
                           in.mujoco_env_visible, need_mujoco));
        i.push_back(Sep());
        i.push_back(Planned(f, "sim.load_mjcf", "Load MJCF Model...", "Loads a MuJoCo model file."));
        model.menus.push_back(std::move(m));
    }

    // ---------------------------------------------------------------- Apps
    {
        Menu m{"apps", "Apps", {}};
        auto& i = m.items;
        if (in.plugin_panels.empty()) {
            i.push_back(Head("No plugin apps loaded"));
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
        i.push_back(Planned(f, "help.documentation", "Documentation", "Opens the user guide."));
        i.push_back(Planned(f, "help.api_reference", "API Reference", "Opens the pycyxwiz reference."));
        i.push_back(Sep());
        i.push_back(Action(f, "help.report_issue", "Report Issue...", "Opens the issue page in your browser."));
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
            i.push_back(Head("Signed in as " + in.user_display_name));
            i.push_back(Action(f, "account.settings", "Account Settings...", "Opens your account settings."));
            i.push_back(Action(f, "account.wallet", "Wallet", "Opens the Wallet panel."));
            i.push_back(Sep());
            i.push_back(Action(f, "account.sign_out", "Sign out", "Signs you out and clears the session."));
        } else {
            i.push_back(Action(f, "account.sign_in", "Sign in...", "Signs in to CyxWiz."));
        }
        model.account = std::move(m);
    }

    return model;
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
