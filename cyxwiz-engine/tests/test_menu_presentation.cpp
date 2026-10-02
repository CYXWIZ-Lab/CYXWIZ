// Menu bar presentation model (TOFIX129 step 1.2): the rules the owner set
// for every menu, checked on the model that drives the menus, the Command
// Palette, the shortcut handler and Preferences > Shortcuts.
#include "../src/core/menu_presentation.h"

#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace cyxwiz::menu;

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

MenuInputs SampleInputs() {
    MenuInputs in;
    in.has_project = true;
    in.signed_in = true;
    in.user_display_name = "Owner";
    in.script_open = true;
    in.panels = {{"CyxWiz Studio", "Workspace", true, ""}, {"Properties", "Workspace", true, ""},
                 {"Patterns", "Workspace", false, ""}, {"Table Viewer", "Data", false, ""},
                 {"Training", "Training", false, ""}, {"Jobs", "Network", false, ""},
                 {"Plugin Manager", "Extensions", false, ""}};
    in.recent_projects = {{"D:/projects/berean/berean.cyxwiz", "berean", false, ""},
                          {"D:/projects/mnist/mnist.cyxwiz", "mnist", false, ""}};
    in.themes = {{"CyxWizDark", "CyxWiz Dark", true, "CyxWiz"}, {"CyxWizLight", "CyxWiz Light", false, "CyxWiz"},
                 {"Dracula", "Dracula", false, "Vibrant"}};
    in.icon_packs = {{"0", "FontAwesome (Default)", true, ""}, {"1", "Tabler Icons", false, ""}};
    in.tutorials = {{"getting_started", "Getting Started", true, ""}, {"first_model", "Your First Model", false, ""}};
    in.plugin_panels = {{"mujoco_viewport", "MuJoCo Viewport", false, "Simulation"}};
    return in;
}

void Walk(const std::vector<MenuItem>& items, int depth, const std::string& path,
          const std::function<void(const MenuItem&, int, const std::string&)>& fn) {
    for (const auto& item : items) {
        fn(item, depth, path);
        if (item.kind == MenuItem::Kind::Submenu) Walk(item.children, depth + 1, path + " > " + item.label, fn);
    }
}

void WalkModel(const MenuModel& model, const std::function<void(const MenuItem&, int, const std::string&)>& fn) {
    for (const auto& menu : model.menus) Walk(menu.items, 1, menu.label, fn);
    Walk(model.account.items, 1, model.account.label, fn);
}

const MenuItem* Find(const MenuModel& model, const std::string& id, const std::string& argument = "") {
    const MenuItem* found = nullptr;
    WalkModel(model, [&](const MenuItem& item, int, const std::string&) {
        if (!found && item.id == id && (argument.empty() || item.argument == argument)) found = &item;
    });
    return found;
}

bool EndsWith(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace

int main() {
    const MenuInputs in = SampleInputs();
    const MenuModel model = BuildMenuModel(in);

    // Rule: 12 menus, File to Help, in this order.
    const std::vector<std::string> expected_menus = {"File", "Edit", "View", "Nodes", "Train", "Data",
                                                     "Tools", "Script", "Deploy", "Simulation", "Apps", "Help"};
    Check(model.menus.size() == expected_menus.size(), "12 menus");
    for (size_t i = 0; i < expected_menus.size(); ++i)
        Check(model.menus[i].label == expected_menus[i], "menu order: " + expected_menus[i]);
    Check(model.account.label == "Account", "account menu");

    // Rule: at most two menu levels; every action has an id, a label and a
    // one-sentence hint; disabled items say why.
    std::set<std::string> dialog_labels;
    std::map<std::string, std::string> label_by_id;
    WalkModel(model, [&](const MenuItem& item, int depth, const std::string& path) {
        Check(depth <= 2, "menu depth <= 2 at " + path);
        if (item.kind == MenuItem::Kind::Separator) return;
        Check(!item.label.empty(), "label at " + path);
        if (item.kind == MenuItem::Kind::Header) return;
        Check(!item.id.empty(), "id for " + item.label);
        Check(!item.hint.empty() && EndsWith(item.hint, "."), "hint sentence for " + item.id);
        if (!item.enabled) Check(!item.disabled_reason.empty(), "disabled reason for " + item.id);
        if (item.planned) Check(!item.enabled, "planned item is disabled: " + item.id);
        if (item.kind == MenuItem::Kind::Action && EndsWith(item.label, "...")) dialog_labels.insert(item.id);
        if (item.kind == MenuItem::Kind::Action && item.argument.empty()) {
            auto it = label_by_id.find(item.id);
            if (it == label_by_id.end()) label_by_id[item.id] = item.label;
            else Check(it->second == item.label, "one label per action: " + item.id);
        }
    });

    // Rule: "..." only on items that open a dialog. This set is the decision.
    const std::set<std::string> dialog_openers = {
        "file.new_project", "file.open_project", "script.new", "script.open", "file.save_as", "file.import_model",
        "file.preferences", "edit.find", "edit.replace", "edit.find_in_files", "edit.replace_in_files",
        "edit.go_to_line", "view.command_palette", "view.theme_editor", "nodes.custom_node_editor",
        "train.optimizer_settings", "train.compare_test_results", "train.export_test_report",
        "train.load_checkpoint", "data.import", "data.create_custom", "deploy.connect", "deploy.deploy",
        "deploy.convert_bin_to_dir", "deploy.convert_dir_to_bin", "deploy.marketplace", "sim.load_mjcf",
        "help.browse_tutorials", "help.shortcuts", "help.report_issue", "help.check_updates", "help.about",
        "account.settings"};
    for (const auto& id : dialog_openers) Check(dialog_labels.count(id) == 1, "dialog opener has ...: " + id);
    for (const auto& id : dialog_labels) Check(dialog_openers.count(id) == 1, "... only on dialog openers: " + id);

    // Rule: every action of the old menu bar still exists (the inventory in
    // survey_menu_bar.md, one id per action).
    const std::vector<std::string> inventory = {
        "file.new_project", "file.open_project", "file.close_project", "file.new_window", "file.open_in_new_window",
        "script.new", "script.open", "file.save", "file.save_as", "file.save_all", "file.auto_save",
        "file.import_model", "export.cyxmodel", "export.safetensors", "export.onnx", "export.gguf",
        "file.open_recent", "file.clear_recent", "account.settings", "file.restart", "file.exit",
        "edit.undo", "edit.redo", "edit.cut", "edit.copy", "edit.paste", "edit.delete", "edit.select_all",
        "edit.go_to_line", "edit.duplicate_line", "edit.move_line_up", "edit.move_line_down", "edit.indent",
        "edit.outdent", "edit.uppercase", "edit.lowercase", "edit.titlecase", "edit.sort_asc", "edit.sort_desc",
        "edit.join_lines", "edit.find", "edit.replace", "edit.find_in_files", "edit.replace_in_files",
        "edit.toggle_line_comment", "edit.toggle_block_comment", "file.preferences",
        "view.panel", "view.save_layout", "view.reset_layout", "view.theme", "view.theme_editor",
        "view.studio_minimap", "view.script_minimap", "tools.idle_log", "tools.verbose_python", "view.fullscreen",
        "tools.profiler", "tools.memory_panel", "tools.system_monitor",
        "nodes.add_dense", "nodes.add_conv", "nodes.add_pooling", "nodes.add_dropout", "nodes.add_batchnorm",
        "nodes.add_attention", "nodes.group", "nodes.ungroup", "nodes.duplicate", "nodes.delete", "view.icon_pack",
        "nodes.custom_node_editor",
        "train.compile", "train.local_debug", "train.start", "train.pause", "train.stop", "train.dashboard",
        "train.optimizer_settings",
        "sim.run", "sim.pause", "sim.stop", "sim.step", "sim.viewport", "sim.env_library", "sim.load_mjcf",
        "train.hyperparam_search", "deploy.serving", "deploy.convert_bin_to_dir", "deploy.convert_dir_to_bin",
        "train.load_checkpoint", "train.run_test", "train.quick_test", "train.view_test_results",
        "train.compare_test_results", "train.export_test_report", "tools.clear_cache", "tools.gc",
        "tools.model_summary", "tools.flops", "tools.architecture_diagram", "tools.lr_finder",
        "data.profiler", "data.missing_values", "data.outliers", "data.correlation",
        "data.normalization", "data.standardization", "data.log_transform", "data.boxcox", "data.feature_scaling",
        "tools.descriptive_stats", "tools.hypothesis_test", "tools.regression", "tools.distribution_fitter",
        "tools.cross_validation", "tools.confusion_matrix", "tools.roc_auc", "tools.pr_curve",
        "tools.learning_curves", "tools.kmeans", "tools.dbscan", "tools.hierarchical", "tools.gmm",
        "tools.cluster_eval", "tools.dim_reduction", "tools.feature_importance", "tools.gradcam", "tools.saliency",
        "tools.nas", "tools.architecture_suggestions", "tools.matrix_calculator", "tools.eigen", "tools.svd",
        "tools.qr", "tools.cholesky", "tools.fft", "tools.spectrogram", "tools.filter_designer",
        "tools.convolution", "tools.wavelet", "tools.gradient_descent", "tools.convexity", "tools.lp", "tools.qp",
        "tools.differentiation", "tools.integration", "tools.decomposition", "tools.acf_pacf",
        "tools.stationarity", "tools.seasonality", "tools.forecasting", "tools.tokenization",
        "tools.word_frequency", "tools.tfidf", "tools.embeddings", "tools.sentiment", "tools.language_model",
        "tools.calculator", "tools.unit_converter", "tools.random_generator", "tools.hash_generator",
        "tools.json_viewer", "tools.regex_tester",
        "data.import", "data.create_custom", "data.statistics", "data.augment",
        "script.python_console", "script.run",
        "deploy.connect", "deploy.quantize_int8", "deploy.quantize_int4", "deploy.quantize_fp16", "deploy.deploy",
        "deploy.marketplace", "apps.panel", "apps.plugin_manager",
        "help.tutorial", "help.browse_tutorials", "help.documentation", "help.shortcuts", "help.api_reference",
        "help.report_issue", "help.check_updates", "help.about", "account.sign_out",
        // not in any menu today
        "view.command_palette", "tools.dnn_inference", "script.stop", "account.wallet"};
    for (const auto& id : inventory) Check(Find(model, id) != nullptr, "inventory action present: " + id);

    // Rule: the Planned set is exactly this (kept visible, never enabled).
    std::set<std::string> planned;
    WalkModel(model, [&](const MenuItem& item, int, const std::string&) { if (item.planned) planned.insert(item.id); });
    const std::set<std::string> expected_planned = {
        "view.fullscreen", "export.gguf", "deploy.quantize_int8", "deploy.quantize_int4", "deploy.quantize_fp16",
        "deploy.marketplace", "sim.run", "sim.pause", "sim.stop", "sim.step", "sim.load_mjcf", "help.documentation",
        "help.api_reference", "help.check_updates", "data.augment"};
    Check(planned == expected_planned, "planned set");

    // Rule: every printed shortcut comes from the table, and within one
    // context a chord names one action.
    const auto& table = ShortcutTable();
    Check(!table.empty(), "shortcut table");
    std::map<std::pair<Context, std::string>, std::string> chord_owner;
    for (const auto& e : table) {
        Check(!e.action_id.empty() && !e.action_label.empty() && !e.chord.empty(), "shortcut entry complete");
        auto key = std::make_pair(e.context, e.chord);
        auto it = chord_owner.find(key);
        if (it == chord_owner.end()) chord_owner[key] = e.action_id;
        else Check(it->second == e.action_id, "chord used once per context: " + e.chord);
        if (e.context == Context::Any) Check(!e.window_handled || e.action_id == "file.exit",
                                             "global chord handled globally: " + e.chord);
    }
    WalkModel(model, [&](const MenuItem& item, int, const std::string&) {
        if (item.shortcut.empty()) return;
        const std::string id = (item.id == "view.panel" && item.argument == "Patterns") ? "view.patterns" : item.id;
        bool in_table = false;
        for (const auto& e : table) if (e.action_id == id && e.chord == item.shortcut) in_table = true;
        Check(in_table, "printed shortcut is documented: " + item.id + " " + item.shortcut);
    });
    // The collisions found in the survey are gone.
    Check(ShortcutFor("file.save_as", Context::Any) == "Ctrl+Shift+S", "Save As chord");
    Check(ShortcutFor("canvas.subgraph", Context::StudioCanvas) != "Ctrl+Shift+S", "subgraph moved off Save As");
    Check(ShortcutFor("script.cell_mode", Context::ScriptEditor) != "Ctrl+Shift+N", "cell mode moved off New Project");
    // Notebook command-mode keys are documented (TOFIX133 P0 item 7).
    Check(ShortcutFor("notebook.delete_cell", Context::ScriptNotebook) == "D, D", "notebook delete documented");
    Check(ShortcutFor("notebook.to_markdown", Context::ScriptNotebook) == "M", "notebook M documented");
    Check(ShortcutFor("script.cell_mode", Context::ScriptNotebook) == "Ctrl+Shift+M", "leave notebook documented");
    Check(ShortcutFor("variables.view_data", Context::Variables) == "Enter" &&
              ShortcutFor("variables.delete", Context::Variables) == "Delete",
          "Variable Explorer keys documented");
    Check(std::string(ContextName(Context::Variables)) == "Variable Explorer", "Variable Explorer context name");
    Check(ShortcutFor("table.slice_step", Context::TableViewer) == "Up, Down", "Data Viewer slice keys documented");
    Check(ShortcutFor("editor.select_occurrences", Context::ScriptEditor) == "Ctrl+Shift+L", "editor keys documented");
    Check(ShortcutFor("tools.model_summary", Context::Any).empty(), "Model Summary has no chord");
    Check(ShortcutFor("data.profiler", Context::Any).empty(), "Data Profiler has no chord");
    // Window-wise: F5 and Ctrl+S do the matching action in the Script Editor.
    Check(ShortcutFor("train.start", Context::Any) == "F5" && ShortcutFor("script.run", Context::ScriptEditor) == "F5",
          "F5 runs in both windows");
    Check(ShortcutFor("file.save", Context::Any) == "Ctrl+S" && ShortcutFor("script.save", Context::ScriptEditor) == "Ctrl+S",
          "Ctrl+S saves in both windows");
    // Every menu path in the table names a real menu.
    std::set<std::string> menu_labels;
    for (const auto& m : model.menus) menu_labels.insert(m.label);
    for (const auto& e : table) {
        if (e.menu_path.empty()) continue;
        const std::string first = e.menu_path.substr(0, e.menu_path.find_first_of(" ,"));
        Check(menu_labels.count(first) == 1, "menu path names a menu: " + e.menu_path);
    }

    // Context: nothing focused disables editing with a reason; the Script
    // Editor enables it; the canvas needs a selection for node actions.
    {
        MenuInputs none = in;
        none.focus = Context::Any;
        const auto m0 = BuildMenuModel(none);
        Check(!Find(m0, "edit.undo")->enabled, "undo disabled with no focus");
        Check(!Find(m0, "nodes.group")->enabled, "group disabled off canvas");
        Check(Find(m0, "edit.find_in_files")->enabled, "find in files works everywhere");

        MenuInputs script = in;
        script.focus = Context::ScriptEditor;
        const auto m1 = BuildMenuModel(script);
        Check(Find(m1, "edit.undo")->enabled, "undo enabled in the Script Editor");
        Check(Find(m1, "edit.go_to_line")->enabled, "go to line enabled in the Script Editor");
        Check(Find(m1, "edit.undo")->shortcut == "Ctrl+Z", "undo chord");

        MenuInputs canvas = in;
        canvas.focus = Context::StudioCanvas;
        canvas.selected_nodes = 0;
        const auto m2 = BuildMenuModel(canvas);
        Check(!Find(m2, "nodes.group")->enabled && !Find(m2, "nodes.group")->disabled_reason.empty(),
              "group needs a selection");
        Check(!Find(m2, "edit.replace")->enabled, "replace is script-only");
        canvas.selected_nodes = 2;
        const auto m3 = BuildMenuModel(canvas);
        Check(Find(m3, "nodes.group")->enabled && Find(m3, "edit.cut")->enabled, "group and cut with a selection");
    }

    // Training state: Start disabled while running; Pause becomes Resume.
    {
        MenuInputs running = in;
        running.training = TrainingState::Running;
        const auto m = BuildMenuModel(running);
        Check(!Find(m, "train.start")->enabled && Find(m, "train.stop")->enabled && Find(m, "train.pause")->enabled,
              "running state");
        MenuInputs paused = in;
        paused.training = TrainingState::Paused;
        const auto mp = BuildMenuModel(paused);
        Check(Find(mp, "train.resume") != nullptr && Find(mp, "train.pause") == nullptr, "paused shows Resume");
        Check(!Find(model, "train.stop")->enabled, "stop disabled when idle");
    }

    // Dynamic lists: panels grouped with headers, themes, packs, recent
    // projects, plugin panels, tutorials.
    {
        const MenuItem* panels = Find(model, "view.panels");
        Check(panels && panels->kind == MenuItem::Kind::Submenu, "panels submenu");
        int headers = 0, entries = 0;
        for (const auto& c : panels->children) {
            if (c.kind == MenuItem::Kind::Header) ++headers;
            if (c.kind == MenuItem::Kind::Action) { ++entries; Check(c.checkable, "panel entry is checkable"); }
        }
        Check(headers == 5 && entries == 7, "panel groups and entries");
        Check(Find(model, "view.panel", "Patterns")->shortcut == "Ctrl+Shift+P", "Patterns panel shortcut");
        Check(Find(model, "view.panel", "CyxWiz Studio")->checked, "visible panel is checked");
        Check(Find(model, "view.theme", "CyxWizDark")->checked && !Find(model, "view.theme", "Dracula")->checked,
              "theme check state");
        Check(Find(model, "file.open_recent", in.recent_projects[1].id)->label == "mnist", "recent project entry");
        Check(Find(model, "apps.panel", "mujoco_viewport") != nullptr, "plugin panel entry");
        Check(Find(model, "help.tutorial", "getting_started")->checked, "completed tutorial is checked");
        Check(Find(model, "account.sign_out") != nullptr && Find(model, "account.sign_in") == nullptr,
              "signed in account menu");
        MenuInputs out = in;
        out.signed_in = false;
        Check(Find(BuildMenuModel(out), "account.sign_in") != nullptr, "signed out account menu");
        MenuInputs no_recent = in;
        no_recent.recent_projects.clear();
        Check(!Find(BuildMenuModel(no_recent), "file.recent")->enabled, "empty recent list is disabled with a reason");
    }

    // ONNX export follows the build.
    Check(!Find(model, "export.onnx")->enabled, "ONNX disabled when not compiled");
    MenuInputs onnx = in;
    onnx.onnx_export_available = true;
    Check(Find(BuildMenuModel(onnx), "export.onnx")->enabled, "ONNX enabled when compiled");

    // Palette: one flat list with menu paths.
    const auto palette = BuildPaletteEntries(model);
    Check(palette.size() > 150, "palette has every action");
    for (const auto& e : palette) Check(!e.menu_path.empty() && !e.label.empty(), "palette entry complete: " + e.id);
    bool has_dnn = false;
    for (const auto& e : palette) if (e.id == "tools.dnn_inference") has_dnn = e.menu_path == "Tools > Model";
    Check(has_dnn, "DNN Inference path");

    std::cout << "menu presentation: " << palette.size() << " palette entries, " << table.size()
              << " shortcut entries, " << planned.size() << " planned items. OK\n";
    return 0;
}
