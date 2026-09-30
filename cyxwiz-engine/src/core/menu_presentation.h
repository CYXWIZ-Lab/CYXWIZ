// Menu bar presentation model (TOFIX129 step 1.2).
//
// One table of actions drives the main menu bar, the Command Palette, the
// global shortcut handler and Preferences > Shortcuts, so a shortcut printed
// in a menu is always the one that is handled and documented. Pure data in,
// data out: no ImGui, no globals, no backend calls. The GUI layer maps action
// ids to handlers and draws the model.
#pragma once

#include <string>
#include <vector>

namespace cyxwiz::menu {

// Which dock window has keyboard focus. Shortcuts are window-wise: the same
// chord can do the matching action in another window (F5 runs a script in
// the Script Editor and starts training elsewhere).
enum class Context { Any, StudioCanvas, ScriptEditor, ScriptDebugging };

const char* ContextName(Context context);

enum class TrainingState { Idle, Running, Paused };

// A panel the sidebar knows about, in sidebar order.
struct PanelEntry {
    std::string name;      // window title as registered with the sidebar
    std::string group;     // Workspace, Data, Training, Network, Extensions
    bool visible = false;
    std::string shortcut;  // printed beside the entry when set
};

// A dynamic entry: a theme preset, an icon pack, a tutorial, a recent
// project, a plugin panel.
struct NamedEntry {
    std::string id;
    std::string label;
    bool checked = false;
    std::string group;  // plugin panels: category
};

struct MenuInputs {
    Context focus = Context::Any;
    bool has_project = false;
    int selected_nodes = 0;
    TrainingState training = TrainingState::Idle;
    bool signed_in = false;
    std::string user_display_name;
    bool script_open = false;
    bool script_running = false;
    bool onnx_export_available = false;
    bool mujoco_loaded = false;
    bool mujoco_viewport_visible = false;
    bool mujoco_env_visible = false;
    bool auto_save = false;
    bool studio_minimap = false;
    bool script_minimap = false;
    bool idle_log = false;
    bool verbose_python = false;
    std::vector<PanelEntry> panels;
    std::vector<std::string> recent_projects;
    std::vector<NamedEntry> themes;
    std::vector<NamedEntry> icon_packs;
    std::vector<NamedEntry> tutorials;      // checked = completed
    std::vector<NamedEntry> plugin_panels;  // checked = visible, group = category
};

struct MenuItem {
    enum class Kind { Action, Separator, Header, Submenu };
    Kind kind = Kind::Action;
    std::string id;         // action id, for example "file.new_project"
    std::string label;      // as shown, with "..." when it opens a dialog
    std::string shortcut;   // printed chord, empty when none
    std::string hint;       // one sentence for the status bar
    std::string argument;   // parametrised actions: preset, path, panel name
    bool enabled = true;
    std::string disabled_reason;
    bool checkable = false;
    bool checked = false;
    bool planned = false;   // kept visible, not available yet
    std::vector<MenuItem> children;  // Submenu only
};

struct Menu {
    std::string id;
    std::string label;
    std::vector<MenuItem> items;
};

struct MenuModel {
    std::vector<Menu> menus;  // File to Help, in drawn order
    Menu account;             // the avatar menu at the right of the bar
};

MenuModel BuildMenuModel(const MenuInputs& inputs);

// One row of Preferences > Shortcuts, and the source of every printed chord.
struct ShortcutEntry {
    Context context = Context::Any;
    std::string action_id;
    std::string action_label;
    std::string chord;
    std::string menu_path;      // where the action also appears, may be empty
    bool window_handled = false; // the focused window handles the key itself;
                                 // listed here, not dispatched by the global handler
    bool planned = false;
};

const std::vector<ShortcutEntry>& ShortcutTable();

// The chord printed beside an action for the given focus: the entry for that
// context, else the Any entry, else empty.
std::string ShortcutFor(const std::string& action_id, Context focus);

// A flattened view for the Command Palette: every action with its menu path.
struct PaletteEntry {
    std::string id;
    std::string label;
    std::string menu_path;  // "File > Export Model"
    std::string shortcut;
    std::string hint;
    std::string argument;
    bool enabled = true;
    std::string disabled_reason;
    bool planned = false;
};

std::vector<PaletteEntry> BuildPaletteEntries(const MenuModel& model);

}  // namespace cyxwiz::menu
