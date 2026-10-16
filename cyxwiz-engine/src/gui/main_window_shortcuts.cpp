// Global keyboard shortcuts, driven by the menu shortcut table (TOFIX129
// step 1.4). Every chord printed in a menu or listed in Preferences >
// Shortcuts is dispatched here, unless the focused window handles it itself.
#include "main_window.h"

#include "../core/keyboard_shortcuts.h"
#include "../core/menu_presentation.h"
#include "panels/toolbar.h"

#include <imgui.h>
#include <spdlog/spdlog.h>

#include <set>
#include <string>
#include <vector>

namespace gui {

namespace {

struct Chord {
    bool ctrl = false;
    bool shift = false;
    bool alt = false;
    ImGuiKey key = ImGuiKey_None;
};

ImGuiKey KeyByName(const std::string& name) {
    if (name.size() == 1) {
        const char c = name[0];
        if (c >= 'A' && c <= 'Z') return static_cast<ImGuiKey>(ImGuiKey_A + (c - 'A'));
        if (c >= '0' && c <= '9') return static_cast<ImGuiKey>(ImGuiKey_0 + (c - '0'));
        if (c == ',') return ImGuiKey_Comma;
        if (c == '/') return ImGuiKey_Slash;
        if (c == '.') return ImGuiKey_Period;
        return ImGuiKey_None;
    }
    if (name.size() >= 2 && name[0] == 'F') {
        const int n = std::atoi(name.c_str() + 1);
        if (n >= 1 && n <= 12) return static_cast<ImGuiKey>(ImGuiKey_F1 + (n - 1));
    }
    if (name == "Delete") return ImGuiKey_Delete;
    if (name == "Tab") return ImGuiKey_Tab;
    if (name == "Esc") return ImGuiKey_Escape;
    if (name == "Enter") return ImGuiKey_Enter;
    if (name == "Space") return ImGuiKey_Space;
    if (name == "Up") return ImGuiKey_UpArrow;
    if (name == "Down") return ImGuiKey_DownArrow;
    return ImGuiKey_None;
}

// "Ctrl+Shift+N" -> modifiers plus key. Unknown keys give ImGuiKey_None.
Chord ParseChord(const std::string& text) {
    Chord chord;
    size_t start = 0;
    while (start <= text.size()) {
        const size_t plus = text.find('+', start);
        const std::string part = text.substr(start, plus == std::string::npos ? std::string::npos : plus - start);
        if (part == "Ctrl") chord.ctrl = true;
        else if (part == "Shift") chord.shift = true;
        else if (part == "Alt") chord.alt = true;
        else if (!part.empty()) chord.key = KeyByName(part);
        if (plus == std::string::npos) break;
        start = plus + 1;
    }
    return chord;
}

bool Pressed(const Chord& chord) {
    if (chord.key == ImGuiKey_None) return false;
    const ImGuiIO& io = ImGui::GetIO();
    if (io.KeyCtrl != chord.ctrl || io.KeyShift != chord.shift || io.KeyAlt != chord.alt) return false;
    return ImGui::IsKeyPressed(chord.key, false);
}

cyxwiz::menu::Context FocusContext(KeyboardContext context) {
    switch (context) {
        case KeyboardContext::NodeEditor: return cyxwiz::menu::Context::StudioCanvas;
        case KeyboardContext::ScriptEditor: return cyxwiz::menu::Context::ScriptEditor;
        default: return cyxwiz::menu::Context::Any;
    }
}

}  // namespace

void MainWindow::HandleGlobalShortcuts() {
    DetectKeyboardContext();
    const KeyboardContext context = KeyboardShortcutManager::Instance().GetActiveContext();

    // Modal dialogs and the completion popup own the keyboard.
    if (context == KeyboardContext::ModalDialog || context == KeyboardContext::CompletionPopup) return;
    if (!toolbar_) return;

    using cyxwiz::menu::Context;
    const Context focus = FocusContext(context);
    const auto& table = cyxwiz::menu::ShortcutTable();

    // Chords the focused window handles itself shadow the global ones with
    // the same keys (F5 runs the script in the Script Editor, not training).
    std::set<std::string> shadowed;
    for (const auto& e : table)
        if (e.context == focus && e.window_handled) shadowed.insert(e.chord);

    for (const auto& e : table) {
        if (e.window_handled || e.planned) continue;
        if (e.context != Context::Any && e.context != focus) continue;
        if (e.context == Context::Any && shadowed.count(e.chord)) continue;
        if (!Pressed(ParseChord(e.chord))) continue;
        spdlog::info("Shortcut {} -> {}", e.chord, e.action_id);
        if (e.action_id == "view.patterns") toolbar_->Dispatch("view.panel", "Patterns");
        else toolbar_->Dispatch(e.action_id);
        return;  // one action per key press
    }
}

}  // namespace gui
