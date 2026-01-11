#pragma once

#include <imgui.h>
#include <string>
#include <functional>
#include <vector>
#include <unordered_map>

namespace gui {

/**
 * Keyboard context - determines which shortcuts are active
 * Higher priority contexts block lower priority ones
 */
enum class KeyboardContext {
    None = 0,

    // Highest priority - modal states
    ModalDialog,           // Save dialogs, confirmations, etc.
    CompletionPopup,       // Auto-completion popup in script editor

    // Panel-specific contexts
    ScriptEditor,          // Script editor panel focused
    NodeEditor,            // Node editor panel focused
    Console,               // Python console focused
    AssetBrowser,          // Asset browser focused
    TableViewer,           // Table viewer focused

    // Lowest priority - global/main window
    MainWindow,            // Default fallback context
};

/**
 * Shortcut definition
 */
struct Shortcut {
    ImGuiKey key;
    bool ctrl = false;
    bool shift = false;
    bool alt = false;
    std::string description;
    std::function<void()> action;

    bool Matches() const {
        ImGuiIO& io = ImGui::GetIO();
        return ImGui::IsKeyPressed(key, false) &&
               io.KeyCtrl == ctrl &&
               io.KeyShift == shift &&
               io.KeyAlt == alt;
    }
};

/**
 * KeyboardShortcutManager - Context-aware keyboard shortcut handling
 *
 * Usage:
 *   1. Call SetActiveContext() based on which panel is focused
 *   2. Call ProcessShortcuts() once per frame
 *   3. Shortcuts only fire in their registered context
 */
class KeyboardShortcutManager {
public:
    static KeyboardShortcutManager& Instance() {
        static KeyboardShortcutManager instance;
        return instance;
    }

    /**
     * Set the currently active keyboard context
     * Call this based on ImGui focus state
     */
    void SetActiveContext(KeyboardContext context) {
        active_context_ = context;
    }

    /**
     * Get the currently active context
     */
    KeyboardContext GetActiveContext() const {
        return active_context_;
    }

    /**
     * Check if a specific context is active
     */
    bool IsContextActive(KeyboardContext context) const {
        return active_context_ == context;
    }

    /**
     * Register a shortcut for a specific context
     */
    void RegisterShortcut(KeyboardContext context, const Shortcut& shortcut) {
        shortcuts_[context].push_back(shortcut);
    }

    /**
     * Clear all shortcuts for a context
     */
    void ClearShortcuts(KeyboardContext context) {
        shortcuts_[context].clear();
    }

    /**
     * Process shortcuts for the current context
     * Returns true if a shortcut was handled
     */
    bool ProcessShortcuts() {
        auto it = shortcuts_.find(active_context_);
        if (it != shortcuts_.end()) {
            for (const auto& shortcut : it->second) {
                if (shortcut.Matches()) {
                    shortcut.action();
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Check if a specific shortcut matches (without executing)
     * Useful for manual shortcut handling
     */
    static bool CheckShortcut(ImGuiKey key, bool ctrl = false, bool shift = false, bool alt = false) {
        ImGuiIO& io = ImGui::GetIO();
        return ImGui::IsKeyPressed(key, false) &&
               io.KeyCtrl == ctrl &&
               io.KeyShift == shift &&
               io.KeyAlt == alt;
    }

    /**
     * Helper to get shortcut string for display (e.g., "Ctrl+N")
     */
    static std::string GetShortcutString(ImGuiKey key, bool ctrl, bool shift, bool alt) {
        std::string result;
        if (ctrl) result += "Ctrl+";
        if (shift) result += "Shift+";
        if (alt) result += "Alt+";
        result += ImGui::GetKeyName(key);
        return result;
    }

private:
    KeyboardShortcutManager() = default;

    KeyboardContext active_context_ = KeyboardContext::MainWindow;
    std::unordered_map<KeyboardContext, std::vector<Shortcut>> shortcuts_;
};

/**
 * Helper macro for checking shortcuts inline
 */
#define SHORTCUT(key, ctrl, shift, alt) \
    KeyboardShortcutManager::CheckShortcut(ImGuiKey_##key, ctrl, shift, alt)

} // namespace gui
