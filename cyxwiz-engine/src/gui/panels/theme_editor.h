#pragma once

#include "../panel.h"
#include "../theme.h"
#include <string>
#include <vector>
#include <map>

namespace gui {

/**
 * Theme Editor Panel
 * Provides a comprehensive UI for customizing ImGui and ImNodes themes.
 * Changes are applied live for immediate preview.
 */
class ThemeEditorPanel : public cyxwiz::Panel {
public:
    ThemeEditorPanel();
    ~ThemeEditorPanel() override = default;

    void Render() override;

    // Theme import/export
    bool SaveTheme(const std::string& name);
    bool LoadTheme(const std::string& path);
    bool ExportTheme(const std::string& path);

    // Get/set if changes have been made
    bool HasUnsavedChanges() const { return has_unsaved_changes_; }

private:
    // Tab rendering functions
    void RenderPresetSelector();
    void RenderImGuiColorsTab();
    void RenderImNodesColorsTab();
    void RenderStyleTab();
    void RenderSaveLoadTab();

    // Color group rendering
    void RenderColorGroup(const char* group_name, const std::vector<std::pair<ImGuiCol_, const char*>>& colors);
    void RenderImNodesColorGroup(const char* group_name, const std::vector<std::pair<int, const char*>>& colors);

    // Style section rendering
    void RenderRoundingSection();
    void RenderBorderSection();
    void RenderPaddingSection();
    void RenderSizeSection();

    // Backup/restore
    void BackupCurrentStyle();
    void RestoreBackupStyle();

    // Check if style differs from backup
    bool StyleDiffersFromBackup() const;

    // UI state
    int current_tab_ = 0;
    bool has_unsaved_changes_ = false;
    bool show_save_dialog_ = false;
    bool show_load_dialog_ = false;
    char theme_name_buffer_[256] = "";
    char theme_path_buffer_[512] = "";

    // Backup of original style
    ImGuiStyle backup_style_;
    bool has_backup_ = false;

    // Color filter
    char color_filter_[128] = "";

    // Collapsed state for color groups
    std::map<std::string, bool> group_collapsed_;

    // Available custom themes
    std::vector<std::string> custom_themes_;
    int selected_custom_theme_ = -1;

    // ImGui color definitions organized by group
    struct ColorGroupDef {
        const char* name;
        std::vector<std::pair<ImGuiCol_, const char*>> colors;
    };
    static const std::vector<ColorGroupDef> kImGuiColorGroups;

    // ImNodes color definitions
    struct ImNodesColorDef {
        int id;
        const char* name;
    };
    static const std::vector<ImNodesColorDef> kImNodesColors;
};

} // namespace gui
