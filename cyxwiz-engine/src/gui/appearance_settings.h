#pragma once

// Engine-wide appearance (tofix121): theme, interface text size, code text
// size and sidebar side. One place applies and saves each setting, so View >
// Theme, Preferences > Appearance, the Script Editor font menu and startup
// always agree. Settings live in EngineConfig and apply to every project.

#include "theme.h"

#include <filesystem>
#include <string>
#include <vector>

namespace gui {

// Startup: read EngineConfig and apply theme, code text size and sidebar.
// The interface text size is applied by the font loader (UiTextPixels).
void ApplyStartupAppearance();

ThemePreset CurrentThemePreset();
void SetThemePreset(ThemePreset preset);

// Interface text size in pixels (13/15/17/20). Setting it rebuilds the font
// atlas before the next frame.
int UiTextPixels();
void SetUiTextPixels(int pixels);

// Code text size as an editor font scale (1.0/1.3/1.6/2.0).
float CodeTextScale();
void SetCodeTextScale(float scale);

bool SidebarOnLeft();
void SetSidebarOnLeft(bool left);

void ResetAppearance();

// True once after SetUiTextPixels/ResetAppearance asked for new fonts; the
// application rebuilds the atlas outside a frame.
bool ConsumeFontRebuildRequest();

// The active theme has a light window background.
bool IsLightTheme();

// ---- Custom themes (Theme Editor) ------------------------------------------
// Saved in the user settings folder (<config>/themes/<name>.json), never the
// project. The saved or loaded theme is remembered and re-applied over its
// base preset at startup; choosing a preset forgets it.
std::filesystem::path CustomThemesDirectory();
std::vector<std::filesystem::path> ListCustomThemes();
bool SaveCustomTheme(const std::string& name, std::filesystem::path* saved_path,
                     std::string* error);
bool LoadCustomTheme(const std::filesystem::path& path, std::string* error);
// File name (no extension) of the remembered custom theme, or "".
std::string ActiveCustomThemeName();
// Re-apply the base preset and the remembered custom theme (discards live edits).
void ReapplySavedTheme();

}  // namespace gui
