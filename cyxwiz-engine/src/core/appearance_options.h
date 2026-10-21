#pragma once

// Engine-wide appearance options (tofix121): the allowed interface and code
// text sizes, their labels, and normalization of saved values. Pure data so
// Preferences, startup and tests agree without ImGui.

#include <array>
#include <string>

namespace cyxwiz::appearance {

struct SizeOption {
    const char* label;  // "Default 15 px"
    int pixels;
};

// Interface text (menus, panels, dialogs).
inline constexpr std::array<SizeOption, 4> kUiTextSizes = {{
    {"Small 13 px", 13}, {"Default 15 px", 15}, {"Large 17 px", 17}, {"Extra large 20 px", 20}}};
inline constexpr int kDefaultUiTextPx = 15;

// Code text (Script Editor, Python REPL, Logs, Commands, terminals); the
// editor mono fonts are loaded at these scales.
struct CodeSizeOption {
    const char* label;  // "16 px"
    float scale;
    int pixels;
};
inline constexpr std::array<CodeSizeOption, 4> kCodeTextSizes = {{
    {"14 px", 1.0f, 14}, {"16 px", 1.3f, 16}, {"20 px", 1.6f, 20}, {"24 px", 2.0f, 24}}};
inline constexpr float kDefaultCodeScale = 1.3f;

// Nearest allowed value (bad or old saved values snap to an option).
int NormalizeUiTextPx(int pixels);
float NormalizeCodeScale(float scale);
int UiTextIndex(int pixels);
int CodeScaleIndex(float scale);

// Theme preset index kept in range (count = number of presets).
int NormalizeThemePreset(int preset, int count);

// Relative luminance of an sRGB colour (0 black .. 1 white); a window
// background above 0.5 is a light theme.
float Luminance(float r, float g, float b);
bool IsLightBackground(float r, float g, float b);

}  // namespace cyxwiz::appearance
