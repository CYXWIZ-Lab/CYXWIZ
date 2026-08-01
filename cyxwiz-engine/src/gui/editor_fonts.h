#pragma once

#include <array>
#include <cmath>

struct ImFont;

namespace cyxwiz::gui {

inline constexpr std::array<float, 4> kEditorFontScales = {1.0f, 1.3f, 1.6f, 2.0f};
inline constexpr std::array<float, 4> kEditorMonoFontPixels = {14.0f, 16.0f, 20.0f, 24.0f};

inline std::array<ImFont*, kEditorFontScales.size()> g_editor_mono_fonts = {};

inline int EditorFontIndexForScale(float scale) {
    int best_index = 0;
    float best_distance = std::abs(scale - kEditorFontScales[0]);
    for (int i = 1; i < static_cast<int>(kEditorFontScales.size()); ++i) {
        float distance = std::abs(scale - kEditorFontScales[i]);
        if (distance < best_distance) {
            best_distance = distance;
            best_index = i;
        }
    }
    return best_index;
}

inline void RegisterEditorMonoFont(float scale, ImFont* font) {
    g_editor_mono_fonts[EditorFontIndexForScale(scale)] = font;
}

inline ImFont* GetEditorMonoFont(float scale) {
    ImFont* font = g_editor_mono_fonts[EditorFontIndexForScale(scale)];
    return font ? font : g_editor_mono_fonts[0];
}

inline float GetEditorMonoFontPixelSize(float scale) {
    return kEditorMonoFontPixels[EditorFontIndexForScale(scale)];
}

} // namespace cyxwiz::gui
