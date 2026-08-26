#pragma once

#include <imgui.h>

#include <array>
#include <cmath>
#include <limits>

namespace cyxwiz::gui {

inline constexpr std::array<float, 4> kEditorFontScales = {1.0f, 1.3f, 1.6f,
                                                           2.0f};
inline constexpr std::array<float, 4> kEditorMonoFontPixels = {14.0f, 16.0f,
                                                               20.0f, 24.0f};
inline constexpr std::array<ImWchar, 17> kTerminalTextGlyphRanges = {
    0x0020, 0x00FF, // Basic Latin and Latin-1 Supplement
    0x2000, 0x206F, // General Punctuation
    0x2190, 0x21FF, // Arrows
    0x2500, 0x259F, // Box Drawing and Block Elements
    0x25A0, 0x25FF, // Geometric Shapes
    0x2600, 0x26FF, // Miscellaneous Symbols
    0x2700, 0x27BF, // Dingbats
    0x2800, 0x28FF, // Braille Patterns used by terminal spinners
    0};
inline constexpr std::array<ImWchar, 9> kTerminalSymbolFallbackGlyphRanges = {
    0x2605, 0x2605, // Black Star
    0x2714, 0x2714, // Heavy Check Mark
    0x2718, 0x2718, // Heavy Ballot X
    0x2800, 0x28FF, // Braille Patterns
    0};

inline std::array<ImFont *, kEditorFontScales.size()> g_editor_mono_fonts = {};

inline void ClearEditorMonoFonts() { g_editor_mono_fonts.fill(nullptr); }

inline ImFont *AddTerminalCapableMonoFont(ImFontAtlas *atlas,
                                          const char *mono_path,
                                          const char *symbol_fallback_path,
                                          float pixel_size,
                                          const ImFontConfig *font_config) {
  ImFont *font = atlas->AddFontFromFileTTF(mono_path, pixel_size, font_config,
                                           kTerminalTextGlyphRanges.data());
  if (!font || !symbol_fallback_path || symbol_fallback_path[0] == '\0')
    return font;

  ImFontConfig fallback_config;
  fallback_config.MergeMode = true;
  fallback_config.PixelSnapH = true;
  if (font_config)
    fallback_config.RasterizerDensity = font_config->RasterizerDensity;
  fallback_config.GlyphMinAdvanceX = pixel_size;
  fallback_config.GlyphMaxAdvanceX = pixel_size;
  atlas->AddFontFromFileTTF(symbol_fallback_path, pixel_size, &fallback_config,
                            kTerminalSymbolFallbackGlyphRanges.data());
  return font;
}

inline char32_t ResolveTerminalDisplayCodepoint(ImFont *font,
                                                char32_t codepoint) {
  if (font &&
      codepoint <= static_cast<char32_t>(std::numeric_limits<ImWchar>::max()) &&
      font->FindGlyphNoFallback(static_cast<ImWchar>(codepoint))) {
    return codepoint;
  }
  if (codepoint == 0x2605)
    return U'*';
  if (codepoint == 0x2714)
    return U'+';
  if (codepoint == 0x2718)
    return U'x';
  if (codepoint >= 0x2800 && codepoint <= 0x28FF) {
    constexpr std::array<char32_t, 4> spinner_fallbacks = {U'|', U'/', U'-',
                                                           U'\\'};
    return spinner_fallbacks[static_cast<std::size_t>(codepoint) %
                             spinner_fallbacks.size()];
  }
  return U'?';
}

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

inline void RegisterEditorMonoFont(float scale, ImFont *font) {
  g_editor_mono_fonts[EditorFontIndexForScale(scale)] = font;
}

inline ImFont *GetEditorMonoFont(float scale) {
  ImFont *font = g_editor_mono_fonts[EditorFontIndexForScale(scale)];
  return font ? font : g_editor_mono_fonts[0];
}

inline float GetEditorMonoFontPixelSize(float scale) {
  return kEditorMonoFontPixels[EditorFontIndexForScale(scale)];
}

} // namespace cyxwiz::gui
