#pragma once

#include <imgui.h>

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <string_view>

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

// Engine-wide code text size (tofix121): one of kEditorFontScales, shared by
// the Script Editor, Python REPL, Logs, Commands and terminals. Set by
// gui/appearance_settings; read by every code view.
inline float g_code_font_scale = 1.3f;
inline float CodeFontScale() { return g_code_font_scale; }

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
  const auto has = [font](char32_t c) {
    return font && font->FindGlyphNoFallback(static_cast<ImWchar>(c));
  };
  if (codepoint == 0x2605)
    return U'*';
  // Check marks (incl. the emoji U+2705) as the heavy check mark, crosses as
  // the heavy ballot X, when the font has them.
  if (codepoint == 0x2714 || codepoint == 0x2705 || codepoint == 0x2713 ||
      codepoint == 0x2611)
    return codepoint != 0x2714 && has(0x2714) ? U'\u2714' : U'+';
  if (codepoint == 0x2718 || codepoint == 0x274C || codepoint == 0x274E ||
      codepoint == 0x2716 || codepoint == 0x2717)
    return codepoint != 0x2718 && has(0x2718) ? U'\u2718' : U'x';
  if (codepoint == 0x26A0)
    return U'!';
  if (codepoint >= 0x2800 && codepoint <= 0x28FF) {
    constexpr std::array<char32_t, 4> spinner_fallbacks = {U'|', U'/', U'-',
                                                           U'\\'};
    return spinner_fallbacks[static_cast<std::size_t>(codepoint) %
                             spinner_fallbacks.size()];
  }
  return U'?';
}

// Text a font can draw (TOFIX157): ImWchar is 16 bits here, so an emoji
// (above U+FFFF) and the invisible variation selector or joiner that follows
// one would each draw as "?". They are left out; a check, a cross or a
// warning sign becomes a sign the font has (ResolveTerminalDisplayCodepoint);
// any other character the font lacks still shows as "?". Plain ASCII is
// returned as it is.
inline std::string DrawableText(ImFont *font, std::string_view utf8) {
  bool ascii = true;
  for (unsigned char c : utf8)
    if (c >= 0x80) {
      ascii = false;
      break;
    }
  if (ascii)
    return std::string(utf8);
  std::string out;
  out.reserve(utf8.size());
  const auto put = [&out](char32_t c) {
    if (c < 0x80) {
      out += static_cast<char>(c);
    } else if (c < 0x800) {
      out += static_cast<char>(0xC0 | (c >> 6));
      out += static_cast<char>(0x80 | (c & 0x3F));
    } else if (c < 0x10000) {
      out += static_cast<char>(0xE0 | (c >> 12));
      out += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (c & 0x3F));
    }
  };
  for (size_t i = 0; i < utf8.size();) {
    const unsigned char lead = static_cast<unsigned char>(utf8[i]);
    const size_t length = lead < 0x80 ? 1 : (lead >> 5) == 0x6 ? 2 : (lead >> 4) == 0xE ? 3 : (lead >> 3) == 0x1E ? 4 : 1;
    if (i + length > utf8.size() || (length == 1 && lead >= 0x80)) {
      put(U'?');  // a broken byte sequence
      ++i;
      continue;
    }
    char32_t c = length == 1 ? lead : lead & (0xFF >> (length + 1));
    for (size_t k = 1; k < length; ++k)
      c = (c << 6) | (static_cast<unsigned char>(utf8[i + k]) & 0x3F);
    i += length;
    if (c > 0xFFFF || c == 0xFE0F || c == 0xFE0E || c == 0x200D)
      continue;  // emoji and the selectors / joiners that go with them
    put(c < 0x80 ? c : ResolveTerminalDisplayCodepoint(font, c));
  }
  return out;
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

// The mono font at the Engine-wide code text size.
inline ImFont *GetCodeFont() { return GetEditorMonoFont(g_code_font_scale); }

inline float GetEditorMonoFontPixelSize(float scale) {
  return kEditorMonoFontPixels[EditorFontIndexForScale(scale)];
}

} // namespace cyxwiz::gui
