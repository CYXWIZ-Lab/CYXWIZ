#pragma once

// Named interface fonts (TOFIX129 step 0.4). The application registers the
// fonts it loads; screens ask for them by role instead of indexing
// io.Fonts->Fonts or scaling the regular font. A font that was not loaded
// returns nullptr and FontScope then changes nothing.

#include <imgui.h>

#include <array>

namespace cyxwiz::ui {

enum class Font { Regular = 0, Medium, Bold, Heading, Count };

inline std::array<ImFont*, static_cast<size_t>(Font::Count)> g_ui_fonts = {};

inline void RegisterFont(Font role, ImFont* font) { g_ui_fonts[static_cast<size_t>(role)] = font; }
inline void ClearFonts() { g_ui_fonts.fill(nullptr); }
inline ImFont* GetFont(Font role) { return g_ui_fonts[static_cast<size_t>(role)]; }

// Pushes the font for the scope when it is loaded.
class FontScope {
public:
    explicit FontScope(Font role) : pushed_(GetFont(role) != nullptr) {
        if (pushed_) ImGui::PushFont(GetFont(role));
    }
    ~FontScope() {
        if (pushed_) ImGui::PopFont();
    }
    FontScope(const FontScope&) = delete;
    FontScope& operator=(const FontScope&) = delete;

private:
    bool pushed_;
};

}  // namespace cyxwiz::ui
