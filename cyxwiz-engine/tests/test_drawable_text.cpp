// Console text a 16-bit ImGui font can draw (TOFIX157): emoji and their
// variation selectors / joiners are left out instead of showing "?"; check,
// cross and warning signs become readable ones. With no font every sign
// falls back to ASCII.
#include "../src/gui/editor_fonts.h"

#include <cstdio>
#include <string>

namespace {

int failures = 0;

void Same(const std::string& got, const std::string& want, const char* what) {
    if (got != want) {
        std::fprintf(stderr, "FAIL: %s: got \"%s\", want \"%s\"\n", what, got.c_str(), want.c_str());
        ++failures;
    }
}

}  // namespace

int main() {
    using cyxwiz::gui::DrawableText;
    Same(DrawableText(nullptr, "plain ASCII stays"), "plain ASCII stays", "ASCII");
    // U+2705 white heavy check mark, U+274C cross mark, U+26A0 + U+FE0F warning.
    Same(DrawableText(nullptr, "\xE2\x9C\x85 saved"), "+ saved", "check mark");
    Same(DrawableText(nullptr, "\xE2\x9D\x8C failed"), "x failed", "cross mark");
    Same(DrawableText(nullptr, "\xE2\x9A\xA0\xEF\xB8\x8F careful"), "! careful", "warning sign and its selector");
    // U+1F680 rocket (4 bytes): left out.
    Same(DrawableText(nullptr, "\xF0\x9F\x9A\x80 start"), " start", "an emoji above U+FFFF");
    // A family emoji: man + ZWJ + woman + ZWJ + girl, all left out.
    Same(DrawableText(nullptr, "a\xF0\x9F\x91\xA8\xE2\x80\x8D\xF0\x9F\x91\xA9\xE2\x80\x8D\xF0\x9F\x91\xA7" "b"), "ab",
         "joined emoji");
    // Latin-1 and other text the font may lack: unchanged when drawable,
    // "?" otherwise (with no font every non-ASCII character is unknown).
    Same(DrawableText(nullptr, "caf\xC3\xA9"), "caf?", "no font: a character it lacks");
    Same(DrawableText(nullptr, "bad \xFF byte"), "bad ? byte", "a broken byte sequence");
    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("drawable text: ASCII, signs, emoji, joiners. OK\n");
    return 0;
}
