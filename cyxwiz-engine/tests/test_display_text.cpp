// Text as the interface draws it: Arabic joined forms and right-to-left order
// (TOFIX129). Expected values are code points written out, so the test reads
// the same in any editor.
#include "../src/core/display_text.h"

#include <cstdlib>
#include <iostream>
#include <string>

using namespace cyxwiz::core;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

std::string U(std::u32string_view cps) {
    std::string out;
    for (char32_t c : cps) {
        if (c < 0x80) {
            out.push_back(static_cast<char>(c));
        } else if (c < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (c >> 6)));
            out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xE0 | (c >> 12)));
            out.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
        }
    }
    return out;
}
}  // namespace

int main() {
    // Left-to-right text is untouched, also text that is not valid UTF-8
    Check(DisplayText("hello, world (1.5%)") == "hello, world (1.5%)", "Latin unchanged");
    Check(DisplayText(U(U"caf\u00E9 \u03C3 \u2264 \u2192")) == U(U"caf\u00E9 \u03C3 \u2264 \u2192"), "Latin-1, Greek, signs unchanged");
    Check(DisplayText(std::string("ab\xFF\xFE", 4)) == std::string("ab\xFF\xFE", 4), "bytes that are not UTF-8 unchanged");
    Check(DisplayText("").empty(), "empty");
    Check(!HasRightToLeft("plain"), "right-to-left: not in Latin text");
    Check(!HasRightToLeft(U(U"\u0661\u0662")), "right-to-left: Arabic digits alone are not");
    Check(HasRightToLeft(U(U"\u0633")), "right-to-left: an Arabic letter is");

    // salam: seen initial, lam-alef final ligature, meem isolated; drawn right to left
    Check(DisplayText(U(U"\u0633\u0644\u0627\u0645")) == U(U"\uFEE1\uFEFC\uFEB3"), "salam: joined, lam-alef, reversed");
    Check(DisplayText(U(U"\u0644\u0627")) == U(U"\uFEFB"), "lam-alef alone: the isolated ligature");
    // marhaba: meem initial, reh final, hah initial (reh does not join on), beh medial, alef final
    const std::string marhaba = U(U"\uFE8E\uFE92\uFEA3\uFEAE\uFEE3");
    Check(DisplayText(U(U"\u0645\u0631\u062D\u0628\u0627")) == marhaba, "marhaba: initial, final, medial forms");
    // Numbers stay left to right inside right-to-left text
    Check(DisplayText(U(U"\u0645\u0631\u062D\u0628\u0627 123")) == "123 " + marhaba, "number after Arabic: left of it, in order");
    Check(DisplayText(U(U"\u0639\u062F\u062F \u0663\u0664\u0665")) == U(U"\u0663\u0664\u0665 \uFEA9\uFEAA\uFECB"),
          "Arabic-Indic digits kept in order; dal does not join on");
    Check(DisplayText(U(U"\u0627\u0644\u0633\u0639\u0631 12.5%")) == U(U"12.5% \uFEAE\uFECC\uFEB4\uFEDF\uFE8D"),
          "a decimal number and its percent sign stay one number");
    // A left-to-right line keeps its order; the Arabic name in it is reversed
    Check(DisplayText(U(U"Name: \u0639\u0644\u064A")) == U(U"Name: \uFEF2\uFEE0\uFECB"), "Arabic inside a Latin line");
    Check(DisplayText(U(U"abc \u0645\u0631\u062D\u0628\u0627 123 def")) == "abc 123 " + marhaba + " def",
          "a number after Arabic in a Latin line goes with the Arabic");
    // Brackets are mirrored in right-to-left text
    Check(DisplayText(U(U"(\u0645\u0631\u062D\u0628\u0627)")) == "(" + marhaba + ")", "brackets mirrored");
    // A mark stays after its letter (ImGui draws it over the letter before it)
    Check(DisplayText(U(U"\u0628\u0650")) == U(U"\uFE8F\u0650"), "kasra after its beh");
    Check(DisplayText(U(U"\u0628\u0650\u0628")) == U(U"\uFE90\uFE91\u0650"), "a mark does not break the joining");
    // Arabic separators belong to the number: 3.14 and 1,000 in Arabic digits
    Check(DisplayText(U(U"\u0627\u0644\u0639\u062F\u062F \u0663\u066B\u0661\u0664")) == U(U"\u0663\u066B\u0661\u0664 \uFEA9\uFEAA\uFECC\uFEDF\uFE8D"), "Arabic decimal separator: the number stays in order");
    Check(DisplayText(U(U"\u0645\u0631 \u0661\u066C\u0660\u0660\u0660")) == U(U"\u0661\u066C\u0660\u0660\u0660 \uFEAE\uFEE3"), "Arabic thousands separator: one number");
    Check(!HasRightToLeft(U(U"3\u066B14")), "a number with an Arabic separator is not right-to-left text");
    // Bytes that are not text in a right-to-left string become U+FFFD, never a NUL
    Check(DisplayText(std::string("\xC0\x80", 2) + U(U"\u0628")).find('\0') == std::string::npos,
          "an overlong NUL is not decoded as NUL");
    // Hebrew: reversed, no joined forms
    Check(DisplayText(U(U"\u05E9\u05DC\u05D5\u05DD")) == U(U"\u05DD\u05D5\u05DC\u05E9"), "Hebrew reversed");
    // Persian letters with forms of their own (peh, gaf, farsi yeh)
    Check(DisplayText(U(U"\u067E\u06AF\u06CC")) == U(U"\uFBFD\uFB95\uFB58"), "Persian: peh initial, gaf medial, yeh final");
    // Two Arabic words: each word joined, the words in right-to-left order
    Check(DisplayText(U(U"\u0628\u0628 \u0644\u0627")) == U(U"\uFEFB \uFE90\uFE91"), "two words, right to left");

    std::cout << "display text: Arabic forms, lam-alef, right-to-left order, numbers, brackets, marks. OK\n";
    return 0;
}
