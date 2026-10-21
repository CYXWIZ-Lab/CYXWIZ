// Find/Replace (TOFIX133 P0 item 13): regex Replace replaces, ReplaceAll
// ends on empty matches, whole word keeps looking; columns follow tabs/UTF-8.
#include "../src/core/text_search.h"

#include <cstdlib>
#include <iostream>
#include <string>

using namespace cyxwiz::textsearch;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}
}  // namespace

int main() {
    Options plain;
    Options word;
    word.whole_word = true;
    Options re;
    re.regex = true;
    re.case_sensitive = true;

    // Whole word: the first hit is part of "item_count", the second is a word.
    const std::string code = "item_count = 0\nfor item in items:\n";
    auto m = FindNext(code, "item", 0, word);
    Check(m && code.substr(m->pos, 4) == "item" && m->pos == 19, "whole word skips item_count and finds item");
    Check(FindAll(code, "item", word).size() == 1, "underscore and letters are word characters");

    // Regex Replace: the selection is a regex match, groups expand.
    auto r = ReplacementFor("lr=0.01", R"(lr=(\d+\.\d+))", "learning_rate=$1", re);
    Check(r && *r == "learning_rate=0.01", "regex replace uses the match, not the pattern text");
    Check(!ReplacementFor("lr=abc", R"(lr=(\d+\.\d+))", "x", re), "non-matching selection is not replaced");
    Check(ReplacementFor("Item", "item", "x", plain).value_or("") == "x", "plain, case-insensitive");

    // ReplaceAll with a pattern that can match empty text ends.
    int count = -1;
    std::string out = ReplaceAll("a1b22c", R"(\d*)", "#", re, &count);
    Check(count == 2 && out == "a#b#c", "empty matches are skipped, loop ends: " + out);

    out = ReplaceAll("x = x + xx", "x", "y", word, &count);
    Check(count == 2 && out == "y = y + xx", "whole-word replace all");

    std::string error;
    Check(FindAll("abc", "(", re, &error).empty() && !error.empty(), "invalid regex reports an error");

    // Previous wraps to the last match.
    m = FindPrevious("ab ab ab", "ab", 0, plain);
    Check(m && m->pos == 6, "previous wraps to the end");
    m = FindPrevious("ab ab ab", "ab", 5, plain);
    Check(m && m->pos == 3, "previous before an offset");

    // Columns: a tab advances to the next stop; UTF-8 characters count once.
    const std::string text = "x\n\tif t:\n\xC3\xA9t\xC3\xA9 = 1\n";
    Position p = ToPosition(text, 3, 4);  // the 'i' after the tab
    Check(p.line == 1 && p.column == 4, "tab expands to column 4");
    const size_t e_at = text.find("t\xC3\xA9");
    p = ToPosition(text, e_at, 4);
    Check(p.line == 2 && p.column == 1, "UTF-8 e-acute is one column");
    Check(ToOffset(text, {1, 4}, 4) == 3, "column 4 on the tab line is the 'i'");
    Check(ToOffset(text, {2, 1}, 4) == e_at, "column 1 after a two-byte character");
    Check(ToOffset(text, {0, 99}, 4) == 1, "column past the end clamps to the line end");

    std::cout << "text search: whole word, regex replace, empty matches, wrap, columns. OK\n";
    return 0;
}
