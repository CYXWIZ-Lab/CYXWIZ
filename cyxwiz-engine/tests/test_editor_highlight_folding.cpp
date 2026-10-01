// Script Editor colouring and folding (TOFIX133 P1 step 1.2): multi-line
// strings, names and calls, the cache, fold regions and folds that follow edits.
#include "../src/core/editor/folding.h"
#include "../src/core/editor/python_highlight.h"
#include "../src/core/editor/text_document.h"

#include <cstdlib>
#include <iostream>
#include <string>

using namespace cyxwiz::editor;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

TokenKind KindAt(const std::vector<Span>& spans, int col) {
    for (const auto& s : spans)
        if (col >= s.start && col < s.start + s.length) return s.kind;
    return TokenKind::Default;
}
}  // namespace

int main() {
    std::vector<Span> spans;

    // One line: keywords, def names, calls, builtins, constants, comments.
    const std::string line = "def parse(x): return len(DEFAULT_URL) + helper(x)  # done";
    HighlightLine(line, {}, spans);
    Check(KindAt(spans, 0) == TokenKind::Keyword, "def is a keyword");
    Check(KindAt(spans, 4) == TokenKind::Function, "name after def");
    Check(KindAt(spans, 21) == TokenKind::Builtin, "len is a builtin");
    Check(KindAt(spans, 25) == TokenKind::Constant, "UPPER_CASE is a constant");
    Check(KindAt(spans, 41) == TokenKind::Function, "a call is a function");
    Check(KindAt(spans, static_cast<int>(line.find("#"))) == TokenKind::Comment, "comment to the end");

    // Strings across lines carry state; a '#' inside a string is text.
    LineState s = HighlightLine("doc = '''first # not a comment", {}, spans);
    Check(s.triple == '\'' && KindAt(spans, 20) == TokenKind::String, "open ''' string");
    s = HighlightLine("middle line", s, spans);
    Check(s.triple == '\'' && KindAt(spans, 0) == TokenKind::String, "inside the string");
    s = HighlightLine("end''' + x", s, spans);
    Check(s.triple == 0 && KindAt(spans, 0) == TokenKind::String && KindAt(spans, 9) == TokenKind::Default,
          "string ends, code resumes");
    HighlightLine("x = f\"{a}\" + rb'\\d'", {}, spans);
    Check(KindAt(spans, 4) == TokenKind::String && KindAt(spans, 13) == TokenKind::String, "prefixed strings");
    HighlightLine("# %% Training", {}, spans);
    Check(KindAt(spans, 0) == TokenKind::CellMarker, "cell marker line");

    // Cache: typing on one line recolours that line, not the file.
    Document d;
    std::string text;
    for (int i = 0; i < 200; ++i) text += "value_" + std::to_string(i) + " = " + std::to_string(i) + "\n";
    d.SetText(text);
    Highlighter h;
    h.Update(d);
    Check(h.RecolouredLastUpdate() == 201, "first pass colours every line");
    d.SetCursor({100, 0});
    d.Type("x");
    h.Update(d);
    Check(h.RecolouredLastUpdate() == 1, "one edited line recoloured");
    d.SetCursor({10, 0});
    d.Type("\"\"\"");
    h.Update(d);
    Check(h.StateAtStart(11).triple == '"', "an opened string reaches the next lines");

    // Fold regions.
    d.SetText("def a():\n    x = 1\n\n    y = 2\n\n\ndef b():\n    pass\n");
    auto regions = FindFoldRegions(d);
    Check(regions.size() == 2 && regions[0].start == 0 && regions[0].end == 3, "def a folds to its last line");
    Check(regions[1].start == 6 && regions[1].end == 7, "def b");
    d.SetText("\"\"\"Module doc\nsecond line\n\"\"\"\nimport os\n");
    regions = FindFoldRegions(d);
    Check(!regions.empty() && regions[0].start == 0 && regions[0].end == 2, "docstring folds");
    d.SetText("# %% load\nx = 1\n\n# %% train\ny = 2\n");
    regions = FindFoldRegions(d);
    Check(regions.size() == 2 && regions[0].end == 1 && regions[1].start == 3 && regions[1].end == 4, "cell sections");

    // Folds, visible lines, reveal, and folds that move with edits.
    d.SetText("a = 1\ndef f():\n    x = 1\n    y = 2\nb = 2\n");
    FoldState folds;
    folds.Update(d);
    folds.Fold(1);
    Check(folds.IsHidden(2) && folds.IsHidden(3) && !folds.IsHidden(4), "lines 2-3 hidden");
    Check(folds.VisibleLines(d.LineCount()) == std::vector<int>({0, 1, 4, 5}), "visible lines skip the fold");
    Check(folds.HiddenCount(1) == 2, "two hidden lines");
    d.SetCursor({0, 0});
    d.Newline();  // a line above the fold
    folds.Update(d);
    Check(folds.IsFolded(2) && !folds.IsFolded(1), "the fold moved down with its header");
    folds.Reveal(3);
    Check(!folds.IsFolded(2), "a cursor inside opens the fold");
    folds.Fold(2);
    d.SetCursor({2, 0});
    d.DeleteLines();  // the header goes away
    folds.Update(d);
    Check(!folds.IsFolded(2), "a fold without its header is dropped");

    std::cout << "editor highlight and folding: names, strings across lines, cache, regions, folds follow edits. OK\n";
    return 0;
}
