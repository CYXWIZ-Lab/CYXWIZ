#include "outline.h"

#include <algorithm>
#include <cctype>

namespace cyxwiz::editor {

namespace {
int Indent(const std::string& line, int tab) {
    int col = 0;
    for (const char c : line) {
        if (c == ' ') ++col;
        else if (c == '\t') col = (col / tab + 1) * tab;
        else return col;
    }
    return -1;  // blank
}

// "def name(" / "async def name(" / "class Name" at the start of the line.
bool ParseHeader(const std::string& line, Scope& out) {
    size_t i = line.find_first_not_of(" \t");
    if (i == std::string::npos) return false;
    if (line.compare(i, 6, "async ") == 0) i = line.find_first_not_of(' ', i + 6);
    std::string kind;
    if (line.compare(i, 4, "def ") == 0) kind = "def";
    else if (line.compare(i, 6, "class ") == 0) kind = "class";
    else return false;
    i = line.find_first_not_of(' ', i + kind.size());
    if (i == std::string::npos) return false;
    size_t j = i;
    while (j < line.size() && (std::isalnum(static_cast<unsigned char>(line[j])) || line[j] == '_')) ++j;
    if (j == i) return false;
    out.kind = kind;
    out.name = line.substr(i, j - i);
    return true;
}
}  // namespace

std::vector<Scope> EnclosingScopes(const Document& document, int line) {
    std::vector<Scope> out;
    const int tab = document.Settings().tab_size;
    line = std::clamp(line, 0, document.LineCount() - 1);
    // The indentation the cursor's line belongs to (a blank line takes the next code line's).
    int level = -1;
    for (int l = line; l < document.LineCount() && level < 0; ++l) level = Indent(document.Line(l), tab);
    if (level < 0) level = 0;
    // A header on the cursor's own line counts as its scope.
    Scope own;
    if (ParseHeader(document.Line(line), own)) {
        own.line = line;
        out.push_back(own);
        level = Indent(document.Line(line), tab);
    }
    for (int l = line - 1; l >= 0 && level > 0; --l) {
        const std::string& text = document.Line(l);
        const int ind = Indent(text, tab);
        if (ind < 0 || ind >= level) continue;
        level = ind;
        Scope s;
        if (ParseHeader(text, s)) {
            s.line = l;
            out.push_back(s);
        }
    }
    std::reverse(out.begin(), out.end());
    return out;
}

}  // namespace cyxwiz::editor
