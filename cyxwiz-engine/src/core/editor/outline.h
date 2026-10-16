// Where the cursor is in the code, for the Script Editor's breadcrumbs
// (TOFIX133 P2): the classes and functions that enclose a line, outermost
// first, found by indentation. No ImGui.
#pragma once

#include "text_document.h"

#include <string>
#include <vector>

namespace cyxwiz::editor {

struct Scope {
    std::string kind;  // "class" or "def"
    std::string name;
    int line = 0;      // where it is defined
};

std::vector<Scope> EnclosingScopes(const Document& document, int line);

}  // namespace cyxwiz::editor
