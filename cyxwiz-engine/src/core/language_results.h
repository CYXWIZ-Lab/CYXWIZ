// Results of the Script Editor's language tools (TOFIX133 P3): what
// python_tools/cyxwiz_intel.py returns, as typed structs. Pure data, no
// ImGui, no Python; parsing never throws (bad input gives empty results).
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace cyxwiz::lang {

struct Completion {
    std::string name;      // "load"
    std::string complete;  // what is still to type after the prefix ("ad")
    std::string kind;      // function, class, module, variable, keyword, property, path, ...
    std::string detail;    // signature, module or inferred type
};

struct Description {  // the selected completion's details
    std::string name;
    std::string kind;
    std::string signature;
    std::string doc;
    std::string module;
};

struct Hover {
    std::string name;
    std::string kind;
    std::string signature;
    std::string type;  // inferred type of a variable
    std::string doc;
    std::string module;
    std::string path;  // where it is defined (empty: builtin or this text)
    int line = 0;
};

struct Signature {
    std::string name;
    std::vector<std::string> params;  // "limit: int=10"
    int index = -1;                   // the parameter being typed (-1: none)
    std::string doc;
};

struct Location {
    std::string name;
    std::string path;  // empty: in the text that was sent
    int line = 0;      // 1-based
    int column = 0;    // 0-based
    std::string module;
    bool in_source = false;
};

struct Problem {
    int line = 0;    // 1-based
    int column = 0;  // 0-based
    bool error = false;  // else a warning
    std::string message;
    std::string code;  // pyflakes message class, "syntax", "internal"
};

std::vector<Completion> ParseCompletions(const std::string& json);
Description ParseDescription(const std::string& json);
Hover ParseHover(const std::string& json);
std::vector<Signature> ParseSignatures(const std::string& json);
std::vector<Location> ParseLocations(const std::string& json);
std::vector<Problem> ParseProblems(const std::string& json);

// Kind of a completion as a one-letter chip and a colour role for the list.
struct KindChip {
    const char* letter;
    int role;  // 0 function, 1 class, 2 module, 3 variable, 4 keyword, 5 other
};
KindChip ChipFor(const std::string& kind);

// "2 errors, 3 warnings" / "1 error" / "No problems".
std::string ProblemSummary(const std::vector<Problem>& problems);

// The columns [first, second) a problem underlines: the identifier at its
// column, or for an unused import the imported name (pyflakes points at the
// start of the statement). At least one column.
std::pair<int, int> ProblemRange(const std::string& line_text, const Problem& problem);

}  // namespace cyxwiz::lang
