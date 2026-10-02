// Variable Explorer presentation (TOFIX133 P5, board 9): what
// python_tools/cyxwiz_vars.py returns, as rows for the shared Variables view
// (the Engine panel and a notebook's Variables). Pure data, no ImGui, no
// Python; parsing never throws.
#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace cyxwiz::vars {

struct Variable {
    std::string name;
    std::string step;       // JSON of its path step, e.g. ["name","df"] or ["item",3]
    std::string type;       // "ndarray[float32]"
    std::string kind;       // number, text, collection, table, array, module, function, class, other
    std::string size;       // "(128, 64)", "3 items", "len=13"
    long long memory = -1;  // bytes of arrays and tables; -1 not known
    std::string value;      // one line
    bool expandable = false;
    bool viewable = false;  // opens in the Data Viewer
    std::string digest;
    long long more = 0;     // a "+ n more" row (children cut at 200); other fields empty
};

std::vector<Variable> ParseVariables(const std::string& json);

// Filter chips (board 9). Modules, functions and classes are under All only.
enum class Chip { All, Tables, Arrays, Numbers, Text, Collections };
constexpr int kChipCount = 6;
const char* ChipLabel(Chip chip);
bool InChip(const Variable& v, Chip chip);
int ChipCount(const std::vector<Variable>& vars, Chip chip);

// Name, type or value contains the text (case-insensitive), and in the chip.
bool Matches(const Variable& v, const std::string& filter, Chip chip);

// "24 B", "32 KB", "705 KB", "1.4 MB", "2.1 GB"; "" when not known.
std::string MemoryText(long long bytes);
// "10 variables · arrays and tables 777 KB" (the memory part only when known).
std::string FooterText(const std::vector<Variable>& vars);

// Names whose value changed since the previous read (new names count).
// The first read of a scope (previous empty) marks nothing.
std::set<std::string> ChangedNames(const std::map<std::string, std::string>& previous_digests,
                                   const std::vector<Variable>& now);
std::map<std::string, std::string> Digests(const std::vector<Variable>& vars);

// Sorting (board 9 columns). Size sorts by element count, Memory by bytes.
enum class Column { Name, Type, Size, Memory, Value };
void Sort(std::vector<Variable>& vars, Column column, bool ascending);
long long ElementCount(const std::string& size);  // "(128, 64)" -> 8192; "3 items" -> 3; -1 unknown

// Paths: a JSON list of steps. A child's path is its parent's plus its step.
std::string PathJson(const std::string& parent_path_json, const std::string& step_json);

// One visible row of the tree.
struct TreeRow {
    const Variable* var = nullptr;
    std::string path;  // JSON
    int depth = 0;
    bool open = false;
    bool loading = false;  // open, children not read yet
};
// Top-level rows (already filtered and sorted) with the open ones' children
// from `children` (keyed by path JSON) under them.
std::vector<TreeRow> Flatten(const std::vector<Variable>& top, const std::set<std::string>& open,
                             const std::map<std::string, std::vector<Variable>>& children);

// "Read after p5_vars.py finished · 12:04:31", "Read on Refresh · 12:05:02".
std::string ReadStatus(const std::string& reason, const std::string& clock);

}  // namespace cyxwiz::vars
