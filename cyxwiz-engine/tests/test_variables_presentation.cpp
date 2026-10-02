// Variable Explorer presentation (TOFIX133 P5): what cyxwiz_vars.py read
// from p5_vars.py in the scratch project (real output, 2026-10-02).
#include "../src/core/variables_presentation.h"

#include <cstdlib>
#include <iostream>
#include <string>

using namespace cyxwiz::vars;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

const char* kTop = R"J([{"name": "config", "step": ["name", "config"], "type": "dict", "kind": "collection", "size": "4 items", "memory": -1, "value": "{'batch': 64, 'epochs': 20, 'labels': ['negative', 'neutral', 'positive'], 'lr': 0.001}", "expandable": true, "viewable": false, "digest": "dict|4 items|{'batch': 64, 'epochs': 20,"}, {"name": "counts", "step": ["name", "counts"], "type": "Series[int64]", "kind": "table", "size": "(3,)", "memory": 48, "value": "count, dtype int64: 2 \u2192 4000, 0 \u2192 4000, 1 \u2192 2000", "expandable": false, "viewable": true, "digest": "Series|(3,)|count, dtype int64: 2 \u2192 4000"}, {"name": "df", "step": ["name", "df"], "type": "DataFrame", "kind": "table", "size": "(10000, 3)", "memory": 722132, "value": "columns: text (str), label (int64), length (int64)", "expandable": true, "viewable": true, "digest": "(10000, 3)|755063793014"}, {"name": "done", "step": ["name", "done"], "type": "bool", "kind": "number", "size": "", "memory": -1, "value": "True", "expandable": false, "viewable": false, "digest": "bool||True"}, {"name": "epochs", "step": ["name", "epochs"], "type": "int", "kind": "number", "size": "", "memory": -1, "value": "20", "expandable": false, "viewable": false, "digest": "int||20"}, {"name": "labels", "step": ["name", "labels"], "type": "list", "kind": "collection", "size": "(3,)", "memory": -1, "value": "['negative', 'neutral', 'positive']", "expandable": true, "viewable": true, "digest": "list|(3,)|['negative', 'neutral', 'posit"}, {"name": "learning_rate", "step": ["name", "learning_rate"], "type": "float", "kind": "number", "size": "", "memory": -1, "value": "0.001", "expandable": false, "viewable": false, "digest": "float||0.001"}, {"name": "name", "step": ["name", "name"], "type": "str", "kind": "text", "size": "len=13", "memory": -1, "value": "'sentiment run'", "expandable": false, "viewable": false, "digest": "str|len=13|'sentiment run'"}, {"name": "scores", "step": ["name", "scores"], "type": "ndarray[float64]", "kind": "array", "size": "(5000,)", "memory": 40000, "value": "float64 \u00b7 min 0 \u00b7 max 1 \u00b7 mean 0.5", "expandable": false, "viewable": true, "digest": "float64|(5000,)|-4fb0eeb0013f5f96"}, {"name": "weights", "step": ["name", "weights"], "type": "ndarray[float32]", "kind": "array", "size": "(128, 64)", "memory": 32768, "value": "float32 \u00b7 min -3.9 \u00b7 max 3.26 \u00b7 mean 0.00183", "expandable": false, "viewable": true, "digest": "float32|(128, 64)|33b5143bd470fd3d"}])J";
const char* kConfig = R"J([{"name": "'epochs'", "step": ["item", 0], "type": "int", "kind": "number", "size": "", "memory": -1, "value": "20", "expandable": false, "viewable": false, "digest": "int||20"}, {"name": "'lr'", "step": ["item", 1], "type": "float", "kind": "number", "size": "", "memory": -1, "value": "0.001", "expandable": false, "viewable": false, "digest": "float||0.001"}, {"name": "'batch'", "step": ["item", 2], "type": "int", "kind": "number", "size": "", "memory": -1, "value": "64", "expandable": false, "viewable": false, "digest": "int||64"}, {"name": "'labels'", "step": ["item", 3], "type": "list", "kind": "collection", "size": "(3,)", "memory": -1, "value": "['negative', 'neutral', 'positive']", "expandable": true, "viewable": true, "digest": "list|(3,)|['negative', 'neutral', 'positive']"}])J";

const Variable* Find(const std::vector<Variable>& vars, const std::string& name) {
    for (const auto& v : vars)
        if (v.name == name) return &v;
    return nullptr;
}
}  // namespace

int main() {
    auto vars = ParseVariables(kTop);
    Check(vars.size() == 10, "ten variables");
    const Variable* df = Find(vars, "df");
    Check(df && df->type == "DataFrame" && df->kind == "table" && df->size == "(10000, 3)" && df->memory == 722132 &&
              df->expandable && df->viewable && df->step == R"(["name","df"])",
          "DataFrame row");
    const Variable* w = Find(vars, "weights");
    Check(w && w->type == "ndarray[float32]" && w->memory == 32768 && !w->expandable, "array row");

    // Chips: All 10, Tables 2, Arrays 2, Numbers 3, Text 1, Collections 2 (board 9
    // said Numbers 4: a miscount on the mockup; bool, int and float are 3).
    const int expected[] = {10, 2, 2, 3, 1, 2};
    for (int i = 0; i < kChipCount; ++i)
        Check(ChipCount(vars, static_cast<Chip>(i)) == expected[i], std::string("chip ") + ChipLabel(static_cast<Chip>(i)));
    Check(Matches(*df, "", Chip::Tables) && !Matches(*df, "", Chip::Arrays), "chip filter");
    Check(Matches(*df, "LENGTH", Chip::All) && Matches(*w, "float32", Chip::All) && !Matches(*w, "zzz", Chip::All),
          "text filter on name, type or value");

    Check(MemoryText(24) == "24 B" && MemoryText(32768) == "32 KB" && MemoryText(722132) == "705 KB" &&
              MemoryText(1500000) == "1.4 MB" && MemoryText(-1).empty(),
          "memory text");
    Check(FooterText(vars) == "10 variables \xC2\xB7 arrays and tables 776 KB", "footer: " + FooterText(vars));

    // Changed marks: none on the first read; then only what changed.
    Check(ChangedNames({}, vars).empty(), "first read marks nothing");
    auto previous = Digests(vars);
    auto later = vars;
    for (auto& v : later)
        if (v.name == "epochs") v.digest = "int||30";
    Variable added;
    added.name = "model";
    added.digest = "x";
    later.push_back(added);
    const auto changed = ChangedNames(previous, later);
    Check(changed.size() == 2 && changed.count("epochs") && changed.count("model"), "changed names");

    // Sorting.
    Sort(vars, Column::Memory, false);
    Check(vars[0].name == "df" && vars[1].name == "scores" && vars[2].name == "weights", "sort by memory");
    Sort(vars, Column::Size, false);
    Check(vars[0].name == "df", "sort by element count");
    Check(ElementCount("(128, 64)") == 8192 && ElementCount("(5000,)") == 5000 && ElementCount("4 items") == 4 &&
              ElementCount("len=13") == 13 && ElementCount("") == -1,
          "element counts");
    Sort(vars, Column::Name, true);
    Check(vars.front().name == "config" && vars.back().name == "weights", "sort by name");

    // Tree: config open, its children under it; labels inside it closed.
    const std::string config_path = PathJson("", R"(["name","config"])");
    Check(config_path == R"([["name","config"]])", "path json");
    std::map<std::string, std::vector<Variable>> children;
    std::set<std::string> open{config_path};
    auto rows = Flatten(vars, open, children);
    Check(rows.size() == 10 && rows[0].open && rows[0].loading, "open row waits for its children");
    children[config_path] = ParseVariables(kConfig);
    rows = Flatten(vars, open, children);
    Check(rows.size() == 14 && rows[1].depth == 1 && rows[1].var->name == "'epochs'" && !rows[1].open, "children rows");
    Check(rows[4].var->name == "'labels'" && rows[4].path == R"([["name","config"],["item",3]])", "child path");

    // "+ n more" rows stay last and are not variables.
    auto more = ParseVariables(R"J([{"name": "b", "digest": "1"}, {"more": 5}, {"name": "a", "digest": "2"}])J");
    Sort(more, Column::Name, true);
    Check(more[0].name == "a" && more[2].more == 5 && ChipCount(more, Chip::All) == 2, "more row");

    Check(ReadStatus("after p5_vars.py finished", "12:04:31") == "Read after p5_vars.py finished \xC2\xB7 12:04:31", "read status");
    Check(ParseVariables("not json").empty() && ParseVariables("{}").empty(), "bad input");
    std::cout << "variables presentation: chips, filter, memory, changed marks, sort, tree. OK\n";
    return 0;
}
