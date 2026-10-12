#pragma once

// Guided SQL steps (TOFIX146 step 5, mockup v5 board 3). A guided step keeps
// its operations as JSON in the SQL step's `guided_ops` parameter; the
// Engine writes the step's query from them, one named part per operation
// with its plain summary as a comment above it. The pipeline runs that query
// like any SQL step, so Run step, checks, run facts and the plan view work
// unchanged. Pure data: no ImGui, no DuckDB.

#include <nlohmann/json.hpp>

#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace cyxwiz::guided {

// The step parameter holding the operations.
inline constexpr const char* kGuidedOpsParameter = "guided_ops";

struct OpInfo {
    const char* id;        // "keep_rows"
    const char* name;      // "Keep rows where"
    const char* category;  // "Rows"
    const char* hint;      // "conditions with all / any groups"
};

// The operation library in display order, grouped by category.
const std::vector<OpInfo>& Library();
const OpInfo* FindOp(std::string_view id);

struct Operation {
    std::string type;  // an OpInfo id
    bool on = true;
    nlohmann::json params = nlohmann::json::object();
};

struct GuidedStep {
    std::vector<Operation> ops;
};

// Defaults a new operation starts with; `column` (may be empty) is a likely
// column for it (the first text column of its input).
nlohmann::json DefaultParams(std::string_view type, const std::string& column);

std::string ToJson(const GuidedStep& step);
std::optional<GuidedStep> FromJson(std::string_view text, std::string* error = nullptr);

// What the operations know about their inputs: the step's input names in
// pin order and, when known, each input's columns.
struct Context {
    std::vector<std::string> inputs;                          // aliases, the first is the main input
    std::map<std::string, std::vector<std::string>> columns;  // alias -> columns
    // Input columns holding whole numbers: their sums stay whole numbers
    // (BIGINT); other sums are decimals (DuckDB's own sum of a whole
    // number is HUGEINT, which a SQL step cannot return).
    std::set<std::string> whole_number_columns;
};

// Columns after each operation, tracked without running anything.
struct Columns {
    bool known = true;  // false when an appended table's columns are not known
    std::vector<std::string> names;
};
// columns[i] is what operation i receives; columns[ops.size()] the result.
std::vector<Columns> TrackColumns(const GuidedStep& step, const Context& context);

struct Generated {
    std::string sql;
    std::vector<std::string> part_names;  // per operation; empty when it is off or has an error
    std::vector<std::string> errors;      // per operation; empty when it is fine
    bool ok = true;                       // false when any operation that is on has an error
};

Generated GenerateSql(const GuidedStep& step, const Context& context);

// One plain line about what the operation does.
std::string Summarise(const Operation& op);

// Ready-made starting points: id, name and the operations.
struct Template {
    const char* id;
    const char* name;
    const char* description;
};
const std::vector<Template>& Templates();
GuidedStep MakeTemplate(std::string_view id, const std::string& text_column);

}  // namespace cyxwiz::guided
