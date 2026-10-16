#include "sql_plan.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <map>
#include <set>

namespace cyxwiz::sqlplan {

namespace {
using nlohmann::json;

std::string Lower(std::string text) {
    for (char& c : text) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return text;
}

// One line, at most `limit` characters.
std::string Shorten(const std::string& text, size_t limit) {
    std::string out;
    bool space = false;
    for (const char c : text) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            space = !out.empty();
            continue;
        }
        if (space) out += ' ';
        space = false;
        out += c;
    }
    if (out.size() > limit) out = out.substr(0, limit) + "...";
    return out;
}

// A list is joined with `separator`: ", " for column lists; "\n" for an
// expression, which DuckDB splits into list items at its newline characters.
std::string DetailText(const json& value, const char* separator = "\n") {
    if (value.is_string()) return value.get<std::string>();
    if (value.is_array()) {
        std::string joined;
        bool first = true;
        for (const auto& item : value) {
            if (!first) joined += separator;
            joined += DetailText(item, separator);
            first = false;
        }
        return joined;
    }
    return value.dump();
}

int64_t ToCount(const json& value) {
    if (value.is_number_integer()) return value.get<int64_t>();
    if (value.is_string()) {
        try {
            return std::stoll(value.get<std::string>());
        } catch (...) {
            return -1;
        }
    }
    return -1;
}

struct Node {
    std::string name;
    std::vector<std::pair<std::string, std::string>> details;
    std::vector<std::string> columns;
    int64_t rows = -1;
    int64_t estimated = -1;
    double seconds = -1.0;
    std::vector<Node> children;
};

bool IsWrapper(const std::string& name) {
    return name == "EXPLAIN_ANALYZE" || name == "RESULT_COLLECTOR" || name == "QUERY" || name.empty();
}

Node ReadNode(const json& j, bool timed) {
    Node node;
    node.name = j.value(timed ? "operator_name" : "name", std::string());
    while (!node.name.empty() && node.name.back() == ' ') node.name.pop_back();
    if (const auto extra = j.find("extra_info"); extra != j.end() && extra->is_object()) {
        for (const auto& [key, value] : extra->items()) {
            if (key == "Estimated Cardinality") {
                node.estimated = ToCount(value);
                continue;
            }
            if (key == "Projections" && value.is_array()) {
                for (const auto& column : value) node.columns.push_back(DetailText(column));
            } else if (key == "Projections" && value.is_string()) {
                node.columns.push_back(value.get<std::string>());
            }
            const bool list = key == "Projections" || key == "Order By" || key == "Aggregates" || key == "Groups";
            node.details.emplace_back(key, DetailText(value, list ? ", " : "\n"));
        }
    }
    if (timed) {
        node.rows = j.contains("operator_cardinality") ? ToCount(j["operator_cardinality"]) : -1;
        node.seconds = j.value("operator_timing", -1.0);
    }
    if (const auto children = j.find("children"); children != j.end() && children->is_array()) {
        for (const auto& child : *children) node.children.push_back(ReadNode(child, timed));
    }
    return node;
}

std::string DetailValue(const PlanOperator& op, const char* key) {
    for (const auto& [k, v] : op.details) {
        if (k == key) return v;
    }
    return {};
}

bool IsInternalProjection(const std::vector<std::string>& columns) {
    if (columns.empty()) return false;
    for (const auto& column : columns) {
        if (column.rfind("#", 0) != 0 && column.find("__internal") == std::string::npos) return false;
    }
    return true;
}

// Children first, then the node: the order rows flow. The inputs of a node
// with several inputs sit one level deeper. Returns the node's output
// columns: its own list, else (a filter, a sort) its inputs' columns.
std::vector<std::string> Flatten(const Node& node, int depth, Plan& plan) {
    if (IsWrapper(node.name)) {
        std::vector<std::string> columns;
        for (const auto& child : node.children) columns = Flatten(child, depth, plan);
        return columns;
    }
    const int child_depth = node.children.size() > 1 ? depth + 1 : depth;
    int64_t rows_in = -1;
    std::vector<std::string> input_columns;
    for (const auto& child : node.children) {
        const auto columns = Flatten(child, child_depth, plan);
        if (child.rows >= 0) rows_in = std::max<int64_t>(rows_in, 0) + child.rows;
        input_columns.insert(input_columns.end(), columns.begin(), columns.end());
    }
    PlanOperator op;
    op.name = node.name;
    op.details = node.details;
    op.columns = node.columns;
    op.depth = depth;
    op.rows = node.rows;
    op.rows_in = node.children.empty() ? -1 : rows_in;
    op.estimated = node.estimated;
    op.seconds = node.seconds;
    op.internal = op.name == "PROJECTION" && IsInternalProjection(op.columns);
    // Columns a projection makes (not passed through from its input).
    std::vector<std::string> made;
    if (op.name == "PROJECTION" && !op.internal) {
        const std::set<std::string> inputs(input_columns.begin(), input_columns.end());
        for (const auto& column : op.columns) {
            if (!inputs.count(column) && column.rfind("#", 0) != 0) made.push_back(column);
        }
    }
    if (op.internal) {
        op.summary = "(internal: DuckDB reorders or compresses columns)";
    } else if (op.name == "PROJECTION") {
        std::string list;
        for (const auto& column : made) list += (list.empty() ? "" : ", ") + column;
        op.summary = made.empty() ? "passes " + std::to_string(op.columns.size()) + " columns"
                                  : "makes " + Shorten(list, 60);
    } else if (op.name == "FILTER") {
        op.summary = "keeps rows where " + Shorten(DetailValue(op, "Expression"), 70);
    } else if (op.name.find("JOIN") != std::string::npos) {
        op.summary = "joins on " + Shorten(DetailValue(op, "Conditions"), 70);
    } else if (op.name == "UNNEST") {
        op.summary = "one row per list element";
    } else if (op.name == "ORDER_BY") {
        op.summary = "sorts by " + Shorten(DetailValue(op, "Order By"), 70);
    } else if (op.name == "COLUMN_DATA_SCAN") {
        op.summary = "rows written in the query (VALUES)";
    } else if (op.name.find("SCAN") != std::string::npos) {
        const std::string table = DetailValue(op, "Table");
        op.summary = table.empty() ? Shorten(DetailValue(op, "Function"), 40) : "reads " + table;
    } else if (!op.details.empty()) {
        op.summary = Shorten(op.details.front().second, 70);
    }
    op.made_columns = made;
    std::vector<std::string> output = op.columns.empty() || op.internal ? input_columns : op.columns;
    plan.ops.push_back(std::move(op));
    return output;
}

// String literals of a text ('...' with '' inside), outside comments.
std::vector<std::string> Literals(std::string_view text) {
    std::vector<std::string> out;
    for (size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '-' && i + 1 < text.size() && text[i + 1] == '-') {
            while (i < text.size() && text[i] != '\n') ++i;
        } else if (text[i] == '/' && i + 1 < text.size() && text[i + 1] == '*') {
            const size_t end = text.find("*/", i + 2);
            i = end == std::string_view::npos ? text.size() : end + 1;
        } else if (text[i] == '\'') {
            std::string literal;
            size_t k = i + 1;
            for (; k < text.size(); ++k) {
                if (text[k] == '\'') {
                    if (k + 1 < text.size() && text[k + 1] == '\'') {
                        literal += "''";
                        ++k;
                        continue;
                    }
                    break;
                }
                literal += text[k];
            }
            out.push_back(literal);
            i = k;
        }
    }
    return out;
}

bool DefinesColumn(const std::string& lower_part, const std::string& column) {
    const std::string needle = "as " + Lower(column);
    for (size_t at = lower_part.find(needle); at != std::string::npos; at = lower_part.find(needle, at + 1)) {
        const bool before = at == 0 || !std::isalnum(static_cast<unsigned char>(lower_part[at - 1]));
        const size_t after = at + needle.size();
        const bool end = after >= lower_part.size() ||
                         !(std::isalnum(static_cast<unsigned char>(lower_part[after])) || lower_part[after] == '_');
        if (before && end) return true;
    }
    return false;
}

// The text call `name(...)` starting at `at`, with balanced brackets
// (quotes respected), or empty.
std::string CallAt(const std::string& text, size_t at) {
    size_t open = at;
    while (open < text.size() && (std::isalnum(static_cast<unsigned char>(text[open])) || text[open] == '_')) ++open;
    if (open >= text.size() || text[open] != '(') return {};
    int depth = 0;
    bool quoted = false;
    for (size_t i = open; i < text.size(); ++i) {
        const char c = text[i];
        if (c == '\'') quoted = !quoted;
        if (quoted) continue;
        if (c == '(') ++depth;
        if (c == ')' && --depth == 0) return text.substr(at, i - at + 1);
    }
    return {};
}

size_t Count(const std::string& text, const std::string& what) {
    size_t n = 0;
    for (size_t at = text.find(what); at != std::string::npos; at = text.find(what, at + what.size())) ++n;
    return n;
}

std::string Thousands(int64_t value) { return sqlstep::FormatCount(value); }
}  // namespace

Plan ParsePlanJson(std::string_view text) {
    Plan plan;
    const json j = json::parse(text.begin(), text.end(), nullptr, false);
    if (j.is_discarded()) {
        plan.error = "the plan is not valid JSON";
        return plan;
    }
    std::vector<Node> roots;
    if (j.is_array()) {
        for (const auto& node : j) roots.push_back(ReadNode(node, false));
    } else if (j.is_object()) {
        plan.timed = true;
        Node root = ReadNode(j, true);
        roots = std::move(root.children);
    } else {
        plan.error = "the plan has an unexpected shape";
        return plan;
    }
    for (const auto& root : roots) Flatten(root, 0, plan);
    for (size_t i = 0; i < plan.ops.size(); ++i) {
        const auto& op = plan.ops[i];
        if (op.internal) ++plan.internal_count;
        if (!plan.timed) continue;
        plan.operator_seconds += std::max(0.0, op.seconds);
        if (plan.slowest < 0 || op.seconds > plan.ops[static_cast<size_t>(plan.slowest)].seconds) {
            plan.slowest = static_cast<int>(i);
        }
    }
    if (plan.timed && !plan.ops.empty()) plan.rows_out = plan.ops.back().rows;
    return plan;
}

void ExplainPlan(Plan& plan, std::string_view sql, const sqlstep::SqlOutline& outline) {
    const size_t parts = outline.parts.size();
    std::vector<std::string> part_text(parts);
    std::vector<std::string> part_lower(parts);
    std::map<std::string, std::set<size_t>> literal_parts;
    for (size_t p = 0; p < parts; ++p) {
        const auto& part = outline.parts[p];
        const size_t end = std::min(part.end_offset, sql.size());
        part_text[p] = std::string(sql.substr(std::min(part.begin_offset, end), end - std::min(part.begin_offset, end)));
        part_lower[p] = Lower(part_text[p]);
        for (const auto& literal : Literals(part_text[p])) {
            if (literal.size() >= 4) literal_parts[literal].insert(p);
        }
    }
    const auto find_from = [&](size_t from, const auto& test) -> int {
        for (size_t p = from; p < parts; ++p) {
            if (test(p)) return static_cast<int>(p);
        }
        return -1;
    };

    size_t cursor = 0;
    for (auto& op : plan.ops) {
        const auto& made = op.made_columns;
        if (op.internal || parts == 0) continue;
        std::string all;
        for (const auto& [key, value] : op.details) all += value + "\n";
        // 1. Literals that belong to one part only.
        std::set<size_t> matched;
        for (const auto& [literal, owners] : literal_parts) {
            if (owners.size() == 1 && all.find("'" + literal + "'") != std::string::npos) matched.insert(*owners.begin());
        }
        int found = -1;
        if (!matched.empty()) {
            for (const size_t p : matched) op.parts.push_back(outline.parts[p].name);
            cursor = std::max(cursor, *matched.begin());
            continue;
        }
        // 2. By what the operator is.
        if (op.name == "COLUMN_DATA_SCAN") {
            found = find_from(0, [&](size_t p) { return part_lower[p].find("values") != std::string::npos; });
        } else if (op.name.find("SCAN") != std::string::npos) {
            const std::string table = Lower(DetailValue(op, "Table"));
            found = find_from(0, [&](size_t p) {
                const auto& reads = outline.parts[p].reads;
                return !table.empty() && std::find(reads.begin(), reads.end(), table) != reads.end();
            });
            if (found < 0) found = find_from(0, [&](size_t p) { return !outline.parts[p].reads.empty(); });
        } else if (op.name.find("JOIN") != std::string::npos) {
            found = find_from(cursor, [&](size_t p) { return part_lower[p].find(" join ") != std::string::npos; });
        } else if (op.name == "UNNEST") {
            found = find_from(cursor, [&](size_t p) { return part_lower[p].find("unnest(") != std::string::npos; });
        } else if (op.name == "ORDER_BY") {
            found = find_from(cursor, [&](size_t p) { return part_lower[p].find("order by") != std::string::npos; });
        } else if (op.name == "PROJECTION") {
            if (!made.empty()) {
                found = find_from(cursor, [&](size_t p) {
                    return std::any_of(made.begin(), made.end(),
                                       [&](const std::string& column) { return DefinesColumn(part_lower[p], column); });
                });
            } else if (cursor + 1 < parts) {
                // Passes the same columns on: DuckDB's next part.
                found = static_cast<int>(cursor + 1);
            }
        }
        if (found >= 0) {
            op.parts.push_back(outline.parts[static_cast<size_t>(found)].name);
            cursor = std::max(cursor, static_cast<size_t>(found));
        }
    }

    // Notes.
    plan.estimates_off = 0;
    for (auto& op : plan.ops) {
        if (op.internal) continue;
        std::vector<std::string> notes;
        if (plan.timed && op.rows_in > op.rows && op.rows >= 0 && op.name == "FILTER") {
            notes.push_back("Removes " + Thousands(op.rows_in - op.rows) + " rows (" + Thousands(op.rows_in) + " -> " +
                            Thousands(op.rows) + ").");
        }
        // The same call repeated inside one operator: DuckDB inlined an
        // earlier part's expression, so it is computed again per row.
        std::string all;
        for (const auto& [key, value] : op.details) all += value + "\n";
        std::string repeated;
        size_t times = 0;
        for (size_t at = 0; at < all.size(); ++at) {
            if (!std::isalpha(static_cast<unsigned char>(all[at])) || (at > 0 && (std::isalnum(static_cast<unsigned char>(all[at - 1])) || all[at - 1] == '_'))) continue;
            const std::string call = CallAt(all, at);
            if (call.size() < 20) continue;
            const size_t n = Count(all, call);
            if (n >= 3 && (n > times || (n == times && call.size() > repeated.size()))) {
                repeated = call;
                times = n;
            }
        }
        if (times >= 3) {
            notes.push_back("The same " + Shorten(repeated, 90) + " runs " + std::to_string(times) +
                            " times per row: DuckDB copied an earlier part's expression into this operator. Keeping "
                            "that part's result as its own column may compute it once.");
        }
        if (plan.timed && op.estimated >= 0 && op.rows >= 0) {
            const double actual = static_cast<double>(std::max<int64_t>(op.rows, 1));
            const double guess = static_cast<double>(std::max<int64_t>(op.estimated, 1));
            if (actual / guess >= 100.0 || guess / actual >= 100.0) {
                ++plan.estimates_off;
                notes.push_back("DuckDB estimated " + Thousands(op.estimated) + " rows; " + Thousands(op.rows) +
                                " came. With estimates this far off the plan's shape may not be the best one.");
            }
        }
        if (plan.timed && op.seconds >= 0.5 && op.rows > 0 && op.rows <= 1000) {
            char per_row[64];
            std::snprintf(per_row, sizeof(per_row), "%.3f", op.seconds / static_cast<double>(op.rows));
            notes.push_back(FormatSeconds(op.seconds) + " for " + Thousands(op.rows) + " rows, about " + per_row +
                            " s per row: each row carries a lot of data (long text), so every expression on it is "
                            "costly.");
        }
        op.notes = std::move(notes);
    }
}

std::string FormatSeconds(double seconds) {
    char text[32];
    if (seconds >= 0.1) std::snprintf(text, sizeof(text), "%.2f s", seconds);
    else if (seconds >= 0.0001) std::snprintf(text, sizeof(text), "%.1f ms", seconds * 1000.0);
    else if (seconds > 0.0) return "<0.1 ms";
    else return "0 ms";
    return text;
}

}  // namespace cyxwiz::sqlplan
