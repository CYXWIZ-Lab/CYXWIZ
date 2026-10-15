#include "guided_sql.h"

#include <algorithm>
#include <cctype>
#include <set>

namespace cyxwiz::guided {

namespace {
using json = nlohmann::json;

std::string Ident(const std::string& name) {
    std::string out = "\"";
    for (const char c : name) out += c == '"' ? std::string("\"\"") : std::string(1, c);
    return out + "\"";
}

std::string Lit(const std::string& text) {
    std::string out = "'";
    for (const char c : text) out += c == '\'' ? std::string("''") : std::string(1, c);
    return out + "'";
}

// A literal text as a regular expression (RE2): metacharacters escaped.
std::string RegexEscape(const std::string& text) {
    std::string out;
    for (const char c : text) {
        if (std::string_view("\\^$.|?*+()[]{}").find(c) != std::string_view::npos) out += '\\';
        out += c;
    }
    return out;
}

std::string Str(const json& params, const char* key, const std::string& fallback = {}) {
    const auto it = params.find(key);
    return it != params.end() && it->is_string() ? it->get<std::string>() : fallback;
}

bool Flag(const json& params, const char* key, bool fallback = false) {
    const auto it = params.find(key);
    return it != params.end() && it->is_boolean() ? it->get<bool>() : fallback;
}

long long Int(const json& params, const char* key, long long fallback = 0) {
    const auto it = params.find(key);
    if (it == params.end()) return fallback;
    if (it->is_number_integer()) return it->get<long long>();
    if (it->is_string()) {
        try {
            return std::stoll(it->get<std::string>());
        } catch (...) {
            return fallback;
        }
    }
    return fallback;
}

std::vector<std::string> StrList(const json& params, const char* key) {
    std::vector<std::string> out;
    const auto it = params.find(key);
    if (it == params.end() || !it->is_array()) return out;
    for (const auto& item : *it) {
        if (item.is_string() && !item.get<std::string>().empty()) out.push_back(item.get<std::string>());
    }
    return out;
}

const json& List(const json& params, const char* key) {
    static const json empty = json::array();
    const auto it = params.find(key);
    return it != params.end() && it->is_array() ? *it : empty;
}

std::string Join(const std::vector<std::string>& items, const std::string& separator) {
    std::string out;
    for (size_t i = 0; i < items.size(); ++i) out += (i ? separator : "") + items[i];
    return out;
}

std::string IdentList(const std::vector<std::string>& names) {
    std::vector<std::string> quoted;
    for (const auto& name : names) quoted.push_back(Ident(name));
    return Join(quoted, ", ");
}

bool IsNumber(const std::string& text) {
    if (text.empty()) return false;
    size_t i = text[0] == '-' ? 1 : 0;
    bool digit = false, dot = false;
    for (; i < text.size(); ++i) {
        if (std::isdigit(static_cast<unsigned char>(text[i]))) digit = true;
        else if (text[i] == '.' && !dot) dot = true;
        else return false;
    }
    return digit;
}

std::vector<std::string> SplitCommaList(const std::string& text) {
    std::vector<std::string> out;
    std::string item;
    for (const char c : text) {
        if (c == ',') {
            out.push_back(item);
            item.clear();
        } else {
            item += c;
        }
    }
    out.push_back(item);
    for (auto& v : out) {
        while (!v.empty() && v.front() == ' ') v.erase(v.begin());
        while (!v.empty() && v.back() == ' ') v.pop_back();
    }
    out.erase(std::remove(out.begin(), out.end(), std::string()), out.end());
    return out;
}

// One condition: column, operator, value.
std::string ConditionSql(const json& condition, std::string* error) {
    const std::string column = Str(condition, "column");
    const std::string op = Str(condition, "op", "equals");
    const std::string value = Str(condition, "value");
    if (column.empty()) {
        *error = "choose a column for every condition";
        return {};
    }
    const std::string c = Ident(column);
    const std::string text = "CAST(" + c + " AS VARCHAR)";
    const auto needs_value = [&]() {
        if (value.empty()) *error = "give a value for '" + column + " " + op + "'";
        return !value.empty();
    };
    if (op == "equals") return needs_value() ? c + " = " + Lit(value) : std::string();
    if (op == "not equals") return needs_value() ? c + " <> " + Lit(value) : std::string();
    if (op == "greater than") return needs_value() ? c + " > " + Lit(value) : std::string();
    if (op == "at least") return needs_value() ? c + " >= " + Lit(value) : std::string();
    if (op == "less than") return needs_value() ? c + " < " + Lit(value) : std::string();
    if (op == "at most") return needs_value() ? c + " <= " + Lit(value) : std::string();
    if (op == "contains") return needs_value() ? "contains(" + text + ", " + Lit(value) + ")" : std::string();
    if (op == "does not contain") return needs_value() ? "NOT contains(" + text + ", " + Lit(value) + ")" : std::string();
    if (op == "starts with") return needs_value() ? "starts_with(" + text + ", " + Lit(value) + ")" : std::string();
    if (op == "ends with") return needs_value() ? "ends_with(" + text + ", " + Lit(value) + ")" : std::string();
    if (op == "matches pattern") return needs_value() ? "regexp_matches(" + text + ", " + Lit(value) + ")" : std::string();
    if (op == "does not match pattern") {
        return needs_value() ? "NOT regexp_matches(" + text + ", " + Lit(value) + ")" : std::string();
    }
    if (op == "is empty") return "(" + c + " IS NULL OR " + text + " = '')";
    if (op == "is not empty") return "(" + c + " IS NOT NULL AND " + text + " <> '')";
    if (op == "is one of" || op == "is not one of") {
        const auto values = SplitCommaList(value);
        if (values.empty()) {
            *error = "list the values for '" + column + " " + op + "', separated by commas";
            return {};
        }
        std::vector<std::string> literals;
        for (const auto& v : values) literals.push_back(Lit(v));
        return text + (op == "is one of" ? " IN (" : " NOT IN (") + Join(literals, ", ") + ")";
    }
    *error = "unknown condition '" + op + "'";
    return {};
}

std::string ConditionText(const json& condition) {
    const std::string op = Str(condition, "op", "equals");
    const std::string value = Str(condition, "value");
    std::string text = Str(condition, "column", "?") + " " + op;
    if (op != "is empty" && op != "is not empty") text += " " + value;
    return text;
}

// The column expression of `add_column`, or empty with an error.
std::string FormulaSql(const json& p, std::string* error) {
    const std::string formula = Str(p, "formula", "length");
    const std::string column = Str(p, "column");
    const std::string column2 = Str(p, "column2");
    const std::string value = Str(p, "value");
    const auto col = [&]() -> std::string {
        if (column.empty()) *error = "choose a column";
        return Ident(column);
    };
    if (formula == "length") return "length(CAST(" + col() + " AS VARCHAR))";
    if (formula == "word count") {
        const std::string c = "trim(CAST(" + col() + " AS VARCHAR))";
        return "CASE WHEN " + c + " = '' THEN 0 ELSE len(regexp_split_to_array(" + c + ", '\\s+')) END";
    }
    if (formula == "upper case") return "upper(CAST(" + col() + " AS VARCHAR))";
    if (formula == "lower case") return "lower(CAST(" + col() + " AS VARCHAR))";
    if (formula == "combine") {
        if (column2.empty()) *error = "choose the second column";
        return "concat_ws(" + Lit(Str(p, "separator", " ")) + ", CAST(" + col() + " AS VARCHAR), CAST(" +
               Ident(column2) + " AS VARCHAR))";
    }
    if (formula == "arithmetic") {
        const std::string op = Str(p, "operator", "+");
        if (op != "+" && op != "-" && op != "*" && op != "/") *error = "choose + - * or /";
        std::string right;
        if (!column2.empty()) right = Ident(column2);
        else if (IsNumber(value)) right = value;
        else *error = "give a number or a second column";
        return "(" + col() + " " + op + " " + right + ")";
    }
    if (formula == "constant") return Lit(value);
    if (formula == "condition") {
        std::string condition_error;
        const std::string condition = ConditionSql(p.value("condition", json::object()), &condition_error);
        if (!condition_error.empty()) *error = condition_error;
        return "CASE WHEN " + condition + " THEN " + Lit(Str(p, "then")) + " ELSE " + Lit(Str(p, "else")) + " END";
    }
    *error = "unknown formula '" + formula + "'";
    return {};
}

// sum() of whole numbers as BIGINT, of anything else as DOUBLE.
std::string SumSql(const std::string& column, const Context& context) {
    const bool whole = context.whole_number_columns.count(column) > 0;
    return std::string("CAST(sum(") + Ident(column) + ") AS " + (whole ? "BIGINT" : "DOUBLE") + ")";
}

bool Contains(const std::vector<std::string>& names, const std::string& name) {
    return std::find(names.begin(), names.end(), name) != names.end();
}

// "SELECT *, expr AS name" or, when the column exists, "SELECT * REPLACE (...)".
std::string WithColumn(const std::string& from, const Columns& in, const std::string& name, const std::string& expr) {
    if (in.known && Contains(in.names, name)) {
        return "SELECT * REPLACE (" + expr + " AS " + Ident(name) + ") FROM " + from;
    }
    return "SELECT *, " + expr + " AS " + Ident(name) + " FROM " + from;
}

void AddColumn(Columns& columns, const std::string& name) {
    if (!Contains(columns.names, name)) columns.names.push_back(name);
}

std::string SplitExpr(const json& p, const std::string& column) {
    const std::string by = Str(p, "by", "paragraphs");
    const std::string c = "CAST(" + Ident(column) + " AS VARCHAR)";
    if (by == "lines") return "regexp_split_to_array(" + c + ", '\\r?\\n')";
    if (by == "sentences") return "regexp_extract_all(" + c + ", '[^.!?]+[.!?]*')";
    if (by == "separator") return "string_split(" + c + ", " + Lit(Str(p, "separator", ",")) + ")";
    // Two or more line breaks, Windows (\r\n) or Unix (\n), spaces between.
    return "regexp_split_to_array(" + c + ", '(\\r?\\n[ \\t]*){2,}')";
}

const char* MeasureFunction(const std::string& fn) {
    if (fn == "sum") return "sum";
    if (fn == "average") return "avg";
    if (fn == "min") return "min";
    if (fn == "max") return "max";
    if (fn == "first") return "first";
    return nullptr;
}

std::string MeasureName(const json& m) {
    const std::string name = Str(m, "name");
    if (!name.empty()) return name;
    const std::string fn = Str(m, "function", "count");
    const std::string column = Str(m, "column");
    if (fn == "count") return "row_count";
    std::string out = fn + "_" + column;
    for (char& ch : out) {
        if (ch == ' ') ch = '_';
    }
    return out;
}
}  // namespace

const std::vector<OpInfo>& Library() {
    static const std::vector<OpInfo> ops = {
        {"keep_rows", "Keep rows where", "Rows", "conditions with all / any groups"},
        {"remove_duplicates", "Remove duplicate rows", "Rows", "by all or chosen columns"},
        {"sort_rows", "Sort rows", "Rows", "one or more columns"},
        {"keep_first", "Keep first / sample rows", "Rows", "top N, random N, every Nth"},
        {"choose_columns", "Choose and rename columns", "Columns", "keep, drop, reorder, rename"},
        {"add_column", "Add a calculated column", "Columns", "formula builder, no SQL"},
        {"change_type", "Change type", "Columns", "text, whole number, decimal, true/false"},
        {"split_merge", "Split or merge columns", "Columns", "by a separator"},
        {"clean_text", "Clean text", "Text", "spaces, markup, footnote marks, case"},
        {"find_replace", "Find and replace", "Text", "plain text or a pattern"},
        {"keep_between", "Keep text between markers", "Text", "cut headers and footers"},
        {"split_into_rows", "Split text into rows", "Text", "paragraphs, lines, sentences"},
        {"extract_pattern", "Extract with a pattern", "Text", "pull a reference, number or code"},
        {"look_up", "Look up from another table", "Combine", "join with a match report"},
        {"append_rows", "Append rows", "Combine", "stack inputs with the same columns"},
        {"group_summarise", "Group and summarise", "Summarise", "count, sum, average, first, join text"},
        {"pivot_unpivot", "Pivot / unpivot", "Summarise", "rows to columns and back"},
        {"assign_split", "Assign train / validation / test", "Splits", "stable hash, by a key"},
        {"number_rows", "Number rows", "Splits", "within groups, in an order"},
        {"add_checks", "Add checks", "Checks", "row counts, allowed values, unique, not empty"},
    };
    return ops;
}

const OpInfo* FindOp(std::string_view id) {
    for (const auto& op : Library()) {
        if (id == op.id) return &op;
    }
    return nullptr;
}

json DefaultParams(std::string_view type, const std::string& column) {
    if (type == "keep_rows") {
        return {{"match", "all"}, {"conditions", json::array({{{"column", column}, {"op", "is not empty"}, {"value", ""}}})}};
    }
    if (type == "remove_duplicates") return {{"columns", json::array()}};
    if (type == "sort_rows") return {{"keys", json::array({{{"column", column}, {"descending", false}}})}};
    if (type == "keep_first") return {{"mode", "first"}, {"count", "1000"}};
    if (type == "choose_columns") return {{"columns", json::array()}};
    if (type == "add_column") return {{"name", "new_column"}, {"formula", "length"}, {"column", column}};
    if (type == "change_type") return {{"column", column}, {"to", "whole number"}, {"on_error", "empty"}};
    if (type == "split_merge") return {{"mode", "split"}, {"column", column}, {"separator", ","}, {"names", json::array({"part_1", "part_2"})}};
    if (type == "clean_text") {
        return {{"column", column}, {"trim", true}, {"collapse_spaces", true}, {"remove_markup", false},
                {"remove_footnote_marks", false}, {"case", "unchanged"}};
    }
    if (type == "find_replace") return {{"column", column}, {"find", ""}, {"replace", ""}, {"pattern", false}, {"match_case", true}};
    if (type == "keep_between") return {{"column", column}, {"start", ""}, {"end", ""}, {"keep_markers", false}, {"if_missing", "keep"}};
    if (type == "split_into_rows") return {{"column", column}, {"by", "paragraphs"}, {"index_name", column.empty() ? "piece_index" : column + "_index"}, {"drop_empty", true}};
    if (type == "extract_pattern") return {{"column", column}, {"pattern", ""}, {"group", "0"}, {"name", "extracted"}};
    if (type == "look_up") {
        return {{"table", ""}, {"key", json::array({{{"kind", "column"}, {"value", column}}})}, {"other_key", ""},
                {"bring", json::array()}, {"no_match", "keep"}};
    }
    if (type == "append_rows") return {{"table", ""}};
    if (type == "group_summarise") {
        return {{"by", json::array({column})}, {"measures", json::array({{{"function", "count"}, {"column", ""}, {"name", "row_count"}}})}};
    }
    if (type == "pivot_unpivot") {
        return {{"mode", "pivot"}, {"names_from", ""}, {"values", ""}, {"values_from", ""}, {"function", "sum"}, {"group_by", json::array()},
                {"columns", json::array()}, {"name_column", "name"}, {"value_column", "value"}};
    }
    if (type == "assign_split") {
        return {{"key", json::array({column})}, {"train", "90"}, {"validation", "5"}, {"test", "5"}, {"salt", "split-v1"}, {"name", "split"}};
    }
    if (type == "number_rows") return {{"name", "row_index"}, {"partition_by", json::array()}, {"order_by", json::array()}};
    if (type == "add_checks") return {{"checks", json::array({{{"kind", "row_count"}, {"min", "1"}, {"max", ""}, {"exact", ""}}})}};
    return json::object();
}

std::string ToJson(const GuidedStep& step) {
    json ops = json::array();
    for (const auto& op : step.ops) ops.push_back({{"type", op.type}, {"on", op.on}, {"params", op.params}});
    json out{{"version", 1}, {"ops", ops}};
    const Context& c = step.saved_context;
    if (!c.inputs.empty()) {
        out["inputs"] = c.inputs;
        out["columns"] = c.columns;
        out["whole_number_columns"] = c.whole_number_columns;
    }
    return out.dump();
}

std::optional<GuidedStep> FromJson(std::string_view text, std::string* error) {
    const json j = json::parse(text.begin(), text.end(), nullptr, false);
    if (j.is_discarded() || !j.is_object() || !j.contains("ops") || !j["ops"].is_array()) {
        if (error) *error = "the guided operations are not valid JSON";
        return std::nullopt;
    }
    if (j.value("version", 0) != 1) {
        if (error) *error = "guided operations version " + std::to_string(j.value("version", 0)) + " is not supported";
        return std::nullopt;
    }
    GuidedStep step;
    for (const auto& item : j["ops"]) {
        Operation op;
        op.type = item.value("type", std::string());
        if (!FindOp(op.type)) {
            if (error) *error = "unknown guided operation '" + op.type + "'";
            return std::nullopt;
        }
        op.on = item.value("on", true);
        op.params = item.contains("params") && item["params"].is_object() ? item["params"] : json::object();
        step.ops.push_back(std::move(op));
    }
    if (j.contains("inputs") && j["inputs"].is_array()) {
        try {
            step.saved_context.inputs = j["inputs"].get<std::vector<std::string>>();
            if (j.contains("columns")) step.saved_context.columns = j["columns"].get<std::map<std::string, std::vector<std::string>>>();
            if (j.contains("whole_number_columns")) {
                step.saved_context.whole_number_columns = j["whole_number_columns"].get<std::set<std::string>>();
            }
        } catch (const std::exception&) {
            step.saved_context = Context{};
        }
    }
    return step;
}

std::vector<Columns> TrackColumns(const GuidedStep& step, const Context& context) {
    std::vector<Columns> out;
    Columns current;
    const std::string main = context.inputs.empty() ? std::string("input") : context.inputs.front();
    const auto found = context.columns.find(main);
    current.known = found != context.columns.end();
    if (current.known) current.names = found->second;
    for (const auto& op : step.ops) {
        out.push_back(current);
        if (!op.on) continue;
        const json& p = op.params;
        const std::string& t = op.type;
        if (t == "choose_columns") {
            Columns next;
            for (const auto& c : List(p, "columns")) {
                if (!c.value("keep", true)) continue;
                const std::string rename = Str(c, "rename");
                next.names.push_back(rename.empty() ? Str(c, "name") : rename);
            }
            current = next;
        } else if (t == "add_column" || t == "extract_pattern" || t == "number_rows" || t == "assign_split") {
            AddColumn(current, Str(p, "name"));
        } else if (t == "split_merge") {
            if (Str(p, "mode", "split") == "split") {
                for (const auto& name : StrList(p, "names")) AddColumn(current, name);
            } else {
                AddColumn(current, Str(p, "name"));
            }
        } else if (t == "split_into_rows") {
            const std::string column = Str(p, "column");
            auto& names = current.names;
            names.erase(std::remove(names.begin(), names.end(), column), names.end());
            names.push_back(column);
            AddColumn(current, Str(p, "index_name"));
        } else if (t == "look_up") {
            for (const auto& b : List(p, "bring")) {
                const std::string as = Str(b, "as");
                AddColumn(current, as.empty() ? Str(b, "column") : as);
            }
        } else if (t == "append_rows") {
            const auto other = context.columns.find(Str(p, "table"));
            if (other == context.columns.end()) current.known = false;
            else for (const auto& name : other->second) AddColumn(current, name);
        } else if (t == "group_summarise") {
            Columns next;
            next.names = StrList(p, "by");
            for (const auto& m : List(p, "measures")) next.names.push_back(MeasureName(m));
            current = next;
        } else if (t == "pivot_unpivot") {
            if (Str(p, "mode", "pivot") == "pivot") {
                Columns next;
                next.names = StrList(p, "group_by");
                for (const auto& v : SplitCommaList(Str(p, "values"))) next.names.push_back(v);
                current = next;
            } else {
                const auto unpivoted = StrList(p, "columns");
                auto& names = current.names;
                for (const auto& c : unpivoted) names.erase(std::remove(names.begin(), names.end(), c), names.end());
                AddColumn(current, Str(p, "name_column", "name"));
                AddColumn(current, Str(p, "value_column", "value"));
            }
        }
    }
    out.push_back(current);
    return out;
}

bool MoveOp(GuidedStep& step, size_t from, size_t to) {
    if (from >= step.ops.size() || to >= step.ops.size()) return false;
    if (from == to) return true;
    Operation moved = std::move(step.ops[from]);
    step.ops.erase(step.ops.begin() + static_cast<std::ptrdiff_t>(from));
    step.ops.insert(step.ops.begin() + static_cast<std::ptrdiff_t>(to), std::move(moved));
    return true;
}

std::string Summarise(const Operation& op) {
    const json& p = op.params;
    const std::string& t = op.type;
    const std::string column = Str(p, "column", "?");
    if (t == "keep_rows") {
        std::vector<std::string> parts;
        for (const auto& c : List(p, "conditions")) parts.push_back(ConditionText(c));
        if (parts.empty()) return "Keep rows where ... (add a condition)";
        return "Keep rows where " + Join(parts, Str(p, "match", "all") == "any" ? " or " : " and ") + ".";
    }
    if (t == "remove_duplicates") {
        const auto columns = StrList(p, "columns");
        return columns.empty() ? "Remove rows that repeat an earlier row." : "Keep the first row for each " + Join(columns, ", ") + ".";
    }
    if (t == "sort_rows") {
        std::vector<std::string> keys;
        for (const auto& k : List(p, "keys")) keys.push_back(Str(k, "column", "?") + (Flag(k, "descending") ? " (high to low)" : ""));
        return "Sort by " + Join(keys, ", ") + ".";
    }
    if (t == "keep_first") {
        const std::string mode = Str(p, "mode", "first");
        const std::string n = Str(p, "count", "?");
        if (mode == "sample") return "Keep a random sample of " + n + " rows (repeatable).";
        if (mode == "every") return "Keep every " + n + "th row.";
        return "Keep the first " + n + " rows.";
    }
    if (t == "choose_columns") {
        int kept = 0, renamed = 0;
        for (const auto& c : List(p, "columns")) {
            if (!c.value("keep", true)) continue;
            ++kept;
            if (!Str(c, "rename").empty()) ++renamed;
        }
        return "Keep " + std::to_string(kept) + " of " + std::to_string(List(p, "columns").size()) + " columns" +
               (renamed ? "; rename " + std::to_string(renamed) : "") + ".";
    }
    if (t == "add_column") return "Add " + Str(p, "name", "?") + " = " + Str(p, "formula", "length") + " of " + column + ".";
    if (t == "change_type") return "Make " + column + " " + Str(p, "to", "?") + ".";
    if (t == "split_merge") {
        return Str(p, "mode", "split") == "split"
                   ? "Split " + column + " at '" + Str(p, "separator") + "' into " + Join(StrList(p, "names"), ", ") + "."
                   : "Merge " + Join(StrList(p, "columns"), ", ") + " into " + Str(p, "name", "?") + ".";
    }
    if (t == "clean_text") {
        std::vector<std::string> what;
        if (Flag(p, "remove_markup")) what.push_back("markup");
        if (Flag(p, "remove_footnote_marks")) what.push_back("footnote marks");
        if (Flag(p, "collapse_spaces")) what.push_back("extra spaces");
        const std::string c = Str(p, "case", "unchanged");
        return "Clean " + column + (what.empty() ? "" : ": remove " + Join(what, ", ")) +
               (Flag(p, "trim") ? ", trim" : "") + (c != "unchanged" ? ", " + c + " case" : "") + ".";
    }
    if (t == "find_replace") {
        return "In " + column + ", replace " + (Flag(p, "pattern") ? "the pattern " : "") + "'" + Str(p, "find") + "' with '" +
               Str(p, "replace") + "'.";
    }
    if (t == "keep_between") {
        return "Keep the part of " + column + " between '" + Str(p, "start", "(start)") + "' and '" + Str(p, "end", "(end)") + "'.";
    }
    if (t == "split_into_rows") {
        const std::string by = Str(p, "by", "paragraphs");
        const std::string piece = by == "separator" ? "piece between '" + Str(p, "separator") + "'"
                                                    : by.substr(0, by.size() - 1);  // paragraphs -> paragraph
        return "Split " + column + " into rows: one per " + piece + ", numbered in " + Str(p, "index_name", "?") + ".";
    }
    if (t == "extract_pattern") return "Extract " + Str(p, "name", "?") + " from " + column + " with a pattern.";
    if (t == "look_up") {
        std::vector<std::string> brought;
        for (const auto& b : List(p, "bring")) brought.push_back(Str(b, "as").empty() ? Str(b, "column") : Str(b, "as"));
        const std::string no_match = Str(p, "no_match", "keep");
        return "Add " + (brought.empty() ? std::string("columns") : Join(brought, ", ")) + " from " + Str(p, "table", "?") +
               ", matching on " + Str(p, "other_key", "?") + (no_match == "remove" ? "; drop rows with no match"
                                                              : no_match == "stop" ? "; stop when a row has no match" : "") + ".";
    }
    if (t == "append_rows") return "Append the rows of " + Str(p, "table", "?") + " (columns matched by name).";
    if (t == "group_summarise") {
        std::vector<std::string> measures;
        for (const auto& m : List(p, "measures")) measures.push_back(MeasureName(m));
        return "One row per " + Join(StrList(p, "by"), ", ") + " with " + Join(measures, ", ") + ".";
    }
    if (t == "pivot_unpivot") {
        return Str(p, "mode", "pivot") == "pivot"
                   ? "Pivot: a column per " + Str(p, "names_from", "?") + " value holding " + Str(p, "function", "sum") + " of " + Str(p, "values_from", "?") + "."
                   : "Unpivot " + Join(StrList(p, "columns"), ", ") + " into " + Str(p, "name_column", "name") + " and " + Str(p, "value_column", "value") + ".";
    }
    if (t == "assign_split") {
        return "Assign " + Str(p, "name", "split") + " " + Str(p, "train", "?") + "/" + Str(p, "validation", "?") + "/" +
               Str(p, "test", "?") + " by a stable hash of " + Join(StrList(p, "key"), ", ") + ".";
    }
    if (t == "number_rows") {
        const auto groups = StrList(p, "partition_by");
        return "Number rows in " + Str(p, "name", "?") + (groups.empty() ? "" : " within each " + Join(groups, ", ")) + ".";
    }
    if (t == "add_checks") return std::to_string(List(p, "checks").size()) + " check(s); they become Check nodes when the step is saved.";
    return t;
}

Generated GenerateSql(const GuidedStep& step, const Context& context) {
    Generated out;
    out.part_names.assign(step.ops.size(), std::string());
    out.errors.assign(step.ops.size(), std::string());
    const auto columns = TrackColumns(step, context);
    const std::string main = context.inputs.empty() ? std::string("input") : context.inputs.front();
    std::string previous = Ident(main);
    std::vector<std::string> ctes;
    // The last operation that is on: "Add checks" must be it (its checks
    // become Check nodes after the step, which see the step's result).
    size_t last_on = step.ops.size();
    for (size_t i = 0; i < step.ops.size(); ++i)
        if (step.ops[i].on) last_on = i;
    for (size_t i = 0; i < step.ops.size(); ++i) {
        const Operation& op = step.ops[i];
        if (!op.on) continue;
        const json& p = op.params;
        const std::string& t = op.type;
        const Columns& in = columns[i];
        std::string error;
        std::string body;
        if (t == "add_checks" && i != last_on) {
            out.errors[i] = "move Add checks to the end: its checks become check nodes after the step";
            out.ok = false;
            continue;
        }
        const std::string column = Str(p, "column");
        const auto need_column = [&]() {
            if (column.empty()) error = "choose a column";
            return Ident(column);
        };
        if (t == "keep_rows") {
            std::vector<std::string> conditions;
            for (const auto& c : List(p, "conditions")) {
                std::string condition = ConditionSql(c, &error);
                if (!error.empty()) break;
                conditions.push_back("(" + condition + ")");
            }
            if (conditions.empty() && error.empty()) error = "add a condition";
            body = "SELECT * FROM " + previous + "\n  WHERE " + Join(conditions, Str(p, "match", "all") == "any" ? "\n     OR " : "\n    AND ");
        } else if (t == "remove_duplicates") {
            const auto keys = StrList(p, "columns");
            if (keys.empty()) {
                body = "SELECT DISTINCT * FROM " + previous;
            } else {
                body = "SELECT * EXCLUDE (__row) FROM (SELECT *, row_number() OVER () AS __row FROM " + previous +
                       ")\n  QUALIFY row_number() OVER (PARTITION BY " + IdentList(keys) + " ORDER BY __row) = 1\n  ORDER BY __row";
            }
        } else if (t == "sort_rows") {
            std::vector<std::string> keys;
            for (const auto& k : List(p, "keys")) {
                if (Str(k, "column").empty()) continue;
                keys.push_back(Ident(Str(k, "column")) + (Flag(k, "descending") ? " DESC" : ""));
            }
            if (keys.empty()) error = "choose a column to sort by";
            body = "SELECT * FROM " + previous + " ORDER BY " + Join(keys, ", ");
        } else if (t == "keep_first") {
            const long long n = Int(p, "count", 0);
            if (n <= 0) error = "give how many rows (a whole number above 0)";
            const std::string mode = Str(p, "mode", "first");
            if (mode == "sample") {
                body = "SELECT * FROM " + previous + " USING SAMPLE reservoir(" + std::to_string(n) + " ROWS) REPEATABLE (42)";
            } else if (mode == "every") {
                body = "SELECT * EXCLUDE (__row) FROM (SELECT *, row_number() OVER () AS __row FROM " + previous +
                       ")\n  WHERE (__row - 1) % " + std::to_string(n) + " = 0";
            } else {
                body = "SELECT * FROM " + previous + " LIMIT " + std::to_string(n);
            }
        } else if (t == "choose_columns") {
            std::vector<std::string> select;
            for (const auto& c : List(p, "columns")) {
                if (!c.value("keep", true)) continue;
                const std::string name = Str(c, "name");
                const std::string rename = Str(c, "rename");
                select.push_back(Ident(name) + (rename.empty() || rename == name ? "" : " AS " + Ident(rename)));
            }
            if (select.empty()) error = "keep at least one column";
            body = "SELECT " + Join(select, ", ") + "\n  FROM " + previous;
        } else if (t == "add_column") {
            const std::string name = Str(p, "name");
            if (name.empty()) error = "name the new column";
            const std::string expr = FormulaSql(p, &error);
            body = WithColumn(previous, in, name, expr);
        } else if (t == "change_type") {
            const std::string to = Str(p, "to", "whole number");
            const char* type = to == "text" ? "VARCHAR" : to == "decimal" ? "DOUBLE" : to == "true/false" ? "BOOLEAN" : "BIGINT";
            const std::string cast = Str(p, "on_error", "empty") == "stop" ? "CAST" : "TRY_CAST";
            const std::string c = need_column();
            body = "SELECT * REPLACE (" + cast + "(" + c + " AS " + type + ") AS " + c + ") FROM " + previous;
        } else if (t == "split_merge") {
            if (Str(p, "mode", "split") == "split") {
                const auto names = StrList(p, "names");
                if (names.empty()) error = "name the new columns";
                std::vector<std::string> select;
                for (size_t k = 0; k < names.size(); ++k) {
                    select.push_back("split_part(CAST(" + need_column() + " AS VARCHAR), " + Lit(Str(p, "separator", ",")) +
                                     ", " + std::to_string(k + 1) + ") AS " + Ident(names[k]));
                }
                body = "SELECT *, " + Join(select, ", ") + " FROM " + previous;
            } else {
                const auto merged = StrList(p, "columns");
                if (merged.size() < 2) error = "choose two or more columns to merge";
                if (Str(p, "name").empty()) error = "name the merged column";
                std::vector<std::string> parts;
                for (const auto& m : merged) parts.push_back("CAST(" + Ident(m) + " AS VARCHAR)");
                body = WithColumn(previous, in, Str(p, "name"),
                                  "concat_ws(" + Lit(Str(p, "separator", " ")) + ", " + Join(parts, ", ") + ")");
            }
        } else if (t == "clean_text") {
            std::string expr = "CAST(" + need_column() + " AS VARCHAR)";
            if (Flag(p, "remove_footnote_marks")) expr = "regexp_replace(" + expr + ", '\\s?\\[[0-9a-z]{1,3}\\]', '', 'g')";
            if (Flag(p, "remove_markup")) {
                expr = "regexp_replace(regexp_replace(" + expr + ", '_([^_\\n]+)_', '\\1', 'g'), '=([^=\\n]+)=', '\\1', 'g')";
            }
            if (Flag(p, "collapse_spaces")) expr = "regexp_replace(" + expr + ", '\\s+', ' ', 'g')";
            if (Flag(p, "trim")) expr = "trim(" + expr + ")";
            const std::string c = Str(p, "case", "unchanged");
            if (c == "lower") expr = "lower(" + expr + ")";
            if (c == "upper") expr = "upper(" + expr + ")";
            body = "SELECT * REPLACE (" + expr + " AS " + Ident(column) + ") FROM " + previous;
        } else if (t == "find_replace") {
            const std::string find = Str(p, "find");
            if (find.empty()) error = "give the text to find";
            const std::string c = "CAST(" + need_column() + " AS VARCHAR)";
            std::string expr;
            if (!Flag(p, "pattern") && Flag(p, "match_case", true)) {
                expr = "replace(" + c + ", " + Lit(find) + ", " + Lit(Str(p, "replace")) + ")";
            } else {
                const std::string pattern = Flag(p, "pattern") ? find : RegexEscape(find);
                expr = "regexp_replace(" + c + ", " + Lit(pattern) + ", " + Lit(Str(p, "replace")) + ", " +
                       (Flag(p, "match_case", true) ? "'g'" : "'gi'") + ")";
            }
            body = "SELECT * REPLACE (" + expr + " AS " + Ident(column) + ") FROM " + previous;
        } else if (t == "keep_between") {
            const std::string start = Str(p, "start");
            const std::string end = Str(p, "end");
            if (start.empty() && end.empty()) error = "give a start marker, an end marker or both";
            const std::string pattern = "(?s)" + (start.empty() ? std::string("^") : RegexEscape(start)) + "(.*?)" +
                                        (end.empty() ? std::string("$") : RegexEscape(end));
            const std::string c = "CAST(" + need_column() + " AS VARCHAR)";
            const std::string group = Flag(p, "keep_markers") ? "0" : "1";
            std::string expr = "regexp_extract(" + c + ", " + Lit(pattern) + ", " + group + ")";
            if (Str(p, "if_missing", "keep") == "keep") {
                expr = "CASE WHEN regexp_matches(" + c + ", " + Lit(pattern) + ") THEN " + expr + " ELSE " + c + " END";
            }
            body = "SELECT * REPLACE (" + expr + " AS " + Ident(column) + ") FROM " + previous;
        } else if (t == "split_into_rows") {
            const std::string c = need_column();
            const std::string index = Str(p, "index_name");
            if (index.empty()) error = "name the column that numbers the pieces";
            std::string pieces = SplitExpr(p, column);
            if (Flag(p, "drop_empty", true)) pieces = "list_filter(" + pieces + ", x -> trim(x) <> '')";
            // Unnest in the select list: no lateral join carrying the whole
            // text into every piece (TOFIX146 run-speed finding).
            body = "SELECT * EXCLUDE (__pieces, " + c + "), unnest(__pieces) AS " + c + ",\n         unnest(range(1, len(__pieces) + 1)) AS " +
                   Ident(index) + "\n  FROM (SELECT *, " + pieces + " AS __pieces FROM " + previous + ")";
        } else if (t == "extract_pattern") {
            const std::string pattern = Str(p, "pattern");
            if (pattern.empty()) error = "give a pattern";
            if (Str(p, "name").empty()) error = "name the new column";
            const long long group = Int(p, "group", 0);
            body = WithColumn(previous, in, Str(p, "name"),
                              "regexp_extract(CAST(" + need_column() + " AS VARCHAR), " + Lit(pattern) + ", " + std::to_string(group) + ")");
        } else if (t == "look_up") {
            const std::string table = Str(p, "table");
            const std::string other_key = Str(p, "other_key");
            if (table.empty() || std::find(context.inputs.begin(), context.inputs.end(), table) == context.inputs.end()) {
                error = "choose the table to look up from (one of the step's inputs)";
            }
            if (other_key.empty()) error = "choose the column to match in " + (table.empty() ? std::string("that table") : table);
            std::vector<std::string> key_parts;
            const auto& key = List(p, "key");
            for (const auto& k : key) {
                const std::string value = Str(k, "value");
                if (Str(k, "kind", "column") == "text") key_parts.push_back(Lit(value));
                else if (!value.empty()) key_parts.push_back(key.size() == 1 ? "l." + Ident(value) : "CAST(l." + Ident(value) + " AS VARCHAR)");
            }
            if (key_parts.empty()) error = "build the key to match from this table's columns";
            std::vector<std::string> brought;
            for (const auto& b : List(p, "bring")) {
                if (Str(b, "column").empty()) continue;
                const std::string as = Str(b, "as").empty() ? Str(b, "column") : Str(b, "as");
                brought.push_back("r." + Ident(Str(b, "column")) + " AS " + Ident(as));
            }
            if (brought.empty()) error = "choose the columns to bring in";
            const std::string key_sql = Join(key_parts, " || ");
            const std::string no_match = Str(p, "no_match", "keep");
            body = "SELECT l.*, " + Join(brought, ", ") + "\n  FROM " + previous + " l\n  " +
                   (no_match == "remove" ? "JOIN " : "LEFT JOIN ") + Ident(table) + " r ON r." + Ident(other_key) + " = " + key_sql;
            if (no_match == "stop") {
                // A separate count of unmatched rows: DuckDB may run later
                // filters first, so a check inside the join's WHERE could
                // never see a row a later operation drops.
                const std::string unmatched = "(SELECT CAST(" + key_sql + " AS VARCHAR) AS k FROM " + previous + " l LEFT JOIN " +
                                              Ident(table) + " r ON r." + Ident(other_key) + " = " + key_sql + " WHERE r." +
                                              Ident(other_key) + " IS NULL)";
                body = "SELECT j.* FROM (" + body + ") j,\n    (SELECT CASE WHEN (SELECT count(*) FROM " + unmatched +
                       ") > 0 THEN error('Look up: ' || (SELECT count(*) FROM " + unmatched + ") || ' row(s) have no match in " +
                       table + "; the first is ' || (SELECT first(k) FROM " + unmatched + ")) ELSE 1 END AS ok) chk\n  WHERE chk.ok = 1";
            }
        } else if (t == "append_rows") {
            const std::string table = Str(p, "table");
            if (table.empty() || std::find(context.inputs.begin(), context.inputs.end(), table) == context.inputs.end()) {
                error = "choose the table to append (one of the step's inputs)";
            }
            body = "SELECT * FROM " + previous + "\n  UNION ALL BY NAME\n  SELECT * FROM " + Ident(table);
        } else if (t == "group_summarise") {
            const auto by = StrList(p, "by");
            std::vector<std::string> select;
            for (const auto& name : by) select.push_back(Ident(name));
            for (const auto& m : List(p, "measures")) {
                const std::string fn = Str(m, "function", "count");
                const std::string col = Str(m, "column");
                std::string expr;
                if (fn == "count") expr = "count(*)";
                else if (col.empty()) error = "choose the column for " + fn;
                else if (fn == "count distinct") expr = "count(DISTINCT " + Ident(col) + ")";
                else if (fn == "join text") expr = "string_agg(CAST(" + Ident(col) + " AS VARCHAR), " + Lit(Str(m, "separator", " ")) + ")";
                else if (fn == "sum") expr = SumSql(col, context);
                else if (const char* f = MeasureFunction(fn)) expr = std::string(f) + "(" + Ident(col) + ")";
                else error = "unknown summary '" + fn + "'";
                select.push_back(expr + " AS " + Ident(MeasureName(m)));
            }
            if (List(p, "measures").empty()) error = "add a summary (count, sum, ...)";
            body = "SELECT " + Join(select, ", ") + "\n  FROM " + previous + (by.empty() ? "" : "\n  GROUP BY " + IdentList(by));
        } else if (t == "pivot_unpivot") {
            if (Str(p, "mode", "pivot") == "pivot") {
                const std::string names = Str(p, "names_from");
                const std::string values = Str(p, "values_from");
                const std::string fn = Str(p, "function", "sum");
                if (names.empty()) error = "choose the column whose values become columns";
                if (values.empty() && fn != "count") error = "choose the column to fill them from";
                const auto group = StrList(p, "group_by");
                const char* f = MeasureFunction(fn);
                const std::string using_sql = fn == "count" ? std::string("count(*)")
                                              : fn == "sum" || !f ? SumSql(values, context)
                                                                  : std::string(f) + "(" + Ident(values) + ")";
                // The values that become columns are listed: without IN (...)
                // DuckDB runs a second statement to find them, and a SQL step
                // is one statement.
                std::vector<std::string> in_list;
                for (const auto& v : SplitCommaList(Str(p, "values"))) in_list.push_back(Lit(v));
                if (in_list.empty()) error = "list the values that become columns (Find values fills them from the data)";
                // Only the columns involved: DuckDB groups a PIVOT by every
                // other column, so a row would follow columns the user did
                // not choose.
                std::vector<std::string> used = group;
                used.push_back(names);
                if (fn != "count") used.push_back(values);
                body = "SELECT * FROM (PIVOT (SELECT " + IdentList(used) + " FROM " + previous + ") ON " + Ident(names) +
                       " IN (" + Join(in_list, ", ") + ") USING " + using_sql +
                       (group.empty() ? "" : " GROUP BY " + IdentList(group)) + ")";
            } else {
                const auto unpivot = StrList(p, "columns");
                if (unpivot.empty()) error = "choose the columns to turn into rows";
                body = "SELECT * FROM (UNPIVOT " + previous + " ON " + IdentList(unpivot) + " INTO NAME " +
                       Ident(Str(p, "name_column", "name")) + " VALUE " + Ident(Str(p, "value_column", "value")) + ")";
            }
        } else if (t == "assign_split") {
            const auto key = StrList(p, "key");
            const long long train = Int(p, "train", -1), validation = Int(p, "validation", -1), test = Int(p, "test", -1);
            if (key.empty()) error = "choose the key the split follows (rows with the same key stay together)";
            if (train < 0 || validation < 0 || test < 0 || train + validation + test != 100) error = "the percentages must add up to 100";
            std::vector<std::string> parts;
            for (const auto& k : key) parts.push_back("CAST(" + Ident(k) + " AS VARCHAR)");
            const std::string bucket = "md5_number(concat_ws('|', " + Join(parts, ", ") + ") || " + Lit(Str(p, "salt")) + ") % 100";
            const std::string expr = "CASE WHEN " + bucket + " < " + std::to_string(train) + " THEN 'train' WHEN " + bucket + " < " +
                                     std::to_string(train + validation) + " THEN 'validation' ELSE 'test' END";
            body = WithColumn(previous, in, Str(p, "name", "split"), expr);
        } else if (t == "number_rows") {
            const auto groups = StrList(p, "partition_by");
            std::vector<std::string> order;
            for (const auto& k : List(p, "order_by")) {
                if (!Str(k, "column").empty()) order.push_back(Ident(Str(k, "column")) + (Flag(k, "descending") ? " DESC" : ""));
            }
            if (Str(p, "name").empty()) error = "name the number column";
            std::string window = groups.empty() ? std::string() : "PARTITION BY " + IdentList(groups);
            if (!order.empty()) window += std::string(window.empty() ? "" : " ") + "ORDER BY " + Join(order, ", ");
            body = WithColumn(previous, in, Str(p, "name"), "row_number() OVER (" + window + ")");
        } else if (t == "add_checks") {
            std::vector<std::string> whens;
            for (const auto& c : List(p, "checks")) {
                const std::string kind = Str(c, "kind", "row_count");
                const std::string col = Str(c, "column");
                if (kind == "row_count") {
                    const std::string n = "(SELECT count(*) FROM " + previous + ")";
                    if (!Str(c, "exact").empty()) whens.push_back(n + " <> " + std::to_string(Int(c, "exact")) + " THEN error('Check failed: expected exactly " + Str(c, "exact") + " rows, found ' || " + n + ")");
                    if (!Str(c, "min").empty()) whens.push_back(n + " < " + std::to_string(Int(c, "min")) + " THEN error('Check failed: expected at least " + Str(c, "min") + " rows, found ' || " + n + ")");
                    if (!Str(c, "max").empty()) whens.push_back(n + " > " + std::to_string(Int(c, "max")) + " THEN error('Check failed: expected at most " + Str(c, "max") + " rows, found ' || " + n + ")");
                } else if (col.empty() && kind != "unique") {
                    error = "choose the column for every check";
                } else if (kind == "allowed_values") {
                    std::vector<std::string> literals;
                    for (const auto& v : SplitCommaList(Str(c, "values"))) literals.push_back(Lit(v));
                    if (literals.empty()) error = "list the allowed values";
                    whens.push_back("(SELECT count(*) FROM " + previous + " WHERE " + Ident(col) + " IS NULL OR CAST(" + Ident(col) +
                                    " AS VARCHAR) NOT IN (" + Join(literals, ", ") + ")) > 0 THEN error('Check failed: " + col +
                                    " has values outside " + Str(c, "values") + "')");
                } else if (kind == "unique") {
                    auto keys = StrList(c, "columns");
                    if (keys.empty() && !col.empty()) keys.push_back(col);
                    if (keys.empty()) error = "choose the column(s) that must be unique";
                    whens.push_back("(SELECT count(*) FROM (SELECT " + IdentList(keys) + " FROM " + previous + " GROUP BY ALL HAVING count(*) > 1)) > 0 THEN error('Check failed: " +
                                    Join(keys, ", ") + " repeats')");
                } else if (kind == "not_empty") {
                    whens.push_back("(SELECT count(*) FROM " + previous + " WHERE " + Ident(col) + " IS NULL OR CAST(" + Ident(col) +
                                    " AS VARCHAR) = '') > 0 THEN error('Check failed: " + col + " is empty in some rows')");
                }
            }
            if (whens.empty() && error.empty()) error = "add a check";
            body = "SELECT p.* FROM " + previous + " p,\n    (SELECT CASE WHEN " + Join(whens, "\n                 WHEN ") +
                   "\n                 ELSE 1 END AS ok) chk\n  WHERE chk.ok = 1";
        } else {
            error = "unknown operation";
        }
        if (!error.empty()) {
            out.errors[i] = error;
            out.ok = false;
            continue;  // left out; the next operation reads the one before
        }
        const std::string name = t + "_" + std::to_string(i + 1);
        out.part_names[i] = name;
        // The summary is a -- comment: keep it on one line.
        std::string summary = Summarise(op);
        for (char& ch : summary) {
            if (ch == '\n' || ch == '\r') ch = ' ';
        }
        ctes.push_back("-- " + std::to_string(i + 1) + ". " + summary + "\n" + name + " AS (\n  " + body + "\n)");
        previous = name;
    }
    std::string sql = "-- Guided step: " + std::to_string(step.ops.size()) +
                      " operation(s), written by the Engine's guided editor.\n"
                      "-- Change it in Guided mode; editing this SQL turns it into an SQL step.\n";
    if (ctes.empty()) {
        sql += "SELECT * FROM " + previous + "\n";
    } else {
        sql += "WITH\n" + Join(ctes, ",\n") + "\nSELECT * FROM " + previous + "\n";
    }
    out.sql = std::move(sql);
    return out;
}

const std::vector<Template>& Templates() {
    static const std::vector<Template> templates = {
        {"paragraphs", "Split documents into paragraphs", "split at blank lines, clean, drop empty, number"},
        {"clean", "Clean a text column", "markup, footnote marks, spaces; drop empty and repeated rows"},
        {"lookup", "Look up and keep matched rows", "bring columns from a second input; rows with no match are dropped"},
        {"split", "Train / validation / test split", "stable hash split with checks"},
    };
    return templates;
}

GuidedStep MakeTemplate(std::string_view id, const std::string& text_column) {
    GuidedStep step;
    const auto add = [&](const char* type, json params) { step.ops.push_back({type, true, std::move(params)}); };
    const std::string c = text_column.empty() ? std::string("text") : text_column;
    if (id == "paragraphs") {
        add("split_into_rows", DefaultParams("split_into_rows", c));
        json clean = DefaultParams("clean_text", c);
        clean["remove_markup"] = true;
        clean["remove_footnote_marks"] = true;
        add("clean_text", clean);
        add("keep_rows", {{"match", "all"}, {"conditions", json::array({{{"column", c}, {"op", "is not empty"}, {"value", ""}}})}});
    } else if (id == "clean") {
        json clean = DefaultParams("clean_text", c);
        clean["remove_markup"] = true;
        add("clean_text", clean);
        add("keep_rows", {{"match", "all"}, {"conditions", json::array({{{"column", c}, {"op", "is not empty"}, {"value", ""}}})}});
        add("remove_duplicates", {{"columns", json::array({c})}});
    } else if (id == "lookup") {
        json lookup = DefaultParams("look_up", c);
        lookup["no_match"] = "remove";
        add("look_up", lookup);
    } else if (id == "split") {
        add("assign_split", DefaultParams("assign_split", c));
        add("add_checks", {{"checks", json::array({{{"kind", "allowed_values"}, {"column", "split"}, {"values", "train, validation, test"}}})}});
    }
    return step;
}

SplitChecks SplitOffChecks(const GuidedStep& step) {
    SplitChecks out;
    out.step = step;
    size_t last_on = step.ops.size();
    for (size_t i = 0; i < step.ops.size(); ++i)
        if (step.ops[i].on) last_on = i;
    if (last_on == step.ops.size() || step.ops[last_on].type != "add_checks") return out;
    for (const auto& c : List(step.ops[last_on].params, "checks")) {
        const std::string kind = Str(c, "kind", "row_count");
        const std::string column = Str(c, "column");
        CheckNode node;
        std::string label;
        auto& p = node.parameters;
        p["check"] = kind;
        if (kind == "row_count") {
            p["expected_rows"] = Str(c, "exact");
            p["min_rows"] = Str(c, "min");
            p["max_rows"] = Str(c, "max");
            if (p["expected_rows"].empty() && p["min_rows"].empty() && p["max_rows"].empty()) continue;
            label = "rows";
        } else if (kind == "unique") {
            auto keys = StrList(c, "columns");
            if (keys.empty() && !column.empty()) keys.push_back(column);
            if (keys.empty()) continue;
            p["columns"] = Join(keys, ", ");
            label = p["columns"] + " unique";
        } else if (column.empty()) {
            continue;
        } else if (kind == "allowed_values") {
            p["column"] = column;
            p["allowed"] = Str(c, "values");
            label = column + " values";
        } else if (kind == "not_empty") {
            p["column"] = column;
            label = column + " not empty";
        } else {
            continue;
        }
        p["check_name"] = label;
        node.name = "Check: " + label;
        out.checks.push_back(std::move(node));
    }
    out.step.ops.erase(out.step.ops.begin() + static_cast<std::ptrdiff_t>(last_on));
    return out;
}

}  // namespace cyxwiz::guided
