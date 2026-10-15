#include "data_studio_checks.h"

#include <cctype>
#include <cstdlib>
#include <set>

namespace cyxwiz::studio {

using gui::NodeType;

namespace {

const CheckKind kKinds[kCheckKindCount] = {CheckKind::RowCount, CheckKind::AllowedValues, CheckKind::Unique,
                                           CheckKind::NotEmpty, CheckKind::Range};

std::string Trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

std::string Get(const std::map<std::string, std::string>& p, const char* key) {
    const auto it = p.find(key);
    return it == p.end() ? std::string() : Trim(it->second);
}

// "a, b ,c" -> "a, b, c" (the operator splits on commas and trims).
std::string ListText(const std::string& list) {
    std::string out, item;
    const auto flush = [&] {
        const std::string t = Trim(item);
        if (!t.empty()) out += (out.empty() ? "" : ", ") + t;
        item.clear();
    };
    for (char ch : list) {
        if (ch == ',') flush();
        else item += ch;
    }
    flush();
    return out;
}

bool WholeCount(const std::string& text, int64_t& value) {
    if (text.empty()) return false;
    for (char ch : text)
        if (!std::isdigit(static_cast<unsigned char>(ch))) return false;
    value = std::strtoll(text.c_str(), nullptr, 10);
    return true;
}

bool Number(const std::string& text, double& value) {
    if (text.empty()) return false;
    char* end = nullptr;
    value = std::strtod(text.c_str(), &end);
    return end && *end == '\0';
}

// "15349" -> "15,349"; other text as written.
std::string CountText(const std::string& text) {
    int64_t n = 0;
    return WholeCount(text, n) ? Thousands(n) : text;
}

// The leading count of a row-count pass ("passed: 15349 rows ..." -> 15349).
bool PassedCount(const std::string& outcome, int64_t& counted) {
    const std::string prefix = "passed: ";
    if (outcome.rfind(prefix, 0) != 0) return false;
    size_t i = prefix.size();
    std::string digits;
    while (i < outcome.size() && std::isdigit(static_cast<unsigned char>(outcome[i]))) digits += outcome[i++];
    return WholeCount(digits, counted);
}

}  // namespace

const char* CheckKindName(CheckKind kind) {
    switch (kind) {
        case CheckKind::RowCount: return "Row count";
        case CheckKind::AllowedValues: return "Allowed values";
        case CheckKind::Unique: return "Unique";
        case CheckKind::NotEmpty: return "Not empty";
        case CheckKind::Range: return "Range";
    }
    return "";
}

const char* CheckKindParameter(CheckKind kind) {
    switch (kind) {
        case CheckKind::RowCount: return "row_count";
        case CheckKind::AllowedValues: return "allowed_values";
        case CheckKind::Unique: return "unique";
        case CheckKind::NotEmpty: return "not_empty";
        case CheckKind::Range: return "range";
    }
    return "row_count";
}

bool CheckParameters(const CheckForm& form, std::map<std::string, std::string>& p, std::string& error) {
    p.clear();
    for (const char* key : {"expected_rows", "min_rows", "max_rows", "count_true_column", "count_column", "count_value",
                            "column", "columns", "allowed", "min_value", "max_value"})
        p[key] = "";
    p["check_name"] = Trim(form.name);
    p["check"] = CheckKindParameter(form.kind);
    p["on_failure"] = "stop";
    error.clear();
    const std::string column = Trim(form.column);
    switch (form.kind) {
        case CheckKind::RowCount: {
            int64_t a = 0, b = 0;
            if (!WholeCount(Trim(form.count), a)) {
                error = "Enter the number of rows (a whole number, 0 or more).";
                return false;
            }
            if (form.count_mode == 3) {
                if (!WholeCount(Trim(form.count_max), b)) {
                    error = "Enter the largest number of rows (a whole number).";
                    return false;
                }
                if (a > b) {
                    error = "The first number is larger than the second.";
                    return false;
                }
                p["min_rows"] = std::to_string(a);
                p["max_rows"] = std::to_string(b);
            } else {
                p[form.count_mode == 1 ? "min_rows" : form.count_mode == 2 ? "max_rows" : "expected_rows"] = std::to_string(a);
            }
            const std::string where_column = Trim(form.where_column);
            if (form.where != 0 && where_column.empty()) {
                error = "Pick the column the rows are counted by.";
                return false;
            }
            if (form.where == 1) {
                if (Trim(form.where_value).empty()) {
                    error = "Enter the value the rows have in " + where_column + ".";
                    return false;
                }
                p["count_column"] = where_column;
                p["count_value"] = Trim(form.where_value);
            } else if (form.where == 2) {
                p["count_true_column"] = where_column;
            }
            return true;
        }
        case CheckKind::AllowedValues:
            if (column.empty()) {
                error = "Pick the column to check.";
                return false;
            }
            if (ListText(form.allowed).empty()) {
                error = "List the allowed values, separated by commas.";
                return false;
            }
            p["column"] = column;
            p["allowed"] = ListText(form.allowed);
            return true;
        case CheckKind::Unique:
            if (ListText(form.columns).empty()) {
                error = "Pick the column(s) whose values must not repeat.";
                return false;
            }
            p["columns"] = ListText(form.columns);
            return true;
        case CheckKind::NotEmpty:
            if (column.empty()) {
                error = "Pick the column to check.";
                return false;
            }
            p["column"] = column;
            return true;
        case CheckKind::Range: {
            if (column.empty()) {
                error = "Pick the column to check.";
                return false;
            }
            const std::string lo = Trim(form.min_value), hi = Trim(form.max_value);
            double a = 0.0, b = 0.0;
            if (lo.empty() && hi.empty()) {
                error = "Enter the smallest value, the largest, or both.";
                return false;
            }
            if ((!lo.empty() && !Number(lo, a)) || (!hi.empty() && !Number(hi, b))) {
                error = "The smallest and largest values must be numbers.";
                return false;
            }
            if (!lo.empty() && !hi.empty() && a > b) {
                error = "The smallest value is larger than the largest.";
                return false;
            }
            p["column"] = column;
            p["min_value"] = lo;
            p["max_value"] = hi;
            return true;
        }
    }
    return false;
}

CheckForm CheckFormFrom(const std::map<std::string, std::string>& p) {
    CheckForm f;
    const std::string check = Get(p, "check");
    for (CheckKind kind : kKinds)
        if (check == CheckKindParameter(kind)) f.kind = kind;
    f.name = Get(p, "check_name");
    const std::string exact = Get(p, "expected_rows"), lo = Get(p, "min_rows"), hi = Get(p, "max_rows");
    if (!exact.empty()) {
        f.count_mode = 0;
        f.count = exact;
    } else if (!lo.empty() && !hi.empty()) {
        f.count_mode = 3;
        f.count = lo;
        f.count_max = hi;
    } else if (!lo.empty()) {
        f.count_mode = 1;
        f.count = lo;
    } else if (!hi.empty()) {
        f.count_mode = 2;
        f.count = hi;
    }
    if (!Get(p, "count_true_column").empty()) {
        f.where = 2;
        f.where_column = Get(p, "count_true_column");
    } else if (!Get(p, "count_column").empty()) {
        f.where = 1;
        f.where_column = Get(p, "count_column");
        f.where_value = Get(p, "count_value");
    }
    f.column = Get(p, "column");
    f.columns = ListText(Get(p, "columns"));
    if (f.kind == CheckKind::Unique && f.columns.empty()) f.columns = f.column;  // the operator reads `column` then
    f.allowed = ListText(Get(p, "allowed"));
    f.min_value = Get(p, "min_value");
    f.max_value = Get(p, "max_value");
    return f;
}

std::string CheckRule(const std::map<std::string, std::string>& p) {
    const CheckForm f = CheckFormFrom(p);
    switch (f.kind) {
        case CheckKind::RowCount: {
            std::string rule;
            if (f.count.empty()) return "no row count set";
            switch (f.count_mode) {
                case 0: rule = "exactly " + CountText(f.count); break;
                case 1: rule = "at least " + CountText(f.count); break;
                case 2: rule = "at most " + CountText(f.count); break;
                default: rule = CountText(f.count) + " to " + CountText(f.count_max); break;
            }
            rule += f.count == "1" && f.count_mode != 3 ? " row" : " rows";
            if (f.where == 1) rule += " where " + f.where_column + " = " + f.where_value;
            else if (f.where == 2) rule += " where " + f.where_column + " is true";
            return rule;
        }
        case CheckKind::AllowedValues:
            return (f.column.empty() ? "[no column]" : f.column) + " is one of " + (f.allowed.empty() ? "[no values]" : f.allowed);
        case CheckKind::Unique:
            if (f.columns.empty()) return "[no columns] are unique";
            return f.columns.find(',') == std::string::npos ? f.columns + " is unique" : f.columns + " together are unique";
        case CheckKind::NotEmpty:
            return (f.column.empty() ? "[no column]" : f.column) + " is never empty";
        case CheckKind::Range: {
            const std::string column = f.column.empty() ? "[no column]" : f.column;
            if (!f.min_value.empty() && !f.max_value.empty()) return column + " from " + f.min_value + " to " + f.max_value;
            if (!f.min_value.empty()) return column + " at least " + f.min_value;
            if (!f.max_value.empty()) return column + " at most " + f.max_value;
            return column + ": no range set";
        }
    }
    return "";
}

std::string CheckDefaultName(const std::map<std::string, std::string>& parameters) {
    std::string name = "Check: " + CheckRule(parameters);
    if (name.size() > 60) name = name.substr(0, 57) + "...";
    return name;
}

std::string ValidatorRule(const std::map<std::string, std::string>& p) {
    std::string rule;
    const auto add = [&](const char* label, const char* key) {
        const std::string list = ListText(Get(p, key));
        if (!list.empty()) rule += (rule.empty() ? "" : " \xC2\xB7 ") + std::string(label) + ": " + list;
    };
    add("required", "required_columns");
    add("not null", "not_null_columns");
    add("unique", "unique_columns");
    return rule.empty() ? "no rules set (open its dialog)" : rule;
}

std::vector<CheckRow> CheckRows(const std::vector<GraphNode>& nodes) {
    const auto by_id = [&](int id) -> const GraphNode* {
        for (const auto& n : nodes)
            if (n.id == id) return &n;
        return nullptr;
    };
    std::vector<CheckRow> rows;
    for (const auto& n : nodes) {
        const bool check = n.type == NodeType::RowCountCheck;
        if (!check && n.type != NodeType::DataValidator) continue;
        CheckRow r;
        r.id = n.id;
        r.validator = !check;
        r.name = n.name;
        // What it reads: the node before it, past any checks chained in front.
        const GraphNode* from = n.input_from.empty() ? nullptr : by_id(n.input_from.front());
        std::set<int> seen{n.id};
        while (from && from->type == NodeType::RowCountCheck && seen.insert(from->id).second)
            from = from->input_from.empty() ? nullptr : by_id(from->input_from.front());
        if (from && from->type != NodeType::RowCountCheck) {
            r.reads = from->name;
            r.reads_id = from->id;
        }
        r.rule = check ? CheckRule(n.parameters) : ValidatorRule(n.parameters);
        if (!n.error.empty()) {
            // A check that ran and found bad data says "<label> failed: ...";
            // anything else stopped it before it could look (a missing column).
            const bool failed = check && n.error.find(" failed: ") != std::string::npos;
            r.result = failed ? "fail" : "could not run";
            // The executor wraps it ("RowCountCheck: operator execution
            // failed: Invalid: 'Check: Law' failed: ..."); the check's words follow.
            r.detail = n.error;
            if (const size_t at = r.detail.find("Invalid: "); at != std::string::npos) r.detail = r.detail.substr(at + 9);
            r.tone = Tone::Error;
        } else if (!n.ran) {
            r.result = "not run";
        } else if (check) {
            const bool passed = n.outcome.rfind("passed", 0) == 0;
            int64_t counted = 0;
            r.result = !passed ? "not run"
                       : CheckFormFrom(n.parameters).kind == CheckKind::RowCount && PassedCount(n.outcome, counted)
                           ? "pass \xC2\xB7 " + Thousands(counted) + (counted == 1 ? " row" : " rows")
                           : "pass";
            r.detail = n.outcome;
            r.tone = passed ? Tone::Ok : Tone::Plain;
        } else {
            // A Data Validator's table is its issues.
            r.result = n.rows == 0 ? "no issues" : n.rows == 1 ? "1 issue" : Thousands(n.rows) + " issues";
            r.tone = n.rows == 0 ? Tone::Ok : Tone::Warn;
        }
        rows.push_back(std::move(r));
    }
    return rows;
}

}  // namespace cyxwiz::studio
