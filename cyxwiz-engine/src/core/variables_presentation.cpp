#include "variables_presentation.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>

namespace cyxwiz::vars {

using json = nlohmann::json;

namespace {
std::string Str(const json& o, const char* key) {
    auto it = o.find(key);
    return it != o.end() && it->is_string() ? it->get<std::string>() : std::string();
}

std::string Lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool ContainsCI(const std::string& text, const std::string& needle_lower) {
    return needle_lower.empty() || Lower(text).find(needle_lower) != std::string::npos;
}
}  // namespace

std::vector<Variable> ParseVariables(const std::string& text) {
    std::vector<Variable> out;
    const json doc = json::parse(text, nullptr, false);
    if (doc.is_discarded() || !doc.is_array()) return out;
    for (const auto& item : doc) {
        if (!item.is_object()) continue;
        Variable v;
        auto more = item.find("more");
        if (more != item.end() && more->is_number_integer()) {
            v.more = more->get<long long>();
            out.push_back(std::move(v));
            continue;
        }
        v.name = Str(item, "name");
        auto step = item.find("step");
        if (step != item.end() && step->is_array()) v.step = step->dump();
        v.type = Str(item, "type");
        v.kind = Str(item, "kind");
        v.size = Str(item, "size");
        auto mem = item.find("memory");
        if (mem != item.end() && mem->is_number_integer()) v.memory = mem->get<long long>();
        v.value = Str(item, "value");
        v.expandable = item.value("expandable", false);
        v.viewable = item.value("viewable", false);
        v.digest = Str(item, "digest");
        out.push_back(std::move(v));
    }
    return out;
}

const char* ChipLabel(Chip chip) {
    switch (chip) {
        case Chip::All: return "All";
        case Chip::Tables: return "Tables";
        case Chip::Arrays: return "Arrays";
        case Chip::Numbers: return "Numbers";
        case Chip::Text: return "Text";
        case Chip::Collections: return "Collections";
    }
    return "All";
}

bool InChip(const Variable& v, Chip chip) {
    switch (chip) {
        case Chip::All: return true;
        case Chip::Tables: return v.kind == "table";
        case Chip::Arrays: return v.kind == "array";
        case Chip::Numbers: return v.kind == "number";
        case Chip::Text: return v.kind == "text";
        case Chip::Collections: return v.kind == "collection";
    }
    return true;
}

int ChipCount(const std::vector<Variable>& vars, Chip chip) {
    int n = 0;
    for (const auto& v : vars)
        if (v.more == 0 && InChip(v, chip)) ++n;
    return n;
}

bool Matches(const Variable& v, const std::string& filter, Chip chip) {
    if (v.more != 0 || !InChip(v, chip)) return false;
    const std::string needle = Lower(filter);
    return ContainsCI(v.name, needle) || ContainsCI(v.type, needle) || ContainsCI(v.value, needle);
}

std::string MemoryText(long long bytes) {
    if (bytes < 0) return {};
    char buf[32];
    if (bytes < 1024) std::snprintf(buf, sizeof(buf), "%lld B", bytes);
    else if (bytes < 1024LL * 1024) std::snprintf(buf, sizeof(buf), "%lld KB", (bytes + 512) / 1024);
    else if (bytes < 1024LL * 1024 * 1024) std::snprintf(buf, sizeof(buf), "%.1f MB", static_cast<double>(bytes) / (1024.0 * 1024.0));
    else std::snprintf(buf, sizeof(buf), "%.1f GB", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
    return buf;
}

std::string FooterText(const std::vector<Variable>& vars) {
    int count = 0;
    long long memory = 0;
    bool any = false;
    for (const auto& v : vars) {
        if (v.more != 0) continue;
        ++count;
        if (v.memory >= 0) {
            memory += v.memory;
            any = true;
        }
    }
    std::string text = std::to_string(count) + (count == 1 ? " variable" : " variables");
    if (any) text += " \xC2\xB7 arrays and tables " + MemoryText(memory);
    return text;
}

std::map<std::string, std::string> Digests(const std::vector<Variable>& vars) {
    std::map<std::string, std::string> out;
    for (const auto& v : vars)
        if (v.more == 0) out[v.name] = v.digest;
    return out;
}

std::set<std::string> ChangedNames(const std::map<std::string, std::string>& previous, const std::vector<Variable>& now) {
    std::set<std::string> out;
    if (previous.empty()) return out;
    for (const auto& v : now) {
        if (v.more != 0) continue;
        auto it = previous.find(v.name);
        if (it == previous.end() || it->second != v.digest) out.insert(v.name);
    }
    return out;
}

long long ElementCount(const std::string& size) {
    if (size.empty()) return -1;
    if (size.front() == '(') {
        long long product = 1;
        bool any = false;
        long long n = 0;
        bool in_number = false;
        for (char c : size) {
            if (std::isdigit(static_cast<unsigned char>(c))) {
                n = n * 10 + (c - '0');
                in_number = true;
            } else if (in_number) {
                product *= n;
                any = true;
                n = 0;
                in_number = false;
            }
        }
        return any ? product : 0;
    }
    long long n = 0;
    bool any = false;
    for (char c : size) {
        if (std::isdigit(static_cast<unsigned char>(c))) {
            n = n * 10 + (c - '0');
            any = true;
        } else if (any) {
            break;
        }
    }
    return any ? n : -1;
}

void Sort(std::vector<Variable>& vars, Column column, bool ascending) {
    auto key_less = [column](const Variable& a, const Variable& b) {
        switch (column) {
            case Column::Name: return Lower(a.name) < Lower(b.name);
            case Column::Type: return Lower(a.type) < Lower(b.type);
            case Column::Size: return ElementCount(a.size) < ElementCount(b.size);
            case Column::Memory: return a.memory < b.memory;
            case Column::Value: return a.value < b.value;
        }
        return false;
    };
    std::stable_sort(vars.begin(), vars.end(), [&](const Variable& a, const Variable& b) {
        if (a.more != b.more) return a.more == 0;  // "+ n more" stays last
        if (key_less(a, b)) return ascending;
        if (key_less(b, a)) return !ascending;
        return Lower(a.name) < Lower(b.name);
    });
}

std::string PathJson(const std::string& parent, const std::string& step) {
    json path = parent.empty() ? json::array() : json::parse(parent, nullptr, false);
    if (path.is_discarded() || !path.is_array()) path = json::array();
    const json s = json::parse(step, nullptr, false);
    if (!s.is_discarded()) path.push_back(s);
    return path.dump();
}

namespace {
void AddRows(const std::vector<Variable>& level, const std::string& parent, int depth, const std::set<std::string>& open,
             const std::map<std::string, std::vector<Variable>>& children, std::vector<TreeRow>& out) {
    for (const auto& v : level) {
        TreeRow row;
        row.var = &v;
        row.depth = depth;
        row.path = v.more != 0 ? parent : PathJson(parent, v.step);
        row.open = v.expandable && open.count(row.path) > 0;
        auto it = children.find(row.path);
        row.loading = row.open && it == children.end();
        out.push_back(row);
        if (row.open && it != children.end()) AddRows(it->second, row.path, depth + 1, open, children, out);
    }
}
}  // namespace

std::vector<TreeRow> Flatten(const std::vector<Variable>& top, const std::set<std::string>& open,
                             const std::map<std::string, std::vector<Variable>>& children) {
    std::vector<TreeRow> out;
    AddRows(top, std::string(), 0, open, children, out);
    return out;
}

std::string Thousands(long long n) {
    std::string digits = std::to_string(n < 0 ? -n : n);
    std::string out;
    for (size_t i = 0; i < digits.size(); ++i) {
        if (i > 0 && (digits.size() - i) % 3 == 0) out += ',';
        out += digits[i];
    }
    return n < 0 ? "-" + out : out;
}

namespace {
std::string ShapeText(const std::vector<long long>& shape) {
    std::string s = "(";
    for (size_t i = 0; i < shape.size(); ++i) s += (i ? ", " : "") + Thousands(shape[i]);
    return s + (shape.size() == 1 ? ",)" : ")");
}
}  // namespace

std::string LiveHeader(const LiveTable& t, const std::string& clock) {
    const std::string where = t.scope_label == "Python session" ? "the Python session" : t.scope_label;
    std::string what;
    if (t.kind == "frame")
        what = Thousands(t.rows) + (t.rows == 1 ? " row" : " rows") + " \xC3\x97 " + Thousands(t.columns) +
               (t.columns == 1 ? " column" : " columns");
    else if (t.kind == "array")
        what = (t.dtype.empty() ? std::string("array ") : t.dtype + " array ") + ShapeText(t.shape);
    else
        what = "list of " + Thousands(t.rows);
    return "Variable " + t.name + " from " + where + " \xC2\xB7 " + what + " \xC2\xB7 read " + clock;
}

std::string LimitText(const LiveTable& t) {
    if (t.shown >= t.rows) return {};
    return "Showing the first " + Thousands(t.shown) + " of " + Thousands(t.rows) + " rows. Sorting and stats use these rows.";
}

std::string SliceText(const LiveTable& t) {
    if (t.kind != "array" || t.shape.size() <= 2) return {};
    std::string s = t.name + "[";
    for (size_t i = 0; i < t.shape.size(); ++i) {
        if (i) s += ", ";
        s += i < t.slice.size() ? std::to_string(t.slice[i]) : std::string(":");
    }
    const size_t n = t.shape.size();
    return s + "] \xC2\xB7 " + Thousands(t.shape[n - 2]) + " \xC3\x97 " + Thousands(t.shape[n - 1]);
}

std::string ReadStatus(const std::string& reason, const std::string& clock) {
    return "Read " + reason + " \xC2\xB7 " + clock;
}

}  // namespace cyxwiz::vars
