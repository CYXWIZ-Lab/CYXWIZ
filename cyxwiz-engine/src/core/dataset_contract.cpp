#include "dataset_contract.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <functional>

namespace cyxwiz {

namespace {

std::string Lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

bool EndsWith(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

const char* TypeText(ColumnFacts::Type t) {
    switch (t) {
        case ColumnFacts::Type::Integer: return "int";
        case ColumnFacts::Type::Float: return "float";
        case ColumnFacts::Type::Boolean: return "bool";
        case ColumnFacts::Type::Text: return "text";
        case ColumnFacts::Type::Temporal: return "date";
        case ColumnFacts::Type::Other: break;
    }
    return "other";
}

std::string Count(size_t n) {
    std::string s = std::to_string(n);
    for (int i = static_cast<int>(s.size()) - 3; i > 0; i -= 3) s.insert(static_cast<size_t>(i), ",");
    return s;
}

constexpr size_t kCategoryValues = 12;  // a number column with this many values or fewer is a category

}  // namespace

const char* RoleLabel(ColumnRole role) {
    switch (role) {
        case ColumnRole::Id: return "ID";
        case ColumnRole::Target: return "Target";
        case ColumnRole::Numeric: return "Numeric";
        case ColumnRole::Category: return "Category";
        case ColumnRole::DateTime: return "Date";
        case ColumnRole::Text: return "Text";
        case ColumnRole::FilePath: return "File path";
        case ColumnRole::Weight: return "Weight";
        case ColumnRole::Ignore: return "Ignore";
    }
    return "Numeric";
}

const char* RoleId(ColumnRole role) {
    switch (role) {
        case ColumnRole::Id: return "id";
        case ColumnRole::Target: return "target";
        case ColumnRole::Numeric: return "numeric";
        case ColumnRole::Category: return "category";
        case ColumnRole::DateTime: return "date";
        case ColumnRole::Text: return "text";
        case ColumnRole::FilePath: return "file_path";
        case ColumnRole::Weight: return "weight";
        case ColumnRole::Ignore: return "ignore";
    }
    return "numeric";
}

std::optional<ColumnRole> RoleFromId(const std::string& id) {
    for (ColumnRole r : {ColumnRole::Id, ColumnRole::Target, ColumnRole::Numeric, ColumnRole::Category, ColumnRole::DateTime,
                         ColumnRole::Text, ColumnRole::FilePath, ColumnRole::Weight, ColumnRole::Ignore})
        if (id == RoleId(r)) return r;
    return std::nullopt;
}

const char* RoleSourceLabel(RoleSource source) {
    switch (source) {
        case RoleSource::Contract: return "contract";
        case RoleSource::User: return "you";
        case RoleSource::Inferred: return "inferred";
    }
    return "inferred";
}

bool IsFeatureRole(ColumnRole role) {
    return role != ColumnRole::Id && role != ColumnRole::Target && role != ColumnRole::Weight && role != ColumnRole::Ignore;
}

const ColumnContract* DatasetContract::Find(const std::string& name) const {
    for (const auto& c : columns)
        if (c.name == name) return &c;
    return nullptr;
}

std::optional<std::string> DatasetContract::Target() const {
    for (const auto& c : columns)
        if (c.role == ColumnRole::Target) return c.name;
    return std::nullopt;
}

bool LooksLikeDate(const std::string& text) {
    // Digits in the shapes dates take: YYYY-MM-DD / YYYY/MM/DD (optionally a
    // time after T or a space), DD/MM/YYYY or MM/DD/YYYY, YYYY-MM.
    std::string s;
    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c)) && s.empty()) continue;
        s += c;
    }
    const auto digits = [&](size_t at, size_t n) {
        if (at + n > s.size()) return false;
        for (size_t i = at; i < at + n; ++i)
            if (!std::isdigit(static_cast<unsigned char>(s[i]))) return false;
        return true;
    };
    const auto sep = [&](size_t at) { return at < s.size() && (s[at] == '-' || s[at] == '/' || s[at] == '.'); };
    if (digits(0, 4) && sep(4) && digits(5, 2)) {
        if (s.size() == 7) return true;  // 2024-05
        if (sep(7) && digits(8, 2)) {
            if (s.size() == 10) return true;
            return s[10] == 'T' || s[10] == ' ';
        }
        return false;
    }
    if (digits(0, 2) && sep(2) && digits(3, 2) && sep(5) && digits(6, 4)) return s.size() == 10 || s[10] == ' ';
    return false;
}

bool LooksLikeMediaPath(const std::string& text) {
    static const char* const kExt[] = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp",
                                       ".wav", ".mp3", ".flac", ".ogg", ".aiff", ".m4a"};
    const std::string s = Lower(text);
    for (const char* e : kExt)
        if (EndsWith(s, e)) return true;
    return false;
}

ColumnRole InferRole(const ColumnFacts& f, std::string* reason) {
    const auto say = [&](const std::string& why) {
        if (reason) *reason = why;
    };
    const std::string lname = Lower(f.name);
    const bool id_name = lname == "id" || EndsWith(lname, "_id") || EndsWith(lname, " id");
    const bool text = f.type == ColumnFacts::Type::Text;
    const bool unique = f.non_null > 1 && f.distinct == f.non_null;
    if (f.type == ColumnFacts::Type::Temporal) {
        say("date type");
        return ColumnRole::DateTime;
    }
    if (text && f.path_share >= 0.9) {
        say("file paths");
        return ColumnRole::FilePath;
    }
    if (text && f.date_share >= 0.9) {
        say("dates");
        return ColumnRole::DateTime;
    }
    // Long text that differs per row is text (reviews, statements), not an ID:
    // IDs are short (a UUID is 36 characters).
    if (text && f.avg_length >= 40.0) {
        say("long text");
        return ColumnRole::Text;
    }
    if (text && unique) {
        say("unique per row");
        return ColumnRole::Id;
    }
    // A row number saved with the data (a CSV's unnamed first column read as
    // "C0", "column0" or "Unnamed: 0", or one named index): an ID, not a measure.
    const auto numbered = [&](const std::string& prefix) {
        return lname.size() > prefix.size() && lname.compare(0, prefix.size(), prefix) == 0 &&
               lname.find_first_not_of("0123456789", prefix.size()) == std::string::npos;
    };
    const bool index_name = lname.empty() || lname == "index" || lname == "#" || numbered("c") || numbered("column") || numbered("unnamed: ");
    if (index_name && unique && f.type == ColumnFacts::Type::Integer) {
        say("row number");
        return ColumnRole::Id;
    }
    if (id_name && (text || f.type == ColumnFacts::Type::Integer)) {
        say(lname == "id" ? "named id" : "name ends in _id");
        return ColumnRole::Id;
    }
    if (f.type == ColumnFacts::Type::Boolean) {
        say(Count(f.distinct) + (f.distinct == 1 ? " value" : " values"));
        return ColumnRole::Category;
    }
    if (text) {
        // Long text, or names that are mostly different per row: text; else a category.
        const double share = f.non_null ? static_cast<double>(f.distinct) / static_cast<double>(f.non_null) : 0.0;
        if (f.avg_length >= 40.0) {
            say("long text");
            return ColumnRole::Text;
        }
        if (share > 0.5) {
            say(Count(f.distinct) + " values, mostly different per row");
            return ColumnRole::Text;
        }
        say(Count(f.distinct) + " values");
        return ColumnRole::Category;
    }
    if (f.type == ColumnFacts::Type::Integer && f.distinct > 0 && f.distinct <= kCategoryValues) {
        say(Count(f.distinct) + " values");
        return ColumnRole::Category;
    }
    say("numbers");
    return ColumnRole::Numeric;
}

std::string SchemaFingerprint(const std::vector<ColumnFacts>& columns) {
    std::string key;
    for (const auto& c : columns) key += c.name + ":" + TypeText(c.type) + ";";
    char buf[24];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(std::hash<std::string>{}(key)));
    return buf;
}

DatasetContract BuildContract(const std::string& source_key, const std::vector<ColumnFacts>& columns,
                              const std::map<std::string, ColumnRole>& contract_roles,
                              const std::map<std::string, ColumnRole>& user_roles) {
    DatasetContract out;
    out.source_key = source_key;
    out.schema_fingerprint = SchemaFingerprint(columns);
    for (const auto& f : columns) {
        ColumnContract c;
        c.name = f.name;
        c.type = TypeText(f.type);
        auto contract = contract_roles.find(f.name);
        auto user = user_roles.find(f.name);
        if (contract != contract_roles.end()) {
            c.role = contract->second;
            c.source = RoleSource::Contract;
            c.reason = contract->second == ColumnRole::Target ? "Data Input label" : "set by the graph";
            if (user != user_roles.end() && user->second != contract->second) c.user_conflict = user->second;
        } else if (user != user_roles.end()) {
            c.role = user->second;
            c.source = RoleSource::User;
            c.reason = "set in Data Studio";
            // Only the graph names the target for training: a user Target is a warning too.
            if (user->second == ColumnRole::Target) {
                for (const auto& [name, role] : contract_roles)
                    if (role == ColumnRole::Target && name != f.name) c.user_conflict = user->second;
            }
        } else {
            c.role = InferRole(f, &c.reason);
            c.source = RoleSource::Inferred;
        }
        out.columns.push_back(std::move(c));
    }
    for (const auto& [name, role] : user_roles) {
        (void)role;
        if (!out.Find(name)) out.unmatched.push_back(name);
    }
    return out;
}

}  // namespace cyxwiz
