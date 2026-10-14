#include "ml_readiness.h"

#include "session_query_engine.h"

#include <arrow/api.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <optional>

namespace cyxwiz::readiness {

namespace {

std::string Q(const std::string& name) { return SessionQueryEngine::QuoteIdentifier(name); }

// Words in a text cell: runs of non-space; an empty text has none.
std::string Words(const std::string& column) {
    const std::string v = "trim(CAST(" + Q(column) + " AS VARCHAR))";
    return "CASE WHEN " + v + " = '' THEN 0 ELSE len(string_split_regex(" + v + ", '\\s+')) END";
}

std::optional<std::string> Cell(const arrow::Table& table, int column, int64_t row) {
    if (column >= table.num_columns() || row >= table.num_rows()) return std::nullopt;
    auto scalar = table.column(column)->GetScalar(row);
    if (!scalar.ok() || !(*scalar)->is_valid) return std::nullopt;
    if ((*scalar)->type->id() == arrow::Type::STRING) return std::static_pointer_cast<arrow::StringScalar>(*scalar)->ToString();
    return (*scalar)->ToString();
}

int64_t IntCell(const arrow::Table& table, int column, int64_t row) {
    const auto s = Cell(table, column, row);
    if (!s) return -1;
    try {
        return static_cast<int64_t>(std::stod(*s));
    } catch (...) {
        return -1;
    }
}

double DoubleCell(const arrow::Table& table, int column, int64_t row) {
    const auto s = Cell(table, column, row);
    if (!s) return std::numeric_limits<double>::quiet_NaN();
    try {
        return std::stod(*s);
    } catch (...) {
        return std::numeric_limits<double>::quiet_NaN();
    }
}

QueryRequest Request(const std::string& sql, const std::vector<std::string>& inputs, const std::string& label) {
    QueryRequest r;
    r.sql = sql;
    r.inputs = inputs;
    r.label = label;
    return r;
}

// 641.5, 555, 690.7
std::string Number(double v) {
    if (!std::isfinite(v)) return "-";
    char buf[32];
    if (std::fabs(v - std::round(v)) < 0.05) return studio::Thousands(static_cast<int64_t>(std::llround(v)));
    std::snprintf(buf, sizeof(buf), "%.1f", v);
    return buf;
}

std::string Share(int64_t part, int64_t whole) {
    if (whole <= 0) return "-";
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%.0f%%", 100.0 * static_cast<double>(part) / static_cast<double>(whole));
    return buf;
}

std::string Join(const std::vector<std::string>& parts, const std::string& last_sep) {
    std::string out;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i) out += i + 1 == parts.size() ? last_sep : ", ";
        out += parts[i];
    }
    return out;
}

std::string Capital(std::string s) {
    if (!s.empty() && s[0] >= 'a' && s[0] <= 'z') s[0] = static_cast<char>(s[0] - 'a' + 'A');
    return s;
}

struct Kind {
    const char* one;           // "group"
    const char* noun;          // "groups"
    const std::string* column; // the role's column
    int64_t Overlap::*count;
};

}  // namespace

Roles PickRoles(const DatasetContract& contract) {
    Roles roles;
    const auto pick = [&](ColumnRole role, std::string& into) {
        for (RoleSource source : {RoleSource::User, RoleSource::Contract, RoleSource::Inferred})
            for (const auto& c : contract.columns)
                if (c.role == role && c.source == source) {
                    into = c.name;
                    return;
                }
    };
    pick(ColumnRole::Text, roles.text);
    pick(ColumnRole::Id, roles.id);
    pick(ColumnRole::Group, roles.group);
    pick(ColumnRole::Target, roles.target);
    return roles;
}

std::string SplitSql(const std::string& table, const Roles& roles) {
    std::string sql = "SELECT count(*) AS rows";
    sql += roles.group.empty() ? ", NULL AS groups" : ", count(DISTINCT " + Q(roles.group) + ") AS groups";
    sql += roles.id.empty() ? ", NULL AS ids" : ", count(DISTINCT " + Q(roles.id) + ") AS ids";
    if (roles.text.empty()) sql += ", NULL AS words_median, NULL AS words_mean";
    else sql += ", median(" + Words(roles.text) + ") AS words_median, avg(" + Words(roles.text) + ") AS words_mean";
    return sql + " FROM " + Q(table);
}

std::string OverlapSql(const std::string& a, const std::string& b, const std::string& column) {
    const std::string c = Q(column);
    return "SELECT count(*) AS n FROM (SELECT CAST(" + c + " AS VARCHAR) FROM " + Q(a) + " WHERE " + c +
           " IS NOT NULL INTERSECT SELECT CAST(" + c + " AS VARCHAR) FROM " + Q(b) + " WHERE " + c + " IS NOT NULL)";
}

std::string ClassSql(const std::string& table, const std::string& target, size_t limit) {
    const std::string c = Q(target);
    return "SELECT CAST(" + c + " AS VARCHAR) AS value, count(*) AS n FROM " + Q(table) + " GROUP BY 1 ORDER BY n DESC, value LIMIT " +
           std::to_string(limit);
}

Facts Measure(const std::string& name, const std::vector<SplitTable>& splits, const Roles& roles, const QueryRunner& run,
              const std::function<bool()>& should_stop) {
    const auto started = std::chrono::steady_clock::now();
    Facts f;
    f.name = name;
    f.roles = roles;
    const auto stopped = [&] {
        if (should_stop && should_stop()) {
            f.error = "Cancelled.";
            return true;
        }
        return false;
    };
    for (const auto& s : splits) {
        if (stopped()) return f;
        SplitFacts sf;
        sf.table = s;
        const QueryResult r = run(Request(SplitSql(s.dataset, roles), {s.dataset}, "ML readiness: " + s.split));
        if (!r.ok || !r.table) {
            sf.error = r.error.empty() ? "The query returned nothing." : r.error;
        } else {
            sf.rows = IntCell(*r.table, 0, 0);
            sf.groups = IntCell(*r.table, 1, 0);
            sf.ids = IntCell(*r.table, 2, 0);
            sf.words_median = DoubleCell(*r.table, 3, 0);
            sf.words_mean = DoubleCell(*r.table, 4, 0);
        }
        if (!roles.target.empty() && sf.error.empty()) {
            const QueryResult c = run(Request(ClassSql(s.dataset, roles.target, 50), {s.dataset}, "ML readiness: classes"));
            if (c.ok && c.table)
                for (int64_t row = 0; row < c.table->num_rows(); ++row)
                    sf.classes.emplace_back(Cell(*c.table, 0, row).value_or("(missing)"), IntCell(*c.table, 1, row));
        }
        f.splits.push_back(std::move(sf));
    }
    for (size_t a = 0; a < f.splits.size(); ++a)
        for (size_t b = a + 1; b < f.splits.size(); ++b) {
            Overlap o;
            o.a = a;
            o.b = b;
            if (f.splits[a].error.empty() && f.splits[b].error.empty()) {
                const auto count = [&](const std::string& column) -> int64_t {
                    if (column.empty() || stopped()) return -1;
                    const auto& ta = f.splits[a].table.dataset;
                    const auto& tb = f.splits[b].table.dataset;
                    const QueryResult r = run(Request(OverlapSql(ta, tb, column), {ta, tb}, "ML readiness: overlap"));
                    return r.ok && r.table ? IntCell(*r.table, 0, 0) : -1;
                };
                o.groups = count(roles.group);
                o.ids = count(roles.id);
                o.texts = count(roles.text);
            }
            if (!f.error.empty()) return f;
            f.overlaps.push_back(o);
        }
    f.elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    return f;
}

View Build(const Facts& f) {
    using studio::Tone;
    View v;
    v.title = f.name;
    std::vector<std::string> names;
    for (const auto& s : f.splits) names.push_back(s.table.split);
    for (size_t i = 0; i < names.size(); ++i) v.splits += (i ? " \xC2\xB7 " : "") + names[i];

    const std::vector<Kind> kinds = {{"group", "groups", &f.roles.group, &Overlap::groups},
                                     {"ID", "IDs", &f.roles.id, &Overlap::ids},
                                     {"identical text", "identical texts", &f.roles.text, &Overlap::texts}};
    const auto counted_noun = [](const Kind& k, int64_t n) { return studio::Thousands(n) + " " + (n == 1 ? k.one : k.noun); };

    // The verdict.
    bool counted = false;
    std::vector<std::string> leaks;
    for (const auto& k : kinds) {
        int64_t total = 0;
        for (const auto& o : f.overlaps)
            if (o.*k.count >= 0) {
                counted = true;
                total += o.*k.count;
            }
        if (total > 0) leaks.push_back(counted_noun(k, total));
    }
    if (f.splits.size() < 2) {
        v.verdict = "One split only";
        v.verdict_tone = Tone::Warn;
        v.verdict_tip = "Leakage is checked between splits: connect a validation or test Data Input to the Data Split.";
    } else if (!leaks.empty()) {
        v.verdict = "Leakage: " + Join(leaks, " and ") + " in two splits";
        v.verdict_tone = Tone::Error;
        v.verdict_tip = "The same values are in more than one split, so validation and test scores read too high. Leakage across splits below says where.";
    } else if (!counted) {
        v.verdict = "Leakage not checked";
        v.verdict_tone = Tone::Warn;
        v.verdict_tip = "Set a Group, ID or Text column in Explore > Profile.";
    } else {
        v.verdict = "No leakage found";
        v.verdict_tone = Tone::Ok;
        v.verdict_tip = "No group, ID or text of one split is in another (exact matches).";
    }

    // Roles.
    {
        Card c;
        c.title = "Roles (dataset contract)";
        const auto role = [&](const char* label, const std::string& column, const char* missing) {
            if (!column.empty()) c.lines.push_back({std::string(label) + "  " + column});
            else c.lines.push_back({std::string(label) + ": " + missing, Tone::Plain, true});
        };
        role("Text", f.roles.text, "none");
        role("ID", f.roles.id, "none");
        role("Group", f.roles.group, "not set (Explore > Profile)");
        role("Target", f.roles.target, f.roles.text.empty() ? "none" : "none (a text corpus)");
        c.lines.push_back({"From the train split; set roles in Explore > Profile.", Tone::Plain, true});
        v.cards.push_back(std::move(c));
    }

    // Split sizes.
    {
        Card c;
        c.title = "Split sizes";
        c.header = {"", "rows"};
        if (!f.roles.group.empty()) c.header.push_back("groups");
        if (!f.roles.id.empty()) c.header.push_back("IDs");
        for (const auto& s : f.splits) {
            if (!s.error.empty()) {
                c.lines.push_back({s.table.split + ": " + s.error, Tone::Error});
                continue;
            }
            std::vector<std::string> row = {s.table.split, studio::Thousands(s.rows)};
            if (!f.roles.group.empty()) row.push_back(studio::Thousands(s.groups));
            if (!f.roles.id.empty()) row.push_back(studio::Thousands(s.ids));
            c.rows.push_back(std::move(row));
        }
        for (const auto& s : f.splits)
            if (s.error.empty() && !f.roles.id.empty() && s.ids >= 0 && s.ids < s.rows)
                c.lines.push_back({s.table.split + ": " + studio::Thousands(s.rows - s.ids) + (s.rows - s.ids == 1 ? " row repeats" : " rows repeat") +
                                       " an ID",
                                   Tone::Warn});
        v.cards.push_back(std::move(c));
    }

    // Leakage across splits.
    if (f.splits.size() >= 2) {
        Card c;
        c.title = "Leakage across splits";
        for (const auto& k : kinds) {
            if (k.column->empty()) {
                c.lines.push_back({Capital(k.noun) + ": no " + (k.count == &Overlap::texts ? "Text" : k.count == &Overlap::ids ? "ID" : "Group") +
                                       " column",
                                   Tone::Plain, true});
                continue;
            }
            int64_t total = 0;
            bool known = false;
            for (const auto& o : f.overlaps)
                if (o.*k.count >= 0) {
                    known = true;
                    total += o.*k.count;
                }
            if (!known) {
                c.lines.push_back({Capital(k.noun) + ": not counted", Tone::Warn});
                continue;
            }
            c.lines.push_back({counted_noun(k, total) + " in two splits (" + *k.column + ")", total ? Tone::Error : Tone::Ok});
            for (const auto& o : f.overlaps)
                if (o.*k.count > 0)
                    c.lines.push_back({"   " + studio::Thousands(o.*k.count) + " in " + f.splits[o.a].table.split + " and " +
                                           f.splits[o.b].table.split,
                                       Tone::Error});
        }
        c.lines.push_back({"Exact matches between each pair of splits.", Tone::Plain, true});
        v.cards.push_back(std::move(c));
    }

    // Text length.
    if (!f.roles.text.empty()) {
        Card c;
        c.title = "Text length (words)";
        c.header = {"", "median", "mean"};
        for (const auto& s : f.splits)
            if (s.error.empty()) c.rows.push_back({s.table.split, Number(s.words_median), Number(s.words_mean)});
        // Splits whose median is 10% or more off the train split's.
        if (!f.splits.empty() && std::isfinite(f.splits[0].words_median) && f.splits[0].words_median > 0) {
            const double base = f.splits[0].words_median;
            std::vector<std::string> shorter, longer;
            double shorter_sum = 0, longer_sum = 0;
            for (size_t i = 1; i < f.splits.size(); ++i) {
                const double m = f.splits[i].words_median;
                if (!std::isfinite(m)) continue;
                const double diff = 100.0 * (m - base) / base;
                if (diff <= -10.0) {
                    shorter.push_back(f.splits[i].table.split);
                    shorter_sum -= diff;
                } else if (diff >= 10.0) {
                    longer.push_back(f.splits[i].table.split);
                    longer_sum += diff;
                }
            }
            const auto say = [&](const std::vector<std::string>& which, double sum, const char* word) {
                if (which.empty()) return;
                const int pct = static_cast<int>(std::lround(sum / static_cast<double>(which.size())));
                c.lines.push_back({Capital(Join(which, " and ")) + (which.size() > 1 ? " are" : " is") + " about " + std::to_string(pct) + "% " +
                                       word + " than " + f.splits[0].table.split + ".",
                                   Tone::Warn});
            };
            say(shorter, shorter_sum, "shorter");
            say(longer, longer_sum, "longer");
        }
        c.lines.push_back({"Median and mean words per row of " + f.roles.text + ".", Tone::Plain, true});
        v.cards.push_back(std::move(c));
    }

    // Class balance.
    {
        Card c;
        c.title = "Class balance";
        if (f.roles.target.empty()) {
            c.lines.push_back({"Appears when a Target is set (the Data Input label, or Explore > Profile).", Tone::Plain, true});
        } else if (!f.splits.empty() && !f.splits[0].classes.empty()) {
            c.header = {f.roles.target};
            for (const auto& s : f.splits) c.header.push_back(s.table.split);
            const auto share_of = [](const SplitFacts& s, const std::string& value) -> double {
                if (s.rows <= 0) return -1;
                for (const auto& [cls, n] : s.classes)
                    if (cls == value) return 100.0 * static_cast<double>(n) / static_cast<double>(s.rows);
                return 0;
            };
            const auto& train = f.splits[0];
            constexpr size_t kShown = 8;
            for (size_t i = 0; i < train.classes.size() && i < kShown; ++i) {
                std::vector<std::string> row = {train.classes[i].first};
                for (const auto& s : f.splits) {
                    int64_t n = 0;
                    for (const auto& [cls, k] : s.classes)
                        if (cls == train.classes[i].first) n = k;
                    row.push_back(Share(n, s.rows));
                }
                c.rows.push_back(std::move(row));
            }
            if (train.classes.size() > kShown)
                c.lines.push_back({"The " + std::to_string(kShown) + " largest of " + std::to_string(train.classes.size()) +
                                       (train.classes.size() >= 50 ? "+" : "") + " classes in " + train.table.split + ".",
                                   Tone::Plain, true});
            const int64_t largest = train.classes.front().second, smallest = train.classes.back().second;
            if (train.classes.size() >= 2 && smallest > 0 && train.classes.size() < 50) {
                const double ratio = static_cast<double>(largest) / static_cast<double>(smallest);
                char buf[96];
                std::snprintf(buf, sizeof(buf), "%s: largest class %s, smallest %s (%.0f to 1).", train.table.split.c_str(),
                              Share(largest, train.rows).c_str(), Share(smallest, train.rows).c_str(), ratio);
                c.lines.push_back({buf, ratio >= 10.0 ? Tone::Warn : Tone::Plain});
            }
            // A class whose share moves 10 points or more between train and another split.
            for (size_t i = 1; i < f.splits.size(); ++i)
                for (const auto& [value, n] : train.classes) {
                    const double a = share_of(train, value), b = share_of(f.splits[i], value);
                    if (a < 0 || b < 0 || std::fabs(a - b) < 10.0) continue;
                    char buf[64];
                    std::snprintf(buf, sizeof(buf), ": %.0f%% in %s, %.0f%% in %s.", a, train.table.split.c_str(), b,
                                  f.splits[i].table.split.c_str());
                    c.lines.push_back({value + buf, Tone::Warn});
                    break;
                }
        }
        if (!f.balance.empty()) c.lines.push_back({"Data Loader balance mode: " + f.balance + ".", Tone::Plain, true});
        v.cards.push_back(std::move(c));
    }
    return v;
}

}  // namespace cyxwiz::readiness
