#include "explore_presentation.h"

#include <cmath>
#include <cstdio>

namespace cyxwiz::explore_view {

namespace {

std::string Thousands(size_t n) {
    std::string s = std::to_string(n);
    for (int i = static_cast<int>(s.size()) - 3; i > 0; i -= 3) s.insert(static_cast<size_t>(i), ",");
    return s;
}

std::string Ident(const std::string& name) {
    std::string out = "\"";
    for (char c : name) out += c == '"' ? std::string("\"\"") : std::string(1, c);
    return out + "\"";
}

std::string Literal(const std::string& text) {
    std::string out = "'";
    for (char c : text) out += c == '\'' ? std::string("''") : std::string(1, c);
    return out + "'";
}

}  // namespace

std::string Number(size_t n) { return Thousands(n); }

std::string Count(size_t n, const char* one, const char* many) {
    return Thousands(n) + " " + (n == 1 ? one : many);
}

std::string Share(size_t part, size_t whole) {
    if (whole == 0) return "0%";
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%.1f%%", 100.0 * static_cast<double>(part) / static_cast<double>(whole));
    return buf;
}

std::vector<std::string> MissingAdvice(double share) {
    if (share <= 0) return {};
    if (share < 5) {
        return {"Few rows lack it: dropping them is usually safe.",
                "Or fill them: the median for numbers, the most frequent value for categories."};
    }
    if (share < 20) {
        return {"A fair share is missing: fill them from similar rows (nearest neighbours) or with a model,",
                "or add a column that says the value was missing."};
    }
    if (share < 50) {
        return {"Much is missing: add a column that says the value was missing,",
                "use a model that accepts missing values,", "and find out why it is missing (at random or not)."};
    }
    return {"More than half is missing: consider dropping the column,", "or check how the data was collected."};
}

const std::vector<std::string>& GeneralMissingAdvice() {
    static const std::vector<std::string> advice = {
        "Numbers: fill with the mean or median, or predict them from other columns.",
        "Categories: fill with the most frequent value, or an \"Unknown\" category.",
        "Time series: carry the last value forward, or interpolate.",
        "Write down how you filled them.",
    };
    return advice;
}

std::string MissingRule(size_t texts_columns) {
    if (texts_columns == 0) return "Missing = empty cells. No column has texts marked as missing;";
    return "Missing = empty cells, plus the texts marked as missing in " + Count(texts_columns, "column", "columns") + ";";
}

std::string MissingRowsSql(const std::string& table, const std::string& column, const std::vector<std::string>& texts) {
    std::string where = Ident(column) + " IS NULL";
    if (!texts.empty()) {
        std::string in;
        for (const auto& t : texts) in += (in.empty() ? "" : ", ") + Literal(t);
        where += " OR CAST(" + Ident(column) + " AS VARCHAR) IN (" + in + ")";
    }
    return "SELECT *\nFROM " + Ident(table) + "\nWHERE " + where;
}

std::string Figure(double v) {
    if (!std::isfinite(v)) return "";
    char buf[32];
    if (std::fabs(v) >= 1e6) std::snprintf(buf, sizeof(buf), "%.4g", v);
    else if (std::fabs(v) >= 1000 || std::fabs(v - std::round(v)) < 1e-9) std::snprintf(buf, sizeof(buf), "%.0f", v);
    else std::snprintf(buf, sizeof(buf), "%.4g", v);
    return buf;
}

std::string OutlierRowsSql(const std::string& table, const std::string& column, double lo, double hi) {
    const auto exact = [](double v) {
        char buf[40];
        std::snprintf(buf, sizeof(buf), "%.17g", v);
        return std::string(buf);
    };
    return "SELECT *\nFROM " + Ident(table) + "\nWHERE " + Ident(column) + " < " + exact(lo) + " OR " + Ident(column) + " > " + exact(hi);
}

const char* CorrelationStrength(double r) {
    const double a = std::fabs(r);
    if (!std::isfinite(a)) return "unknown";
    if (a >= 0.8) return "very strong";
    if (a >= 0.6) return "strong";
    if (a >= 0.4) return "moderate";
    if (a >= 0.2) return "weak";
    return "none";
}

std::string CorrelationSentence(double r) {
    if (!std::isfinite(r)) return "No correlation can be measured: a column has no spread in the rows both have.";
    const std::string strength = CorrelationStrength(r);
    if (strength == "none") return "No linear relation: one tells little about the other.";
    if (r > 0) return strength + " positive: when one is larger, so is the other.";
    return strength + " negative: when one is larger, the other tends to be smaller.";
}

std::string CorrelationCsv(const std::vector<std::string>& columns, const std::vector<double>& r) {
    const size_t n = columns.size();
    std::vector<std::vector<std::string>> rows;
    rows.push_back({"column"});
    for (const auto& c : columns) rows.back().push_back(c);
    for (size_t i = 0; i < n; ++i) {
        rows.push_back({columns[i]});
        for (size_t j = 0; j < n; ++j) {
            const double v = i * n + j < r.size() ? r[i * n + j] : NAN;
            char buf[32] = "";
            if (std::isfinite(v)) std::snprintf(buf, sizeof(buf), "%.6f", v);
            rows.back().push_back(buf);
        }
    }
    return Csv(rows);
}

const char* SkewWords(double skewness) {
    if (!std::isfinite(skewness)) return "";
    if (skewness > 0.2) return "longer right tail";
    if (skewness < -0.2) return "longer left tail";
    return "about symmetric";
}

const char* FigureHint(const std::string& label) {
    static const std::pair<const char*, const char*> kHints[] = {
        {"Count", "values present"},
        {"Mean", "the average"},
        {"Median", "the middle value"},
        {"Mode", "the most frequent value (the smallest of a tie)"},
        {"Std", "typical distance from the mean (sample)"},
        {"Variance", "std squared"},
        {"SEM", "how precisely the mean is known: std / sqrt(count)"},
        {"Min", "the smallest value"},
        {"Max", "the largest value"},
        {"Range", "max - min"},
        {"Sum", "all values added"},
        {"Q1", "25% of the values are below it"},
        {"Q3", "75% of the values are below it"},
        {"IQR", "Q3 - Q1: the middle half"},
        {"Skewness", "0 = symmetric; positive = a longer right tail"},
        {"Kurtosis", "excess: 0 = like a normal; negative = flatter, fewer extremes"},
        {"CV", "std as a share of the mean"},
    };
    for (const auto& [name, hint] : kHints)
        if (label == name) return hint;
    return "";
}

std::string Csv(const std::vector<std::vector<std::string>>& rows) {
    std::string out;
    for (const auto& row : rows) {
        for (size_t i = 0; i < row.size(); ++i) {
            if (i) out += ",";
            const std::string& cell = row[i];
            if (cell.find_first_of(",\"\n") == std::string::npos) {
                out += cell;
                continue;
            }
            out += "\"";
            for (char c : cell) out += c == '"' ? std::string("\"\"") : std::string(1, c);
            out += "\"";
        }
        out += "\n";
    }
    return out;
}

guided::GuidedStep KeepRowsWithValue(const std::string& column, const std::vector<std::string>& texts) {
    guided::GuidedStep step;
    nlohmann::json params = guided::DefaultParams("keep_rows", column);
    if (!texts.empty()) {
        std::string list;
        for (const auto& t : texts) list += (list.empty() ? "" : ", ") + t;
        params["conditions"].push_back({{"column", column}, {"op", "is not one of"}, {"value", list}});
    }
    step.ops.push_back({"keep_rows", true, std::move(params)});
    return step;
}

}  // namespace cyxwiz::explore_view
