#include "explore_presentation.h"

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
