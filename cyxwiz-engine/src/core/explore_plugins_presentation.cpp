#include "explore_plugins_presentation.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>

namespace cyxwiz::explore_plugins {

namespace {

bool Reads(ColumnKind kind, const Column& c) {
    switch (kind) {
        case ColumnKind::Numeric: return c.numeric;
        case ColumnKind::Text: return c.text;
        case ColumnKind::Any: break;
    }
    return true;
}

const char* KindOf(const Column& c) { return c.numeric ? "numbers" : c.text ? "text" : "other values"; }

std::string Plural(int64_t n, const char* one, const char* many) {
    return std::to_string(n) + " " + (n == 1 ? one : many);
}

std::string Grouped(int64_t n) {
    std::string digits = std::to_string(n < 0 ? -n : n), out;
    for (size_t i = 0; i < digits.size(); ++i) {
        if (i > 0 && (digits.size() - i) % 3 == 0) out += ',';
        out += digits[i];
    }
    return n < 0 ? "-" + out : out;
}

std::string CsvField(const std::string& s) {
    if (s.find_first_of(",\"\n\r") == std::string::npos) return s;
    std::string out = "\"";
    for (char c : s) out += c == '"' ? std::string("\"\"") : std::string(1, c);
    return out + "\"";
}

std::string Lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

}  // namespace

std::string Label(const Analytic& a) { return a.category.empty() ? a.name : a.name + " (" + a.category + ")"; }

std::string FromLine(const Analytic& a) {
    std::string from = "from " + (a.plugin_name.empty() ? std::string("a plugin") : a.plugin_name);
    if (!a.plugin_version.empty()) from += " " + a.plugin_version;
    return from;
}

std::vector<std::string> RefusedReasons(const Analytic& a, const std::vector<Column>& columns) {
    std::vector<std::string> reasons;
    reasons.reserve(columns.size());
    const char* wants = a.kind == ColumnKind::Numeric ? "numbers" : "text";
    for (const auto& c : columns)
        reasons.push_back(Reads(a.kind, c) ? std::string()
                                           : std::string(KindOf(c)) + " · this analytic reads " + wants);
    return reasons;
}

std::vector<std::string> DefaultColumns(const Analytic& a, const std::vector<Column>& columns) {
    std::vector<std::string> picked;
    for (const auto& c : columns) {
        if (a.max_columns > 0 && static_cast<int>(picked.size()) >= a.max_columns) break;
        if (Reads(a.kind, c)) picked.push_back(c.name);
    }
    return picked;
}

void KeepReadable(const Analytic& a, const std::vector<Column>& columns, std::vector<std::string>& picked) {
    std::erase_if(picked, [&](const std::string& name) {
        const auto it = std::find_if(columns.begin(), columns.end(), [&](const Column& c) { return c.name == name; });
        return it == columns.end() || !Reads(a.kind, *it);
    });
    if (a.max_columns > 0 && static_cast<int>(picked.size()) > a.max_columns) picked.resize(static_cast<size_t>(a.max_columns));
}

std::string ColumnsHint(const Analytic& a, const std::vector<Column>& columns) {
    if (a.kind == ColumnKind::Any) return {};
    const auto readable = std::count_if(columns.begin(), columns.end(), [&](const Column& c) { return Reads(a.kind, c); });
    return std::string(a.kind == ColumnKind::Numeric ? "numbers" : "text") + " only: " + std::to_string(readable) +
           " of " + Plural(static_cast<int64_t>(columns.size()), "column", "columns");
}

RunState CanRun(const Analytic& a, const std::vector<std::string>& picked, int64_t rows, bool plugin_loaded) {
    RunState s;
    const int n = static_cast<int>(picked.size());
    if (!plugin_loaded) {
        s.reason = (a.plugin_name.empty() ? std::string("Its plugin") : a.plugin_name) +
                   " is not loaded, so this analytic cannot run.";
    } else if (n < a.min_columns) {
        s.reason = "Pick at least " + Plural(a.min_columns, "column", "columns") + " (this analytic reads " +
                   std::to_string(a.min_columns) + (a.max_columns == a.min_columns ? "" : " or more") + ").";
    } else if (a.max_columns > 0 && n > a.max_columns) {
        s.reason = "Pick at most " + Plural(a.max_columns, "column", "columns") + ".";
    } else {
        s.enabled = true;
    }
    if (n > 0) s.summary = "reads " + Grouped(rows) + (rows == 1 ? " row" : " rows") + " of " + Plural(n, "column", "columns");
    return s;
}

std::string Key(const std::string& dataset, const std::string& analytic_id, const std::vector<std::string>& picked) {
    std::string key = dataset + '\x1f' + analytic_id;
    for (const auto& p : picked) key += '\x1f' + p;
    return key;
}

std::string NumberText(double v) {
    char buf[64];
    const double m = std::fabs(v);
    if (!std::isfinite(v)) return std::isnan(v) ? "NaN" : (v > 0 ? "inf" : "-inf");
    if (m != 0.0 && (m < 1e-4 || m >= 1e6)) std::snprintf(buf, sizeof(buf), "%.4g", v);
    else if (m >= 1000.0) std::snprintf(buf, sizeof(buf), "%.1f", v);
    else std::snprintf(buf, sizeof(buf), "%.4f", v);
    return buf;
}

AnswerView Present(const Result& r) {
    AnswerView view;
    if (r.has_scalar) view.headline = NumberText(r.scalar);
    double top = 0.0;
    for (double v : r.values)
        if (std::isfinite(v)) top = std::max(top, std::fabs(v));
    for (size_t i = 0; i < r.values.size(); ++i) {
        Bar bar;
        bar.label = i < r.labels.size() && !r.labels[i].empty() ? r.labels[i] : "value " + std::to_string(i + 1);
        bar.value = NumberText(r.values[i]);
        bar.fraction = top > 0.0 && std::isfinite(r.values[i]) ? static_cast<float>(std::fabs(r.values[i]) / top) : 0.0f;
        view.bars.push_back(std::move(bar));
    }
    view.note = r.text;
    view.value_header = r.name.empty() ? "value" : Lower(r.name);
    std::string by = "Computed by " + (r.plugin_name.empty() ? std::string("a plugin") : r.plugin_name);
    if (!r.plugin_version.empty()) by += " " + r.plugin_version;
    by += " on " + Grouped(r.rows) + (r.rows == 1 ? " row" : " rows");
    const double ms = std::max(0.0, r.milliseconds);
    char took[32];
    if (ms < 1000.0) std::snprintf(took, sizeof(took), "%.0f ms", ms);
    else std::snprintf(took, sizeof(took), "%.1f s", ms / 1000.0);
    by += std::string(" in ") + took;
    if (!r.finished_at.empty()) by += ", " + r.finished_at;
    view.by_line = by;
    return view;
}

std::string CopyText(const Result& r) {
    const AnswerView v = Present(r);
    std::string out;
    if (r.has_scalar) out += r.name + '\t' + v.headline + '\n';
    if (!v.bars.empty()) out += "column\t" + v.value_header + '\n';
    for (const auto& b : v.bars) out += b.label + '\t' + b.value + '\n';
    return out;
}

std::string Csv(const Result& r) {
    std::string out = "column," + CsvField(r.name.empty() ? std::string("value") : Lower(r.name)) + '\n';
    if (r.has_scalar) out += "(all)," + NumberText(r.scalar) + '\n';
    for (size_t i = 0; i < r.values.size(); ++i) {
        const std::string label = i < r.labels.size() ? r.labels[i] : "value " + std::to_string(i + 1);
        char value[64];
        std::snprintf(value, sizeof(value), "%.17g", r.values[i]);
        out += CsvField(label) + ',' + value + '\n';
    }
    return out;
}

}  // namespace cyxwiz::explore_plugins
