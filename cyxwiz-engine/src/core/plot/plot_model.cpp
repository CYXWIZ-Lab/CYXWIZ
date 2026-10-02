#include "plot_model.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>

namespace cyxwiz::plot {

const std::vector<KindInfo>& Kinds() {
    static const std::vector<KindInfo> kinds = {
        {Kind::Line, "line", "Line", Group::Basic, kEncY, kEncX | kEncColor, true, "X (optional: row number)", "Y values"},
        {Kind::Scatter, "scatter", "Scatter", Group::Basic, kEncX | kEncY, kEncColor, true, "X values", "Y values"},
        {Kind::Bar, "bar", "Bar", Group::Basic, kEncX, kEncY, false, "Categories", "Values (optional: count rows)"},
        {Kind::Histogram, "histogram", "Histogram", Group::Basic, kEncX, kEncColor, false, "Values", ""},
        {Kind::Area, "area", "Area", Group::Basic, kEncY, kEncX | kEncColor, true, "X (optional: row number)", "Y values"},
        {Kind::Step, "step", "Step", Group::Basic, kEncY, kEncX | kEncColor, true, "X (optional: row number)", "Y values"},
        {Kind::Stem, "stem", "Stem", Group::Basic, kEncY, kEncX, true, "X (optional: row number)", "Y values"},
        {Kind::Pie, "pie", "Pie", Group::Basic, kEncX, kEncY, false, "Categories", "Values (optional: count rows)"},
        {Kind::Box, "box", "Box", Group::Distribution, kEncY, kEncColor, true, "", "Values"},
        {Kind::Violin, "violin", "Violin", Group::Distribution, kEncY, kEncColor, true, "", "Values"},
        {Kind::ErrorBars, "error_bars", "Error bars", Group::Distribution, kEncX | kEncY, 0, false, "Groups", "Values (mean and spread)"},
        {Kind::Heatmap, "heatmap", "Heatmap", Group::GridDensity, kEncX | kEncY, 0, false, "Columns (categories)", "Rows (categories)"},
        {Kind::Histogram2D, "histogram_2d", "2D histogram", Group::GridDensity, kEncX | kEncY, 0, false, "X values", "Y values"},
    };
    return kinds;
}

const KindInfo& Info(Kind kind) {
    for (const auto& k : Kinds())
        if (k.kind == kind) return k;
    return Kinds().front();
}

const KindInfo* FindKind(const std::string& id) {
    for (const auto& k : Kinds())
        if (id == k.id) return &k;
    return nullptr;
}

const char* GroupLabel(Group group) {
    switch (group) {
        case Group::Basic: return "Basic";
        case Group::Distribution: return "Distribution";
        case Group::GridDensity: return "Grid and density";
    }
    return "";
}

std::string SpecToJson(const PlotSpec& s) {
    nlohmann::json j;
    j["version"] = PlotSpec::kVersion;
    j["kind"] = Info(s.kind).id;
    j["x"] = s.x_column;
    j["y"] = s.y_columns;
    j["color"] = s.color_column;
    j["title"] = s.title;
    j["x_label"] = s.x_label;
    j["y_label"] = s.y_label;
    j["bins"] = s.bins;
    j["smooth"] = s.smooth;
    j["density"] = s.density;
    j["show_mean"] = s.show_mean;
    j["show_median"] = s.show_median;
    j["log_x"] = s.log_x;
    j["log_y"] = s.log_y;
    j["legend"] = s.legend;
    return j.dump();
}

bool SpecFromJson(const std::string& text, PlotSpec& s, std::string* problem) {
    const auto j = nlohmann::json::parse(text, nullptr, false);
    const auto fail = [&](const std::string& why) {
        if (problem) *problem = why;
        return false;
    };
    if (j.is_discarded() || !j.is_object()) return fail("not a plot spec (JSON object expected)");
    const int version = j.value("version", 0);
    if (version < 1 || version > PlotSpec::kVersion)
        return fail("plot spec version " + std::to_string(version) + " is not supported");
    const KindInfo* kind = FindKind(j.value("kind", std::string()));
    if (!kind) return fail("unknown plot kind '" + j.value("kind", std::string()) + "'");
    PlotSpec out;
    out.kind = kind->kind;
    out.x_column = j.value("x", std::string());
    if (j.contains("y") && j["y"].is_array())
        for (const auto& y : j["y"])
            if (y.is_string()) out.y_columns.push_back(y.get<std::string>());
    out.color_column = j.value("color", std::string());
    out.title = j.value("title", std::string());
    out.x_label = j.value("x_label", std::string());
    out.y_label = j.value("y_label", std::string());
    out.bins = std::clamp(j.value("bins", 30), 1, 1000);
    out.smooth = std::clamp(j.value("smooth", 0), 0, 100000);
    out.density = j.value("density", false);
    out.show_mean = j.value("show_mean", false);
    out.show_median = j.value("show_median", false);
    out.log_x = j.value("log_x", false);
    out.log_y = j.value("log_y", false);
    out.legend = j.value("legend", true);
    s = std::move(out);
    return true;
}

std::string MissingEncoding(const PlotSpec& s) {
    const KindInfo& k = Info(s.kind);
    if ((k.required & kEncX) && s.x_column.empty()) return std::string("Choose ") + k.x_hint + ".";
    if ((k.required & kEncY) && s.y_columns.empty()) return std::string("Choose ") + k.y_hint + ".";
    return "";
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

std::string DataLabel::Text() const {
    const std::string n = Thousands(static_cast<long long>(shown));
    const std::string all = Thousands(static_cast<long long>(total));
    switch (state) {
        case State::Exact: return "exact \xC2\xB7 all " + n + " values";
        case State::Reduced: return "reduced \xC2\xB7 " + n + " of " + all + " points";
        case State::Sampled: return "sampled \xC2\xB7 " + n + " of " + all + " rows";
        case State::Truncated:
            return total > shown ? "first " + n + " of " + all + " rows" : "first " + n + " rows";
    }
    return "";
}

namespace {
double Quantile(const std::vector<double>& sorted, double q) {
    if (sorted.empty()) return 0.0;
    const double pos = q * static_cast<double>(sorted.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = std::min(lo + 1, sorted.size() - 1);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - static_cast<double>(lo));
}
}  // namespace

ColumnStats Summarize(const std::vector<double>& values) {
    ColumnStats st;
    std::vector<double> finite;
    finite.reserve(values.size());
    double sum = 0.0;
    for (double v : values) {
        if (std::isfinite(v)) {
            finite.push_back(v);
            sum += v;
        } else {
            ++st.missing;
        }
    }
    st.count = finite.size();
    if (finite.empty()) return st;
    std::sort(finite.begin(), finite.end());
    st.min = finite.front();
    st.max = finite.back();
    st.mean = sum / static_cast<double>(finite.size());
    st.median = Quantile(finite, 0.5);
    st.q1 = Quantile(finite, 0.25);
    st.q3 = Quantile(finite, 0.75);
    return st;
}

}  // namespace cyxwiz::plot
