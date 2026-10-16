#include "python_plot_request.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>

namespace cyxwiz::plot {

namespace {

std::string Joined(const std::vector<std::string>& names) {
    std::string out;
    for (size_t i = 0; i < names.size() && i < 12; ++i) out += (i ? ", " : "") + names[i];
    if (names.size() > 12) out += ", ...";
    return out;
}

std::string Thousands(size_t n) {
    const std::string digits = std::to_string(n);
    std::string out;
    for (size_t i = 0; i < digits.size(); ++i) {
        if (i > 0 && (digits.size() - i) % 3 == 0) out += ',';
        out += digits[i];
    }
    return out;
}

std::string Lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

}  // namespace

bool ParsePythonPlotRequest(const std::string& json, PythonPlotRequest& out, std::string* error) {
    const auto fail = [&](const std::string& why) {
        if (error) *error = why;
        return false;
    };
    const auto j = nlohmann::json::parse(json, nullptr, false);
    if (j.is_discarded() || !j.is_object() || !j.contains("spec") || !j["spec"].is_object()) return fail("not a plot request");
    PythonPlotRequest r;
    std::string problem;
    if (!SpecFromJson(j["spec"].dump(), r.spec, &problem)) return fail(problem);
    if (j.contains("columns") && j["columns"].is_array())
        for (const auto& c : j["columns"])
            if (c.is_string()) r.columns.push_back(c.get<std::string>());
    r.source = j.value("source", std::string());
    r.file = j.value("file", std::string());
    r.line = j.value("line", 0);
    r.title = r.spec.title;

    const KindInfo& info = Info(r.spec.kind);
    const std::string kind = Lower(info.label);
    // Every column the spec names must be sent.
    std::vector<std::string> used;
    const auto use = [&](const std::string& c) {
        if (!c.empty()) used.push_back(c);
    };
    use(r.spec.x_column);
    for (const auto& y : r.spec.y_columns) use(y);
    use(r.spec.color_column);
    use(r.spec.value_column);
    use(r.spec.z_column);
    use(r.spec.u_column);
    use(r.spec.v_column);
    for (const auto& c : r.spec.spread_columns) use(c);
    for (const auto& c : used)
        if (std::find(r.columns.begin(), r.columns.end(), c) == r.columns.end())
            return fail("column '" + c + "' is not in the data (columns: " + Joined(r.columns) + ")");
    // The kind's required columns, worded as the Plot window asks for them.
    if ((info.required & kEncX) && r.spec.x_column.empty())
        return fail(kind + " needs " + Lower(info.x_hint[0] ? info.x_hint : "an X column"));
    if ((info.required & kEncY) && r.spec.y_columns.empty())
        return fail(kind + " needs " + Lower(info.y_hint[0] ? info.y_hint : "a Y column"));
    if ((info.required & kEncZ) && r.spec.z_column.empty()) return fail(kind + " needs a Z column");
    if ((info.required & kEncValue) && r.spec.value_column.empty())
        return fail(kind + " needs " + Lower(info.value_hint[0] ? info.value_hint : "a value column"));
    if ((info.required & kEncVector) && (r.spec.u_column.empty() || r.spec.v_column.empty()))
        return fail(kind + " needs the arrow columns (u and v)");
    if (r.spec.kind == Kind::Surface && r.spec.surface_from == PlotSpec::SurfaceFrom::XYZ &&
        (r.spec.x_column.empty() || r.spec.y_columns.empty() || r.spec.z_column.empty()))
        return fail("surface from rows needs X, Y and Z (or a 2D array of heights)");
    out = std::move(r);
    return true;
}

std::string PythonPlotSourceText(const PythonPlotRequest& request, size_t rows) {
    std::string s = "from Python";
    if (!request.source.empty()) s += " \xC2\xB7 " + request.source;
    s += " \xC2\xB7 " + Thousands(rows) + (rows == 1 ? " row" : " rows");
    if (!request.file.empty()) s += " \xC2\xB7 " + std::filesystem::path(request.file).filename().string();
    // Code run from the editor or the Console has no file: the line alone.
    if (request.line > 0) s += (request.file.empty() ? " \xC2\xB7 line " : " line ") + std::to_string(request.line);
    return s;
}

}  // namespace cyxwiz::plot
