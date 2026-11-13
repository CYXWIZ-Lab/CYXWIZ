#include "optimization_plots.h"

#include "../ui_tokens.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace cyxwiz::optplot {

namespace {

using optcalc::Number;

// A plot's frame: the area on screen, its data range, and the gutter the
// y labels sit in.
struct Frame {
    ImVec2 at;
    float width = 0, height = 0, left = 0, bottom = 0;
    double x0 = 0, x1 = 1, y0 = 0, y1 = 1;

    ImVec2 P(double x, double y) const {
        const float w = width - left, h = height - bottom;
        return ImVec2(at.x + left + static_cast<float>((x - x0) / (x1 - x0)) * w,
                      at.y + h - static_cast<float>((y - y0) / (y1 - y0)) * h);
    }
    ImVec2 Min() const { return ImVec2(at.x + left, at.y); }
    ImVec2 Max() const { return ImVec2(at.x + width, at.y + height - bottom); }
};

// Reserves the area and draws the background and the axis end labels.
Frame Begin(float width, float height, double x0, double x1, double y0, double y1, const std::string& x_lo,
            const std::string& x_hi, const std::string& y_lo, const std::string& y_hi, const char* x_label,
            const char* y_label) {
    const ui::Tokens& t = ui::CurrentTokens();
    Frame f;
    f.at = ImGui::GetCursorScreenPos();
    f.width = width;
    f.height = height;
    f.left = (std::max)(ImGui::CalcTextSize(y_lo.c_str()).x, ImGui::CalcTextSize(y_hi.c_str()).x) + 8.0f;
    f.bottom = ImGui::GetTextLineHeight() + 4.0f;
    f.x0 = x0;
    f.x1 = x1 > x0 ? x1 : x0 + 1;
    f.y0 = y0;
    f.y1 = y1 > y0 ? y1 : y0 + 1;
    ImGui::Dummy(ImVec2(width, height));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(f.Min(), f.Max(), ui::ToU32(t.plot_bg), 4.0f);
    const ImU32 faint = ui::ToU32(t.text_faint);
    const float base = f.at.y + height - f.bottom + 2.0f;
    dl->AddText(ImVec2(f.at.x + f.left, base), faint, x_lo.c_str());
    dl->AddText(ImVec2(f.at.x + width - ImGui::CalcTextSize(x_hi.c_str()).x, base), faint, x_hi.c_str());
    if (x_label && *x_label)
        dl->AddText(ImVec2(f.at.x + f.left + (width - f.left - ImGui::CalcTextSize(x_label).x) * 0.5f, base), faint, x_label);
    dl->AddText(ImVec2(f.at.x, f.at.y), faint, y_hi.c_str());
    dl->AddText(ImVec2(f.at.x, f.at.y + height - f.bottom - ImGui::GetTextLineHeight()), faint, y_lo.c_str());
    if (y_label && *y_label) dl->AddText(ImVec2(f.at.x, f.at.y + (height - f.bottom) * 0.5f - ImGui::GetTextLineHeight() * 0.5f), faint, y_label);
    return f;
}

void Clip(const Frame& f) { ImGui::GetWindowDrawList()->PushClipRect(f.Min(), f.Max(), true); }

void Contours(const Frame& f, const optcalc::SurfaceMap& map) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const size_t n = map.levels.size();
    for (size_t k = 0; k < n; ++k) {
        ImVec4 c = t.accent_text;
        c.w = 0.85f - 0.55f * static_cast<float>(k) / static_cast<float>((std::max<size_t>)(1, n - 1));  // low lines stronger
        const ImU32 colour = ui::ToU32(c);
        for (const auto& s : map.levels[k].segments) dl->AddLine(f.P(s.x1, s.y1), f.P(s.x2, s.y2), colour, 1.0f);
    }
}

void Diamond(ImDrawList* dl, ImVec2 c, float r, ImU32 colour) {
    dl->AddQuad(ImVec2(c.x, c.y - r), ImVec2(c.x + r, c.y), ImVec2(c.x, c.y + r), ImVec2(c.x - r, c.y), colour, 1.5f);
}

}  // namespace

void Map(const optcalc::SurfaceMap& map, const std::vector<Path>& paths, const std::vector<Mark>& marks, float width,
         float height) {
    const ui::Tokens& t = ui::CurrentTokens();
    const Frame f = Begin(width, height, map.x_min, map.x_max, map.y_min, map.y_max, Number(map.x_min, 4), Number(map.x_max, 4),
                          Number(map.y_min, 4), Number(map.y_max, 4), "x", "y");
    ImDrawList* dl = ImGui::GetWindowDrawList();
    Clip(f);
    Contours(f, map);
    for (const Mark& m : marks) {
        const ImVec2 p = f.P(m.x, m.y);
        if (m.diamond) Diamond(dl, p, 6.0f, ui::ToU32(t.text_bright));
        else dl->AddCircleFilled(p, 4.5f, ui::ToU32(t.text_bright));
        if (!m.caption.empty()) dl->AddText(ImVec2(p.x + 8.0f, p.y - ImGui::GetTextLineHeight()), ui::ToU32(t.text_bright), m.caption.c_str());
    }
    for (const Path& path : paths) {
        const size_t count = path.shown == 0 ? path.points.size() : (std::min)(path.points.size(), path.shown + 1);
        std::vector<ImVec2> line;
        line.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            const auto& q = path.points[i];
            if (q.size() < 2 || !std::isfinite(q[0]) || !std::isfinite(q[1])) break;
            // Far outside the view a path is drawn to the edge, not to its overflow.
            const double x = std::clamp(q[0], map.x_min - 10 * (map.x_max - map.x_min), map.x_max + 10 * (map.x_max - map.x_min));
            const double y = std::clamp(q[1], map.y_min - 10 * (map.y_max - map.y_min), map.y_max + 10 * (map.y_max - map.y_min));
            line.push_back(f.P(x, y));
        }
        const ImU32 colour = ui::ToU32(path.colour);
        if (line.size() > 1) dl->AddPolyline(line.data(), static_cast<int>(line.size()), colour, ImDrawFlags_None, 2.0f);
        if (!line.empty()) dl->AddCircleFilled(line.back(), 4.0f, colour);
    }
    dl->PopClipRect();
}

void Program(const optcalc::ProgramPlot& plot, const std::vector<std::string>& row_names,
             const std::vector<std::string>& variables, const optcalc::SurfaceMap* map, float width, float height) {
    const ui::Tokens& t = ui::CurrentTokens();
    const Frame f = Begin(width, height, plot.x_min, plot.x_max, plot.y_min, plot.y_max, Number(plot.x_min, 3),
                          Number(plot.x_max, 3), Number(plot.y_min, 3), Number(plot.y_max, 3),
                          variables.size() > 0 ? variables[0].c_str() : "", variables.size() > 1 ? variables[1].c_str() : "");
    ImDrawList* dl = ImGui::GetWindowDrawList();
    Clip(f);
    if (map && map->ok()) Contours(f, *map);
    if (plot.region.size() >= 3) {
        std::vector<ImVec2> polygon;
        for (const auto& p : plot.region) polygon.push_back(f.P(p.first, p.second));
        ImVec4 fill = t.accent;
        fill.w = 0.28f;
        dl->AddConvexPolyFilled(polygon.data(), static_cast<int>(polygon.size()), ui::ToU32(fill));
    }
    // The axes through 0 when they are in view.
    const ImU32 faint = ui::ToU32(t.text_faint);
    if (plot.y_min < 0 && plot.y_max > 0) dl->AddLine(f.P(plot.x_min, 0), f.P(plot.x_max, 0), faint, 1.0f);
    if (plot.x_min < 0 && plot.x_max > 0) dl->AddLine(f.P(0, plot.y_min), f.P(0, plot.y_max), faint, 1.0f);
    for (size_t i = 0; i < plot.lines.size(); ++i) {
        const auto& s = plot.lines[i];
        if (s.x1 == s.x2 && s.y1 == s.y2) continue;
        const ImU32 colour = ui::ToU32(ui::SeriesColour(t, i));
        const ImVec2 a = f.P(s.x1, s.y1), b = f.P(s.x2, s.y2);
        dl->AddLine(a, b, colour, 1.5f);
        if (i < row_names.size()) {
            const ImVec2 label((a.x + b.x) * 0.5f + 4.0f, (a.y + b.y) * 0.5f - ImGui::GetTextLineHeight());
            dl->AddText(label, colour, row_names[i].c_str());
        }
    }
    if (plot.has_best) {
        const ImVec2 p = f.P(plot.best_x, plot.best_y);
        dl->AddCircleFilled(p, 5.0f, ui::ToU32(t.success));
        const std::string caption = "best (" + Number(plot.best_x, 4) + ", " + Number(plot.best_y, 4) + ")";
        dl->AddText(ImVec2(p.x + 8.0f, p.y - ImGui::GetTextLineHeight() - 2.0f), ui::ToU32(t.success), caption.c_str());
    }
    dl->PopClipRect();
}

void ShadedCurve(const std::vector<double>& xs, const std::vector<double>& ys, double from, double to, float width,
                 float height) {
    const ui::Tokens& t = ui::CurrentTokens();
    double lo = 0.0, hi = 0.0;
    for (const double y : ys)
        if (std::isfinite(y)) {
            lo = (std::min)(lo, y);
            hi = (std::max)(hi, y);
        }
    if (!(hi > lo)) hi = lo + 1.0;
    const double pad = 0.08 * (hi - lo);
    const double x0 = xs.empty() ? 0 : xs.front(), x1 = xs.empty() ? 1 : xs.back();
    const Frame f = Begin(width, height, x0, x1, lo - pad, hi + pad, Number(x0, 4), Number(x1, 4), Number(lo, 4), Number(hi, 4), "x",
                          "f(x)");
    ImDrawList* dl = ImGui::GetWindowDrawList();
    Clip(f);
    ImVec4 fill = t.accent;
    fill.w = 0.30f;
    const ImU32 shade = ui::ToU32(fill);
    // Strips side by side: anti-aliased edges would overlap into stripes.
    const ImDrawListFlags flags = dl->Flags;
    dl->Flags &= ~ImDrawListFlags_AntiAliasedFill;
    for (size_t i = 0; i + 1 < xs.size(); ++i) {
        const double a = xs[i], b = xs[i + 1];
        if (b < from || a > to || !std::isfinite(ys[i]) || !std::isfinite(ys[i + 1])) continue;
        // A strip to the x axis, as two triangles (the area may cross 0).
        const ImVec2 p0 = f.P(a, 0), p1 = f.P(a, ys[i]), p2 = f.P(b, ys[i + 1]), p3 = f.P(b, 0);
        dl->AddTriangleFilled(p0, p1, p2, shade);
        dl->AddTriangleFilled(p0, p2, p3, shade);
    }
    dl->Flags = flags;
    dl->AddLine(f.P(x0, 0), f.P(x1, 0), ui::ToU32(t.text_faint), 1.0f);
    std::vector<ImVec2> line;
    for (size_t i = 0; i < xs.size(); ++i) {
        if (!std::isfinite(ys[i])) {
            if (line.size() > 1) dl->AddPolyline(line.data(), static_cast<int>(line.size()), ui::ToU32(t.accent_text), ImDrawFlags_None, 2.0f);
            line.clear();
            continue;
        }
        line.push_back(f.P(xs[i], ys[i]));
    }
    if (line.size() > 1) dl->AddPolyline(line.data(), static_cast<int>(line.size()), ui::ToU32(t.accent_text), ImDrawFlags_None, 2.0f);
    dl->PopClipRect();
}

void LogLines(const std::vector<double>& xs, const std::vector<LogLine>& lines, bool log_x, const char* x_label,
              const char* y_label, float width, float height) {
    const ui::Tokens& t = ui::CurrentTokens();
    const auto lx = [log_x](double x) { return log_x ? std::log10(x) : x; };
    double x0 = std::numeric_limits<double>::infinity(), x1 = -x0, y0 = x0, y1 = -x0;
    for (const double x : xs)
        if (std::isfinite(x) && (!log_x || x > 0)) {
            x0 = (std::min)(x0, lx(x));
            x1 = (std::max)(x1, lx(x));
        }
    for (const auto& line : lines)
        for (const double y : line.ys)
            if (std::isfinite(y) && y > 0) {
                y0 = (std::min)(y0, std::log10(y));
                y1 = (std::max)(y1, std::log10(y));
            }
    if (!std::isfinite(x0) || !std::isfinite(y0)) {
        ImGui::Dummy(ImVec2(width, height));
        return;
    }
    y0 = std::floor(y0);
    y1 = std::ceil(y1);
    if (!(y1 > y0)) y1 = y0 + 1;
    const auto decade = [](double e) { return "1e" + std::to_string(static_cast<int>(e)); };
    // A log x axis reads right to left when it runs from large steps to small ones.
    const bool reversed = log_x && xs.size() > 1 && xs.front() > xs.back();
    const std::string x_lo = log_x ? decade(std::round(reversed ? x1 : x0)) : Number(x0, 4);
    const std::string x_hi = log_x ? decade(std::round(reversed ? x0 : x1)) : Number(x1, 4);
    Frame f = Begin(width, height, reversed ? -x1 : x0, reversed ? -x0 : x1, y0, y1, x_lo, x_hi, decade(y0), decade(y1), x_label, y_label);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    Clip(f);
    const ImU32 grid = ui::ToU32(t.border_soft);
    for (double e = y0 + 1; e < y1; e += 1) dl->AddLine(f.P(f.x0, e), f.P(f.x1, e), grid, 1.0f);
    for (size_t k = 0; k < lines.size(); ++k) {
        std::vector<ImVec2> pts;
        for (size_t i = 0; i < xs.size() && i < lines[k].ys.size(); ++i) {
            const double y = lines[k].ys[i];
            if (!std::isfinite(y) || y <= 0 || (log_x && xs[i] <= 0)) continue;
            const double x = reversed ? -lx(xs[i]) : lx(xs[i]);
            pts.push_back(f.P(x, std::log10(y)));
        }
        if (pts.size() > 1) dl->AddPolyline(pts.data(), static_cast<int>(pts.size()), ui::ToU32(lines[k].colour), ImDrawFlags_None, 2.0f);
        const float lh = ImGui::GetTextLineHeight();
        dl->AddText(ImVec2(f.Max().x - 90.0f, f.Min().y + 4.0f + lh * static_cast<float>(k)), ui::ToU32(lines[k].colour), lines[k].name.c_str());
    }
    dl->PopClipRect();
}

}  // namespace cyxwiz::optplot
