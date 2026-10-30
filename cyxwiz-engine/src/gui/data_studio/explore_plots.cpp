#include "explore_plots.h"

#include "../../core/stats_presentation.h"
#include "../ui_tokens.h"

#include <imgui.h>

#include <algorithm>
#include <cmath>

namespace cyxwiz::explore_plot {

namespace {

namespace sv = stats_view;

constexpr size_t kPoints = 4000;

void DashedLine(ImDrawList* dl, ImVec2 a, ImVec2 b, ImU32 colour) {
    const float dx = b.x - a.x, dy = b.y - a.y, len = std::sqrt(dx * dx + dy * dy);
    if (len <= 0.0f) return;
    for (float d = 0.0f; d < len; d += 8.0f) {
        const float e = std::min(d + 4.0f, len);
        dl->AddLine(ImVec2(a.x + dx * d / len, a.y + dy * d / len), ImVec2(a.x + dx * e / len, a.y + dy * e / len), colour, 1.0f);
    }
}

// The x axis under a plot: low end, the title centred, high end.
void XAxis(const ImVec2& at, float width, const std::string& lo, const char* title, const std::string& hi) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::TextColored(t.text_faint, "%s", lo.c_str());
    ImGui::SameLine();
    const float x_start = ImGui::GetCursorPosX();
    ImGui::SetCursorPosX(std::max(x_start, x_start + (width - ImGui::CalcTextSize(title).x) * 0.5f - ImGui::CalcTextSize(lo.c_str()).x));
    ImGui::TextColored(t.text_faint, "%s", title);
    ImGui::SameLine();
    ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(), at.x - ImGui::GetWindowPos().x + ImGui::GetScrollX() + width - ImGui::CalcTextSize(hi.c_str()).x));
    ImGui::TextColored(t.text_faint, "%s", hi.c_str());
}

void Extent(const std::vector<double>& v, double& lo, double& hi) {
    for (const double x : v)
        if (std::isfinite(x)) {
            lo = std::min(lo, x);
            hi = std::max(hi, x);
        }
}

}  // namespace

void Scatter(const std::vector<double>& xs, const std::vector<double>& ys, float width, float height,
             const std::vector<std::pair<double, double>>* curve, int guide, const char* x_label, const char* y_label,
             const std::vector<int>* groups) {
    const ui::Tokens& t = ui::CurrentTokens();
    const size_t n = std::min(xs.size(), ys.size());
    if (n == 0) return;
    double x0 = INFINITY, x1 = -INFINITY, y0 = INFINITY, y1 = -INFINITY;
    for (size_t i = 0; i < n; ++i) {
        x0 = std::min(x0, xs[i]);
        x1 = std::max(x1, xs[i]);
        y0 = std::min(y0, ys[i]);
        y1 = std::max(y1, ys[i]);
    }
    if (curve)
        for (const auto& [cx, cy] : *curve) {
            y0 = std::min(y0, cy);
            y1 = std::max(y1, cy);
        }
    if (guide == 2) {
        x0 = y0 = std::min(x0, y0);
        x1 = y1 = std::max(x1, y1);
    }
    if (guide == 1) {
        y0 = std::min(y0, 0.0);
        y1 = std::max(y1, 0.0);
    }
    if (!(x1 > x0)) x1 = x0 + 1.0;
    if (!(y1 > y0)) y1 = y0 + 1.0;
    const double px = (x1 - x0) * 0.04, py = (y1 - y0) * 0.06;
    x0 -= px;
    x1 += px;
    y0 -= py;
    y1 += py;
    const ImVec2 at = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(width, height));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(at, ImVec2(at.x + width, at.y + height), ui::ToU32(t.plot_bg), 4.0f);
    const auto P = [&](double x, double y) {
        return ImVec2(at.x + static_cast<float>((x - x0) / (x1 - x0)) * width, at.y + height - static_cast<float>((y - y0) / (y1 - y0)) * height);
    };
    if (guide == 1) {
        const ImVec2 a = P(x0, 0.0), b = P(x1, 0.0);
        for (float x = a.x; x < b.x; x += 8.0f) dl->AddLine(ImVec2(x, a.y), ImVec2(std::min(x + 4.0f, b.x), a.y), ui::ToU32(t.text_faint), 1.0f);
    } else if (guide == 2) {
        dl->AddLine(P(x0, x0), P(x1, x1), ui::ToU32(t.text_faint), 1.0f);
    }
    ImVec4 dot = t.accent_text;
    dot.w = 0.7f;
    const size_t stride = std::max<size_t>(1, n / kPoints);
    ImVec4 faint = t.text_faint;
    faint.w = 0.55f;
    for (size_t i = 0; i < n; i += stride) {
        ImVec4 c = dot;
        if (groups && i < groups->size()) {
            const int g = (*groups)[i];
            c = g < 0 ? faint : ui::SeriesColour(t, static_cast<size_t>(g));
            if (g >= 0) c.w = 0.8f;
        }
        dl->AddCircleFilled(P(xs[i], ys[i]), 2.4f, ui::ToU32(c));
    }
    if (curve && curve->size() > 1) {
        std::vector<ImVec2> line;
        for (const auto& [cx, cy] : *curve) line.push_back(P(cx, cy));
        dl->AddPolyline(line.data(), static_cast<int>(line.size()), ui::ToU32(t.caution), ImDrawFlags_None, 2.0f);
    }
    const std::string lx = sv::Num(x0 + px), hx = sv::Num(x1 - px), ly = sv::Num(y0 + py), hy = sv::Num(y1 - py);
    dl->AddText(ImVec2(at.x + 4.0f, at.y + 2.0f), ui::ToU32(t.text_faint), (std::string(y_label) + "  " + hy).c_str());
    dl->AddText(ImVec2(at.x + 4.0f, at.y + height - ImGui::GetTextLineHeight() - 2.0f), ui::ToU32(t.text_faint), ly.c_str());
    XAxis(at, width, lx, x_label, hx);
}

void Bars(const std::vector<int64_t>& bins, double lo, double hi, float width, float height) {
    const ui::Tokens& t = ui::CurrentTokens();
    const ImVec2 at = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(width, height));
    if (bins.empty()) return;
    const int64_t top = *std::max_element(bins.begin(), bins.end());
    if (top <= 0) return;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(at, ImVec2(at.x + width, at.y + height), ui::ToU32(t.plot_bg), 4.0f);
    const float bw = width / static_cast<float>(bins.size());
    ImVec4 fill = t.accent_text;
    fill.w = 0.75f;
    for (size_t i = 0; i < bins.size(); ++i) {
        if (bins[i] == 0) continue;
        const float h = (height - 8.0f) * static_cast<float>(bins[i]) / static_cast<float>(top);
        const float x = at.x + static_cast<float>(i) * bw;
        dl->AddRectFilled(ImVec2(x + 1.0f, at.y + height - h), ImVec2(x + std::max(2.0f, bw - 1.0f), at.y + height), ui::ToU32(fill), 2.0f,
                          ImDrawFlags_RoundCornersTop);
    }
    const std::string l = sv::Num(lo), h = sv::Num(hi);
    ImGui::TextColored(t.text_faint, "%s", l.c_str());
    ImGui::SameLine();
    ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(), at.x - ImGui::GetWindowPos().x + ImGui::GetScrollX() + width - ImGui::CalcTextSize(h.c_str()).x));
    ImGui::TextColored(t.text_faint, "%s", h.c_str());
}

void Lines(const std::vector<double>& xs, const std::vector<Line>& lines, float width, float height, bool own_scale,
           double mark_x, const std::string& mark, const char* x_label, size_t first_colour) {
    if (xs.empty()) return;
    LinesOver(xs, lines, width, height, own_scale, mark_x, mark, x_label, sv::Num(xs.front()), sv::Num(xs.back()), first_colour);
}

void LinesOver(const std::vector<double>& xs, const std::vector<Line>& lines, float width, float height, bool own_scale,
               double mark_x, const std::string& mark, const char* x_label, const std::string& lo_text, const std::string& hi_text,
               size_t first_colour) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (xs.empty() || lines.empty()) return;
    double x0 = xs.front(), x1 = xs.back();
    if (!(x1 > x0)) x1 = x0 + 1.0;
    const double px = (x1 - x0) * 0.06;
    x0 -= px;
    x1 += px;
    // Each line's scale (or one shared), the data range and the padded one.
    std::vector<std::pair<double, double>> data(lines.size(), {INFINITY, -INFINITY});
    for (size_t l = 0; l < lines.size(); ++l) Extent(lines[l].ys, data[l].first, data[l].second);
    if (!own_scale) {
        double lo = INFINITY, hi = -INFINITY;
        for (const auto& [a, b] : data) {
            lo = std::min(lo, a);
            hi = std::max(hi, b);
        }
        for (auto& d : data) d = {lo, hi};
    }
    std::vector<std::pair<double, double>> scale;
    for (auto& [lo, hi] : data) {
        if (!std::isfinite(lo)) lo = 0.0, hi = 1.0;
        if (!(hi > lo)) hi = lo + 1.0;
        const double pad = (hi - lo) * 0.1;
        scale.emplace_back(lo - pad, hi + pad);
    }
    const ImVec2 at = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(width, height));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(at, ImVec2(at.x + width, at.y + height), ui::ToU32(t.plot_bg), 4.0f);
    // A text row on top (the scale's high end, the legend), the lines below it.
    const float line_h = ImGui::GetTextLineHeight();
    const float top = at.y + line_h + 6.0f, bottom = at.y + height - line_h - 4.0f;
    const auto X = [&](double x) { return at.x + static_cast<float>((x - x0) / (x1 - x0)) * width; };
    const auto Y = [&](size_t l, double y) {
        return bottom - static_cast<float>((y - scale[l].first) / (scale[l].second - scale[l].first)) * (bottom - top);
    };
    if (std::isfinite(mark_x)) {
        const float mx = X(mark_x);
        DashedLine(dl, ImVec2(mx, top), ImVec2(mx, at.y + height - 2.0f), ui::ToU32(t.text_faint));
        const float w = ImGui::CalcTextSize(mark.c_str()).x;
        const float tx = mx + 4.0f + w <= at.x + width ? mx + 4.0f : mx - 4.0f - w;
        dl->AddText(ImVec2(tx, at.y + height - line_h - 2.0f), ui::ToU32(t.text_dim), mark.c_str());
    }
    float legend_x = at.x + width - 4.0f;
    const size_t stride = std::max<size_t>(1, xs.size() / kPoints);  // a long series: every n-th point
    // The legend from the right; the lines in order, so a later line (a trend) draws over an earlier one (the series).
    for (size_t l = lines.size(); l-- > 0;) {
        legend_x -= ImGui::CalcTextSize(lines[l].name.c_str()).x + 18.0f;
        dl->AddCircleFilled(ImVec2(legend_x + 5.0f, at.y + 2.0f + line_h * 0.5f), 4.0f, ui::ToU32(ui::SeriesColour(t, first_colour + l)));
        dl->AddText(ImVec2(legend_x + 13.0f, at.y + 2.0f), ui::ToU32(t.text_dim), lines[l].name.c_str());
    }
    for (size_t l = 0; l < lines.size(); ++l) {
        const ImU32 c = ui::ToU32(ui::SeriesColour(t, first_colour + l));
        std::vector<ImVec2> pts;
        for (size_t i = 0; i < xs.size() && i < lines[l].ys.size(); i += stride)
            if (std::isfinite(lines[l].ys[i])) pts.emplace_back(X(xs[i]), Y(l, lines[l].ys[i]));
        if (pts.size() > 1) dl->AddPolyline(pts.data(), static_cast<int>(pts.size()), c, ImDrawFlags_None, 2.0f);
        if (stride == 1 && pts.size() <= 400)
            for (const auto& q : pts) dl->AddCircleFilled(q, 3.0f, c);
    }
    dl->AddText(ImVec2(at.x + 4.0f, at.y + 2.0f), ui::ToU32(t.text_faint), (lines[0].name + " " + sv::Num(data[0].second)).c_str());
    const std::string low = sv::Num(data[0].first);
    if (!std::isfinite(mark_x) || X(mark_x) > at.x + ImGui::CalcTextSize(low.c_str()).x + 12.0f)
        dl->AddText(ImVec2(at.x + 4.0f, at.y + height - line_h - 2.0f), ui::ToU32(t.text_faint), low.c_str());
    XAxis(at, width, lo_text, x_label, hi_text);
}

void Curve(const std::vector<double>& ys, float width, float height, double level, const std::string& caption,
           const char* x_label, const char* y_label) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (ys.size() < 2) return;
    double y0 = INFINITY, y1 = -INFINITY;
    Extent(ys, y0, y1);
    if (std::isfinite(level)) {
        y0 = std::min(y0, level);
        y1 = std::max(y1, level);
    }
    if (!(y1 > y0)) y1 = y0 + 1.0;
    const double pad = (y1 - y0) * 0.06;
    const double lo = y0 - pad, hi = y1 + pad;
    const ImVec2 at = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(width, height));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(at, ImVec2(at.x + width, at.y + height), ui::ToU32(t.plot_bg), 4.0f);
    const double n = static_cast<double>(ys.size() - 1);
    const auto P = [&](double i, double y) {
        return ImVec2(at.x + 4.0f + static_cast<float>(i / n) * (width - 8.0f), at.y + height - static_cast<float>((y - lo) / (hi - lo)) * height);
    };
    const size_t stride = std::max<size_t>(1, ys.size() / 1500);
    std::vector<ImVec2> pts;
    for (size_t i = 0; i < ys.size(); i += stride) pts.push_back(P(static_cast<double>(i), ys[i]));
    pts.push_back(P(n, ys.back()));
    dl->AddPolyline(pts.data(), static_cast<int>(pts.size()), ui::ToU32(ui::SeriesColour(t, 0)), ImDrawFlags_None, 2.0f);
    if (std::isfinite(level)) {
        const float y = P(0.0, level).y;
        DashedLine(dl, ImVec2(at.x + 4.0f, y), ImVec2(at.x + width - 4.0f, y), ui::ToU32(t.caution));
        dl->AddText(ImVec2(at.x + 8.0f, y - ImGui::GetTextLineHeight() - 2.0f), ui::ToU32(t.text_dim), caption.c_str());
    }
    dl->AddText(ImVec2(at.x + 4.0f, at.y + 2.0f), ui::ToU32(t.text_faint), (std::string(y_label) + "  " + sv::Num(y1)).c_str());
    XAxis(at, width, "1", x_label, sv::Num(n + 1.0));
}

void Dendrogram(const std::vector<std::array<double, 4>>& xs, const std::vector<std::array<double, 4>>& ys,
                const std::vector<int64_t>& leaf_rows, double cut, float width, float height) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (xs.empty() || leaf_rows.empty()) return;
    double top = 0.0;
    for (const auto& y : ys)
        for (const double v : y) top = std::max(top, v);
    if (std::isfinite(cut)) top = std::max(top, cut);
    if (!(top > 0.0)) top = 1.0;
    const double ceiling = top * 1.08;
    const float line_h = ImGui::GetTextLineHeight();
    const ImVec2 at = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(width, height + line_h + 4.0f));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(at, ImVec2(at.x + width, at.y + height), ui::ToU32(t.plot_bg), 4.0f);
    const double span = 10.0 * static_cast<double>(leaf_rows.size());
    const auto P = [&](double x, double y) {
        return ImVec2(at.x + static_cast<float>(x / span) * width, at.y + height - static_cast<float>(y / ceiling) * (height - line_h - 4.0f));
    };
    const ImU32 c = ui::ToU32(ui::SeriesColour(t, 0));
    for (size_t l = 0; l < xs.size() && l < ys.size(); ++l) {
        const ImVec2 pts[4] = {P(xs[l][0], ys[l][0]), P(xs[l][1], ys[l][1]), P(xs[l][2], ys[l][2]), P(xs[l][3], ys[l][3])};
        dl->AddPolyline(pts, 4, c, ImDrawFlags_None, 1.6f);
    }
    if (std::isfinite(cut)) {
        const float y = P(0.0, cut).y;
        DashedLine(dl, ImVec2(at.x + 4.0f, y), ImVec2(at.x + width - 4.0f, y), ui::ToU32(t.caution));
    }
    dl->AddText(ImVec2(at.x + 4.0f, at.y + 2.0f), ui::ToU32(t.text_faint), ("merge distance  " + sv::Num(top)).c_str());
    for (size_t i = 0; i < leaf_rows.size(); ++i) {
        const std::string label = std::to_string(leaf_rows[i]);
        const float w = ImGui::CalcTextSize(label.c_str()).x;
        dl->AddText(ImVec2(P(10.0 * static_cast<double>(i) + 5.0, 0.0).x - w * 0.5f, at.y + height + 2.0f), ui::ToU32(t.text_faint),
                    label.c_str());
    }
}

void Band(const std::vector<double>& xs, const Line& line, const Line& centre, const std::vector<double>& low, const std::vector<double>& high,
          float width, float height, double mark_x, const std::string& mark, const char* x_label, const std::string& lo_text, const std::string& hi_text) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (xs.size() < 2) return;
    double x0 = xs.front(), x1 = xs.back();
    if (!(x1 > x0)) x1 = x0 + 1.0;
    const double px = (x1 - x0) * 0.03;
    x0 -= px;
    x1 += px;
    double lo = INFINITY, hi = -INFINITY;
    Extent(line.ys, lo, hi);
    Extent(centre.ys, lo, hi);
    Extent(low, lo, hi);
    Extent(high, lo, hi);
    if (!std::isfinite(lo)) lo = 0.0, hi = 1.0;
    if (!(hi > lo)) hi = lo + 1.0;
    const double pad = (hi - lo) * 0.1;
    const double y0 = lo - pad, y1 = hi + pad;
    const ImVec2 at = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(width, height));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(at, ImVec2(at.x + width, at.y + height), ui::ToU32(t.plot_bg), 4.0f);
    const float line_h = ImGui::GetTextLineHeight();
    const float top = at.y + line_h + 6.0f, bottom = at.y + height - line_h - 4.0f;
    const auto X = [&](double x) { return at.x + static_cast<float>((x - x0) / (x1 - x0)) * width; };
    const auto Y = [&](double y) { return bottom - static_cast<float>((y - y0) / (y1 - y0)) * (bottom - top); };
    // The band: one quad per step where both edges are known.
    ImVec4 fill = ui::SeriesColour(t, 1);
    fill.w = 0.22f;
    const size_t stride = std::max<size_t>(1, xs.size() / kPoints);  // a long series: every n-th point
    for (size_t i = 0; i + stride < xs.size() && i + stride < low.size() && i + stride < high.size(); i += stride) {
        const size_t j = i + stride;
        if (!std::isfinite(low[i]) || !std::isfinite(high[i]) || !std::isfinite(low[j]) || !std::isfinite(high[j])) continue;
        const ImVec2 q[4] = {ImVec2(X(xs[i]), Y(high[i])), ImVec2(X(xs[j]), Y(high[j])), ImVec2(X(xs[j]), Y(low[j])), ImVec2(X(xs[i]), Y(low[i]))};
        dl->AddConvexPolyFilled(q, 4, ui::ToU32(fill));
    }
    if (std::isfinite(mark_x)) {
        const float mx = X(mark_x);
        DashedLine(dl, ImVec2(mx, top), ImVec2(mx, at.y + height - 2.0f), ui::ToU32(t.text_faint));
        const float w = ImGui::CalcTextSize(mark.c_str()).x;
        const float tx = mx + 4.0f + w <= at.x + width ? mx + 4.0f : mx - 4.0f - w;
        dl->AddText(ImVec2(tx, at.y + height - line_h - 2.0f), ui::ToU32(t.text_dim), mark.c_str());
    }
    float legend_x = at.x + width - 4.0f;
    const Line* both[2] = {&line, &centre};
    for (int l = 1; l >= 0; --l) {
        const ImU32 c = ui::ToU32(ui::SeriesColour(t, static_cast<size_t>(l)));
        const float thickness = l == 0 ? 1.6f : 2.0f;
        std::vector<ImVec2> pts;
        for (size_t i = 0; i < xs.size() && i < both[l]->ys.size(); i += stride) {
            if (std::isfinite(both[l]->ys[i])) {
                pts.emplace_back(X(xs[i]), Y(both[l]->ys[i]));
                continue;
            }
            if (pts.size() > 1) dl->AddPolyline(pts.data(), static_cast<int>(pts.size()), c, ImDrawFlags_None, thickness);
            pts.clear();
        }
        if (pts.size() > 1) dl->AddPolyline(pts.data(), static_cast<int>(pts.size()), c, ImDrawFlags_None, thickness);
        if (both[l]->name.empty()) continue;
        legend_x -= ImGui::CalcTextSize(both[l]->name.c_str()).x + 18.0f;
        dl->AddCircleFilled(ImVec2(legend_x + 5.0f, at.y + 2.0f + line_h * 0.5f), 4.0f, c);
        dl->AddText(ImVec2(legend_x + 13.0f, at.y + 2.0f), ui::ToU32(t.text_dim), both[l]->name.c_str());
    }
    dl->AddText(ImVec2(at.x + 4.0f, at.y + 2.0f), ui::ToU32(t.text_faint), sv::Num(hi).c_str());
    dl->AddText(ImVec2(at.x + 4.0f, at.y + height - line_h - 2.0f), ui::ToU32(t.text_faint), sv::Num(lo).c_str());
    XAxis(at, width, lo_text, x_label, hi_text);
}

void Stems(const std::vector<double>& values, const std::vector<double>& lower, const std::vector<double>& upper, float width, float height, int season,
           const char* y_label) {
    const ui::Tokens& t = ui::CurrentTokens();
    const size_t n = values.size();
    if (n < 2) return;
    double lo = -0.2, hi = 1.0;
    for (size_t k = 1; k < n; ++k) {
        if (std::isfinite(values[k])) lo = std::min(lo, values[k] - 0.05);
        if (k < lower.size() && std::isfinite(lower[k])) lo = std::min(lo, lower[k] - 0.05);
    }
    const ImVec2 at = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(width, height));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(at, ImVec2(at.x + width, at.y + height), ui::ToU32(t.plot_bg), 4.0f);
    const float line_h = ImGui::GetTextLineHeight();
    const float top = at.y + line_h + 4.0f, bottom = at.y + height - 4.0f;
    const float step = (width - 16.0f) / static_cast<float>(n);
    const auto X = [&](size_t k) { return at.x + 8.0f + (static_cast<float>(k) + 0.5f) * step; };
    const auto Y = [&](double y) { return bottom - static_cast<float>((y - lo) / (hi - lo)) * (bottom - top); };
    // The band's half-width around 0 (the backend centres it on the value).
    ImVec4 fill = t.text_faint;
    fill.w = 0.18f;
    for (size_t k = 1; k < n && k < lower.size() && k < upper.size(); ++k) {
        if (!std::isfinite(lower[k]) || !std::isfinite(upper[k])) continue;
        const double half = 0.5 * (upper[k] - lower[k]);
        dl->AddRectFilled(ImVec2(X(k) - step * 0.5f, Y(half)), ImVec2(X(k) + step * 0.5f, Y(-half)), ui::ToU32(fill));
    }
    const float zero = Y(0.0);
    dl->AddLine(ImVec2(at.x + 4.0f, zero), ImVec2(at.x + width - 4.0f, zero), ui::ToU32(t.text_faint), 1.0f);
    const ImU32 strong = ui::ToU32(ui::SeriesColour(t, 0));
    ImVec4 weak_c = ui::SeriesColour(t, 0);
    weak_c.w = 0.45f;
    const ImU32 weak = ui::ToU32(weak_c);
    for (size_t k = 0; k < n; ++k) {
        if (!std::isfinite(values[k])) continue;
        const bool outside = k > 0 && k < lower.size() && k < upper.size() && std::isfinite(lower[k]) && std::isfinite(upper[k]) &&
                             std::fabs(values[k]) > 0.5 * (upper[k] - lower[k]);
        if (season > 0 && k > 0 && static_cast<int>(k) % season == 0) DashedLine(dl, ImVec2(X(k), top), ImVec2(X(k), bottom), ui::ToU32(t.caution));
        const ImU32 c = k == 0 ? ui::ToU32(t.text_faint) : outside ? strong : weak;
        dl->AddLine(ImVec2(X(k), zero), ImVec2(X(k), Y(values[k])), c, std::max(1.5f, std::min(4.0f, step * 0.5f)));
        dl->AddCircleFilled(ImVec2(X(k), Y(values[k])), std::max(2.0f, std::min(3.5f, step * 0.3f)), c);
    }
    dl->AddText(ImVec2(at.x + 4.0f, at.y + 2.0f), ui::ToU32(t.text_faint), (std::string(y_label) + "  1").c_str());
    XAxis(at, width, "0", "lag", std::to_string(n - 1));
}

void PositionBars(const std::vector<double>& values, const std::vector<std::string>& names, float width, float height) {
    const ui::Tokens& t = ui::CurrentTokens();
    const size_t n = values.size();
    if (n == 0) return;
    double lo = 0.0, hi = 0.0;
    Extent(values, lo, hi);
    if (!(hi > lo)) hi = lo + 1.0;
    const double pad = (hi - lo) * 0.1;
    lo -= pad;
    hi += pad;
    const float line_h = ImGui::GetTextLineHeight();
    const ImVec2 at = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(width, height + line_h + 4.0f));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(at, ImVec2(at.x + width, at.y + height), ui::ToU32(t.plot_bg), 4.0f);
    const float bw = (width - 8.0f) / static_cast<float>(n);
    const auto Y = [&](double y) { return at.y + height - 2.0f - static_cast<float>((y - lo) / (hi - lo)) * (height - line_h - 6.0f); };
    const float zero = Y(0.0);
    dl->AddLine(ImVec2(at.x + 4.0f, zero), ImVec2(at.x + width - 4.0f, zero), ui::ToU32(t.text_faint), 1.0f);
    ImVec4 up = ui::SeriesColour(t, 0), down = ui::SeriesColour(t, 1);
    up.w = down.w = 0.8f;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(values[i])) continue;
        const float x = at.x + 4.0f + static_cast<float>(i) * bw;
        const float y = Y(values[i]);
        dl->AddRectFilled(ImVec2(x + 2.0f, std::min(y, zero)), ImVec2(x + std::max(3.0f, bw - 2.0f), std::max(y, zero)), ui::ToU32(values[i] >= 0 ? up : down), 2.0f);
        if (i < names.size()) {
            const float w = ImGui::CalcTextSize(names[i].c_str()).x;
            if (w <= bw) dl->AddText(ImVec2(x + (bw - w) * 0.5f, at.y + height + 2.0f), ui::ToU32(t.text_faint), names[i].c_str());
        }
    }
    dl->AddText(ImVec2(at.x + 4.0f, at.y + 2.0f), ui::ToU32(t.text_faint), sv::Num(hi - pad).c_str());
}

}  // namespace cyxwiz::explore_plot
