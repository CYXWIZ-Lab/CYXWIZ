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

}  // namespace

void Scatter(const std::vector<double>& xs, const std::vector<double>& ys, float width, float height,
             const std::vector<std::pair<double, double>>* curve, int guide, const char* x_label, const char* y_label) {
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
    for (size_t i = 0; i < n; i += stride) dl->AddCircleFilled(P(xs[i], ys[i]), 2.4f, ui::ToU32(dot));
    if (curve && curve->size() > 1) {
        std::vector<ImVec2> line;
        for (const auto& [cx, cy] : *curve) line.push_back(P(cx, cy));
        dl->AddPolyline(line.data(), static_cast<int>(line.size()), ui::ToU32(t.caution), ImDrawFlags_None, 2.0f);
    }
    const std::string lx = sv::Num(x0 + px), hx = sv::Num(x1 - px), ly = sv::Num(y0 + py), hy = sv::Num(y1 - py);
    dl->AddText(ImVec2(at.x + 4.0f, at.y + 2.0f), ui::ToU32(t.text_faint), (std::string(y_label) + "  " + hy).c_str());
    dl->AddText(ImVec2(at.x + 4.0f, at.y + height - ImGui::GetTextLineHeight() - 2.0f), ui::ToU32(t.text_faint), ly.c_str());
    ImGui::TextColored(t.text_faint, "%s", lx.c_str());
    ImGui::SameLine();
    const std::string mid = x_label;
    const float x_start = ImGui::GetCursorPosX();
    ImGui::SetCursorPosX(std::max(x_start, x_start + (width - ImGui::CalcTextSize(mid.c_str()).x) * 0.5f - ImGui::CalcTextSize(lx.c_str()).x));
    ImGui::TextColored(t.text_faint, "%s", mid.c_str());
    ImGui::SameLine();
    ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(), at.x - ImGui::GetWindowPos().x + ImGui::GetScrollX() + width - ImGui::CalcTextSize(hx.c_str()).x));
    ImGui::TextColored(t.text_faint, "%s", hx.c_str());
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

}  // namespace cyxwiz::explore_plot
