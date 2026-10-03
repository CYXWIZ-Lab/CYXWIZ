// 3D plots in the shared plot view (TOFIX134 P4, approved board 16): Scatter 3D,
// Line 3D and Surface drawn with ImPlot3D's box, axes and camera.
//
// ImGui here uses 16-bit indices and ImPlot3D's own items stop at 65,535
// vertices per plot, so the view draws the data itself:
//   - Surface fill, wireframe and floor contours go into ImPlot3D's depth-
//     sorted 3D draw list (one vertex per grid point, a vertex budget, wire
//     lines every Nth row and column);
//   - Scatter 3D points and Line 3D paths are projected, sorted back to front
//     and drawn into the window's draw list, which splits large lists itself.
// The fill is coloured on the theme scale and lit by a light fixed to the
// data (like matplotlib's shaded surfaces); ImPlot3D colours by height only.

#include "plot_view.h"

#include "plot_style.h"
#include "../ui_tokens.h"

#include <implot.h>
#include <implot3d.h>
#include <implot3d_internal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>

namespace cyxwiz::plot {

namespace {

constexpr unsigned kVertexBudget = 60000;  // of ImPlot3D's 65,535 per plot
constexpr int kWireLines = 48;              // wire lines per direction at most
constexpr double kPi = 3.14159265358979323846;

ImU32 Shade(ImVec4 c, float k, float alpha = 1.0f) {
    c.x = std::min(1.0f, c.x * k);
    c.y = std::min(1.0f, c.y * k);
    c.z = std::min(1.0f, c.z * k);
    c.w = alpha;
    return ImGui::ColorConvertFloat4ToU32(c);
}

// A value on the surface's scale (grid range) or the points' colour scale.
ImVec4 SurfaceColour(const Prepared& p, double v) {
    const double span = p.grid_hi - p.grid_lo;
    const float t = static_cast<float>(span > 0 ? std::clamp((v - p.grid_lo) / span, 0.0, 1.0) : 0.0);
    return ImPlot::SampleColormap(t, p.grid_diverging ? DivergingColormap() : SequentialColormap());
}

ImVec4 PointColour(const Prepared& p, double v) {
    if (!std::isfinite(v)) return ui::CurrentTokens().text_faint;
    const double span = p.colour_max - p.colour_min;
    const float t = static_cast<float>(span > 0 ? std::clamp((v - p.colour_min) / span, 0.0, 1.0) : 0.0);
    return ImPlot::SampleColormap(t, p.colour_diverging ? DivergingColormap() : SequentialColormap());
}

// Depth as the 3D list sorts it (larger: nearer the viewer), from NDC.
float Depth(const ImPlot3DPlot& plot, const ImPlot3DPoint& ndc) {
    return (plot.Rotation * ndc).z;
}

// Elevation and azimuth (degrees) of a rotation made by ImPlot3DQuat::FromElAz.
void ElAzOf(const ImPlot3DQuat& q, double& el, double& az) {
    const ImPlot3DPoint up = q * ImPlot3DPoint(0, 0, 1);
    const double e = std::atan2(up.z, up.y);
    const ImPlot3DPoint ex = q * ImPlot3DPoint(1, 0, 0);
    const double s = ex.y * std::sin(e) - ex.z * std::cos(e);
    el = e * 180.0 / kPi;
    az = std::atan2(s, ex.x) * 180.0 / kPi;
}

// One line segment (screen pixels) as a quad in the 3D list.
void AddSegment(ImDrawList3D& dl, ImVec2 a, ImVec2 b, float half, ImU32 col, ImVec2 uv, float z) {
    ImVec2 d(b.x - a.x, b.y - a.y);
    const float len = std::sqrt(d.x * d.x + d.y * d.y);
    if (len < 1e-3f) return;
    d = ImVec2(-d.y / len * half, d.x / len * half);
    ImDrawVert* v = dl._VtxWritePtr;
    v[0].pos = ImVec2(a.x + d.x, a.y + d.y);
    v[1].pos = ImVec2(b.x + d.x, b.y + d.y);
    v[2].pos = ImVec2(b.x - d.x, b.y - d.y);
    v[3].pos = ImVec2(a.x - d.x, a.y - d.y);
    for (int i = 0; i < 4; ++i) {
        v[i].uv = uv;
        v[i].col = col;
    }
    dl._VtxWritePtr += 4;
    const unsigned base = dl._VtxCurrentIdx;
    ImDrawIdx* idx = dl._IdxWritePtr;
    idx[0] = static_cast<ImDrawIdx>(base);
    idx[1] = static_cast<ImDrawIdx>(base + 1);
    idx[2] = static_cast<ImDrawIdx>(base + 2);
    idx[3] = static_cast<ImDrawIdx>(base);
    idx[4] = static_cast<ImDrawIdx>(base + 2);
    idx[5] = static_cast<ImDrawIdx>(base + 3);
    dl._IdxWritePtr += 6;
    dl._ZWritePtr[0] = z;
    dl._ZWritePtr[1] = z;
    dl._ZWritePtr += 2;
    dl._VtxCurrentIdx += 4;
}

struct GridGeometry {
    const Prepared& p;
    double dx, dy;
    explicit GridGeometry(const Prepared& prepared)
        : p(prepared), dx((prepared.x_max - prepared.x_min) / prepared.grid_cols), dy((prepared.y_max - prepared.y_min) / prepared.grid_rows) {}
    double X(int c) const { return p.x_min + (c + 0.5) * dx; }
    double Y(int r) const { return p.y_max - (r + 0.5) * dy; }
    double Z(int r, int c) const { return p.grid[static_cast<size_t>(r) * static_cast<size_t>(p.grid_cols) + static_cast<size_t>(c)]; }
};

}  // namespace

void PlotView::Draw3D(ImVec2 size) {
    const Prepared& p = data_;
    const Kind kind = p.spec.kind;
    const ui::Tokens& t = ui::CurrentTokens();
    const bool surface = kind == Kind::Surface;
    const bool bar = surface || p.colour_scale;
    const float bar_w = 60.0f + ImGui::GetTextLineHeight() + 8.0f;
    ImVec2 plot_size = size;
    if (bar) plot_size.x = std::max(120.0f, size.x - bar_w - 16.0f);

    ImPlot3DStyle& st = ImPlot3D::GetStyle();
    st.Colors[ImPlot3DCol_FrameBg] = ImVec4(0, 0, 0, 0);
    st.Colors[ImPlot3DCol_PlotBg] = t.plot_bg;
    st.Colors[ImPlot3DCol_PlotBorder] = ImVec4(0, 0, 0, 0);
    st.Colors[ImPlot3DCol_AxisText] = t.text_dim;
    st.Colors[ImPlot3DCol_AxisGrid] = t.plot_grid;
    st.Colors[ImPlot3DCol_AxisTick] = t.plot_grid;
    st.Colors[ImPlot3DCol_LegendBg] = ui::WithAlpha(t.bg_panel, 0.92f);
    st.Colors[ImPlot3DCol_LegendBorder] = ImVec4(0, 0, 0, 0);
    st.Colors[ImPlot3DCol_LegendText] = t.text;
    st.Colors[ImPlot3DCol_TitleText] = t.text_bright;
    // Room around the box for the axis names (drawn outside it, clipped by the frame).
    st.PlotPadding = ImVec2(36, 36);

    ImPlot3DFlags flags = ImPlot3DFlags_NoTitle | ImPlot3DFlags_NoMouseText | ImPlot3DFlags_NoMenus;
    if (!legend_ || surface || p.series.size() < 2) flags |= ImPlot3DFlags_NoLegend;
    if (!ImPlot3D::BeginPlot("##plot3d", plot_size, flags)) return;
    ImPlot3DPlot& plot = *ImPlot3D::GetCurrentPlot();

    const std::string xl = p.spec.x_label.empty() ? p.spec.x_column : p.spec.x_label;
    const std::string yl = p.spec.y_label.empty() ? (p.spec.y_columns.empty() ? std::string() : p.spec.y_columns.front()) : p.spec.y_label;
    const std::string zl = p.z_label;
    ImPlot3D::SetupAxes(xl.c_str(), yl.c_str(), zl.c_str());
    // Surfaces span the grid's cell centres; points their own range.
    double x0 = p.x_min, x1 = p.x_max, y0 = p.y_min, y1 = p.y_max;
    if (surface && p.grid_cols > 0 && p.grid_rows > 0) {
        const GridGeometry g(p);
        x0 = g.X(0);
        x1 = g.X(p.grid_cols - 1);
        y0 = g.Y(p.grid_rows - 1);
        y1 = g.Y(0);
    }
    const ImPlot3DCond cond = fit_ ? ImPlot3DCond_Always : ImPlot3DCond_Once;
    ImPlot3D::SetupAxesLimits(x0, x1 > x0 ? x1 : x0 + 1, y0, y1 > y0 ? y1 : y0 + 1, p.z_min, p.z_max > p.z_min ? p.z_max : p.z_min + 1, cond);
    fit_ = false;
    // The view: a preset asked for, else the saved one (once).
    if (view_apply_) {
        if (std::isfinite(view_el_)) ImPlot3D::SetupBoxRotation(static_cast<float>(view_el_), static_cast<float>(view_az_), true, ImPlot3DCond_Always);
        else ImPlot3D::SetupBoxRotation(plot.InitialRotation, true, ImPlot3DCond_Always);
        view_apply_ = false;
    } else if (std::isfinite(p.spec.view_elevation) && std::isfinite(p.spec.view_azimuth)) {
        ImPlot3D::SetupBoxRotation(static_cast<float>(p.spec.view_elevation), static_cast<float>(p.spec.view_azimuth), false, ImPlot3DCond_Once);
    }
    ImPlot3D::SetupLock();

    const ImVec2 uv = ImGui::GetDrawListSharedData()->TexUvWhitePixel;
    const ImPlot3DPoint lo = plot.RangeMin(), hi = plot.RangeMax();
    const auto inside = [&](double x, double y, double z) {
        return x >= lo.x && x <= hi.x && y >= lo.y && y <= hi.y && z >= lo.z && z <= hi.z;
    };
    hover3d_.clear();
    ImDrawList3D& dl = plot.DrawList;

    if (surface && p.grid_rows >= 2 && p.grid_cols >= 2) {
        const GridGeometry g(p);
        const int R = p.grid_rows, C = p.grid_cols;
        // Rows and columns drawn: every step-th, within the vertex budget.
        int step = 1;
        while (static_cast<unsigned>((R + step - 1) / step) * static_cast<unsigned>((C + step - 1) / step) > kVertexBudget / 2) ++step;
        std::vector<int> rows, cols;
        for (int r = 0; r < R; r += step) rows.push_back(r);
        for (int c = 0; c < C; c += step) cols.push_back(c);
        if (rows.back() != R - 1) rows.push_back(R - 1);
        if (cols.back() != C - 1) cols.push_back(C - 1);
        const int nr = static_cast<int>(rows.size()), nc = static_cast<int>(cols.size());
        // Positions in NDC (shading and depth) and on screen.
        std::vector<ImPlot3DPoint> ndc(static_cast<size_t>(nr) * static_cast<size_t>(nc));
        std::vector<ImVec2> pix(ndc.size());
        std::vector<char> ok(ndc.size(), 0);
        for (int i = 0; i < nr; ++i)
            for (int j = 0; j < nc; ++j) {
                const double z = g.Z(rows[static_cast<size_t>(i)], cols[static_cast<size_t>(j)]);
                const size_t k = static_cast<size_t>(i) * static_cast<size_t>(nc) + static_cast<size_t>(j);
                if (!std::isfinite(z)) continue;
                const ImPlot3DPoint pt(static_cast<float>(g.X(cols[static_cast<size_t>(j)])), static_cast<float>(g.Y(rows[static_cast<size_t>(i)])),
                                       static_cast<float>(z));
                if (!inside(pt.x, pt.y, pt.z)) continue;
                ndc[k] = ImPlot3D::PlotToNDC(pt);
                pix[k] = ImPlot3D::PlotToPixels(pt);
                ok[k] = 1;
                hover3d_.push_back({pix[k], pt.x, pt.y, pt.z, NAN});
            }
        const auto at = [&](int i, int j) { return static_cast<size_t>(i) * static_cast<size_t>(nc) + static_cast<size_t>(j); };
        const bool fill = p.spec.surface_draw != PlotSpec::SurfaceDraw::Lines;
        const bool lines = p.spec.surface_draw != PlotSpec::SurfaceDraw::Fill;
        if (fill) {
            // A light fixed to the data, from the upper left front.
            ImPlot3DPoint light(-0.45f, -0.55f, 0.7f);
            light.Normalize();
            size_t quads = 0, verts = 0;
            for (size_t k = 0; k < ok.size(); ++k) verts += ok[k] ? 1 : 0;
            for (int i = 0; i + 1 < nr; ++i)
                for (int j = 0; j + 1 < nc; ++j)
                    if (ok[at(i, j)] && ok[at(i + 1, j)] && ok[at(i, j + 1)] && ok[at(i + 1, j + 1)]) ++quads;
            if (verts > 0 && dl._VtxCurrentIdx + verts < ImDrawList3D::MaxIdx()) {
                dl.PrimReserve(static_cast<int>(quads * 6), static_cast<int>(verts));
                std::vector<unsigned> index(ok.size(), 0);
                for (int i = 0; i < nr; ++i)
                    for (int j = 0; j < nc; ++j) {
                        const size_t k = at(i, j);
                        if (!ok[k]) continue;
                        // Normal from the neighbours (one side at the edges and open cells).
                        const auto n_at = [&](int ii, int jj) { return ii >= 0 && jj >= 0 && ii < nr && jj < nc && ok[at(ii, jj)]; };
                        const ImPlot3DPoint& c0 = ndc[k];
                        const ImPlot3DPoint ex = (n_at(i, j + 1) ? ndc[at(i, j + 1)] : c0) - (n_at(i, j - 1) ? ndc[at(i, j - 1)] : c0);
                        const ImPlot3DPoint ey = (n_at(i - 1, j) ? ndc[at(i - 1, j)] : c0) - (n_at(i + 1, j) ? ndc[at(i + 1, j)] : c0);
                        ImPlot3DPoint n = ex.Cross(ey);
                        float shade = 1.0f;
                        if (p.spec.shade && n.LengthSquared() > 0) {
                            n.Normalize();
                            if (n.z < 0) n = -n;
                            shade = 0.62f + 0.45f * std::max(0.0f, n.Dot(light));
                        }
                        const double z = g.Z(rows[static_cast<size_t>(i)], cols[static_cast<size_t>(j)]);
                        ImDrawVert& v = *dl._VtxWritePtr++;
                        v.pos = pix[k];
                        v.uv = uv;
                        v.col = Shade(SurfaceColour(p, z), shade);
                        index[k] = dl._VtxCurrentIdx++;
                    }
                for (int i = 0; i + 1 < nr; ++i)
                    for (int j = 0; j + 1 < nc; ++j) {
                        const size_t a = at(i, j), b = at(i, j + 1), c = at(i + 1, j + 1), d = at(i + 1, j);
                        if (!(ok[a] && ok[b] && ok[c] && ok[d])) continue;
                        ImDrawIdx* idx = dl._IdxWritePtr;
                        idx[0] = static_cast<ImDrawIdx>(index[a]);
                        idx[1] = static_cast<ImDrawIdx>(index[b]);
                        idx[2] = static_cast<ImDrawIdx>(index[c]);
                        idx[3] = static_cast<ImDrawIdx>(index[a]);
                        idx[4] = static_cast<ImDrawIdx>(index[c]);
                        idx[5] = static_cast<ImDrawIdx>(index[d]);
                        dl._IdxWritePtr += 6;
                        dl._ZWritePtr[0] = Depth(plot, (ndc[a] + ndc[b] + ndc[c]) / 3.0f);
                        dl._ZWritePtr[1] = Depth(plot, (ndc[a] + ndc[c] + ndc[d]) / 3.0f);
                        dl._ZWritePtr += 2;
                    }
            }
        }
        if (lines) {
            // Wire lines every wstep-th row and column, a little above the fill.
            const int wstep = std::max(1, std::max(nr, nc) / kWireLines);
            // On a fill: dark lines (they read on light and dark parts); alone: the first series colour.
            const ImU32 col = fill ? Shade(t.plot_bg, 1.0f, 0.6f) : ImGui::ColorConvertFloat4ToU32(t.series[0]);
            std::vector<std::pair<size_t, size_t>> segs;
            for (int i = 0; i < nr; i += wstep)
                for (int j = 0; j + 1 < nc; ++j)
                    if (ok[at(i, j)] && ok[at(i, j + 1)]) segs.push_back({at(i, j), at(i, j + 1)});
            for (int j = 0; j < nc; j += wstep)
                for (int i = 0; i + 1 < nr; ++i)
                    if (ok[at(i, j)] && ok[at(i + 1, j)]) segs.push_back({at(i, j), at(i + 1, j)});
            const size_t room = (ImDrawList3D::MaxIdx() - dl._VtxCurrentIdx) / 4;
            if (segs.size() > room) segs.resize(room);
            dl.PrimReserve(static_cast<int>(segs.size() * 6), static_cast<int>(segs.size() * 4));
            for (const auto& [a, b] : segs)
                AddSegment(dl, pix[a], pix[b], 0.5f, col, uv, Depth(plot, (ndc[a] + ndc[b]) * 0.5f) + 1e-3f);
            // Segments too short to draw reserved space they did not use.
            const int unused_v = static_cast<int>(dl.VtxBuffer.Data + dl.VtxBuffer.Size - dl._VtxWritePtr);
            if (unused_v > 0) dl.PrimUnreserve(unused_v / 4 * 6, unused_v);
        }
        // Contour lines on the floor (the lowest Z of the box).
        if (p.spec.floor_contours && !p.contour_segments.empty()) {
            const float zf = lo.z;
            size_t count = 0;
            for (const auto& s : p.contour_segments) count += s.size() / 4;
            const size_t room = (ImDrawList3D::MaxIdx() - dl._VtxCurrentIdx) / 4;
            if (count <= room) {
                dl.PrimReserve(static_cast<int>(count * 6), static_cast<int>(count * 4));
                for (size_t l = 0; l < p.contour_segments.size(); ++l) {
                    const ImU32 col = Shade(SurfaceColour(p, p.contour_levels[l]), 1.0f, 0.9f);
                    const auto& s = p.contour_segments[l];
                    for (size_t i = 0; i + 3 < s.size(); i += 4) {
                        const ImPlot3DPoint a(static_cast<float>(s[i]), static_cast<float>(s[i + 1]), zf);
                        const ImPlot3DPoint b(static_cast<float>(s[i + 2]), static_cast<float>(s[i + 3]), zf);
                        AddSegment(dl, ImPlot3D::PlotToPixels(a), ImPlot3D::PlotToPixels(b), 0.6f, col, uv,
                                   Depth(plot, (ImPlot3D::PlotToNDC(a) + ImPlot3D::PlotToNDC(b)) * 0.5f));
                    }
                }
                const int unused_v = static_cast<int>(dl.VtxBuffer.Data + dl.VtxBuffer.Size - dl._VtxWritePtr);
                if (unused_v > 0) dl.PrimUnreserve(unused_v / 4 * 6, unused_v);
            }
        }
    } else {
        // Points and paths: projected here, sorted back to front, drawn in pixels.
        ImDrawList* draw = ImPlot3D::GetPlotDrawList();
        draw->PushClipRect(plot.PlotRect.Min, plot.PlotRect.Max, true);
        const bool line = kind == Kind::Line3D;
        struct Pt {
            ImVec2 pos;
            float depth;
            ImU32 col;
            float radius;
        };
        std::vector<Pt> pts;
        size_t total = 0;
        for (const auto& s : p.series) total += s.x.size();
        const float radius = total > 20000 ? 1.5f : total > 2000 ? 2.2f : 3.0f;
        for (size_t si = 0; si < p.series.size(); ++si) {
            const Series& s = p.series[si];
            const ImU32 series_col = ImGui::ColorConvertFloat4ToU32(ColourOf(si));
            std::vector<ImVec2> path;
            for (size_t i = 0; i < s.x.size() && i < s.y.size() && i < s.z3.size(); ++i) {
                const ImPlot3DPoint pt(static_cast<float>(s.x[i]), static_cast<float>(s.y[i]), static_cast<float>(s.z3[i]));
                if (!inside(pt.x, pt.y, pt.z)) {
                    if (line && path.size() > 1) draw->AddPolyline(path.data(), static_cast<int>(path.size()), series_col, ImDrawFlags_None, 1.8f);
                    path.clear();
                    continue;
                }
                const ImVec2 px = ImPlot3D::PlotToPixels(pt);
                const double cv = i < s.c.size() ? s.c[i] : NAN;
                hover3d_.push_back({px, pt.x, pt.y, pt.z, cv});
                if (line) {
                    path.push_back(px);
                    continue;
                }
                float r = radius;
                if (i < s.z.size() && std::isfinite(s.z[i]) && p.size_max > p.size_min)
                    r = 2.0f + 8.0f * static_cast<float>((s.z[i] - p.size_min) / (p.size_max - p.size_min));
                const ImU32 col = p.colour_scale ? Shade(PointColour(p, cv), 1.0f, 0.85f) : (series_col & 0x00FFFFFFu) | 0xD9000000u;
                pts.push_back({px, Depth(plot, ImPlot3D::PlotToNDC(pt)), col, r});
            }
            if (line && path.size() > 1) draw->AddPolyline(path.data(), static_cast<int>(path.size()), series_col, ImDrawFlags_None, 1.8f);
        }
        std::sort(pts.begin(), pts.end(), [](const Pt& a, const Pt& b) { return a.depth < b.depth; });
        for (const auto& q : pts) {
            if (q.radius <= 1.6f) draw->AddRectFilled(ImVec2(q.pos.x - 1.2f, q.pos.y - 1.2f), ImVec2(q.pos.x + 1.2f, q.pos.y + 1.2f), q.col);
            else draw->AddCircleFilled(q.pos, q.radius, q.col, 10);
        }
        draw->PopClipRect();
        // Legend entries for groups: ImPlot3D lists items it drew, so each
        // series adds one with a single point outside the box (nothing shows).
        if (p.series.size() > 1) {
            const double nan = NAN;
            for (size_t si = 0; si < p.series.size(); ++si) {
                const ImVec4 c = ColourOf(si);
                if (line) {
                    ImPlot3D::SetNextLineStyle(c, 2.0f);
                    ImPlot3D::PlotLine(p.series[si].label.c_str(), &nan, &nan, &nan, 1);
                } else {
                    ImPlot3D::SetNextMarkerStyle(ImPlot3DMarker_Circle, 4.0f, c, 0.0f, c);
                    ImPlot3D::PlotScatter(p.series[si].label.c_str(), &nan, &nan, &nan, 1);
                }
            }
        }
    }

    // Hover: the nearest drawn point or surface vertex within 10 pixels.
    const ImVec2 mouse = ImGui::GetMousePos();
    if (ImGui::IsMouseHoveringRect(plot.PlotRect.Min, plot.PlotRect.Max) && !ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        int best = -1;
        float best_d = 100.0f;
        for (size_t i = 0; i < hover3d_.size(); ++i) {
            const float dx = hover3d_[i].pos.x - mouse.x, dy = hover3d_[i].pos.y - mouse.y;
            const float d = dx * dx + dy * dy;
            if (d < best_d) {
                best_d = d;
                best = static_cast<int>(i);
            }
        }
        if (best >= 0) {
            const Hover3D& h = hover3d_[static_cast<size_t>(best)];
            ImGui::GetForegroundDrawList()->AddCircle(h.pos, 5.0f, ImGui::ColorConvertFloat4ToU32(t.text_bright), 12, 1.5f);
            ImGui::BeginTooltip();
            ImGui::TextColored(t.text_dim, "%s", xl.c_str());
            ImGui::SameLine();
            ImGui::Text("%g", h.x);
            ImGui::TextColored(t.text_dim, "%s", yl.c_str());
            ImGui::SameLine();
            ImGui::Text("%g", h.y);
            ImGui::TextColored(t.text_dim, "%s", zl.c_str());
            ImGui::SameLine();
            ImGui::Text("%g", h.z);
            if (std::isfinite(h.c)) {
                ImGui::TextColored(t.text_dim, "%s", p.colour_label.c_str());
                ImGui::SameLine();
                ImGui::Text("%g", h.c);
            }
            ImGui::EndTooltip();
        }
    }

    // The view the user turned to, for saving with the plot.
    if (!plot.Held && on_view_changed) {
        double el = 0, az = 0;
        ElAzOf(plot.Rotation, el, az);
        const bool moved = !std::isfinite(saved_el_) || std::fabs(el - saved_el_) > 0.5 || std::fabs(az - saved_az_) > 0.5;
        if (moved && plot.AnimationTime <= 0.0f && plot.Initialized) {
            const bool first = !std::isfinite(saved_el_);
            saved_el_ = el;
            saved_az_ = az;
            if (!first) on_view_changed(el, az);
        }
    }
    ImPlot3D::EndPlot();
    frame_min_ = ImGui::GetItemRectMin();
    frame_max_ = ImGui::GetItemRectMax();

    if (bar) {
        ImGui::SameLine();
        if (surface)
            ImPlot::ColormapScale((zl + "##scale3d").c_str(), p.grid_lo, p.grid_hi > p.grid_lo ? p.grid_hi : p.grid_lo + 1.0, ImVec2(bar_w, plot_size.y),
                                  "%g", 0, p.grid_diverging ? DivergingColormap() : SequentialColormap());
        else
            ImPlot::ColormapScale(p.colour_label.c_str(), p.colour_min, p.colour_max, ImVec2(bar_w, plot_size.y), "%g", 0,
                                  p.colour_diverging ? DivergingColormap() : SequentialColormap());
    }
}

void PlotView::SetView(double elevation, double azimuth) {
    view_el_ = elevation;
    view_az_ = azimuth;
    view_apply_ = true;
}

}  // namespace cyxwiz::plot
