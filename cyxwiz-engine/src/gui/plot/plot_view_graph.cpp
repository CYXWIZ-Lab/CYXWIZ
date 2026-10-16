// Network and Tree in the shared plot view (TOFIX134 P4 group 2, approved
// board 17). The layout comes from the prepare step (0..1, y from the top);
// the view draws it in plot coordinates so ImPlot pans and zooms, lets the
// user move a node (a press on a node makes the view the active item, so
// neither the plot nor the window moves), folds tree branches (laid out again without them), and
// shows a node's links on hover.

#include "plot_view.h"

#include "../../core/plot/plot_graph.h"
#include "../ui_tokens.h"

#include <imgui_internal.h>  // ImRect
#include <implot.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>

namespace cyxwiz::plot {

namespace {

std::string Num(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%g", v);
    return buf;
}

// The text cut to fit `width` pixels, ending in "..." when cut. A name such
// as "leaf 1.3: single" or "2: tempo <= 0.5" that does not fit first drops
// its id ("single"): what the node says matters more than its number.
std::string Shorten(const std::string& full, float width) {
    if (ImGui::CalcTextSize(full.c_str()).x <= width) return full;
    const size_t colon = full.find(": ");
    const std::string text = colon != std::string::npos && colon + 2 < full.size() ? full.substr(colon + 2) : full;
    if (ImGui::CalcTextSize(text.c_str()).x <= width) return text;
    size_t lo = 0, hi = text.size();
    while (lo < hi) {
        const size_t mid = (lo + hi + 1) / 2;
        if (ImGui::CalcTextSize((text.substr(0, mid) + "...").c_str()).x <= width) lo = mid;
        else hi = mid - 1;
    }
    while (lo > 0 && (static_cast<unsigned char>(text[lo]) & 0xC0) == 0x80) --lo;  // not inside a UTF-8 character
    return text.substr(0, lo) + "...";
}

void Row(const ImVec4& dim, const char* name, const std::string& value) {
    ImGui::TextColored(dim, "%s", name);
    ImGui::SameLine();
    ImGui::TextUnformatted(value.c_str());
}

}  // namespace

void PlotView::DrawGraph(ImVec2 size) {
    const Prepared& p = data_;
    const ui::Tokens& t = ui::CurrentTokens();
    const bool tree = p.spec.kind == Kind::Tree;
    const size_t n = p.graph_nodes.size();
    // Positions (x right, y up) after the user's moves and tree folds.
    if (graph_pos_.size() != n) {
        graph_pos_.assign(n, ImVec2(0, 0));
        graph_moved_.assign(n, 0);
        graph_fold_.assign(n, 0);
        graph_layout_dirty_ = true;
    }
    std::vector<char> visible(n, 1);
    if (tree) {
        // Hidden: anything under a folded node.
        for (size_t i = 0; i < n; ++i) {
            int a = p.graph_nodes[i].parent, steps = 0;
            while (a >= 0 && steps++ < static_cast<int>(n)) {
                if (graph_fold_[static_cast<size_t>(a)]) {
                    visible[i] = 0;
                    break;
                }
                a = p.graph_nodes[static_cast<size_t>(a)].parent;
            }
        }
    }
    if (graph_layout_dirty_) {
        graph_layout_dirty_ = false;
        if (tree) {
            // Lay the visible part out again (folded branches leave no gaps).
            std::vector<int> index(n, -1), parent;
            for (size_t i = 0; i < n; ++i)
                if (visible[i]) {
                    index[i] = static_cast<int>(parent.size());
                    parent.push_back(-1);
                }
            for (size_t i = 0; i < n; ++i)
                if (visible[i] && p.graph_nodes[i].parent >= 0) parent[static_cast<size_t>(index[i])] = index[static_cast<size_t>(p.graph_nodes[i].parent)];
            const auto pos = TreeLayout(parent);
            for (size_t i = 0; i < n; ++i)
                if (visible[i] && !graph_moved_[i]) {
                    const Point2& q = pos[static_cast<size_t>(index[i])];
                    graph_pos_[i] = p.spec.tree_left_right ? ImVec2(static_cast<float>(q.y), static_cast<float>(1.0 - q.x))
                                                           : ImVec2(static_cast<float>(q.x), static_cast<float>(1.0 - q.y));
                }
        } else {
            for (size_t i = 0; i < n; ++i)
                if (!graph_moved_[i]) graph_pos_[i] = ImVec2(static_cast<float>(p.graph_nodes[i].x), static_cast<float>(1.0 - p.graph_nodes[i].y));
        }
    }

    // Moving a node: a press on the node under the mouse (last frame's place)
    // makes this view the active item before the plot is drawn, so neither
    // the plot (pan) nor the window (move) takes the drag. Held and moved more
    // than 3 px: the node follows; let go without moving: a click (folds).
    const ImVec2 mouse = ImGui::GetMousePos();
    const ImGuiID grab_id = ImGui::GetID("##graph_node");
    int clicked = -1;
    if (graph_drag_ < 0 && hot_graph_ >= 0 && static_cast<size_t>(hot_graph_) < n && ImGui::IsMouseClicked(ImGuiMouseButton_Left) &&
        ImGui::IsMouseHoveringRect(hot_min_, hot_max_)) {  // hot: the plot was hovered last frame
        graph_drag_ = hot_graph_;
        graph_press_ = mouse;
        graph_dragging_ = false;
        ImGui::SetActiveID(grab_id, ImGui::GetCurrentWindow());
    }
    if (graph_drag_ >= 0) {
        if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            if (!graph_dragging_) clicked = graph_drag_;
            graph_drag_ = -1;
            if (ImGui::GetActiveID() == grab_id) ImGui::ClearActiveID();
        } else {
            ImGui::KeepAliveID(grab_id);
            if ((mouse.x - graph_press_.x) * (mouse.x - graph_press_.x) + (mouse.y - graph_press_.y) * (mouse.y - graph_press_.y) > 9.0f)
                graph_dragging_ = true;
        }
    }
    ImPlotFlags flags = ImPlotFlags_NoTitle | ImPlotFlags_NoMenus | ImPlotFlags_NoMouseText | ImPlotFlags_NoLegend;
    if (!tree) flags |= ImPlotFlags_Equal;
    if (graph_drag_ >= 0) flags |= ImPlotFlags_NoInputs;  // a held node: no pan or zoom
    const ImVec2 plot_px(size.x > 0 ? size.x : ImGui::GetContentRegionAvail().x, size.y > 0 ? size.y : ImGui::GetContentRegionAvail().y);
    if (!ImPlot::BeginPlot("##graph", size, flags)) return;
    ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_NoDecorations);
    // Fit leaves room for half a tree box beside the outer nodes, in pixels:
    // a margin m of the 0..1 layout is m / (1 + 2m) of the plot.
    double margin_x = 0.06, margin_y = 0.08;
    if (tree) {
        const auto margin = [](double half_px, float plot) {
            return plot > 4.0f * half_px ? std::clamp(half_px / (plot - 2.0 * half_px), 0.04, 0.4) : 0.4;
        };
        const double half_h = ImGui::GetTextLineHeight() + 8.0;
        margin_x = margin(p.spec.tree_left_right ? 114.0 : 80.0, plot_px.x);  // left-right: half the widest box beside a level
        margin_y = margin(half_h, plot_px.y);
    }
    ImPlot::SetupAxesLimits(-margin_x, 1.0 + margin_x, -margin_y, 1.0 + margin_y, fit_ ? ImPlotCond_Always : ImPlotCond_Once);
    fit_ = false;
    ImDrawList* dl = ImPlot::GetPlotDrawList();
    const auto px = [&](size_t i) { return ImPlot::PlotToPixels(graph_pos_[i].x, graph_pos_[i].y); };

    // Sizes: links, weight or the same (the tree's boxes size to their names).
    double most = 0;
    for (const auto& nd : p.graph_nodes)
        most = std::max(most, p.spec.node_size == PlotSpec::NodeSize::Weight && std::isfinite(nd.weight) ? nd.weight : static_cast<double>(nd.links));
    const auto radius = [&](size_t i) {
        const auto& nd = p.graph_nodes[i];
        if (p.spec.node_size == PlotSpec::NodeSize::Same || most <= 0) return 5.0f;
        const double v = p.spec.node_size == PlotSpec::NodeSize::Weight && std::isfinite(nd.weight) ? nd.weight : static_cast<double>(nd.links);
        return 3.0f + 10.0f * static_cast<float>(std::sqrt(std::max(0.0, v) / most));
    };
    const float line = ImGui::GetTextLineHeight();
    // A tree box is at most as wide as the room to its neighbours on the same
    // level (top-down) or the room between levels (left-right), so boxes never
    // cover each other; a name that does not fit ends in "...".
    constexpr float kMaxBox = 220.0f;  // a split such as "0: album_total_tracks <= 1.5" fits
    std::vector<float> room(n, kMaxBox);
    if (tree) {
        std::vector<std::pair<ImVec2, size_t>> at;
        for (size_t i = 0; i < n; ++i)
            if (visible[i]) at.push_back({px(i), i});
        if (p.spec.tree_left_right) {
            std::vector<float> xs;
            for (const auto& a : at) xs.push_back(a.first.x);
            std::sort(xs.begin(), xs.end());
            float gap = kMaxBox;
            for (size_t k = 1; k < xs.size(); ++k)
                if (xs[k] - xs[k - 1] > 2.0f) gap = std::min(gap, xs[k] - xs[k - 1] - 24.0f);
            std::fill(room.begin(), room.end(), std::max(24.0f, gap));
        } else {
            // Same level: the same pixel row.
            std::sort(at.begin(), at.end(), [](const auto& a, const auto& b) {
                return std::lround(a.first.y) != std::lround(b.first.y) ? a.first.y < b.first.y : a.first.x < b.first.x;
            });
            for (size_t k = 0; k < at.size(); ++k) {
                float w = kMaxBox;
                if (k > 0 && std::lround(at[k - 1].first.y) == std::lround(at[k].first.y)) w = std::min(w, at[k].first.x - at[k - 1].first.x - 6.0f);
                if (k + 1 < at.size() && std::lround(at[k + 1].first.y) == std::lround(at[k].first.y))
                    w = std::min(w, at[k + 1].first.x - at[k].first.x - 6.0f);
                room[at[k].second] = std::max(24.0f, w);
            }
        }
    }
    const auto box = [&](size_t i) {
        const ImVec2 c = px(i);
        const float w = std::min(room[i], ImGui::CalcTextSize(p.graph_nodes[i].name.c_str()).x + 14.0f);
        const float h = line * (std::isfinite(p.graph_nodes[i].weight) ? 2.0f : 1.0f) + 6.0f;
        return ImRect(ImVec2(c.x - w * 0.5f, c.y - h * 0.5f), ImVec2(c.x + w * 0.5f, c.y + h * 0.5f));
    };

    if (graph_drag_ >= 0 && graph_dragging_ && static_cast<size_t>(graph_drag_) < n) {
        const ImPlotPoint m = ImPlot::GetPlotMousePos();
        graph_pos_[static_cast<size_t>(graph_drag_)] = ImVec2(static_cast<float>(m.x), static_cast<float>(m.y));
        graph_moved_[static_cast<size_t>(graph_drag_)] = 1;
    }

    // The node under the mouse (for hover, moving and folding): the mouse in
    // the plot's rect (while a node is held the plot does not count as hovered).
    hot_graph_ = -1;
    const ImVec2 pp = ImPlot::GetPlotPos(), ps = ImPlot::GetPlotSize();
    const bool inside_plot = ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem) &&
                             ImGui::IsMouseHoveringRect(pp, ImVec2(pp.x + ps.x, pp.y + ps.y));
    if (inside_plot || graph_drag_ >= 0) {
        float best = 1e30f;
        for (size_t i = 0; i < n; ++i) {
            if (!visible[i]) continue;
            if (tree) {
                if (box(i).Contains(mouse)) hot_graph_ = static_cast<int>(i);
            } else {
                const ImVec2 c = px(i);
                const float d = (c.x - mouse.x) * (c.x - mouse.x) + (c.y - mouse.y) * (c.y - mouse.y);
                const float r = radius(i) + 3.0f;
                if (d <= r * r && d < best) {
                    best = d;
                    hot_graph_ = static_cast<int>(i);
                }
            }
        }
    }
    const int focus = graph_drag_ >= 0 ? graph_drag_ : hot_graph_;
    // Where a press grabs the node next frame.
    hot_min_ = hot_max_ = ImVec2(0, 0);
    if (hot_graph_ >= 0) {
        const size_t h = static_cast<size_t>(hot_graph_);
        if (tree) {
            const ImRect r = box(h);
            hot_min_ = r.Min;
            hot_max_ = r.Max;
        } else {
            const ImVec2 c = px(h);
            const float r = radius(h) + 3.0f;
            hot_min_ = ImVec2(c.x - r, c.y - r);
            hot_max_ = ImVec2(c.x + r, c.y + r);
        }
    }

    // Links: weight as width; the focused node's links bright.
    double max_w = 0;
    for (const auto& l : p.graph_links) max_w = std::max(max_w, l.weight);
    for (const auto& l : p.graph_links) {
        const size_t a = static_cast<size_t>(l.a), b = static_cast<size_t>(l.b);
        if (!visible[a] || !visible[b]) continue;
        const bool hot = focus >= 0 && (l.a == focus || l.b == focus);
        const ImU32 col = ui::ToU32(ui::WithAlpha(hot ? t.text_bright : t.text_dim, hot ? 0.85f : (focus >= 0 ? 0.12f : 0.35f)));
        const ImVec2 pa = px(a), pb = px(b);
        if (tree) {
            // A curve from the parent's edge to the child's.
            const ImRect ra = box(a), rb = box(b);
            if (p.spec.tree_left_right) {
                const ImVec2 s(ra.Max.x, pa.y), e(rb.Min.x, pb.y);
                const float mx = (s.x + e.x) * 0.5f;
                dl->AddBezierCubic(s, ImVec2(mx, s.y), ImVec2(mx, e.y), e, col, 1.4f);
            } else {
                const ImVec2 s(pa.x, ra.Max.y), e(pb.x, rb.Min.y);
                const float my = (s.y + e.y) * 0.5f;
                dl->AddBezierCubic(s, ImVec2(s.x, my), ImVec2(e.x, my), e, col, 1.4f);
            }
            continue;
        }
        const float width = 0.6f + 2.6f * static_cast<float>(max_w > 0 ? l.weight / max_w : 0.0);
        dl->AddLine(pa, pb, col, width);
        if (p.spec.directed) {
            // An arrowhead at the target's edge.
            const ImVec2 d(pb.x - pa.x, pb.y - pa.y);
            const float len = std::sqrt(d.x * d.x + d.y * d.y);
            if (len > 1.0f) {
                const ImVec2 u(d.x / len, d.y / len);
                const ImVec2 tip(pb.x - u.x * (radius(b) + 1.0f), pb.y - u.y * (radius(b) + 1.0f));
                const float s = 6.0f;
                dl->AddTriangleFilled(tip, ImVec2(tip.x - u.x * s - u.y * s * 0.5f, tip.y - u.y * s + u.x * s * 0.5f),
                                      ImVec2(tip.x - u.x * s + u.y * s * 0.5f, tip.y - u.y * s - u.x * s * 0.5f), col);
            }
        }
    }

    // Nodes and their names.
    std::vector<ImRect> placed;
    std::vector<size_t> order;
    for (size_t i = 0; i < n; ++i)
        if (visible[i]) order.push_back(i);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        const auto& na = p.graph_nodes[a];
        const auto& nb = p.graph_nodes[b];
        return (std::isfinite(na.weight) ? na.weight : na.links) > (std::isfinite(nb.weight) ? nb.weight : nb.links);
    });
    if (tree) {
        for (size_t i : order) {
            const auto& nd = p.graph_nodes[i];
            const ImRect r = box(i);
            const ImVec4 c = ColourOf(static_cast<size_t>(std::max(0, nd.group)));
            const bool hot = static_cast<int>(i) == focus;
            dl->AddRectFilled(r.Min, r.Max, ui::ToU32(ui::Mix(t.plot_bg, c, hot ? 0.55f : 0.32f)), 5.0f);
            dl->AddRectFilled(ImVec2(r.Min.x, r.Max.y - 3.0f), r.Max, ui::ToU32(c), 2.0f);
            dl->PushClipRect(r.Min, r.Max, true);
            dl->AddText(ImVec2(r.Min.x + 7.0f, r.Min.y + 3.0f), ui::ToU32(t.text_bright), Shorten(nd.name, r.GetWidth() - 14.0f).c_str());
            if (std::isfinite(nd.weight)) dl->AddText(ImVec2(r.Min.x + 7.0f, r.Min.y + 3.0f + line), ui::ToU32(t.text_dim), Num(nd.weight).c_str());
            dl->PopClipRect();
            if (graph_fold_[i]) {
                // How many are folded away under it.
                int hidden = 0;
                std::function<void(int)> count = [&](int v) {
                    for (size_t k = 0; k < n; ++k)
                        if (p.graph_nodes[k].parent == v) {
                            ++hidden;
                            count(static_cast<int>(k));
                        }
                };
                count(static_cast<int>(i));
                const std::string more = "+" + std::to_string(hidden);
                // Where the hidden children would be: right of the box (left-right) or under it.
                const ImVec2 at = p.spec.tree_left_right ? ImVec2(r.Max.x + 4.0f, r.Min.y + 2.0f)
                                                         : ImVec2(r.GetCenter().x - ImGui::CalcTextSize(more.c_str()).x * 0.5f, r.Max.y + 3.0f);
                dl->AddText(at, ui::ToU32(t.accent_text), more.c_str());
            }
        }
    } else {
        const size_t label_cap = p.spec.node_labels == PlotSpec::NodeLabels::None ? 0
                                 : p.spec.node_labels == PlotSpec::NodeLabels::All ? n
                                                                                   : static_cast<size_t>(p.spec.label_top);
        // Draw small first so big nodes sit on top.
        for (auto it = order.rbegin(); it != order.rend(); ++it) {
            const size_t i = *it;
            const ImVec4 c = ColourOf(static_cast<size_t>(std::max(0, p.graph_nodes[i].group)));
            const bool dim = focus >= 0 && static_cast<int>(i) != focus;
            dl->AddCircleFilled(px(i), radius(i), ui::ToU32(ui::WithAlpha(c, dim ? 0.55f : 1.0f)), 20);
            if (static_cast<int>(i) == focus) dl->AddCircle(px(i), radius(i) + 3.0f, ui::ToU32(t.text_bright), 24, 1.5f);
        }
        size_t labelled = 0;
        for (size_t i : order) {
            if (labelled >= label_cap) break;
            const auto& nd = p.graph_nodes[i];
            const ImVec2 c = px(i);
            const ImVec2 ts = ImGui::CalcTextSize(nd.name.c_str());
            const ImRect r(ImVec2(c.x + radius(i) + 3.0f, c.y - ts.y * 0.5f), ImVec2(c.x + radius(i) + 3.0f + ts.x, c.y + ts.y * 0.5f));
            bool clash = false;
            for (const auto& q : placed) clash = clash || q.Overlaps(r);
            if (clash && p.spec.node_labels != PlotSpec::NodeLabels::All) continue;
            placed.push_back(r);
            ++labelled;
            dl->AddRectFilled(ImVec2(r.Min.x - 2.0f, r.Min.y), ImVec2(r.Max.x + 2.0f, r.Max.y), ui::ToU32(ui::WithAlpha(t.plot_bg, 0.6f)), 3.0f);
            dl->AddText(r.Min, ui::ToU32(t.text_bright), nd.name.c_str());
        }
    }

    // Folding: a click (not a drag) on a tree node with children.
    if (tree && clicked >= 0 && static_cast<size_t>(clicked) < n) {
        bool has_children = false;
        for (const auto& nd : p.graph_nodes) has_children = has_children || nd.parent == clicked;
        if (has_children) {
            graph_fold_[static_cast<size_t>(clicked)] ^= 1;
            graph_layout_dirty_ = true;
            fit_ = true;  // the tree is laid out again in 0..1, wider boxes need new margins
        }
    }

    // Hover card.
    if (hot_graph_ >= 0 && graph_drag_ < 0) {
        const auto& nd = p.graph_nodes[static_cast<size_t>(hot_graph_)];
        ImGui::BeginTooltip();
        ImGui::TextColored(t.text_bright, "%s", nd.name.c_str());
        if (tree) {
            Row(t.text_dim, "parent", nd.parent >= 0 ? p.graph_nodes[static_cast<size_t>(nd.parent)].name : std::string("(a root)"));
            Row(t.text_dim, "level", std::to_string(nd.depth));
            if (std::isfinite(nd.weight)) Row(t.text_dim, p.spec.value_column.empty() ? "value" : p.spec.value_column.c_str(), Num(nd.weight));
            if (!nd.label.empty()) Row(t.text_dim, p.spec.color_column.c_str(), nd.label);
            int children = 0;
            for (const auto& c : p.graph_nodes) children += c.parent == hot_graph_ ? 1 : 0;
            if (children) ImGui::TextColored(t.text_faint, "%d children \xC2\xB7 click to %s", children, graph_fold_[static_cast<size_t>(hot_graph_)] ? "unfold" : "fold");
        } else {
            Row(t.text_dim, "links", std::to_string(nd.links));
            if (!p.spec.value_column.empty()) Row(t.text_dim, "weight", Num(nd.weight));
            if (static_cast<size_t>(nd.group) < p.graph_groups.size() && p.graph_groups.size() > 1)
                Row(t.text_dim, "group", p.graph_groups[static_cast<size_t>(nd.group)]);
            // Its strongest links.
            std::vector<std::pair<double, std::string>> nb;
            for (const auto& l : p.graph_links) {
                if (l.a == hot_graph_) nb.push_back({l.weight, p.graph_nodes[static_cast<size_t>(l.b)].name});
                else if (l.b == hot_graph_) nb.push_back({l.weight, p.graph_nodes[static_cast<size_t>(l.a)].name});
            }
            std::stable_sort(nb.begin(), nb.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            for (size_t k = 0; k < nb.size() && k < 6; ++k)
                ImGui::TextColored(t.text_dim, "  %s%s", nb[k].second.c_str(), p.spec.value_column.empty() ? "" : (" (" + Num(nb[k].first) + ")").c_str());
            if (nb.size() > 6) ImGui::TextColored(t.text_faint, "  +%zu more", nb.size() - 6);
            ImGui::TextColored(t.text_faint, "Drag to move it");
        }
        ImGui::EndTooltip();
    }
    ImPlot::EndPlot();
    frame_min_ = ImGui::GetItemRectMin();
    frame_max_ = ImGui::GetItemRectMax();
}

void PlotView::ResetGraphLayout() {
    std::fill(graph_moved_.begin(), graph_moved_.end(), 0);
    std::fill(graph_fold_.begin(), graph_fold_.end(), 0);
    graph_layout_dirty_ = true;
    fit_ = true;
}

}  // namespace cyxwiz::plot
