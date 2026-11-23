#include "word_map_view.h"

#include "../core/explore_presentation.h"

#include <cyxwiz/text_processing.h>
#include "plot/plot_view.h"
#include "separate_windows.h"
#include "ui_buttons.h"
#include "ui_tokens.h"
#include "ui_widgets.h"

#include <implot3d.h>
#include <implot3d_internal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <unordered_set>

namespace cyxwiz {

namespace {

constexpr size_t kMoreLabels = 70;  // labels beyond the word and its neighbours, as room allows

std::string Fixed(double v, int digits) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.*f", digits, v);
    return buf;
}

std::string Percent(double share) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.1f%%", share * 100.0);
    return buf;
}

ImVec4 Mix(const ImVec4& a, const ImVec4& b, float f) {
    return ImVec4(a.x + (b.x - a.x) * f, a.y + (b.y - a.y) * f, a.z + (b.z - a.z) * f, a.w);
}

void Faint(const std::string& text) {
    if (text.empty()) return;
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(ui::CurrentTokens().text_faint, "%s", text.c_str());
    ImGui::PopTextWrapPos();
}

void Heading(const char* text) { ImGui::TextColored(ui::CurrentTokens().text_dim, "%s", text); }

}  // namespace

WordMapView::WordMapView(std::string id) : id_(std::move(id)) {}

void WordMapView::SetSource(Source source) {
    if (source.key == source_.key && source.vectors == source_.vectors) return;
    source_ = std::move(source);
    job_.Reset();
    map_.reset();
    map_index_.clear();
    leans_.clear();
    asked_key_.clear();
    neighbours_for_.clear();
    neighbours_.clear();
    error_.clear();
    refit_ = true;
    if (source_.groups.empty() && colour_ == 1) colour_ = 0;
}

std::string WordMapView::Title() const {
    return "Word map" + (source_.title.empty() ? std::string() : ": " + source_.title) + "###wordmap_" + id_;
}

void WordMapView::Open() {
    open_ = true;
    gui::BringToFront(Title());
}

void WordMapView::LookUp(const std::string& word) {
    std::snprintf(query_, sizeof query_, "%s", word.c_str());
    word_ = word_map::QueryWord(word);
    neighbours_for_.clear();
    focus_word_ = true;
    refit_ = true;
}

void WordMapView::FindNeighbours() {
    const std::string key = source_.key + "|" + word_ + "|" + std::to_string(neighbours_count_);
    if (key == neighbours_for_) return;
    neighbours_for_ = key;
    neighbours_.clear();
    neighbours_error_.clear();
    if (word_.empty() || !source_.vectors) return;
    neighbours_ = WordEmbeddings::Nearest(*source_.vectors, word_, neighbours_count_, &neighbours_error_);
}

// The layout for the commonest words and the word's neighbours that are not
// among them (a new one only when the set changes).
void WordMapView::Ask() {
    if (!source_.vectors) return;
    std::vector<std::string> extra;
    std::string wanted = source_.key + "|" + std::to_string(words_);
    if (!word_.empty()) {
        extra.push_back(word_);
        for (const auto& [w, c] : neighbours_) extra.push_back(w);
        for (const std::string& w : extra) wanted += "|" + w;
    }
    if (wanted != words_for_) {
        words_for_ = wanted;
        const size_t n = static_cast<size_t>(std::max(50, words_));
        words_list_ = word_map::MapWords(*source_.vectors, n, extra);
        // Only the extra words that are past the commonest change the layout.
        const size_t base = extra.empty() ? words_list_.size() : word_map::MapWords(*source_.vectors, n, {}).size();
        extra_key_.clear();
        for (size_t i = base; i < words_list_.size(); ++i) extra_key_ += "|" + words_list_[i];
    }
    const std::string key =
        source_.key + "|" + word_map::LayoutName(layout_) + "|" + std::to_string(words_) + "|" + std::to_string(clusters_) + extra_key_;
    if (key != asked_key_) {
        asked_key_ = key;
        word_map::MapRequest request;
        request.words = words_list_;
        request.layout = layout_;
        request.clusters = clusters_;
        const auto vectors = source_.vectors;
        job_.Ask(std::string("Word map: ") + word_map::LayoutName(layout_), key, [vectors, request](LambdaTask& task) {
            return word_map::Place(*vectors, request, [&task] { return task.ShouldStop(); },
                                   [&task](float done, const std::string& text) { task.ReportProgress(done, text); });
        });
    }
    if (const word_map::MapAnswer* a = job_.Answer(asked_key_)) {
        if (shown_key_ != asked_key_) {
            shown_key_ = asked_key_;
            map_ = std::make_shared<word_map::MapAnswer>(*a);
            map_index_.clear();
            for (size_t i = 0; i < map_->words.size(); ++i) map_index_.emplace(map_->words[i], i);
            leans_ = source_.groups.empty() ? std::vector<int>{} : word_map::LeansTo(map_->words, source_.word_rows, source_.group_rows);
            if (focus_word_) refit_ = true;
        }
        error_.clear();
    } else if (!job_.Error().empty()) {
        error_ = job_.Error();
    }
}

ImVec4 WordMapView::ColourOf(size_t i) const {
    const ui::Tokens& t = ui::CurrentTokens();
    if (colour_ == 2 || !map_) return t.accent_text;
    if (colour_ == 1) return i < leans_.size() && leans_[i] >= 0 ? ui::SeriesColour(t, static_cast<size_t>(leans_[i])) : t.text_faint;
    return i < map_->cluster.size() ? ui::SeriesColour(t, static_cast<size_t>(std::max(0, map_->cluster[i]))) : t.accent_text;
}

void WordMapView::DrawPreview(ImVec2 size) {
    if (!source_.vectors) return;
    FindNeighbours();
    Ask();
    DrawMap(size, false);
}

void WordMapView::DrawWindow() {
    if (!open_) return;
    const std::string title = Title();
    // The first time: over its host in the main window, most of its size;
    // from there its tab docks it anywhere or moves it out (A8).
    const ImGuiViewport* main = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(main->WorkPos.x + main->WorkSize.x * 0.5f, main->WorkPos.y + main->WorkSize.y * 0.5f), ImGuiCond_FirstUseEver,
                            ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(main->WorkSize.x * 0.8f, main->WorkSize.y * 0.8f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowViewport(main->ID);
    gui::NextWindowMayLeave(title.c_str());
    const bool expanded = ImGui::Begin(title.c_str(), &open_);
    gui::TabMenu(title.c_str(), &open_);
    if (expanded) {
        if (!source_.vectors) {
            Faint("No vectors yet: learn them in Explore > Text > Embeddings, or train or open them in a Word Embeddings node.");
        } else {
            FindNeighbours();
            Ask();
            const float side = 270.0f;
            if (ImGui::BeginTable("##wordmap_split", 2, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_NoSavedSettings | ImGuiTableFlags_Resizable)) {
                ImGui::TableSetupColumn("controls", ImGuiTableColumnFlags_WidthFixed, side);
                ImGui::TableSetupColumn("map", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::BeginChild("##wordmap_controls", ImVec2(0, ImGui::GetContentRegionAvail().y), ImGuiChildFlags_None);
                DrawControls();
                ImGui::EndChild();
                ImGui::TableNextColumn();
                const ImVec2 avail = ImGui::GetContentRegionAvail();
                DrawMap(ImVec2(avail.x, std::max(300.0f, avail.y - ImGui::GetTextLineHeightWithSpacing() * 1.6f)), true);
                ImGui::EndTable();
            }
        }
    }
    ImGui::End();
}

void WordMapView::DrawControls() {
    const ui::Tokens& t = ui::CurrentTokens();
    Heading("LOOK UP A WORD");
    ImGui::SetNextItemWidth(-1.0f);
    if (ImGui::InputTextWithHint("##wordmap_query", "type a word, then Enter", query_, sizeof query_, ImGuiInputTextFlags_EnterReturnsTrue) ||
        (ImGui::IsItemDeactivatedAfterEdit() && word_ != query_))
        LookUp(query_);
    if (word_.empty()) {
        Faint("Or double-click a word on the map.");
    } else if (!neighbours_error_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.error, "%s", neighbours_error_.c_str());
        ImGui::PopTextWrapPos();
    } else if (neighbours_.empty()) {
        Faint("\"" + word_ + "\" has no vector: it is not seen often enough.");
    } else {
        Faint("Nearest to \"" + word_ + "\" by the vectors' cosine:");
        for (size_t i = 0; i < neighbours_.size(); ++i) {
            const auto& [w, c] = neighbours_[i];
            ImGui::PushID(static_cast<int>(i));
            const std::string cosine = Fixed(c, 2);
            if (ImGui::Selectable(w.c_str(), false)) LookUp(w);
            ImGui::SameLine(ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(cosine.c_str()).x + ImGui::GetCursorPosX() - 8.0f);
            ImGui::TextColored(t.text_faint, "%s", cosine.c_str());
            ImGui::PopID();
        }
    }
    ImGui::Spacing();
    Heading("NEIGHBOURS SHOWN");
    if (ui::NumberField("##wordmap_k", &neighbours_count_, 1, 50, 1, 56.0f, "How many nearest words are marked and joined")) neighbours_for_.clear();
    ImGui::Spacing();
    Heading("LAYOUT");
    static const char* const kLayout[] = {"t-SNE", "UMAP", "PCA"};
    int layout = static_cast<int>(layout_);
    if (ExploreChoices("##wordmap_layout", kLayout, 3, &layout)) {
        layout_ = static_cast<word_map::Layout>(layout);
        refit_ = true;
    }
    if (layout_ == word_map::Layout::PCA && map_ && map_->request.layout == word_map::Layout::PCA)
        Faint("PCA: the three axes hold " + Percent(map_->share[0]) + ", " + Percent(map_->share[1]) + " and " + Percent(map_->share[2]) +
              " of the spread: faithful but crowded.");
    else if (layout_ == word_map::Layout::PCA)
        Faint("PCA: the vectors' three widest directions (their share is said once placed).");
    else
        Faint(std::string(word_map::LayoutName(layout_)) + ": near words stay near; distances between far clusters mean little.");
    ImGui::Spacing();
    Heading("VIEW");
    static const char* const kView[] = {"3D", "flat (from above)"};
    int view = flat_ ? 1 : 0;
    if (ExploreChoices("##wordmap_view", kView, 2, &view)) {
        flat_ = view == 1;
        refit_ = true;
    }
    if (ui::SecondaryButton("Reset view")) {
        refit_ = true;
        focus_word_ = false;
        spin_ = false;
    }
    ImGui::SameLine();
    if (ui::SecondaryButton(spin_ ? "Stop turning" : "Turn slowly")) spin_ = !spin_;
    ImGui::Spacing();
    Heading("COLOUR");
    static const char* const kColour[] = {"clusters", "leans to a group", "one colour"};
    if (source_.groups.empty()) {
        static const char* const kTwo[] = {"clusters", "one colour"};
        int c = colour_ == 2 ? 1 : 0;
        if (ExploreChoices("##wordmap_colour2", kTwo, 2, &c)) colour_ = c == 1 ? 2 : 0;
    } else {
        ExploreChoices("##wordmap_colour", kColour, 3, &colour_);
    }
    if (colour_ == 0) {
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.text_dim, "k-means of");
        ImGui::SameLine();
        ui::NumberField("##wordmap_clusters", &clusters_, 2, 20, 1, 48.0f, "How many clusters the vectors are split into");
        ImGui::SameLine();
        ImGui::TextColored(t.text_dim, "clusters");
    }
    if (map_ && colour_ != 2) {
        const size_t count = colour_ == 0 ? map_->cluster_names.size() : source_.groups.size();
        for (size_t k = 0; k < count; ++k) {
            const ImVec2 at = ImGui::GetCursorScreenPos();
            const float h = ImGui::GetTextLineHeight();
            ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(at.x + 5.0f, at.y + h * 0.5f), 4.5f, ui::ToU32(ui::SeriesColour(t, k)), 12);
            ImGui::Dummy(ImVec2(12.0f, h));
            ImGui::SameLine(0.0f, 6.0f);
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(t.text_dim, "%s", (colour_ == 0 ? map_->cluster_names[k] : source_.groups[k]).c_str());
            ImGui::PopTextWrapPos();
        }
        if (colour_ == 1) Faint("Grey: no group (a word leans to a group when it is twice as common there, in 30 of its rows or more).");
    }
    ImGui::Spacing();
    Heading("WORDS ON THE MAP");
    ui::NumberField("##wordmap_words", &words_, 100, 5000, 100, 72.0f, "The commonest words placed (common English words left out); the word looked up and its neighbours are always added");
    ImGui::SameLine();
    ImGui::TextColored(t.text_faint, "commonest, plus the neighbours");
    ImGui::Spacing();
    Heading("LABELS");
    static const char* const kLabels[] = {"and room for more", "the word and its neighbours"};
    int labels = more_labels_ ? 0 : 1;
    if (ExploreChoices("##wordmap_labels", kLabels, 2, &labels)) more_labels_ = labels == 0;
    ImGui::Spacing();
    if (ui::SecondaryButton("Export PNG...")) export_pending_ = "word_map.png";
}

void WordMapView::DrawMap(ImVec2 size, bool full) {
    const ui::Tokens& t = ui::CurrentTokens();
    // Placing: a bar over the map (or over the empty plot the first time).
    const auto placing = [&](const ImVec2& at, float width) {
        std::string message;
        const float done = job_.Progress(&message);
        ImDrawList* fg = ImGui::GetWindowDrawList();
        const std::string text = std::string("Placing the words (") + word_map::LayoutName(layout_) + ", on the device)" +
                                 (message.empty() ? std::string("...") : ": " + message);
        fg->AddText(ImVec2(at.x + 12.0f, at.y + 10.0f), ui::ToU32(t.text_dim), text.c_str());
        const float y = at.y + 14.0f + ImGui::GetTextLineHeight();
        fg->AddRectFilled(ImVec2(at.x + 12.0f, y), ImVec2(at.x + width - 12.0f, y + 4.0f), ui::ToU32(t.bg_raised), 2.0f);
        if (done > 0.0f)
            fg->AddRectFilled(ImVec2(at.x + 12.0f, y), ImVec2(at.x + 12.0f + (width - 24.0f) * std::min(1.0f, done), y + 4.0f), ui::ToU32(t.accent), 2.0f);
    };
    if (!map_) {
        const ImVec2 at = ImGui::GetCursorScreenPos();
        ImGui::Dummy(size);
        ImGui::GetWindowDrawList()->AddRectFilled(at, ImVec2(at.x + size.x, at.y + size.y), ui::ToU32(t.plot_bg), 4.0f);
        if (!error_.empty()) ImGui::GetWindowDrawList()->AddText(ImVec2(at.x + 12.0f, at.y + 10.0f), ui::ToU32(t.error), error_.c_str());
        else placing(at, size.x);
        return;
    }
    const word_map::MapAnswer& m = *map_;
    // The view: the whole map, or around the word and its neighbours.
    std::vector<std::string> near_words;
    for (const auto& [w, c] : neighbours_) near_words.push_back(w);
    const bool focus = focus_word_ && map_index_.count(word_);
    const word_map::Focus f = focus ? word_map::FocusOn(m, word_, near_words) : word_map::Focus{};
    const double h = f.half;
    const size_t qi = map_index_.count(word_) ? map_index_.at(word_) : SIZE_MAX;
    std::unordered_map<size_t, double> near;
    for (const auto& [w, c] : neighbours_)
        if (const auto it = map_index_.find(w); it != map_index_.end()) near.emplace(it->second, c);
    const bool lit_some = qi != SIZE_MAX;
    struct Dot {
        size_t i;
        ImVec2 px;
        float depth;
    };
    std::vector<Dot> dots;
    dots.reserve(m.words.size());
    ImRect area;
    ImDrawList* dl = nullptr;
    bool hovered_plot = false;
    if (full) {
        // The window: ImPlot3D's box, to turn, zoom and move with the mouse.
        ImPlot3DStyle& st = ImPlot3D::GetStyle();
        st.Colors[ImPlot3DCol_FrameBg] = ImVec4(0, 0, 0, 0);
        st.Colors[ImPlot3DCol_PlotBg] = t.plot_bg;
        st.Colors[ImPlot3DCol_PlotBorder] = ImVec4(0, 0, 0, 0);
        st.Colors[ImPlot3DCol_AxisGrid] = ui::WithAlpha(t.plot_grid, 0.5f);
        st.Colors[ImPlot3DCol_AxisTick] = ImVec4(0, 0, 0, 0);
        st.PlotPadding = ImVec2(6, 6);
        const std::string plot_id = "##wordmap_full_" + id_;
        if (!ImPlot3D::BeginPlot(plot_id.c_str(), size, ImPlot3DFlags_CanvasOnly | ImPlot3DFlags_NoMenus)) return;
        ImPlot3DPlot& plot = *ImPlot3D::GetCurrentPlot();
        const ImPlot3DAxisFlags axis = ImPlot3DAxisFlags_NoLabel | ImPlot3DAxisFlags_NoTickLabels;
        ImPlot3D::SetupAxes("", "", "", axis, axis, axis);
        ImPlot3D::SetupAxesLimits(f.centre[0] - h, f.centre[0] + h, f.centre[1] - h, f.centre[1] + h, flat_ ? -0.05 : f.centre[2] - h,
                                  flat_ ? 0.05 : f.centre[2] + h, refit_ ? ImPlot3DCond_Always : ImPlot3DCond_Once);
        refit_ = false;
        if (flat_) {
            ImPlot3D::SetupBoxRotation(90.0f, 0.0f, false, ImPlot3DCond_Always);
        } else if (spin_) {
            spin_azimuth_ += ImGui::GetIO().DeltaTime * 8.0f;
            ImPlot3D::SetupBoxRotation(22.0f, spin_azimuth_, false, ImPlot3DCond_Always);
        }
        ImPlot3D::SetupLock();
        const ImPlot3DPoint lo = plot.RangeMin(), hi = plot.RangeMax();
        for (size_t i = 0; i < m.words.size(); ++i) {
            const ImPlot3DPoint pt(static_cast<float>(m.xyz[i][0]), static_cast<float>(m.xyz[i][1]), flat_ ? 0.0f : static_cast<float>(m.xyz[i][2]));
            if (pt.x < lo.x || pt.x > hi.x || pt.y < lo.y || pt.y > hi.y || pt.z < lo.z || pt.z > hi.z) continue;
            dots.push_back({i, ImPlot3D::PlotToPixels(pt), (plot.Rotation * ImPlot3D::PlotToNDC(pt)).z});
        }
        area = plot.PlotRect;
        dl = ImPlot3D::GetPlotDrawList();
        hovered_plot = plot.Hovered;
    } else {
        // The preview: a camera turning slowly round the word, drawn here so
        // that it takes no mouse input (the page scrolls over it).
        const ImVec2 at = ImGui::GetCursorScreenPos();
        ImGui::Dummy(size);
        area = ImRect(at, ImVec2(at.x + size.x, at.y + size.y));
        dl = ImGui::GetWindowDrawList();
        dl->AddRectFilled(area.Min, area.Max, ui::ToU32(t.plot_bg), 4.0f);
        spin_azimuth_ += ImGui::GetIO().DeltaTime * 12.0f;
        const float yaw = spin_azimuth_ * 3.14159265f / 180.0f, pitch = 22.0f * 3.14159265f / 180.0f;
        const float cy = std::cos(yaw), sy = std::sin(yaw), cp = std::cos(pitch), sp = std::sin(pitch);
        const ImVec2 centre = area.GetCenter();
        const float scale = std::min(size.x, size.y) * 0.42f;
        for (size_t i = 0; i < m.words.size(); ++i) {
            const float x = static_cast<float>((m.xyz[i][0] - f.centre[0]) / h), y = static_cast<float>((m.xyz[i][1] - f.centre[1]) / h),
                        z = static_cast<float>((m.xyz[i][2] - f.centre[2]) / h);
            if (std::fabs(x) > 1.0f || std::fabs(y) > 1.0f || std::fabs(z) > 1.0f) continue;
            const float rx = x * cy - y * sy, ry = x * sy + y * cy;  // turned about the vertical
            const float depth = ry * cp - z * sp, up = ry * sp + z * cp;  // then tilted towards the eye
            dots.push_back({i, ImVec2(centre.x + rx * scale, centre.y - up * scale), -depth});
        }
    }
    std::sort(dots.begin(), dots.end(), [](const Dot& a, const Dot& b) { return a.depth < b.depth; });
    dl->PushClipRect(area.Min, area.Max, true);
    // The lines from the word to its neighbours, under the dots.
    std::unordered_map<size_t, ImVec2> at;
    for (const Dot& d : dots) at.emplace(d.i, d.px);
    if (qi != SIZE_MAX && at.count(qi))
        for (const auto& [j, c] : near)
            if (at.count(j)) dl->AddLine(at[qi], at[j], ui::ToU32(ui::WithAlpha(t.caution, 0.75f)), 1.5f);
    const size_t n = std::max<size_t>(1, m.words.size());
    for (const Dot& d : dots) {
        const bool lit = !lit_some || d.i == qi || near.count(d.i);
        ImVec4 c = ColourOf(d.i);
        c.w = 0.9f;
        if (!lit) c = Mix(c, t.plot_bg, 0.62f);
        float r = (full ? 2.0f : 1.4f) + (full ? 2.6f : 1.6f) * (1.0f - std::min(1.0f, static_cast<float>(d.i) / static_cast<float>(n)));
        if (d.i == qi) r = full ? 7.5f : 5.5f;
        else if (near.count(d.i)) r = full ? 5.0f : 3.8f;
        dl->AddCircleFilled(d.px, r, ui::ToU32(d.i == qi ? t.accent : c), 12);
        if (d.i == qi) dl->AddCircle(d.px, r + 3.0f, ui::ToU32(t.text_bright), 16, 1.5f);
        else if (near.count(d.i)) dl->AddCircle(d.px, r + 1.5f, ui::ToU32(t.caution), 12, 1.2f);
    }
    // The dot under the mouse.
    size_t hovered = SIZE_MAX;
    if (full && hovered_plot) {
        const ImVec2 mouse = ImGui::GetMousePos();
        float best = 9.0f * 9.0f;
        for (auto it = dots.rbegin(); it != dots.rend(); ++it) {
            const float dx = it->px.x - mouse.x, dy = it->px.y - mouse.y;
            if (dx * dx + dy * dy < best) {
                best = dx * dx + dy * dy;
                hovered = it->i;
            }
        }
        if (hovered != SIZE_MAX && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) LookUp(m.words[hovered]);
    }
    // Labels, most wanted first, none over another.
    std::vector<ImVec4> boxes;
    const auto label = [&](size_t i, const std::string& text, int kind) {  // 0 the word, 1 a neighbour, 2 hovered, 3 others
        if (!at.count(i)) return false;
        const ImVec2 p = at[i];
        const ImVec2 ts = ImGui::CalcTextSize(text.c_str());
        const float w = ts.x + 8.0f, hgt = ts.y + 2.0f;
        const auto free = [&](const ImVec2& a) {
            const ImVec2 b(a.x + w, a.y + hgt);
            if (a.x < area.Min.x || b.x > area.Max.x || a.y < area.Min.y || b.y > area.Max.y) return false;
            for (const ImVec4& o : boxes)
                if (a.x < o.z && b.x > o.x && a.y < o.w && b.y > o.y) return false;
            return true;
        };
        // Right of the dot, left, above, below, then stepped up and down on
        // either side (the word, its neighbours and the hovered one search
        // further): the first place that is free.
        std::vector<ImVec2> sides = {ImVec2(p.x + 8.0f, p.y - hgt * 0.5f), ImVec2(p.x - 8.0f - w, p.y - hgt * 0.5f),
                                     ImVec2(p.x - w * 0.5f, p.y - 8.0f - hgt), ImVec2(p.x - w * 0.5f, p.y + 8.0f)};
        if (kind != 3)
            for (int k = 1; k <= 6; ++k)
                for (const float dir : {-1.0f, 1.0f}) {
                    const float dy = dir * static_cast<float>(k) * (hgt + 2.0f);
                    sides.push_back(ImVec2(p.x + 8.0f, p.y - hgt * 0.5f + dy));
                    sides.push_back(ImVec2(p.x - 8.0f - w, p.y - hgt * 0.5f + dy));
                }
        ImVec2 a = sides[0];
        bool placed = false;
        for (const ImVec2& s : sides)
            if (free(s)) {
                a = s;
                placed = true;
                break;
            }
        if (!placed && kind == 3) return false;  // the word, its neighbours and the hovered one always show
        const ImVec2 b(a.x + w, a.y + hgt);
        boxes.push_back(ImVec4(a.x - 2.0f, a.y - 2.0f, b.x + 2.0f, b.y + 2.0f));
        if (p.y < a.y - 1.0f || p.y > b.y + 1.0f)  // moved off its dot: a leader line
            dl->AddLine(p, ImVec2(std::clamp(p.x, a.x, b.x), p.y < a.y ? a.y : b.y), ui::ToU32(ui::WithAlpha(t.text_faint, 0.6f)), 1.0f);
        const ImVec4 bg = kind == 0 ? t.accent : kind == 1 || kind == 2 ? t.bg_raised : ui::WithAlpha(t.plot_bg, 0.72f);
        dl->AddRectFilled(a, b, ui::ToU32(bg), 3.0f);
        if (kind == 1) dl->AddRectFilled(a, ImVec2(a.x + 2.0f, b.y), ui::ToU32(t.caution), 0.0f);
        dl->AddText(ImVec2(a.x + 4.0f, a.y + 1.0f), ui::ToU32(kind == 0 ? t.text_bright : kind == 3 ? t.text_dim : t.text), text.c_str());
        return true;
    };
    if (hovered != SIZE_MAX) {
        std::string text = m.words[hovered];
        if (colour_ == 0 && hovered < m.cluster.size() && static_cast<size_t>(m.cluster[hovered]) < m.cluster_names.size())
            text += "  (" + m.cluster_names[static_cast<size_t>(m.cluster[hovered])] + ")";
        label(hovered, text, 2);
    }
    if (qi != SIZE_MAX && qi != hovered) label(qi, m.words[qi], 0);
    for (const auto& [w, c] : neighbours_)
        if (const auto it = map_index_.find(w); it != map_index_.end() && it->second != hovered) label(it->second, w + " " + Fixed(c, 2), 1);
    if (full && more_labels_) {
        size_t placed = 0;
        for (auto it = dots.rbegin(); it != dots.rend() && placed < kMoreLabels; ++it) {
            if (it->i == qi || it->i == hovered || near.count(it->i) || it->i > 400) continue;  // the commonest, nearest the eye
            if (label(it->i, m.words[it->i], 3)) ++placed;
        }
    }
    if (job_.Running()) placing(area.Min, area.GetWidth());  // a new layout on its way (the old one shown)
    dl->PopClipRect();
    if (!full) return;
    const ImVec2 plot_min = area.Min, plot_max = area.Max;
    ImPlot3D::EndPlot();
    {
        ImGui::TextColored(t.text_faint, "%s words on the map, %s%s  -  drag to turn, scroll to zoom, right-drag to move, double-click a word to look it up",
                           explore_view::Number(m.words.size()).c_str(), word_map::LayoutName(m.request.layout), flat_ ? " seen from above" : " in 3D");
        if (!export_pending_.empty()) {
            const std::string name = export_pending_;
            export_pending_.clear();
            const auto& hooks = plot::Hooks();
            if (hooks.capture_png && hooks.save_path)
                if (const auto path = hooks.save_path("Export the word map", "png", name))
                    hooks.capture_png(plot_min, plot_max, [file = *path](std::vector<unsigned char> png) {
                        std::ofstream out(file, std::ios::binary);
                        out.write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
                    });
        }
    }
}

}  // namespace cyxwiz
