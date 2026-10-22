#include "groups_view.h"

#include "explore_plots.h"

#include "../../core/file_dialogs.h"
#include "../../core/explore_presentation.h"
#include "../../core/groups_presentation.h"
#include "../../core/session_query_service.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <fstream>

namespace cyxwiz {

namespace {

namespace gv = groups_view;
using groups::Method;

QueryResult RunNow(const QueryRequest& r) { return SessionQueryService::Instance().RunNow(r); }

bool Measurable(const ProfiledColumn& c) { return c.numeric && std::isfinite(c.min) && std::isfinite(c.max) && c.max > c.min; }

std::string Key(const std::string& dataset, const groups::GroupsRequest& r) {
    std::string key = dataset + "|" + gv::MethodLabel(r.method) + "|" + (r.standardise ? "z" : "raw") + "|" + r.compare;
    for (const auto& c : r.columns) key += "|" + c;
    switch (r.method) {
        case Method::KMeans: key += "|" + std::to_string(r.k) + "|" + std::to_string(r.seed); break;
        case Method::DBSCAN: key += "|" + std::to_string(r.eps) + "|" + std::to_string(r.min_samples) + "|" + r.metric; break;
        case Method::Hierarchical: key += "|" + std::to_string(r.k) + "|" + r.linkage + "|" + r.metric; break;
        case Method::Mixture: key += "|" + std::to_string(r.k) + "|" + std::to_string(r.seed) + "|" + r.covariance; break;
    }
    return key;
}

// The how-many question depends on the columns, scaling and the method's own settings, not k.
std::string HowManyKey(const std::string& dataset, const groups::GroupsRequest& r, int from, int to) {
    groups::GroupsRequest q = r;
    q.k = 0;
    q.eps = 0.0;
    q.compare.clear();
    return Key(dataset, q) + "|" + std::to_string(from) + "|" + std::to_string(to);
}

void Label(const char* text) {
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(ui::CurrentTokens().text_dim, "%s", text);
    ImGui::SameLine();
}

void Faint(const std::string& text) {
    if (text.empty()) return;
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(ui::CurrentTokens().text_faint, "%s", text.c_str());
    ImGui::PopTextWrapPos();
}

void Wrapped(const ImVec4& colour, const std::string& text) {
    if (text.empty()) return;
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(colour, "%s", text.c_str());
    ImGui::PopTextWrapPos();
}

void Heading(const char* text) {
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(ui::CurrentTokens().text_dim, "%s", text);
    ImGui::PopTextWrapPos();
}

void NumberCell(const std::string& text, const ImVec4* colour = nullptr) {
    ImGui::TableNextColumn();
    const float w = ImGui::CalcTextSize(text.c_str()).x;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetContentRegionAvail().x - w));
    if (colour) ImGui::TextColored(*colour, "%s", text.c_str());
    else ImGui::TextUnformatted(text.c_str());
}

// A combo of fixed choices; true when it changed.
bool Choice(const char* id, std::string& value, std::initializer_list<const char*> choices, float width) {
    bool changed = false;
    ImGui::SetNextItemWidth(width);
    if (ImGui::BeginCombo(id, value.c_str())) {
        for (const char* c : choices)
            if (ImGui::Selectable(c, value == c)) {
                changed = value != c;
                value = c;
            }
        ImGui::EndCombo();
    }
    return changed;
}

// The height of a table of `rows` rows that may scroll sideways (a ScrollX
// table needs one; without it nothing is drawn).
float TableHeight(size_t rows) {
    return static_cast<float>(rows) * (ImGui::GetTextLineHeight() + 2.0f * ImGui::GetStyle().CellPadding.y) + ImGui::GetStyle().ScrollbarSize + 4.0f;
}

// A ScrollX table's inner width: the room its columns need, or the line.
float InnerWidth(const std::vector<std::string>& headers, float first) {
    float w = first;
    for (const auto& h : headers) w += std::max(ImGui::CalcTextSize(h.c_str()).x, 56.0f) + 2.0f * ImGui::GetStyle().CellPadding.x + 8.0f;
    return std::max(w, ImGui::GetContentRegionAvail().x);
}

// A coloured dot and its group's name and size, in a line that wraps.
void Legend(const groups::GroupsAnswer& a) {
    const ui::Tokens& t = ui::CurrentTokens();
    ui::FlowRow row;
    const auto item = [&](const ImVec4& colour, const std::string& text) {
        row.Next(ImGui::CalcTextSize(text.c_str()).x + 22.0f);
        const ImVec2 at = ImGui::GetCursorScreenPos();
        ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(at.x + 5.0f, at.y + ImGui::GetTextLineHeight() * 0.5f + 1.0f), 4.5f, ui::ToU32(colour));
        ImGui::SetCursorScreenPos(ImVec2(at.x + 14.0f, at.y));
        ImGui::TextColored(t.text_dim, "%s", text.c_str());
    };
    for (const auto& g : a.groups) item(ui::SeriesColour(t, static_cast<size_t>(g.label)), gv::GroupName(g.label) + " (" + std::to_string(g.rows) + ")");
    if (a.noise > 0) item(t.text_faint, "noise (" + std::to_string(a.noise) + ")");
}

}  // namespace

const groups::GroupsAnswer* GroupsView::Shown(const std::string& dataset) const {
    return dataset == dataset_ && !key_.empty() ? job_.Answer(key_) : nullptr;
}

void GroupsView::Render(const ExploreContext& context) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (!ExploreProfileReady(context, "Choose a table in the list to find which of its rows group together.")) return;
    const DatasetProfile& p = *context.profile;
    if (seen_ != context.profile) {
        seen_ = context.profile;
        job_.Reset();
        how_many_job_.Reset();
        message_.clear();
    }
    dataset_ = context.dataset;
    RenderControls(p);
    if (!message_.empty()) Wrapped(message_error_ ? t.error : t.text_dim, message_);
    ImGui::Spacing();
    if (request_.columns.empty()) {
        key_.clear();
        Faint("Add the number columns that describe the rows (two or more).");
        return;
    }
    if (how_many_open_) {
        RenderHowMany(context);
        ImGui::Spacing();
    }
    key_ = Key(context.dataset, request_);
    const std::string dataset = context.dataset;
    const groups::GroupsRequest req = request_;
    const explore::MissingTexts texts = context.missing_texts;
    job_.Ask(std::string("Groups: ") + gv::MethodLabel(req.method), key_,
             [dataset, req, texts](LambdaTask&) { return groups::RunGroups(dataset, req, RunNow, texts); });
    const groups::GroupsAnswer* a = job_.Answer(key_);
    if (!a) {
        if (!job_.Error().empty()) Wrapped(t.error, job_.Error());
        else Faint("Grouping on the device...");
        return;
    }
    RenderAnswer(*a, p);
}

void GroupsView::RenderControls(const DatasetProfile& p) {
    std::vector<plot::ColumnSummary> numbers;
    for (const auto& c : p.columns) {
        if (!Measurable(c)) continue;
        plot::ColumnSummary s;
        s.name = c.facts.name;
        s.numeric = true;
        numbers.push_back(std::move(s));
    }
    std::erase_if(request_.columns, [&](const std::string& n) {
        const ProfiledColumn* c = p.Find(n);
        return !c || !Measurable(*c);
    });
    // Default: every number column, up to eight.
    if (request_.columns.empty())
        for (size_t i = 0; i < numbers.size() && i < 8; ++i) request_.columns.push_back(numbers[i].name);

    ui::FlowRow row;
    row.Next(ImGui::CalcTextSize("Columns").x + 430.0f);
    Label("Columns");
    picker_.PickMany("##groups_columns", request_.columns, numbers, true, std::clamp(ImGui::GetContentRegionAvail().x, 160.0f, 420.0f));
    static const char* const kScale[] = {"Standardise", "As they are"};
    int scale = request_.standardise ? 0 : 1;
    row.Next(ImGui::CalcTextSize("ScaleStandardiseAs they are").x + 50.0f);
    Label("Scale");
    if (ExploreChoices("##groups_scale", kScale, 2, &scale)) request_.standardise = scale == 0;
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Standardise: each column as z-scores, so a column in large units does not decide the groups alone");

    ui::FlowRow method;
    static const char* const kMethods[] = {"K-Means", "DBSCAN", "Hierarchical", "Gaussian mixture"};
    int m = static_cast<int>(request_.method);
    method.Next(ImGui::CalcTextSize("MethodK-MeansDBSCANHierarchicalGaussian mixture").x + 110.0f);
    Label("Method");
    if (ExploreChoices("##groups_method", kMethods, 4, &m)) request_.method = static_cast<Method>(m);
    RenderMethodSettings();
    const char* how = how_many_open_ ? "Hide how many" : "How many groups?";
    ImGui::SameLine(0.0f, 16.0f);
    if (ImGui::GetContentRegionAvail().x < ImGui::CalcTextSize(how).x + 8.0f) ImGui::NewLine();
    if (ui::LinkButton(how_many_open_ ? "Hide how many" : "How many groups?")) how_many_open_ = !how_many_open_;
}

void GroupsView::RenderMethodSettings() {
    const auto setting = [](const char* label, float width) {
        ImGui::SameLine(0.0f, 16.0f);
        if (ImGui::GetContentRegionAvail().x < ImGui::CalcTextSize(label).x + width) ImGui::NewLine();
        Label(label);
    };
    switch (request_.method) {
        case Method::KMeans:
        case Method::Mixture:
            setting(request_.method == Method::KMeans ? "Groups" : "Components", 120.0f);
            ui::NumberField("##groups_k", &request_.k, 2, 50, 1, 40.0f, "How many groups to make");
            setting("Seed", 110.0f);
            {
                int seed = static_cast<int>(request_.seed);
                if (ui::NumberField("##groups_seed", &seed, 1, 2147483647, 0, 90.0f, "The same seed gives the same groups"))
                    request_.seed = static_cast<uint32_t>(seed);
            }
            if (request_.method == Method::Mixture) {
                setting("Covariance", 110.0f);
                Choice("##groups_cov", request_.covariance, {"full", "tied", "diag", "spherical"}, 100.0f);
            }
            break;
        case Method::DBSCAN: {
            setting("eps", 150.0f);
            ui::NumberFieldOptions o;
            o.min = 1e-6;
            o.max = 1e6;
            o.step = 0.05;
            o.format = "%.3g";
            o.width = 70.0f;
            o.hint = "How near two rows must be to be neighbours (in the scaled units)";
            ui::NumberField("##groups_eps", &request_.eps, o);
            setting("min samples", 120.0f);
            ui::NumberField("##groups_min", &request_.min_samples, 1, 1000, 1, 40.0f, "Neighbours (the row itself included) that make a core row");
            setting("metric", 110.0f);
            Choice("##groups_metric", request_.metric, {"euclidean", "manhattan", "cosine"}, 110.0f);
            break;
        }
        case Method::Hierarchical:
            setting("Groups", 120.0f);
            ui::NumberField("##groups_k", &request_.k, 2, 50, 1, 40.0f, "Where to cut the tree");
            setting("linkage", 110.0f);
            if (Choice("##groups_linkage", request_.linkage, {"ward", "complete", "average", "single"}, 100.0f) && request_.linkage == "ward")
                request_.metric = "euclidean";
            if (request_.linkage != "ward") {
                setting("metric", 110.0f);
                Choice("##groups_hmetric", request_.metric, {"euclidean", "manhattan", "cosine"}, 110.0f);
            }
            break;
    }
}

void GroupsView::RenderAnswer(const groups::GroupsAnswer& a, const DatasetProfile& p) {
    const ui::Tokens& t = ui::CurrentTokens();
    std::vector<std::pair<std::string, std::string>> figures;
    if (a.scored) {
        figures = {{"silhouette", gv::Score(a.silhouette)}, {"Davies-Bouldin", gv::Score(a.davies_bouldin)},
                   {"Calinski-Harabasz", explore_view::Figure(a.calinski_harabasz)}};
    }
    figures.emplace_back("rows used", std::to_string(a.used));
    ExploreFigures("##groups_scores", figures);
    Wrapped(t.text, gv::Summary(a));
    const std::string method = gv::MethodLine(a);
    if (!method.empty()) Faint(method + ".");
    Faint(gv::LeftOut(a.rows, a.used));
    ImGui::Spacing();

    const float avail = ImGui::GetContentRegionAvail().x;
    const bool split = avail >= 820.0f;
    if (split && ImGui::BeginTable("##groups_split", 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_NoSavedSettings)) {
        ImGui::TableSetupColumn("scatter", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("profile", ImGuiTableColumnFlags_WidthStretch, 1.1f);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
    }
    {
        const float w = split ? ImGui::GetContentRegionAvail().x : avail;
        Heading("THE ROWS ON THE FIRST TWO COMPONENTS");
        const std::string xl = std::isfinite(a.x_ratio) ? "component 1 (" + gv::Share(a.x_ratio) + ")" : "component 1";
        const std::string yl = std::isfinite(a.y_ratio) ? "component 2 (" + gv::Share(a.y_ratio) + ")" : "component 2";
        explore_plot::Scatter(a.x, a.y, w, std::clamp(w * 0.62f, 160.0f, 300.0f), nullptr, 0, xl.c_str(), yl.c_str(), &a.labels);
        Legend(a);
    }
    if (split) ImGui::TableNextColumn();
    else ImGui::Spacing();
    RenderProfile(a);
    if (split) ImGui::EndTable();
    ImGui::Spacing();
    RenderCompare(a, p);
    ImGui::Spacing();
    RenderActions(a);
}

void GroupsView::RenderProfile(const groups::GroupsAnswer& a) {
    const ui::Tokens& t = ui::CurrentTokens();
    Heading("WHAT EACH GROUP IS (MEANS)");
    const size_t d = a.request.columns.size();
    std::vector<std::string> headers = {"rows", "silhouette"};
    headers.insert(headers.end(), a.request.columns.begin(), a.request.columns.end());
    if (ImGui::BeginTable("##groups_profile", static_cast<int>(d + 3),
                          ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg | ImGuiTableFlags_NoSavedSettings | ImGuiTableFlags_ScrollX,
                          ImVec2(0.0f, TableHeight(a.groups.size() + 2)), InnerWidth(headers, 90.0f))) {
        ImGui::TableSetupColumn("group", ImGuiTableColumnFlags_WidthStretch, 1.2f);
        ImGui::TableSetupColumn("rows", ImGuiTableColumnFlags_WidthStretch, 0.7f);
        ImGui::TableSetupColumn("silhouette", ImGuiTableColumnFlags_WidthStretch, 0.9f);
        for (size_t j = 0; j < d; ++j) ImGui::TableSetupColumn(a.request.columns[j].c_str(), ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        Heading("group");
        NumberCell("rows", &t.text_dim);
        NumberCell("silhouette", &t.text_dim);
        for (const auto& c : a.request.columns) NumberCell(c, &t.text_dim);
        for (const auto& g : a.groups) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextColored(ui::SeriesColour(t, static_cast<size_t>(g.label)), "%s", gv::GroupName(g.label).c_str());
            NumberCell(std::to_string(g.rows));
            NumberCell(gv::Score(g.silhouette));
            for (const double m : g.means) NumberCell(explore_view::Figure(m));
        }
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextColored(t.text_dim, "all rows");
        NumberCell(std::to_string(a.used), &t.text_dim);
        NumberCell(gv::Score(a.silhouette), &t.text_dim);
        for (const double m : a.overall_means) NumberCell(explore_view::Figure(m), &t.text_dim);
        ImGui::EndTable();
    }
    for (size_t g = 0; g < a.groups.size(); ++g)
        Wrapped(t.text, gv::GroupName(a.groups[g].label) + ": " + gv::Profile(a, g) + ".");
}

void GroupsView::RenderCompare(const groups::GroupsAnswer& a, const DatasetProfile& p) {
    const ui::Tokens& t = ui::CurrentTokens();
    std::vector<plot::ColumnSummary> columns;
    for (const auto& c : p.columns) {
        if (c.facts.distinct < 2 || c.facts.distinct > 50) continue;
        if (std::find(request_.columns.begin(), request_.columns.end(), c.facts.name) != request_.columns.end()) continue;
        plot::ColumnSummary s;
        s.name = c.facts.name;
        s.numeric = c.numeric;
        s.distinct = c.facts.distinct;
        s.has_stats = true;
        columns.push_back(std::move(s));
    }
    Label("Compare with a column");
    picker_.Pick("##groups_compare", request_.compare, columns, false, "(none)", 220.0f);
    if (a.request.compare.empty() || a.compare_values.empty()) return;
    const std::string agreement = "agreement (adjusted Rand) " + gv::Score(a.adjusted_rand);
    ImGui::SameLine(0.0f, 16.0f);
    if (ImGui::GetContentRegionAvail().x < ImGui::CalcTextSize(agreement.c_str()).x) ImGui::NewLine();
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "%s", agreement.c_str());
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("1: the groups are the column's values; 0: no better than chance");
    const size_t v = a.compare_values.size();
    if (ImGui::BeginTable("##groups_crosstab", static_cast<int>(v + 1),
                          ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg | ImGuiTableFlags_NoSavedSettings | ImGuiTableFlags_ScrollX,
                          ImVec2(0.0f, TableHeight(a.groups.size() + 1)), InnerWidth(a.compare_values, 90.0f))) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        Heading("group");
        for (const auto& value : a.compare_values) NumberCell(value, &t.text_dim);
        for (size_t g = 0; g < a.groups.size() && g < a.crosstab.size(); ++g) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextColored(ui::SeriesColour(t, static_cast<size_t>(a.groups[g].label)), "%s", gv::GroupName(a.groups[g].label).c_str());
            for (const int64_t n : a.crosstab[g]) NumberCell(std::to_string(n), n == 0 ? &t.text_faint : nullptr);
        }
        ImGui::EndTable();
    }
    Wrapped(t.text, gv::CompareSentence(a));
}

void GroupsView::RenderActions(const groups::GroupsAnswer& a) {
    if (add_to_graph) {
        if (ui::PrimaryButton("Add as a step", true, nullptr, ui::ButtonSize::Small)) {
            message_ = add_to_graph(a.request, true);
            message_error_ = false;
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Puts the %s node after the data in use: the rest of the graph gets a cluster_id column", gv::MethodLabel(a.request.method));
        ImGui::SameLine();
        if (ui::SecondaryButton("Keep as a node")) {
            message_ = add_to_graph(a.request, false);
            message_error_ = false;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Adds the %s node as a branch reading the data in use", gv::MethodLabel(a.request.method));
        ImGui::SameLine();
    }
    if (ui::SecondaryButton("Export CSV...")) Export(a);
    ImGui::SameLine();
    if (ui::SecondaryButton("Copy")) {
        std::string text = std::string(gv::MethodLabel(a.request.method)) + ": " + gv::Summary(a) + "\n";
        for (size_t g = 0; g < a.groups.size(); ++g) text += gv::GroupName(a.groups[g].label) + ": " + gv::Profile(a, g) + "\n";
        const std::string compare = gv::CompareSentence(a);
        if (!compare.empty()) text += compare + "\n";
        ImGui::SetClipboardText(text.c_str());
        message_ = "Copied the groups.";
        message_error_ = false;
    }
}

void GroupsView::RenderHowMany(const ExploreContext& context) {
    const ui::Tokens& t = ui::CurrentTokens();
    const Method m = request_.method;
    if (m != Method::DBSCAN) {
        Label("How many groups? from");
        ui::NumberField("##groups_from", &from_, 2, 49, 1, 40.0f);
        ImGui::SameLine(0.0f, 12.0f);
        if (ImGui::GetContentRegionAvail().x < ImGui::CalcTextSize("to").x + 130.0f) ImGui::NewLine();
        Label("to");
        ui::NumberField("##groups_to", &to_, 3, 50, 1, 40.0f);
        to_ = std::max(to_, from_ + 1);
    }
    const std::string key = HowManyKey(context.dataset, request_, from_, to_);
    groups::HowManyRequest req;
    req.base = request_;
    req.from = from_;
    req.to = to_;
    const std::string dataset = context.dataset;
    const explore::MissingTexts texts = context.missing_texts;
    how_many_job_.Ask(std::string("Groups: how many (") + gv::MethodLabel(m) + ")", key,
                      [dataset, req, texts](LambdaTask&) { return groups::RunHowMany(dataset, req, RunNow, texts); });
    const groups::HowManyAnswer* h = how_many_job_.Answer(key);
    if (!h) {
        if (!how_many_job_.Error().empty()) Wrapped(t.error, how_many_job_.Error());
        else Faint(m == Method::DBSCAN ? "Measuring the distances to the neighbours..." : "Trying each number of groups on the device...");
        return;
    }
    const float w = ImGui::GetContentRegionAvail().x;
    const float height = std::clamp(w * 0.32f, 150.0f, 240.0f);
    std::vector<double> ks(h->ks.begin(), h->ks.end());
    const std::string suggested = h->suggested > 0 ? "suggested " + std::to_string(h->suggested) : std::string();
    switch (m) {
        case Method::KMeans:
            Heading("INERTIA (ELBOW) AND SILHOUETTE BY NUMBER OF GROUPS");
            explore_plot::Lines(ks, {{"inertia", h->inertia}, {"silhouette", h->silhouette}}, w, height, true,
                                h->suggested > 0 ? h->suggested : NAN, suggested, "groups");
            break;
        case Method::Mixture:
            Heading("BIC AND AIC BY NUMBER OF COMPONENTS (LOWER IS BETTER)");
            explore_plot::Lines(ks, {{"BIC", h->bic}, {"AIC", h->aic}}, w, height, false, h->suggested > 0 ? h->suggested : NAN, suggested,
                                "components");
            break;
        case Method::Hierarchical: {
            Heading("THE TREE'S TOP MERGES, AND THE SILHOUETTE OF EACH CUT");
            const auto top = groups::TopOfTree(h->linkage, h->used, 12);
            // The cut between the merges that leave k and k - 1 groups.
            double cut = NAN;
            const int64_t below = h->used - request_.k - 1;
            if (below >= 0 && below + 1 < static_cast<int64_t>(h->linkage.size()))
                cut = (h->linkage[static_cast<size_t>(below)][2] + h->linkage[static_cast<size_t>(below + 1)][2]) / 2.0;
            explore_plot::Dendrogram(top.xs, top.ys, top.leaf_rows, cut, w, height);
            Faint("Below each leaf: its rows. The dashed line cuts the tree into " + std::to_string(request_.k) + " groups.");
            explore_plot::Lines(ks, {{"silhouette", h->silhouette}}, w, height * 0.7f, false, h->suggested > 0 ? h->suggested : NAN,
                                suggested, "groups");
            break;
        }
        case Method::DBSCAN: {
            const int k = std::max(1, request_.min_samples - 1);
            Heading("DISTANCE TO EACH ROW'S NEIGHBOURS, SORTED (PICK EPS AT THE KNEE)");
            const std::string ylabel = std::to_string(k) + (k == 1 ? "st" : k == 2 ? "nd" : k == 3 ? "rd" : "th") + " neighbour";
            explore_plot::Curve(h->kdistances, w, height, request_.eps, "eps " + explore_view::Figure(request_.eps), "rows, sorted", ylabel.c_str());
            break;
        }
    }
    Wrapped(t.text, gv::HowManySentence(*h));
    if (m == Method::DBSCAN) {
        if (std::isfinite(h->knee) && h->knee != request_.eps && ui::SecondaryButton(("Use eps " + explore_view::Figure(h->knee)).c_str())) request_.eps = h->knee;
    } else if (h->suggested > 0 && h->suggested != request_.k) {
        if (ui::SecondaryButton(("Use " + std::to_string(h->suggested)).c_str())) request_.k = h->suggested;
    }
}

void GroupsView::Export(const groups::GroupsAnswer& a) {
    const std::string name = std::string("groups_") + gv::MethodLabel(a.request.method) + ".csv";
    if (const auto path = FileDialogs::SaveFile("Export groups", {{"CSV", "csv"}}, nullptr, name.c_str())) {
        std::ofstream out(*path, std::ios::binary);
        out << gv::GroupsCsv(a);
        message_ = out ? "Saved " + *path : "Could not write " + *path;
        message_error_ = !out;
    }
}

}  // namespace cyxwiz
