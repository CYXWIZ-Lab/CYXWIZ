#include "projection_view.h"

#include "explore_plots.h"

#include "../../core/explore_presentation.h"
#include "../../core/file_dialogs.h"
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
#include <set>

namespace cyxwiz {

namespace {

namespace gv = groups_view;
using groups::ProjectionMethod;

QueryResult RunNow(const QueryRequest& r) { return SessionQueryService::Instance().RunNow(r); }

bool Measurable(const ProfiledColumn& c) { return c.numeric && std::isfinite(c.min) && std::isfinite(c.max) && c.max > c.min; }

std::string Key(const std::string& dataset, const groups::ProjectionRequest& r) {
    std::string key = dataset + "|" + gv::ProjectionLabel(r.method) + "|" + (r.standardise ? "z" : "raw") + "|" + r.colour;
    for (const auto& c : r.columns) key += "|" + c;
    switch (r.method) {
        case ProjectionMethod::PCA: break;
        case ProjectionMethod::TSNE:
            key += "|" + std::to_string(r.perplexity) + "|" + std::to_string(r.learning_rate) + "|" + std::to_string(r.iterations) + "|" +
                   std::to_string(r.seed);
            break;
        case ProjectionMethod::UMAP: key += "|" + std::to_string(r.neighbors) + "|" + std::to_string(r.min_dist) + "|" + std::to_string(r.seed); break;
    }
    return key;
}

void Label(const char* text) {
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(ui::CurrentTokens().text_dim, "%s", text);
    ImGui::SameLine();
}

void Wrapped(const ImVec4& colour, const std::string& text) {
    if (text.empty()) return;
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(colour, "%s", text.c_str());
    ImGui::PopTextWrapPos();
}

void Faint(const std::string& text) { Wrapped(ui::CurrentTokens().text_faint, text); }

void Heading(const char* text) { Wrapped(ui::CurrentTokens().text_dim, text); }

void NumberCell(const std::string& text, const ImVec4* colour = nullptr) {
    ImGui::TableNextColumn();
    const float w = ImGui::CalcTextSize(text.c_str()).x;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetContentRegionAvail().x - w));
    if (colour) ImGui::TextColored(*colour, "%s", text.c_str());
    else ImGui::TextUnformatted(text.c_str());
}

// The next setting on this line, or the next when it does not fit.
void Setting(const char* label, float width) {
    ImGui::SameLine(0.0f, 16.0f);
    if (ImGui::GetContentRegionAvail().x < ImGui::CalcTextSize(label).x + width) ImGui::NewLine();
    Label(label);
}

// The groups can colour the points when they were found on the same columns
// (then the rows used are the same).
bool SameColumns(const groups::GroupsAnswer& g, const groups::ProjectionRequest& r) {
    return std::set<std::string>(g.request.columns.begin(), g.request.columns.end()) ==
           std::set<std::string>(r.columns.begin(), r.columns.end());
}

}  // namespace

void ProjectionView::Render(const ExploreContext& context) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (!ExploreProfileReady(context, "Choose a table in the list to see its rows in two dimensions.")) return;
    const DatasetProfile& p = *context.profile;
    if (seen_ != context.profile) {
        seen_ = context.profile;
        job_.Reset();
        message_.clear();
    }
    const groups::GroupsAnswer* shown = groups_shown ? groups_shown(context.dataset) : nullptr;
    RenderControls(p, shown);
    if (!message_.empty()) Wrapped(message_error_ ? t.error : t.text_dim, message_);
    ImGui::Spacing();
    if (request_.columns.size() < 2) {
        Faint("Add two number columns or more.");
        return;
    }
    const std::string key = Key(context.dataset, request_);
    const std::string dataset = context.dataset;
    const groups::ProjectionRequest req = request_;
    const explore::MissingTexts texts = context.missing_texts;
    if (key == stopped_key_) {
        Faint("Stopped. Change a setting to run again.");
        return;
    }
    if (key != asked_key_) {
        asked_key_ = key;
        progress_ = std::make_shared<Progress>();
    }
    std::shared_ptr<Progress> progress = progress_;
    job_.Ask(std::string("Projection: ") + gv::ProjectionLabel(req.method), key, [dataset, req, texts, progress](LambdaTask& task) {
        groups::ProjectionHooks hooks;
        hooks.should_stop = [&task] { return task.ShouldStop(); };
        hooks.progress = [&task, progress](float f, const std::string& what) {
            progress->done = f;
            {
                std::lock_guard<std::mutex> lock(progress->mutex);
                progress->text = what;
            }
            task.ReportProgress(f, what);
        };
        return groups::RunProjection(dataset, req, RunNow, texts, hooks);
    });
    const groups::ProjectionAnswer* a = job_.Answer(key);
    if (!a) {
        if (job_.Running()) RenderRunning();
        else if (!job_.Error().empty()) Wrapped(t.error, job_.Error());
        return;
    }
    RenderAnswer(*a, shown && SameColumns(*shown, a->request) ? shown : nullptr);
}

void ProjectionView::RenderControls(const DatasetProfile& p, const groups::GroupsAnswer* shown) {
    std::vector<plot::ColumnSummary> numbers, all;
    for (const auto& c : p.columns) {
        plot::ColumnSummary s;
        s.name = c.facts.name;
        s.numeric = c.numeric;
        s.distinct = c.facts.distinct;
        s.has_stats = true;
        all.push_back(s);
        if (Measurable(c)) numbers.push_back(std::move(s));
    }
    std::erase_if(request_.columns, [&](const std::string& n) {
        const ProfiledColumn* c = p.Find(n);
        return !c || !Measurable(*c);
    });
    if (request_.columns.empty())
        for (size_t i = 0; i < numbers.size() && i < 8; ++i) request_.columns.push_back(numbers[i].name);
    if (!request_.colour.empty() && !p.Find(request_.colour)) request_.colour.clear();

    ui::FlowRow row;
    row.Next(ImGui::CalcTextSize("Columns").x + 430.0f);
    Label("Columns");
    picker_.PickMany("##projection_columns", request_.columns, numbers, true, std::clamp(ImGui::GetContentRegionAvail().x, 160.0f, 420.0f));
    static const char* const kScale[] = {"Standardise", "As they are"};
    int scale = request_.standardise ? 0 : 1;
    row.Next(ImGui::CalcTextSize("ScaleStandardiseAs they are").x + 50.0f);
    Label("Scale");
    if (ExploreChoices("##projection_scale", kScale, 2, &scale)) request_.standardise = scale == 0;

    static const char* const kMethods[] = {"PCA", "t-SNE", "UMAP"};
    int m = static_cast<int>(request_.method);
    ui::FlowRow method;
    method.Next(ImGui::CalcTextSize("MethodPCAt-SNEUMAP").x + 90.0f);
    Label("Method");
    if (ExploreChoices("##projection_method", kMethods, 3, &m)) request_.method = static_cast<ProjectionMethod>(m);
    switch (request_.method) {
        case ProjectionMethod::PCA: break;
        case ProjectionMethod::TSNE: {
            Setting("perplexity", 120.0f);
            ui::NumberField("##projection_perplexity", &request_.perplexity, 1, 500, 5, 40.0f, "About how many neighbours each row keeps close");
            Setting("learning rate", 110.0f);
            ui::NumberFieldOptions o;
            o.min = 1.0;
            o.max = 10000.0;
            o.format = "%.0f";
            o.width = 60.0f;
            ui::NumberField("##projection_lr", &request_.learning_rate, o);
            Setting("iterations", 120.0f);
            ui::NumberField("##projection_iterations", &request_.iterations, 250, 10000, 250, 50.0f, "Steps of the layout");
            break;
        }
        case ProjectionMethod::UMAP: {
            Setting("neighbours", 120.0f);
            ui::NumberField("##projection_neighbors", &request_.neighbors, 2, 200, 1, 40.0f, "Neighbours (the row itself included) that shape the layout");
            Setting("min distance", 110.0f);
            ui::NumberFieldOptions o;
            o.min = 0.0;
            o.max = 1.0;
            o.step = 0.05;
            o.format = "%.2f";
            o.width = 50.0f;
            o.hint = "How tightly points may pack together";
            ui::NumberField("##projection_min_dist", &request_.min_dist, o);
            break;
        }
    }
    if (request_.method != ProjectionMethod::PCA) {
        Setting("Seed", 110.0f);
        int seed = static_cast<int>(request_.seed);
        if (ui::NumberField("##projection_seed", &seed, 1, 2147483647, 0, 90.0f, "Picks the sample when there are more than 5,000 rows"))
            request_.seed = static_cast<uint32_t>(seed);
    }

    // Colour by: none, the groups of Explore > Groups (same columns), or a column.
    const bool groups_ok = shown && SameColumns(*shown, request_);
    if (by_groups_ && !groups_ok) by_groups_ = false;
    const std::string current = by_groups_ ? std::string("groups (") + gv::MethodLabel(shown->request.method) + ")"
                                : request_.colour.empty() ? std::string("(none)") : request_.colour;
    Label("Colour by");
    ImGui::SetNextItemWidth(std::clamp(ImGui::GetContentRegionAvail().x, 140.0f, 240.0f));
    if (ImGui::BeginCombo("##projection_colour", current.c_str())) {
        if (ImGui::Selectable("(none)", !by_groups_ && request_.colour.empty())) {
            by_groups_ = false;
            request_.colour.clear();
        }
        if (groups_ok) {
            const std::string label = std::string("groups (") + gv::MethodLabel(shown->request.method) + ")";
            if (ImGui::Selectable(label.c_str(), by_groups_)) {
                by_groups_ = true;
                request_.colour.clear();
            }
        }
        for (const auto& c : all)
            if (ImGui::Selectable(c.name.c_str(), !by_groups_ && request_.colour == c.name)) {
                by_groups_ = false;
                request_.colour = c.name;
            }
        ImGui::EndCombo();
    }
    if (!groups_ok && ImGui::IsItemHovered())
        ImGui::SetTooltip("Run Explore > Groups on these same columns to colour the points by its groups");
}

void ProjectionView::RenderRunning() {
    const ui::Tokens& t = ui::CurrentTokens();
    std::string text;
    {
        std::lock_guard<std::mutex> lock(progress_->mutex);
        text = progress_->text;
    }
    ImGui::TextColored(t.accent_text, ICON_FA_SPINNER " %s",
                       text.empty() ? (std::string("Projecting with ") + gv::ProjectionLabel(request_.method) + " on the device...").c_str()
                                    : text.c_str());
    if (request_.method == ProjectionMethod::TSNE) {
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, t.accent_text);
        ImGui::ProgressBar(std::clamp(progress_->done.load(), 0.0f, 1.0f), ImVec2(-FLT_MIN, 0.0f), "");
        ImGui::PopStyleColor();
    }
    if (ui::SecondaryButton("Cancel")) {
        job_.Cancel();
        stopped_key_ = asked_key_;
    }
    if (request_.method == ProjectionMethod::UMAP && ImGui::IsItemHovered())
        ImGui::SetTooltip("UMAP finishes its layout in the background; its answer is not shown");
}

void ProjectionView::RenderAnswer(const groups::ProjectionAnswer& a, const groups::GroupsAnswer* shown) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (a.rows > a.used) Faint(gv::LeftOut(a.rows, a.used));
    Wrapped(t.text, gv::ProjectionSentence(a));

    // The colours of the points.
    gv::Colouring colours;
    if (by_groups_ && shown) {
        std::vector<int> labels;
        for (const int64_t r : a.used_rows) labels.push_back(r >= 0 && static_cast<size_t>(r) < shown->labels.size() ? shown->labels[static_cast<size_t>(r)] : -1);
        colours = gv::ColourGroups(labels);
    } else if (!a.request.colour.empty()) {
        colours = gv::ColourPoints(a.colour, a.colour_numeric);
    }

    const float avail = ImGui::GetContentRegionAvail().x;
    const bool pca = a.request.method == ProjectionMethod::PCA;
    const bool split = pca && avail >= 820.0f;
    if (split && ImGui::BeginTable("##projection_split", 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_NoSavedSettings)) {
        ImGui::TableSetupColumn("plot", ImGuiTableColumnFlags_WidthStretch, 1.3f);
        ImGui::TableSetupColumn("components", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
    }
    {
        const float w = split ? ImGui::GetContentRegionAvail().x : avail;
        const std::string xl = gv::AxisTitle(a, 0), yl = gv::AxisTitle(a, 1);
        explore_plot::Scatter(a.x, a.y, w, std::clamp(w * 0.62f, 180.0f, 340.0f), nullptr, 0, xl.c_str(), yl.c_str(),
                              colours.index.empty() ? nullptr : &colours.index);
        // The legend, in a line that wraps.
        ui::FlowRow row;
        for (size_t i = 0; i < colours.legend.size(); ++i) {
            const auto& [name, n] = colours.legend[i];
            const std::string text = name + " (" + std::to_string(n) + ")";
            row.Next(ImGui::CalcTextSize(text.c_str()).x + 22.0f);
            const ImVec2 at = ImGui::GetCursorScreenPos();
            const ImVec4 c = name == "noise" ? t.text_faint : ui::SeriesColour(t, i);
            ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(at.x + 5.0f, at.y + ImGui::GetTextLineHeight() * 0.5f + 1.0f), 4.5f, ui::ToU32(c));
            ImGui::SetCursorScreenPos(ImVec2(at.x + 14.0f, at.y));
            ImGui::TextColored(t.text_dim, "%s", text.c_str());
        }
    }
    if (pca) {
        if (split) ImGui::TableNextColumn();
        else ImGui::Spacing();
        RenderComponents(a);
        if (split) ImGui::EndTable();
    }
    ImGui::Spacing();
    RenderActions(a);
}

void ProjectionView::RenderComponents(const groups::ProjectionAnswer& a) {
    const ui::Tokens& t = ui::CurrentTokens();
    Heading("EXPLAINED VARIANCE");
    const float bar_w = std::max(80.0f, ImGui::GetContentRegionAvail().x - 190.0f);
    double cumulative = 0.0;
    for (size_t c = 0; c < a.ratio.size(); ++c) {
        cumulative += a.ratio[c];
        ImGui::TextColored(t.text_dim, "component %zu", c + 1);
        ImGui::SameLine(110.0f);
        const ImVec2 at = ImGui::GetCursorScreenPos();
        const float h = ImGui::GetTextLineHeight();
        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->AddRectFilled(ImVec2(at.x, at.y + 2.0f), ImVec2(at.x + bar_w, at.y + h - 2.0f), ui::ToU32(t.plot_bg), 3.0f);
        dl->AddRectFilled(ImVec2(at.x, at.y + 2.0f), ImVec2(at.x + bar_w * static_cast<float>(a.ratio[c]), at.y + h - 2.0f),
                          ui::ToU32(ui::SeriesColour(t, 0)), 3.0f);
        ImGui::Dummy(ImVec2(bar_w, h));
        ImGui::SameLine();
        ImGui::Text("%s", gv::Share(a.ratio[c]).c_str());
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s with the components before it", gv::Share(cumulative).c_str());
    }
    ImGui::Spacing();
    Heading("WHAT THE COMPONENTS ARE (LOADINGS)");
    if (a.loadings.empty()) return;
    if (ImGui::BeginTable("##projection_loadings", 3, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg | ImGuiTableFlags_NoSavedSettings)) {
        ImGui::TableSetupColumn("column", ImGuiTableColumnFlags_WidthStretch, 2.0f);
        ImGui::TableSetupColumn("1", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("2", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextColored(t.text_dim, "column");
        NumberCell("component 1", &t.text_dim);
        NumberCell("component 2", &t.text_dim);
        for (size_t j = 0; j < a.request.columns.size(); ++j) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(a.request.columns[j].c_str());
            for (size_t c = 0; c < 2; ++c) {
                const double v = c < a.loadings.size() && j < a.loadings[c].size() ? a.loadings[c][j] : NAN;
                NumberCell(gv::Score(v), std::fabs(v) >= 0.4 ? &t.text_bright : &t.text_faint);
            }
        }
        ImGui::EndTable();
    }
}

void ProjectionView::RenderActions(const groups::ProjectionAnswer& a) {
    if (add_to_graph) {
        const char* node = gv::ProjectionLabel(a.request.method);
        if (ui::PrimaryButton("Add as a step", true, nullptr, ui::ButtonSize::Small)) {
            message_ = add_to_graph(a.request, true);
            message_error_ = false;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Puts the %s node after the data in use: the rest of the graph gets its columns beside the others", node);
        ImGui::SameLine();
        if (ui::SecondaryButton("Keep as a node")) {
            message_ = add_to_graph(a.request, false);
            message_error_ = false;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Adds the %s node as a branch reading the data in use", node);
        ImGui::SameLine();
    }
    if (ui::SecondaryButton("Export CSV...")) Export(a);
    ImGui::SameLine();
    if (ui::SecondaryButton("Copy")) {
        const std::string text = gv::ProjectionSentence(a) + "\n";
        ImGui::SetClipboardText(text.c_str());
        message_ = "Copied the summary.";
        message_error_ = false;
    }
}

void ProjectionView::Export(const groups::ProjectionAnswer& a) {
    const std::string name = std::string("projection_") + gv::ProjectionLabel(a.request.method) + ".csv";
    if (const auto path = FileDialogs::SaveFile("Export projection", {{"CSV", "csv"}}, nullptr, name.c_str())) {
        std::ofstream out(*path, std::ios::binary);
        out << gv::ProjectionCsv(a);
        message_ = out ? "Saved " + *path : "Could not write " + *path;
        message_error_ = !out;
    }
}

}  // namespace cyxwiz
