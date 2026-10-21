#include "tests_view.h"

#include "../../core/file_dialogs.h"
#include "../../core/session_query_service.h"
#include "../../core/stats_presentation.h"
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

namespace sv = stats_view;
using stats::Question;

QueryResult RunNow(const QueryRequest& r) { return SessionQueryService::Instance().RunNow(r); }

bool Measurable(const ProfiledColumn& c) { return c.numeric && std::isfinite(c.min) && std::isfinite(c.max) && c.max > c.min; }

// A column that splits rows into a few groups (2 .. 20 values); `exactly` asks for that many.
bool Grouping(const ProfiledColumn& c, size_t exactly = 0) {
    const size_t d = c.facts.distinct;
    return exactly ? d == exactly : d >= 2 && d <= stats::kMaxGroups;
}

const ProfiledColumn* FirstWhere(const DatasetProfile& p, bool (*want)(const ProfiledColumn&)) {
    for (const auto& c : p.columns)
        if (want(c)) return &c;
    return nullptr;
}

std::string Key(const std::string& dataset, const stats::TestRequest& r) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "|%d|%d|%d|%.17g|%.17g", static_cast<int>(r.question), r.two_columns ? 1 : 0, r.welch ? 1 : 0,
                  r.confidence, r.value);
    return dataset + "|" + r.values + "|" + r.groups + "|" + r.second + buf;
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

void Heading(const char* text) { ImGui::TextColored(ui::CurrentTokens().text_dim, "%s", text); }

void NumberCell(const std::string& text, const ImVec4* colour = nullptr) {
    ImGui::TableNextColumn();
    const float w = ImGui::CalcTextSize(text.c_str()).x;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetContentRegionAvail().x - w));
    if (colour) ImGui::TextColored(*colour, "%s", text.c_str());
    else ImGui::TextUnformatted(text.c_str());
}

bool BeginTable(const char* id, int columns) {
    return ImGui::BeginTable(id, columns, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg | ImGuiTableFlags_NoSavedSettings);
}

void Headers(std::initializer_list<const char*> names) {
    ImGui::TableNextRow();
    bool first = true;
    for (const char* n : names) {
        if (first) {
            ImGui::TableNextColumn();
            Heading(n);
            first = false;
        } else {
            NumberCell(n, &ui::CurrentTokens().text_dim);
        }
    }
}

// The figures of the answer, label / value rows.
std::vector<std::pair<std::string, std::string>> AnswerRows(const stats::TestAnswer& a) {
    const auto& m = a.main;
    const auto& r = a.request;
    const std::string ci = sv::Level(r.confidence) + " CI";
    std::vector<std::pair<std::string, std::string>> rows;
    switch (r.question) {
        case Question::TwoGroups:
            rows = {{"difference", sv::Num(m.estimate)}, {ci, sv::Range(m.ci_low, m.ci_high)}, {"t", sv::Num(m.statistic)},
                    {"df", sv::Num(m.df)}, {"p", sv::P(m.p)}, {"Cohen's d", sv::Num(m.effect) + " (" + sv::DWords(m.effect) + ")"}};
            break;
        case Question::Paired:
            rows = {{"mean change", sv::Num(m.estimate)}, {ci, sv::Range(m.ci_low, m.ci_high)}, {"t", sv::Num(m.statistic)},
                    {"df", sv::Num(m.df)}, {"p", sv::P(m.p)}, {"d (paired)", sv::Num(m.effect) + " (" + sv::DWords(m.effect) + ")"}};
            break;
        case Question::SeveralGroups:
            rows = {{"F", sv::Num(m.statistic) + (r.welch ? " (Welch)" : " (classic)")}, {"df", sv::Num(m.df) + ", " + sv::Num(m.df2)},
                    {"p", sv::P(m.p)}, {"eta\xC2\xB2", sv::Num(m.effect) + " (" + sv::EtaWords(m.effect) + ")"}};
            break;
        case Question::OneValue:
            rows = {{"mean", sv::Num(m.estimate)}, {ci + " of the mean", sv::Range(m.ci_low, m.ci_high)}, {"t", sv::Num(m.statistic)},
                    {"df", sv::Num(m.df)}, {"p", sv::P(m.p)}, {"Cohen's d", sv::Num(m.effect) + " (" + sv::DWords(m.effect) + ")"}};
            break;
        case Question::TwoCategories:
            rows = {{"chi\xC2\xB2", sv::Num(m.statistic)}, {"df", sv::Num(m.df)}, {"p", sv::P(m.p)},
                    {"Cramer's V", sv::Num(m.effect) + " (" + sv::VWords(m.effect) + ")"}};
            break;
    }
    return rows;
}

}  // namespace

void TestsView::Render(const ExploreContext& context) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (!ExploreProfileReady(context, "Choose a table in the list to test a question about it.")) return;
    const DatasetProfile& p = *context.profile;
    if (seen_ != context.profile) {
        seen_ = context.profile;
        job_.Reset();
        message_.clear();
    }
    RenderControls(p);
    if (!message_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(message_error_ ? t.error : t.text_dim, "%s", message_.c_str());
        ImGui::PopTextWrapPos();
    }
    ImGui::Spacing();
    // The columns the question needs (a question switch fills them on the next frame).
    const auto& q = request_;
    const bool labels = q.question == Question::TwoCategories;
    const bool needs_second = labels || q.question == Question::Paired || (q.question == Question::TwoGroups && q.two_columns);
    const bool needs_groups = labels || ((q.question == Question::TwoGroups && !q.two_columns) || q.question == Question::SeveralGroups);
    if ((!labels && q.values.empty()) || (needs_groups && q.groups.empty()) || (needs_second && q.second.empty())) {
        Faint(needs_groups && q.groups.empty() ? "This table has no column that splits its rows into a few groups; pick one."
                                               : "Pick the columns to test.");
        return;
    }
    const std::string key = Key(context.dataset, request_);
    const std::string dataset = context.dataset;
    const stats::TestRequest req = request_;
    const explore::MissingTexts texts = context.missing_texts;
    job_.Ask("Tests: " + std::string(sv::QuestionLabel(req.question)), key,
             [dataset, req, texts](LambdaTask&) { return stats::RunTest(dataset, req, RunNow, texts); });
    const stats::TestAnswer* a = job_.Answer(key);
    if (!a) {
        if (!job_.Error().empty()) {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(t.error, "%s", job_.Error().c_str());
            ImGui::PopTextWrapPos();
        } else {
            Faint("Working out the answer on the device...");
        }
        return;
    }
    const float avail = ImGui::GetContentRegionAvail().x;
    if (avail >= 760.0f && ImGui::BeginTable("##tests_split", 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_NoSavedSettings)) {
        ImGui::TableSetupColumn("groups", ImGuiTableColumnFlags_WidthStretch, 1.15f);
        ImGui::TableSetupColumn("answer", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        RenderGroups(*a);
        ImGui::TableNextColumn();
        RenderAnswer(*a);
        ImGui::EndTable();
    } else {
        RenderGroups(*a);
        ImGui::Spacing();
        RenderAnswer(*a);
    }
}

void TestsView::RenderControls(const DatasetProfile& p) {
    const ui::Tokens& t = ui::CurrentTokens();
    std::vector<plot::ColumnSummary> numbers, all;
    for (const auto& c : p.columns) {
        plot::ColumnSummary s;
        s.name = c.facts.name;
        s.numeric = c.numeric;
        s.distinct = c.facts.distinct;
        s.has_stats = true;
        all.push_back(s);
        if (Measurable(c)) numbers.push_back(s);
    }
    // Defaults: a number column, and a column that splits the rows into groups.
    const auto valid_number = [&](const std::string& name) {
        const ProfiledColumn* c = p.Find(name);
        return c && Measurable(*c);
    };
    if (!valid_number(request_.values)) {
        const ProfiledColumn* c = FirstWhere(p, Measurable);
        request_.values = c ? c->facts.name : std::string();
    }
    const auto pick_group = [&](size_t exactly) {
        for (const auto& c : p.columns)
            if (!c.numeric && Grouping(c, exactly)) return c.facts.name;
        for (const auto& c : p.columns)
            if (Grouping(c, exactly) && c.facts.name != request_.values) return c.facts.name;
        return std::string();
    };
    if (!p.Find(request_.groups)) request_.groups = pick_group(request_.question == Question::TwoGroups ? 2 : 0);
    if (!p.Find(request_.second)) {
        if (request_.question == Question::TwoCategories) {
            for (const auto& c : p.columns)
                if (!c.numeric && Grouping(c) && c.facts.name != request_.groups) {
                    request_.second = c.facts.name;
                    break;
                }
        } else {
            for (const auto& c : p.columns)
                if (Measurable(c) && c.facts.name != request_.values) {
                    request_.second = c.facts.name;
                    break;
                }
        }
    }

    // The question.
    static const char* const kQuestions[] = {"Two groups", "Paired columns", "Several groups", "One column vs a value", "Two categories"};
    int q = static_cast<int>(request_.question);
    Label("Question");
    if (ui::SegmentedControl("##tests_question", kQuestions, 5, &q)) {
        const Question before = request_.question;
        request_.question = static_cast<Question>(q);
        // The columns that suit the new question.
        if ((request_.question == Question::TwoCategories) != (before == Question::TwoCategories)) request_.second.clear();
        const ProfiledColumn* g = p.Find(request_.groups);
        if (request_.question == Question::TwoGroups && (!g || g->facts.distinct != 2)) request_.groups = pick_group(2);
        if (request_.question == Question::SeveralGroups && (!g || !Grouping(*g) || g->facts.distinct < 3)) {
            const std::string three = [&] {
                for (const auto& c : p.columns)
                    if (Grouping(c) && c.facts.distinct >= 3 && !c.numeric) return c.facts.name;
                return pick_group(0);
            }();
            request_.groups = three;
        }
        message_.clear();
    }

    ui::FlowRow row;
    switch (request_.question) {
        case Question::TwoGroups:
        case Question::SeveralGroups:
            row.Next(ImGui::CalcTextSize("Values").x + 230.0f);
            Label("Values");
            picker_.Pick("##tests_values", request_.values, numbers, true, nullptr, 220.0f);
            if (request_.question == Question::TwoGroups && request_.two_columns) {
                row.Next(ImGui::CalcTextSize("and").x + 230.0f);
                Label("and");
                picker_.Pick("##tests_second", request_.second, numbers, true, nullptr, 220.0f);
            } else {
                row.Next(ImGui::CalcTextSize("split by").x + 230.0f);
                Label("split by");
                picker_.Pick("##tests_groups", request_.groups, all, false, nullptr, 220.0f);
            }
            if (request_.question == Question::TwoGroups) {
                const char* other = request_.two_columns ? "or a group column" : "or two columns";
                row.Next(ui::ButtonWidth(other, ui::ButtonSize::Small));
                if (ui::LinkButton(other)) request_.two_columns = !request_.two_columns;
            }
            break;
        case Question::Paired:
            row.Next(ImGui::CalcTextSize("Before").x + 230.0f);
            Label("Before");
            picker_.Pick("##tests_before", request_.values, numbers, true, nullptr, 220.0f);
            row.Next(ImGui::CalcTextSize("after").x + 230.0f);
            Label("after");
            picker_.Pick("##tests_after", request_.second, numbers, true, nullptr, 220.0f);
            break;
        case Question::OneValue: {
            row.Next(ImGui::CalcTextSize("Values").x + 230.0f);
            Label("Values");
            picker_.Pick("##tests_values", request_.values, numbers, true, nullptr, 220.0f);
            row.Next(ImGui::CalcTextSize("compared with").x + 110.0f);
            Label("compared with");
            ui::NumberFieldOptions value;
            value.format = "%g";
            value.width = 90.0f;
            value.hint = "The value the mean is compared with";
            ui::NumberField("##tests_value", &request_.value, value);
            break;
        }
        case Question::TwoCategories:
            row.Next(ImGui::CalcTextSize("Rows").x + 230.0f);
            Label("Rows");
            picker_.Pick("##tests_rows", request_.groups, all, false, nullptr, 220.0f);
            row.Next(ImGui::CalcTextSize("columns").x + 230.0f);
            Label("columns");
            picker_.Pick("##tests_cols", request_.second, all, false, nullptr, 220.0f);
            break;
    }

    ui::FlowRow settings;
    if (request_.question == Question::TwoGroups || request_.question == Question::SeveralGroups) {
        static const char* const kTwo[] = {"Unequal (Welch)", "Equal (Student)"};
        static const char* const kSeveral[] = {"Unequal (Welch)", "Equal (classic)"};
        settings.Next(ImGui::CalcTextSize("Variances").x + ImGui::CalcTextSize("Unequal (Welch)Equal (Student)").x + 60.0f);
        Label("Variances");
        int v = request_.welch ? 0 : 1;
        if (ui::SegmentedControl("##tests_variances", request_.question == Question::TwoGroups ? kTwo : kSeveral, 2, &v))
            request_.welch = v == 0;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Welch's test does not assume the groups spread alike; it is the safer default");
    }
    settings.Next(ImGui::CalcTextSize("alpha").x + 160.0f);
    Label("alpha");
    double alpha = 1.0 - request_.confidence;
    ui::NumberFieldOptions options;
    options.min = 0.001;
    options.max = 0.5;
    options.format = "%g";
    options.width = 60.0f;
    options.hint = "The significance level: a p-value below it counts as significant; the CI is at 1 - alpha";
    if (ui::NumberField("##tests_alpha", &alpha, options)) request_.confidence = 1.0 - alpha;
    ImGui::SameLine();
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_faint, "typed; 0.001 .. 0.5");
}

void TestsView::RenderGroups(const stats::TestAnswer& a) {
    const ui::Tokens& t = ui::CurrentTokens();
    const auto& r = a.request;
    if (r.question == Question::TwoCategories) {
        Heading("COUNTS (EXPECTED)");
        if (BeginTable("##tests_counts", static_cast<int>(a.col_names.size()) + 1)) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            Heading(r.groups.c_str());
            for (const auto& c : a.col_names) NumberCell(c, &t.text_dim);
            for (size_t i = 0; i < a.counts.size(); ++i) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(a.row_names[i].c_str());
                for (size_t j = 0; j < a.counts[i].size(); ++j) {
                    ImGui::TableNextColumn();
                    const std::string count = sv::Num(a.counts[i][j]);
                    const std::string expected = " (" + sv::Num(a.expected[i][j]) + ")";
                    const float w = ImGui::CalcTextSize((count + expected).c_str()).x;
                    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetContentRegionAvail().x - w));
                    ImGui::TextUnformatted(count.c_str());
                    ImGui::SameLine(0.0f, 0.0f);
                    ImGui::TextColored(a.expected[i][j] < 5.0 ? t.warning : t.text_faint, "%s", expected.c_str());
                }
            }
            ImGui::EndTable();
        }
        Faint(sv::LeftOut(a));
        return;
    }
    Heading(r.question == Question::Paired ? "THE COLUMNS" : r.question == Question::OneValue ? "THE VALUES" : "THE GROUPS");
    if (BeginTable("##tests_groups", 5)) {
        ImGui::TableSetupColumn("name", ImGuiTableColumnFlags_WidthStretch, 1.4f);
        for (int i = 0; i < 4; ++i) ImGui::TableSetupColumn("figure", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        Headers({r.question == Question::SeveralGroups || (r.question == Question::TwoGroups && !r.two_columns) ? r.groups.c_str() : "",
                 "n", "mean", "SD", "normal? (p)"});
        const double alpha = 1.0 - r.confidence;
        for (const auto& g : a.groups) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(g.name.c_str());
            NumberCell(sv::Num(static_cast<double>(g.n)));
            NumberCell(sv::Num(g.mean));
            NumberCell(sv::Num(g.sd));
            if (!std::isfinite(g.normal_p)) NumberCell("-", &t.text_faint);
            else if (g.normal_p < alpha) NumberCell("no (" + sv::P(g.normal_p) + ")", &t.caution);
            else NumberCell("yes (" + sv::P(g.normal_p) + ")");
        }
        ImGui::EndTable();
    }
    Faint(sv::LeftOut(a));
    RenderBoxes(a, ImGui::GetContentRegionAvail().x);
    if (!a.pairs.empty()) {
        ImGui::Spacing();
        Heading("WHICH PAIRS DIFFER (TUKEY HSD)");
        if (BeginTable("##tests_pairs", 5)) {
            ImGui::TableSetupColumn("pair", ImGuiTableColumnFlags_WidthStretch, 1.8f);
            for (int i = 0; i < 4; ++i) ImGui::TableSetupColumn("figure", ImGuiTableColumnFlags_WidthStretch, 1.0f);
            Headers({"pair", "difference", sv::Level(r.confidence).c_str(), "p", ""});
            for (const auto& pr : a.pairs) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted((a.groups[pr.a].name + " - " + a.groups[pr.b].name).c_str());
                NumberCell(sv::Num(pr.difference));
                NumberCell(sv::Range(pr.ci_low, pr.ci_high));
                NumberCell(sv::PairP(pr.p));
                const bool differ = std::string(sv::PairVerdict(pr, r.confidence)) == "differ";
                NumberCell(sv::PairVerdict(pr, r.confidence), differ ? &t.success : &t.text_dim);
            }
            ImGui::EndTable();
        }
    }
}

// One box per group on a shared scale: whiskers min .. max, box Q1 .. Q3, the median.
void TestsView::RenderBoxes(const stats::TestAnswer& a, float width) {
    const ui::Tokens& t = ui::CurrentTokens();
    std::vector<const stats::GroupFigures*> boxes;
    for (const auto& g : a.groups)
        if (std::isfinite(g.min) && std::isfinite(g.max)) boxes.push_back(&g);
    if (a.request.question == Question::Paired && boxes.size() == 3) boxes.pop_back();  // the differences have their own scale
    if (boxes.empty()) return;
    double lo = boxes[0]->min, hi = boxes[0]->max;
    for (const auto* g : boxes) {
        lo = std::min(lo, g->min);
        hi = std::max(hi, g->max);
    }
    if (!(hi > lo)) return;
    float label_w = 0.0f;
    for (const auto* g : boxes) label_w = std::max(label_w, ImGui::CalcTextSize(g->name.c_str()).x);
    label_w = std::min(label_w + 10.0f, width * 0.35f);
    const float row_h = 22.0f, plot_w = std::max(60.0f, width - label_w - 8.0f);
    const ImVec2 at = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(width, row_h * static_cast<float>(boxes.size()) + 4.0f));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const auto px = [&](double v) { return at.x + label_w + plot_w * static_cast<float>((v - lo) / (hi - lo)); };
    for (size_t i = 0; i < boxes.size(); ++i) {
        const auto& g = *boxes[i];
        const float y = at.y + row_h * static_cast<float>(i);
        const ImVec4 colour = i % 2 ? t.caution : t.accent_text;
        const ImVec4 clip(at.x, y, at.x + label_w - 6.0f, y + row_h);
        dl->AddText(nullptr, 0.0f, ImVec2(at.x, y + 3.0f), ui::ToU32(t.text_dim), g.name.c_str(), nullptr, 0.0f, &clip);
        const float mid = y + row_h * 0.5f;
        dl->AddLine(ImVec2(px(g.min), mid), ImVec2(px(g.q1), mid), ui::ToU32(colour), 1.0f);
        dl->AddLine(ImVec2(px(g.q3), mid), ImVec2(px(g.max), mid), ui::ToU32(colour), 1.0f);
        ImVec4 fill = colour;
        fill.w = 0.45f;
        dl->AddRectFilled(ImVec2(px(g.q1), y + 4.0f), ImVec2(std::max(px(g.q3), px(g.q1) + 2.0f), y + row_h - 4.0f), ui::ToU32(fill), 3.0f);
        dl->AddLine(ImVec2(px(g.median), y + 4.0f), ImVec2(px(g.median), y + row_h - 4.0f), ui::ToU32(t.text_bright), 2.0f);
    }
    const std::string l = sv::Num(lo), h = sv::Num(hi);
    const float x0 = ImGui::GetCursorPosX();
    ImGui::SetCursorPosX(x0 + label_w);
    ImGui::TextColored(t.text_faint, "%s", l.c_str());
    ImGui::SameLine();
    ImGui::SetCursorPosX(x0 + std::max(label_w, width - ImGui::CalcTextSize(h.c_str()).x));
    ImGui::TextColored(t.text_faint, "%s", h.c_str());
}

void TestsView::RenderAnswer(const stats::TestAnswer& a) {
    const ui::Tokens& t = ui::CurrentTokens();
    Heading("ANSWER");
    ExploreFigures("##tests_figures", AnswerRows(a));
    const sv::Sentence s = sv::Answer(a);
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(s.significant ? t.success : t.text, "%s", s.text.c_str());
    ImGui::PopTextWrapPos();
    const auto checks = sv::Checks(a);
    if (!checks.empty()) {
        ImGui::Spacing();
        Heading("CHECKS");
        ImGui::PushTextWrapPos(0.0f);
        for (const auto& c : checks) {
            ImGui::Bullet();
            ImGui::SameLine();
            ImGui::TextColored(t.text_dim, "%s", c.c_str());
        }
        ImGui::PopTextWrapPos();
    }
    ImGui::Spacing();
    if (ui::SecondaryButton("Copy")) {
        std::string text = s.text + "\n";
        for (const auto& [k, v] : AnswerRows(a)) text += k + ": " + v + "\n";
        ImGui::SetClipboardText(text.c_str());
        message_ = "Copied the answer.";
        message_error_ = false;
    }
    ImGui::SameLine();
    if (ui::SecondaryButton("Export CSV...")) Export(a);
    if (keep_as_node) {
        ImGui::SameLine();
        if (ui::LinkButton("Keep as a node")) keep_as_node(a.request);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Adds a Hypothesis Test node with these settings after the data in use: the test runs with the graph");
    }
}

void TestsView::Export(const stats::TestAnswer& a) {
    const std::string name = a.request.values.empty() ? "test.csv" : a.request.values + "_test.csv";
    if (const auto path = FileDialogs::SaveFile("Export test", {{"CSV", "csv"}}, nullptr, name.c_str())) {
        std::ofstream out(*path, std::ios::binary);
        out << sv::TestCsv(a);
        message_ = out ? "Saved " + *path : "Could not write " + *path;
        message_error_ = !out;
    }
}

}  // namespace cyxwiz
