// Training Dashboard: word-vector runs (TOFIX152, mockup approved
// 2026-10-14). word2vec and GloVe training of a Word Embeddings node shows
// as its own run behind the Run switcher: status, steps, cards, the cost or
// loss per epoch, the result, a nearest-words check and the session's runs.
// The model run's view (training_plot_panel.cpp) is left as it was.

#include "training_plot_panel.h"

#include "training_dashboard_look.h"
#include "../icons.h"

#include <imgui.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>

namespace cyxwiz {

using namespace dash;

namespace {

constexpr int kWordVectorColour = 4;
constexpr float kWordVectorChartHeight = 260.0f;

double SteadySeconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

ImVec4 PillColour(const std::string& pill, const DashColors& c) {
    if (pill == "Preparing") return c.info;
    if (pill == "Completed" || pill == "Training") return c.success;
    if (pill == "Failed") return c.error;
    return c.caution;  // Cancelled
}

std::string FileName(const std::string& path) {
    const std::string name = std::filesystem::path(path).filename().string();
    return name.empty() ? path : name;
}

bool WriteText(const std::string& path, const std::string& text, std::string& message) {
    std::error_code error;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path(), error);
    std::ofstream out(path, std::ios::binary);
    out << text;
    if (!out) {
        message = "Could not write " + path;
        return false;
    }
    message = "Saved " + path;
    return true;
}

}  // namespace

int TrainingPlotPanel::RecordWordVectorFigures(const OperatorFigures& figures) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    const size_t before = word_vector_runs_.Runs().size();
    const WordVectorRun* run = word_vector_runs_.Record(figures, SteadySeconds());
    if (!run) return 0;
    if (run->state == WordVectorRun::State::Completed && run->used_by.empty() && word_vector_hooks_.used_by) {
        const int number = run->number;
        word_vector_runs_.SetUsedBy(number, word_vector_hooks_.used_by(run->node_id, run->saved));
        run = word_vector_runs_.Find(number);
    }
    if (word_vector_runs_.Runs().size() > before) {
        // The dashboard opens on the run that just started.
        shown_run_ = run->number;
        shown_run_chosen_ = false;
    }
    return run->number;
}

void TrainingPlotPanel::RecordWordVectorProgress(int node_id, float step_share, const std::string& message) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    word_vector_runs_.Progress(node_id, step_share, message);
}

void TrainingPlotPanel::EndWordVectorRuns(int node_id, bool cancelled, const std::string& error) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (node_id < 0) {
        word_vector_runs_.EndAllLive(cancelled, error);
    } else {
        word_vector_runs_.EndLive(node_id, cancelled, error);
    }
}

std::string TrainingPlotPanel::WordVectorRunLine(int node_id) const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    const auto& runs = word_vector_runs_.Runs();
    for (auto it = runs.rbegin(); it != runs.rend(); ++it) {
        if (it->node_id == node_id) return it->Live() ? cyxwiz::WordVectorRunLine(*it, SteadySeconds()) : std::string();
    }
    return {};
}

void TrainingPlotPanel::ShowWordVectorRun(int node_id) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    const auto& runs = word_vector_runs_.Runs();
    for (auto it = runs.rbegin(); it != runs.rend(); ++it) {
        if (it->node_id != node_id) continue;
        shown_run_ = it->number;
        shown_run_chosen_ = true;
        break;
    }
    visible_ = true;
    focus_requested_ = true;
}

bool TrainingPlotPanel::HasWordVectorRun(int node_id) const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    for (const auto& run : word_vector_runs_.Runs()) {
        if (run.node_id == node_id) return true;
    }
    return false;
}

// The model run's state in a few words, for its tab.
std::string TrainingPlotPanel::ModelRunSummary() const {
    char text[96];
    if (is_preparing_) return "preparing";
    if (is_training_) {
        std::snprintf(text, sizeof(text), "epoch %d / %d", current_epoch_, total_epochs_);
        return text;
    }
    if (total_training_time_ > 0.0f) {
        const char* how = terminal_status_ == "completed" || terminal_status_.empty() ? "finished" : "stopped";
        std::snprintf(text, sizeof(text), "%s at epoch %d / %d", how, last_executed_epoch_, total_epochs_);
        return text;
    }
    return train_loss_.values.empty() ? "no run yet" : "";
}

void TrainingPlotPanel::RenderRunSwitcher() {
    const DashColors c = CurrentDashColors();
    const auto& runs = word_vector_runs_.Runs();
    // Back to the model once its training starts, unless a run was picked.
    if (!shown_run_chosen_ && shown_run_ != 0 && is_training_ && !is_preparing_) {
        const WordVectorRun* shown = word_vector_runs_.Find(shown_run_);
        if (!shown || !shown->Live()) shown_run_ = 0;
    }
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(c.muted, "Run");
    ImGui::SameLine(0.0f, 10.0f);
    const std::string model = ModelRunSummary();
    if (DashRunTab("##run_model", "Model", model.c_str(), c.info, shown_run_ == 0, c)) {
        shown_run_ = 0;
        shown_run_chosen_ = true;
    }
    // The latest three runs, and the shown one when it is older.
    const size_t first = runs.size() > 3 ? runs.size() - 3 : 0;
    for (size_t i = 0; i < runs.size(); ++i) {
        if (i < first && runs[i].number != shown_run_) continue;
        const WordVectorRunView view = BuildWordVectorRunView(runs[i], SteadySeconds());
        std::string label = view.switcher_label;
        if (runs.size() > 1) label += " #" + std::to_string(runs[i].number);
        const std::string id = "##run_" + std::to_string(runs[i].number);
        ImGui::SameLine(0.0f, 6.0f);
        if (ImGui::GetContentRegionAvail().x < ImGui::CalcTextSize((label + view.switcher_sub).c_str()).x + 60.0f) {
            ImGui::NewLine();
        }
        if (DashRunTab(id.c_str(), label.c_str(), view.switcher_sub.c_str(), c.accent_text,
                       shown_run_ == runs[i].number, c)) {
            shown_run_ = runs[i].number;
            shown_run_chosen_ = true;
        }
    }
    ImGui::Spacing();
}

void TrainingPlotPanel::RenderWordVectorRun(const WordVectorRun& run) {
    const WordVectorRunView view = BuildWordVectorRunView(run, SteadySeconds());
    RenderWordVectorStatus(run, view);
    RenderWordVectorSteps(view);
    RenderWordVectorCards(view);
    RenderWordVectorChart(run, view);
    if (run.state == WordVectorRun::State::Completed) RenderWordVectorResult(run, view);
    RenderWordVectorRunsTable();
    RenderActiveTaskSummary();
}

void TrainingPlotPanel::RenderWordVectorStatus(const WordVectorRun& run, const WordVectorRunView& view) {
    const DashColors c = CurrentDashColors();
    const ImVec4 colour = PillColour(view.pill, c);
    if (BeginDashCard("##wv_status", c)) {
        ImGui::AlignTextToFramePadding();
        DashPill(view.pill.c_str(), colour);
        ImGui::SameLine(0.0f, 10.0f);
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(c.text, "%s", view.title.c_str());
        ImGui::SameLine(0.0f, 10.0f);
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(c.muted, "%s", view.subtitle.c_str());

        // Cancel (live) or Open node dialog, then Actions, on the right.
        const char* first = run.Live() ? "Cancel" : "Open node dialog";
        const char* actions = "Actions " ICON_FA_CHEVRON_DOWN;
        const float needed = DashButtonWidth(first) + DashButtonWidth(actions) + ImGui::GetStyle().ItemSpacing.x;
        ImGui::SameLine();
        if (ImGui::GetContentRegionAvail().x < needed) ImGui::NewLine();
        ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(), ImGui::GetWindowContentRegionMax().x - needed));
        if (DashButton(first, DashButtonKind::Secondary, c)) {
            if (run.Live() && word_vector_hooks_.cancel) word_vector_hooks_.cancel(run.node_id);
            if (!run.Live() && word_vector_hooks_.open_node_dialog) word_vector_hooks_.open_node_dialog(run.node_id);
        }
        ImGui::SameLine();
        if (DashButton(actions, DashButtonKind::Secondary, c)) ImGui::OpenPopup("##wv_actions");
        if (ImGui::BeginPopup("##wv_actions")) {
            const std::string folder = word_vector_hooks_.export_folder ? word_vector_hooks_.export_folder() : "";
            if (ImGui::MenuItem("Export curve CSV", nullptr, false, !run.figures.empty() && !folder.empty())) {
                WriteText((std::filesystem::path(folder) /
                           ("word_vector_run_" + std::to_string(run.number) + "_" + view.figure_name + ".csv"))
                              .string(),
                          WordVectorCurveCsv(run), word_vector_export_message_);
            }
            if (ImGui::MenuItem("Export runs CSV", nullptr, false, !folder.empty())) {
                WriteText((std::filesystem::path(folder) / "word_vector_runs.csv").string(),
                          WordVectorRunsCsv(word_vector_runs_.Runs()), word_vector_export_message_);
            }
            if (ImGui::MenuItem("Open node dialog", nullptr, false,
                                static_cast<bool>(word_vector_hooks_.open_node_dialog))) {
                word_vector_hooks_.open_node_dialog(run.node_id);
            }
            ImGui::EndPopup();
        }
        DashProgress(view.progress, colour, c);
        if (!view.error.empty()) {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(c.caution, "%s", view.error.c_str());
            ImGui::PopTextWrapPos();
        }
        if (!word_vector_export_message_.empty()) {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(c.faint, "%s", word_vector_export_message_.c_str());
            ImGui::PopTextWrapPos();
        }
    }
    EndDashCard();
}

void TrainingPlotPanel::RenderWordVectorSteps(const WordVectorRunView& view) {
    const DashColors c = CurrentDashColors();
    const int count = static_cast<int>(view.steps.size());
    const float available = ImGui::GetContentRegionAvail().x;
    const int columns = std::clamp(static_cast<int>(available / 170.0f), 1, std::max(1, count));
    ImGui::Spacing();
    ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(4.0f, 4.0f));
    if (ImGui::BeginTable("##wv_steps", columns, ImGuiTableFlags_SizingStretchSame | ImGuiTableFlags_NoPadOuterX)) {
        for (int i = 0; i < count; ++i) {
            const auto& step = view.steps[static_cast<size_t>(i)];
            ImGui::TableNextColumn();
            ImGui::PushID(i);
            if (BeginDashCard("##step", c)) {
                using State = WordVectorRunView::Step::State;
                const char* icon = step.state == State::Done  ? ICON_FA_CIRCLE_CHECK
                                   : step.state == State::Now ? ICON_FA_SPINNER
                                                              : ICON_FA_CIRCLE;
                const ImVec4 icon_colour = step.state == State::Done  ? c.success
                                           : step.state == State::Now ? c.accent_text
                                                                      : c.faint;
                ImGui::TextColored(icon_colour, "%s", icon);
                ImGui::SameLine(0.0f, 6.0f);
                ImGui::TextColored(step.state == State::Todo ? c.faint : c.text, "%s", step.name.c_str());
                ImGui::PushTextWrapPos(0.0f);
                ImGui::TextColored(c.faint, "%s", step.detail.empty() ? " " : step.detail.c_str());
                ImGui::PopTextWrapPos();
            }
            EndDashCard();
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    ImGui::PopStyleVar();
}

void TrainingPlotPanel::RenderWordVectorCards(const WordVectorRunView& view) {
    const DashColors c = CurrentDashColors();
    const int count = static_cast<int>(view.cards.size());
    const float available = ImGui::GetContentRegionAvail().x;
    const int columns = std::clamp(static_cast<int>(available / 190.0f), 1, std::max(1, count));
    ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(4.0f, 4.0f));
    if (ImGui::BeginTable("##wv_cards", columns, ImGuiTableFlags_SizingStretchSame | ImGuiTableFlags_NoPadOuterX)) {
        const ImVec4 dots[4] = {c.accent_text, c.warning, c.info, c.info};
        for (int i = 0; i < count; ++i) {
            const auto& card = view.cards[static_cast<size_t>(i)];
            ImGui::TableNextColumn();
            std::string line1 = card.line1;
            ImVec4 line1_colour = c.muted;
            if (card.trend != 0) {
                line1 = std::string(card.trend < 0 ? ICON_FA_CARET_DOWN : ICON_FA_CARET_UP) + " " + line1;
                line1_colour = card.trend < 0 ? c.success : c.caution;
            }
            const std::string id = "##wv_card_" + std::to_string(i);
            DashKpiCard(id.c_str(), card.label.c_str(), dots[i % 4], card.value.c_str(), "", line1.c_str(),
                        line1_colour, card.line2.c_str(), c.faint, c);
        }
        ImGui::EndTable();
    }
    ImGui::PopStyleVar();
}

void TrainingPlotPanel::RenderWordVectorChart(const WordVectorRun& run, const WordVectorRunView& view) {
    const DashColors c = CurrentDashColors();
    ImGui::Spacing();
    if (BeginDashCard("##wv_chart", c)) {
        DashCardTitle(view.chart_title.c_str(), "hover a point for its value", c);
        if (run.figures.empty()) {
            ImGui::Dummy(ImVec2(0.0f, 40.0f));
            const float width = ImGui::CalcTextSize(view.chart_empty.c_str()).x;
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, (ImGui::GetContentRegionAvail().x - width) * 0.5f));
            ImGui::TextColored(c.faint, "%s", view.chart_empty.c_str());
            ImGui::Dummy(ImVec2(0.0f, 40.0f));
        } else {
            WordVectorChart& chart = word_vector_chart_;
            if (!chart.view) chart.view = std::make_unique<plot::PlotView>("word_vector_curve");
            if (chart.run != run.number || chart.version != word_vector_runs_.Version()) {
                plot::Prepared p;
                p.spec.kind = plot::Kind::Line;
                p.spec.title = view.chart_title;
                p.spec.x_label = "Epoch";
                p.spec.y_label = run.GloVe() ? "Cost" : "Loss";
                p.spec.legend = false;
                plot::Series s;
                s.label = run.GloVe() ? "cost per word pair" : "loss per example";
                for (size_t i = 0; i < run.figures.size(); ++i) {
                    s.x.push_back(static_cast<double>(i + 1));
                    s.y.push_back(run.figures[i]);
                }
                s.x_sorted = true;
                s.colour = kWordVectorColour;
                s.markers = true;
                p.series.push_back(std::move(s));
                p.label = plot::DataLabel{plot::DataLabel::State::Exact, run.figures.size(), run.figures.size()};
                const bool fit = chart.run != run.number;
                chart.view->SetData(std::move(p));
                if (fit) chart.view->RequestFit();
                chart.run = run.number;
                chart.version = word_vector_runs_.Version();
            }
            plot::PlotView::Options o;
            o.export_name = "word_vector_run_" + std::to_string(run.number) + "_" + view.figure_name;
            // Every planned epoch on the axis, so the curve fills in as it runs.
            o.x_range = {true, 0.5, std::max<double>(run.epochs, static_cast<double>(run.figures.size())) + 0.5,
                         true};
            // The values' own range: costs of 0.006 to 0.034 sit flat on a fit around 0.
            const auto [low, high] = std::minmax_element(run.figures.begin(), run.figures.end());
            const double pad = *high > *low ? (*high - *low) * 0.08 : std::abs(*high) * 0.1 + 1e-9;
            o.y_range = {true, *low > 0.0 ? std::max(*low - pad, *low * 0.5) : *low - pad, *high + pad};
            chart.view->Draw(ImVec2(-1.0f, kWordVectorChartHeight), o);
        }
    }
    EndDashCard();
}

void TrainingPlotPanel::RenderWordVectorResult(const WordVectorRun& run, const WordVectorRunView& view) {
    const DashColors c = CurrentDashColors();
    ImGui::Spacing();
    const bool wide = ImGui::GetContentRegionAvail().x >= 760.0f;
    if (!ImGui::BeginTable("##wv_result_row", wide ? 2 : 1,
                           ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_NoPadOuterX)) {
        return;
    }
    if (wide) {
        ImGui::TableSetupColumn("result", ImGuiTableColumnFlags_WidthStretch, 1.4f);
        ImGui::TableSetupColumn("check", ImGuiTableColumnFlags_WidthStretch, 1.0f);
    }
    ImGui::TableNextColumn();
    if (BeginDashCard("##wv_result", c)) {
        DashCardTitle("Result", "", c);
        if (ImGui::BeginTable("##wv_result_rows", 2, ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("Lined up").x + 16.0f);
            ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch);
            for (const auto& [label, value] : view.result) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(c.muted, "%s", label.c_str());
                ImGui::TableNextColumn();
                ImGui::PushTextWrapPos(0.0f);
                ImGui::TextColored(c.text, "%s", value.c_str());
                ImGui::PopTextWrapPos();
            }
            ImGui::EndTable();
        }
    }
    EndDashCard();
    ImGui::TableNextColumn();
    RenderWordVectorQuickCheck(run);
    ImGui::EndTable();
}

void TrainingPlotPanel::RenderWordVectorQuickCheck(const WordVectorRun& run) {
    const DashColors c = CurrentDashColors();
    QuickCheck& check = quick_checks_[run.number];
    if (!check.looked_up && word_vector_hooks_.nearest && !run.saved.empty()) {
        // Starts on a frequent word of the vocabulary.
        if (check.word.empty() && word_vector_hooks_.words) {
            check.word = WordVectorQuickWord(word_vector_hooks_.words(run.saved));
            std::snprintf(check.input, sizeof(check.input), "%s", check.word.c_str());
        }
        if (!check.word.empty()) check.rows = word_vector_hooks_.nearest(run.saved, check.word, check.note);
        check.looked_up = true;
    }
    if (BeginDashCard("##wv_check", c)) {
        const std::string title = "Quick check \xC2\xB7 nearest to \"" + check.word + "\"";
        DashCardTitle(check.word.empty() ? "Quick check" : title.c_str(), "", c);
        ImGui::SetNextItemWidth(std::min(240.0f, ImGui::GetContentRegionAvail().x));
        if (ImGui::InputTextWithHint("##wv_word", "another word, then Enter", check.input, sizeof(check.input),
                                     ImGuiInputTextFlags_EnterReturnsTrue) &&
            word_vector_hooks_.nearest && !run.saved.empty()) {
            check.word = check.input;
            check.note.clear();
            check.rows = word_vector_hooks_.nearest(run.saved, check.word, check.note);
        }
        if (run.saved.empty()) check.note = "The vectors were not saved to a file.";
        if (!check.note.empty()) {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(c.caution, "%s", check.note.c_str());
            ImGui::PopTextWrapPos();
        }
        if (!check.rows.empty() && ImGui::BeginTable("##wv_nearest", 2, ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("word", ImGuiTableColumnFlags_WidthStretch, 3.0f);
            ImGui::TableSetupColumn("cosine", ImGuiTableColumnFlags_WidthStretch, 1.0f);
            for (size_t i = 0; i < check.rows.size() && i < 6; ++i) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(c.text, "%s", check.rows[i].first.c_str());
                ImGui::TableNextColumn();
                ImGui::TextColored(c.muted, "%.3f", check.rows[i].second);
            }
            ImGui::EndTable();
        }
        if (word_vector_hooks_.open_node_dialog) {
            ImGui::PushStyleColor(ImGuiCol_Text, c.accent_text);
            ImGui::TextUnformatted("More words in the node dialog");
            ImGui::PopStyleColor();
            if (ImGui::IsItemHovered()) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
            if (ImGui::IsItemClicked()) word_vector_hooks_.open_node_dialog(run.node_id);
        }
    }
    EndDashCard();
}

void TrainingPlotPanel::RenderWordVectorRunsTable() {
    const DashColors c = CurrentDashColors();
    const auto& runs = word_vector_runs_.Runs();
    ImGui::Spacing();
    if (BeginDashCard("##wv_runs", c)) {
        DashCardTitle("Word vector runs", "this session", c);
        const char* headers[] = {"Run", "Method", "Settings", "Words", "Per epoch", "First to last", "Time",
                                 "Saved"};
        // Columns as wide as their text; a narrow dashboard scrolls the table.
        const float height = ImGui::GetTextLineHeightWithSpacing() * (static_cast<float>(runs.size()) + 1.0f) +
                             ImGui::GetStyle().ScrollbarSize + 8.0f;
        if (ImGui::BeginTable("##wv_runs_table", 8,
                              ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_ScrollX,
                              ImVec2(0.0f, std::min(height, 320.0f)))) {
            for (int i = 0; i < 8; ++i) ImGui::TableSetupColumn(headers[i]);
            ImGui::TableHeadersRow();
            // Newest first; a row shows its run.
            for (auto it = runs.rbegin(); it != runs.rend(); ++it) {
                const WordVectorRunRow row = BuildWordVectorRunRow(*it);
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::PushID(it->number);
                if (ImGui::Selectable(row.run.c_str(), shown_run_ == it->number,
                                      ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap)) {
                    shown_run_ = it->number;
                    shown_run_chosen_ = true;
                }
                ImGui::PopID();
                for (const std::string* cell : {&row.method, &row.settings, &row.words, &row.per_epoch, &row.figures,
                                                &row.time, &row.saved}) {
                    ImGui::TableNextColumn();
                    ImGui::TextColored(c.text, "%s", cell->c_str());
                }
            }
            ImGui::EndTable();
        }
        ImGui::TextColored(c.faint, "Model runs keep their own Run Comparison table; the two kinds are not mixed.");
    }
    EndDashCard();
}

}  // namespace cyxwiz
