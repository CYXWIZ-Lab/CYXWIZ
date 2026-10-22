#include "folds_view.h"

#include "../../core/file_dialogs.h"
#include "../../core/project_manager.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_fonts.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"

#include <imgui.h>
#include <implot.h>

#include <algorithm>
#include <cfloat>
#include <filesystem>
#include <fstream>

namespace cyxwiz {

namespace {

void Wrapped(const ImVec4& colour, const std::string& text) {
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(colour, "%s", text.c_str());
    ImGui::PopTextWrapPos();
}

void Label(const char* text) {
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(ui::CurrentTokens().text_dim, "%s", text);
    ImGui::SameLine();
}

}  // namespace

FoldsView::FoldsView(Mode mode) : mode_(mode) {}

bool FoldsView::IsRunning() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return running_;
}

void FoldsView::Started() {
    std::lock_guard<std::mutex> lock(mutex_);
    running_ = true;
    started_ = std::chrono::steady_clock::now();
    progress_ = {};
    live_.clear();
    note_.clear();
}

void FoldsView::Progress(const CrossValidationProgress& progress) {
    std::lock_guard<std::mutex> lock(mutex_);
    progress_ = progress;
}

void FoldsView::FoldDone(const CrossValidationRun& done) {
    std::lock_guard<std::mutex> lock(mutex_);
    live_.push_back(done);
}

void FoldsView::Finished(CrossValidationOutcome outcome) {
    std::lock_guard<std::mutex> lock(mutex_);
    running_ = false;
    outcome_ = std::move(outcome);
    ++outcome_version_;
}

CrossValidationSettings FoldsView::Settings() {
    CrossValidationSettings s;
    s.folds = folds_;
    s.stratified = stratified_ == 0;
    s.seed = static_cast<uint32_t>(std::max(seed_, 0));
    if (mode_ == Mode::LearningCurve) folds_view::ParseShares(shares_text_, s.train_shares, shares_error_);
    return s;
}

void FoldsView::Render() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        RenderLocked();
    }
    // Run and Cancel call into MainWindow, which reports back into this view:
    // after the lock is let go.
    if (pending_run_) {
        const CrossValidationSettings settings = *pending_run_;
        pending_run_.reset();
        if (run) run(settings);
    }
    if (pending_cancel_) {
        pending_cancel_ = false;
        if (cancel) cancel();
    }
}

void FoldsView::RenderLocked() {
    const ui::Tokens& t = ui::CurrentTokens();
    Wrapped(t.text_dim, mode_ == Mode::CrossValidation
                            ? "Trains the open graph once per fold with fresh weights, each time on the other folds, and scores "
                              "the fold it did not see. Epochs, batch size, optimiser and device come from the graph; the "
                              "graph's own test rows stay out."
                            : "Trains the open graph on shares of each fold's training rows and scores those rows and the fold, "
                              "to show whether more rows would help. Loss and accuracy per epoch of a run are in the Training "
                              "Dashboard.");
    RenderSettings(running_);
    if (running_) {
        RenderRunning();
    } else if (outcome_) {
        const CrossValidationOutcome& o = *outcome_;
        if (o.cancelled) {
            Wrapped(t.text_dim, "Cancelled; nothing was kept. Run starts again.");
        } else if (!o.ok) {
            Wrapped(t.error, "Could not finish: " + o.error);
        } else if (mode_ == Mode::CrossValidation) {
            RenderCrossValidation(o);
        } else {
            RenderLearningCurve(o);
        }
    }
    if (!note_.empty()) Wrapped(note_failed_ ? t.error : t.success, note_);
    ImGui::Spacing();
    Wrapped(t.text_faint, "Trains on the selected device; the trained model in use, its checkpoints and the Training Dashboard "
                          "stay as they are. It waits while Train runs.");
}

void FoldsView::RenderSettings(bool running) {
    const ui::Tokens& t = ui::CurrentTokens();
    ui::FlowRow row;
    const float field = ui::ButtonWidth("+", ui::ButtonSize::Small) * 2.0f + 60.0f;
    row.Next(ImGui::CalcTextSize("Folds").x + field);
    Label("Folds");
    ui::NumberField("##folds", &folds_, 2, 20, 1, 40.0f, "Folds, 2 to 20");
    if (mode_ == Mode::CrossValidation) {
        static const char* const kShares[] = {"Yes", "No"};
        row.Next(ImGui::CalcTextSize("Keep class shares Yes No").x + 40.0f);
        Label("Keep class shares");
        ui::SegmentedControl("##stratified", kShares, 2, &stratified_);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Classes: each fold keeps the table's class shares (stratified)");
    }
    row.Next(ImGui::CalcTextSize("Seed").x + 90.0f);
    Label("Seed");
    ui::NumberField("##seed", &seed_, 0, 999999, 0, 70.0f, "Draws the folds (and the learning curve's subsets)");
    if (mode_ == Mode::LearningCurve) {
        row.Next(ImGui::CalcTextSize("Training rows").x + 230.0f);
        Label("Training rows");
        ImGui::SetNextItemWidth(220.0f);
        if (ImGui::InputText("##shares", shares_text_, sizeof(shares_text_))) {
            std::vector<double> shares;
            shares_error_.clear();
            folds_view::ParseShares(shares_text_, shares, shares_error_);
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Shares of each fold's training rows, e.g. 10%%, 25%%, 50%%, 100%%");
    }
    ImGui::Spacing();
    const std::string blocker = running ? std::string("It is running") : run_blocker ? run_blocker() : std::string();
    const std::string why = !blocker.empty() ? blocker : shares_error_;
    if (ui::PrimaryButton(outcome_ && !running ? "Run again" : "Run", why.empty() && run, why.c_str())) {
        CrossValidationSettings settings = Settings();
        if (shares_error_.empty() && run) pending_run_ = settings;
    }
    if (running && cancel) {
        ImGui::SameLine();
        if (ui::SecondaryButton("Cancel")) pending_cancel_ = true;
    }
    if (outcome_ && outcome_->ok && !running) {
        ImGui::SameLine();
        if (ui::SecondaryButton("Export CSV...", true, nullptr, ui::ButtonSize::Small)) Export();
    }
    if (!shares_error_.empty() && mode_ == Mode::LearningCurve) Wrapped(t.error, shares_error_);
    ImGui::Spacing();
}

void FoldsView::RenderRunning() {
    const ui::Tokens& t = ui::CurrentTokens();
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started_).count();
    if (progress_.runs == 0) {
        ImGui::TextColored(t.text_dim, ICON_FA_SPINNER " Compiling the graph and preparing its data...");
        return;
    }
    ImGui::TextColored(t.accent_text, ICON_FA_SPINNER " %s", folds_view::ProgressLine(progress_, elapsed).c_str());
    const float done = (static_cast<float>(progress_.run - 1) +
                        (progress_.epochs > 0 ? static_cast<float>(progress_.epoch) / static_cast<float>(progress_.epochs) : 0.0f)) /
                       static_cast<float>(std::max(1, progress_.runs));
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, t.accent_text);
    ImGui::ProgressBar(std::clamp(done, 0.0f, 1.0f), ImVec2(-FLT_MIN, 0.0f), "");
    ImGui::PopStyleColor();
    if (live_.empty()) return;
    const bool regression = progress_.regression;
    const bool curve = mode_ == Mode::LearningCurve;
    if (ImGui::BeginTable("##live", curve ? 5 : 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit)) {
        ImGui::TableSetupColumn("Fold");
        if (curve) ImGui::TableSetupColumn("training share");
        for (const char* c : {"training rows", "validation rows", regression ? "MAE" : "accuracy"}) ImGui::TableSetupColumn(c);
        ImGui::TableHeadersRow();
        for (const auto& r : live_) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%d", r.fold);
            if (curve) {
                ImGui::TableNextColumn();
                ImGui::Text("%.0f%%", r.share * 100.0);
            }
            ImGui::TableNextColumn();
            ImGui::Text("%lld", static_cast<long long>(r.train_rows));
            ImGui::TableNextColumn();
            ImGui::Text("%lld", static_cast<long long>(r.val_rows));
            ImGui::TableNextColumn();
            if (regression) ImGui::Text("%.4g", r.val.mae);
            else ImGui::Text("%.1f%%", r.val.accuracy * 100.0);
        }
        ImGui::EndTable();
    }
}

void FoldsView::RenderCrossValidation(const CrossValidationOutcome& o) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (drawn_version_ != outcome_version_) {
        drawn_version_ = outcome_version_;
        cv_ = folds_view::BuildCrossValidationView(o, single_test ? single_test() : std::nullopt);
    }
    {
        ui::FontScope heading(ui::Font::Heading);
        ImGui::TextUnformatted(cv_.headline.c_str());
    }
    ui::FlowRow chips;
    for (const auto& pill : cv_.pills) {
        chips.Next(ui::ChipWidth(pill.c_str()));
        ui::Chip(pill.c_str(), t.text_dim);
    }
    Wrapped(t.text, cv_.sentence);
    if (o.test_rows_left_out > 0)
        Wrapped(t.text_dim, std::to_string(o.rows) + " rows used; the graph's " + std::to_string(o.test_rows_left_out) +
                                " test rows were left out.");
    ImGui::Spacing();
    const bool side = ImGui::GetContentRegionAvail().x > 760.0f;
    const ImGuiTableFlags split = ImGuiTableFlags_SizingStretchSame;
    if (side && !ImGui::BeginTable("##cv_split", 2, split)) return;
    if (side) ImGui::TableNextColumn();
    const float plot_h = 220.0f;
    if (ImPlot::BeginPlot("##fold_bars", ImVec2(-1, plot_h), ImPlotFlags_NoMenus | ImPlotFlags_NoLegend)) {
        std::vector<double> xs, ys = cv_.bars;
        for (size_t i = 0; i < ys.size(); ++i) xs.push_back(static_cast<double>(i + 1));
        if (cv_.percent)
            for (double& y : ys) y *= 100.0;
        const auto [lo, hi] = std::minmax_element(ys.begin(), ys.end());
        const double pad = std::max((*hi - *lo) * 0.6, cv_.percent ? 2.0 : std::abs(*hi) * 0.05 + 1e-9);
        ImPlot::SetupAxes("fold", cv_.percent ? (cv_.bars_label + " (%)").c_str() : cv_.bars_label.c_str());
        ImPlot::SetupAxisLimits(ImAxis_X1, 0.4, static_cast<double>(ys.size()) + 0.6, ImPlotCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, std::max(0.0, *lo - pad), cv_.percent ? std::min(100.0, *hi + pad * 0.5) : *hi + pad * 0.5,
                                ImPlotCond_Always);
        ImPlot::SetupAxisTicks(ImAxis_X1, 1.0, static_cast<double>(ys.size()), static_cast<int>(ys.size()));
        ImPlot::SetNextFillStyle(ui::WithAlpha(t.series[0], 0.75f));
        ImPlot::PlotBars("folds", xs.data(), ys.data(), static_cast<int>(ys.size()), 0.6);
        const double mean = cv_.percent ? cv_.bars_mean * 100.0 : cv_.bars_mean;
        ImPlot::SetNextLineStyle(t.caution, 1.5f);
        ImPlot::PlotInfLines("mean", &mean, 1, ImPlotInfLinesFlags_Horizontal);
        ImPlot::EndPlot();
    }
    if (side) ImGui::TableNextColumn();
    const int columns = 3 + static_cast<int>(cv_.columns.size());
    if (ImGui::BeginTable("##folds", columns, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit)) {
        ImGui::TableSetupColumn("Fold");
        ImGui::TableSetupColumn("training rows");
        ImGui::TableSetupColumn("validation rows");
        for (const auto& c : cv_.columns) ImGui::TableSetupColumn(c.c_str());
        ImGui::TableHeadersRow();
        const auto line = [&](const folds_view::FoldRow& r, bool total) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextColored(total ? t.text_dim : t.text, "%s", r.fold.c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(r.train_rows.c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(r.val_rows.c_str());
            for (size_t i = 0; i < r.scores.size(); ++i) {
                ImGui::TableNextColumn();
                const bool watch = r.flagged && i == (cv_.regression ? 0 : 2);
                ImGui::TextColored(watch ? t.warning : t.text, "%s", r.scores[i].c_str());
            }
        };
        for (const auto& r : cv_.rows) line(r, false);
        line(cv_.mean, true);
        ImGui::EndTable();
    }
    if (!cv_.note.empty()) Wrapped(t.caution, cv_.note);
    if (side) ImGui::EndTable();
}

void FoldsView::RenderLearningCurve(const CrossValidationOutcome& o) {
    const ui::Tokens& t = ui::CurrentTokens();
    const auto scores = folds_view::CurveScores(o.regression);
    score_ = std::clamp(score_, 0, static_cast<int>(scores.size()) - 1);
    Label("Score");
    ImGui::SetNextItemWidth(140.0f);
    if (ImGui::BeginCombo("##score", folds_view::CurveScoreName(scores[static_cast<size_t>(score_)]))) {
        for (int i = 0; i < static_cast<int>(scores.size()); ++i)
            if (ImGui::Selectable(folds_view::CurveScoreName(scores[static_cast<size_t>(i)]), i == score_)) score_ = i;
        ImGui::EndCombo();
    }
    if (drawn_version_ != outcome_version_ || drawn_score_ != score_) {
        drawn_version_ = outcome_version_;
        drawn_score_ = score_;
        curve_ = folds_view::BuildLearningCurveView(o, scores[static_cast<size_t>(score_)]);
    }
    const bool side = ImGui::GetContentRegionAvail().x > 760.0f;
    if (side && !ImGui::BeginTable("##curve_split", 2, ImGuiTableFlags_SizingStretchProp)) return;
    if (side) {
        ImGui::TableSetupColumn("plot", 0, 1.3f);
        ImGui::TableSetupColumn("table", 0, 1.0f);
        ImGui::TableNextColumn();
    }
    const double k = curve_.percent ? 100.0 : 1.0;
    std::vector<double> x, train, val, train_lo, train_hi, val_lo, val_hi;
    for (const auto& p : curve_.points) {
        x.push_back(p.rows);
        train.push_back(p.train_mean * k);
        val.push_back(p.val_mean * k);
        // A share's band stays inside 0 .. 100%.
        const auto clip = [&](double y) { return curve_.percent ? std::clamp(y, 0.0, 100.0) : y; };
        train_lo.push_back(clip((p.train_mean - p.train_sd) * k));
        train_hi.push_back(clip((p.train_mean + p.train_sd) * k));
        val_lo.push_back(clip((p.val_mean - p.val_sd) * k));
        val_hi.push_back(clip((p.val_mean + p.val_sd) * k));
    }
    const std::string y_label = curve_.score_name + (curve_.percent ? " (%)" : "");
    if (ImPlot::BeginPlot("##curve", ImVec2(-1, 260.0f), ImPlotFlags_NoMenus)) {
        ImPlot::SetupAxes("training rows", y_label.c_str(), ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        const int n = static_cast<int>(x.size());
        ImPlot::SetNextFillStyle(t.series[0], 0.18f);
        ImPlot::PlotShaded("##train_band", x.data(), train_lo.data(), train_hi.data(), n);
        ImPlot::SetNextFillStyle(t.series[2], 0.18f);
        ImPlot::PlotShaded("##val_band", x.data(), val_lo.data(), val_hi.data(), n);
        ImPlot::SetNextLineStyle(t.series[0], 2.0f);
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 4.0f);
        ImPlot::PlotLine("training", x.data(), train.data(), n);
        ImPlot::SetNextLineStyle(t.series[2], 2.0f);
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 4.0f);
        ImPlot::PlotLine("validation", x.data(), val.data(), n);
        ImPlot::EndPlot();
    }
    ImGui::TextColored(t.text_faint, "Mean of %d folds; the band is one standard deviation.", o.folds);
    if (side) ImGui::TableNextColumn();
    if (ImGui::BeginTable("##curve_table", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit)) {
        for (const char* c : {"rows", "training", "validation", "gap"}) ImGui::TableSetupColumn(c);
        ImGui::TableHeadersRow();
        for (const auto& r : curve_.table) {
            ImGui::TableNextRow();
            for (const auto& cell : r) {
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(cell.c_str());
            }
        }
        ImGui::EndTable();
    }
    Wrapped(t.text, curve_.sentence);
    Wrapped(t.text_dim, curve_.advice);
    if (side) ImGui::EndTable();
}

void FoldsView::Export() {
    if (!outcome_) return;
    const std::string root = ProjectManager::Instance().GetProjectRoot();
    const std::string folder = root.empty() ? std::string() : (std::filesystem::path(root) / "exports").string();
    const char* name = mode_ == Mode::CrossValidation ? "cross_validation.csv" : "learning_curve.csv";
    const auto path = FileDialogs::SaveFile("Export", {{"CSV Files", "csv"}, {"All Files", "*"}}, folder.empty() ? nullptr : folder.c_str(), name);
    if (!path) return;
    std::ofstream out(*path, std::ios::binary);
    out << folds_view::OutcomeCsv(*outcome_);
    note_failed_ = !out.good();
    note_ = note_failed_ ? "Could not write " + *path : "Saved " + *path;
}

}  // namespace cyxwiz
