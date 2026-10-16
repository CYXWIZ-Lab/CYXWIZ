#include "test_results_panel.h"
#include "../../core/compiled_node_presentation.h"
#include "../../core/file_dialogs.h"
#include "../../core/project_manager.h"
#include "../../core/test_results_export.h"
#include "../ui_buttons.h"
#include "../ui_fonts.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <filesystem>

namespace cyxwiz {

namespace {

ImVec4 ToneColour(test_report::Tone tone) {
    const ui::Tokens& t = ui::CurrentTokens();
    switch (tone) {
        case test_report::Tone::Ok: return t.success;
        case test_report::Tone::Warning: return t.warning;
        case test_report::Tone::Error: return t.error;
        case test_report::Tone::Info: return t.info;
        case test_report::Tone::Neutral: return t.text_dim;
    }
    return t.text_dim;
}

ImVec4 MetricColour(float value) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (value >= 0.9f) return t.success;
    if (value >= 0.7f) return t.warning;
    if (value >= 0.5f) return t.caution;
    return t.error;
}

void Wrapped(const ImVec4& colour, const std::string& text) {
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(colour, "%s", text.c_str());
    ImGui::PopTextWrapPos();
}

// Where exports go first: the project's exports folder.
std::string ExportsFolder() {
    const std::string root = ProjectManager::Instance().GetProjectRoot();
    return root.empty() ? std::string() : (std::filesystem::path(root) / "exports").string();
}

}  // namespace

TestResultsPanel::TestResultsPanel()
    : Panel("Test Results", false)  // Hidden by default
{
}

void TestResultsPanel::SetResults(const TestingMetrics& results) {
    std::lock_guard<std::mutex> lock(results_mutex_);
    if (test_report::IsFailedRun(results)) {
        // The last successful results stay (and stay what Comparison compares).
        failure_message_ = results.status_message;
        spdlog::warn("TestResultsPanel: the test failed: {}", results.status_message);
    } else {
        if (has_results_) {
            previous_results_ = results_;
            has_previous_ = true;
        }
        results_ = results;
        has_results_ = true;
        failure_message_.clear();
    }
    export_note_.clear();
    view_dirty_ = true;
}

void TestResultsPanel::Clear() {
    std::lock_guard<std::mutex> lock(results_mutex_);
    results_ = TestingMetrics();
    previous_results_ = TestingMetrics();
    has_results_ = false;
    has_previous_ = false;
    failure_message_.clear();
    export_note_.clear();
    view_dirty_ = true;
}

void TestResultsPanel::ExportReport() {
    std::lock_guard<std::mutex> lock(results_mutex_);
    ExportToJSON();
}

void TestResultsPanel::ShowComparison() {
    Show();
    select_comparison_tab_ = true;
}

void TestResultsPanel::Render() {
    if (!visible_) return;
    ImGui::SetNextWindowSize(ImVec2(760, 560), ImGuiCond_FirstUseEver);
    if (ImGui::Begin(name_.c_str(), &visible_)) {
        std::lock_guard<std::mutex> lock(results_mutex_);
        if (view_dirty_) {
            view_ = test_report::BuildView(results_, has_previous_ ? &previous_results_ : nullptr);
            view_dirty_ = false;
        }
        const ui::Tokens& t = ui::CurrentTokens();
        if (!failure_message_.empty()) {
            ImGui::TextColored(t.error, ICON_FA_CIRCLE_XMARK " The test failed");
            Wrapped(t.text, failure_message_);
            Wrapped(t.text_dim, has_results_ ? "The last successful run's results are below."
                                             : "Task View shows the failed testing task; engine_log.txt has the details.");
            ImGui::Dummy(ImVec2(0.0f, t.space_sm));
        }
        if (!has_results_) {
            if (failure_message_.empty())
                ui::EmptyState(ICON_FA_GAUGE, "No test results yet",
                               "Train the graph, then Train > Test > Run Test (F8).");
        } else {
            RenderHeader();
            const float footer_h = ui::ButtonHeight(ui::ButtonSize::Regular) + ImGui::GetStyle().ItemSpacing.y * 2.0f;
            if (ImGui::BeginTabBar("##test_results_tabs", ImGuiTabBarFlags_FittingPolicyScroll)) {
                const auto tab = [&](const char* label, void (TestResultsPanel::*body)(), ImGuiTabItemFlags flags = 0) {
                    if (ImGui::BeginTabItem(label, nullptr, flags)) {
                        ImGui::BeginChild("##tab_body", ImVec2(0.0f, -footer_h));
                        (this->*body)();
                        ImGui::EndChild();
                        ImGui::EndTabItem();
                    }
                };
                tab("Overview###overview", &TestResultsPanel::RenderOverviewTab);
                if (view_.classification) {
                    tab("Confusion matrix###matrix", &TestResultsPanel::RenderConfusionMatrixTab);
                    tab("Classes###classes", &TestResultsPanel::RenderClassesTab);
                    if (!results_.metric_learning_mode) tab("Curves###curves", &TestResultsPanel::RenderCurvesTab);
                    tab("Predictions###predictions", &TestResultsPanel::RenderPredictionsTab);
                }
                tab("Comparison###comparison", &TestResultsPanel::RenderComparisonTab,
                    select_comparison_tab_ ? ImGuiTabItemFlags_SetSelected : ImGuiTabItemFlags_None);
                select_comparison_tab_ = false;
                ImGui::EndTabBar();
            }
            RenderFooter();
        }
    }
    ImGui::End();
}

void TestResultsPanel::RenderHeader() {
    const ui::Tokens& t = ui::CurrentTokens();
    {
        ui::FontScope heading(ui::Font::Heading);
        ImGui::TextColored(ToneColour(view_.headline_tone), "%s", view_.headline.c_str());
    }
    Wrapped(t.text_dim, view_.subtitle);
    if (!view_.note.empty()) Wrapped(t.caution, view_.note);
    ui::FlowRow chips;
    for (const auto& chip : view_.chips) {
        chips.Next(ui::ChipWidth(chip.text.c_str()));
        ui::Chip(chip.text.c_str(), ToneColour(chip.tone));
    }
    ImGui::Dummy(ImVec2(0.0f, t.space_xs));
}

void TestResultsPanel::RenderOverviewTab() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (!view_.tiles.empty()) {
        ui::FlowRow tiles;
        for (const auto& tile : view_.tiles) {
            float w = 0.0f;
            {
                ui::FontScope medium(ui::Font::Medium);
                w = ImGui::CalcTextSize(tile.value.c_str()).x;
            }
            w = (std::max)(w, ImGui::CalcTextSize(tile.label.c_str()).x) + t.space_lg;
            tiles.Next(w);
            ImGui::BeginGroup();
            {
                ui::FontScope medium(ui::Font::Medium);
                ImGui::TextUnformatted(tile.value.c_str());
            }
            ImGui::TextColored(t.text_dim, "%s", tile.label.c_str());
            ImGui::EndGroup();
            ImGui::SameLine(0.0f, 0.0f);
            ImGui::Dummy(ImVec2(t.space_lg, 0.0f));
        }
        ImGui::NewLine();
        ImGui::Dummy(ImVec2(0.0f, t.space_sm));
    }
    if (view_.facts.empty()) return;
    if (ImGui::BeginTable("##facts", 2, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("key", ImGuiTableColumnFlags_WidthStretch, 0.35f);
        ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        for (const auto& [key, value] : view_.facts) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            Wrapped(t.text_dim, key);
            ImGui::TableNextColumn();
            Wrapped(key.rfind("Worst", 0) == 0 ? t.error : key == "Best class" ? t.success : t.text, value);
        }
        ImGui::EndTable();
    }
}

void TestResultsPanel::RenderConfusionMatrixTab() {
    const ui::Tokens& t = ui::CurrentTokens();
    const auto& mx = view_.matrix;
    if (mx.counts.empty()) {
        ImGui::TextColored(t.text_dim, "No confusion matrix for this run.");
        return;
    }
    static const char* const kModes[] = {"Counts", "Share of each true class"};
    ui::SegmentedControl("##matrix_mode", kModes, 2, &matrix_mode_);
    if (!mx.largest_confusion.empty()) Wrapped(t.text_dim, mx.largest_confusion);
    ImGui::Dummy(ImVec2(0.0f, t.space_xs));

    const int n = static_cast<int>(mx.counts.size());
    const ImGuiTableFlags flags = ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingFixedFit;
    if (!ImGui::BeginTable("##confusion", n + 1, flags)) return;
    ImGui::TableSetupScrollFreeze(1, 1);
    ImGui::TableSetupColumn("true \\ predicted");
    for (int j = 0; j < n; ++j) ImGui::TableSetupColumn(mx.names[static_cast<size_t>(j)].c_str());
    ImGui::TableHeadersRow();
    for (int i = 0; i < n; ++i) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextColored(t.text_dim, "%s", mx.names[static_cast<size_t>(i)].c_str());
        const int total = mx.row_totals[static_cast<size_t>(i)];
        for (int j = 0; j < n; ++j) {
            ImGui::TableNextColumn();
            const auto& row = mx.counts[static_cast<size_t>(i)];
            const int value = static_cast<size_t>(j) < row.size() ? row[static_cast<size_t>(j)] : 0;
            const float intensity = mx.max_count > 0 ? static_cast<float>(value) / static_cast<float>(mx.max_count) : 0.0f;
            const ImVec4 cell = i == j ? ui::Mix(t.bg_raised, t.success, 0.25f + 0.55f * intensity)
                                       : ui::Mix(t.bg_raised, t.error, value > 0 ? 0.15f + 0.6f * intensity : 0.0f);
            ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, ImGui::GetColorU32(cell));
            char label[32];
            if (matrix_mode_ == 1) std::snprintf(label, sizeof(label), "%.2f", total > 0 ? static_cast<double>(value) / total : 0.0);
            else std::snprintf(label, sizeof(label), "%d", value);
            ImGui::TextColored(ui::ReadableOn(t, cell), "%s", label);
            if (ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                ImGui::Text("True %s, predicted %s", mx.names[static_cast<size_t>(i)].c_str(),
                            mx.names[static_cast<size_t>(j)].c_str());
                ImGui::Text("%d of %d (%.1f%% of the true class)", value, total, total > 0 ? 100.0 * value / total : 0.0);
                ImGui::EndTooltip();
            }
        }
    }
    ImGui::EndTable();
}

void TestResultsPanel::RenderClassesTab() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (view_.classes.empty()) {
        ImGui::TextColored(t.text_dim, "No per-class metrics for this run.");
        return;
    }
    ImGui::SetNextItemWidth((std::min)(240.0f, ImGui::GetContentRegionAvail().x));
    ImGui::InputTextWithHint("##filter", "Filter classes", filter_text_, sizeof(filter_text_));
    // Fit to the headers and numbers; a narrow panel scrolls sideways.
    const ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_Sortable | ImGuiTableFlags_ScrollY |
                                  ImGuiTableFlags_ScrollX | ImGuiTableFlags_SizingFixedFit;
    if (!ImGui::BeginTable("##classes", 7, flags)) return;
    ImGui::TableSetupScrollFreeze(1, 1);
    ImGui::TableSetupColumn("Class", ImGuiTableColumnFlags_DefaultSort);
    for (const char* column : {"Precision", "Recall", "F1", "Found", "False alarms", "Support"})
        ImGui::TableSetupColumn(column);
    ImGui::TableHeadersRow();
    if (ImGuiTableSortSpecs* specs = ImGui::TableGetSortSpecs(); specs && specs->SpecsCount > 0) {
        static const test_report::ClassSort kColumns[] = {
            test_report::ClassSort::Name,  test_report::ClassSort::Precision,   test_report::ClassSort::Recall,
            test_report::ClassSort::F1,    test_report::ClassSort::Found,       test_report::ClassSort::FalseAlarms,
            test_report::ClassSort::Support};
        const auto& spec = specs->Specs[0];
        sort_by_ = kColumns[std::clamp(static_cast<int>(spec.ColumnIndex), 0, 6)];
        sort_ascending_ = spec.SortDirection == ImGuiSortDirection_Ascending;
    }
    const auto rows = test_report::SortClasses(view_.classes, sort_by_, sort_ascending_, filter_text_);
    for (const auto& row : rows) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(row.name.c_str());
        if (row.never_predicted && ImGui::IsItemHovered()) ImGui::SetTooltip("Never predicted in this run.");
        ImGui::TableNextColumn();
        ImGui::TextColored(MetricColour(row.precision), "%.4f", row.precision);
        ImGui::TableNextColumn();
        ImGui::TextColored(MetricColour(row.recall), "%.4f", row.recall);
        ImGui::TableNextColumn();
        ImGui::TextColored(MetricColour(row.f1), "%.4f", row.f1);
        ImGui::TableNextColumn();
        ImGui::Text("%d", row.found);
        ImGui::TableNextColumn();
        ImGui::Text("%d", row.false_alarms);
        ImGui::TableNextColumn();
        ImGui::Text("%d", row.support);
    }
    ImGui::EndTable();
}

void TestResultsPanel::RenderCurvesTab() {
    const ui::Tokens& t = ui::CurrentTokens();
    const auto& roc = results_.roc;
    const auto& pr = results_.pr;
    if (!roc || !roc->success || !pr || !pr->success) {
        ui::EmptyState(ICON_FA_CHART_AREA, "No curves for this run",
                       results_.curves_note.empty() ? "Run the test again to compute ROC and precision-recall."
                                                    : results_.curves_note.c_str());
        return;
    }
    const bool multiclass = !roc->class_auc.empty();
    {
        ui::FontScope heading(ui::Font::Heading);
        ImGui::Text("AUC %.4f", roc->auc);
        ImGui::SameLine(0.0f, 24.0f);
        ImGui::Text("AP %.4f", pr->average_precision);
    }
    Wrapped(t.text_dim, multiclass ? "One-vs-rest per class; AUC and AP are the macro means."
                                   : "Positive class: 1. F1 is one threshold; the curves show every threshold.");
    if (!results_.curves_note.empty()) Wrapped(t.caution, results_.curves_note);
    ImGui::Spacing();

    const float plot_height = (std::max)(ImGui::GetContentRegionAvail().y - ImGui::GetStyle().ItemSpacing.y, 220.0f);
    if (!ImGui::BeginTable("##curves", 2, ImGuiTableFlags_SizingStretchSame)) return;
    ImGui::TableNextColumn();
    if (ImPlot::BeginPlot("ROC", ImVec2(-1, plot_height), ImPlotFlags_NoMenus)) {
        ImPlot::SetupAxes("False positive rate", "True positive rate");
        ImPlot::SetupAxesLimits(-0.02, 1.02, -0.02, 1.02, ImPlotCond_Always);
        const double diagonal[] = {0.0, 1.0};
        ImPlot::SetNextLineStyle(ui::WithAlpha(t.text_dim, 0.6f));
        ImPlot::PlotLine("##chance", diagonal, diagonal, 2);
        if (multiclass) {
            for (size_t c = 0; c < roc->class_fpr.size(); ++c) {
                if (roc->class_fpr[c].empty()) continue;  // no test sample of the class
                char label[160];
                std::snprintf(label, sizeof(label), "%s (AUC %.3f)",
                              test_report::ClassName(results_, static_cast<int>(c)).c_str(), roc->class_auc[c]);
                ImPlot::SetNextLineStyle(ui::SeriesColour(t, c), 2.0f);
                ImPlot::PlotLine(label, roc->class_fpr[c].data(), roc->class_tpr[c].data(),
                                 static_cast<int>(roc->class_fpr[c].size()));
            }
        } else {
            ImPlot::SetNextLineStyle(t.series[1], 2.0f);
            ImPlot::PlotLine("ROC", roc->fpr.data(), roc->tpr.data(), static_cast<int>(roc->fpr.size()));
        }
        ImPlot::EndPlot();
    }
    ImGui::TableNextColumn();
    if (ImPlot::BeginPlot("Precision-recall", ImVec2(-1, plot_height), ImPlotFlags_NoMenus)) {
        ImPlot::SetupAxes("Recall", "Precision");
        ImPlot::SetupAxesLimits(-0.02, 1.02, -0.02, 1.02, ImPlotCond_Always);
        if (multiclass) {
            for (size_t c = 0; c < pr->class_recall.size(); ++c) {
                if (pr->class_recall[c].empty()) continue;
                char label[160];
                std::snprintf(label, sizeof(label), "%s (AP %.3f)",
                              test_report::ClassName(results_, static_cast<int>(c)).c_str(), pr->class_ap[c]);
                ImPlot::SetNextLineStyle(ui::SeriesColour(t, c), 2.0f);
                ImPlot::PlotLine(label, pr->class_recall[c].data(), pr->class_precision[c].data(),
                                 static_cast<int>(pr->class_recall[c].size()));
            }
        } else {
            ImPlot::SetNextLineStyle(t.series[0], 2.0f);
            ImPlot::PlotLine("PR", pr->recall.data(), pr->precision.data(), static_cast<int>(pr->recall.size()));
        }
        ImPlot::EndPlot();
    }
    ImGui::EndTable();
}

void TestResultsPanel::RenderPredictionsTab() {
    const ui::Tokens& t = ui::CurrentTokens();
    const size_t samples = std::min(results_.predictions.size(), results_.ground_truth.size());
    if (samples == 0) {
        ImGui::TextColored(t.text_dim, "No predictions were kept for this run.");
        return;
    }
    const std::string all = "All " + FormatCompiledCount(static_cast<long long>(samples));
    const std::string mistakes = "Only mistakes " + FormatCompiledCount(static_cast<long long>(view_.mistakes.size()));
    const char* modes[] = {all.c_str(), mistakes.c_str()};
    ui::SegmentedControl("##predictions_mode", modes, 2, &predictions_mode_);
    const ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp;
    if (!ImGui::BeginTable("##predictions", 4, flags)) return;
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableSetupColumn("#", 0, 0.5f);
    ImGui::TableSetupColumn("Predicted", 0, 1.2f);
    ImGui::TableSetupColumn("Actual", 0, 1.2f);
    ImGui::TableSetupColumn("Confidence", 0, 1.2f);
    ImGui::TableHeadersRow();
    // The clipper walks the rows that are shown, so scrolling stays even
    // with only the mistakes.
    const bool only_mistakes = predictions_mode_ == 1;
    const int count = only_mistakes ? static_cast<int>(view_.mistakes.size()) : static_cast<int>(samples);
    ImGuiListClipper clipper;
    clipper.Begin(count);
    while (clipper.Step()) {
        for (int r = clipper.DisplayStart; r < clipper.DisplayEnd; ++r) {
            const int i = only_mistakes ? view_.mistakes[static_cast<size_t>(r)] : r;
            const int predicted = results_.predictions[static_cast<size_t>(i)];
            const int actual = results_.ground_truth[static_cast<size_t>(i)];
            const float confidence =
                static_cast<size_t>(i) < results_.confidences.size() ? results_.confidences[static_cast<size_t>(i)] : 0.0f;
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%d", i);
            ImGui::TableNextColumn();
            ImGui::TextColored(predicted == actual ? t.success : t.error, "%s",
                               test_report::ClassName(results_, predicted).c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(test_report::ClassName(results_, actual).c_str());
            ImGui::TableNextColumn();
            char overlay[16];
            std::snprintf(overlay, sizeof(overlay), "%.0f%%", confidence * 100.0f);
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram,
                                  confidence > 0.8f ? t.success : confidence > 0.5f ? t.warning : t.caution);
            ImGui::ProgressBar(confidence, ImVec2(-FLT_MIN, 0.0f), overlay);
            ImGui::PopStyleColor();
        }
    }
    ImGui::EndTable();
}

void TestResultsPanel::RenderComparisonTab() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (view_.comparison.empty()) {
        Wrapped(t.text_dim, "One run so far. Run a second test to compare it with this one.");
        return;
    }
    Wrapped(t.text_dim, "The run before beside the latest run. Change is latest minus before; for loss, MAE and RMSE "
                        "a negative change is an improvement.");
    if (ImGui::BeginTable("##comparison", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Metric", 0, 1.4f);
        ImGui::TableSetupColumn("Before", 0, 1.0f);
        ImGui::TableSetupColumn("Latest", 0, 1.0f);
        ImGui::TableSetupColumn("Change", 0, 1.0f);
        ImGui::TableHeadersRow();
        for (const auto& row : view_.comparison) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(row.metric.c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(row.previous.c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(row.latest.c_str());
            ImGui::TableNextColumn();
            ImGui::TextColored(ToneColour(row.change_tone), "%s", row.change.c_str());
        }
        ImGui::EndTable();
    }
    if (!view_.comparison_note.empty()) Wrapped(t.text_dim, view_.comparison_note);
}

void TestResultsPanel::RenderFooter() {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::Dummy(ImVec2(0.0f, 2.0f));
    ui::FlowRow row;
    row.Next(ui::ButtonWidth("Export CSV...", ui::ButtonSize::Small));
    if (ui::SecondaryButton("Export CSV...", true, nullptr, ui::ButtonSize::Small)) ExportToCSV();
    row.Next(ui::ButtonWidth("Export JSON...", ui::ButtonSize::Small));
    if (ui::SecondaryButton("Export JSON...", true, nullptr, ui::ButtonSize::Small)) ExportToJSON();
    row.Next(ui::ButtonWidth("Clear", ui::ButtonSize::Small));
    const bool clear = ui::DangerButton("Clear", true, nullptr, ui::ButtonSize::Small);
    if (run_test_) {
        row.Next(ui::ButtonWidth("Run test again", ui::ButtonSize::Regular));
        if (ui::PrimaryButton("Run test again")) run_test_();
    }
    if (!export_note_.empty()) Wrapped(export_failed_ ? t.error : t.success, export_note_);
    if (clear) {
        results_ = TestingMetrics();
        previous_results_ = TestingMetrics();
        has_results_ = false;
        has_previous_ = false;
        failure_message_.clear();
        export_note_.clear();
        view_dirty_ = true;
    }
}

void TestResultsPanel::ExportToCSV() {
    const std::string folder = ExportsFolder();
    const auto path = FileDialogs::SaveFile("Export Test Results", {{"CSV Files", "csv"}, {"All Files", "*"}},
                                            folder.empty() ? nullptr : folder.c_str(), "test_results.csv");
    if (!path) return;
    std::string error;
    export_failed_ = !WriteTestResultsCsv(results_, *path, &error);
    export_note_ = export_failed_ ? "Could not export: " + error : "Saved " + *path;
}

void TestResultsPanel::ExportToJSON() {
    const std::string folder = ExportsFolder();
    const auto path = FileDialogs::SaveFile("Export Test Results", {{"JSON Files", "json"}, {"All Files", "*"}},
                                            folder.empty() ? nullptr : folder.c_str(), "test_results.json");
    if (!path) return;
    std::string error;
    export_failed_ = !WriteTestResultsJson(results_, *path, &error);
    export_note_ = export_failed_ ? "Could not export: " + error : "Saved " + *path;
}

} // namespace cyxwiz
