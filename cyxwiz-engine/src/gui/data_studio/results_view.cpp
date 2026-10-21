#include "results_view.h"

#include "../../core/arrow_dataset.h"
#include "../../core/data_registry.h"
#include "../../core/project_manager.h"
#include "../../core/kept_tests.h"
#include "../../core/test_predictions_metrics.h"
#include "../../core/test_predictions_table.h"
#include "../panels/test_results_panel.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"

#include <arrow/api.h>
#include <imgui.h>

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <ctime>
#include <filesystem>

namespace cyxwiz {

void ResultsView::RefreshRuns() {
    const double now = ImGui::GetTime();
    if (runs_time_ >= 0.0 && now - runs_time_ < 5.0) return;
    runs_time_ = now;
    const auto current = run_ < static_cast<int>(runs_.size()) ? runs_[static_cast<size_t>(run_)].folder
                                                               : std::filesystem::path();
    runs_ = FindGenerationPreviewRuns(ProjectManager::Instance().GetCheckpointsPath());
    run_ = 0;
    for (size_t i = 0; i < runs_.size(); ++i)
        if (runs_[i].folder == current) run_ = static_cast<int>(i);
}

void ResultsView::Render() {
    RefreshRuns();
    const bool have_predictions = DataRegistry::Instance().GetArrowDataset(kTestPredictionsDataset) != nullptr;
    if (!source_chosen_) source_ = runs_.empty() && have_predictions ? 1 : 0;
    static const char* kSources[] = {"Generations", "Test run"};
    if (ui::SegmentedControl("##results_source", kSources, 2, &source_)) source_chosen_ = true;
    ImGui::Spacing();
    if (source_ == 0) RenderPreviews();
    else RenderTest();
}

void ResultsView::RenderPreviews() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (runs_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_dim, "No training run of this project wrote generation previews. Turn them on in the "
                                       "Data Loader's Generation tab of a language model's graph; each run's samples "
                                       "show here and in the Training Dashboard.");
        ImGui::PopTextWrapPos();
        return;
    }
    const auto& run = runs_[static_cast<size_t>(run_)];
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "Run");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(std::min(420.0f, ImGui::GetContentRegionAvail().x * 0.7f));
    const std::string run_label = run.graph + " \xC2\xB7 " + run.run_id;
    if (ImGui::BeginCombo("##run", run_label.c_str())) {
        for (size_t i = 0; i < runs_.size(); ++i) {
            const std::string label = runs_[i].graph + " \xC2\xB7 " + runs_[i].run_id + "  (" +
                                      std::to_string(runs_[i].files) + (runs_[i].files == 1 ? " pass)" : " passes)");
            if (ImGui::Selectable(label.c_str(), static_cast<int>(i) == run_)) run_ = static_cast<int>(i);
        }
        ImGui::EndCombo();
    }
    preview_view_.SetFolder(run.folder);
    ImGui::Spacing();
    preview_view_.Render();
}

size_t ResultsView::PreviewRunCount() {
    RefreshRuns();
    return runs_.size();
}

int64_t ResultsView::TestRowCount() const {
    const auto dataset = DataRegistry::Instance().GetArrowDataset(kTestPredictionsDataset);
    const auto table = dataset ? dataset->GetArrowTable() : nullptr;
    return table ? table->num_rows() : 0;
}

void ResultsView::RenderTest() {
    const ui::Tokens& t = ui::CurrentTokens();
    RenderKeepDialog();
    const auto dataset = DataRegistry::Instance().GetArrowDataset(kTestPredictionsDataset);
    const auto table = dataset ? dataset->GetArrowTable() : nullptr;
    ImGui::PushTextWrapPos(0.0f);
    if (!table) {
        ImGui::TextColored(t.text_dim, "No test run in this session. Test a model (Train > Test > Run Test): its results open in "
                                       "the Test Results panel, and its per-row predictions are published as "
                                       "\"test_predictions\".");
        ImGui::PopTextWrapPos();
        return;
    }
    // The run read back from its table: the same calculation and words as
    // the Test Results panel.
    const std::string key = std::to_string(reinterpret_cast<uintptr_t>(table.get())) + "#" + std::to_string(table->num_rows());
    if (key != predictions_key_) {
        predictions_key_ = key;
        TestingMetrics run;
        test_error_.clear();
        if (MetricsFromPredictionsTable(*table, run, test_error_)) test_view_ = test_report::BuildView(run, nullptr);
    }
    if (!test_error_.empty()) {
        ImGui::TextColored(t.error, "The test predictions cannot be read: %s", test_error_.c_str());
        ImGui::PopTextWrapPos();
        return;
    }
    ImGui::TextColored(t.text_dim, "LATEST TEST");
    ImGui::TextColored(TestToneColour(test_view_.headline_tone), "%s", test_view_.headline.c_str());
    ImGui::TextColored(t.text_dim, "%s", test_view_.subtitle.c_str());
    std::string chips;
    for (const auto& chip : test_view_.chips) chips += (chips.empty() ? "" : " \xC2\xB7 ") + chip.text;
    if (!chips.empty()) ImGui::TextColored(t.text_faint, "%s", chips.c_str());
    ImGui::Spacing();
    ImGui::TextColored(t.text_dim, "The Test Results panel shows the rest: %s", test_view_.classification
        ? "the confusion matrix, the classes, the curves, every row by its dataset row, and the error by the values of a column."
        : "every row by its dataset row, the largest errors first, and the error by the values of a column.");
    ImGui::PopTextWrapPos();
    if (ui::SecondaryButton("Open Test Results", static_cast<bool>(on_open_test_results), "Not available here") &&
        on_open_test_results)
        on_open_test_results();
    RenderKeptTests();
}

std::string ResultsView::KeepBlocker(bool graph_open) const {
    if (!DataRegistry::Instance().GetArrowDataset(kTestPredictionsDataset)) return "Test a model first (Train > Test > Run Test)";
    if (ProjectManager::Instance().GetProjectRoot().empty()) return "Open or create a project: the test is kept in its results folder";
    if (!graph_open) return "No graph is open";
    return "";
}

void ResultsView::RenderKeepDialog() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (keep_requested_) {
        keep_requested_ = false;
        const std::string name = (graph_name.empty() || graph_name == "unsaved graph" ? std::string("Model") : graph_name) + " test";
        std::snprintf(keep_name_, sizeof(keep_name_), "%s", name.c_str());
        ImGui::OpenPopup("Keep this test as a node");
    }
    ImGui::SetNextWindowSize(ImVec2(520.0f, 0.0f), ImGuiCond_Appearing);
    if (!ImGui::BeginPopupModal("Keep this test as a node", nullptr, ImGuiWindowFlags_NoSavedSettings)) return;
    const auto dataset = DataRegistry::Instance().GetArrowDataset(kTestPredictionsDataset);
    const auto rows = dataset ? dataset->GetArrowTable() : nullptr;
    const std::string root = ProjectManager::Instance().GetProjectRoot();
    const std::string file = KeptTestFileName(keep_name_, std::time(nullptr));
    ImGui::TextColored(t.text_dim, "Name");
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::InputText("##keep_name", keep_name_, sizeof(keep_name_));
    ImGui::PushTextWrapPos(0.0f);
    ImGui::Text("Saves the %lld test rows to results/tests/%s in the project.", static_cast<long long>(rows ? rows->num_rows() : 0),
                file.c_str());
    ImGui::TextColored(t.text_dim, "Adds a Data Input reading that file and a Test Evaluation node after it. The file is a snapshot: "
                                   "testing again does not change it. Its rows still point at the tested table, so the "
                                   "worst rows and the error by column value keep working.");
    ImGui::PopTextWrapPos();
    ImGui::Spacing();
    const bool named = keep_name_[0] != '\0';
    if (ui::PrimaryButton("Keep", named && rows && !root.empty(), "Give it a name")) {
        const std::filesystem::path relative = std::filesystem::path("results") / "tests" / file;
        std::string error;
        keep_failed_ = !WriteKeptTest(*rows, keep_name_, KeptTestsFolder(root) / file, error);
        if (keep_failed_) {
            keep_note_ = "Could not keep the test: " + error;
        } else {
            if (on_keep) on_keep(relative.generic_string(), keep_name_);
            keep_note_ = "Kept as " + relative.generic_string() + ": added the Data Input \"" + std::string(keep_name_) +
                         "\" and a Test Evaluation node (Ctrl+Z on the canvas removes them; the file stays).";
            kept_time_ = -1.0;  // list it now
        }
        ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ui::SecondaryButton("Cancel")) ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
}

void ResultsView::RenderKeptTests() {
    const ui::Tokens& t = ui::CurrentTokens();
    const std::string root = ProjectManager::Instance().GetProjectRoot();
    const double now = ImGui::GetTime();
    if (kept_time_ < 0.0 || now - kept_time_ > 2.0) {
        kept_time_ = now;
        kept_ = root.empty() ? std::vector<KeptTest>{} : ListKeptTests(KeptTestsFolder(root));
    }
    if (!keep_note_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(keep_failed_ ? t.error : t.success, "%s", keep_note_.c_str());
        ImGui::PopTextWrapPos();
    }
    if (kept_.empty()) return;
    ImGui::Spacing();
    ImGui::TextColored(t.text_dim, "KEPT TESTS");
    for (const auto& k : kept_) {
        ImGui::PushID(k.path.string().c_str());
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(k.name.c_str());
        ImGui::SameLine();
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.text_faint, "%s", ("results/tests/" + k.path.filename().string()).c_str());
        ImGui::SameLine();
        if (ui::LinkButton("Open") && on_open_kept) on_open_kept(k.path.string());
        ImGui::PopID();
    }
}

}  // namespace cyxwiz
