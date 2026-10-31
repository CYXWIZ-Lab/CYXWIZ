#include "results_view.h"

#include "../../core/arrow_dataset.h"
#include "../../core/data_registry.h"
#include "../../core/project_manager.h"
#include "../../core/kept_tests.h"
#include "../../core/test_predictions_table.h"
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

ResultsView::ResultsView() {
    test_.keep = [this] { RequestKeep(); };
    test_.keep_blocker = [this] { return KeepBlocker(graph_open); };
    // The folds' sentence sets the latest test beside them.
    for (const auto& view : {cross_validation_, learning_curve_})
        view->single_test = [this] { return test_.LatestScore(); };
}

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
    // A language model's project opens on its previews, any other on Test.
    if (!view_chosen_) view_ = !runs_.empty() && !test_.HasResults() ? View::Generations : View::Test;
    static const struct {
        const char* label;
        View view;
    } kViews[] = {{"Test", View::Test},
                  {"Cross-validation", View::CrossValidation},
                  {"Learning curve", View::LearningCurve},
                  {"Generations", View::Generations}};
    for (size_t i = 0; i < std::size(kViews); ++i) {
        if (i > 0) ImGui::SameLine();
        if (ui::FilterChip(kViews[i].label, kViews[i].label, "", view_ == kViews[i].view)) Show(kViews[i].view);
    }
    ImGui::Spacing();
    RenderKeepDialog();
    switch (view_) {
        case View::Test: test_.Render(); break;
        case View::CrossValidation: cross_validation_->Render(); break;
        case View::LearningCurve: learning_curve_->Render(); break;
        case View::Generations: RenderPreviews(); break;
    }
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

std::string ResultsView::KeepBlocker(bool open) const {
    if (!DataRegistry::Instance().GetArrowDataset(kTestPredictionsDataset)) return "Test a model first (Train > Test > Run Test)";
    if (ProjectManager::Instance().GetProjectRoot().empty()) return "Open or create a project: the test is kept in its results folder";
    if (!open) return "No graph is open";
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
        if (!WriteKeptTest(*rows, keep_name_, KeptTestsFolder(root) / file, error)) {
            test_.SetNote("Could not keep the test: " + error, true);
        } else {
            if (on_keep) on_keep(relative.generic_string(), keep_name_);
            test_.SetNote("Kept as " + relative.generic_string() + ": added the Data Input \"" + std::string(keep_name_) +
                              "\" and a Test Evaluation node (Ctrl+Z on the canvas removes them; the file stays). Showing "
                              "lists it under Kept.",
                          false);
        }
        ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ui::SecondaryButton("Cancel")) ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
}

}  // namespace cyxwiz
