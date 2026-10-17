#include "results_view.h"

#include "../../core/arrow_dataset.h"
#include "../../core/data_registry.h"
#include "../../core/project_manager.h"
#include "../../core/test_predictions_metrics.h"
#include "../../core/test_predictions_table.h"
#include "../panels/test_results_panel.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"

#include <arrow/api.h>
#include <imgui.h>

#include <algorithm>
#include <cstdio>

namespace cyxwiz {

namespace {

std::string Of(size_t n, size_t of) { return std::to_string(n) + " of " + std::to_string(of); }

// A figure with its label under it, as the board's summary tiles.
void Tile(const char* label, const std::string& value, const std::string& note, const ImVec4& colour) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::BeginGroup();
    ImGui::TextColored(t.text_dim, "%s", label);
    ImGui::TextColored(colour, "%s", value.c_str());
    if (!note.empty()) ImGui::TextColored(t.text_faint, "%s", note.c_str());
    ImGui::EndGroup();
}

void Mark(bool ok, bool applies) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (!applies) ImGui::TextColored(t.text_faint, "-");
    else ImGui::TextColored(ok ? t.success : t.error, "%s", ok ? ICON_FA_CHECK : ICON_FA_XMARK);
}

}  // namespace

void ResultsView::RefreshRuns() {
    const double now = ImGui::GetTime();
    if (runs_time_ >= 0.0 && now - runs_time_ < 5.0) return;
    runs_time_ = now;
    const std::string current = run_ < static_cast<int>(runs_.size()) ? runs_[static_cast<size_t>(run_)].folder : std::string();
    runs_ = studio::FindPreviewRuns(ProjectManager::Instance().GetCheckpointsPath());
    run_ = 0;
    for (size_t i = 0; i < runs_.size(); ++i)
        if (runs_[i].folder == current) run_ = static_cast<int>(i);
}

void ResultsView::LoadEpoch() {
    if (runs_.empty()) return;
    const auto& run = runs_[static_cast<size_t>(run_)];
    const int e = epoch_index_ < 0 || epoch_index_ >= static_cast<int>(run.epochs.size()) ? run.epochs.back()
                                                                                          : run.epochs[static_cast<size_t>(epoch_index_)];
    const std::string key = run.folder + "#" + std::to_string(e);
    if (key == loaded_key_) return;
    loaded_key_ = key;
    selected_sample_ = -1;
    epoch_error_.clear();
    if (!studio::LoadPreviewEpoch(std::filesystem::path(run.folder) / ("epoch_" + std::to_string(e) + ".json"), epoch_, epoch_error_))
        epoch_ = {};
}

void ResultsView::Render() {
    RefreshRuns();
    const bool have_predictions = DataRegistry::Instance().GetArrowDataset(kTestPredictionsDataset) != nullptr;
    if (!source_chosen_) source_ = runs_.empty() && have_predictions ? 1 : 0;
    static const char* kSources[] = {"Language model previews", "Test run"};
    if (ui::SegmentedControl("##results_source", kSources, 2, &source_)) source_chosen_ = true;
    ImGui::Spacing();
    if (source_ == 0) RenderPreviews();
    else RenderTest();
}

void ResultsView::RenderPreviews() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (runs_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_dim, "No training run of this project wrote generation previews. A causal language model's training "
                                       "writes them each epoch when its Output node has preview prompts; they show here.");
        ImGui::PopTextWrapPos();
        return;
    }
    // The run and the epoch.
    const auto& run = runs_[static_cast<size_t>(run_)];
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "Run");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(std::min(380.0f, ImGui::GetContentRegionAvail().x * 0.6f));
    const std::string run_label = run.graph + " \xC2\xB7 " + run.run_id;
    if (ImGui::BeginCombo("##run", run_label.c_str())) {
        for (size_t i = 0; i < runs_.size(); ++i) {
            const std::string label = runs_[i].graph + " \xC2\xB7 " + runs_[i].run_id + "  (" + std::to_string(runs_[i].epochs.size()) +
                                      (runs_[i].epochs.size() == 1 ? " epoch)" : " epochs)");
            if (ImGui::Selectable(label.c_str(), static_cast<int>(i) == run_)) {
                run_ = static_cast<int>(i);
                epoch_index_ = -1;
            }
        }
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "Epoch");
    ImGui::SameLine();
    const int last = static_cast<int>(run.epochs.size()) - 1;
    const int shown_index = epoch_index_ < 0 || epoch_index_ > last ? last : epoch_index_;
    ImGui::SetNextItemWidth(90.0f);
    if (ImGui::BeginCombo("##epoch", std::to_string(run.epochs[static_cast<size_t>(shown_index)]).c_str())) {
        for (int i = 0; i <= last; ++i)
            if (ImGui::Selectable(std::to_string(run.epochs[static_cast<size_t>(i)]).c_str(), i == shown_index)) epoch_index_ = i;
        ImGui::EndCombo();
    }
    LoadEpoch();
    if (!epoch_error_.empty()) {
        ImGui::TextColored(t.error, "%s", epoch_error_.c_str());
        return;
    }
    ImGui::TextColored(t.text_faint, "%zu preview prompts \xC2\xB7 %s, %d new tokens%s", epoch_.samples.size(), epoch_.sampling.c_str(),
                       epoch_.max_new_tokens, epoch_.cancelled ? " \xC2\xB7 cancelled" : "");
    ImGui::Spacing();

    // The summary tiles.
    const auto s = studio::SummarizePreview(epoch_);
    const size_t n = s.samples;
    const auto colour = [&](size_t good) { return good == n ? t.success : good == 0 ? t.error : t.warning; };
    const float tile_w = 150.0f;
    const auto next = [&] {
        ImGui::SameLine(0.0f, 28.0f);
        if (ImGui::GetContentRegionAvail().x < tile_w) ImGui::NewLine();
    };
    if (s.with_refs) {
        Tile("Right book", Of(s.right_book, s.with_refs), "", colour(s.right_book));
        next();
        Tile("Right chapter", Of(s.right_chapter, s.with_refs), "", colour(s.right_chapter));
        next();
        Tile("Right verse", Of(s.right_verse, s.with_refs), "", colour(s.right_verse));
        next();
    }
    std::string note = std::to_string(s.hit_limit) + " hit the limit";
    if (s.loops) note += ", " + std::to_string(s.loops) + (s.loops == 1 ? " loops" : " loop");
    Tile("Stopped by itself", Of(s.stopped_itself, n), note, colour(s.stopped_itself));
    ImGui::Spacing();

    // One row per prompt.
    if (ImGui::BeginTable("##previews", s.with_refs ? 7 : 3,
                          ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_PadOuterX |
                              ImGuiTableFlags_NoSavedSettings)) {
        const float mark_w = ImGui::CalcTextSize("chapter").x + 8.0f;
        ImGui::TableSetupColumn("question", ImGuiTableColumnFlags_WidthStretch, 3.0f);
        if (s.with_refs) {
            // References are short: sized to the longest, never cut.
            float ref_w = ImGui::CalcTextSize("model cited").x;
            for (const auto& x : epoch_.samples)
                ref_w = std::max({ref_w, ImGui::CalcTextSize(x.evidence_ref.c_str()).x, ImGui::CalcTextSize(x.cited_ref.c_str()).x});
            ImGui::TableSetupColumn("evidence", ImGuiTableColumnFlags_WidthFixed, ref_w + 8.0f);
            ImGui::TableSetupColumn("model cited", ImGuiTableColumnFlags_WidthFixed, ref_w + 8.0f);
            ImGui::TableSetupColumn("book", ImGuiTableColumnFlags_WidthFixed, mark_w);
            ImGui::TableSetupColumn("chapter", ImGuiTableColumnFlags_WidthFixed, mark_w);
            ImGui::TableSetupColumn("verse", ImGuiTableColumnFlags_WidthFixed, mark_w);
        } else {
            ImGui::TableSetupColumn("answer", ImGuiTableColumnFlags_WidthStretch, 3.0f);
        }
        ImGui::TableSetupColumn("stop", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("token limit \xC2\xB7 loops").x + 8.0f);
        ImGui::TableHeadersRow();
        for (size_t i = 0; i < epoch_.samples.size(); ++i) {
            const auto& x = epoch_.samples[i];
            ImGui::PushID(static_cast<int>(i));
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            std::string q = x.question.size() > 110 ? x.question.substr(0, 107) + "..." : x.question;
            ui::PushListRowColours();
            if (ImGui::Selectable(q.c_str(), selected_sample_ == static_cast<int>(i), ImGuiSelectableFlags_SpanAllColumns))
                selected_sample_ = selected_sample_ == static_cast<int>(i) ? -1 : static_cast<int>(i);
            ui::PopListRowColours();
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Click: show the model's answer under the table");
            if (s.with_refs) {
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(x.evidence_ref.c_str());
                ImGui::TableNextColumn();
                ImGui::TextColored(x.cited_ref.empty() ? t.text_faint : t.text, "%s", x.cited_ref.empty() ? "none" : x.cited_ref.c_str());
                ImGui::TableNextColumn();
                Mark(x.book, x.has_ref);
                ImGui::TableNextColumn();
                Mark(x.chapter, x.has_ref);
                ImGui::TableNextColumn();
                Mark(x.verse, x.has_ref);
            } else {
                ImGui::TableNextColumn();
                const std::string a = x.generated.size() > 110 ? x.generated.substr(0, 107) + "..." : x.generated;
                ImGui::TextColored(t.text_dim, "%s", a.c_str());
            }
            ImGui::TableNextColumn();
            const std::string stop = studio::StopWords(x.stop) + (x.loops ? " \xC2\xB7 loops" : "");
            ImGui::TextColored(studio::StopWords(x.stop) == "end" && !x.loops ? t.text_dim : t.warning, "%s", stop.c_str());
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    if (selected_sample_ >= 0 && selected_sample_ < static_cast<int>(epoch_.samples.size())) {
        const auto& x = epoch_.samples[static_cast<size_t>(selected_sample_)];
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ui::Mix(t.bg_window, t.text, 0.035f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.0f, 10.0f));
        ImGui::BeginChild("##answer", ImVec2(0.0f, 0.0f), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);
        ImGui::PopStyleVar();
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_dim, "QUESTION");
        ImGui::TextUnformatted(x.question.c_str());
        ImGui::Spacing();
        ImGui::TextColored(t.text_dim, "THE MODEL'S ANSWER");
        ImGui::TextUnformatted(x.generated.c_str());
        ImGui::PopTextWrapPos();
        ImGui::EndChild();
        ImGui::PopStyleColor();
    }

    // Where it goes wrong.
    ImGui::Spacing();
    ImGui::TextColored(t.text_dim, "WHERE IT GOES WRONG (SLICES)");
    if (ImGui::BeginTable("##slices", s.with_refs ? 3 : 2, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_NoSavedSettings)) {
        ImGui::TableSetupColumn("slice");
        ImGui::TableSetupColumn("rows");
        if (s.with_refs) ImGui::TableSetupColumn("right verse");
        ImGui::TableHeadersRow();
        for (const auto& slice : studio::PreviewSlices(epoch_)) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(slice.name.c_str());
            ImGui::TableNextColumn();
            ImGui::Text("%zu", slice.rows);
            if (s.with_refs) {
                ImGui::TableNextColumn();
                ImGui::Text("%zu", slice.right_verse);
            }
        }
        ImGui::EndTable();
    }
    ImGui::Spacing();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_faint, "These are the run's preview prompts. A full table for an evaluation set (every validation and test "
                                     "prompt) comes from batch evaluation (TOFIX149 item 4).");
    ImGui::PopTextWrapPos();
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
}

}  // namespace cyxwiz
