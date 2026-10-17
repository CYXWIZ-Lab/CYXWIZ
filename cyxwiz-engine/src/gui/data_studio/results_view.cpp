#include "results_view.h"

#include "../../core/arrow_dataset.h"
#include "../../core/async_task_manager.h"
#include "../../core/data_registry.h"
#include "../../core/parquet_backed_dataset.h"
#include "../../core/project_manager.h"
#include "../../core/test_predictions_table.h"
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

std::string Percent(double x) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%.1f%%", x * 100.0);
    return buf;
}

// A measured value (an error, a target) to four significant digits.
std::string Figure(double x) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.4g", x);
    return buf;
}

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

ResultsView::~ResultsView() {
    if (slice_task_) AsyncTaskManager::Instance().Cancel(slice_task_);
}

void ResultsView::RenderTest() {
    const ui::Tokens& t = ui::CurrentTokens();
    const auto dataset = DataRegistry::Instance().GetArrowDataset(kTestPredictionsDataset);
    const auto table = dataset ? dataset->GetArrowTable() : nullptr;
    if (!table) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_dim, "No test run in this session. Test a model (Train > Test); its per-row predictions are "
                                       "published as \"test_predictions\" and show here: a classifier's accuracy, confusion per "
                                       "class and worst rows, a regression's errors, and the error by the values of a column.");
        ImGui::PopTextWrapPos();
        return;
    }
    const std::string key = std::to_string(reinterpret_cast<uintptr_t>(table.get())) + "#" + std::to_string(table->num_rows());
    if (key != predictions_key_) {
        predictions_key_ = key;
        regression_ = studio::IsRegressionPredictions(table);
        if (regression_) regression_results_ = studio::RegressionFromPredictions(table);
        else classifier_ = studio::ClassifierFromPredictions(table);
        row_errors_ = studio::RowErrorsFromPredictions(table);
        // The tested table's columns that can be sliced (not lists or bytes).
        slice_columns_.clear();
        std::shared_ptr<arrow::Schema> schema;
        if (row_errors_.ok) {
            if (const auto memory = DataRegistry::Instance().GetArrowDataset(row_errors_.source_dataset)) {
                if (const auto source = memory->GetArrowTable()) schema = source->schema();
            } else if (const auto disk = DataRegistry::Instance().GetParquetBackedDataset(row_errors_.source_dataset)) {
                schema = disk->GetSchema();
            }
        }
        if (schema)
            for (const auto& field : schema->fields())
                if (!arrow::is_nested(field->type()->id()) && field->type()->id() != arrow::Type::BINARY &&
                    field->type()->id() != arrow::Type::LARGE_BINARY)
                    slice_columns_.push_back(field->name());
        if (slice_task_) AsyncTaskManager::Instance().Cancel(slice_task_);
        slice_task_ = 0;
        slice_column_.clear();
        slices_ = {};
        slices_error_.clear();
    }
    if (regression_) RenderRegression();
    else RenderClassifier();
    ImGui::Spacing();
    RenderErrorsByValue();
}

void ResultsView::RenderClassifier() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (!classifier_.ok) {
        ImGui::TextColored(t.error, "The test predictions cannot be read: %s", classifier_.error.c_str());
        return;
    }
    const auto& r = classifier_;
    Tile("Accuracy", Percent(r.accuracy), std::to_string(r.rows) + " test rows", r.accuracy >= 0.9 ? t.success : r.accuracy >= 0.6 ? t.warning : t.error);
    ImGui::SameLine(0.0f, 28.0f);
    Tile("Wrong", std::to_string(r.rows - static_cast<int64_t>(r.accuracy * static_cast<double>(r.rows) + 0.5)), "rows", t.text);
    ImGui::Spacing();

    ImGui::TextColored(t.text_dim, "CONFUSION (ROWS: TRUE CLASS, COLUMNS: PREDICTED)");
    const int k = static_cast<int>(r.classes.size());
    // A scrolling table with height 0 fills the rest of the window: size it
    // to its rows (the header and one per class) and the scrollbar.
    const ImGuiStyle& style = ImGui::GetStyle();
    const float confusion_h = static_cast<float>(k + 1) * (ImGui::GetTextLineHeight() + style.CellPadding.y * 2.0f) +
                              style.ScrollbarSize + style.CellPadding.y * 2.0f;
    if (k <= 20 && ImGui::BeginTable("##confusion", k + 1, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_NoSavedSettings |
                                                              ImGuiTableFlags_ScrollX,
                                     ImVec2(0.0f, confusion_h))) {
        ImGui::TableSetupColumn("");
        for (const auto& c : r.classes) ImGui::TableSetupColumn(c.c_str());
        ImGui::TableHeadersRow();
        for (int a = 0; a < k; ++a) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextColored(t.text_dim, "%s", r.classes[static_cast<size_t>(a)].c_str());
            for (int b = 0; b < k; ++b) {
                ImGui::TableNextColumn();
                const int64_t v = r.confusion[static_cast<size_t>(a)][static_cast<size_t>(b)];
                ImGui::TextColored(a == b ? t.success : v ? t.error : t.text_faint, "%lld", static_cast<long long>(v));
            }
        }
        ImGui::EndTable();
    } else if (k > 20) {
        ImGui::TextColored(t.text_faint, "%d classes: the per-class table below lists them.", k);
    }
    ImGui::Spacing();
    if (ImGui::BeginTable("##per_class", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_NoSavedSettings)) {
        ImGui::TableSetupColumn("class");
        ImGui::TableSetupColumn("rows");
        ImGui::TableSetupColumn("precision");
        ImGui::TableSetupColumn("recall");
        ImGui::TableHeadersRow();
        for (const auto& c : r.per_class) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(c.name.c_str());
            ImGui::TableNextColumn();
            ImGui::Text("%lld", static_cast<long long>(c.support));
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(Percent(c.precision).c_str());
            ImGui::TableNextColumn();
            ImGui::TextColored(c.recall < 0.6 ? t.warning : t.text, "%s", Percent(c.recall).c_str());
        }
        ImGui::EndTable();
    }
    ImGui::Spacing();
    ImGui::TextColored(t.text_dim, "WORST ROWS (WRONG, MOST CONFIDENT FIRST)");
    if (r.worst.empty()) {
        ImGui::TextColored(t.success, "No wrong row.");
    } else if (ImGui::BeginTable("##worst", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_NoSavedSettings)) {
        const bool by_dataset = r.worst.front().dataset_row >= 0;
        ImGui::TableSetupColumn(by_dataset ? "dataset row" : "test row");
        ImGui::TableSetupColumn("true");
        ImGui::TableSetupColumn("predicted");
        ImGui::TableSetupColumn("confidence");
        ImGui::TableHeadersRow();
        for (const auto& w : r.worst) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%lld", static_cast<long long>(by_dataset ? w.dataset_row + 1 : w.row));
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(w.truth.c_str());
            ImGui::TableNextColumn();
            ImGui::TextColored(t.error, "%s", w.predicted.c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(Percent(w.confidence).c_str());
        }
        ImGui::EndTable();
    }
}

void ResultsView::RenderRegression() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (!regression_results_.ok) {
        ImGui::TextColored(t.error, "The test predictions cannot be read: %s", regression_results_.error.c_str());
        return;
    }
    const auto& r = regression_results_;
    const std::string rows = std::to_string(r.rows) + " test rows" +
                             (r.targets > 1 ? ", " + std::to_string(r.targets) + " targets" : std::string());
    Tile("MAE", Figure(r.mae), rows, t.text);
    ImGui::SameLine(0.0f, 28.0f);
    Tile("RMSE", Figure(r.rmse), "", t.text);
    ImGui::SameLine(0.0f, 28.0f);
    Tile("Bias", Figure(r.bias), r.bias > 0 ? "predicts high" : r.bias < 0 ? "predicts low" : "", t.text);
    ImGui::Spacing();
    ImGui::TextColored(t.text_dim, r.targets > 1 ? "LARGEST ERRORS (THE FIRST TARGET SHOWN)" : "LARGEST ERRORS");
    if (!r.worst.empty() &&
        ImGui::BeginTable("##worst_regression", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_NoSavedSettings)) {
        const bool by_dataset = r.worst.front().dataset_row >= 0;
        ImGui::TableSetupColumn(by_dataset ? "dataset row" : "test row");
        ImGui::TableSetupColumn("true");
        ImGui::TableSetupColumn("predicted");
        ImGui::TableSetupColumn("error");
        ImGui::TableHeadersRow();
        for (const auto& w : r.worst) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%lld", static_cast<long long>(by_dataset ? w.dataset_row + 1 : w.row));
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(Figure(w.truth).c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(Figure(w.predicted).c_str());
            ImGui::TableNextColumn();
            ImGui::TextColored(t.warning, "%s", Figure(w.error).c_str());
        }
        ImGui::EndTable();
    }
}

void ResultsView::SliceBy(const std::string& column) {
    if (slice_task_) AsyncTaskManager::Instance().Cancel(slice_task_);
    slice_column_ = column;
    slices_ = {};
    slices_error_.clear();
    const std::string source = row_errors_.source_dataset;
    const std::vector<int64_t> row_ids = row_errors_.row_ids;
    const std::vector<double> errors = row_errors_.errors;
    struct Outcome {
        studio::ErrorSlices slices;
        std::string error;
    };
    auto outcome = std::make_shared<Outcome>();
    std::weak_ptr<int> alive = alive_;
    slice_task_ = AsyncTaskManager::Instance().RunAsync(
        "Error by " + column,
        [source, column, row_ids, errors, outcome](LambdaTask& task) {
            studio::ColumnGatherer gatherer(row_ids);
            std::string error;
            bool ok = true;
            if (const auto memory = DataRegistry::Instance().GetArrowDataset(source)) {
                const auto table = memory->GetArrowTable();
                ok = table && gatherer.Add(table->GetColumnByName(column), error);
            } else if (const auto disk = DataRegistry::Instance().GetParquetBackedDataset(source)) {
                const int groups = disk->GetNumRowGroups();
                for (int g = 0; ok && g < groups; ++g) {
                    if (task.ShouldStop()) {
                        task.MarkFailed("Cancelled.");
                        return;
                    }
                    task.ReportProgress(static_cast<float>(g) / static_cast<float>(std::max(1, groups)),
                                        "row group " + std::to_string(g + 1) + " of " + std::to_string(groups));
                    ok = gatherer.Add(disk->ReadRowGroupColumn(g, column), error);
                }
            } else {
                ok = false;
                error = "'" + source + "' is no longer loaded; load it again to slice by its columns";
            }
            studio::GatheredColumn values;
            if (ok) ok = gatherer.Finish(values, error);
            if (!ok) {
                outcome->error = error.empty() ? "the column could not be read" : error;
                task.MarkFailed(outcome->error);
                return;
            }
            outcome->slices = studio::SliceErrors(values, errors);
        },
        nullptr,
        [this, alive, outcome](bool /*success*/, const std::string& /*message*/) {
            if (alive.expired()) return;
            slice_task_ = 0;
            slices_error_ = outcome->error == "Cancelled." ? std::string() : outcome->error;
            slices_ = outcome->slices;
        },
        alive_);
}

void ResultsView::RenderErrorsByValue() {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::TextColored(t.text_dim, "ERROR BY COLUMN VALUE");
    if (!row_errors_.ok) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_faint, "Not available: %s. Test again to record them.", row_errors_.error.c_str());
        ImGui::PopTextWrapPos();
        return;
    }
    if (slice_columns_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_faint, "The tested table '%s' is no longer loaded; load it again to slice the error by its columns.",
                           row_errors_.source_dataset.c_str());
        ImGui::PopTextWrapPos();
        return;
    }
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "Column");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(std::min(280.0f, ImGui::GetContentRegionAvail().x * 0.6f));
    if (ImGui::BeginCombo("##slice_column", slice_column_.empty() ? "Choose a column" : slice_column_.c_str())) {
        for (const auto& c : slice_columns_)
            if (ImGui::Selectable(c.c_str(), c == slice_column_)) SliceBy(c);
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_faint, "of '%s'", row_errors_.source_dataset.c_str());
    if (slice_column_.empty()) return;
    if (slice_task_) {
        ImGui::TextColored(t.text_dim, "%s Reading '%s'...", ICON_FA_SPINNER, slice_column_.c_str());
        return;
    }
    if (!slices_error_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.error, "%s", slices_error_.c_str());
        ImGui::PopTextWrapPos();
        return;
    }
    const bool classifier = row_errors_.classifier;
    const auto measure = [&](double x) { return classifier ? Percent(x) : Figure(x); };
    if (ImGui::BeginTable("##by_value", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_NoSavedSettings)) {
        ImGui::TableSetupColumn(slices_.binned ? "range" : "value");
        ImGui::TableSetupColumn("test rows");
        ImGui::TableSetupColumn(classifier ? "wrong" : "mean |error|");
        ImGui::TableHeadersRow();
        for (const auto& s : slices_.slices) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(s.value.c_str());
            ImGui::TableNextColumn();
            ImGui::Text("%lld", static_cast<long long>(s.rows));
            ImGui::TableNextColumn();
            // Above the overall error by a quarter or more stands out.
            ImGui::TextColored(s.error > slices_.overall * 1.25 && s.error > 0.0 ? t.warning : t.text, "%s", measure(s.error).c_str());
        }
        ImGui::EndTable();
    }
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_faint, "Overall %s over %zu test rows.%s A value with few rows can differ by chance.",
                       measure(slices_.overall).c_str(), row_errors_.errors.size(),
                       slices_.binned ? " Numbers are grouped in ranges of about equal rows." : "");
    ImGui::PopTextWrapPos();
}

}  // namespace cyxwiz
