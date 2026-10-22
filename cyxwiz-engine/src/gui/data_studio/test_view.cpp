#include "test_view.h"

#include "explore_plots.h"

#include "../../core/arrow_dataset.h"
#include "../../core/compiled_node_presentation.h"
#include "../../core/data_registry.h"
#include "../../core/dataset_catalog.h"
#include "../../core/file_dialogs.h"
#include "../../core/kept_tests.h"
#include "../../core/project_manager.h"
#include "../../core/test_results_export.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_fonts.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"

#include <cyxwiz/data_transform.h>

#include <arrow/api.h>
#include <implot.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <filesystem>

namespace cyxwiz {

namespace {

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

// The dataset row shown for test row i: 1-based, "" when not recorded.
std::string DatasetRow(const TestingMetrics& m, size_t i, size_t samples) {
    if (m.row_ids.size() != samples || i >= samples) return {};
    return std::to_string(m.row_ids[i] + 1);
}

std::string Fixed(double v, int decimals) {
    char text[48];
    std::snprintf(text, sizeof(text), "%.*f", decimals, v);
    return text;
}

std::shared_ptr<arrow::Table> RegisteredTable(const std::string& name) {
    const auto dataset = DataRegistry::Instance().GetArrowDataset(name);
    return dataset ? dataset->GetArrowTable() : nullptr;
}

// A combo of `options` (by value); true when the pick changed.
bool Pick(const char* id, std::string& value, const std::vector<std::string>& options, const char* none = nullptr,
          float width = 200.0f) {
    bool changed = false;
    ImGui::SetNextItemWidth((std::min)(width, ImGui::GetContentRegionAvail().x));
    const std::string shown = value.empty() && none ? none : value;
    if (ImGui::BeginCombo(id, shown.c_str())) {
        if (none && ImGui::Selectable(none, value.empty())) {
            changed = !value.empty();
            value.clear();
        }
        for (const auto& option : options)
            if (ImGui::Selectable(option.c_str(), option == value)) {
                changed = changed || option != value;
                value = option;
            }
        ImGui::EndCombo();
    }
    return changed;
}

void Label(const char* text) {
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(ui::CurrentTokens().text_dim, "%s", text);
    ImGui::SameLine();
}

}  // namespace

ImVec4 TestToneColour(test_report::Tone tone) {
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

void TestView::SetResults(const TestingMetrics& results) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (test_report::IsFailedRun(results)) {
        // The last successful results stay (and stay what Comparison compares).
        failure_ = results.status_message;
        spdlog::warn("Results > Test: the test failed: {}", results.status_message);
    } else {
        if (has_results_) {
            previous_ = results_;
            has_previous_ = true;
        }
        results_ = results;
        has_results_ = true;
        failure_.clear();
    }
    source_ = Source::Latest;
    note_.clear();
    dirty_ = true;
}

bool TestView::HasResults() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return has_results_;
}

void TestView::ShowKept(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    LoadKept(path);
}

void TestView::LoadKept(const std::string& path) {
    source_ = Source::Kept;
    dirty_ = true;
    std::string error;
    const auto rows = ReadKeptTest(path, error);
    TestingMetrics kept;
    if (!rows || !MetricsFromPredictionsTable(*rows, kept, error)) {
        kept_error_ = "Cannot open the kept test " + path + ": " + error;
        kept_path_ = path;
        kept_ = TestingMetrics{};
        spdlog::warn("Results > Test: {}", kept_error_);
        return;
    }
    std::string name = std::filesystem::path(path).stem().string();
    if (const auto metadata = rows->schema()->metadata()) {
        const int at = metadata->FindKey(kKeptTestNameKey);
        if (at >= 0 && !metadata->value(at).empty()) name = metadata->value(at);
    }
    kept_ = std::move(kept);
    kept_name_ = name;
    kept_path_ = path;
    kept_error_.clear();
}

void TestView::ShowComparison() {
    std::lock_guard<std::mutex> lock(mutex_);
    source_ = Source::Latest;  // Comparison is the latest run and the one before
    select_comparison_ = true;
    dirty_ = true;
}

void TestView::ShowTable() {
    std::lock_guard<std::mutex> lock(mutex_);
    source_ = Source::Table;
    dirty_ = true;
}

bool TestView::ExportReport() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!Shown()) return false;
    Export(true);
    return true;
}

const TestingMetrics* TestView::Shown() const {
    switch (source_) {
        case Source::Latest: return has_results_ ? &results_ : nullptr;
        case Source::Kept: return kept_error_.empty() && !kept_path_.empty() ? &kept_ : nullptr;
        case Source::Table: return table_answer_ ? &table_answer_->metrics : nullptr;
    }
    return nullptr;
}

void TestView::Rebuild() {
    const TestingMetrics* m = Shown();
    dirty_ = false;
    ++shown_version_;
    curves_key_.clear();
    if (!m) {
        view_ = test_report::View{};
        rows_ = nullptr;
        by_value_.SetRows(nullptr);
        return;
    }
    view_ = test_report::BuildView(*m, source_ == Source::Latest && has_previous_ ? &previous_ : nullptr);
    rows_ = PredictionsTableFromMetrics(*m);
    by_value_.SetRows(rows_);
    const int classes = static_cast<int>(m->per_class_metrics.size());
    if (positive_ >= classes) positive_ = -1;
}

void TestView::Render() {
    std::lock_guard<std::mutex> lock(mutex_);
    const ui::Tokens& t = ui::CurrentTokens();
    if (source_ == Source::Latest && !failure_.empty()) {
        ImGui::TextColored(t.error, ICON_FA_CIRCLE_XMARK " The test failed");
        Wrapped(t.text, failure_);
        Wrapped(t.text_dim, has_results_ ? "The last successful run's results are below."
                                         : "Task View shows the failed testing task; engine_log.txt has the details.");
        ImGui::Dummy(ImVec2(0.0f, t.space_sm));
    }
    RenderSource();
    if (source_ == Source::Table) {
        const TableAnswer* answer = table_job_.Answer(table_key_);
        if (answer != table_answer_) {
            table_answer_ = answer;
            dirty_ = true;
        }
    }
    if (dirty_) Rebuild();
    const TestingMetrics* m = Shown();
    if (!m) {
        if (source_ == Source::Latest && failure_.empty())
            ui::EmptyState(ICON_FA_GAUGE, "No test results yet",
                           "Train the graph, then Train > Test > Run Test (F8). Or show A table... of predictions.");
        if (source_ == Source::Kept && !kept_error_.empty()) Wrapped(t.error, kept_error_);
        return;
    }
    RenderHeader(*m);
    const bool classification = view_.classification;
    const bool curves = classification && !m->metric_learning_mode && m->roc && m->roc->success && m->pr && m->pr->success;
    const float footer_h = ui::ButtonHeight(ui::ButtonSize::Regular) + ImGui::GetStyle().ItemSpacing.y * 2.0f;
    if (ImGui::BeginTabBar("##test_sections", ImGuiTabBarFlags_FittingPolicyScroll)) {
        const auto section = [&](const char* label, const auto& body, ImGuiTabItemFlags flags = 0) {
            if (!ImGui::BeginTabItem(label, nullptr, flags)) return;
            // Fills the job above the buttons; in a short dock it keeps a
            // usable height and the job scrolls to the buttons.
            ImGui::BeginChild("##section", ImVec2(0.0f, (std::max)(ImGui::GetContentRegionAvail().y - footer_h, 300.0f)));
            body();
            ImGui::EndChild();
            ImGui::EndTabItem();
        };
        section("Overview###overview", [&] { RenderOverview(); });
        if (classification) {
            section("Confusion matrix###matrix", [&] { RenderMatrix(); });
            section("Classes###classes", [&] { RenderClasses(); });
            if (!m->metric_learning_mode) section("Curves###curves", [&] { RenderCurves(*m); });
            if (curves) section("Threshold###threshold", [&] { RenderThreshold(*m); });
            section("Predictions###predictions", [&] { RenderPredictions(*m); });
        } else if (m->regression_mode) {
            section("Plots###plots", [&] { RenderRegressionPlots(*m); });
            section("Predictions###predictions", [&] { RenderRegressionRows(*m); });
        }
        if (rows_) section("By column value###by_value", [&] { by_value_.Render(); });
        if (source_ == Source::Latest)
            section("Comparison###comparison", [&] { RenderComparison(); },
                    select_comparison_ ? ImGuiTabItemFlags_SetSelected : ImGuiTabItemFlags_None);
        select_comparison_ = false;
        ImGui::EndTabBar();
    }
    RenderFooter(*m);
}

// Showing: the latest test, a kept test (results/tests in the project), or a table.
void TestView::RenderSource() {
    const ui::Tokens& t = ui::CurrentTokens();
    const std::string root = ProjectManager::Instance().GetProjectRoot();
    Label("Showing");
    ImGui::SetNextItemWidth((std::min)(300.0f, ImGui::GetContentRegionAvail().x));
    const std::string current = source_ == Source::Kept    ? "Kept: " + kept_name_
                                : source_ == Source::Table ? std::string("A table...")
                                                           : std::string(has_results_ ? "Latest test" : "Latest test (none yet)");
    if (ImGui::BeginCombo("##showing", current.c_str())) {
        if (ImGui::Selectable(has_results_ ? "Latest test" : "Latest test (none yet)", source_ == Source::Latest)) {
            source_ = Source::Latest;
            dirty_ = true;
        }
        const auto kept = root.empty() ? std::vector<KeptTest>{} : ListKeptTests(KeptTestsFolder(root));
        for (const auto& k : kept) {
            const std::string label = "Kept: " + k.name + "##" + k.path.string();
            if (ImGui::Selectable(label.c_str(), source_ == Source::Kept && kept_path_ == k.path.string())) LoadKept(k.path.string());
        }
        if (kept.empty() && !root.empty()) ImGui::TextColored(t.text_faint, "No kept tests yet (Keep as node keeps one)");
        if (ImGui::Selectable("A table...", source_ == Source::Table)) {
            source_ = Source::Table;
            dirty_ = true;
        }
        ImGui::EndCombo();
    }
    if (source_ == Source::Kept && !kept_path_.empty()) {
        ImGui::SameLine();
        ImGui::AlignTextToFramePadding();
        std::error_code ec;
        const std::string shown = root.empty() ? kept_path_ : std::filesystem::relative(kept_path_, root, ec).generic_string();
        ImGui::TextColored(t.text_faint, "%s", (ec || shown.empty() ? kept_path_ : shown).c_str());
    }
    if (source_ == Source::Table) RenderTableChoices();
    ImGui::Dummy(ImVec2(0.0f, t.space_xs));
}

// A table read as a test: the table, classes or numbers, the true, predicted
// and score columns (a first guess by name), read in the background.
void TestView::RenderTableChoices() {
    const ui::Tokens& t = ui::CurrentTokens();
    // The tables in memory (the catalog lists every kind of dataset).
    std::vector<std::string> tables;
    for (const auto& entry : DatasetCatalog::Instance().List())
        if (DataRegistry::Instance().GetArrowDataset(entry.name)) tables.push_back(entry.name);
    ImGui::SameLine();
    Pick("##table", table_name_, tables, "Pick a table", 240.0f);
    const auto table = table_name_.empty() ? nullptr : RegisteredTable(table_name_);
    if (!table) {
        Wrapped(t.text_dim, tables.empty() ? "No table is loaded. Bring one in (a Data Input), or run a step that makes one."
                                           : "Pick a table with a true column and a predicted column, and scores for the "
                                             "curves: a model scored in a script, an imported CSV, a node's result.");
        return;
    }
    if (columns_for_ != table_name_) {
        columns_for_ = table_name_;
        columns_ = GuessPredictionColumns(*table);
    }
    columns_.dataset = table_name_;
    std::vector<std::string> names, numbers;
    for (const auto& field : table->schema()->fields()) {
        names.push_back(field->name());
        if (field->type()->id() != arrow::Type::STRING && field->type()->id() != arrow::Type::LARGE_STRING &&
            field->type()->id() != arrow::Type::BOOL)
            numbers.push_back(field->name());
    }
    ui::FlowRow row;
    static const char* const kKinds[] = {"Classes", "Numbers"};
    int kind = columns_.numbers ? 1 : 0;
    row.Next(ImGui::CalcTextSize("Classes Numbers").x + 40.0f);
    if (ui::SegmentedControl("##kind", kKinds, 2, &kind)) {
        columns_.numbers = kind == 1;
        if (columns_.numbers) columns_.scores.clear();
    }
    row.Next(ImGui::CalcTextSize("True").x + 200.0f);
    Label("True");
    Pick("##truth", columns_.truth, names, "pick", 180.0f);
    row.Next(ImGui::CalcTextSize("Predicted").x + 200.0f);
    Label("Predicted");
    Pick("##predicted", columns_.predicted, names, "pick", 180.0f);
    if (!columns_.numbers) {
        // Scores: none, one per class (score_<class>), or one column (two classes).
        std::vector<std::string> per_class;
        for (const auto& name : names)
            if (name.rfind("score_", 0) == 0) per_class.push_back(name);
        std::string scores = columns_.scores.empty() ? std::string() : columns_.scores.size() == 1 ? columns_.scores[0]
                                                                                                    : "score_ (one per class)";
        std::vector<std::string> options;
        if (per_class.size() >= 2) options.push_back("score_ (one per class)");
        for (const auto& name : numbers)
            if (name != columns_.truth && name != columns_.predicted) options.push_back(name);
        row.Next(ImGui::CalcTextSize("Scores").x + 220.0f);
        Label("Scores");
        if (Pick("##scores", scores, options, "none (no curves)", 200.0f)) {
            columns_.positive.clear();
            if (scores.empty()) columns_.scores.clear();
            else if (scores == "score_ (one per class)") columns_.scores = per_class;
            else columns_.scores = {scores};
        }
    }

    const std::string key = table_name_ + "#" + std::to_string(reinterpret_cast<uintptr_t>(table.get())) + "|" + columns_.truth +
                            "|" + columns_.predicted + "|" + (columns_.numbers ? "n" : "c") + "|" + columns_.positive + "|" +
                            [&] {
                                std::string s;
                                for (const auto& c : columns_.scores) s += c + ",";
                                return s;
                            }();
    if (columns_.truth.empty() || columns_.predicted.empty()) {
        Wrapped(t.text_dim, "Pick the true column and the predicted column.");
        return;
    }
    if (key != table_key_) {
        table_key_ = key;
        const PredictionColumns columns = columns_;
        table_job_.Ask("Reading a table as a test", key, [table, columns](LambdaTask&) {
            TableAnswer a;
            if (!MetricsFromColumns(*table, columns, a.metrics, a.left_out, a.error) && a.error.empty())
                a.error = "the columns cannot be read as a test";
            return a;
        });
    }
    if (table_job_.Running()) {
        ImGui::TextColored(t.text_dim, ICON_FA_SPINNER " Reading the table...");
    } else if (!table_job_.Error().empty()) {
        Wrapped(t.error, table_job_.Error());
    } else if (const TableAnswer* a = table_job_.Answer(table_key_)) {
        const auto& m = a->metrics;
        // One score for two classes: whose score it is.
        if (columns_.scores.size() == 1 && m.per_class_metrics.size() == 2) {
            std::vector<std::string> classes = {test_report::ClassName(m, 0), test_report::ClassName(m, 1)};
            std::string of = columns_.positive.empty() ? classes[1] : columns_.positive;
            Label("The score is the chance of");
            if (Pick("##score_of", of, classes, nullptr, 160.0f)) columns_.positive = of;
        }
        if (a->left_out > 0)
            Wrapped(t.text_dim, FormatCompiledCount(a->left_out) + " rows without a true or predicted value (or score) are left out.");
    }
}

void TestView::RenderHeader(const TestingMetrics& m) {
    const ui::Tokens& t = ui::CurrentTokens();
    (void)m;
    {
        ui::FontScope heading(ui::Font::Heading);
        ImGui::TextColored(TestToneColour(view_.headline_tone), "%s", view_.headline.c_str());
    }
    Wrapped(t.text_dim, view_.subtitle);
    if (!view_.note.empty()) Wrapped(t.caution, view_.note);
    ui::FlowRow chips;
    for (const auto& chip : view_.chips) {
        chips.Next(ui::ChipWidth(chip.text.c_str()));
        ui::Chip(chip.text.c_str(), TestToneColour(chip.tone));
    }
    ImGui::Dummy(ImVec2(0.0f, t.space_xs));
}

void TestView::RenderOverview() {
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

void TestView::RenderMatrix() {
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
    if (n > 60) {
        Wrapped(t.text_dim, std::to_string(n) + " classes are too many to draw as a grid; Classes lists each one's figures.");
        return;
    }
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
                ImGui::Text("True %s, predicted %s", mx.names[static_cast<size_t>(i)].c_str(), mx.names[static_cast<size_t>(j)].c_str());
                ImGui::Text("%d of %d (%.1f%% of the true class)", value, total, total > 0 ? 100.0 * value / total : 0.0);
                ImGui::EndTooltip();
            }
        }
    }
    ImGui::EndTable();
}

void TestView::RenderClasses() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (view_.classes.empty()) {
        ImGui::TextColored(t.text_dim, "No per-class metrics for this run.");
        return;
    }
    ImGui::SetNextItemWidth((std::min)(240.0f, ImGui::GetContentRegionAvail().x));
    ImGui::InputTextWithHint("##filter", "Filter classes", filter_, sizeof(filter_));
    // Fit to the headers and numbers; a narrow panel scrolls sideways.
    const ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_Sortable | ImGuiTableFlags_ScrollY |
                                  ImGuiTableFlags_ScrollX | ImGuiTableFlags_SizingFixedFit;
    if (!ImGui::BeginTable("##classes", 7, flags)) return;
    ImGui::TableSetupScrollFreeze(1, 1);
    ImGui::TableSetupColumn("Class", ImGuiTableColumnFlags_DefaultSort);
    for (const char* column : {"Precision", "Recall", "F1", "Found", "False alarms", "Support"}) ImGui::TableSetupColumn(column);
    ImGui::TableHeadersRow();
    if (ImGuiTableSortSpecs* specs = ImGui::TableGetSortSpecs(); specs && specs->SpecsCount > 0) {
        static const test_report::ClassSort kColumns[] = {
            test_report::ClassSort::Name, test_report::ClassSort::Precision, test_report::ClassSort::Recall,
            test_report::ClassSort::F1,   test_report::ClassSort::Found,     test_report::ClassSort::FalseAlarms,
            test_report::ClassSort::Support};
        const auto& spec = specs->Specs[0];
        sort_by_ = kColumns[std::clamp(static_cast<int>(spec.ColumnIndex), 0, 6)];
        sort_ascending_ = spec.SortDirection == ImGuiSortDirection_Ascending;
    }
    for (const auto& row : test_report::SortClasses(view_.classes, sort_by_, sort_ascending_, filter_)) {
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

// The positive class (two classes) or the class against the rest, by name.
void TestView::RenderPositiveClass(const TestingMetrics& m) {
    const int classes = static_cast<int>(m.per_class_metrics.size());
    const int shown = positive_ >= 0 ? positive_ : classes == 2 ? 1 : 0;
    Label(classes == 2 ? "Positive class" : "Class against the rest");
    ImGui::SetNextItemWidth((std::min)(200.0f, ImGui::GetContentRegionAvail().x));
    const std::string current = test_report::ClassName(m, shown);
    if (ImGui::BeginCombo("##positive", current.c_str())) {
        for (int c = 0; c < classes; ++c) {
            ImGui::PushID(c);
            if (ImGui::Selectable(test_report::ClassName(m, c).c_str(), c == shown)) positive_ = c;
            ImGui::PopID();
        }
        ImGui::EndCombo();
    }
}

bool TestView::EnsureCurves(const TestingMetrics& m) {
    const int classes = static_cast<int>(m.per_class_metrics.size());
    const int positive = positive_ >= 0 ? positive_ : classes == 2 ? 1 : 0;
    const std::string key = std::to_string(shown_version_) + ":" + std::to_string(positive);
    if (key != curves_key_) {
        curves_key_ = key;
        curves_error_.clear();
        if (!CurvesForClass(m, positive, curves_, curves_error_) && curves_error_.empty()) curves_error_ = "no curve for this class";
    }
    return curves_error_.empty();
}

void TestView::RenderCurves(const TestingMetrics& m) {
    const ui::Tokens& t = ui::CurrentTokens();
    const auto& roc = m.roc;
    const auto& pr = m.pr;
    if (!roc || !roc->success || !pr || !pr->success) {
        ui::EmptyState(ICON_FA_CHART_AREA, "No curves for this run",
                       m.curves_note.empty() ? "Run the test again to compute ROC and precision-recall." : m.curves_note.c_str());
        return;
    }
    const bool multiclass = !roc->class_auc.empty();
    {
        ui::FontScope heading(ui::Font::Heading);
        ImGui::Text("AUC %.4f", roc->auc);
        ImGui::SameLine(0.0f, 24.0f);
        ImGui::Text("AP %.4f", pr->average_precision);
    }
    Wrapped(t.text_dim, multiclass ? "One-vs-rest per class; AUC and AP are the macro means. The ring marks the picked class at the threshold."
                                   : "The ring marks the threshold (Threshold sets it); the curves show every threshold.");
    if (!m.curves_note.empty()) Wrapped(t.caution, m.curves_note);
    RenderPositiveClass(m);
    const bool have = EnsureCurves(m);
    if (!have) Wrapped(t.caution, curves_error_);
    const test_report::ThresholdFigures at =
        have ? test_report::AtThreshold(curves_.roc, curves_.positives, curves_.negatives, threshold_) : test_report::ThresholdFigures{};
    ImGui::Spacing();

    const float plot_height = (std::max)(ImGui::GetContentRegionAvail().y - ImGui::GetStyle().ItemSpacing.y, 220.0f);
    if (!ImGui::BeginTable("##curves", 2, ImGuiTableFlags_SizingStretchSame)) return;
    const int picked = positive_ >= 0 ? positive_ : static_cast<int>(m.per_class_metrics.size()) == 2 ? 1 : 0;
    const auto ring = [&](double x, double y) {
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 6.0f, ImVec4(0, 0, 0, 0), 2.0f, t.text);
        ImPlot::PlotScatter("##threshold", &x, &y, 1);
    };
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
                std::snprintf(label, sizeof(label), "%s (AUC %.3f)", test_report::ClassName(m, static_cast<int>(c)).c_str(),
                              roc->class_auc[c]);
                ImPlot::SetNextLineStyle(ui::SeriesColour(t, c), static_cast<int>(c) == picked ? 3.0f : 1.5f);
                ImPlot::PlotLine(label, roc->class_fpr[c].data(), roc->class_tpr[c].data(), static_cast<int>(roc->class_fpr[c].size()));
            }
        } else if (have) {
            ImPlot::SetNextLineStyle(t.series[1], 2.0f);
            ImPlot::PlotLine("ROC", curves_.roc.fpr.data(), curves_.roc.tpr.data(), static_cast<int>(curves_.roc.fpr.size()));
        }
        if (at.ok) ring(static_cast<double>(at.fp) / curves_.negatives, at.recall);
        ImPlot::EndPlot();
    }
    ImGui::TableNextColumn();
    if (ImPlot::BeginPlot("Precision-recall", ImVec2(-1, plot_height), ImPlotFlags_NoMenus)) {
        ImPlot::SetupAxes("Recall", "Precision");
        ImPlot::SetupAxesLimits(-0.02, 1.02, -0.02, 1.02, ImPlotCond_Always);
        // Steps, so the area is AP (precision at a recall holds back to the recall before).
        const auto stairs = [](const char* label, std::vector<double> recall, std::vector<double> precision) {
            std::reverse(recall.begin(), recall.end());
            std::reverse(precision.begin(), precision.end());
            ImPlot::PlotStairs(label, recall.data(), precision.data(), static_cast<int>(recall.size()), ImPlotStairsFlags_PreStep);
        };
        if (multiclass) {
            for (size_t c = 0; c < pr->class_recall.size(); ++c) {
                if (pr->class_recall[c].empty()) continue;
                char label[160];
                std::snprintf(label, sizeof(label), "%s (AP %.3f)", test_report::ClassName(m, static_cast<int>(c)).c_str(),
                              pr->class_ap[c]);
                ImPlot::SetNextLineStyle(ui::SeriesColour(t, c), static_cast<int>(c) == picked ? 3.0f : 1.5f);
                stairs(label, pr->class_recall[c], pr->class_precision[c]);
            }
        } else if (have) {
            ImPlot::SetNextLineStyle(t.series[0], 2.0f);
            stairs("PR", curves_.pr.recall, curves_.pr.precision);
        }
        if (at.ok && at.tp + at.fp > 0) ring(at.recall, at.precision);
        ImPlot::EndPlot();
    }
    ImGui::EndTable();
}

// The figures at a threshold, read off the class's ROC (no pass over the rows).
void TestView::RenderThreshold(const TestingMetrics& m) {
    const ui::Tokens& t = ui::CurrentTokens();
    RenderPositiveClass(m);
    if (!EnsureCurves(m)) {
        Wrapped(t.caution, curves_error_);
        return;
    }
    const std::string positive = test_report::ClassName(m, positive_ >= 0 ? positive_ : m.per_class_metrics.size() == 2 ? 1 : 0);
    const bool two = m.per_class_metrics.size() == 2;
    const std::string rest = two ? test_report::ClassName(m, positive_ == 0 ? 1 : 0) : "the rest";
    ImGui::SameLine();
    Label("Threshold");
    ui::NumberFieldOptions options;
    options.min = 0.0;
    options.max = 1.0;
    options.step = 0.01;
    options.format = "%.3f";
    options.hint = "The class is predicted when its score is at least this";
    ui::NumberField("##threshold", &threshold_, options);
    ImGui::SameLine();
    const double best = test_report::BestF1Threshold(curves_.roc, curves_.positives, curves_.negatives);
    if (ui::SecondaryButton("Best F1", std::isfinite(best), "No threshold gives a positive", ui::ButtonSize::Small))
        threshold_ = best;
    const test_report::ThresholdFigures f = test_report::AtThreshold(curves_.roc, curves_.positives, curves_.negatives, threshold_);
    if (!f.ok) {
        Wrapped(t.text_dim, "The test needs samples of " + positive + " and of " + rest + ".");
        return;
    }
    Wrapped(t.text_dim, "At " + Fixed(threshold_, 3) + ": " + positive + " when its score is at least " + Fixed(threshold_, 3) +
                            ". AUC " + Fixed(curves_.roc.auc, 4) + ", AP " + Fixed(curves_.pr.average_precision, 4) + ".");
    ImGui::Spacing();
    if (ImGui::BeginTable("##threshold_split", 2, ImGuiTableFlags_SizingStretchSame)) {
        ImGui::TableNextColumn();
        ExploreFigures("##threshold_figures", {{"precision", Fixed(f.precision, 4)},
                                               {"recall (found)", Fixed(f.recall, 4)},
                                               {"specificity", Fixed(f.specificity, 4)},
                                               {"F1", Fixed(f.f1, 4)},
                                               {"balanced accuracy", Fixed(f.balanced_accuracy, 4)},
                                               {"MCC", Fixed(f.mcc, 4)},
                                               {"accuracy", Fixed(f.accuracy, 4)}});
        ImGui::TableNextColumn();
        if (ImGui::BeginTable("##threshold_counts", 3, ImGuiTableFlags_SizingFixedFit)) {
            ImGui::TableSetupColumn("");
            ImGui::TableSetupColumn(("predicted " + positive).c_str());
            ImGui::TableSetupColumn(("predicted " + rest).c_str());
            ImGui::TableHeadersRow();
            const auto line = [&](const std::string& name, int a, int b, bool positive_row) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(t.text_dim, "actual %s", name.c_str());
                ImGui::TableNextColumn();
                ImGui::TextColored(positive_row ? t.success : t.warning, "%d", a);
                ImGui::TableNextColumn();
                ImGui::TextColored(positive_row ? t.warning : t.success, "%d", b);
            };
            line(positive, f.tp, f.fn, true);
            line(rest, f.fp, f.tn, false);
            ImGui::EndTable();
        }
        Wrapped(t.text_dim, std::to_string(f.fp) + " of " + rest + " called " + positive + ", " + std::to_string(f.fn) + " " +
                                positive + " missed, of " + FormatCompiledCount(f.tp + f.fp + f.tn + f.fn) + ".");
        ImGui::EndTable();
    }
}

void TestView::RenderPredictions(const TestingMetrics& m) {
    const ui::Tokens& t = ui::CurrentTokens();
    const size_t samples = std::min(m.predictions.size(), m.ground_truth.size());
    if (samples == 0) {
        ImGui::TextColored(t.text_dim, "No predictions were kept for this run.");
        return;
    }
    const std::string all = "All " + FormatCompiledCount(static_cast<long long>(samples));
    const std::string mistakes =
        "Only mistakes " + FormatCompiledCount(static_cast<long long>(view_.mistakes.size())) + ", most confident first";
    const char* modes[] = {all.c_str(), mistakes.c_str()};
    ui::SegmentedControl("##predictions_mode", modes, 2, &predictions_mode_);
    const bool by_dataset = m.row_ids.size() == samples;
    const ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp;
    if (!ImGui::BeginTable("##predictions", by_dataset ? 5 : 4, flags)) return;
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableSetupColumn("#", 0, 0.5f);
    if (by_dataset) ImGui::TableSetupColumn("Dataset row", 0, 0.8f);
    ImGui::TableSetupColumn("Predicted", 0, 1.2f);
    ImGui::TableSetupColumn("Actual", 0, 1.2f);
    ImGui::TableSetupColumn("Confidence", 0, 1.2f);
    ImGui::TableHeadersRow();
    // The clipper walks the rows that are shown, so scrolling stays even with only the mistakes.
    const bool only_mistakes = predictions_mode_ == 1;
    ImGuiListClipper clipper;
    clipper.Begin(only_mistakes ? static_cast<int>(view_.mistakes.size()) : static_cast<int>(samples));
    while (clipper.Step()) {
        for (int r = clipper.DisplayStart; r < clipper.DisplayEnd; ++r) {
            const size_t i = static_cast<size_t>(only_mistakes ? view_.mistakes[static_cast<size_t>(r)] : r);
            const int predicted = m.predictions[i];
            const int actual = m.ground_truth[i];
            const float confidence = i < m.confidences.size() ? m.confidences[i] : 0.0f;
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%zu", i);
            if (by_dataset) {
                ImGui::TableNextColumn();
                const std::string row = DatasetRow(m, i, samples);
                ImGui::TextUnformatted(row.c_str());
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Row %s of '%s' (1 is its first row)", row.c_str(), m.source_dataset.c_str());
            }
            ImGui::TableNextColumn();
            ImGui::TextColored(predicted == actual ? t.success : t.error, "%s", test_report::ClassName(m, predicted).c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(test_report::ClassName(m, actual).c_str());
            ImGui::TableNextColumn();
            char overlay[16];
            std::snprintf(overlay, sizeof(overlay), "%.0f%%", confidence * 100.0f);
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, confidence > 0.8f ? t.success : confidence > 0.5f ? t.warning : t.caution);
            ImGui::ProgressBar(confidence, ImVec2(-FLT_MIN, 0.0f), overlay);
            ImGui::PopStyleColor();
        }
    }
    ImGui::EndTable();
}

void TestView::RenderRegressionRows(const TestingMetrics& m) {
    const ui::Tokens& t = ui::CurrentTokens();
    const size_t width = static_cast<size_t>(std::max(m.regression_width, 0));
    const size_t samples = width ? m.regression_truth.size() / width : 0;
    if (samples == 0) {
        Wrapped(t.text_dim, m.regression_note.empty() ? "No rows were kept for this run." : m.regression_note);
        return;
    }
    const std::string all = "All " + FormatCompiledCount(static_cast<long long>(samples));
    const char* modes[] = {all.c_str(), "Largest errors first"};
    ui::SegmentedControl("##regression_rows_mode", modes, 2, &regression_rows_mode_);
    if (width > 1)
        Wrapped(t.text_dim, std::to_string(width) + " targets: the first is shown; the order uses each row's mean |error| over all of them.");
    const bool by_dataset = m.row_ids.size() == samples;
    const ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp;
    if (!ImGui::BeginTable("##regression_rows", by_dataset ? 5 : 4, flags)) return;
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableSetupColumn("#", 0, 0.5f);
    if (by_dataset) ImGui::TableSetupColumn("Dataset row", 0, 0.8f);
    ImGui::TableSetupColumn("True", 0, 1.0f);
    ImGui::TableSetupColumn("Predicted", 0, 1.0f);
    ImGui::TableSetupColumn("Error", 0, 1.0f);
    ImGui::TableHeadersRow();
    const bool largest = regression_rows_mode_ == 1 && view_.largest_errors.size() == samples;
    ImGuiListClipper clipper;
    clipper.Begin(static_cast<int>(samples));
    while (clipper.Step()) {
        for (int r = clipper.DisplayStart; r < clipper.DisplayEnd; ++r) {
            const size_t i = largest ? static_cast<size_t>(view_.largest_errors[static_cast<size_t>(r)]) : static_cast<size_t>(r);
            const float truth = m.regression_truth[i * width];
            const float predicted = m.regression_predictions[i * width];
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%zu", i);
            if (by_dataset) {
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(DatasetRow(m, i, samples).c_str());
            }
            ImGui::TableNextColumn();
            ImGui::Text("%.4g", truth);
            ImGui::TableNextColumn();
            ImGui::Text("%.4g", predicted);
            ImGui::TableNextColumn();
            ImGui::TextColored(t.warning, "%+.4g", predicted - truth);
        }
    }
    ImGui::EndTable();
}

// Predicted vs actual, residuals vs predicted, the residual histogram (first
// target), drawn as Explore > Regression draws its plots.
void TestView::RenderRegressionPlots(const TestingMetrics& m) {
    const ui::Tokens& t = ui::CurrentTokens();
    const size_t width = static_cast<size_t>(std::max(m.regression_width, 0));
    const size_t samples = width ? m.regression_truth.size() / width : 0;
    if (samples == 0) {
        Wrapped(t.text_dim, m.regression_note.empty() ? "No rows were kept for this run." : m.regression_note);
        return;
    }
    if (plotted_version_ != shown_version_) {
        plotted_version_ = shown_version_;
        truth_.resize(samples);
        predicted_.resize(samples);
        residuals_.resize(samples);
        for (size_t i = 0; i < samples; ++i) {
            truth_[i] = m.regression_truth[i * width];
            predicted_[i] = m.regression_predictions[i * width];
            residuals_[i] = predicted_[i] - truth_[i];
        }
        const auto [lo, hi] = std::minmax_element(residuals_.begin(), residuals_.end());
        bins_lo_ = *lo;
        bins_hi_ = *hi;
        bins_ = DataTransform::Histogram(residuals_, 24, bins_lo_, bins_hi_);
    }
    static const char* const kPlots[] = {"Predicted vs actual", "Residuals vs predicted", "Residual histogram"};
    ui::SegmentedControl("##regression_plot", kPlots, 3, &plot_);
    if (width > 1) Wrapped(t.text_dim, std::to_string(width) + " targets: the first is drawn.");
    const float w = ImGui::GetContentRegionAvail().x;
    const float h = std::clamp(ImGui::GetContentRegionAvail().y - ImGui::GetTextLineHeightWithSpacing() * 3.0f, 140.0f, 420.0f);
    double bias = 0.0;
    for (const double r : residuals_) bias += r;
    bias /= static_cast<double>(residuals_.size());
    switch (plot_) {
        case 1:
            explore_plot::Scatter(predicted_, residuals_, w, h, nullptr, 1, "predicted", "residual (predicted - actual)");
            Wrapped(t.text_dim, "Points spread evenly around 0, with no funnel or curve, mean the errors are about the same size everywhere.");
            break;
        case 2:
            explore_plot::Bars(bins_, bins_lo_, bins_hi_, w, h * 0.8f);
            Wrapped(t.text_dim, "Residuals (predicted - actual); their mean, the bias, is " + Fixed(bias, 4) + ".");
            break;
        default:
            explore_plot::Scatter(truth_, predicted_, w, h, nullptr, 2, "actual", "predicted");
            Wrapped(t.text_dim, "On the line: right. Above it: predicted too high; below: too low.");
            break;
    }
}

void TestView::RenderComparison() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (view_.comparison.empty()) {
        Wrapped(t.text_dim, "One run so far. Run a second test to compare it with this one.");
        return;
    }
    Wrapped(t.text_dim, "The run before beside the latest run. Change is latest minus before; for loss, MAE and RMSE a negative "
                        "change is an improvement.");
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
            ImGui::TextColored(TestToneColour(row.change_tone), "%s", row.change.c_str());
        }
        ImGui::EndTable();
    }
    if (!view_.comparison_note.empty()) Wrapped(t.text_dim, view_.comparison_note);
}

void TestView::RenderFooter(const TestingMetrics& m) {
    const ui::Tokens& t = ui::CurrentTokens();
    (void)m;
    ImGui::Dummy(ImVec2(0.0f, 2.0f));
    ui::FlowRow row;
    row.Next(ui::ButtonWidth("Export CSV...", ui::ButtonSize::Small));
    if (ui::SecondaryButton("Export CSV...", true, nullptr, ui::ButtonSize::Small)) Export(false);
    row.Next(ui::ButtonWidth("Export JSON...", ui::ButtonSize::Small));
    if (ui::SecondaryButton("Export JSON...", true, nullptr, ui::ButtonSize::Small)) Export(true);
    const bool latest = source_ == Source::Latest;
    if (latest && keep) {
        const std::string blocker = keep_blocker ? keep_blocker() : std::string();
        row.Next(ui::ButtonWidth("Keep as node...", ui::ButtonSize::Small));
        if (ui::SecondaryButton("Keep as node...", blocker.empty(), blocker.c_str(), ui::ButtonSize::Small)) keep();
    }
    if (latest) {
        row.Next(ui::ButtonWidth("Clear", ui::ButtonSize::Small));
        if (ui::DangerButton("Clear", true, nullptr, ui::ButtonSize::Small)) ImGui::OpenPopup("Clear the latest test?");
    }
    if (run_test && latest) {
        row.Next(ui::ButtonWidth("Run test again", ui::ButtonSize::Regular));
        if (ui::PrimaryButton("Run test again")) run_test();
    }
    if (!note_.empty()) Wrapped(note_failed_ ? t.error : t.success, note_);
    if (ImGui::BeginPopupModal("Clear the latest test?", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings)) {
        ImGui::TextUnformatted("The latest test and the one before it are cleared from Results.");
        ImGui::TextColored(t.text_dim, "Kept tests stay; test_predictions stays until the next test.");
        ImGui::Spacing();
        if (ui::DangerButton("Clear")) {
            results_ = TestingMetrics();
            previous_ = TestingMetrics();
            has_results_ = false;
            has_previous_ = false;
            failure_.clear();
            note_.clear();
            dirty_ = true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ui::SecondaryButton("Cancel")) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
}

void TestView::Export(bool json) {
    const TestingMetrics* m = Shown();
    if (!m) return;
    const std::string folder = ExportsFolder();
    const auto path = json ? FileDialogs::SaveFile("Export Test Results", {{"JSON Files", "json"}, {"All Files", "*"}},
                                                   folder.empty() ? nullptr : folder.c_str(), "test_results.json")
                           : FileDialogs::SaveFile("Export Test Results", {{"CSV Files", "csv"}, {"All Files", "*"}},
                                                   folder.empty() ? nullptr : folder.c_str(), "test_results.csv");
    if (!path) return;
    std::string error;
    note_failed_ = !(json ? WriteTestResultsJson(*m, *path, &error) : WriteTestResultsCsv(*m, *path, &error));
    note_ = note_failed_ ? "Could not export: " + error : "Saved " + *path;
}

}  // namespace cyxwiz
