#include "model_panel.h"

#include "../../core/extension_node_metadata.h"
#include "../../core/file_dialogs.h"
#include "../../core/live_graph_compile.h"
#include "../../core/training_manager.h"
#include "../icons.h"
#include "../node_editor.h"
#include "../ui_buttons.h"
#include "../ui_fonts.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"

#include <imgui.h>
#include <implot.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>

namespace cyxwiz {

namespace {

void Paragraph(const std::string& text, const ImVec4* colour = nullptr) {
    if (text.empty()) return;
    if (colour) ImGui::PushStyleColor(ImGuiCol_Text, *colour);
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(text.c_str());
    ImGui::PopTextWrapPos();
    if (colour) ImGui::PopStyleColor();
}

const ImVec4& StatusTone(CompiledStatusKind kind) {
    const auto& t = ui::CurrentTokens();
    switch (kind) {
        case CompiledStatusKind::Ok: return t.success;
        case CompiledStatusKind::Failed: return t.caution;
        case CompiledStatusKind::Pending: return t.running;
        case CompiledStatusKind::None: break;
    }
    return t.pending;
}

// One tile: a muted label, the value large, the line under it.
void Tile(const char* id, const char* label, const std::string& value, const std::string& note) {
    const auto& t = ui::CurrentTokens();
    ui::BeginCard(id);
    ImGui::TextColored(t.text_dim, "%s", label);
    {
        const ui::FontScope heading(ui::Font::Heading);
        ImGui::TextColored(t.text_bright, "%s", value.c_str());
    }
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_dim, "%s", note.c_str());
    ImGui::PopTextWrapPos();
    ui::EndCard();
}

std::string Percent(double share) {
    if (share <= 0.0) return "";
    if (share < 0.01) return "<1 %";
    char text[16];
    std::snprintf(text, sizeof(text), "%.0f %%", share * 100.0);
    return text;
}

}  // namespace

void ModelPanel::Open(const std::string& tab) {
    visible_ = true;
    open_tab_ = tab;
    ImGui::SetWindowFocus("Model###ModelTools");
}

void ModelPanel::Refresh() {
    const LiveGraphCompile* live = node_editor_ ? node_editor_->GetLiveCompile() : nullptr;
    const uint64_t key = live ? live->ResultSerial() * 16 + (live->ParametersCounting() ? 1 : 0) +
                                    (live->ParametersCounted() ? 2 : 0) + static_cast<uint64_t>(live->State()) * 4
                              : 0;
    const uint64_t full_key = key * 100003ULL + static_cast<uint64_t>(batch_);
    if (full_key == summary_key_ && summary_.has_model == (live && live->Config())) return;
    summary_key_ = full_key;
    ModelSummaryInputs inputs;
    if (live) {
        inputs.state = live->State();
        inputs.config = live->Config();
        inputs.layer_parameters = live->LayerParameters();
        inputs.parameters_counted = live->ParametersCounted();
        inputs.parameters_counting = live->ParametersCounting();
    }
    if (node_editor_) {
        for (const auto& node : node_editor_->GetNodes()) {
            if (const auto metadata = ResolveNodeMetadata(node)) inputs.type_labels[node.id] = metadata->name;
        }
    }
    inputs.batch = batch_ > 0 ? static_cast<size_t>(batch_) : 0;
    summary_ = BuildModelSummary(inputs);
    // Keep the selected layer across recompiles; else the busiest one.
    selected_row_ = 0;
    bool found = false;
    for (size_t i = 0; i < summary_.rows.size(); ++i) {
        if (summary_.rows[i].node_id == selected_node_ && selected_node_ >= 0) {
            selected_row_ = i;
            found = true;
        }
    }
    if (!found) {
        double best = -1.0;
        for (size_t i = 0; i < summary_.rows.size(); ++i) {
            if (summary_.rows[i].share > best) {
                best = summary_.rows[i].share;
                selected_row_ = i;
            }
        }
        selected_node_ = summary_.rows.empty() ? -1 : summary_.rows[selected_row_].node_id;
    }
}

void ModelPanel::Render() {
    if (!visible_) return;
    ImGui::SetNextWindowSize(ImVec2(1180, 760), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin(ICON_FA_LAYER_GROUP " Model###ModelTools", &visible_)) {
        ImGui::End();
        return;
    }
    const auto& t = ui::CurrentTokens();
    Refresh();

    {
        const ui::FontScope medium(ui::Font::Medium);
        ImGui::TextColored(t.text_bright, "%s", "Model");
    }
    ImGui::SameLine();
    ImGui::TextColored(StatusTone(summary_.kind), "%s", summary_.status.c_str());
    if (!summary_.status_note.empty()) {
        ImGui::SameLine();
        ImGui::TextColored(t.text_dim, "%s", summary_.status_note.c_str());
    }

    if (ImGui::BeginTabBar("##model_tabs")) {
        const auto flags = [this](const char* id) {
            return open_tab_ == id ? ImGuiTabItemFlags_SetSelected : ImGuiTabItemFlags_None;
        };
        if (ImGui::BeginTabItem("Summary", nullptr, flags("summary"))) {
            RenderSummary();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Learning rate", nullptr, flags("learning_rate"))) {
            RenderLearningRate();
            ImGui::EndTabItem();
        }
        open_tab_.clear();
        ImGui::EndTabBar();
    }
    ImGui::End();
}

void ModelPanel::RenderSummary() {
    if (!summary_.has_model) {
        ui::EmptyState(ICON_FA_LAYER_GROUP, "No model to summarize", summary_.empty_text.c_str());
        return;
    }
    RenderTiles();
    const float side = 360.0f;
    const bool wide = ImGui::GetContentRegionAvail().x >= 820.0f;
    if (wide) {
        if (ImGui::BeginTable("##summary_layout", 2, ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("##layers", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("##side", ImGuiTableColumnFlags_WidthFixed, side);
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            RenderLayers(ImGui::GetContentRegionAvail().y);
            ImGui::TableNextColumn();
            ImGui::BeginChild("##side_scroll", ImVec2(0.0f, ImGui::GetContentRegionAvail().y));
            RenderSide();
            ImGui::EndChild();
            ImGui::EndTable();
        }
    } else {
        // Narrow: the side cards go under the table, nothing is cut off.
        ImGui::BeginChild("##narrow_scroll", ImVec2(0.0f, 0.0f));
        RenderLayers(std::max(240.0f, ImGui::GetContentRegionAvail().y * 0.6f));
        RenderSide();
        ImGui::EndChild();
    }
}

void ModelPanel::RenderTiles() {
    if (!ImGui::BeginTable("##tiles", 4, ImGuiTableFlags_SizingStretchSame)) return;
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    Tile("parameters", "Parameters", summary_.parameters_text, summary_.parameters_note);
    ImGui::TableNextColumn();
    Tile("multiply_adds", "Multiply-adds a sample", summary_.multiply_adds_text, summary_.multiply_adds_note);
    ImGui::TableNextColumn();
    Tile("weights", "Weights", summary_.weights_text, summary_.weights_note);
    ImGui::TableNextColumn();
    const std::string activations = "Activations, batch " + std::to_string(summary_.batch);
    Tile("activations", activations.c_str(), summary_.activations_text, summary_.activations_note);
    ImGui::EndTable();
}

void ModelPanel::RenderLayers(float height) {
    const auto& t = ui::CurrentTokens();
    const ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp |
                                  ImGuiTableFlags_PadOuterX;
    if (!ImGui::BeginTable("##layers", 7, flags, ImVec2(0.0f, height))) return;
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("000").x);
    ImGui::TableSetupColumn("Layer", ImGuiTableColumnFlags_WidthStretch, 1.6f);
    ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthStretch, 1.0f);
    ImGui::TableSetupColumn("Output", ImGuiTableColumnFlags_WidthStretch, 1.1f);
    ImGui::TableSetupColumn("Parameters", ImGuiTableColumnFlags_WidthStretch, 0.9f);
    ImGui::TableSetupColumn("Multiply-adds", ImGuiTableColumnFlags_WidthStretch, 1.1f);
    ImGui::TableSetupColumn("Share", ImGuiTableColumnFlags_WidthStretch, 1.0f);
    ImGui::TableHeadersRow();
    for (size_t i = 0; i < summary_.rows.size(); ++i) {
        const auto& row = summary_.rows[i];
        ImGui::PushID(row.node_id >= 0 ? row.node_id : -static_cast<int>(i) - 1);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        const bool chosen = i == selected_row_;
        char index[16];
        std::snprintf(index, sizeof(index), "%zu", i + 1);
        if (ImGui::Selectable(index, chosen, ImGuiSelectableFlags_SpanAllColumns)) {
            selected_row_ = i;
            selected_node_ = row.node_id;
        }
        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && node_editor_ &&
            row.node_id >= 0) {
            node_editor_->FocusNode(row.node_id);
        }
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(row.name.c_str());
        ImGui::TableNextColumn();
        ImGui::TextColored(t.text_dim, "%s", row.kind.c_str());
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(row.output.c_str());
        ImGui::TableNextColumn();
        if (!row.parameters_known) ImGui::TextColored(t.text_faint, "-");
        else if (row.parameters == 0) ImGui::TextColored(t.text_faint, "0");
        else ImGui::TextUnformatted(FormatCompiledCount(row.parameters).c_str());
        ImGui::TableNextColumn();
        if (!row.multiply_adds.counted) {
            ImGui::TextColored(t.warning, "not counted");
            if (ImGui::IsItemHovered()) ui::Tooltip(row.multiply_adds.formula.c_str());
        } else if (row.multiply_adds.value == 0) {
            ImGui::TextColored(t.text_faint, "0");
        } else {
            ImGui::TextUnformatted(FormatCompiledCount(row.multiply_adds.value).c_str());
        }
        ImGui::TableNextColumn();
        if (row.share > 0.0) {
            const float width = ImGui::GetContentRegionAvail().x * 0.55f;
            const ImVec2 at = ImGui::GetCursorScreenPos();
            const float h = ImGui::GetTextLineHeight() * 0.45f;
            const float y = at.y + (ImGui::GetTextLineHeight() - h) * 0.5f;
            ImGui::GetWindowDrawList()->AddRectFilled(
                ImVec2(at.x, y), ImVec2(at.x + std::max(2.0f, width * static_cast<float>(row.share)), y + h),
                ImGui::GetColorU32(t.accent_text), h * 0.5f);
            ImGui::Dummy(ImVec2(width, ImGui::GetTextLineHeight()));
            ImGui::SameLine();
            ImGui::TextColored(t.text_dim, "%s", Percent(row.share).c_str());
        }
        ImGui::PopID();
    }
    ImGui::EndTable();
}

void ModelPanel::RenderSide() {
    const auto& t = ui::CurrentTokens();
    if (selected_row_ < summary_.rows.size()) {
        const auto& row = summary_.rows[selected_row_];
        ui::BeginCard("selected");
        ImGui::TextColored(t.text_bright, "%s", row.name.c_str());
        ImGui::SameLine();
        ImGui::TextColored(t.text_dim, "%s", row.kind.c_str());
        Paragraph(row.how);
        ImGui::Spacing();
        const bool on_canvas = node_editor_ && row.node_id >= 0;
        if (ui::SecondaryButton(ICON_FA_CROSSHAIRS " Show on canvas", on_canvas, "This row has no node of its own.")) {
            node_editor_->FocusNode(row.node_id);
        }
        ui::EndCard();
    }

    for (size_t i = 0; i < summary_.advice.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));
        ui::BeginCard("advice");
        ImGui::TextColored(t.text_bright, "%s", summary_.advice[i].title.c_str());
        Paragraph(summary_.advice[i].text, &t.text_dim);
        ui::EndCard();
        ImGui::PopID();
    }

    ui::BeginCard("source");
    ImGui::TextColored(t.text_bright, "%s", "Where the numbers come from");
    Paragraph("The graph's compile, the same one Properties shows as AS COMPILED: branches as linked, "
              "padding as set. Parameters are counted from the built model (learnable only); multiply-adds "
              "as PyTorch's flop counter counts them: matrix products, convolutions and attention, "
              "element-wise layers 0.",
              &t.text_dim);
    ImGui::Spacing();
    ui::FieldLabel("Batch for the activations");
    int batch = batch_ > 0 ? batch_ : static_cast<int>(summary_.batch);
    if (ui::NumberField("##batch", &batch, 1, 65536, 1, 80.0f, "The graph's batch size until changed here.")) {
        batch_ = batch;
    }
    ui::EndCard();

    if (ui::SecondaryButton(ICON_FA_COPY " Copy as text")) {
        ImGui::SetClipboardText(ModelSummaryText(summary_).c_str());
        message_ = "Copied the summary as text.";
        message_error_ = false;
    }
    ImGui::SameLine();
    if (ui::SecondaryButton("Export CSV...")) {
        if (const auto path = FileDialogs::SaveFile("Export the model summary", {{"CSV", "csv"}}, nullptr,
                                                    "model_summary.csv")) {
            std::ofstream out(*path, std::ios::binary);
            out << ModelSummaryCsv(summary_);
            message_ = out ? "Saved " + *path : "Could not write " + *path;
            message_error_ = !out;
        }
    }
    if (!message_.empty()) Paragraph(message_, message_error_ ? &t.warning : &t.text_dim);
}

void ModelPanel::RangeLaunchEnded(const std::string& why) {
    range_running_ = false;
    range_message_ = why;
    range_message_error_ = true;
}

void ModelPanel::RenderLearningRate() {
    auto& manager = TrainingManager::Instance();
    // A finished test: take its curve in once.
    const auto outcome = manager.GetLastRangeTest();
    if (outcome.serial != range_seen_serial_) {
        range_seen_serial_ = outcome.serial;
        range_running_ = false;
        range_points_ = outcome.points;
        range_stop_reason_ = outcome.stop_reason;
        range_message_error_ = !outcome.failure.empty();
        range_message_ = !outcome.failure.empty() ? "The test failed: " + outcome.failure
                         : outcome.cancelled     ? "Stopped before the test ended; the curve so far is shown."
                                                 : std::string();
    } else if (range_running_ && manager.IsTrainingActive()) {
        const auto live = manager.GetCurrentMetrics();
        range_points_ = live.range_points;
        range_stop_reason_.clear();
    }

    const float side = 320.0f;
    const bool wide = ImGui::GetContentRegionAvail().x >= 760.0f;
    if (wide && ImGui::BeginTable("##lr_layout", 2, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("##settings", ImGuiTableColumnFlags_WidthFixed, side);
        ImGui::TableSetupColumn("##result", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        RenderRangeSettings();
        ImGui::TableNextColumn();
        RenderRangeResult();
        ImGui::EndTable();
    } else if (!wide) {
        ImGui::BeginChild("##lr_narrow", ImVec2(0.0f, 0.0f));
        RenderRangeSettings();
        RenderRangeResult();
        ImGui::EndChild();
    }
}

void ModelPanel::RenderRangeSettings() {
    const auto& t = ui::CurrentTokens();
    const LiveGraphCompile* live = node_editor_ ? node_editor_->GetLiveCompile() : nullptr;
    const TrainingConfiguration* config = live ? live->Config() : nullptr;
    ui::BeginCard("range_settings");
    Paragraph("Trains a copy of the graph's model on its Train rows, raising the learning rate every step, and "
              "records the loss. The trained model in use, its checkpoints and the Dashboard's runs are not changed.",
              &t.text_dim);
    ImGui::Spacing();
    ui::FieldLabel("From");
    ui::NumberFieldOptions rate;
    rate.min = 1e-12;
    rate.max = 1e6;
    rate.format = "%g";
    rate.width = 110.0f;
    rate.hint = "The first step's learning rate.";
    ui::NumberField("##range_from", &range_.start, rate);
    ui::FieldLabel("To");
    rate.hint = "The last step's learning rate.";
    ui::NumberField("##range_to", &range_.end, rate);
    ui::FieldLabel("Steps");
    ui::NumberField("##range_steps", &range_.steps, 10, 2000, 10, 80.0f, "Optimizer steps from the first rate to the last.");
    if (config) {
        ImGui::Spacing();
        const gui::MLNode* optimizer = node_editor_ ? node_editor_->FindStep(config->optimizer_node_id) : nullptr;
        ImGui::TextColored(t.text_dim, "Batch size");
        ImGui::SameLine(110.0f);
        ImGui::Text("%d, the graph's", config->batch_size);
        ImGui::TextColored(t.text_dim, "Optimizer");
        ImGui::SameLine(110.0f);
        ImGui::Text("%s, from the graph", optimizer ? optimizer->name.c_str() : "-");
        ImGui::TextColored(t.text_dim, "Rows");
        ImGui::SameLine(110.0f);
        ImGui::TextUnformatted("the Train rows of the Data Split");
    }
    ImGui::Spacing();
    char rule[120];
    std::snprintf(rule, sizeof(rule), "Stops early once the smoothed loss (%g) passes %g x its lowest: the rate is then far too high.",
                  range_.smoothing, range_.stop_factor);
    Paragraph(rule, &t.text_faint);
    ImGui::Spacing();

    std::string blocked;
    if (!config) blocked = "Open a training graph first.";
    else if (!config->is_valid) blocked = "The graph does not compile; the Compile popup (F7) says why.";
    else if (TrainingManager::Instance().IsTrainingActive()) blocked = range_running_ ? "The test is running." : "Train is running.";
    else if (!(range_.start > 0.0 && range_.end > range_.start)) blocked = "To must be above From, both above 0.";
    if (ui::PrimaryButton("Run", blocked.empty(), blocked.c_str())) {
        range_message_.clear();
        range_points_.clear();
        range_stop_reason_.clear();
        const std::string refused = start_range_test_ ? start_range_test_(range_) : "The Engine cannot start a test here.";
        range_running_ = refused.empty();
        range_message_ = refused;
        range_message_error_ = !refused.empty();
    }
    ImGui::SameLine();
    const bool running = range_running_ && TrainingManager::Instance().IsTrainingActive();
    if (ui::SecondaryButton("Stop", running, "No test is running.")) TrainingManager::Instance().StopTraining();
    if (range_running_) {
        const int done = static_cast<int>(range_points_.size());
        ImGui::TextColored(t.running, "%s", done == 0 ? "Preparing the data..." : ("Step " + std::to_string(done) + " of " + std::to_string(range_.steps)).c_str());
    }
    if (!range_message_.empty()) Paragraph(range_message_, range_message_error_ ? &t.warning : &t.text_dim);
    ui::EndCard();
}

void ModelPanel::RenderRangeResult() {
    const auto& t = ui::CurrentTokens();
    if (range_points_.empty()) {
        ui::EmptyState(ICON_FA_CHART_LINE, range_running_ ? "Waiting for the first step" : "No range test yet",
                       "Run trains for a few dozen steps; the curve shows where the loss falls fastest.");
        return;
    }
    const auto analysis = AnalyseLearningRateRange(range_points_);
    const bool finished = !range_running_;

    static const char* const kRules[] = {"Steepest fall", "Lowest / 10"};
    int rule = steepest_rule_ ? 0 : 1;
    ImGui::TextColored(t.text_dim, "Suggest by");
    ImGui::SameLine();
    if (ui::SegmentedControl("##range_rule", kRules, 2, &rule)) steepest_rule_ = rule == 0;
    ImGui::SameLine();
    std::string status = "smoothed loss (" + FormatLearningRate(range_.smoothing) + ") · log scale · " +
                         std::to_string(range_points_.size()) + " of " + std::to_string(range_.steps) + " steps";
    if (finished && !range_stop_reason_.empty()) status += " · " + range_stop_reason_;
    ImGui::TextColored(t.text_dim, "%s", status.c_str());

    std::vector<double> x, y;
    for (const auto& point : range_points_) {
        if (!std::isfinite(point.smoothed)) break;
        x.push_back(point.learning_rate);
        y.push_back(point.smoothed);
    }
    const float plot_h = std::max(220.0f, ImGui::GetContentRegionAvail().y - 150.0f);
    if (!x.empty() && ImPlot::BeginPlot("##range_curve", ImVec2(-1, plot_h), ImPlotFlags_NoMenus | ImPlotFlags_NoLegend)) {
        ImPlot::SetupAxes("learning rate", "smoothed loss", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
        ImPlot::SetNextLineStyle(t.series[0], 2.0f);
        ImPlot::PlotLine("smoothed", x.data(), y.data(), static_cast<int>(x.size()));
        if (analysis.ok) {
            ImPlot::SetNextLineStyle(t.text_faint, 1.0f);
            ImPlot::PlotInfLines("lowest", &analysis.lowest_rate, 1);
            ImPlot::Annotation(analysis.lowest_rate, analysis.lowest_loss, t.text_dim, ImVec2(8, -12), true,
                               "lowest %s at %s", FormatLearningRate(analysis.lowest_loss).c_str(),
                               FormatLearningRate(analysis.lowest_rate).c_str());
            const double suggested = steepest_rule_ ? analysis.steepest_rate : analysis.lowest_over_ten;
            ImPlot::SetNextLineStyle(t.success, 1.5f);
            ImPlot::PlotInfLines("suggested", &suggested, 1);
            if (steepest_rule_) {
                const double sy = y[analysis.steepest];
                ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 5.0f, t.success, 0.0f);
                ImPlot::PlotScatter("steepest", &suggested, &sy, 1);
            }
        }
        ImPlot::EndPlot();
    }

    if (!finished) return;
    const LiveGraphCompile* live = node_editor_ ? node_editor_->GetLiveCompile() : nullptr;
    const TrainingConfiguration* config = live ? live->Config() : nullptr;
    const gui::MLNode* optimizer =
        config && node_editor_ ? node_editor_->FindStep(config->optimizer_node_id) : nullptr;
    const auto words = DescribeLearningRateRange(analysis, steepest_rule_, optimizer ? optimizer->name : std::string(),
                                                 config ? config->learning_rate : 0.0);
    ui::BeginCard("range_words");
    ImGui::TextColored(t.text_bright, "%s", words.headline.c_str());
    Paragraph(words.why, &t.text_dim);
    ImGui::Spacing();
    if (analysis.ok) {
        const bool can_use = optimizer != nullptr;
        if (ui::PrimaryButton(words.use_label.c_str(), can_use, "The graph has no optimizer node.")) {
            if (node_editor_->SetStepParameters(optimizer->id, {{"learning_rate", words.suggested_text}})) {
                range_message_ = optimizer->name + "'s learning rate is now " + words.suggested_text + " (Undo puts it back).";
                range_message_error_ = false;
            }
        }
        ImGui::SameLine();
    }
    if (ui::SecondaryButton(ICON_FA_COPY " Copy the curve")) {
        ImGui::SetClipboardText(LearningRateRangeText(range_points_, '\t').c_str());
        range_message_ = "Copied the curve as tab-separated rows.";
        range_message_error_ = false;
    }
    ui::EndCard();
}

}  // namespace cyxwiz
