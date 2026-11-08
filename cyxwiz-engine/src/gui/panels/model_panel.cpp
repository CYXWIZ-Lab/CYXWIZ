#include "model_panel.h"

#include "../../core/extension_node_metadata.h"
#include "../../core/file_dialogs.h"
#include "../../core/live_graph_compile.h"
#include "../../core/training_manager.h"
#include "../../core/async_task_manager.h"
#include "../../core/test_manager.h"
#include "../icons.h"
#include "../node_editor.h"
#include "../ui_buttons.h"
#include "../ui_fonts.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"
#include "output_renderer.h"

#include <cyxwiz/device.h>
#include <stb_image_write.h>

#include <imgui.h>
#include <implot.h>

#include <algorithm>
#include <cctype>
#include <map>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>

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

ModelPanel::~ModelPanel() { ReleaseExplainImages(); }

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
        if (ImGui::BeginTabItem("Explain", nullptr, flags("explain"))) {
            RenderExplain();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Health", nullptr, flags("health"))) {
            RenderHealth();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Projector", nullptr, flags("projector"))) {
            RenderProjector();
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

// ---- Explain -----------------------------------------------------------------

namespace {

constexpr int kExplainSide = 168;  // px of each map on screen

ImVec4 Mix(const ImVec4& a, const ImVec4& b, float t) {
    return ImVec4(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t, 1.0f);
}

// The theme's sequential scale at 0..1.
ImVec4 Sequential(float v) {
    const auto& stops = ui::CurrentTokens().scale_sequential;
    const float x = std::clamp(v, 0.0f, 1.0f) * 3.0f;
    const int i = std::min(2, static_cast<int>(x));
    return Mix(stops[i], stops[i + 1], x - static_cast<float>(i));
}

// The theme's diverging scale at -1..1.
ImVec4 Diverging(float v) {
    const auto& stops = ui::CurrentTokens().scale_diverging;
    const float x = std::clamp(v, -1.0f, 1.0f);
    return x < 0.0f ? Mix(stops[1], stops[0], -x) : Mix(stops[1], stops[2], x);
}

struct Rgba {
    int width = 0, height = 0;
    std::vector<unsigned char> pixels;
};

// height x width colours, each pixel `scale` times (nearest), RGBA.
Rgba Picture(size_t height, size_t width, int scale, const std::function<ImVec4(size_t)>& colour) {
    Rgba out;
    out.width = static_cast<int>(width) * scale;
    out.height = static_cast<int>(height) * scale;
    out.pixels.resize(static_cast<size_t>(out.width) * out.height * 4);
    for (int y = 0; y < out.height; ++y) {
        for (int x = 0; x < out.width; ++x) {
            const ImVec4 c = colour(static_cast<size_t>(y / scale) * width + static_cast<size_t>(x / scale));
            unsigned char* p = &out.pixels[(static_cast<size_t>(y) * out.width + x) * 4];
            p[0] = static_cast<unsigned char>(std::clamp(c.x, 0.0f, 1.0f) * 255.0f + 0.5f);
            p[1] = static_cast<unsigned char>(std::clamp(c.y, 0.0f, 1.0f) * 255.0f + 0.5f);
            p[2] = static_cast<unsigned char>(std::clamp(c.z, 0.0f, 1.0f) * 255.0f + 0.5f);
            p[3] = 255;
        }
    }
    return out;
}

// The four pictures of an outcome: input, Grad-CAM over it, saliency, IG.
std::vector<Rgba> ExplainPictures(const ExplainOutcome& out) {
    const size_t h = out.height, w = out.width, c = std::max<size_t>(1, out.channels);
    const int scale = std::max(1, kExplainSide / static_cast<int>(std::max<size_t>(1, std::max(h, w))));
    const auto [lo_it, hi_it] = std::minmax_element(out.input.begin(), out.input.end());
    const float lo = *lo_it, span = std::max(1e-12f, *hi_it - *lo_it);
    const auto pixel = [&](size_t i) {
        if (c >= 3)
            return ImVec4((out.input[i * c] - lo) / span, (out.input[i * c + 1] - lo) / span, (out.input[i * c + 2] - lo) / span, 1.0f);
        const float g = (out.input[i * c] - lo) / span;
        return ImVec4(g, g, g, 1.0f);
    };
    const auto peak = [](const std::vector<float>& v, bool absolute) {
        float m = 0.0f;
        for (float x : v) m = std::max(m, absolute ? std::fabs(x) : x);
        return m > 0.0f ? m : 1.0f;
    };
    std::vector<Rgba> pictures;
    pictures.push_back(Picture(h, w, scale, pixel));
    const float cam_peak = out.grad_cam.empty() ? 1.0f : peak(out.grad_cam, false);
    pictures.push_back(out.grad_cam.empty() ? Rgba{}
                                            : Picture(h, w, scale, [&](size_t i) {
                                                  const float a = out.grad_cam[i] / cam_peak;
                                                  return Mix(pixel(i), Sequential(a), 0.35f + 0.5f * a);
                                              }));
    const float sal_peak = peak(out.saliency, false);
    pictures.push_back(Picture(h, w, scale, [&](size_t i) { return Sequential(out.saliency[i] / sal_peak); }));
    const float ig_peak = peak(out.integrated, true);
    pictures.push_back(Picture(h, w, scale, [&](size_t i) { return Diverging(out.integrated[i] / ig_peak); }));
    return pictures;
}

}  // namespace

void ModelPanel::ReleaseExplainImages() {
    for (auto& id : explain_images_) {
        if (id) OutputRenderer::DeleteTexture(id);
        id = 0;
    }
}

void ModelPanel::UploadExplainImages() {
    ReleaseExplainImages();
    if (!explain_.ok || explain_.height <= 1) return;
    const auto pictures = ExplainPictures(explain_);
    for (size_t i = 0; i < pictures.size() && i < 4; ++i) {
        if (!pictures[i].pixels.empty())
            explain_images_[i] = OutputRenderer::CreateTextureFromRGBA(pictures[i].pixels.data(), pictures[i].width,
                                                                        pictures[i].height);
    }
    explain_image_size_[0] = pictures.front().width;
    explain_image_size_[1] = pictures.front().height;
}

void ModelPanel::RenderExplain() {
    const auto& t = ui::CurrentTokens();
    auto& tests = TestManager::Instance();
    const std::shared_ptr<IExecutableModel> model = TrainingManager::Instance().GetActiveModel();
    if (!model) {
        ui::EmptyState(ICON_FA_LAYER_GROUP, "No trained model",
                       "Train the graph, or load a checkpoint (Train > Load Checkpoint): Explain looks inside the model "
                       "Test and Inference use.");
        return;
    }
    if (!tests.HasResults() || tests.IsTestingActive() || tests.GetLastResults().kept_inputs.empty()) {
        ui::EmptyState(ICON_FA_LAYER_GROUP, tests.IsTestingActive() ? "Testing..." : "No test rows yet",
                       "Run Test (F8): it keeps the inputs of the rows it got wrong and of the first rows, and "
                       "Explain shows them here.");
        return;
    }
    const TestingMetrics& test = tests.GetLastResults();
    const double identity = static_cast<double>(test.total_samples) * 1e6 + test.total_time_seconds;
    if (identity != test_seen_) {
        test_seen_ = identity;
        explain_sample_ = -1;
        explain_asked_.clear();
        explain_shown_.clear();
        explain_ = ExplainOutcome{};
        ReleaseExplainImages();
    }
    const std::vector<std::string>& names = test.confusion_matrix.class_names;
    const auto choices = ExplainRowChoices(test, explain_wrong_only_, names);
    if (explain_sample_ < 0 && !choices.empty()) explain_sample_ = choices.front().sample;

    // A finished job.
    if (explain_job_) {
        std::lock_guard<std::mutex> lock(explain_job_->mutex);
        if (explain_job_->done) {
            explain_ = explain_job_->outcome;
            explain_shown_ = explain_asked_;
            explain_job_.reset();
            UploadExplainImages();
        }
    }
    StartExplain(ExplainRowChoices(test, false, names));

    const float side = 300.0f;
    const bool wide = ImGui::GetContentRegionAvail().x >= 820.0f;
    if (wide && ImGui::BeginTable("##explain_layout", 2, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("##rows", ImGuiTableColumnFlags_WidthFixed, side);
        ImGui::TableSetupColumn("##maps", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        RenderExplainControls(choices, names);
        ImGui::TableNextColumn();
        RenderExplainResult(names);
        ImGui::EndTable();
    } else if (!wide) {
        ImGui::BeginChild("##explain_narrow", ImVec2(0.0f, 0.0f));
        RenderExplainControls(choices, names);
        RenderExplainResult(names);
        ImGui::EndChild();
    }
    (void)t;
}

void ModelPanel::RenderExplainControls(const std::vector<ExplainRowChoice>& choices,
                                       const std::vector<std::string>& class_names) {
    const auto& t = ui::CurrentTokens();
    const TestingMetrics& test = TestManager::Instance().GetLastResults();
    ui::BeginCard("explain_rows");
    ImGui::TextColored(t.text_dim, "Rows the last test kept");
    int wrong_count = 0;
    for (const auto& kept : test.kept_inputs)
        if (static_cast<size_t>(kept.sample) < test.predictions.size() &&
            static_cast<size_t>(kept.sample) < test.ground_truth.size() &&
            test.predictions[kept.sample] != test.ground_truth[kept.sample])
            ++wrong_count;
    const std::string all = "All " + std::to_string(test.kept_inputs.size());
    const std::string wrong = "Wrong only (" + std::to_string(wrong_count) + ")";
    const char* const filters[] = {all.c_str(), wrong.c_str()};
    int filter = explain_wrong_only_ ? 1 : 0;
    if (ui::SegmentedControl("##explain_filter", filters, 2, &filter)) explain_wrong_only_ = filter == 1;
    ImGui::BeginChild("##explain_list", ImVec2(0.0f, 220.0f));
    for (const auto& choice : choices) {
        ImGui::PushID(choice.sample);
        if (choice.wrong) ImGui::PushStyleColor(ImGuiCol_Text, t.caution);
        if (ImGui::Selectable(choice.label.c_str(), choice.sample == explain_sample_)) explain_sample_ = choice.sample;
        if (choice.wrong) ImGui::PopStyleColor();
        ImGui::PopID();
    }
    ImGui::EndChild();
    Paragraph("Test keeps the inputs of every row it got wrong and of its first " +
                  std::to_string(TestingMetrics::kKeptFirstSamples) + " rows (up to 64 MB).",
              &t.text_faint);

    ImGui::Spacing();
    ImGui::TextColored(t.text_dim, "Explain the score of");
    static const char* const kTargets[] = {"Predicted", "True", "Class"};
    ui::SegmentedControl("##explain_target", kTargets, 3, &explain_target_mode_);
    if (explain_target_mode_ == 2) {
        const int classes = std::max(1, test.probability_classes > 0 ? test.probability_classes
                                                                      : static_cast<int>(class_names.size()));
        ui::NumberField("##explain_class", &explain_class_, 0, classes - 1, 1, 60.0f, "The class number.");
        if (explain_class_ < static_cast<int>(class_names.size())) {
            ImGui::SameLine();
            ImGui::TextColored(t.text_dim, "%s", class_names[explain_class_].c_str());
        }
    }
    ui::FieldLabel("Integrated Gradients steps");
    ui::NumberField("##ig_steps", &ig_steps_, 8, 512, 8, 70.0f, "Points on the path from the baseline; all in one batch.");
    if (!explain_.grad_cam_layer.empty()) {
        ImGui::TextColored(t.text_dim, "Grad-CAM layer");
        ImGui::SameLine();
        ImGui::Text("%s (the last convolution)", explain_.grad_cam_layer.c_str());
    }
    ImGui::Spacing();
    const bool can_save = explain_.ok && explain_images_[0] != 0;
    if (ui::SecondaryButton("Save the images...", can_save, "Pick a row with an image input.")) {
        const std::string name = "explain_row_" + std::to_string(explain_row_.row) + ".png";
        if (const auto path = FileDialogs::SaveFile("Save the maps (four PNG files)", {{"PNG", "png"}}, nullptr, name.c_str())) {
            const auto pictures = ExplainPictures(explain_);
            const std::filesystem::path base(*path);
            const char* const parts[] = {"input", "gradcam", "saliency", "integrated_gradients"};
            int written = 0;
            for (size_t i = 0; i < pictures.size() && i < 4; ++i) {
                if (pictures[i].pixels.empty()) continue;
                const auto file = base.parent_path() / (base.stem().string() + "_" + parts[i] + ".png");
                written += stbi_write_png(file.string().c_str(), pictures[i].width, pictures[i].height, 4,
                                          pictures[i].pixels.data(), pictures[i].width * 4) != 0;
            }
            explain_message_ = "Saved " + std::to_string(written) + " images beside " + base.string();
        }
    }
    if (!explain_message_.empty()) Paragraph(explain_message_, &t.text_dim);
    ui::EndCard();
}

void ModelPanel::StartExplain(const std::vector<ExplainRowChoice>& all) {
    if (explain_job_ || explain_sample_ < 0) return;
    const auto found = std::find_if(all.begin(), all.end(), [this](const ExplainRowChoice& c) { return c.sample == explain_sample_; });
    if (found == all.end()) return;
    int target = -1;
    if (explain_target_mode_ == 1) target = found->truth;
    if (explain_target_mode_ == 2) target = explain_class_;
    const std::string key = std::to_string(found->sample) + "/" + std::to_string(target) + "/" + std::to_string(ig_steps_);
    if (key == explain_asked_) return;
    auto& training = TrainingManager::Instance();
    const std::shared_ptr<IExecutableModel> model = training.GetActiveModel();
    const ActiveModelInfo info = training.GetActiveModelInfo();
    if (!model || !info.evaluation_config) return;
    if (training.IsTrainingActive()) {
        explain_message_ = "Training is running; Explain waits for it to finish.";
        return;
    }
    const TestingMetrics& test = TestManager::Instance().GetLastResults();
    ExplainRequest request;
    request.sample = test.kept_inputs[found->kept].values;
    request.target_class = target;
    request.ig_steps = ig_steps_;
    explain_asked_ = key;
    explain_row_ = *found;
    explain_message_.clear();
    auto job = std::make_shared<ExplainJob>();
    explain_job_ = job;
    const TrainingConfiguration config = *info.evaluation_config;
    const auto device = info.model_device;
    AsyncTaskManager::Instance().RunAsync(
        "Explain row " + std::to_string(found->row),
        [job, model, config, request, device](LambdaTask&) {
            ExplainOutcome outcome;
            if (device) {
                const auto activation = Device(device->type, device->device_id).ActivateExact(false);
                if (!activation.success) outcome.error = "Could not switch to the device that holds the model: " + activation.message;
            }
            if (outcome.error.empty()) outcome = ExplainSample(*model, config, request);
            std::lock_guard<std::mutex> lock(job->mutex);
            job->outcome = std::move(outcome);
            job->done = true;
        },
        nullptr, nullptr, alive_);
}

void ModelPanel::RenderExplainResult(const std::vector<std::string>& class_names) {
    const auto& t = ui::CurrentTokens();
    if (explain_job_ && explain_shown_.empty()) {
        ImGui::TextColored(t.running, "Working out the maps...");
        return;
    }
    if (explain_shown_.empty()) return;
    if (!explain_.ok) {
        ui::BeginCard("explain_error");
        Paragraph(explain_.error, &t.warning);
        ui::EndCard();
        return;
    }
    const auto words = DescribeExplanation(explain_, explain_row_.row, explain_row_.truth, class_names);
    ui::BeginCard("explain_words");
    ImGui::PushStyleColor(ImGuiCol_Text, words.wrong ? t.caution : t.success);
    Paragraph(words.headline);
    ImGui::PopStyleColor();
    if (explain_job_) ImGui::TextColored(t.running, "Working out the next maps...");
    ui::EndCard();

    if (explain_.height > 1 && explain_images_[0]) {
        const float side = static_cast<float>(kExplainSide);
        const char* const titles[] = {"Input", "Grad-CAM", "Saliency", "Integrated Gradients"};
        const std::string captions[] = {
            std::to_string(explain_.height) + " x " + std::to_string(explain_.width) + " x " +
                std::to_string(explain_.channels) + ", as the model sees it after the graph's steps.",
            explain_.grad_cam.empty() ? explain_.grad_cam_note
                                      : "Where " + explain_.grad_cam_layer + " found what raised the score: its channels "
                                        "weighted by their mean gradient, ReLU, resized to the input.",
            "|d score / d pixel|: the pixels whose change moves the score most. Noisy by nature.",
            "Each pixel's share of the score, summed along the path from the baseline. " + words.completeness};
        if (ImGui::BeginTable("##explain_maps", 4, ImGuiTableFlags_SizingStretchSame)) {
            ImGui::TableNextRow();
            for (int i = 0; i < 4; ++i) {
                ImGui::TableNextColumn();
                if (explain_images_[i]) {
                    ImGui::Image(static_cast<ImTextureID>(static_cast<intptr_t>(explain_images_[i])), ImVec2(side, side));
                } else {
                    ImGui::Dummy(ImVec2(side, side));
                }
                ImGui::TextColored(t.text_bright, "%s", titles[i]);
                Paragraph(captions[i], &t.text_dim);
            }
            ImGui::EndTable();
        }
    } else {
        // A table row: one bar per feature.
        std::vector<double> left, right, saliency, integrated;
        for (size_t i = 0; i < explain_.saliency.size(); ++i) {
            left.push_back(static_cast<double>(i) - 0.2);
            right.push_back(static_cast<double>(i) + 0.2);
            saliency.push_back(explain_.saliency[i]);
            integrated.push_back(i < explain_.integrated.size() ? explain_.integrated[i] : 0.0);
        }
        if (ImPlot::BeginPlot("##explain_features", ImVec2(-1, 260.0f), ImPlotFlags_NoMenus)) {
            ImPlot::SetupAxes("feature", "attribution", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::SetNextFillStyle(t.series[0], 0.8f);
            ImPlot::PlotBars("Integrated Gradients", left.data(), integrated.data(), static_cast<int>(left.size()), 0.4);
            ImPlot::SetNextFillStyle(t.series[2], 0.8f);
            ImPlot::PlotBars("|saliency|", right.data(), saliency.data(), static_cast<int>(right.size()), 0.4);
            ImPlot::EndPlot();
        }
        Paragraph(words.completeness + " " + explain_.grad_cam_note, &t.text_dim);
    }
    ImGui::Spacing();
    Paragraph("Grad-CAM, saliency and Integrated Gradients as Captum (PyTorch) computes them, on the trained model's "
              "own device; the baseline is the value 0 before the graph's Normalize (a black image).",
              &t.text_faint);
}

// ---- Health ------------------------------------------------------------------

HealthView ModelPanel::CurrentHealth() const {
    auto& training = TrainingManager::Instance();
    const TrainingHealth health =
        training.IsTrainingActive() ? training.GetCurrentMetrics().health : training.GetLastMetrics().health;
    std::vector<std::string> names;
    const LiveGraphCompile* live = node_editor_ ? node_editor_->GetLiveCompile() : nullptr;
    const ActiveModelInfo info = training.GetActiveModelInfo();
    const TrainingConfiguration* config =
        info.evaluation_config ? &*info.evaluation_config : (live ? live->Config() : nullptr);
    if (config)
        for (const auto& layer : config->layers) names.push_back(layer.name);
    return BuildHealthView(health, names);
}

namespace {

// Bars of a histogram in a box at the cursor.
void Histogram(const std::vector<long long>& counts, ImVec2 size, const ImVec4& colour, const ImVec4* background) {
    const ImVec2 at = ImGui::GetCursorScreenPos();
    ImDrawList* draw = ImGui::GetWindowDrawList();
    if (background) draw->AddRectFilled(at, ImVec2(at.x + size.x, at.y + size.y), ImGui::GetColorU32(*background), 4.0f);
    const long long peak = counts.empty() ? 0 : *std::max_element(counts.begin(), counts.end());
    if (peak > 0) {
        const float pad = background ? 6.0f : 1.0f;
        const float width = (size.x - 2 * pad) / static_cast<float>(counts.size());
        for (size_t i = 0; i < counts.size(); ++i) {
            const float h = std::max(1.0f, (size.y - 2 * pad) * static_cast<float>(counts[i]) / static_cast<float>(peak));
            const float x = at.x + pad + width * static_cast<float>(i);
            draw->AddRectFilled(ImVec2(x, at.y + size.y - pad - h), ImVec2(x + std::max(1.0f, width - 1.0f), at.y + size.y - pad),
                                ImGui::GetColorU32(colour));
        }
    }
    ImGui::Dummy(size);
}

}  // namespace

void ModelPanel::RenderHealth() {
    const auto& t = ui::CurrentTokens();
    const HealthView view = CurrentHealth();
    if (!view.has_health) {
        ui::EmptyState(ICON_FA_LAYER_GROUP, "No health record yet", view.empty_text.c_str());
        return;
    }
    const bool live = TrainingManager::Instance().IsTrainingActive();
    const HealthRow* selected = nullptr;
    for (const auto& row : view.rows)
        if (row.layer == health_selected_) selected = &row;
    if (!selected && !view.rows.empty()) {
        // The worst layer first.
        selected = &view.rows.front();
        for (const auto& row : view.rows)
            if (row.dead_warning) selected = &row;
        health_selected_ = selected->layer;
    }

    const float side = 360.0f;
    const bool wide = ImGui::GetContentRegionAvail().x >= 860.0f;
    const auto main_column = [&]() {
        ui::BeginCard("health_headline");
        ImGui::PushStyleColor(ImGuiCol_Text, view.warning ? t.caution : t.success);
        Paragraph(view.headline);
        ImGui::PopStyleColor();
        Paragraph(view.detail, &t.text_dim);
        ui::EndCard();
        const ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_PadOuterX;
        if (ImGui::BeginTable("##health_rows", 5, flags)) {
            ImGui::TableSetupColumn("Layer", ImGuiTableColumnFlags_WidthStretch, 1.2f);
            ImGui::TableSetupColumn("Weight size by epoch (||W||)", ImGuiTableColumnFlags_WidthStretch, 1.6f);
            ImGui::TableSetupColumn("Mean gradient by epoch", ImGuiTableColumnFlags_WidthStretch, 1.6f);
            ImGui::TableSetupColumn("Weights, last epoch", ImGuiTableColumnFlags_WidthFixed, 130.0f);
            ImGui::TableSetupColumn("Dead after its ReLU", ImGuiTableColumnFlags_WidthStretch, 1.1f);
            ImGui::TableHeadersRow();
            for (const auto& row : view.rows) {
                ImGui::PushID(static_cast<int>(row.layer));
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                if (ImGui::Selectable(row.name.c_str(), row.layer == health_selected_, ImGuiSelectableFlags_SpanAllColumns))
                    health_selected_ = row.layer;
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(row.weight_norms.c_str());
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(row.gradient_norms.c_str());
                ImGui::TableNextColumn();
                Histogram(row.histogram, ImVec2(120.0f, ImGui::GetTextLineHeight()), t.series[0], nullptr);
                ImGui::TableNextColumn();
                ImGui::TextColored(row.dead_warning ? t.caution : (row.dead == "no ReLU" ? t.text_dim : t.success), "%s",
                                   row.dead.c_str());
                ImGui::PopID();
            }
            ImGui::EndTable();
        }
        Paragraph(std::string(live ? "Live: the training running now, " : "The last training, ") + view.epochs +
                      ". Recorded once an epoch: the steps' mean gradient size (no extra pass), the weights at the "
                      "epoch's end, dead units on the validation pass.",
                  &t.text_faint);
    };
    const auto side_column = [&]() {
        if (selected) {
            ui::BeginCard("health_selected");
            ImGui::TextColored(t.text_bright, "%s: weights at the last epoch", selected->name.c_str());
            const float width = ImGui::GetContentRegionAvail().x;
            Histogram(selected->histogram, ImVec2(width, 150.0f), t.series[0], &t.bg_input);
            ImGui::TextColored(t.text_dim, "%s", selected->low.c_str());
            const float high_width = ImGui::CalcTextSize(selected->high.c_str()).x;
            ImGui::SameLine(std::max(0.0f, width - high_width));
            ImGui::TextColored(t.text_dim, "%s", selected->high.c_str());
            Paragraph(selected->words, &t.text_dim);
            ui::EndCard();
        }
        ui::BeginCard("health_advice");
        ImGui::TextColored(t.text_bright, "%s", "What to try");
        Paragraph(view.advice, &t.text_dim);
        ui::EndCard();
    };
    if (wide && ImGui::BeginTable("##health_layout", 2, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("##main", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("##side", ImGuiTableColumnFlags_WidthFixed, side);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        main_column();
        ImGui::TableNextColumn();
        side_column();
        ImGui::EndTable();
    } else if (!wide) {
        ImGui::BeginChild("##health_narrow", ImVec2(0.0f, 0.0f));
        main_column();
        side_column();
        ImGui::EndChild();
    }
}

// ---- Projector ----------------------------------------------------------------

namespace {

ImVec4 ClassColour(int k) {
    const auto& t = ui::CurrentTokens();
    if (k < 0) return t.text_faint;
    if (k < ui::Tokens::kSeriesCount) return t.series[k];
    return t.node[(k * 7) % ui::Tokens::kNodeHueCount];
}

}  // namespace

void ModelPanel::RenderProjector() {
    const auto& t = ui::CurrentTokens();
    auto& tests = TestManager::Instance();
    if (!tests.HasResults() || tests.IsTestingActive()) {
        ui::EmptyState(ICON_FA_LAYER_GROUP, tests.IsTestingActive() ? "Testing..." : "No test yet",
                       "Run Test (F8): it records what each layer makes of every test row, and Projector draws it in "
                       "two dimensions.");
        return;
    }
    const TestingMetrics& test = tests.GetLastResults();
    const auto layers = ProjectorLayers(test);
    if (layers.empty()) {
        ui::EmptyState(ICON_FA_LAYER_GROUP, "No layer features in the last test",
                       test.layer_features_note.empty()
                           ? "Projector reads a classifier's layers on a layer stack; run a full Test (not a quick test)."
                           : test.layer_features_note.c_str());
        return;
    }
    if (std::none_of(layers.begin(), layers.end(), [this](const ProjectorLayer& l) { return l.layer == projector_layer_; })) {
        // The last hidden layer: the features the head reads.
        projector_layer_ = layers.size() >= 2 ? layers[layers.size() - 2].layer : layers.back().layer;
    }
    std::vector<std::string> names;
    if (const auto info = TrainingManager::Instance().GetActiveModelInfo(); info.evaluation_config)
        for (const auto& layer : info.evaluation_config->layers) names.push_back(layer.name);
    const auto layer_name = [&](size_t index) {
        return index < names.size() && !names[index].empty() ? names[index] : "layer " + std::to_string(index);
    };

    // A finished job; else start one for the choices.
    if (projector_job_) {
        std::lock_guard<std::mutex> lock(projector_job_->mutex);
        if (projector_job_->done) {
            projector_ = projector_job_->outcome;
            projector_shown_ = projector_asked_;
            projector_job_.reset();
        }
    }
    const std::string key = std::to_string(test.total_samples) + "/" + std::to_string(test.total_time_seconds) + "/" +
                            std::to_string(projector_layer_) + "/" + std::to_string(projector_method_);
    if (!projector_job_ && key != projector_asked_) {
        projector_asked_ = key;
        auto job = std::make_shared<ProjectorJob>();
        projector_job_ = job;
        ProjectorRequest request;
        request.layer = projector_layer_;
        request.method = static_cast<ProjectorMethod>(projector_method_);
        const TestingMetrics copy = test;
        AsyncTaskManager::Instance().RunAsync(
            "Projector: " + layer_name(projector_layer_),
            [job, copy, request](LambdaTask& task) {
                ProjectorOutcome outcome = ProjectTestRows(copy, request, [&task] { return task.ShouldStop(); });
                std::lock_guard<std::mutex> lock(job->mutex);
                job->outcome = std::move(outcome);
                job->done = true;
            },
            nullptr, nullptr, alive_);
    }

    const float side = 300.0f;
    const bool wide = ImGui::GetContentRegionAvail().x >= 820.0f;
    const auto controls = [&]() {
        ui::BeginCard("projector_controls");
        Paragraph("What a layer makes of the test rows: its output for each row, brought down to two dimensions as in "
                  "Data Studio's Projection tab.",
                  &t.text_dim);
        ImGui::Spacing();
        ui::FieldLabel("Layer");
        const std::string current = layer_name(projector_layer_);
        ImGui::SetNextItemWidth(-1);
        if (ImGui::BeginCombo("##projector_layer", current.c_str())) {
            for (const auto& l : layers) {
                const std::string label = layer_name(l.layer) + "  (" + std::to_string(l.width) + " values a row)";
                if (ImGui::Selectable(label.c_str(), l.layer == projector_layer_)) projector_layer_ = l.layer;
            }
            ImGui::EndCombo();
        }
        ui::FieldLabel("Method");
        static const char* const kMethods[] = {"PCA", "t-SNE", "UMAP"};
        ui::SegmentedControl("##projector_method", kMethods, 3, &projector_method_);
        ui::FieldLabel("Colour by");
        static const char* const kColours[] = {"True class", "Predicted", "Right / wrong"};
        ui::SegmentedControl("##projector_colour", kColours, 3, &projector_colour_);
        ImGui::Spacing();
        if (projector_colour_ != 2) {
            ImGui::TextColored(t.text_dim, "Classes (click to show one)");
            const int classes = std::max(test.probability_classes, static_cast<int>(test.confusion_matrix.class_names.size()));
            for (int k = 0; k < classes; ++k) {
                ImGui::PushID(k);
                const ImVec4 colour = ClassColour(k);
                const std::string label = k < static_cast<int>(test.confusion_matrix.class_names.size())
                                              ? test.confusion_matrix.class_names[k]
                                              : std::to_string(k);
                ImGui::ColorButton("##swatch", colour, ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoBorder,
                                   ImVec2(ImGui::GetTextLineHeight(), ImGui::GetTextLineHeight()));
                ImGui::SameLine();
                if (ImGui::Selectable(label.c_str(), projector_only_ == k, 0, ImVec2(120.0f, 0.0f)))
                    projector_only_ = projector_only_ == k ? -1 : k;
                ImGui::PopID();
            }
        }
        ImGui::Spacing();
        if (ui::SecondaryButton("Send to Projection tab", static_cast<bool>(send_to_projection_),
                                "Data Studio is not open.")) {
            if (auto table = ProjectorFeaturesTable(test, projector_layer_)) {
                std::string name = "model_features_" + layer_name(projector_layer_);
                for (auto& c : name)
                    if (!std::isalnum(static_cast<unsigned char>(c))) c = '_';
                send_to_projection_(table, name);
                projector_message_ = "Sent " + std::to_string(table->num_rows()) + " rows as '" + name + "'.";
            }
        }
        if (!projector_message_.empty()) Paragraph(projector_message_, &t.text_dim);
        ui::EndCard();
    };
    if (wide && ImGui::BeginTable("##projector_layout", 2, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("##controls", ImGuiTableColumnFlags_WidthFixed, side);
        ImGui::TableSetupColumn("##plot", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        controls();
        ImGui::TableNextColumn();
        RenderProjectorPlot(test.confusion_matrix.class_names);
        ImGui::EndTable();
    } else if (!wide) {
        ImGui::BeginChild("##projector_narrow", ImVec2(0.0f, 0.0f));
        controls();
        RenderProjectorPlot(test.confusion_matrix.class_names);
        ImGui::EndChild();
    }
}

void ModelPanel::RenderProjectorPlot(const std::vector<std::string>& class_names) {
    const auto& t = ui::CurrentTokens();
    if (projector_shown_.empty()) {
        ImGui::TextColored(t.running, "Projecting the test rows...");
        return;
    }
    if (!projector_.ok) {
        Paragraph(projector_.error, &t.warning);
        return;
    }
    if (projector_job_) ImGui::TextColored(t.running, "Working out the next projection...");
    ImGui::TextColored(t.text_dim, "%s", projector_.note.c_str());
    const auto name_of = [&](int k) {
        return k >= 0 && static_cast<size_t>(k) < class_names.size() ? class_names[k] : std::to_string(k);
    };
    const float height = std::max(260.0f, ImGui::GetContentRegionAvail().y - 8.0f);
    if (!ImPlot::BeginPlot("##projector", ImVec2(-1, height), ImPlotFlags_NoMenus | ImPlotFlags_NoLegend)) return;
    ImPlot::SetupAxes(projector_.x_label.c_str(), projector_.y_label.c_str(), ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
    // One series a colour; mistakes on top, ringed.
    std::map<int, std::pair<std::vector<double>, std::vector<double>>> groups, mistakes;
    for (const auto& p : projector_.points) {
        if (projector_only_ >= 0 && p.truth != projector_only_ && projector_colour_ != 2) continue;
        const int key = projector_colour_ == 0 ? p.truth : projector_colour_ == 1 ? p.predicted : (p.wrong ? 1 : 0);
        auto& target = p.wrong ? mistakes[key] : groups[key];
        target.first.push_back(p.x);
        target.second.push_back(p.y);
    }
    const auto colour_of = [&](int key) {
        if (projector_colour_ == 2) return key == 1 ? t.caution : t.text_faint;
        return ClassColour(key);
    };
    for (auto& [key, xy] : groups) {
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 3.0f, ui::WithAlpha(colour_of(key), 0.8f), 0.0f);
        ImPlot::PlotScatter(("##g" + std::to_string(key)).c_str(), xy.first.data(), xy.second.data(),
                            static_cast<int>(xy.first.size()));
    }
    for (auto& [key, xy] : mistakes) {
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 5.0f, colour_of(key), 1.5f, t.text_bright);
        ImPlot::PlotScatter(("##m" + std::to_string(key)).c_str(), xy.first.data(), xy.second.data(),
                            static_cast<int>(xy.first.size()));
    }
    // The point under the mouse: its row, and a click opens it in Explain.
    if (ImPlot::IsPlotHovered()) {
        const ImPlotPoint mouse = ImPlot::GetPlotMousePos();
        const ImPlotRect limits = ImPlot::GetPlotLimits();
        const double sx = limits.X.Size() > 0 ? limits.X.Size() : 1.0, sy = limits.Y.Size() > 0 ? limits.Y.Size() : 1.0;
        const ProjectorPoint* nearest = nullptr;
        double best = 0.0004;  // within about 2 % of the plot
        for (const auto& p : projector_.points) {
            const double dx = (p.x - mouse.x) / sx, dy = (p.y - mouse.y) / sy, d = dx * dx + dy * dy;
            if (d < best) {
                best = d;
                nearest = &p;
            }
        }
        if (nearest) {
            const TestingMetrics& test = TestManager::Instance().GetLastResults();
            const bool kept = std::any_of(test.kept_inputs.begin(), test.kept_inputs.end(),
                                          [&](const TestingMetrics::KeptInput& k) { return k.sample == nearest->sample; });
            ImGui::BeginTooltip();
            ImGui::Text("row %lld: a %s, read as %s (%.1f %%)", nearest->row, name_of(nearest->truth).c_str(),
                        name_of(nearest->predicted).c_str(), nearest->confidence * 100.0f);
            ImGui::TextColored(t.text_dim, "%s", kept ? "Click to open it in Explain." : "Its input was not kept by the test.");
            ImGui::EndTooltip();
            if (kept && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                explain_sample_ = nearest->sample;
                explain_target_mode_ = 0;
                open_tab_ = "explain";
            }
        }
    }
    ImPlot::EndPlot();
}

}  // namespace cyxwiz
