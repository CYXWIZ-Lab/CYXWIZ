#include "model_panel.h"

#include "../../core/extension_node_metadata.h"
#include "../../core/file_dialogs.h"
#include "../../core/live_graph_compile.h"
#include "../icons.h"
#include "../node_editor.h"
#include "../ui_buttons.h"
#include "../ui_fonts.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"

#include <imgui.h>

#include <algorithm>
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

}  // namespace cyxwiz
