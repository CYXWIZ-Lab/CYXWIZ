// Properties panel: the "AS COMPILED" card (TOFIX123, re-skinned in A7).
// Draws the CompiledNodeCard of the panel's view (built by
// cyxwiz::BuildCompiledNodeCard over the latest background compile), so the
// panel shows what the compiler built for the selected node. The node type's
// support axes live under Details.
#include "properties.h"

#include "../core/compiled_node_presentation.h"
#include "icons.h"
#include "node_editor.h"
#include "properties_rows.h"
#include "ui_buttons.h"
#include "ui_tokens.h"
#include "ui_widgets.h"

#include <imgui.h>

#include <algorithm>

namespace gui {

namespace {

const ImVec4& StatusColour(cyxwiz::CompiledStatusKind kind) {
    const auto& t = cyxwiz::ui::CurrentTokens();
    switch (kind) {
        case cyxwiz::CompiledStatusKind::Ok: return t.success;
        case cyxwiz::CompiledStatusKind::Failed: return t.caution;
        case cyxwiz::CompiledStatusKind::Pending: return t.running;
        case cyxwiz::CompiledStatusKind::None: break;
    }
    return t.pending;
}

void Wrapped(const ImVec4& colour, const std::string& text) {
    if (text.empty()) return;
    ImGui::PushStyleColor(ImGuiCol_Text, colour);
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(text.c_str());
    ImGui::PopTextWrapPos();
    ImGui::PopStyleColor();
}

void KeyValueRows(const char* id, const std::vector<std::pair<std::string, std::string>>& rows) {
    if (rows.empty() || !ImGui::BeginTable(id, 2, ImGuiTableFlags_SizingStretchProp)) return;
    const auto& t = cyxwiz::ui::CurrentTokens();
    for (const auto& [key, value] : rows) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextColored(t.text_dim, "%s", key.c_str());
        ImGui::TableNextColumn();
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextUnformatted(value.c_str());
        ImGui::PopTextWrapPos();
    }
    ImGui::EndTable();
}

void RenderNodeTypeSupport(const cyxwiz::NodeMetadata& metadata) {
    if (metadata.support_axes.empty()) return;
    const auto& t = cyxwiz::ui::CurrentTokens();
    ImGui::Spacing();
    ImGui::TextColored(t.text_dim, "Node type support");
    for (const auto& axis : metadata.support_axes) {
        ImGui::Text("%s:", axis.name.c_str());
        ImGui::SameLine();
        ImGui::TextColored(axis.supported ? t.success : t.error, "%s", axis.value.c_str());
        if (!axis.reason.empty()) {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(t.text_dim, "  %s", axis.reason.c_str());
            ImGui::PopTextWrapPos();
        }
    }
}

}  // namespace

void Properties::RenderCompiledCard(MLNode& node) {
    if (!view_.has_compiled || !node_editor_) return;
    const auto& t = cyxwiz::ui::CurrentTokens();
    const auto& card = view_.compiled;

    cyxwiz::ui::BeginCard("compiled_card");
    ImGui::TextColored(t.text_dim, "AS COMPILED");
    const float status_width = cyxwiz::ui::ChipWidth(card.status.c_str());
    if (cyxwiz::ui::SameLineRight(status_width)) {
        cyxwiz::ui::Chip(card.status.c_str(), StatusColour(card.kind));
    } else {
        cyxwiz::ui::Chip(card.status.c_str(), StatusColour(card.kind));
    }
    Wrapped(t.text_dim, card.status_note);
    if (!card.role.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextUnformatted(card.role.c_str());
        ImGui::PopTextWrapPos();
    }
    if (card.select_node_id >= 0 && cyxwiz::ui::LinkButton(card.select_label.c_str())) {
        node_editor_->FocusNode(card.select_node_id);
    }
    for (const auto& issue : card.issues) {
        ImGui::TextColored(issue.error ? t.error : t.warning, "%s",
                           issue.error ? ICON_FA_TRIANGLE_EXCLAMATION : ICON_FA_CIRCLE_INFO);
        ImGui::SameLine();
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextUnformatted(issue.message.c_str());
        ImGui::PopTextWrapPos();
    }

    if (card.has_shapes) {
        ImGui::Spacing();
        if (ImGui::BeginTable("##shapes", 3, ImGuiTableFlags_SizingStretchProp)) {
            const std::string batch_header = "Batch of " + std::to_string(card.batch);
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TableNextColumn();
            ImGui::TextColored(t.text_dim, "Per sample");
            ImGui::TableNextColumn();
            ImGui::TextColored(t.text_dim, "%s", batch_header.c_str());
            const auto row = [&](const char* label, const std::string& sample, const std::string& batch) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(t.text_dim, "%s", label);
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(sample.c_str());
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(batch.c_str());
            };
            row("Input", card.input_sample, card.input_batch);
            row("Output", card.output_sample, card.output_batch);
            ImGui::EndTable();
        }
        Wrapped(t.text_dim, card.input_note);
        ImGui::Spacing();
        KeyValueRows("##sizes", {{"Output memory per batch", card.output_memory},
                                 {"Learnable parameters", card.parameters},
                                 {"Parameter memory", card.parameter_memory}});
        Wrapped(t.text_dim, card.parameter_note);
    } else if (!card.no_shapes_text.empty()) {
        ImGui::Spacing();
        Wrapped(t.pending, card.no_shapes_text);
    }

    const bool open = compiled_details_open_.count(node.id) > 0;
    if (properties_rows::Disclosure("Details##compiled", open) != open) {
        if (open) {
            compiled_details_open_.erase(node.id);
        } else {
            compiled_details_open_.insert(node.id);
        }
    }
    if (open) {
        ImGui::Spacing();
        KeyValueRows("##details", card.details);
        if (view_metadata_) RenderNodeTypeSupport(*view_metadata_);
    }
    cyxwiz::ui::EndCard();
}

}  // namespace gui
