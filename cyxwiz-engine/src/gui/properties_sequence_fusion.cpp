// Properties panel: the word + POS fusion card on a Concatenate (TOFIX112,
// re-skinned in A7). It draws the SequenceFusionCard of the panel's view
// (cyxwiz::BuildSequenceFusionCard, the verdict the graph compiler uses), so
// the panel and a compile never disagree.
#include "properties.h"

#include "../core/sequence_fusion_presentation.h"
#include "icons.h"
#include "node_editor.h"
#include "ui_buttons.h"
#include "ui_tokens.h"
#include "ui_widgets.h"

#include <imgui.h>

namespace gui {

namespace {

void KeyValueRows(const char* table_id, const std::vector<std::pair<std::string, std::string>>& rows) {
    if (rows.empty() || !ImGui::BeginTable(table_id, 2, ImGuiTableFlags_SizingStretchProp)) return;
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

}  // namespace

void Properties::RenderFusionCard(MLNode& node) {
    const auto& card = view_.fusion;
    if (!card.applies || !node_editor_) return;
    const auto& t = cyxwiz::ui::CurrentTokens();
    const ImVec4& status_colour = card.compiles ? t.success : t.caution;

    cyxwiz::ui::BeginCard("sequence_fusion");
    ImGui::TextColored(status_colour, "%s", card.compiles ? ICON_FA_CIRCLE_CHECK : ICON_FA_TRIANGLE_EXCLAMATION);
    ImGui::SameLine();
    ImGui::TextUnformatted("Word + POS fusion");
    if (cyxwiz::ui::SameLineRight(cyxwiz::ui::ChipWidth(card.status.c_str()))) {
        cyxwiz::ui::Chip(card.status.c_str(), status_colour);
    } else {
        cyxwiz::ui::Chip(card.status.c_str(), status_colour);
    }

    if (!card.compiles) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextUnformatted(card.reason.c_str());
        ImGui::TextColored(t.text_dim, "%s", card.fix.c_str());
        ImGui::PopTextWrapPos();
        if (card.action != cyxwiz::SequenceFusionAction::None &&
            cyxwiz::ui::SecondaryButton(card.action_label.c_str())) {
            if (card.action == cyxwiz::SequenceFusionAction::SetFeatureAxis) {
                node.parameters["dim"] = "-1";
                NotifyEdited();
            } else if (MLNode* data_input = node_editor_->GetNodeForConfiguration(card.data_input_node_id)) {
                ConfigureNode(data_input);
            }
        }
    }

    ImGui::Spacing();
    KeyValueRows("##rows", card.rows);
    if (card.compiles) {
        ImGui::TextColored(t.text_dim, "Vocabulary sizes are set from the data when training starts.");
    }

    if (!card.details.empty()) {
        const bool open = fusion_details_open_.count(node.id) > 0;
        if (cyxwiz::ui::LinkButton(open ? "Hide" : "Details")) {
            if (open) {
                fusion_details_open_.erase(node.id);
            } else {
                fusion_details_open_.insert(node.id);
            }
        }
        if (open) {
            ImGui::Spacing();
            KeyValueRows("##details", card.details);
        }
    }
    cyxwiz::ui::EndCard();
}

}  // namespace gui
