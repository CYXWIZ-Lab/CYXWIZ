// Properties panel: the word + POS fusion card on a Concatenate (TOFIX112).
// It renders cyxwiz::BuildSequenceFusionCard, the verdict the graph compiler
// uses, so the panel and a compile never disagree.
#include "properties.h"

#include "../core/sequence_fusion_presentation.h"
#include "icons.h"
#include "node_editor.h"
#include "ui_buttons.h"

#include <imgui.h>

#include <algorithm>

namespace gui {

namespace {

const ImVec4 kVerified(0.24f, 0.84f, 0.55f, 1.0f);
const ImVec4 kFailed(0.96f, 0.63f, 0.29f, 1.0f);

void KeyValueRows(const char* table_id, const std::vector<std::pair<std::string, std::string>>& rows) {
    if (rows.empty() || !ImGui::BeginTable(table_id, 2, ImGuiTableFlags_SizingStretchProp)) return;
    const ImVec4 muted = ImGui::GetStyle().Colors[ImGuiCol_TextDisabled];
    for (const auto& [key, value] : rows) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextColored(muted, "%s", key.c_str());
        ImGui::TableNextColumn();
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextUnformatted(value.c_str());
        ImGui::PopTextWrapPos();
    }
    ImGui::EndTable();
}

}  // namespace

void Properties::RenderSequenceFusionSection(MLNode& node) {
    if (node.type != NodeType::Concatenate || !node_editor_) return;
    const auto card = cyxwiz::BuildSequenceFusionCard(node_editor_->GetNodes(), node_editor_->GetLinks(), node.id);
    if (!card.applies) return;

    ImGui::Spacing();
    ImGui::PushID("sequence_fusion");
    if (ImGui::BeginChild("##card", ImVec2(0.0f, 0.0f),
                          ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY |
                              ImGuiChildFlags_AlwaysUseWindowPadding)) {
        const ImVec4 muted = ImGui::GetStyle().Colors[ImGuiCol_TextDisabled];
        const ImVec4 status_color = card.compiles ? kVerified : kFailed;
        ImGui::TextColored(status_color, "%s", card.compiles ? ICON_FA_CIRCLE_CHECK : ICON_FA_TRIANGLE_EXCLAMATION);
        ImGui::SameLine();
        ImGui::TextUnformatted("Word + POS fusion");
        ImGui::SameLine();
        const float status_width = ImGui::CalcTextSize(card.status.c_str()).x;
        ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(),
                                      ImGui::GetContentRegionMax().x - status_width));
        ImGui::TextColored(status_color, "%s", card.status.c_str());

        if (!card.compiles) {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextUnformatted(card.reason.c_str());
            ImGui::TextColored(muted, "%s", card.fix.c_str());
            ImGui::PopTextWrapPos();
            if (card.action != cyxwiz::SequenceFusionAction::None &&
                cyxwiz::ui::SecondaryButton(card.action_label.c_str())) {
                if (card.action == cyxwiz::SequenceFusionAction::SetFeatureAxis) {
                    node.parameters["dim"] = "-1";
                    InvalidateShapes();
                } else if (MLNode* data_input = node_editor_->GetNodeForConfiguration(card.data_input_node_id)) {
                    ConfigureNode(data_input);
                }
            }
        }

        ImGui::Spacing();
        KeyValueRows("##rows", card.rows);
        if (card.compiles) {
            ImGui::TextColored(muted, "Vocabulary sizes are set from the data when training starts.");
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
                ImGui::Separator();
                KeyValueRows("##details", card.details);
            }
        }
    }
    ImGui::EndChild();
    ImGui::PopID();
}

}  // namespace gui
