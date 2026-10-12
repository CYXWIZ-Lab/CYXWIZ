// Subgraph Properties (TOFIX146 D5): what the subgraph is (a Preparation
// Recipe that runs, or a visual group that only tidies the canvas), its steps
// in run order with their last run, Expand / Collapse and the role switch,
// and Details. Replaces the generic sections, whose free-text fields let
// recipe_role and recipe_contract_version be typed past the recipe check.
#include "properties.h"

#include "../core/node_metadata_registry.h"
#include "../core/sql_step_presentation.h"
#include "icons.h"
#include "node_editor.h"
#include "sql_workbench_ui.h"
#include "ui_buttons.h"
#include "ui_tokens.h"

#include <imgui.h>
#include <imgui_stdlib.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace gui {

namespace {
namespace ui = cyxwiz::ui;
using sqlwb::Dim;
using sqlwb::DimWrapped;

std::string StepTypeName(NodeType type) {
    const auto* metadata = cyxwiz::NodeMetadataRegistry::Instance().GetMetadata(type);
    return metadata ? metadata->name : std::string("Node");
}

std::string StepIcon(NodeType type) {
    const auto* metadata = cyxwiz::NodeMetadataRegistry::Instance().GetMetadata(type);
    return metadata && !metadata->icon.empty() ? metadata->icon : std::string(ICON_FA_CIRCLE);
}

}  // namespace

void Properties::RenderSubgraph(MLNode& node, const cyxwiz::NodeMetadata* metadata) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (!node_editor_) {
        RenderGenericSections(node, metadata, false);
        return;
    }
    const auto summary = node_editor_->GetSubgraphSummary(node.id);
    if (!summary.found) {
        DimWrapped("This subgraph has no stored steps.");
        return;
    }
    using Request = NodeEditor::SubgraphRequest;

    // Header: name, role, and what that role means.
    ImGui::PushStyleColor(ImGuiCol_Text, t.text_bright);
    ImGui::TextUnformatted(node.name.c_str());
    ImGui::PopStyleColor();
    const std::string role = summary.recipe ? ICON_FA_GEARS " Preparation Recipe" : ICON_FA_OBJECT_GROUP " Visual group";
    sqlwb::AlignRight(ImGui::CalcTextSize(role.c_str()).x);
    sqlwb::Coloured(role, summary.recipe ? t.accent_text : t.text_dim);

    const std::string count = std::to_string(summary.steps.size()) + (summary.steps.size() == 1 ? " step" : " steps");
    if (summary.recipe) {
        DimWrapped("Runs its " + count + " in order as one stage: one dataset in, one prepared dataset out, the "
                   "same result as running the steps as separate nodes.");
    } else {
        DimWrapped("Tidies " + count + " on the canvas. A visual group does not run while collapsed; expand it, "
                   "or make it a Preparation Recipe, to run its steps.");
    }

    ImGui::Dummy(ImVec2(0.0f, t.space_xs));
    if (summary.expanded) {
        if (ui::SecondaryButton(ICON_FA_COMPRESS " Collapse", true, nullptr, ui::ButtonSize::Regular)) {
            node_editor_->RequestSubgraphAction(node.id, Request::Collapse);
        }
    } else if (ui::SecondaryButton(ICON_FA_EXPAND " Expand", true, nullptr, ui::ButtonSize::Regular)) {
        node_editor_->RequestSubgraphAction(node.id, Request::Expand);
    }
    // The role switch goes beside Expand, or under it when the panel is narrow.
    const char* role_switch = summary.recipe ? "Make visual group" : "Make Preparation Recipe";
    ImGui::SameLine();
    if (ImGui::GetContentRegionAvail().x < ui::ButtonWidth(role_switch, ui::ButtonSize::Regular)) ImGui::NewLine();
    if (summary.recipe) {
        if (ui::SecondaryButton("Make visual group", true, nullptr, ui::ButtonSize::Regular)) {
            node_editor_->RequestSubgraphAction(node.id, Request::MakeGroup);
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Stop running these steps as one stage; they only tidy the canvas.");
        }
    } else {
        if (ui::SecondaryButton("Make Preparation Recipe", true, nullptr, ui::ButtonSize::Regular)) {
            node_editor_->RequestSubgraphAction(node.id, Request::MakeRecipe);
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Run these steps as one stage. The steps are checked first; if they do not qualify, "
                              "you are told why and nothing changes.");
        }
    }

    // Steps in run order, with each one's latest run. A click shows the step
    // on the canvas (expanding the subgraph first when needed).
    ImGui::Dummy(ImVec2(0.0f, t.space_sm));
    ImGui::PushStyleColor(ImGuiCol_Text, t.text_bright);
    ImGui::TextUnformatted(summary.recipe ? "Steps, in run order" : "Nodes");
    ImGui::PopStyleColor();
    ImGui::SameLine();
    Dim("click one to show it");
    const auto& facts = node_editor_->GetNodeRunFacts();
    const float width = ImGui::GetContentRegionAvail().x;
    for (size_t i = 0; i < summary.steps.size(); ++i) {
        const auto& step = summary.steps[i];
        ImGui::PushID(step.id);
        const std::string label = std::to_string(i + 1) + ". " + StepIcon(step.type) + "  " + step.name;
        std::string result;
        ImVec4 result_colour = t.text_dim;
        if (const auto it = facts.find(step.id); it != facts.end()) {
            if (!it->second.error.empty()) {
                result = "failed";
                result_colour = t.caution;
            } else if (it->second.rows >= 0) {
                result = cyxwiz::sqlstep::FormatCount(it->second.rows) + " rows";
            } else if (!it->second.outcome.empty()) {
                result = it->second.outcome.rfind("passed", 0) == 0 ? "passed" : it->second.outcome;
                result_colour = it->second.outcome.rfind("passed", 0) == 0 ? t.success : t.text_dim;
            }
        }
        const ImVec2 start = ImGui::GetCursorScreenPos();
        const float line = ImGui::GetTextLineHeight();
        if (ImGui::Selectable("##step", false, ImGuiSelectableFlags_None, ImVec2(width, line * 2.0f))) {
            node_editor_->RequestSubgraphAction(node.id, Request::ShowStep, step.id);
        }
        if (ImGui::IsItemHovered() && !summary.expanded) ImGui::SetTooltip("Expand the subgraph and show this step");
        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->PushClipRect(start, ImVec2(start.x + width, start.y + line * 2.0f), true);
        dl->AddText(start, ImGui::GetColorU32(t.text), label.c_str());
        const float indent = ImGui::CalcTextSize("00. ").x;
        dl->AddText(ImVec2(start.x + indent, start.y + line), ImGui::GetColorU32(t.text_dim),
                    StepTypeName(step.type).c_str());
        if (!result.empty()) {
            const float result_w = ImGui::CalcTextSize(result.c_str()).x;
            dl->AddText(ImVec2(start.x + width - result_w, start.y + line), ImGui::GetColorU32(result_colour),
                        result.c_str());
        }
        dl->PopClipRect();
        ImGui::PopID();
    }
    if (summary.steps.empty()) DimWrapped("No steps.");

    // Details: every fact, none of it typed by hand.
    ImGui::Dummy(ImVec2(0.0f, t.space_sm));
    const bool open = subgraph_details_open_.count(node.id) > 0;
    if (ui::LinkButton(open ? "Hide details" : "Details")) {
        if (open) subgraph_details_open_.erase(node.id);
        else subgraph_details_open_.insert(node.id);
    }
    if (!open) return;

    Dim("Name");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::InputText("##subgraph_name", &node.name);

    std::vector<std::pair<std::string, std::string>> rows;
    rows.emplace_back("Node ID", std::to_string(node.id));
    rows.emplace_back("Type", metadata ? metadata->name + " (" + cyxwiz::GetCategoryDisplayName(metadata->category) + ")"
                                       : std::string("Subgraph"));
    rows.emplace_back("Role", summary.recipe ? "Preparation Recipe" : "Visual group");
    if (summary.recipe) {
        rows.emplace_back("Recipe contract", summary.recipe_contract.empty() ? "-" : summary.recipe_contract);
    }
    rows.emplace_back("Shown", summary.expanded ? "Expanded on the canvas" : "Collapsed");
    rows.emplace_back("Pins", std::to_string(summary.inputs) + " in, " + std::to_string(summary.outputs) + " out");
    std::string ids;
    for (const auto& step : summary.steps) ids += (ids.empty() ? "" : ", ") + std::to_string(step.id);
    rows.emplace_back("Step node IDs", ids.empty() ? "-" : ids);
    std::string raw;
    for (const auto& [key, value] : node.parameters) raw += (raw.empty() ? "" : "\n") + key + " = " + value;
    rows.emplace_back("Parameters", raw.empty() ? "-" : raw);
    float key_w = 0.0f;
    for (const auto& row : rows) key_w = std::max(key_w, ImGui::CalcTextSize(row.first.c_str()).x);
    if (ImGui::BeginTable("##subgraph_details", 2, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("key", ImGuiTableColumnFlags_WidthFixed, key_w);
        ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        for (const auto& [key, value] : rows) {
            ImGui::TableNextColumn();
            Dim(key);
            ImGui::TableNextColumn();
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextUnformatted(value.c_str());
            ImGui::PopTextWrapPos();
        }
        ImGui::EndTable();
    }
    DimWrapped("The role is set with the buttons above (or the canvas menu), which check the recipe first; it is "
               "not edited as text.");
}

}  // namespace gui
