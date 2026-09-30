// Properties panel: the "AS COMPILED" card (TOFIX123). Renders
// cyxwiz::BuildCompiledNodeCard over the latest background compile of the
// canvas, so the panel shows what the compiler built for the selected node.
// Replaces the legacy editor-side Shape Info; the node type's support axes
// (formerly "Support Truth" under General) live under Details.
#include "../core/extension_node_metadata.h"
#include "properties.h"

#include "../core/compiled_node_presentation.h"
#include "../core/live_graph_compile.h"
#include "../core/node_metadata_registry.h"
#include "icons.h"
#include "node_editor.h"
#include "ui_buttons.h"

#include <imgui.h>

#include <algorithm>

namespace gui {

namespace {

const ImVec4 kVerified(0.24f, 0.84f, 0.55f, 1.0f);
const ImVec4 kFailed(0.96f, 0.63f, 0.29f, 1.0f);
const ImVec4 kPending(0.70f, 0.65f, 1.00f, 1.0f);
const ImVec4 kNotYet(0.64f, 0.69f, 0.78f, 1.0f);

ImVec4 StatusColor(cyxwiz::CompiledStatusKind kind) {
    switch (kind) {
        case cyxwiz::CompiledStatusKind::Ok: return kVerified;
        case cyxwiz::CompiledStatusKind::Failed: return kFailed;
        case cyxwiz::CompiledStatusKind::Pending: return kPending;
        case cyxwiz::CompiledStatusKind::None: break;
    }
    return kNotYet;
}

void Wrapped(const ImVec4& color, const std::string& text) {
    if (text.empty()) return;
    ImGui::PushStyleColor(ImGuiCol_Text, color);
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(text.c_str());
    ImGui::PopTextWrapPos();
    ImGui::PopStyleColor();
}

void KeyValueTable(const char* id, const std::vector<std::pair<std::string, std::string>>& rows) {
    if (rows.empty() || !ImGui::BeginTable(id, 2, ImGuiTableFlags_SizingStretchProp)) return;
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

void RenderNodeTypeSupport(const cyxwiz::NodeMetadata& metadata) {
    if (metadata.support_axes.empty()) return;
    ImGui::Spacing();
    ImGui::TextDisabled("Node type support");
    for (const auto& axis : metadata.support_axes) {
        ImGui::Text("%s:", axis.name.c_str());
        ImGui::SameLine();
        ImGui::TextColored(axis.supported ? ImVec4(0.35f, 0.85f, 0.45f, 1.0f) : ImVec4(1.0f, 0.35f, 0.35f, 1.0f),
                           "%s", axis.value.c_str());
        if (!axis.reason.empty()) {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextDisabled("  %s", axis.reason.c_str());
            ImGui::PopTextWrapPos();
        }
    }
}

}  // namespace

void Properties::RenderCompiledSection(MLNode& node) {
    if (!node_editor_) return;
    const auto& nodes = node_editor_->GetNodes();
    const auto& links = node_editor_->GetLinks();

    cyxwiz::CompiledNodeInputs inputs;
    const auto metadata = cyxwiz::ResolveNodeMetadata(node);
    inputs.type_label = metadata ? metadata->name : std::string();
    if (live_compile_) {
        inputs.state = live_compile_->State();
        inputs.config = live_compile_->Config();
        inputs.layer_parameters = live_compile_->LayerParameters();
        inputs.parameters_counted = live_compile_->ParametersCounted();
        inputs.parameters_counting = live_compile_->ParametersCounting();
    }
    inputs.data_loaded = AnyDataInputLoaded();
    const auto card = cyxwiz::BuildCompiledNodeCard(nodes, links, node.id, inputs);

    ImGui::Spacing();
    ImGui::PushID("compiled_card");
    if (ImGui::BeginChild("##card", ImVec2(0.0f, 0.0f),
                          ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY |
                              ImGuiChildFlags_AlwaysUseWindowPadding)) {
        const ImVec4 muted = ImGui::GetStyle().Colors[ImGuiCol_TextDisabled];
        ImGui::TextColored(muted, "AS COMPILED");
        ImGui::SameLine();
        const float status_width = ImGui::CalcTextSize(card.status.c_str()).x;
        ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(), ImGui::GetContentRegionMax().x - status_width));
        ImGui::TextColored(StatusColor(card.kind), "%s", card.status.c_str());
        Wrapped(muted, card.status_note);
        if (!card.role.empty()) {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextUnformatted(card.role.c_str());
            ImGui::PopTextWrapPos();
        }
        if (card.select_node_id >= 0 && cyxwiz::ui::LinkButton(card.select_label.c_str())) {
            node_editor_->FocusNode(card.select_node_id);
        }
        for (const auto& issue : card.issues) {
            ImGui::TextColored(issue.error ? kFailed : ImVec4(0.90f, 0.75f, 0.29f, 1.0f), "%s",
                               issue.error ? ICON_FA_TRIANGLE_EXCLAMATION : ICON_FA_CIRCLE_INFO);
            ImGui::SameLine();
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextUnformatted(issue.message.c_str());
            ImGui::PopTextWrapPos();
        }

        if (card.has_shapes) {
            ImGui::Separator();
            if (ImGui::BeginTable("##shapes", 3, ImGuiTableFlags_SizingStretchProp)) {
                const std::string batch_header = "Batch of " + std::to_string(card.batch);
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TableNextColumn();
                ImGui::TextColored(muted, "Per sample");
                ImGui::TableNextColumn();
                ImGui::TextColored(muted, "%s", batch_header.c_str());
                const auto row = [&](const char* label, const std::string& sample, const std::string& batch) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::TextColored(muted, "%s", label);
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(sample.c_str());
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(batch.c_str());
                };
                row("Input", card.input_sample, card.input_batch);
                row("Output", card.output_sample, card.output_batch);
                ImGui::EndTable();
            }
            Wrapped(muted, card.input_note);
            ImGui::Separator();
            KeyValueTable("##sizes", {{"Output memory per batch", card.output_memory},
                                      {"Learnable parameters", card.parameters},
                                      {"Parameter memory", card.parameter_memory}});
            Wrapped(muted, card.parameter_note);
        } else if (!card.no_shapes_text.empty()) {
            ImGui::Separator();
            Wrapped(ImVec4(0.64f, 0.69f, 0.78f, 1.0f), card.no_shapes_text);
        }

        const bool open = compiled_details_open_.count(node.id) > 0;
        if (cyxwiz::ui::LinkButton(open ? "Hide##compiled" : "Details##compiled")) {
            if (open) {
                compiled_details_open_.erase(node.id);
            } else {
                compiled_details_open_.insert(node.id);
            }
        }
        if (open) {
            ImGui::Separator();
            KeyValueTable("##details", card.details);
            if (metadata) RenderNodeTypeSupport(*metadata);
        }
    }
    ImGui::EndChild();
    ImGui::PopID();
}

}  // namespace gui
