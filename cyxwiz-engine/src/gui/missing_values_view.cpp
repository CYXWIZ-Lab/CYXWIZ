#include "missing_values_view.h"

#include "icons.h"
#include "ui_buttons.h"
#include "ui_tokens.h"
#include "ui_widgets.h"

#include <imgui.h>

namespace cyxwiz::ui {

namespace {

void WrappedText(const ImVec4& colour, const std::string& text) {
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(colour, "%s", text.c_str());
    ImGui::PopTextWrapPos();
}

}  // namespace

MissingValuesGroupEvents MissingValuesGroup(const char* id, const missing_values_fix::Group& g, bool label,
                                            missing_values_fix::NumberFill& fill) {
    MissingValuesGroupEvents events;
    if (!g.present) return events;
    const Tokens& t = CurrentTokens();
    BeginCard(id, /*raised=*/true);
    ImGui::TextColored(g.resolved ? t.success : t.error, "%s", g.resolved ? ICON_FA_CIRCLE_CHECK : ICON_FA_CIRCLE_XMARK);
    ImGui::SameLine();
    CardTitle(g.title.c_str(), g.subtitle.c_str());
    if (!g.rows.empty() && ImGui::BeginTable("##columns", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Column", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthStretch, 0.5f);
        ImGui::TableSetupColumn("Missing", ImGuiTableColumnFlags_WidthStretch, 0.8f);
        ImGui::TableSetupColumn(g.fix_header.c_str(), ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableHeadersRow();
        for (const auto& row : g.rows) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(row.column.c_str());
            ImGui::TableNextColumn();
            ImGui::TextColored(t.text_dim, "%s", row.type.c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(row.missing.c_str());
            ImGui::TableNextColumn();
            ImGui::TextColored(t.text_dim, "%s", row.fix.c_str());
        }
        ImGui::EndTable();
    }
    if (!g.resolved) {
        if (g.can_fill || g.can_drop || !g.can_leave_out.empty()) {
            FlowRow actions;
            if (g.can_fill) {
                static const char* kFills[] = {"Mean", "Median", "0"};
                int choice = static_cast<int>(fill);
                actions.Next(ImGui::CalcTextSize("Fill numbers with").x);
                ImGui::AlignTextToFramePadding();
                ImGui::TextColored(t.text_dim, "Fill numbers with");
                actions.Next(SegmentedControlWidth(kFills, 3));
                if (SegmentedControl("##fill", kFills, 3, &choice)) {
                    fill = static_cast<missing_values_fix::NumberFill>(choice);
                    events.fill_changed = true;
                }
                actions.Next(ButtonWidth("Fill these columns", ButtonSize::Regular));
                if (SecondaryButton("Fill these columns")) events.fill = true;
            }
            if (g.can_drop) {
                const char* drop = label ? "Drop rows without a label" : "Drop these rows";
                actions.Next(ButtonWidth(drop, ButtonSize::Regular));
                if (SecondaryButton(drop)) events.drop = true;
            }
            if (!g.can_leave_out.empty()) {
                actions.Next(ButtonWidth("Actions", ButtonSize::Regular));
                if (SecondaryButton("Actions")) ImGui::OpenPopup("##leave_out");
                if (ImGui::BeginPopup("##leave_out")) {
                    for (const auto& column : g.can_leave_out) {
                        const std::string item = "Leave " + column + " out of the features";
                        if (ImGui::MenuItem(item.c_str())) events.leave_out = column;
                    }
                    ImGui::EndPopup();
                }
            }
        }
        if (!g.note.empty()) WrappedText(t.text_faint, g.note);
    } else if (!g.done_text.empty()) {
        // What was added, with Undo after it on the line when it fits (mockup board 1).
        const float spacing = ImGui::GetStyle().ItemSpacing.x;
        const float needed = ImGui::CalcTextSize(g.done_text.c_str()).x + spacing + ImGui::CalcTextSize("Undo").x;
        if (needed <= ImGui::GetContentRegionAvail().x) {
            ImGui::AlignTextToFramePadding();
            ImGui::TextColored(t.success, "%s", g.done_text.c_str());
            ImGui::SameLine();
        } else {
            WrappedText(t.success, g.done_text);
        }
        if (LinkButton("Undo")) events.undo = true;
    }
    EndCard();
    ImGui::Dummy(ImVec2(0.0f, t.space_xs));
    return events;
}

}  // namespace cyxwiz::ui
