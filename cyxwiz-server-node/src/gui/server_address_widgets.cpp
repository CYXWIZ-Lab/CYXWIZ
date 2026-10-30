// server_address_widgets.cpp
#include "gui/server_address_widgets.h"

#include "gui/theme.h"
#include "gui/ui_widgets.h"
#include "gui/server_panel.h"

#include <imgui.h>

namespace cyxwiz::servernode::gui {

namespace proto = cyxwiz::protocol;

void RenderServerRows(const proto::ServerAddressView& view) {
    const auto& semantic = ::gui::GetTheme().Semantic();
    const ImVec4 muted = ui::Muted();
    if (!ImGui::BeginTable("##server_rows", 3, ImGuiTableFlags_SizingStretchProp)) return;
    ImGui::TableSetupColumn("Service", ImGuiTableColumnFlags_WidthStretch, 1.0f);
    ImGui::TableSetupColumn("Address", ImGuiTableColumnFlags_WidthStretch, 2.2f);
    ImGui::TableSetupColumn("Status", ImGuiTableColumnFlags_WidthStretch, 1.1f);
    for (const auto& row : view.rows) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(row.name.c_str());
        ImGui::TableNextColumn();
        ImGui::PushFont(GetSafeFont(FONT_MONO));
        ImGui::TextUnformatted(row.address.c_str());
        ImGui::PopFont();
        ImGui::TableNextColumn();
        const char* word = proto::ReachWord(row.reach);
        switch (row.reach) {
            case proto::ServiceReach::Reachable: ImGui::TextColored(semantic.success, "%s", word); break;
            case proto::ServiceReach::NotReachable: ImGui::TextColored(semantic.warning, "%s", word); break;
            case proto::ServiceReach::Checking:
                ui::Spinner("##reach", ImGui::GetTextLineHeight() * 0.4f, 2.0f);
                ImGui::SameLine(0.0f, 8.0f);
                ImGui::TextColored(muted, "%s", word);
                break;
            case proto::ServiceReach::NotChecked: ImGui::TextColored(muted, "%s", word); break;
        }
    }
    ImGui::EndTable();
}

}  // namespace cyxwiz::servernode::gui
