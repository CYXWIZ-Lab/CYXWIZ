#include "explore_tab.h"

#include "../../core/explore_presentation.h"
#include "../icons.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"

#include <imgui.h>

#include <algorithm>

namespace cyxwiz {

bool ExploreProfileReady(const ExploreContext& context, const char* empty_hint) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (context.dataset.empty()) {
        ui::EmptyState(ICON_FA_TABLE, "Pick a dataset", empty_hint);
        return false;
    }
    if (context.profile) return true;
    if (context.profiling) {
        ImGui::TextColored(t.info, ICON_FA_SPINNER " Profiling %s (Task View shows the steps)", context.shown.c_str());
    } else if (!context.error.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.error, "%s", context.error.c_str());
        ImGui::PopTextWrapPos();
    }
    return false;
}

void ExploreFigures(const char* id, const std::vector<std::pair<std::string, std::string>>& rows) {
    const ui::Tokens& t = ui::CurrentTokens();
    float label_w = 0.0f, value_w = 0.0f;
    for (const auto& [label, value] : rows) {
        label_w = std::max(label_w, ImGui::CalcTextSize(label.c_str()).x);
        value_w = std::max(value_w, ImGui::CalcTextSize(value.c_str()).x);
    }
    const float gap = ImGui::GetStyle().CellPadding.x * 4.0f;
    const int pairs = ImGui::GetContentRegionAvail().x >= 2.0f * (label_w + value_w + gap) ? 2 : 1;
    if (!ImGui::BeginTable(id, pairs * 2, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_NoSavedSettings)) return;
    for (int p = 0; p < pairs; ++p) {
        ImGui::TableSetupColumn("##k", ImGuiTableColumnFlags_WidthFixed, label_w);
        ImGui::TableSetupColumn("##v", ImGuiTableColumnFlags_WidthStretch);
    }
    for (size_t i = 0; i < rows.size(); ++i) {
        if (i % static_cast<size_t>(pairs) == 0) ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextColored(t.text_dim, "%s", rows[i].first.c_str());
        if (const char* hint = explore_view::FigureHint(rows[i].first); *hint && ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip))
            ImGui::SetTooltip("%s", hint);
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(rows[i].second.c_str());
    }
    ImGui::EndTable();
}

}  // namespace cyxwiz
