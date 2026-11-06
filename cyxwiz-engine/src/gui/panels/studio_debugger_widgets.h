#pragma once

// Shared pieces of the Studio Debugger's screens (the panel's layout and its
// Model > Layers view): sub-view selector, cards and right alignment.

#include "studio_debugger_colors.h"
#include "../ui_buttons.h"

#include <imgui.h>

#include <algorithm>

namespace cyxwiz {

// Sub-view selector: a segmented control when it fits, a combo otherwise,
// so no view depends on widening the panel.
inline void SubViewSelector(const char* id, const char* const* labels, int count, int* selected) {
    if (count <= 1) {
        return;
    }
    float needed = 0.0f;
    for (int i = 0; i < count; ++i) {
        needed += ImGui::CalcTextSize(labels[i]).x + 24.0f;
    }
    if (needed <= ImGui::GetContentRegionAvail().x) {
        ui::SegmentedControl(id, labels, count, selected);
        return;
    }
    ImGui::SetNextItemWidth(std::min(260.0f, ImGui::GetContentRegionAvail().x));
    ImGui::PushID(id);
    if (ImGui::BeginCombo("##view", labels[*selected])) {
        for (int i = 0; i < count; ++i) {
            if (ImGui::Selectable(labels[i], *selected == i)) {
                *selected = i;
            }
        }
        ImGui::EndCombo();
    }
    ImGui::PopID();
}

// Right-aligns the next item group of `width` on the current line when it
// fits; otherwise it starts on a new line.
inline void RightAlignOnLine(float width) {
    ImGui::SameLine();
    const float available = ImGui::GetContentRegionAvail().x;
    if (available >= width) {
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + available - width);
    } else {
        ImGui::NewLine();
    }
}

inline bool BeginCard(const char* id) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, DebuggerPanelBg());
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 8.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.0f, 10.0f));
    return ImGui::BeginChild(id, ImVec2(0.0f, 0.0f),
                             ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);
}

inline void EndCard() {
    ImGui::EndChild();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();
}

inline void CardHeading(const char* text) {
    ImGui::PushStyleColor(ImGuiCol_Text, DebuggerBright());
    ImGui::TextUnformatted(text);
    ImGui::PopStyleColor();
}

}  // namespace cyxwiz
