#pragma once

// Small drawing helpers shared by the SQL Workbench files and the SQL step
// Properties view (TOFIX146): text in the theme's tones, right alignment that
// wraps instead of overlapping, and outline-free panels.

#include "ui_tokens.h"

#include <imgui.h>

#include <string>

namespace gui::sqlwb {

inline void Dim(const std::string& text) {
    ImGui::PushStyleColor(ImGuiCol_Text, cyxwiz::ui::CurrentTokens().text_dim);
    ImGui::TextUnformatted(text.c_str());
    ImGui::PopStyleColor();
}

inline void DimWrapped(const std::string& text) {
    ImGui::PushTextWrapPos(0.0f);
    Dim(text);
    ImGui::PopTextWrapPos();
}

inline void Coloured(const std::string& text, const ImVec4& colour) {
    ImGui::PushStyleColor(ImGuiCol_Text, colour);
    ImGui::TextUnformatted(text.c_str());
    ImGui::PopStyleColor();
}

inline void ColouredWrapped(const std::string& text, const ImVec4& colour) {
    ImGui::PushTextWrapPos(0.0f);
    Coloured(text, colour);
    ImGui::PopTextWrapPos();
}

// Puts the next item at the right edge of this line, or on the next line
// when it would overlap what is already there.
inline void AlignRight(float width) {
    ImGui::SameLine();
    const float x = ImGui::GetContentRegionMax().x - width;
    if (x < ImGui::GetCursorPosX()) ImGui::NewLine();
    else ImGui::SetCursorPosX(x);
}

// A panel one tone off its surface, no outline. Always pair with EndPanel.
inline bool BeginPanel(const char* id, const ImVec2& size, const ImVec4& background,
                       ImGuiChildFlags extra_flags = ImGuiChildFlags_None) {
    const auto& t = cyxwiz::ui::CurrentTokens();
    ImGui::PushStyleColor(ImGuiCol_ChildBg, background);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, t.rounding_card);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, t.card_padding);
    return ImGui::BeginChild(id, size, ImGuiChildFlags_AlwaysUseWindowPadding | extra_flags);
}

inline void EndPanel() {
    ImGui::EndChild();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();
}

}  // namespace gui::sqlwb
