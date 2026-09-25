#include "ui_buttons.h"

#include <imgui.h>

namespace cyxwiz::ui {
namespace {

// Brand palette from the approved verification mockup (tofix119 C).
constexpr ImVec4 kPrimary = ImVec4(0.357f, 0.239f, 0.961f, 1.0f);
constexpr ImVec4 kPrimaryHover = ImVec4(0.439f, 0.333f, 0.980f, 1.0f);
constexpr ImVec4 kPrimaryActive = ImVec4(0.290f, 0.192f, 0.820f, 1.0f);
constexpr ImVec4 kSecondaryHover = ImVec4(0.094f, 0.133f, 0.231f, 1.0f);
constexpr ImVec4 kSecondaryActive = ImVec4(0.141f, 0.196f, 0.322f, 1.0f);
constexpr ImVec4 kBorder = ImVec4(0.141f, 0.196f, 0.322f, 1.0f);
constexpr ImVec4 kText = ImVec4(0.906f, 0.925f, 0.961f, 1.0f);
constexpr ImVec4 kLink = ImVec4(0.702f, 0.651f, 1.0f, 1.0f);
constexpr ImVec4 kLinkHover = ImVec4(0.812f, 0.776f, 1.0f, 1.0f);
constexpr ImVec4 kDanger = ImVec4(1.0f, 0.482f, 0.447f, 1.0f);
constexpr ImVec4 kDangerHover = ImVec4(0.302f, 0.106f, 0.114f, 1.0f);
constexpr float kRounding = 6.0f;

ImVec2 Padding(ButtonSize size) {
    return size == ButtonSize::Small ? ImVec2(10.0f, 4.0f) : ImVec2(16.0f, 8.0f);
}

void ReasonTooltip(bool enabled, const char* reason) {
    if (!enabled && reason && *reason &&
        ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip("%s", reason);
    }
}

bool Styled(const char* label, bool enabled, const char* reason, ButtonSize size,
            const ImVec4& fill, const ImVec4& hover, const ImVec4& active,
            const ImVec4& text, bool border, float width = 0.0f) {
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, kRounding);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, Padding(size));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, border ? 1.0f : 0.0f);
    ImGui::PushStyleColor(ImGuiCol_Button, fill);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hover);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, active);
    ImGui::PushStyleColor(ImGuiCol_Text, text);
    ImGui::PushStyleColor(ImGuiCol_Border, kBorder);
    if (!enabled) ImGui::BeginDisabled();
    const bool clicked = ImGui::Button(label, ImVec2(width, 0.0f));
    if (!enabled) ImGui::EndDisabled();
    ImGui::PopStyleColor(5);
    ImGui::PopStyleVar(3);
    ReasonTooltip(enabled, reason);
    return clicked && enabled;
}

}  // namespace

bool PrimaryButton(const char* label, bool enabled, const char* disabled_reason,
                   ButtonSize size, float width) {
    return Styled(label, enabled, disabled_reason, size, kPrimary, kPrimaryHover,
                  kPrimaryActive, ImVec4(1.0f, 1.0f, 1.0f, 1.0f), false, width);
}

bool SecondaryButton(const char* label, bool enabled, const char* disabled_reason,
                     ButtonSize size, float width) {
    return Styled(label, enabled, disabled_reason, size, ImVec4(0, 0, 0, 0),
                  kSecondaryHover, kSecondaryActive, kText, true, width);
}

bool DangerButton(const char* label, bool enabled, const char* disabled_reason,
                  ButtonSize size) {
    return Styled(label, enabled, disabled_reason, size, ImVec4(0, 0, 0, 0),
                  kDangerHover, kDangerHover, kDanger, true);
}

bool LinkButton(const char* label, bool enabled) {
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2.0f, 4.0f));
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_Text, kLink);
    if (!enabled) ImGui::BeginDisabled();
    const bool clicked = ImGui::Button(label);
    if (!enabled) ImGui::EndDisabled();
    const bool hovered = ImGui::IsItemHovered();
    ImGui::PopStyleColor(4);
    ImGui::PopStyleVar();
    if (hovered && enabled) {
        const ImVec2 min = ImGui::GetItemRectMin();
        const ImVec2 max = ImGui::GetItemRectMax();
        ImGui::GetWindowDrawList()->AddLine(
            ImVec2(min.x + 2.0f, max.y - 3.0f), ImVec2(max.x - 2.0f, max.y - 3.0f),
            ImGui::GetColorU32(kLinkHover));
        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    }
    return clicked && enabled;
}

float ButtonWidth(const char* label, ButtonSize size) {
    return ImGui::CalcTextSize(label, nullptr, true).x + Padding(size).x * 2.0f + 2.0f;
}

}  // namespace cyxwiz::ui
