#include "ui_buttons.h"

#include <imgui.h>
#include <imgui_internal.h>

namespace cyxwiz::ui {
namespace {

// Brand palette from the approved verification mockup (tofix119 C).
constexpr ImVec4 kPrimary = ImVec4(0.357f, 0.239f, 0.961f, 1.0f);
constexpr ImVec4 kPrimaryHover = ImVec4(0.439f, 0.333f, 0.980f, 1.0f);
constexpr ImVec4 kPrimaryActive = ImVec4(0.290f, 0.192f, 0.820f, 1.0f);
// Neutral and status colours: dark values from the mockup, light values
// when the active theme has a light window background (Engine light
// themes). Refreshed at the start of every button call.
ImVec4 kSecondaryHover, kSecondaryActive, kBorder, kText, kLink, kLinkHover;
ImVec4 kDanger, kDangerHover, kChipBg, kChipText, kChipTextHover;

void RefreshColors() {
    const ImVec4 bg = ImGui::GetStyle().Colors[ImGuiCol_WindowBg];
    const bool light = (0.2126f * bg.x + 0.7152f * bg.y + 0.0722f * bg.z) > 0.5f;
    if (light) {
        kSecondaryHover = ImVec4(0.90f, 0.91f, 0.94f, 1.0f);
        kSecondaryActive = ImVec4(0.84f, 0.86f, 0.91f, 1.0f);
        kBorder = ImVec4(0.74f, 0.77f, 0.84f, 1.0f);
        kText = ImVec4(0.12f, 0.13f, 0.16f, 1.0f);
        kLink = ImVec4(0.290f, 0.192f, 0.820f, 1.0f);
        kLinkHover = ImVec4(0.357f, 0.239f, 0.961f, 1.0f);
        kDanger = ImVec4(0.75f, 0.15f, 0.13f, 1.0f);
        kDangerHover = ImVec4(0.98f, 0.88f, 0.88f, 1.0f);
        kChipBg = ImVec4(0.95f, 0.96f, 0.98f, 1.0f);
        kChipText = ImVec4(0.20f, 0.22f, 0.27f, 1.0f);
        kChipTextHover = ImVec4(0.05f, 0.05f, 0.08f, 1.0f);
    } else {
        kSecondaryHover = ImVec4(0.094f, 0.133f, 0.231f, 1.0f);
        kSecondaryActive = ImVec4(0.141f, 0.196f, 0.322f, 1.0f);
        kBorder = ImVec4(0.141f, 0.196f, 0.322f, 1.0f);
        kText = ImVec4(0.906f, 0.925f, 0.961f, 1.0f);
        kLink = ImVec4(0.702f, 0.651f, 1.0f, 1.0f);
        kLinkHover = ImVec4(0.812f, 0.776f, 1.0f, 1.0f);
        kDanger = ImVec4(1.0f, 0.482f, 0.447f, 1.0f);
        kDangerHover = ImVec4(0.302f, 0.106f, 0.114f, 1.0f);
        kChipBg = ImVec4(0.078f, 0.102f, 0.145f, 1.0f);
        kChipText = ImVec4(0.79f, 0.82f, 0.88f, 1.0f);
        kChipTextHover = ImVec4(0.95f, 0.96f, 0.97f, 1.0f);
    }
}
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
    RefreshColors();
    return Styled(label, enabled, disabled_reason, size, kPrimary, kPrimaryHover,
                  kPrimaryActive, ImVec4(1.0f, 1.0f, 1.0f, 1.0f), false, width);
}

bool SecondaryButton(const char* label, bool enabled, const char* disabled_reason,
                     ButtonSize size, float width) {
    RefreshColors();
    return Styled(label, enabled, disabled_reason, size, ImVec4(0, 0, 0, 0),
                  kSecondaryHover, kSecondaryActive, kText, true, width);
}

bool DangerButton(const char* label, bool enabled, const char* disabled_reason,
                  ButtonSize size) {
    RefreshColors();
    return Styled(label, enabled, disabled_reason, size, ImVec4(0, 0, 0, 0),
                  kDangerHover, kDangerHover, kDanger, true);
}

bool LinkButton(const char* label, bool enabled) {
    RefreshColors();
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

float ToggleChipWidth(const char* label, const char* count) {
    const float gap = 5.0f;
    return ImGui::CalcTextSize(label, nullptr, true).x + gap +
           ImGui::CalcTextSize(count).x + 18.0f;
}

bool ToggleChip(const char* id, const char* label, const char* count, bool on,
                unsigned int accent_rgba, bool emphasise) {
    RefreshColors();
    const float height = ImGui::GetFrameHeight() - 2.0f;
    const float width = ToggleChipWidth(label, count);
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const bool clicked = ImGui::InvisibleButton(id, ImVec2(width, height));
    const bool hovered = ImGui::IsItemHovered();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 max(pos.x + width, pos.y + height);
    const float alpha = on ? 1.0f : 0.45f;
    const ImVec4 fill = hovered ? kSecondaryHover : kChipBg;
    const auto faded = [alpha](ImVec4 color) {
        color.w *= alpha;
        return ImGui::GetColorU32(color);
    };
    dl->AddRectFilled(pos, max, faded(fill), height * 0.5f);
    const ImVec4 border = emphasise && on ? ImVec4(0.42f, 0.16f, 0.16f, 1.0f) : kBorder;
    dl->AddRect(pos, max, faded(border), height * 0.5f);
    ImVec4 accent = ImGui::ColorConvertU32ToFloat4(accent_rgba);
    accent.w *= alpha;
    const float text_y = pos.y + (height - ImGui::GetTextLineHeight()) * 0.5f;
    const ImVec2 label_size = ImGui::CalcTextSize(label, nullptr, true);
    const char* label_end = ImGui::FindRenderedTextEnd(label);
    dl->AddText(ImVec2(pos.x + 9.0f, text_y), ImGui::GetColorU32(accent), label, label_end);
    const ImVec4 count_color = emphasise && on ? accent : ImVec4(0.55f, 0.58f, 0.65f, alpha);
    dl->AddText(ImVec2(pos.x + 9.0f + label_size.x + 5.0f, text_y),
                ImGui::GetColorU32(count_color), count);
    if (hovered) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    return clicked;
}

float ChipButtonWidth(const char* label) {
    return ImGui::CalcTextSize(label, nullptr, true).x + 22.0f;
}

bool ChipButton(const char* label) {
    RefreshColors();
    const float height = ImGui::GetFrameHeight() - 2.0f;
    const float width = ChipButtonWidth(label);
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const bool clicked = ImGui::InvisibleButton(label, ImVec2(width, height));
    const bool hovered = ImGui::IsItemHovered();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 max(pos.x + width, pos.y + height);
    dl->AddRectFilled(pos, max,
                      ImGui::GetColorU32(hovered ? kSecondaryHover : kChipBg),
                      height * 0.5f);
    dl->AddRect(pos, max, ImGui::GetColorU32(hovered ? kPrimary : kBorder), height * 0.5f);
    const char* end = ImGui::FindRenderedTextEnd(label);
    dl->AddText(ImVec2(pos.x + 11.0f, pos.y + (height - ImGui::GetTextLineHeight()) * 0.5f),
                ImGui::GetColorU32(hovered ? kChipTextHover : kChipText),
                label, end);
    if (hovered) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    return clicked;
}

bool SegmentedControl(const char* id, const char* const* labels, int count, int* selected) {
    RefreshColors();
    ImGui::PushID(id);
    const float height = ImGui::GetFrameHeight();
    const ImVec2 start = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    bool changed = false;
    float x = start.x;
    for (int i = 0; i < count; ++i) {
        const float width = ImGui::CalcTextSize(labels[i], nullptr, true).x + 24.0f;
        ImGui::SetCursorScreenPos(ImVec2(x, start.y));
        ImGui::PushID(i);
        if (ImGui::InvisibleButton("##segment", ImVec2(width, height)) && *selected != i) {
            *selected = i;
            changed = true;
        }
        const bool hovered = ImGui::IsItemHovered();
        ImGui::PopID();
        const bool on = *selected == i;
        const ImVec2 min(x, start.y);
        const ImVec2 max(x + width, start.y + height);
        ImDrawFlags corners = ImDrawFlags_RoundCornersNone;
        if (i == 0) corners |= ImDrawFlags_RoundCornersLeft;
        if (i == count - 1) corners |= ImDrawFlags_RoundCornersRight;
        if (on || hovered) {
            const ImVec4 fill = on ? ImVec4(kPrimary.x, kPrimary.y, kPrimary.z, 0.85f) : kSecondaryHover;
            dl->AddRectFilled(min, max, ImGui::GetColorU32(fill), kRounding, corners);
        }
        const ImVec2 text = ImGui::CalcTextSize(labels[i], nullptr, true);
        dl->AddText(ImVec2(x + (width - text.x) * 0.5f, start.y + (height - text.y) * 0.5f),
                    ImGui::GetColorU32(on ? ImVec4(1, 1, 1, 1) : kText), labels[i],
                    ImGui::FindRenderedTextEnd(labels[i]));
        if (i > 0 && !on && *selected != i - 1)
            dl->AddLine(ImVec2(x, start.y + 4.0f), ImVec2(x, start.y + height - 4.0f),
                        ImGui::GetColorU32(kBorder));
        if (hovered) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        x += width;
    }
    dl->AddRect(start, ImVec2(x, start.y + height), ImGui::GetColorU32(kBorder), kRounding);
    ImGui::SetCursorScreenPos(start);
    ImGui::Dummy(ImVec2(x - start.x, height));
    ImGui::PopID();
    return changed;
}

float ButtonWidth(const char* label, ButtonSize size) {
    return ImGui::CalcTextSize(label, nullptr, true).x + Padding(size).x * 2.0f + 2.0f;
}

}  // namespace cyxwiz::ui
