#include "ui_buttons.h"
#include "ui_tokens.h"

#include <imgui.h>
#include <imgui_internal.h>

namespace cyxwiz::ui {
namespace {

// Colours come from the shared tokens (ui_tokens.h): one light/dark
// decision, one brand purple, theme-derived neutrals. Refreshed at the
// start of every button call.
ImVec4 kPrimary, kPrimaryHover, kPrimaryActive;
ImVec4 kSecondary, kSecondaryHover, kSecondaryActive, kText, kLink, kLinkHover;
ImVec4 kDanger, kDangerHover, kChipBg, kChipText, kChipTextHover;

void RefreshColors() {
    const Tokens& t = CurrentTokens();
    const ImVec4 ink = t.light ? ImVec4(0, 0, 0, 1) : ImVec4(1, 1, 1, 1);
    kPrimary = t.accent;
    kPrimaryHover = t.accent_hover;
    kPrimaryActive = t.accent_active;
    // No outlines on buttons (owner rule 2026-10-01): a secondary button is a
    // fill one step off the window, stronger on hover.
    kSecondary = Mix(t.bg_window, ink, t.light ? 0.06f : 0.07f);
    kSecondaryHover = Mix(t.bg_window, ink, t.light ? 0.11f : 0.13f);
    kSecondaryActive = Mix(t.bg_window, ink, t.light ? 0.16f : 0.18f);
    kText = t.text_bright;
    kLink = t.accent_text;
    kLinkHover = t.light ? t.accent : Mix(t.accent_text, ink, 0.35f);
    kDanger = t.error;
    kDangerHover = Mix(t.bg_window, t.error, t.light ? 0.15f : 0.25f);
    kChipBg = t.bg_raised;
    kChipText = Mix(t.text, t.text_dim, 0.30f);
    kChipTextHover = t.text_bright;
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
            const ImVec4& text, float width = 0.0f) {
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, kRounding);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, Padding(size));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_Button, fill);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hover);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, active);
    ImGui::PushStyleColor(ImGuiCol_Text, text);
    if (!enabled) ImGui::BeginDisabled();
    const bool clicked = ImGui::Button(label, ImVec2(width, 0.0f));
    if (!enabled) ImGui::EndDisabled();
    ImGui::PopStyleColor(4);
    ImGui::PopStyleVar(3);
    ReasonTooltip(enabled, reason);
    return clicked && enabled;
}

}  // namespace

bool PrimaryButton(const char* label, bool enabled, const char* disabled_reason,
                   ButtonSize size, float width) {
    RefreshColors();
    return Styled(label, enabled, disabled_reason, size, kPrimary, kPrimaryHover,
                  kPrimaryActive, ImVec4(1.0f, 1.0f, 1.0f, 1.0f), width);
}

bool SecondaryButton(const char* label, bool enabled, const char* disabled_reason,
                     ButtonSize size, float width) {
    RefreshColors();
    return Styled(label, enabled, disabled_reason, size, kSecondary, kSecondaryHover, kSecondaryActive, kText, width);
}

bool DangerButton(const char* label, bool enabled, const char* disabled_reason,
                  ButtonSize size) {
    RefreshColors();
    // Red text, a red tint on hover; no outline.
    return Styled(label, enabled, disabled_reason, size, ImVec4(0, 0, 0, 0), kDangerHover, kDangerHover, kDanger);
}

bool GhostButton(const char* label, bool enabled, const char* disabled_reason, bool on, ButtonSize size) {
    RefreshColors();
    return Styled(label, enabled, disabled_reason, size, on ? kSecondary : ImVec4(0, 0, 0, 0), kSecondaryHover,
                  kSecondaryActive, kText);
}

float StatusPillWidth(const char* text) {
    return ImGui::CalcTextSize(text, nullptr, true).x + 12.0f + 8.0f + 8.0f + 12.0f;
}

bool StatusPill(const char* id, const char* text, const ImVec4& dot) {
    RefreshColors();
    const float height = ImGui::GetFrameHeight();
    const float width = StatusPillWidth(text);
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const bool clicked = ImGui::InvisibleButton(id, ImVec2(width, height));
    const bool hovered = ImGui::IsItemHovered();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(pos, ImVec2(pos.x + width, pos.y + height), ImGui::GetColorU32(hovered ? kSecondaryHover : kChipBg),
                      height * 0.5f);
    dl->AddCircleFilled(ImVec2(pos.x + 16.0f, pos.y + height * 0.5f), 4.0f, ImGui::GetColorU32(dot));
    dl->AddText(ImVec2(pos.x + 28.0f, pos.y + (height - ImGui::GetTextLineHeight()) * 0.5f),
                ImGui::GetColorU32(hovered ? kChipTextHover : kChipText), text, ImGui::FindRenderedTextEnd(text));
    if (hovered) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    return clicked;
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
    // Emphasis is a red tint, not an outline (no outlines on buttons).
    if (emphasise && on) dl->AddRectFilled(pos, max, ImGui::GetColorU32(ImVec4(0.42f, 0.16f, 0.16f, 0.55f)), height * 0.5f);
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
    // One tinted track; the chosen segment is filled. No outlines or dividers.
    float total = 0.0f;
    for (int i = 0; i < count; ++i) total += ImGui::CalcTextSize(labels[i], nullptr, true).x + 24.0f;
    dl->AddRectFilled(start, ImVec2(start.x + total, start.y + height), ImGui::GetColorU32(kSecondary), kRounding);
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
        if (hovered) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        x += width;
    }
    ImGui::SetCursorScreenPos(start);
    ImGui::Dummy(ImVec2(x - start.x, height));
    ImGui::PopID();
    return changed;
}

float ButtonWidth(const char* label, ButtonSize size) {
    return ImGui::CalcTextSize(label, nullptr, true).x + Padding(size).x * 2.0f + 2.0f;
}

}  // namespace cyxwiz::ui
