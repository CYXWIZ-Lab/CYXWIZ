// ui_widgets.cpp - the Server Node's shared controls (TOFIX139 P2).
#include "gui/ui_widgets.h"

#include "gui/theme.h"

#include <imgui_internal.h>

#include <algorithm>
#include <cmath>

namespace cyxwiz::servernode::gui::ui {

namespace {

ImVec4 Mix(const ImVec4& a, const ImVec4& b, float t) {
    return ImVec4(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t);
}

float Luminance(const ImVec4& c) { return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z; }

// Text that reads on the accent: whichever of the theme's text and window
// background colours contrasts more with it.
ImVec4 OnAccent(const ImVec4& accent) {
    const ImGuiStyle& style = ImGui::GetStyle();
    const ImVec4 text = style.Colors[ImGuiCol_Text];
    const ImVec4 bg = style.Colors[ImGuiCol_WindowBg];
    const float la = Luminance(accent);
    return std::fabs(Luminance(text) - la) >= std::fabs(Luminance(bg) - la) ? text : bg;
}

const float kRounding = 6.0f;
const ImVec2 kButtonPadding(16.0f, 10.0f);
const ImVec2 kFieldPadding(12.0f, 10.0f);

}  // namespace

ImVec4 Accent() { return ::gui::GetTheme().GetAccentColor(); }

ImVec4 Muted() { return ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]; }

ImVec4 Tinted(const ImVec4& color, float strength) {
    // Inside a card (an opaque child) the tint sits on the card's surface.
    const ImGuiStyle& style = ImGui::GetStyle();
    const ImVec4 bg = style.Colors[ImGuiCol_ChildBg].w > 0.0f ? style.Colors[ImGuiCol_ChildBg]
                                                               : style.Colors[ImGuiCol_WindowBg];
    ImVec4 mixed = Mix(bg, color, std::clamp(strength, 0.0f, 1.0f));
    mixed.w = 1.0f;
    return mixed;
}

bool PrimaryButton(const char* label, float width, bool enabled, const char* disabled_reason, bool busy) {
    const ImGuiStyle& style = ImGui::GetStyle();
    const ImVec4 accent = Accent();
    const bool clickable = enabled && !busy;
    if (clickable) {
        ImGui::PushStyleColor(ImGuiCol_Button, accent);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, Mix(accent, style.Colors[ImGuiCol_Text], 0.15f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, Mix(accent, style.Colors[ImGuiCol_WindowBg], 0.2f));
        ImGui::PushStyleColor(ImGuiCol_Text, OnAccent(accent));
    } else if (busy) {
        ImGui::PushStyleColor(ImGuiCol_Button, accent);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, accent);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, accent);
        ImGui::PushStyleColor(ImGuiCol_Text, OnAccent(accent));
    } else {
        ImGui::PushStyleColor(ImGuiCol_Button, style.Colors[ImGuiCol_FrameBg]);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, style.Colors[ImGuiCol_FrameBg]);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, style.Colors[ImGuiCol_FrameBg]);
        ImGui::PushStyleColor(ImGuiCol_Text, style.Colors[ImGuiCol_TextDisabled]);
    }
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, kRounding);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, kButtonPadding);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);

    const float w = width < 0.0f ? ImGui::GetContentRegionAvail().x : width;
    bool pressed = false;
    if (busy) {
        // The label and a spinner, centred on a button nobody can press.
        ImGui::PushID(label);
        const ImVec2 text_size = ImGui::CalcTextSize(label, nullptr, true);
        const float radius = text_size.y * 0.42f;
        const float height = text_size.y + kButtonPadding.y * 2.0f;
        const float bw = w > 0.0f ? w : text_size.x + radius * 2.0f + 10.0f + kButtonPadding.x * 2.0f;
        ImGui::Button("##busy", ImVec2(bw, height));
        const ImVec2 min = ImGui::GetItemRectMin();
        const float total = radius * 2.0f + 10.0f + text_size.x;
        const float x = min.x + (bw - total) * 0.5f;
        const ImVec2 after = ImGui::GetCursorScreenPos();
        ImGui::SetCursorScreenPos(ImVec2(x, min.y + (height - radius * 2.0f) * 0.5f));
        const ImVec4 on_accent = OnAccent(accent);
        Spinner("##spin", radius, 2.5f, &on_accent);
        ImGui::SameLine(0.0f, 10.0f);
        ImGui::SetCursorScreenPos(ImVec2(x + radius * 2.0f + 10.0f, min.y + kButtonPadding.y));
        ImGui::TextUnformatted(label);
        ImGui::SetCursorScreenPos(after);
        ImGui::PopID();
    } else {
        pressed = ImGui::Button(label, ImVec2(w, 0.0f)) && clickable;
        if (!clickable && disabled_reason && disabled_reason[0] != '\0' &&
            ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip("%s", disabled_reason);
        }
    }

    ImGui::PopStyleVar(3);
    ImGui::PopStyleColor(4);
    return pressed;
}

ImVec2 FieldPadding() { return kFieldPadding; }

bool SecondaryButton(const char* label, float width, bool enabled, const ImVec2* padding) {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImGui::PushStyleColor(ImGuiCol_Button, style.Colors[ImGuiCol_FrameBg]);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, enabled ? style.Colors[ImGuiCol_FrameBgHovered] : style.Colors[ImGuiCol_FrameBg]);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, enabled ? style.Colors[ImGuiCol_FrameBgActive] : style.Colors[ImGuiCol_FrameBg]);
    ImGui::PushStyleColor(ImGuiCol_Text, enabled ? style.Colors[ImGuiCol_Text] : style.Colors[ImGuiCol_TextDisabled]);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, kRounding);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, padding ? *padding : ImVec2(14.0f, 8.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
    const float w = width < 0.0f ? ImGui::GetContentRegionAvail().x : width;
    const bool pressed = ImGui::Button(label, ImVec2(w, 0.0f)) && enabled;
    ImGui::PopStyleVar(3);
    ImGui::PopStyleColor(4);
    return pressed;
}

bool LinkButton(const char* label, bool enabled) {
    const ImVec4 color = enabled ? Accent() : Muted();
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_Text, color);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
    const bool pressed = ImGui::Button(label) && enabled;
    if (enabled && ImGui::IsItemHovered()) {
        const ImVec2 min = ImGui::GetItemRectMin();
        const ImVec2 max = ImGui::GetItemRectMax();
        ImGui::GetWindowDrawList()->AddLine(ImVec2(min.x, max.y - 1.0f), ImVec2(max.x, max.y - 1.0f),
                                            ImGui::GetColorU32(color), 1.0f);
        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    }
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor(4);
    return pressed;
}

void Spinner(const char* id, float radius, float thickness, const ImVec4* color) {
    ImGuiWindow* window = ImGui::GetCurrentWindow();
    if (window->SkipItems) return;
    ImGuiContext& g = *GImGui;
    const ImGuiID item_id = window->GetID(id);
    const ImVec2 pos = window->DC.CursorPos;
    const ImVec2 size(radius * 2.0f, radius * 2.0f);
    const ImRect bb(pos, ImVec2(pos.x + size.x, pos.y + size.y));
    ImGui::ItemSize(bb, 0.0f);
    if (!ImGui::ItemAdd(bb, item_id)) return;

    window->DrawList->PathClear();
    const float t = static_cast<float>(g.Time);
    const int num_segments = 24;
    const int start = static_cast<int>(std::fabs(std::sin(t * 1.8f) * (num_segments - 5)));
    const float a_min = IM_PI * 2.0f * static_cast<float>(start) / static_cast<float>(num_segments);
    const float a_max = IM_PI * 2.0f * static_cast<float>(num_segments - 3) / static_cast<float>(num_segments);
    const ImVec2 centre(pos.x + radius, pos.y + radius);
    for (int i = 0; i < num_segments; ++i) {
        const float a = a_min + (static_cast<float>(i) / static_cast<float>(num_segments)) * (a_max - a_min);
        window->DrawList->PathLineTo(
            ImVec2(centre.x + std::cos(a + t * 4.0f) * radius, centre.y + std::sin(a + t * 4.0f) * radius));
    }
    const ImVec4 c = color ? *color : Accent();
    window->DrawList->PathStroke(ImGui::GetColorU32(c), false, thickness);
}

bool TextField(const char* id, char* buf, std::size_t buf_size, const char* placeholder, ImGuiInputTextFlags flags,
               bool enabled, float width) {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImGui::PushStyleColor(ImGuiCol_FrameBg, style.Colors[ImGuiCol_FrameBg]);
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, style.Colors[ImGuiCol_FrameBg]);
    ImGui::PushStyleColor(ImGuiCol_FrameBgActive, style.Colors[ImGuiCol_FrameBg]);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, kRounding);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, kFieldPadding);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
    ImGui::SetNextItemWidth(width);
    if (!enabled) ImGui::BeginDisabled();
    const bool entered = ImGui::InputTextWithHint(id, placeholder ? placeholder : "", buf, buf_size, flags);
    if (!enabled) ImGui::EndDisabled();
    ImGui::PopStyleVar(3);
    ImGui::PopStyleColor(3);
    return entered && (flags & ImGuiInputTextFlags_EnterReturnsTrue);
}

}  // namespace cyxwiz::servernode::gui::ui
