#include "training_dashboard_look.h"

#include "../ui_tokens.h"

#include <imgui.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace cyxwiz::dash {

ImVec4 MixColor(const ImVec4& a, const ImVec4& b, float t) {
    return ImVec4(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t,
                  a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t);
}

ImVec4 WithAlpha(ImVec4 color, float alpha) {
    color.w = alpha;
    return color;
}

DashColors CurrentDashColors() {
    const cyxwiz::ui::Tokens& t = cyxwiz::ui::CurrentTokens();
    DashColors c;
    c.window = t.bg_window;
    c.light = t.light;
    const ImVec4 lift = t.text_bright;
    c.text = t.text;
    c.muted = t.text_dim;
    c.faint = t.text_faint;
    c.card = WithAlpha(MixColor(c.window, lift, c.light ? 0.035f : 0.03f), 1.0f);
    c.track = WithAlpha(MixColor(c.window, lift, c.light ? 0.09f : 0.08f), 1.0f);
    c.border = t.border;
    c.accent = t.accent;
    c.accent_text = t.accent_text;
    c.success = t.success;
    c.warning = t.warning;
    c.caution = t.caution;
    c.error = t.error;
    c.info = t.info;
    return c;
}

// Card: a filled, rounded, borderless group sized to its content.
bool BeginDashCard(const char* id, const DashColors& c, float width) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, c.card);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 8.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(14.0f, 10.0f));
    const bool open = ImGui::BeginChild(
        id, ImVec2(width, 0.0f),
        ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PopStyleVar(3);
    ImGui::PopStyleColor();
    return open;
}

void EndDashCard() {
    ImGui::EndChild();
}

void DashCardTitle(const char* title, const char* subtitle, const DashColors& c) {
    ImGui::TextColored(c.text, "%s", title);
    if (subtitle && *subtitle) {
        ImGui::SameLine(0.0f, 8.0f);
        ImGui::TextColored(c.muted, "%s", subtitle);
    }
}

// Status pill: tinted background, a dot and the label in the status colour.
void DashPill(const char* label, const ImVec4& color, bool dot) {
    ImDrawList* draw = ImGui::GetWindowDrawList();
    const ImVec2 text = ImGui::CalcTextSize(label);
    const float pad_x = 9.0f;
    const float pad_y = 3.0f;
    const float dot_space = dot ? 12.0f : 0.0f;
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const ImVec2 size(text.x + pad_x * 2.0f + dot_space, text.y + pad_y * 2.0f);
    draw->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + size.y),
                        ImGui::GetColorU32(WithAlpha(color, 0.16f)), size.y * 0.5f);
    if (dot) {
        draw->AddCircleFilled(ImVec2(pos.x + pad_x + 3.5f, pos.y + size.y * 0.5f), 3.5f,
                              ImGui::GetColorU32(color));
    }
    draw->AddText(ImVec2(pos.x + pad_x + dot_space, pos.y + pad_y),
                  ImGui::GetColorU32(color), label);
    ImGui::Dummy(size);
}

// Thin rounded progress bar.
void DashProgress(float fraction, const ImVec4& color, const DashColors& c,
                  float width, float height) {
    fraction = std::clamp(fraction, 0.0f, 1.0f);
    if (width <= 0.0f) {
        width = ImGui::GetContentRegionAvail().x;
    }
    ImDrawList* draw = ImGui::GetWindowDrawList();
    const float line_h = ImGui::GetTextLineHeight();
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const float y = pos.y + (line_h - height) * 0.5f;
    draw->AddRectFilled(ImVec2(pos.x, y), ImVec2(pos.x + width, y + height),
                        ImGui::GetColorU32(c.track), height * 0.5f);
    if (fraction > 0.0f) {
        draw->AddRectFilled(ImVec2(pos.x, y),
                            ImVec2(pos.x + std::max(height, width * fraction), y + height),
                            ImGui::GetColorU32(color), height * 0.5f);
    }
    ImGui::Dummy(ImVec2(width, line_h));
}


bool DashButton(const char* label, DashButtonKind kind, const DashColors& c) {
    const ImVec4 fg = kind == DashButtonKind::Danger ? c.error : c.text;
    const ImVec4 hover = kind == DashButtonKind::Danger
        ? WithAlpha(c.error, 0.14f)
        : WithAlpha(c.text, 0.07f);
    // A fill instead of an outline (no outlines on buttons).
    ImGui::PushStyleColor(ImGuiCol_Button, kind == DashButtonKind::Danger ? ImVec4(0, 0, 0, 0) : WithAlpha(c.text, 0.05f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hover);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, WithAlpha(hover, hover.w * 1.8f));
    ImGui::PushStyleColor(ImGuiCol_Border,
                          kind == DashButtonKind::Danger ? WithAlpha(c.error, 0.45f) : c.border);
    ImGui::PushStyleColor(ImGuiCol_Text, fg);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10.0f, 4.0f));
    const bool pressed = ImGui::Button(label);
    ImGui::PopStyleVar(3);
    ImGui::PopStyleColor(5);
    return pressed;
}

float DashButtonWidth(const char* label) {
    return ImGui::CalcTextSize(label, nullptr, true).x + 20.0f;
}

// Toggle chip: filled with the accent tint while on.
bool DashChip(const char* label, bool* value, const DashColors& c) {
    const bool on = *value;
    ImGui::PushStyleColor(ImGuiCol_Button, on ? WithAlpha(c.accent, 0.24f) : WithAlpha(c.text, 0.05f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                          on ? WithAlpha(c.accent, 0.34f) : WithAlpha(c.text, 0.07f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, WithAlpha(c.accent, 0.45f));
    ImGui::PushStyleColor(ImGuiCol_Border, on ? WithAlpha(c.accent, 0.55f) : c.border);
    ImGui::PushStyleColor(ImGuiCol_Text, on ? c.accent_text : c.muted);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 11.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10.0f, 3.0f));
    const bool pressed = ImGui::Button(label);
    ImGui::PopStyleVar(3);
    ImGui::PopStyleColor(5);
    if (pressed) {
        *value = !*value;
    }
    return pressed;
}

// Bold UI font for metric values (falls back to the current font).
ImFont* DashValueFont() {
    ImGuiIO& io = ImGui::GetIO();
    for (ImFont* font : io.Fonts->Fonts) {
        if (!font || !font->Sources || font->SourcesCount <= 0) {
            continue;
        }
        const char* name = font->Sources[0].Name;
        if (std::strstr(name, "Bold") && !std::strstr(name, "Mono")) {
            return font;
        }
    }
    return ImGui::GetFont();
}

void DashBigValue(const char* text, const ImVec4& color) {
    ImFont* font = DashValueFont();
    const float size = ImGui::GetFontSize() * 1.45f;
    const ImVec2 extent = font->CalcTextSizeA(size, FLT_MAX, 0.0f, text);
    ImGui::GetWindowDrawList()->AddText(font, size, ImGui::GetCursorScreenPos(),
                                        ImGui::GetColorU32(color), text);
    ImGui::Dummy(extent);
}

std::string FormatEpochValue(double epoch) {
    char buffer[32];
    if (std::abs(epoch - std::round(epoch)) < 1e-6) {
        std::snprintf(buffer, sizeof(buffer), "%d", static_cast<int>(std::round(epoch)));
    } else {
        std::snprintf(buffer, sizeof(buffer), "%.2f", epoch);
    }
    return buffer;
}

// Small borderless icon button, tinted with the accent while active.
bool DashIconButton(const char* id_label, bool active, const DashColors& c) {
    ImGui::PushStyleColor(ImGuiCol_Button, active ? WithAlpha(c.accent, 0.24f) : ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, WithAlpha(c.text, 0.08f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, WithAlpha(c.accent, 0.40f));
    ImGui::PushStyleColor(ImGuiCol_Text, active ? c.accent_text : c.muted);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 5.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6.0f, 2.0f));
    const bool pressed = ImGui::Button(id_label);
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor(4);
    return pressed;
}

void DashKpiCard(const char* id, const char* label, const ImVec4& dot, const char* value, const char* tooltip,
                 const char* line1, const ImVec4& line1_color, const char* line2, const ImVec4& line2_color,
                 const DashColors& c) {
    if (BeginDashCard(id, c)) {
        const float line = ImGui::GetTextLineHeight();
        const ImVec2 pos = ImGui::GetCursorScreenPos();
        ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(pos.x + 4.0f, pos.y + line * 0.5f), 4.0f,
                                                    ImGui::GetColorU32(dot));
        ImGui::Dummy(ImVec2(8.0f, line));
        ImGui::SameLine(0.0f, 6.0f);
        ImGui::TextColored(c.muted, "%s", label);
        DashBigValue(value, c.text);
        if (tooltip && *tooltip && ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", tooltip);
        }
        ImGui::TextColored(line1_color, "%s", line1);
        ImGui::TextColored(line2_color, "%s", line2);
    }
    EndDashCard();
}

bool DashRunTab(const char* id, const char* label, const char* sub, const ImVec4& dot, bool active,
                const DashColors& c) {
    const ImVec2 pad(10.0f, 5.0f);
    const float dot_space = 14.0f;
    const ImVec2 label_size = ImGui::CalcTextSize(label);
    const ImVec2 sub_size = (sub && *sub) ? ImGui::CalcTextSize(sub) : ImVec2(0.0f, 0.0f);
    const float gap = (sub && *sub) ? 8.0f : 0.0f;
    const ImVec2 size(pad.x * 2.0f + dot_space + label_size.x + gap + sub_size.x, label_size.y + pad.y * 2.0f);
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const bool pressed = ImGui::InvisibleButton(id, size);
    const bool hovered = ImGui::IsItemHovered();
    ImDrawList* draw = ImGui::GetWindowDrawList();
    const ImVec4 fill = active ? c.track : WithAlpha(c.text, hovered ? 0.07f : 0.04f);
    draw->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + size.y), ImGui::GetColorU32(fill), 6.0f);
    if (active) {
        draw->AddRectFilled(ImVec2(pos.x + 6.0f, pos.y + size.y - 2.0f), ImVec2(pos.x + size.x - 6.0f, pos.y + size.y),
                            ImGui::GetColorU32(c.accent), 1.0f);
    }
    draw->AddCircleFilled(ImVec2(pos.x + pad.x + 4.0f, pos.y + size.y * 0.5f), 4.0f, ImGui::GetColorU32(dot));
    const float x = pos.x + pad.x + dot_space;
    draw->AddText(ImVec2(x, pos.y + pad.y), ImGui::GetColorU32(active ? c.text : c.muted), label);
    if (sub && *sub) {
        draw->AddText(ImVec2(x + label_size.x + gap, pos.y + pad.y), ImGui::GetColorU32(c.faint), sub);
    }
    if (hovered) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    return pressed;
}

}  // namespace cyxwiz::dash
