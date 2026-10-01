#include "ui_widgets.h"

#include "icons.h"
#include "ui_buttons.h"

#include <imgui.h>
#include <imgui_internal.h>

#include <algorithm>
#include <cstring>

namespace cyxwiz::ui {

namespace {

constexpr float kTooltipWrap = 420.0f;

ImVec2 WorkSize() {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    return viewport ? viewport->WorkSize : ImVec2(1280.0f, 800.0f);
}

ImVec2 WorkCenter() {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    return viewport ? viewport->GetWorkCenter() : ImVec2(640.0f, 400.0f);
}

}  // namespace

// ---------------------------------------------------------------------------
// Headers and cards

void SectionHeader(const char* title, const char* subtitle) {
    const Tokens& t = CurrentTokens();
    ImGui::Spacing();
    ImGui::TextColored(t.text_bright, "%s", title);
    if (subtitle && subtitle[0]) {
        ImGui::SameLine(0.0f, t.space_lg);
        ImGui::TextColored(t.text_dim, "%s", subtitle);
    }
    ImGui::Spacing();
}

void BeginCard(const char* id) {
    const Tokens& t = CurrentTokens();
    ImGui::PushID(id);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, t.rounding_card);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 1.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, t.card_padding);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_panel);
    ImGui::PushStyleColor(ImGuiCol_Border, t.border);
    ImGui::BeginChild("##card", ImVec2(0.0f, 0.0f),
                      ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);
}

void EndCard() {
    ImGui::EndChild();
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(3);
    ImGui::PopID();
    ImGui::Spacing();
}

void CardHeader(const char* title, const char* subtitle, const StatusStyle* status) {
    const Tokens& t = CurrentTokens();
    ImGui::TextColored(t.text_bright, "%s", title);
    if (subtitle && subtitle[0]) {
        ImGui::SameLine(0.0f, t.space_md);
        ImGui::TextColored(t.text_dim, "%s", subtitle);
    }
    if (status) {
        const std::string text = std::string(status->icon) + " " + status->label;
        const float width = ImGui::CalcTextSize(text.c_str()).x;
        const float x = ImGui::GetWindowContentRegionMax().x - width;
        if (x > ImGui::GetCursorPosX()) {
            ImGui::SameLine(x);
            ImGui::TextColored(status->colour, "%s", text.c_str());
        } else {
            ImGui::TextColored(status->colour, "%s", text.c_str());
        }
    }
    ImGui::Separator();
    ImGui::Spacing();
}

// ---------------------------------------------------------------------------
// Key/value details

void KeyValueTable(const char* id, const std::vector<KeyValue>& rows, int pairs_per_line, bool copy_on_click) {
    const Tokens& t = CurrentTokens();
    std::vector<const KeyValue*> shown;
    for (const auto& row : rows)
        if (!row.value.empty()) shown.push_back(&row);
    if (shown.empty()) return;

    const int pairs = std::clamp(pairs_per_line, 1, 2);
    const ImGuiTableFlags flags = ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersOuter |
                                  ImGuiTableFlags_PadOuterX | ImGuiTableFlags_NoSavedSettings;
    if (!ImGui::BeginTable(id, pairs * 2, flags)) return;
    for (int p = 0; p < pairs; ++p) {
        ImGui::TableSetupColumn("key", ImGuiTableColumnFlags_WidthStretch, 0.8f);
        ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch, 1.6f);
    }
    for (size_t i = 0; i < shown.size(); ++i) {
        if (i % pairs == 0) ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextColored(t.text_dim, "%s", shown[i]->key.c_str());
        ImGui::TableNextColumn();
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextUnformatted(shown[i]->value.c_str());
        ImGui::PopTextWrapPos();
        if (copy_on_click) {
            if (ImGui::IsItemHovered()) {
                ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                ImGui::SetTooltip("Click to copy");
            }
            if (ImGui::IsItemClicked()) ImGui::SetClipboardText(shown[i]->value.c_str());
        }
    }
    ImGui::EndTable();
}

// ---------------------------------------------------------------------------
// Status vocabulary

void StatusText(Status status, const char* text) {
    const StatusStyle s = StatusStyleFor(status);
    ImGui::TextColored(s.colour, "%s %s", s.icon, text && text[0] ? text : s.label);
}

void StatusChip(Status status, const char* text) {
    const Tokens& t = CurrentTokens();
    const StatusStyle s = StatusStyleFor(status);
    const char* label = text && text[0] ? text : s.label;
    const float height = ImGui::GetFrameHeight() - 4.0f;
    const float dot = 6.0f;
    const ImVec2 text_size = ImGui::CalcTextSize(label);
    const ImVec2 size(t.space_md * 2.0f + dot + t.space_sm + text_size.x, height);
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    ImGui::Dummy(size);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + size.y), ToU32(WithAlpha(s.colour, 0.16f)), height * 0.5f);
    const float cy = pos.y + height * 0.5f;
    dl->AddCircleFilled(ImVec2(pos.x + t.space_md + dot * 0.5f, cy), dot * 0.5f, ToU32(s.colour));
    dl->AddText(ImVec2(pos.x + t.space_md + dot + t.space_sm, cy - text_size.y * 0.5f), ToU32(s.colour), label);
}

void StatusLegend(std::initializer_list<Status> statuses) {
    const Tokens& t = CurrentTokens();
    ImGui::TextColored(t.text_dim, "Status:");
    for (Status status : statuses) {
        const StatusStyle s = StatusStyleFor(status);
        ImGui::SameLine(0.0f, t.space_lg);
        ImGui::TextColored(s.colour, "%s", s.icon);
        ImGui::SameLine(0.0f, t.space_xs);
        ImGui::TextColored(t.text_dim, "%s", s.label);
    }
}

// ---------------------------------------------------------------------------
// Tooltips

void Tooltip(const char* text) {
    if (!text || !text[0]) return;
    if (!ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled | ImGuiHoveredFlags_DelayNormal)) return;
    ImGui::BeginTooltip();
    ImGui::PushTextWrapPos(kTooltipWrap);
    ImGui::TextUnformatted(text);
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
}

void HelpMarker(const char* text) {
    const Tokens& t = CurrentTokens();
    ImGui::TextColored(t.text_dim, "(?)");
    Tooltip(text);
}

// ---------------------------------------------------------------------------
// Search field

bool SearchField(const char* id, char* buffer, size_t size, const char* hint, float width) {
    const Tokens& t = CurrentTokens();
    const std::string hint_text = std::string(ICON_FA_MAGNIFYING_GLASS) + "  " + (hint ? hint : "Search");
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, t.rounding_field);
    ImGui::PushStyleColor(ImGuiCol_FrameBg, t.bg_input);
    ImGui::SetNextItemWidth(width > 0.0f ? width : -FLT_MIN);
    const bool changed = ImGui::InputTextWithHint(id, hint_text.c_str(), buffer, size);
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
    return changed;
}

// ---------------------------------------------------------------------------
// Flow row

FlowRow::FlowRow(float right_edge)
    : right(right_edge > 0.0f ? right_edge : ImGui::GetWindowContentRegionMax().x) {}

bool FlowRow::Next(float item_width) {
    if (first) {
        first = false;
        return false;
    }
    const float x = ImGui::GetItemRectMax().x - ImGui::GetWindowPos().x + ImGui::GetStyle().ItemSpacing.x;
    if (x + item_width <= right) {
        ImGui::SameLine();
        return false;
    }
    return true;  // the caller draws on the new line ImGui already started
}

// ---------------------------------------------------------------------------
// Empty state

void EmptyState(const char* icon, const char* title, const char* hint) {
    const Tokens& t = CurrentTokens();
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const float block = (icon && icon[0] ? ImGui::GetTextLineHeight() * 2.0f : 0.0f) +
                        ImGui::GetTextLineHeightWithSpacing() * (hint && hint[0] ? 2.0f : 1.0f);
    const float top = std::max(0.0f, (avail.y - block) * 0.4f);
    ImGui::Dummy(ImVec2(0.0f, top));
    auto centred = [&](const char* text, const ImVec4& colour, float scale) {
        ImGui::SetWindowFontScale(scale);
        const float width = ImGui::CalcTextSize(text).x;
        ImGui::SetCursorPosX(std::max(0.0f, (ImGui::GetWindowContentRegionMax().x - width) * 0.5f));
        ImGui::TextColored(colour, "%s", text);
        ImGui::SetWindowFontScale(1.0f);
    };
    if (icon && icon[0]) centred(icon, t.text_faint, 2.0f);
    centred(title, t.text_dim, 1.0f);
    if (hint && hint[0]) centred(hint, t.text_faint, 1.0f);
}

// ---------------------------------------------------------------------------
// Dialogs

namespace {
float FooterHeight() {
    const ImGuiStyle& style = ImGui::GetStyle();
    return ImGui::GetFrameHeight() + CurrentTokens().button_padding.y + style.ItemSpacing.y * 3.0f + 1.0f;
}
}  // namespace

bool BeginDialog(const char* title, const DialogOptions& options) {
    const Tokens& t = CurrentTokens();
    const ImVec2 work = WorkSize();
    const ImVec2 size(std::min(options.size.x, work.x * 0.92f), std::min(options.size.y, work.y * 0.92f));
    ImGui::SetNextWindowPos(WorkCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(size, ImGuiCond_Appearing);
    ImGui::SetNextWindowSizeConstraints(options.min_size, ImVec2(work.x * 0.95f, work.y * 0.95f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, t.rounding_card);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(t.space_xl, t.space_lg));
    const ImGuiWindowFlags flags = options.resizable ? ImGuiWindowFlags_None : ImGuiWindowFlags_NoResize;
    const bool open = ImGui::BeginPopupModal(title, options.open, flags);
    ImGui::PopStyleVar(2);
    if (!open) return false;
    ImGui::BeginChild("##dialog_body", ImVec2(0.0f, -FooterHeight()), ImGuiChildFlags_None);
    return true;
}

DialogResult EndDialog(const char* primary_label, const char* secondary_label, bool primary_enabled,
                       const char* primary_disabled_reason, bool danger) {
    const Tokens& t = CurrentTokens();
    ImGui::EndChild();
    ImGui::Separator();
    ImGui::Spacing();
    DialogResult result = DialogResult::None;
    const float primary_w = ButtonWidth(primary_label, ButtonSize::Regular);
    const float secondary_w = secondary_label && secondary_label[0] ? ButtonWidth(secondary_label, ButtonSize::Regular) : 0.0f;
    const float total = primary_w + (secondary_w > 0.0f ? secondary_w + t.space_md : 0.0f);
    const float x = ImGui::GetWindowContentRegionMax().x - total;
    if (x > ImGui::GetCursorPosX()) ImGui::SetCursorPosX(x);
    const bool primary = danger ? DangerButton(primary_label, primary_enabled, primary_disabled_reason, ButtonSize::Regular)
                                : PrimaryButton(primary_label, primary_enabled, primary_disabled_reason);
    if (primary) result = DialogResult::Primary;
    if (secondary_w > 0.0f) {
        ImGui::SameLine(0.0f, t.space_md);
        if (SecondaryButton(secondary_label, true, nullptr, ButtonSize::Regular)) result = DialogResult::Secondary;
    }
    if (result == DialogResult::None && ImGui::IsKeyPressed(ImGuiKey_Escape)) result = DialogResult::Secondary;
    if (result != DialogResult::None) ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
    return result;
}

DialogResult ConfirmDialog(const char* title, const char* message, const char* confirm_label, bool danger,
                           const std::vector<std::string>& bullets) {
    const Tokens& t = CurrentTokens();
    ImGui::SetNextWindowPos(WorkCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSizeConstraints(ImVec2(460.0f, 0.0f), ImVec2(640.0f, WorkSize().y * 0.9f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, t.rounding_card);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(t.space_xl, t.space_lg));
    const bool open = ImGui::BeginPopupModal(title, nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::PopStyleVar(2);
    if (!open) return DialogResult::None;

    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + 600.0f);
    ImGui::TextUnformatted(message);
    ImGui::PopTextWrapPos();
    if (!bullets.empty()) {
        ImGui::Spacing();
        for (const auto& b : bullets) ImGui::BulletText("%s", b.c_str());
    }
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    DialogResult result = DialogResult::None;
    const float total = ButtonWidth(confirm_label, ButtonSize::Regular) + t.space_md + ButtonWidth("Cancel", ButtonSize::Regular);
    const float x = ImGui::GetWindowContentRegionMax().x - total;
    if (x > ImGui::GetCursorPosX()) ImGui::SetCursorPosX(x);
    const bool confirmed = danger ? DangerButton(confirm_label, true, nullptr, ButtonSize::Regular)
                                  : PrimaryButton(confirm_label);
    if (confirmed) result = DialogResult::Primary;
    ImGui::SameLine(0.0f, t.space_md);
    if (SecondaryButton("Cancel", true, nullptr, ButtonSize::Regular)) result = DialogResult::Secondary;
    if (result == DialogResult::None && ImGui::IsKeyPressed(ImGuiKey_Escape)) result = DialogResult::Secondary;
    if (result != DialogResult::None) ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
    return result;
}

bool MessageDialog(const char* title, const char* message) {
    const Tokens& t = CurrentTokens();
    ImGui::SetNextWindowPos(WorkCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSizeConstraints(ImVec2(420.0f, 0.0f), ImVec2(640.0f, WorkSize().y * 0.9f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, t.rounding_card);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(t.space_xl, t.space_lg));
    const bool open = ImGui::BeginPopupModal(title, nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::PopStyleVar(2);
    if (!open) return false;
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + 600.0f);
    ImGui::TextUnformatted(message);
    ImGui::PopTextWrapPos();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    const float x = ImGui::GetWindowContentRegionMax().x - ButtonWidth("OK", ButtonSize::Regular);
    if (x > ImGui::GetCursorPosX()) ImGui::SetCursorPosX(x);
    bool dismissed = PrimaryButton("OK") || ImGui::IsKeyPressed(ImGuiKey_Escape) || ImGui::IsKeyPressed(ImGuiKey_Enter);
    if (dismissed) ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
    return dismissed;
}

}  // namespace cyxwiz::ui
