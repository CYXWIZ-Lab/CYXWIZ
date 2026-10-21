#include "ui_widgets.h"

#include "icons.h"
#include "ui_buttons.h"
#include "ui_fonts.h"

#include <imgui.h>
#include <imgui_internal.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unordered_map>

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
    {
        FontScope medium(Font::Medium);
        ImGui::TextColored(t.text_bright, "%s", title);
    }
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
    // A card is a tinted area, not an outline (owner rule 2026-10-01).
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, t.card_padding);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_panel);
    ImGui::PushStyleColor(ImGuiCol_Border, t.border);
    ImGui::BeginChild("##card", ImVec2(0.0f, 0.0f),
                      ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);
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
    {
        FontScope medium(Font::Medium);
        ImGui::TextColored(t.text_bright, "%s", title);
    }
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

void FactsTable(const char* id, const std::vector<KeyValue>& rows) {
    const Tokens& t = CurrentTokens();
    if (!ImGui::BeginTable(id, 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_NoSavedSettings)) return;
    ImGui::TableSetupColumn("key", ImGuiTableColumnFlags_WidthStretch, 0.28f);
    ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch, 1.0f);
    for (const auto& row : rows) {
        if (row.value.empty()) continue;
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextColored(t.text_dim, "%s", row.key.c_str());
        ImGui::TableNextColumn();
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextUnformatted(row.value.c_str());
        ImGui::PopTextWrapPos();
    }
    ImGui::EndTable();
}

// ---------------------------------------------------------------------------
// Status vocabulary

void StatusText(Status status, const char* text) {
    const StatusStyle s = StatusStyleFor(status);
    ImGui::TextColored(s.colour, "%s %s", s.icon, text && text[0] ? text : s.label);
}

void Chip(const char* text, const ImVec4& colour) {
    const Tokens& t = CurrentTokens();
    const float line_height = ImGui::GetFrameHeight() - 4.0f;
    const float dot = 6.0f;
    const float inset = t.space_md * 2.0f + dot + t.space_sm;
    ImVec2 text_size = ImGui::CalcTextSize(text);
    // Wider than the room left: the text wraps inside a taller pill (a narrow
    // panel cut the chip off at its edge).
    float wrap = 0.0f;
    const float avail = ImGui::GetContentRegionAvail().x;
    if (inset + text_size.x > avail && avail > inset + ImGui::GetFontSize() * 2.0f) {
        wrap = avail - inset;
        text_size = ImGui::CalcTextSize(text, nullptr, false, wrap);
    }
    const float height = std::max(line_height, text_size.y + (line_height - ImGui::GetTextLineHeight()));
    const ImVec2 size(inset + text_size.x, height);
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    ImGui::Dummy(size);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + size.y), ToU32(WithAlpha(colour, 0.16f)), line_height * 0.5f);
    const float top = pos.y + (height - text_size.y) * 0.5f;
    dl->AddCircleFilled(ImVec2(pos.x + t.space_md + dot * 0.5f, top + ImGui::GetTextLineHeight() * 0.5f), dot * 0.5f,
                        ToU32(colour));
    dl->AddText(ImGui::GetFont(), ImGui::GetFontSize(), ImVec2(pos.x + t.space_md + dot + t.space_sm, top),
                ToU32(colour), text, nullptr, wrap);
}

float ChipWidth(const char* text) {
    const Tokens& t = CurrentTokens();
    return t.space_md * 2.0f + 6.0f + t.space_sm + ImGui::CalcTextSize(text).x;
}

void StatusChip(Status status, const char* text) {
    const StatusStyle s = StatusStyleFor(status);
    Chip(text && text[0] ? text : s.label, s.colour);
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

bool SameLineRight(float width, float gap) {
    // Window-local x where the last item ended (GetCursorPosX would already
    // be at the start of the next line).
    const float last_end = ImGui::GetItemRectMax().x - ImGui::GetWindowPos().x + ImGui::GetScrollX();
    const float x = ImGui::GetContentRegionMax().x - width;
    if (x < last_end + gap) return false;
    ImGui::SameLine(x);
    return true;
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
// Settings column + results area

namespace {
constexpr float kHandleWidth = 7.0f;
constexpr float kMinColumnWidth = 200.0f;
}  // namespace

void BeginSettingsColumn(const char* id, float default_width) {
    const Tokens& t = CurrentTokens();
    ImGui::PushID(id);
    float* width = ImGui::GetStateStorage()->GetFloatRef(ImGui::GetID("##column_width"), default_width);
    const float avail = ImGui::GetContentRegionAvail().x;
    *width = std::clamp(*width, kMinColumnWidth, std::max(kMinColumnWidth, avail * 0.6f));

    // Areas are told apart by tone, not outlines (owner rule 2026-10-01): the
    // column sits one step above the window, so the input-toned fields read on it.
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, Mix(t.bg_window, t.bg_raised, 0.4f));
    ImGui::BeginChild("##settings", ImVec2(*width, 0.0f), ImGuiChildFlags_AlwaysUseWindowPadding);
    ImGui::PopStyleColor();
}

void BeginResultsArea() {
    const Tokens& t = CurrentTokens();
    ImGui::EndChild();

    // Drag handle between the panes: a short grip in the surface tone.
    float* width = ImGui::GetStateStorage()->GetFloatRef(ImGui::GetID("##column_width"));
    ImGui::SameLine(0.0f, 0.0f);
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const float height = std::max(ImGui::GetContentRegionAvail().y, 1.0f);
    ImGui::InvisibleButton("##handle", ImVec2(kHandleWidth, height));
    const bool hovered = ImGui::IsItemHovered();
    const bool active = ImGui::IsItemActive();
    if (hovered || active) ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    if (active) *width += ImGui::GetIO().MouseDelta.x;
    const float grip = 28.0f;
    const ImVec2 grip_min(pos.x + kHandleWidth * 0.5f - 1.0f, pos.y + (height - grip) * 0.5f);
    ImGui::GetWindowDrawList()->AddRectFilled(grip_min, ImVec2(grip_min.x + 2.0f, grip_min.y + grip),
                                              ToU32(hovered || active ? t.border : t.border_soft), 1.0f);

    ImGui::SameLine(0.0f, 0.0f);
    ImGui::BeginChild("##results", ImVec2(0.0f, 0.0f), ImGuiChildFlags_AlwaysUseWindowPadding);
}

void EndResultsArea() {
    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopID();
}

void SettingsSection(const char* title) {
    const Tokens& t = CurrentTokens();
    std::string upper(title);
    for (char& c : upper) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    if (ImGui::GetCursorPosY() > ImGui::GetStyle().WindowPadding.y + 1.0f) ImGui::Spacing();
    FontScope font(Font::Medium);
    ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
    ImGui::TextUnformatted(upper.c_str());
    ImGui::PopStyleColor();
}

void FieldLabel(const char* label) {
    const Tokens& t = CurrentTokens();
    ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
    // In a table the next cell inherits the frame baseline of the widget in the
    // cell before it, which drew this label a frame padding lower; start at the top.
    ImGui::GetCurrentWindow()->DC.CurrLineTextBaseOffset = 0.0f;
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(label);
    ImGui::PopTextWrapPos();
    ImGui::PopStyleColor();
    ImGui::SetNextItemWidth(-FLT_MIN);
}

void BeginInsetList(const char* id, float height) {
    const Tokens& t = CurrentTokens();
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, t.rounding_field);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_input);
    // Checkbox frames on the raised tone, so an unchecked box shows on the list.
    ImGui::PushStyleColor(ImGuiCol_FrameBg, t.bg_raised);
    const ImGuiChildFlags flags =
        ImGuiChildFlags_AlwaysUseWindowPadding | (height > 0.0f ? ImGuiChildFlags_None : ImGuiChildFlags_AutoResizeY);
    ImGui::BeginChild(id, ImVec2(0.0f, height), flags);
}

void EndInsetList() {
    ImGui::EndChild();
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(2);
}

// ---------------------------------------------------------------------------
// Empty state

bool NumberField(const char* id, double* value, const NumberFieldOptions& o) {
    // Drafts by widget id while a field is being typed in.
    static std::unordered_map<ImGuiID, double> drafts;
    const ImGuiID key = ImGui::GetID(id);
    const auto clamp = [&](double v) { return std::isfinite(v) ? std::clamp(v, o.min, o.max) : *value; };
    bool changed = false;
    ImGui::PushID(id);
    auto it = drafts.find(key);
    double shown = it != drafts.end() ? it->second : *value;
    ImGui::SetNextItemWidth(o.width);
    if (ImGui::InputDouble("##number", &shown, 0.0, 0.0, o.format)) drafts[key] = shown;
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        if (auto d = drafts.find(key); d != drafts.end()) {
            const double v = clamp(d->second);
            drafts.erase(d);
            if (v != *value) {
                *value = v;
                changed = true;
            }
        }
    } else if (!ImGui::IsItemActive()) {
        drafts.erase(key);
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip)) {
        char range[96] = "";
        const bool low = o.min > -1e299, high = o.max < 1e299;
        char lo[32], hi[32];
        std::snprintf(lo, sizeof(lo), o.format, o.min);
        std::snprintf(hi, sizeof(hi), o.format, o.max);
        if (low && high) std::snprintf(range, sizeof(range), "%s to %s", lo, hi);
        else if (low) std::snprintf(range, sizeof(range), "at least %s", lo);
        else if (high) std::snprintf(range, sizeof(range), "at most %s", hi);
        if (o.hint && *range) ImGui::SetTooltip("%s (%s)", o.hint, range);
        else if (o.hint) ImGui::SetTooltip("%s", o.hint);
        else if (*range) ImGui::SetTooltip("%s", range);
    }
    if (o.step > 0.0) {
        for (const auto& [label, sign] : {std::pair<const char*, double>{"-", -1.0}, {"+", 1.0}}) {
            ImGui::SameLine(0.0f, 4.0f);
            if (SecondaryButton(label)) {
                // Whole steps from the value, so 1.5 + 0.1 is 1.6, not 1.6000000000000001.
                const double v = clamp(std::round((*value + sign * o.step) / o.step) * o.step);
                if (v != *value) {
                    *value = v;
                    changed = true;
                }
            }
        }
    }
    ImGui::PopID();
    return changed;
}

bool NumberField(const char* id, float* value, const NumberFieldOptions& options) {
    double v = *value;
    if (!NumberField(id, &v, options)) return false;
    *value = static_cast<float>(v);
    return true;
}

bool NumberField(const char* id, int* value, int min, int max, int step, float width, const char* hint) {
    NumberFieldOptions o;
    o.min = min;
    o.max = max;
    o.step = step;
    o.format = "%.0f";
    o.width = width;
    o.hint = hint;
    double v = *value;
    if (!NumberField(id, &v, o)) return false;
    *value = static_cast<int>(std::lround(v));
    return true;
}

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
    // The title and the hint wrap into centred lines that fit the area (a
    // single line ran past both edges of a narrow panel).
    const float max_width = std::min(std::max(avail.x - ImGui::GetStyle().ItemSpacing.x * 2.0f, 80.0f), 420.0f);
    auto wrapped = [&](const char* text, const ImVec4& colour) {
        std::string line;
        std::string word;
        const auto flush = [&]() {
            if (!line.empty()) centred(line.c_str(), colour, 1.0f);
            line.clear();
        };
        for (const char* c = text;; ++c) {
            if (*c == ' ' || *c == '\0') {
                const std::string candidate = line.empty() ? word : line + " " + word;
                if (!line.empty() && ImGui::CalcTextSize(candidate.c_str()).x > max_width) {
                    flush();
                    line = word;
                } else {
                    line = candidate;
                }
                word.clear();
                if (*c == '\0') break;
            } else {
                word += *c;
            }
        }
        flush();
    };
    if (icon && icon[0]) centred(icon, t.text_faint, 2.0f);
    wrapped(title, t.text_dim);
    if (hint && hint[0]) wrapped(hint, t.text_faint);
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
                       const char* primary_disabled_reason, bool danger, bool close_on_primary) {
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
    if (result == DialogResult::Secondary || (result == DialogResult::Primary && close_on_primary)) {
        ImGui::CloseCurrentPopup();
    }
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
