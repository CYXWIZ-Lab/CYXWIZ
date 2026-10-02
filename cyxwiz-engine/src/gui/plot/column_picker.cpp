#include "column_picker.h"

#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"

#include <imgui.h>

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cstring>

namespace cyxwiz::plot {

namespace {

std::string Lower(const char* s) {
    std::string out(s);
    for (char& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

int FindColumn(const std::vector<ColumnSummary>& columns, const char* name) {
    const std::string want = Lower(name);
    if (want.empty()) return -1;
    for (size_t i = 0; i < columns.size(); ++i)
        if (Lower(columns[i].name.c_str()) == want) return static_cast<int>(i);
    return -1;
}

bool Contains(const std::vector<std::string>& values, const std::string& v) {
    return std::find(values.begin(), values.end(), v) != values.end();
}

void PickerSize(float width) {
    ImGui::SetNextItemWidth(width);
    ImGui::SetNextWindowSizeConstraints(ImVec2(std::max(width, 340.0f), 0.0f), ImVec2(FLT_MAX, 480.0f));
}

}  // namespace

void ColumnPicker::Opened(const char* id) {
    if (open_id_ == id) return;
    open_id_ = id;
    search_[0] = '\0';
    cursor_ = 0;
    focus_search_ = true;
}

int ColumnPicker::DrawList(const std::vector<ColumnSummary>& columns, bool numeric_only,
                           const std::vector<std::string>& chosen, const std::string& current, bool many,
                           std::vector<int>* matches_out) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (focus_search_) {
        ImGui::SetKeyboardFocusHere();
        focus_search_ = false;
    }
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::InputTextWithHint("##search", ICON_FA_MAGNIFYING_GLASS "  Search columns", search_, sizeof(search_))) cursor_ = 0;

    // The columns that fit this field, then those matching the search.
    const std::string needle = Lower(search_);
    std::vector<int> matches;
    size_t eligible = 0, one_value = 0;
    for (size_t i = 0; i < columns.size(); ++i) {
        const ColumnSummary& c = columns[i];
        if (numeric_only && !c.numeric) continue;
        ++eligible;
        if (c.OneValue()) {
            ++one_value;
            if (hide_one_value_) continue;
        }
        if (!needle.empty() && Lower(c.name.c_str()).find(needle) == std::string::npos) continue;
        matches.push_back(static_cast<int>(i));
    }
    if (one_value > 0) {
        const std::string hide = "Hide columns with one value (" + std::to_string(one_value) + ")";
        if (ImGui::Checkbox(hide.c_str(), &hide_one_value_)) cursor_ = 0;
        ImGui::SameLine();
    }
    const std::string count = std::to_string(matches.size()) + " of " + std::to_string(eligible);
    ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(), ImGui::GetWindowContentRegionMax().x - ImGui::CalcTextSize(count.c_str()).x));
    ImGui::TextColored(t.text_dim, "%s", count.c_str());

    // Keys: Up / Down move the highlight, Enter picks it.
    bool moved = false;
    if (!matches.empty()) {
        if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) {
            cursor_ = std::min(cursor_ + 1, static_cast<int>(matches.size()) - 1);
            moved = true;
        }
        if (ImGui::IsKeyPressed(ImGuiKey_UpArrow)) {
            cursor_ = std::max(cursor_ - 1, 0);
            moved = true;
        }
    }
    cursor_ = std::clamp(cursor_, 0, std::max(0, static_cast<int>(matches.size()) - 1));
    int hit = -1;
    if (!matches.empty() && (ImGui::IsKeyPressed(ImGuiKey_Enter) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter))) {
        hit = matches[static_cast<size_t>(cursor_)];
        focus_search_ = many;
    }

    const float row_h = ImGui::GetTextLineHeightWithSpacing();
    const float list_h = std::min(320.0f, static_cast<float>(std::max<size_t>(1, matches.size())) * row_h + 4.0f);
    ImGui::BeginChild("##columns", ImVec2(0, list_h));
    if (matches.empty()) ImGui::TextColored(t.text_dim, "No column matches.");
    ImGuiListClipper clip;
    clip.Begin(static_cast<int>(matches.size()), row_h);
    if (moved) clip.IncludeItemByIndex(cursor_);
    while (clip.Step()) {
        for (int k = clip.DisplayStart; k < clip.DisplayEnd; ++k) {
            const ColumnSummary& c = columns[static_cast<size_t>(matches[static_cast<size_t>(k)])];
            const bool picked = many ? Contains(chosen, c.name) : c.name == current;
            ImGui::PushID(matches[static_cast<size_t>(k)]);
            const float x0 = ImGui::GetCursorPosX();
            const float avail = ImGui::GetContentRegionAvail().x;
            if (ImGui::Selectable("##row", k == cursor_ || (!many && picked), ImGuiSelectableFlags_NoAutoClosePopups,
                                  ImVec2(avail, 0))) {
                hit = matches[static_cast<size_t>(k)];
                cursor_ = k;
            }
            if (moved && k == cursor_) ImGui::SetScrollHereY();
            float x = x0 + 4.0f;
            if (many) {
                ImGui::SameLine(x);
                ImGui::TextColored(picked ? t.accent : ui::WithAlpha(t.text, 0.0f), ICON_FA_CHECK);
                x += 20.0f;
            }
            ImGui::SameLine(x);
            ImGui::TextColored(t.text_dim, "%s", c.numeric ? "#" : "Aa");
            ImGui::SameLine(x + 24.0f);
            ImGui::TextUnformatted(c.name.c_str());
            const std::string stats = c.Text();
            if (!stats.empty()) {
                const float w = ImGui::CalcTextSize(stats.c_str()).x;
                ImGui::SameLine(std::max(x0 + avail - w - 4.0f, ImGui::GetCursorPosX() + 12.0f));
                ImGui::TextColored(t.text_dim, "%s", stats.c_str());
            }
            ImGui::PopID();
        }
    }
    ImGui::EndChild();
    ImGui::TextColored(t.text_faint, "%s", many ? "Up / Down to move, Enter to add or remove. # number, Aa text."
                                                : "Up / Down to move, Enter to pick. # number, Aa text.");
    // Esc closes the picker at once (the search box would take the first).
    if (ImGui::IsKeyPressed(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
    if (matches_out) *matches_out = std::move(matches);
    return hit;
}

bool ColumnPicker::Pick(const char* id, std::string& value, const std::vector<ColumnSummary>& columns, bool numeric_only,
                        const char* none, float width) {
    bool changed = false;
    PickerSize(width);
    const std::string preview = value.empty() ? std::string(none ? none : "(choose)") : value;
    if (ImGui::BeginCombo(id, preview.c_str(), ImGuiComboFlags_HeightLargest)) {
        if (ImGui::IsWindowAppearing()) Opened(id);
        if (none && ImGui::Selectable(none, value.empty())) {
            value.clear();
            changed = true;
        }
        const int hit = DrawList(columns, numeric_only, {}, value, false, nullptr);
        if (hit >= 0) {
            changed = value != columns[static_cast<size_t>(hit)].name;
            value = columns[static_cast<size_t>(hit)].name;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndCombo();
    } else if (open_id_ == id) {
        open_id_.clear();
    }
    return changed;
}

bool ColumnPicker::PickMany(const char* id, std::vector<std::string>& values, const std::vector<ColumnSummary>& columns,
                            bool numeric_only, float width) {
    const ui::Tokens& t = ui::CurrentTokens();
    bool changed = false;
    std::string preview = "(choose)";
    if (!values.empty()) {
        preview = values[0];
        if (values.size() > 1) preview += ", " + values[1];
        if (values.size() > 2) preview += " +" + std::to_string(values.size() - 2);
    }
    PickerSize(width);
    if (ImGui::BeginCombo(id, preview.c_str(), ImGuiComboFlags_HeightLargest)) {
        if (ImGui::IsWindowAppearing()) Opened(id);
        std::vector<int> matches;
        const int hit = DrawList(columns, numeric_only, values, "", true, &matches);
        if (hit >= 0) {
            const std::string& name = columns[static_cast<size_t>(hit)].name;
            if (Contains(values, name)) values.erase(std::remove(values.begin(), values.end(), name), values.end());
            else values.push_back(name);
            changed = true;
        }
        const std::string add_all = matches.size() == 1 ? std::string("Add 1 match")
                                                        : "Add all " + std::to_string(matches.size()) + " matches";
        if (ui::LinkButton(add_all.c_str(), !matches.empty())) {
            for (int m : matches)
                if (!Contains(values, columns[static_cast<size_t>(m)].name)) values.push_back(columns[static_cast<size_t>(m)].name);
            changed = true;
        }
        ImGui::SameLine();
        if (ui::LinkButton("Clear##all", !values.empty())) {
            values.clear();
            changed = true;
        }
        // A range of columns in table order ("pixel400" to "pixel409").
        ImGui::TextColored(t.text_dim, "Range");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(110.0f);
        ImGui::InputTextWithHint("##from", "first column", range_from_, sizeof(range_from_));
        ImGui::SameLine();
        ImGui::TextColored(t.text_dim, "to");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(110.0f);
        ImGui::InputTextWithHint("##to", "last column", range_to_, sizeof(range_to_));
        int a = FindColumn(columns, range_from_), b = FindColumn(columns, range_to_);
        if (a > b) std::swap(a, b);
        std::vector<std::string> in_range;
        if (a >= 0 && b >= 0)
            for (int i = a; i <= b; ++i)
                if (!numeric_only || columns[static_cast<size_t>(i)].numeric) in_range.push_back(columns[static_cast<size_t>(i)].name);
        ImGui::SameLine();
        const std::string add = "Add " + std::to_string(in_range.size()) + "##range";
        if (ui::SecondaryButton(add.c_str(), !in_range.empty(), "Type the first and last column of the range")) {
            for (const auto& name : in_range)
                if (!Contains(values, name)) values.push_back(name);
            changed = true;
        }
        ImGui::EndCombo();
    } else if (open_id_ == id) {
        open_id_.clear();
    }

    // Chips under the field: the first three (click to remove), then the rest.
    if (!values.empty()) {
        const float right = ImGui::GetCursorPosX() + width;
        const size_t shown = std::min<size_t>(3, values.size());
        for (size_t i = 0; i < shown; ++i) {
            const std::string chip_id = std::string(id) + "_chip" + std::to_string(i);
            if (i > 0) {
                ImGui::SameLine(0.0f, 4.0f);
                if (ImGui::GetCursorPosX() + ui::FilterChipWidth(values[i].c_str(), ICON_FA_XMARK) > right) ImGui::NewLine();
            }
            if (ui::FilterChip(chip_id.c_str(), values[i].c_str(), ICON_FA_XMARK, false)) {
                values.erase(values.begin() + static_cast<std::ptrdiff_t>(i));
                changed = true;
                break;
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Remove %s", values[i].c_str());
        }
        if (values.size() > shown) {
            ImGui::SameLine(0.0f, 6.0f);
            ImGui::TextColored(t.text_dim, "+%zu more", values.size() - shown);
        }
    }
    return changed;
}

}  // namespace cyxwiz::plot
