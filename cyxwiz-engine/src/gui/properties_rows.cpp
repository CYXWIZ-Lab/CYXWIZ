#include "properties_rows.h"

#include "icons.h"
#include "ui_tokens.h"
#include "ui_widgets.h"

#include <imgui.h>

#include <algorithm>
#include <cfloat>
#include <string>

namespace gui::properties_rows {

namespace {

using cyxwiz::properties_view::ChipKind;
using cyxwiz::properties_view::Truth;

struct Frame {
    const cyxwiz::properties_view::View* view = nullptr;
    std::set<std::string>* shown = nullptr;
    bool details = false;
    bool narrow = false;  // the panel is too narrow for chips beside a value
};
Frame g_frame;

// Below this the rows give the value the room: a smaller label share and the
// status as a coloured mark (its text on hover) instead of a chip. A chip and
// an 84 px label left a name one letter per line in a 200 px panel.
constexpr float kNarrowPanel = 320.0f;

// The label column: a share of the panel, never narrower than a short label.
float LabelWidth() {
    const float avail = ImGui::GetContentRegionAvail().x;
    if (g_frame.narrow) return std::clamp(avail * 0.30f, 52.0f, 84.0f);
    return std::clamp(avail * 0.38f, 84.0f, 220.0f);
}

// The status column fits the widest chip this view will show (a mark when narrow).
float StatusWidth() {
    float width = ImGui::CalcTextSize(ICON_FA_CIRCLE_CHECK).x;
    if (!g_frame.view || g_frame.narrow) return width;
    for (const auto& truth : g_frame.view->truths) {
        if (!truth.chips.empty() && truth.chips.front().kind != ChipKind::Ok)
            width = std::max(width, cyxwiz::ui::ChipWidth(truth.chips.front().text.c_str()));
    }
    return width;
}

void TruthTooltip(const Truth& truth) {
    if (!ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) return;
    const auto& t = cyxwiz::ui::CurrentTokens();
    ImGui::BeginTooltip();
    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 30.0f);
    for (const auto& chip : truth.chips) {
        ImGui::TextColored(ChipColour(chip.kind), "%s", chip.text.c_str());
        ImGui::SameLine();
    }
    ImGui::NewLine();
    ImGui::TextColored(t.text_dim, "%s", truth.provenance.c_str());
    for (const auto& [key, value] : truth.aliases) ImGui::TextColored(t.text_dim, "Alias %s = %s", key.c_str(), value.c_str());
    if (!truth.message.empty()) ImGui::TextUnformatted(truth.message.c_str());
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
}

// A path or a name breaks after its separators, not letter by letter
// (ImGui wraps at spaces only, so a long name in a narrow column split
// mid-word). A part wider than the column still wraps.
void BreakableText(const std::string& path, const ImVec4& colour) {
    const float right = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
    size_t start = 0;
    while (start < path.size()) {
        const size_t separator = path.find_first_of("/\\_-.", start);
        const size_t end = separator == std::string::npos ? path.size() : separator + 1;
        const std::string part = path.substr(start, end - start);
        if (start > 0) {
            ImGui::SameLine(0.0f, 0.0f);
            if (ImGui::GetCursorScreenPos().x + ImGui::CalcTextSize(part.c_str()).x > right) ImGui::NewLine();
        }
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(colour, "%s", part.c_str());
        ImGui::PopTextWrapPos();
        start = end;
    }
}

// One run of characters a line break can only split mid-word: a path or a
// name such as sft_train_windows.parquet.
bool Breakable(const std::string& value) {
    return value.size() > 1 && value.find_first_of("/\\_-.") != std::string::npos &&
           value.find_first_of(" \n") == std::string::npos;
}

void DetailsRow(const Truth& truth) {
    const auto& t = cyxwiz::ui::CurrentTokens();
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    // Spans the editor and status columns.
    ImGui::TableSetColumnIndex(1);
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_dim, "%s", truth.provenance.c_str());
    for (const auto& [key, value] : truth.aliases) ImGui::TextColored(t.text_dim, "Alias %s = %s", key.c_str(), value.c_str());
    if (!truth.message.empty()) ImGui::TextColored(t.text_dim, "%s", truth.message.c_str());
    ImGui::PopTextWrapPos();
}

}  // namespace

void BeginFrame(const cyxwiz::properties_view::View* view, std::set<std::string>* shown_keys, bool show_details) {
    g_frame.view = view;
    g_frame.shown = shown_keys;
    g_frame.details = show_details;
}

Rows::Rows(const char* id) {
    const auto& t = cyxwiz::ui::CurrentTokens();
    g_frame.narrow = ImGui::GetContentRegionAvail().x < kNarrowPanel;
    ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(t.space_sm, t.space_xs));
    ok = ImGui::BeginTable(id, 3, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_NoPadOuterX);
    if (ok) {
        ImGui::TableSetupColumn("##label", ImGuiTableColumnFlags_WidthFixed, LabelWidth());
        ImGui::TableSetupColumn("##editor", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("##status", ImGuiTableColumnFlags_WidthFixed, StatusWidth());
    }
}

Rows::~Rows() {
    if (ok) ImGui::EndTable();
    ImGui::PopStyleVar();
}

void Label(const char* label, bool required, const char* tooltip) {
    const auto& t = cyxwiz::ui::CurrentTokens();
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::AlignTextToFramePadding();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(label);
    ImGui::PopTextWrapPos();
    if (required) {
        ImGui::SameLine(0.0f, 2.0f);
        ImGui::TextColored(t.warning, "*");
    }
    if (tooltip && tooltip[0]) cyxwiz::ui::Tooltip(tooltip);
    ImGui::TableSetColumnIndex(1);
    ImGui::SetNextItemWidth(-FLT_MIN);
}

void Note(const char* text) {
    if (!text || !text[0]) return;
    const auto& t = cyxwiz::ui::CurrentTokens();
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(1);
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_dim, "%s", text);
    ImGui::PopTextWrapPos();
}

void Status(const char* key) {
    if (!g_frame.view) return;
    const Truth* truth = cyxwiz::properties_view::FindTruth(g_frame.view->truths, key ? key : "");
    if (!truth) return;
    if (g_frame.shown) g_frame.shown->insert(truth->canonical_key);
    ImGui::TableSetColumnIndex(2);
    if (!truth->chips.empty()) {
        // The first status names the row; the rest are in the hover text.
        const auto& chip = truth->chips.front();
        if (chip.kind == ChipKind::Ok) {
            ImGui::AlignTextToFramePadding();
            ImGui::TextColored(ChipColour(ChipKind::Ok), "%s", ICON_FA_CIRCLE_CHECK);
        } else if (g_frame.narrow) {
            ImGui::AlignTextToFramePadding();
            ImGui::TextColored(ChipColour(chip.kind), "%s", ICON_FA_CIRCLE);
        } else {
            Chip(chip);
        }
        TruthTooltip(*truth);
    }
    if (g_frame.details) DetailsRow(*truth);
}

void ReadOnly(const char* label, const std::string& value, const char* key) {
    const auto& t = cyxwiz::ui::CurrentTokens();
    Label(label);
    ImGui::AlignTextToFramePadding();
    Value(value.empty() ? "(not set)" : value, t.text_bright);
    Status(key);
}

void Value(const std::string& text, const ImVec4& colour) {
    if (Breakable(text)) {
        BreakableText(text, colour);
        return;
    }
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(colour, "%s", text.c_str());
    ImGui::PopTextWrapPos();
}

bool Disclosure(const char* label, bool open) {
    ImGui::SetNextItemOpen(open, ImGuiCond_Always);
    return ImGui::TreeNodeEx(label, ImGuiTreeNodeFlags_NoTreePushOnOpen | ImGuiTreeNodeFlags_NoAutoOpenOnLog);
}

const ImVec4& ChipColour(ChipKind kind) {
    const auto& t = cyxwiz::ui::CurrentTokens();
    switch (kind) {
        case ChipKind::Ok: return t.success;
        case ChipKind::Dialog: return t.pending;
        case ChipKind::Default:
        case ChipKind::Alias:
        case ChipKind::Stale:
        case ChipKind::Planned:
        case ChipKind::External: return t.warning;
        case ChipKind::Missing:
        case ChipKind::Conflict:
        case ChipKind::Unsupported:
        case ChipKind::Deprecated: return t.error;
        case ChipKind::Info: return t.info;
    }
    return t.info;
}

void Chip(const cyxwiz::properties_view::Chip& chip) {
    cyxwiz::ui::Chip(chip.text.c_str(), ChipColour(chip.kind));
}

}  // namespace gui::properties_rows
