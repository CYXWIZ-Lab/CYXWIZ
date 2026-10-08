#pragma once

// The Properties panel's settings rows (TOFIX129 A7): one table per group,
// label column, editor column, status column. Every editor in the panel -
// the metadata-driven one and the per-node custom ones - draws its rows
// through these calls, so the truth chip of a setting appears on its row
// wherever the row is drawn, and the panel knows which truths were shown.
//
//   {
//     properties_rows::Rows rows("##training");
//     properties_rows::Label("Epochs");
//     if (ImGui::InputInt("##epochs", &epochs)) ...;
//     properties_rows::Status("epochs");
//   }

#include "../core/properties_presentation.h"

#include <set>
#include <string>

namespace gui::properties_rows {

// Called by the panel once per frame before any row: the view the truths
// come from, the set that collects the keys whose truth was drawn, and
// whether the provenance and message are shown under the rows.
void BeginFrame(const cyxwiz::properties_view::View* view,
                std::set<std::string>* shown_keys,
                bool show_details);

// One table of rows; draws nothing when the table cannot open (`ok` false).
struct Rows {
    explicit Rows(const char* id);
    ~Rows();
    Rows(const Rows&) = delete;
    Rows& operator=(const Rows&) = delete;
    bool ok = false;
};

// Starts a row: the label cell (with " *" when required and `tooltip` on
// hover), then moves to the editor cell with the next item filling it.
void Label(const char* label, bool required = false, const char* tooltip = nullptr);
// A muted note on its own full-width row (a hint under a setting).
void Note(const char* text);
// The status cell for the setting `key`: its truth chip, with the provenance
// and message on hover, and under the row when details are shown. Call after
// the editor, once per row.
void Status(const char* key);
// A read-only value row: label, value text, the status of `key`.
void ReadOnly(const char* label, const std::string& value, const char* key);

// A chip in the shared status colours.
void Chip(const cyxwiz::properties_view::Chip& chip);
const ImVec4& ChipColour(cyxwiz::properties_view::ChipKind kind);

}  // namespace gui::properties_rows
