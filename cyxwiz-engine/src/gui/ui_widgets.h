#pragma once

// Shared building blocks for CyxWiz screens (Engine and Installer), promoted
// from the reference screens of tofix119 C and tofix121 (TOFIX129 step 0.5).
// Every helper reads the tokens (ui_tokens.h) and never types a colour.
//
//   SectionHeader   title with a muted subtitle on one line
//   BeginCard/EndCard   one bordered card per real-world object
//   KeyValueTable   the Details block: two pairs per line, wrapped values
//   StatusText/StatusChip/StatusLegend   the shared status vocabulary
//   Tooltip/HelpMarker   delayed tooltip, also on disabled items
//   SearchField     text field with a magnifier hint
//   FlowRow         toolbar items that wrap to the next line when they do not fit
//   EmptyState      what to do next when there is nothing to show
//   BeginDialog/EndDialog   sized, centred modal with a scrolling body and fixed footer
//   ConfirmDialog/MessageDialog   one-call modals with the shared button order

#include "ui_tokens.h"

#include <imgui.h>

#include <cstddef>
#include <initializer_list>
#include <string>
#include <vector>

namespace cyxwiz::ui {

void SectionHeader(const char* title, const char* subtitle = nullptr);

// A card is a bordered child with window padding that grows with its
// content. Always pair with EndCard. `id` must be a stable identity (a
// device name, a pack id), never a loop index.
void BeginCard(const char* id);
void EndCard();
// Header line inside a card: title, muted subtitle, optional status at the right.
void CardHeader(const char* title, const char* subtitle = nullptr, const StatusStyle* status = nullptr);

struct KeyValue {
    std::string key;
    std::string value;
};
// Skips rows with an empty value. `pairs_per_line` is 1 or 2. With
// `copy_on_click`, clicking a value copies it and says so.
void KeyValueTable(const char* id, const std::vector<KeyValue>& rows, int pairs_per_line = 2,
                   bool copy_on_click = false);

// Icon and word in the status colour. `text` replaces the standard word.
void StatusText(Status status, const char* text = nullptr);
// Tinted pill with a dot and the word; `text` replaces the standard word.
void StatusChip(Status status, const char* text = nullptr);
// "Status:" followed by the icon and word of each status, muted.
void StatusLegend(std::initializer_list<Status> statuses);

// Shown after the usual delay, also over disabled items, wrapped at 420 px.
void Tooltip(const char* text);
// A muted "(?)" that shows `text` on hover.
void HelpMarker(const char* text);

// Returns true when the text changed. `width` 0 = fill the line.
bool SearchField(const char* id, char* buffer, size_t size, const char* hint = "Search", float width = 0.0f);

// Toolbar items that wrap: call Next(item_width) before drawing each item;
// the item stays on the current line when it fits, else starts a new one.
struct FlowRow {
    explicit FlowRow(float right_edge = 0.0f);
    bool Next(float item_width);  // returns true when a new line was started
    float right;
    bool first = true;
};

// Centred empty state: icon (may be empty), title, muted hint.
void EmptyState(const char* icon, const char* title, const char* hint = nullptr);

struct DialogOptions {
    ImVec2 size = ImVec2(560.0f, 400.0f);      // first size, clamped to 92% of the work area
    ImVec2 min_size = ImVec2(420.0f, 240.0f);
    bool resizable = true;
    bool* open = nullptr;                      // title-bar close box when set
};
enum class DialogResult { None, Primary, Secondary };

// Call ImGui::OpenPopup(title) once, then every frame:
//   if (BeginDialog(title, options)) { ...body... ; result = EndDialog("Save"); }
// The body scrolls above a fixed footer with Primary then Secondary.
bool BeginDialog(const char* title, const DialogOptions& options = {});
DialogResult EndDialog(const char* primary_label, const char* secondary_label = "Cancel",
                       bool primary_enabled = true, const char* primary_disabled_reason = nullptr,
                       bool danger = false);

// One-call confirmation: wrapped message, optional bullets, then the action
// (Primary, or Danger when `danger`) and Cancel. Call OpenPopup(title) once.
DialogResult ConfirmDialog(const char* title, const char* message, const char* confirm_label,
                           bool danger = false, const std::vector<std::string>& bullets = {});
// One-call notice with an OK button. Returns true when dismissed.
bool MessageDialog(const char* title, const char* message);

}  // namespace cyxwiz::ui
