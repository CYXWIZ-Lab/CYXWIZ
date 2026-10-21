// ui_widgets.h - the Server Node's shared controls (TOFIX139 P2). Every
// colour comes from the ImGui style the theme set and from the theme's accent
// and semantic colours; screens never push colour literals of their own.
#pragma once

#include <imgui.h>

#include <cstddef>

namespace cyxwiz::servernode::gui::ui {

// The theme's accent (buttons, links, spinners).
ImVec4 Accent();
// Secondary text.
ImVec4 Muted();
// A colour mixed towards the window background; strength 0 = background, 1 = the colour.
ImVec4 Tinted(const ImVec4& color, float strength);

// The one main action of a screen or dialog: filled with the accent. A disabled
// button shows its reason on hover; a busy one shows a spinner and cannot be
// pressed. width 0 = fit the label; -1 = fill the line.
bool PrimaryButton(const char* label, float width = 0.0f, bool enabled = true, const char* disabled_reason = nullptr,
                   bool busy = false);

// A supporting action: surface fill, no outline. padding overrides the
// button's own (to sit level with a text field, pass FieldPadding()).
bool SecondaryButton(const char* label, float width = 0.0f, bool enabled = true, const ImVec2* padding = nullptr);

// The padding text fields use, for controls that sit beside one.
ImVec2 FieldPadding();

// Text in the accent colour with an underline and a hand cursor on hover.
bool LinkButton(const char* label, bool enabled = true);

// A turning arc; colour defaults to the accent.
void Spinner(const char* id, float radius, float thickness, const ImVec4* color = nullptr);

// A text field on the surface tone with no outline. Returns true when the
// flags ask for Enter and it was pressed. width -1 = fill the line.
bool TextField(const char* id, char* buf, std::size_t buf_size, const char* placeholder,
               ImGuiInputTextFlags flags = 0, bool enabled = true, float width = -1.0f);

}  // namespace cyxwiz::servernode::gui::ui
