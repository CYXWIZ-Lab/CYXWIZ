#pragma once

// CyxWiz button styles for Engine and Installer screens. Use these instead of
// ImGui::Button / ImGui::SmallButton so actions share one look: filled primary,
// bordered secondary, text link, and danger. Each returns true when clicked
// while enabled; a disabled button still shows `disabled_reason` on hover.

namespace cyxwiz::ui {

enum class ButtonSize { Small, Regular };

bool PrimaryButton(const char* label, bool enabled = true,
                   const char* disabled_reason = nullptr,
                   ButtonSize size = ButtonSize::Regular, float width = 0.0f);
bool SecondaryButton(const char* label, bool enabled = true,
                     const char* disabled_reason = nullptr,
                     ButtonSize size = ButtonSize::Small, float width = 0.0f);
bool LinkButton(const char* label, bool enabled = true);
bool DangerButton(const char* label, bool enabled = true,
                  const char* disabled_reason = nullptr,
                  ButtonSize size = ButtonSize::Small);

// Pill toggle with a label in `accent_rgba` (0xAABBGGRR, as ImU32) and a
// muted count; dimmed when off. Returns true when clicked. `id` keeps the
// widget identity stable while the count text changes.
bool ToggleChip(const char* id, const char* label, const char* count, bool on,
                unsigned int accent_rgba, bool emphasise = false);
// Pill-shaped action (quick commands): monospace-friendly label, bordered.
bool ChipButton(const char* label);
float ChipButtonWidth(const char* label);

// Segmented choice (one of several options side by side). Returns true when
// the selection changed; `selected` is updated.
bool SegmentedControl(const char* id, const char* const* labels, int count, int* selected);

// Width a ToggleChip will take.
float ToggleChipWidth(const char* label, const char* count);

// Width a button will take, for sizing fixed table columns.
float ButtonWidth(const char* label, ButtonSize size);

}  // namespace cyxwiz::ui
