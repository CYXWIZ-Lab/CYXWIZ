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

// Width a button will take, for sizing fixed table columns.
float ButtonWidth(const char* label, ButtonSize size);

}  // namespace cyxwiz::ui
