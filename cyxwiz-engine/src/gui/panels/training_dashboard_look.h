#pragma once

// The Training Dashboard's look (cards, pills, progress, buttons), shared by
// the model run's view and the word-vector runs' view. This file is also
// compiled into test targets that draw the dashboard; those compile
// ui_tokens.cpp too, so the look reads the shared tokens but keeps its own
// button helpers.

#include <imgui.h>

#include <string>

namespace cyxwiz::dash {

struct DashColors {
    ImVec4 text;
    ImVec4 muted;
    ImVec4 faint;
    ImVec4 window;
    ImVec4 card;
    ImVec4 track;
    ImVec4 border;
    ImVec4 accent;
    ImVec4 accent_text;
    ImVec4 success;
    ImVec4 warning;
    ImVec4 caution;
    ImVec4 error;
    ImVec4 info;
    bool light = false;
};

enum class DashButtonKind { Secondary, Danger };

ImVec4 MixColor(const ImVec4& a, const ImVec4& b, float t);
ImVec4 WithAlpha(ImVec4 color, float alpha);
DashColors CurrentDashColors();
// Card: a filled, rounded, borderless group sized to its content.
bool BeginDashCard(const char* id, const DashColors& c, float width = 0.0f);
void EndDashCard();
void DashCardTitle(const char* title, const char* subtitle, const DashColors& c);
// Status pill: tinted background, a dot and the label in the status colour.
void DashPill(const char* label, const ImVec4& color, bool dot = true);
// Thin rounded progress bar.
void DashProgress(float fraction, const ImVec4& color, const DashColors& c,
                  float width = -1.0f, float height = 6.0f);
bool DashButton(const char* label, DashButtonKind kind, const DashColors& c);
float DashButtonWidth(const char* label);
// Toggle chip: filled with the accent tint while on.
bool DashChip(const char* label, bool* value, const DashColors& c);
// Bold UI font for metric values (falls back to the current font).
ImFont* DashValueFont();
void DashBigValue(const char* text, const ImVec4& color);
std::string FormatEpochValue(double epoch);
// Small borderless icon button, tinted with the accent while active.
bool DashIconButton(const char* id_label, bool active, const DashColors& c);
// KPI card: a dot and label, the big value (with a hover tooltip) and two
// lines below it.
void DashKpiCard(const char* id, const char* label, const ImVec4& dot, const char* value, const char* tooltip,
                 const char* line1, const ImVec4& line1_color, const char* line2, const ImVec4& line2_color,
                 const DashColors& c);
// Run tab of the Run switcher: a dot, the label and a faint second part; the
// shown run is raised with the accent line under it.
bool DashRunTab(const char* id, const char* label, const char* sub, const ImVec4& dot, bool active,
                const DashColors& c);

}  // namespace cyxwiz::dash
