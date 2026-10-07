#pragma once

struct ImGuiViewport;

namespace cyxwiz::plot {

// Sets the plot view hooks (PNG capture, save dialog, clipboard image).
// Called once by the application after ImGui is set up.
void InstallPlotViewHooks();

// Reads the plot images requested this frame from the window just drawn
// (the main window, or a window of its own: TOFIX129 A8). Called by the
// application after the main window is rendered and before its buffers are
// swapped; the other windows call it through the renderer hook below.
void CompletePlotCaptures(ImGuiViewport* viewport);

// Once, after the renderer backend is set up: plots in windows of their own
// are captured too.
void CapturePlotsInSeparateWindows();

}  // namespace cyxwiz::plot
