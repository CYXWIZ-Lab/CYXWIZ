#pragma once

namespace cyxwiz::plot {

// Sets the plot view hooks (PNG capture, save dialog, clipboard image).
// Called once by the application after ImGui is set up.
void InstallPlotViewHooks();

// Reads the plot images requested this frame. Called by the application
// after the frame is rendered and before the buffers are swapped.
void CompletePlotCaptures();

}  // namespace cyxwiz::plot
