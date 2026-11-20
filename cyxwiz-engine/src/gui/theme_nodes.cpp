// Canvas (ImNodes) shape for the theme.
//
// No colours here (TOFIX129 A4, one colour system): the canvas draws its own
// node boxes, pins and links (node families, data types, compile state); the
// grid and mini-map follow the window (Theme::HarmonizeSurfaces); the box
// selector and the subgraph frames follow the accent (ApplyEngineColours).

#include "theme.h"

#include <imgui.h>
#include <imnodes.h>

namespace gui {

void Theme::ApplyImNodesStyle() {
    ImNodesStyle& style = ImNodes::GetStyle();
    style.NodeCornerRounding = config_.frame_rounding;
    style.NodePadding = ImVec2(8.0f, 8.0f);
    style.NodeBorderThickness = 1.0f;

    // Links
    style.LinkThickness = 3.0f;
    style.LinkLineSegmentsPerLength = 0.1f;
    style.LinkHoverDistance = 10.0f;

    // Pins
    style.PinCircleRadius = 5.0f;
    style.PinQuadSideLength = 7.0f;
    style.PinTriangleSideLength = 9.0f;
    style.PinLineThickness = 1.5f;
    style.PinHoverRadius = 10.0f;
    style.PinOffset = 0.0f;

    // Grid
    style.GridSpacing = 32.0f;
    style.Flags = ImNodesStyleFlags_GridLines | ImNodesStyleFlags_GridLinesPrimary | ImNodesStyleFlags_NodeOutline;

    // Link width: heavier in High Contrast, lighter in the editor and terminal looks.
    switch (current_preset_) {
        case ThemePreset::HighContrast:    style.LinkThickness = 4.0f; break;
        case ThemePreset::CyxOSFluent:
        case ThemePreset::CyxOSOffice:     style.LinkThickness = 2.5f; break;
        case ThemePreset::CyxOSCoder:
        case ThemePreset::CyxOSTuiClassic:
        case ThemePreset::CyxOSTuiMatrix:
        case ThemePreset::CyxOSTuiAmber:   style.LinkThickness = 2.0f; break;
        default: break;
    }
}

} // namespace gui
