#include "plot_style.h"

#include "../ui_tokens.h"

namespace cyxwiz::plot {

namespace {

// ImPlot colormaps cannot change once added, so the light and dark variants
// are registered under their own names the first time they are needed.
ImPlotColormap Registered(const char* name, const ImVec4* colours, int count, bool qualitative) {
    const ImPlotColormap existing = ImPlot::GetColormapIndex(name);
    return existing >= 0 ? existing : ImPlot::AddColormap(name, colours, count, qualitative);
}

}  // namespace

ImPlotColormap SeriesColormap() {
    const ui::Tokens& t = ui::CurrentTokens();
    return Registered(t.light ? "CyxWiz Series Light" : "CyxWiz Series Dark", t.series, ui::Tokens::kSeriesCount, true);
}

ImPlotColormap SequentialColormap() {
    const ui::Tokens& t = ui::CurrentTokens();
    return Registered(t.light ? "CyxWiz Sequential Light" : "CyxWiz Sequential Dark", t.scale_sequential, 4, false);
}

ImPlotColormap DivergingColormap() {
    const ui::Tokens& t = ui::CurrentTokens();
    return Registered(t.light ? "CyxWiz Diverging Light" : "CyxWiz Diverging Dark", t.scale_diverging, 3, false);
}

void ApplyPlotStyle() {
    const ui::Tokens& t = ui::CurrentTokens();
    ImPlotStyle& style = ImPlot::GetStyle();
    ImVec4* c = style.Colors;
    const ImVec4 none(0, 0, 0, 0);
    // Series colours come from the colormap; IMPLOT_AUTO_COL keeps that.
    c[ImPlotCol_Line] = IMPLOT_AUTO_COL;
    c[ImPlotCol_Fill] = IMPLOT_AUTO_COL;
    c[ImPlotCol_MarkerOutline] = IMPLOT_AUTO_COL;
    c[ImPlotCol_MarkerFill] = IMPLOT_AUTO_COL;
    c[ImPlotCol_ErrorBar] = t.text_dim;
    // One surface: no frame or borders, the plot area a shade of the window.
    c[ImPlotCol_FrameBg] = none;
    c[ImPlotCol_PlotBg] = t.plot_bg;
    c[ImPlotCol_PlotBorder] = none;
    c[ImPlotCol_LegendBg] = ui::WithAlpha(t.bg_panel, 0.92f);
    c[ImPlotCol_LegendBorder] = none;
    c[ImPlotCol_LegendText] = t.text;
    c[ImPlotCol_TitleText] = t.text_bright;
    c[ImPlotCol_InlayText] = t.text;
    c[ImPlotCol_AxisText] = t.text_dim;
    c[ImPlotCol_AxisGrid] = t.plot_grid;
    c[ImPlotCol_AxisTick] = t.plot_grid;
    c[ImPlotCol_AxisBg] = none;
    c[ImPlotCol_AxisBgHovered] = t.hover;
    c[ImPlotCol_AxisBgActive] = t.selection;
    c[ImPlotCol_Selection] = t.accent_text;
    c[ImPlotCol_Crosshairs] = t.text_dim;
    style.PlotBorderSize = 0.0f;
    style.Colormap = SeriesColormap();
}

}  // namespace cyxwiz::plot
