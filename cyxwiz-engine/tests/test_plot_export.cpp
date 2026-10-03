// Plot exports (TOFIX134 P1 step 1.4): CSV with all values (also when the
// drawing was reduced), quoting, and SVG for every kind. When
// CYXWIZ_TEST_PYTHON names a Python, each SVG is parsed as XML by it.
#include "../src/core/plot/plot_export.h"
#include "../src/core/plot/plot_prepare.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace cyxwiz::plot;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

size_t Lines(const std::string& s) {
    size_t n = 0;
    for (char c : s) n += c == '\n';
    return n;
}

size_t Count(const std::string& s, const std::string& what) {
    size_t n = 0;
    for (size_t at = s.find(what); at != std::string::npos; at = s.find(what, at + what.size())) ++n;
    return n;
}

void ParsesAsXml(const std::string& svg, const std::string& what) {
    const char* python = std::getenv("CYXWIZ_TEST_PYTHON");
    if (!python) return;
    const auto file = std::filesystem::temp_directory_path() / "cyxwiz_plot_export_test.svg";
    std::ofstream(file, std::ios::binary) << svg;
    const std::string cmd = std::string("\"\"") + python +
                            "\" -c \"import sys, xml.etree.ElementTree as E; E.parse(sys.argv[1])\" \"" + file.string() + "\"\"";
    Check(std::system(cmd.c_str()) == 0, what + ": the SVG is not well-formed XML");
    std::filesystem::remove(file);
}

SourceColumn Numbers(const std::string& name, std::vector<double> v) {
    SourceColumn c;
    c.name = name;
    c.numbers = std::move(v);
    return c;
}

SourceColumn Text(const std::string& name, std::vector<std::string> v) {
    SourceColumn c;
    c.name = name;
    c.numeric = false;
    c.text = std::move(v);
    return c;
}
}  // namespace

int main() {
    // A long line: the drawing is reduced, the CSV has every row.
    Source src;
    std::vector<double> x(10000), y(10000);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<double>(i);
        y[i] = std::sin(static_cast<double>(i) / 100.0);
    }
    src.columns.push_back(Numbers("step", x));
    src.columns.push_back(Numbers("loss", y));
    src.columns.push_back(Text("label", std::vector<std::string>(10000, "a, \"quoted\"")));
    PlotSpec line;
    line.kind = Kind::Line;
    line.x_column = "step";
    line.y_columns = {"loss"};
    line.title = "loss <by> step & more";
    Prepared p = Prepare(line, src);
    Check(p.label.state == DataLabel::State::Reduced, "the line is reduced for drawing");
    const std::string csv = ToCsv(p);
    Check(Lines(csv) == 10001 && csv.rfind("series,x,y\n", 0) == 0, "CSV: header and all 10,000 rows");
    Check(csv.find("loss,9999,") != std::string::npos, "CSV: the last row");

    // Quoting of names with commas and quotes.
    PlotSpec bar;
    bar.kind = Kind::Bar;
    bar.x_column = "label";
    const std::string bar_csv = ToCsv(Prepare(bar, src));
    Check(bar_csv.find("\"a, \"\"quoted\"\"\",10000") != std::string::npos, "CSV: a name with a comma and quotes is quoted");

    // SVG: escaped title, a polyline, closed document.
    AxisRange range{0, 10000, -1.1, 1.1, false};
    std::string svg = ToSvg(p, range, SvgStyle{});
    Check(svg.rfind("<svg ", 0) == 0 && svg.find("</svg>") != std::string::npos, "SVG document");
    Check(svg.find("loss &lt;by&gt; step &amp; more") != std::string::npos, "SVG: title escaped");
    Check(Count(svg, "<polyline") == 1, "SVG: one line");
    ParsesAsXml(svg, "line");

    // Histogram: one rect per bin.
    PlotSpec hist;
    hist.kind = Kind::Histogram;
    hist.x_column = "loss";
    hist.bins = 25;
    const Prepared hp = Prepare(hist, src);
    svg = ToSvg(hp, AxisRange{-1, 1, 0, 600, false}, SvgStyle{});
    Check(Count(svg, "<rect x=") == 25 + 1, "SVG histogram: 25 bars (and the clip area)");
    Check(Lines(ToCsv(hp)) == 26, "CSV histogram: header and 25 bins");
    ParsesAsXml(svg, "histogram");

    // Every kind writes a document (and parses when Python is there).
    Source cat;
    cat.columns.push_back(Text("pred", {"cat", "dog", "cat", "cat", "bird"}));
    cat.columns.push_back(Text("truth", {"cat", "dog", "dog", "cat", "bird"}));
    cat.columns.push_back(Numbers("v", {1, 2, 3, 4, 5}));
    cat.columns.push_back(Numbers("w", {2, 1, 4, 3, 5}));
    cat.columns.push_back(Numbers("prob", {0.1, 0.9, 0.4, 0.2, 0.7}));
    cat.columns.push_back(Text("country", {"France", "Spain", "Peru", "Chile", "Japan"}));
    for (const auto& k : Kinds()) {
        PlotSpec s;
        s.kind = k.kind;
        const bool categories = k.kind == Kind::Bar || k.kind == Kind::Pie || k.kind == Kind::ErrorBars || k.kind == Kind::Heatmap;
        if (k.required & kEncX) s.x_column = categories ? "pred" : "v";
        if (k.required & kEncY) s.y_columns = {k.kind == Kind::Heatmap ? "truth" : "w"};
        if (k.kind == Kind::ErrorBars) s.y_columns = {"v"};
        // Model results: actual labels against predictions, scores and probabilities.
        if (k.kind == Kind::Confusion || k.kind == Kind::Roc || k.kind == Kind::PrCurve || k.kind == Kind::Calibration) s.x_column = "truth";
        if (k.kind == Kind::Confusion) s.y_columns = {"pred"};
        if (k.kind == Kind::Calibration) s.y_columns = {"prob"};
        if (k.kind == Kind::Importance) s.x_column = "pred";
        if (k.kind == Kind::Sankey) s.y_columns = {"truth", "pred"};
        if (k.kind == Kind::Treemap) s.y_columns = {"truth", "pred"};
        if (k.kind == Kind::MapRegions) {
            s.x_column = "country";
            s.y_columns = {"v"};
        }
        if (k.kind == Kind::Matrix || k.kind == Kind::PairPlot || k.kind == Kind::Parallel || k.kind == Kind::Image) s.y_columns = {"v", "w"};
        // 3D: x, y and z (a surface from X, Y, Z; its cells need one row each here).
        if (Info(k.kind).group == Group::ThreeD) {
            s.x_column = "v";
            s.y_columns = {"w"};
            s.z_column = "prob";
            s.min_cell_rows = 1;
        }
        // Vector fields read a grid of arrows: a 6 x 6 rotation.
        Source field;
        if (k.required & kEncVector) {
            std::vector<double> fx, fy, fu, fv;
            for (int j = 0; j < 6; ++j)
                for (int i = 0; i < 6; ++i) {
                    fx.push_back(i);
                    fy.push_back(j);
                    fu.push_back(2.5 - j);
                    fv.push_back(i - 2.5);
                }
            field.columns.push_back(Numbers("x", fx));
            field.columns.push_back(Numbers("y", fy));
            field.columns.push_back(Numbers("u", fu));
            field.columns.push_back(Numbers("v", fv));
            s.x_column = "x";
            s.y_columns = {"y"};
            s.u_column = "u";
            s.v_column = "v";
        }
        const Prepared kp = Prepare(s, (k.required & kEncVector) ? field : cat);
        Check(kp.problem.empty(), std::string(k.id) + ": prepared (" + kp.problem + ")");
        const std::string doc = ToSvg(kp, AxisRange{0, 5, 0, 6, false}, SvgStyle{});
        Check(doc.find("</svg>") != std::string::npos, std::string(k.id) + ": SVG written");
        Check(!ToCsv(kp).empty(), std::string(k.id) + ": CSV written");
        ParsesAsXml(doc, k.id);
    }
    Check(HexColour(1.0f, 0.5f, 0.0f) == "#ff8000", "hex colour");

    // 3D: the surface CSV lists the filled cells at their centres, bottom row first.
    {
        Source grid;
        grid.columns.push_back(Numbers("V1", {1, 2}));
        grid.columns.push_back(Numbers("V2", {3, NAN}));
        PlotSpec gs;
        gs.kind = Kind::Surface;
        gs.surface_from = PlotSpec::SurfaceFrom::Grid;
        gs.y_columns = {"V1", "V2"};
        const std::string surface_csv = ToCsv(Prepare(gs, grid));
        Check(surface_csv == "column,row,value\n1,1,1\n2,1,3\n1,2,2\n", "surface CSV: " + surface_csv);
        Check(ToSvg(Prepare(gs, grid), AxisRange{}, SvgStyle{}).find("saved as PNG or CSV") != std::string::npos, "3D SVG says PNG or CSV");
    }
    // Colour picker in SVG: a picked scale and a picked series colour.
    {
        Source picked_src;
        picked_src.columns.push_back(Numbers("x", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}));
        picked_src.columns.push_back(Numbers("y", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}));
        picked_src.columns.push_back(Numbers("epoch", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}));
        PlotSpec picked_scale;
        picked_scale.kind = Kind::Scatter;
        picked_scale.x_column = "x";
        picked_scale.y_columns = {"y"};
        picked_scale.color_column = "epoch";
        picked_scale.colour_scale = "viridis";
        const std::string picked_svg = ToSvg(Prepare(picked_scale, picked_src), AxisRange{0, 13, 0, 13, false}, SvgStyle{});
        Check(picked_svg.find("#440154") != std::string::npos, "the lowest epoch takes Viridis' low end");
        PlotSpec picked_line;
        picked_line.kind = Kind::Line;
        picked_line.x_column = "x";
        picked_line.y_columns = {"y"};
        picked_line.series_colours = {"#5ec8b8"};
        Check(ToSvg(Prepare(picked_line, picked_src), AxisRange{0, 13, 0, 13, false}, SvgStyle{}).find("#5ec8b8") != std::string::npos,
              "a picked series colour");
    }

    // Colour scale (P2 board 6): the CSV keeps the colour value, the SVG
    // points take the scale's ends, a missing value the dim text colour.
    Source sc;
    sc.columns.push_back(Numbers("x", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}));
    sc.columns.push_back(Numbers("y", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}));
    sc.columns.push_back(Numbers("epoch", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, NAN}));
    PlotSpec scatter;
    scatter.kind = Kind::Scatter;
    scatter.x_column = "x";
    scatter.y_columns = {"y"};
    scatter.color_column = "epoch";
    const Prepared sp = Prepare(scatter, sc);
    Check(sp.colour_scale, "13 epoch values: a scale");
    const std::string sc_csv = ToCsv(sp);
    Check(sc_csv.rfind("series,x,y,epoch\n", 0) == 0 && sc_csv.find("y,12,12,12\n") != std::string::npos &&
              sc_csv.find("y,13,13,\n") != std::string::npos,
          "CSV: colour column, missing value empty");
    SvgStyle style;
    svg = ToSvg(sp, AxisRange{0, 13, 0, 13, false}, style);
    Check(svg.find("fill=\"" + style.scale_low + "\"") != std::string::npos &&
              svg.find("fill=\"" + style.scale_high + "\"") != std::string::npos &&
              svg.find("fill=\"" + style.text_dim + "\" fill-opacity") != std::string::npos,
          "SVG: low and high ends of the scale, missing in dim text");
    ParsesAsXml(svg, "scatter with a colour scale");
    std::cout << "plot export: CSV keeps all rows, quoting, SVG escaped and written for all 38 kinds, colour scale, 3D CSV, picked scale and series colour. OK\n";
    return 0;
}
