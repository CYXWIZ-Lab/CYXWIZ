// A plot asked for from Python (TOFIX134 P5): the request is read, the
// columns it uses must be sent, required columns are worded like the Plot
// window, and the source line names the data and the calling script.

#include "../src/core/plot/python_plot_request.h"

#include <cstdlib>
#include <iostream>
#include <string>

using namespace cyxwiz::plot;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

std::string Request(const std::string& spec, const std::string& columns, const std::string& extra = "") {
    return R"({"spec": {"version": 1, )" + spec + R"(}, "columns": [)" + columns + "]" + extra + "}";
}
}  // namespace

int main() {
    PythonPlotRequest r;
    std::string error;

    // A scatter on two DataFrame columns, coloured by a third.
    Check(ParsePythonPlotRequest(Request(R"("kind": "scatter", "x": "artist_popularity", "y": ["track_popularity"], "color": "album_type", "title": "Popularity")",
                                         R"("artist_popularity", "track_popularity", "album_type")",
                                         R"(, "source": "df", "file": "D:/p/p5_api.py", "line": 7)"),
                                 r, &error),
          "scatter reads: " + error);
    Check(r.spec.kind == Kind::Scatter && r.spec.color_column == "album_type", "spec kept");
    Check(r.title == "Popularity" && r.columns.size() == 3 && r.line == 7, "title, columns, line");
    Check(PythonPlotSourceText(r, 8582) == "from Python \xC2\xB7 df \xC2\xB7 8,582 rows \xC2\xB7 p5_api.py line 7",
          "source text: " + PythonPlotSourceText(r, 8582));

    // A column the spec uses but the call did not send.
    Check(!ParsePythonPlotRequest(Request(R"("kind": "scatter", "x": "a", "y": ["nope"])", R"("a", "b")"), r, &error), "missing column refused");
    Check(error == "column 'nope' is not in the data (columns: a, b)", "missing column words: " + error);

    // Required columns, in the Plot window's words.
    Check(!ParsePythonPlotRequest(Request(R"("kind": "scatter", "x": "a")", R"("a")"), r, &error), "scatter without Y refused");
    Check(error == "scatter needs y values", "scatter words: " + error);
    Check(!ParsePythonPlotRequest(Request(R"("kind": "histogram")", R"("a")"), r, &error) && error == "histogram needs values",
          "histogram words: " + error);
    Check(!ParsePythonPlotRequest(Request(R"("kind": "scatter3d", "x": "a", "y": ["b"])", R"("a", "b")"), r, &error) &&
              error == "scatter 3d needs a Z column",
          "3D words: " + error);
    Check(!ParsePythonPlotRequest(Request(R"("kind": "quiver", "x": "a", "y": ["b"])", R"("a", "b")"), r, &error) &&
              error.find("arrow columns") != std::string::npos,
          "quiver words: " + error);
    Check(!ParsePythonPlotRequest(Request(R"("kind": "surface", "surface_from": "xyz", "x": "a")", R"("a")"), r, &error) &&
              error.find("surface from rows") != std::string::npos,
          "surface words: " + error);

    // A surface from grid columns needs no X / Y / Z; a bar only its categories.
    Check(ParsePythonPlotRequest(Request(R"("kind": "surface", "surface_from": "grid", "y": ["c1", "c2"])", R"("c1", "c2")"), r, &error),
          "grid surface: " + error);
    Check(ParsePythonPlotRequest(Request(R"("kind": "bar", "x": "album_type")", R"("album_type")"), r, &error) && r.title.empty(),
          "bar counts: " + error);
    Check(PythonPlotSourceText(r, 1) == "from Python \xC2\xB7 1 row", "no source, no file: " + PythonPlotSourceText(r, 1));
    r.line = 11;  // run from the editor or the Console: no file, the line
    Check(PythonPlotSourceText(r, 2) == "from Python \xC2\xB7 2 rows \xC2\xB7 line 11", "line only: " + PythonPlotSourceText(r, 2));

    // Not a request, a bad spec.
    Check(!ParsePythonPlotRequest("[]", r, &error) && error == "not a plot request", "not a request");
    Check(!ParsePythonPlotRequest(Request(R"("kind": "pyramid")", ""), r, &error) && error == "unknown plot kind 'pyramid'",
          "unknown kind: " + error);

    std::cout << "test_python_plot_request: all checks passed\n";
    return 0;
}
