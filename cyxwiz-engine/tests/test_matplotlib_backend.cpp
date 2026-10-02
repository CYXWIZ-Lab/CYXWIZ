// MatplotlibBackend safety (TOFIX134 P0 item 2): the plot commands must not
// overwrite the user's names in __main__, titles and labels are data (quotes,
// backslashes and newlines cannot break or inject code), numbers keep their
// digits, and figures are closed. Needs matplotlib: set
// CYXWIZ_TEST_SITE_PACKAGES to a site-packages folder that has it (same
// Python minor version); without it the test reports a skip.
#include "plotting/backends/matplotlib_backend.h"

#include <pybind11/embed.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace py = pybind11;

namespace {
int failures = 0;
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}
}  // namespace

int main() {
    py::scoped_interpreter interpreter;
    if (const char* site = std::getenv("CYXWIZ_TEST_SITE_PACKAGES"))
        py::module_::import("sys").attr("path").attr("insert")(0, site);
    try {
        py::module_::import("matplotlib");
        py::module_::import("numpy");
    } catch (const py::error_already_set&) {
        std::cout << "matplotlib backend: SKIPPED (no matplotlib; set CYXWIZ_TEST_SITE_PACKAGES)\n";
        return 0;
    }
    py::dict main = py::module_::import("__main__").attr("__dict__");
    main["x"] = "user x";
    main["fig"] = "user fig";
    main["data"] = 42;

    const auto file = std::filesystem::temp_directory_path() / "cyxwiz_matplotlib_backend_test.png";
    std::filesystem::remove(file);
    {
        py::gil_scoped_release release;  // the backend takes the GIL itself, like in the Engine
        cyxwiz::plotting::MatplotlibBackend backend;
        Check(backend.Initialize(400, 300), "initialize");
        // A plain plot first: the names it uses (x, y, fig, ax) must stay out
        // of __main__, and its numbers keep all their digits.
        {
            backend.BeginPlot("plain");
            const double px[] = {0.1234567890123456, 2.0};
            const double py_[] = {1.0, 2.0};
            backend.PlotLine("plain", px, py_, 2);
            backend.EndPlot();
            py::gil_scoped_acquire gil;
            py::object ax = py::module_::import("matplotlib.pyplot").attr("gca")();
            py::object line = ax.attr("get_lines")()[py::int_(0)];
            Check(py::list(line.attr("get_xdata")())[0].cast<double>() == 0.1234567890123456, "plain plot: full digits");
            py::dict m = py::module_::import("__main__").attr("__dict__");
            Check(m["x"].cast<std::string>() == "user x", "plain plot: __main__ x untouched");
        }
        backend.SetAxisLabel(0, "it's \"x\"\nsecond line");
        backend.SetAxisLabel(1, "back\\slash");
        backend.BeginPlot("O'Brien's plot'); import os; os.getcwd('");
        const double xs[] = {0.1234567890123456, 2.0, 3.0};
        const double ys[] = {1.0000000001, 2.0, 3.0};
        backend.PlotLine("l'ine", xs, ys, 3);
        backend.EndPlot();
        Check(backend.SaveToFile(file.string().c_str()), "save");
        {
            py::gil_scoped_acquire gil;
            // The plot is still open (saved, not shown): its title is the whole text, as data.
            py::object plt = py::module_::import("matplotlib.pyplot");
            py::object ax = plt.attr("gca")();
            Check(ax.attr("get_title")().cast<std::string>() == "O'Brien's plot'); import os; os.getcwd('",
                  "title kept as text");
            Check(ax.attr("get_xlabel")().cast<std::string>() == "it's \"x\"\nsecond line", "x label kept");
            py::object line = ax.attr("get_lines")()[py::int_(0)];
            const double x0 = py::list(line.attr("get_xdata")())[0].cast<double>();
            Check(x0 == 0.1234567890123456, "numbers keep their digits");
        }
        backend.Shutdown();
    }
    Check(main["x"].cast<std::string>() == "user x", "__main__ x untouched");
    Check(main["fig"].cast<std::string>() == "user fig", "__main__ fig untouched");
    Check(main["data"].cast<int>() == 42, "__main__ data untouched");
    Check(std::filesystem::exists(file), "the plot was saved");
    Check(py::len(py::module_::import("matplotlib.pyplot").attr("get_fignums")()) == 0, "figures closed");
    std::filesystem::remove(file);
    if (failures) return 1;
    std::cout << "matplotlib backend: namespace, text as data, precision, figures closed. OK\n";
    return 0;
}
