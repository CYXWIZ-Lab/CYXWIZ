#include "plot_script.h"

#include "python_literal.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace cyxwiz::plotscript {

namespace {
void WriteList(std::ostringstream& out, const char* name, const std::vector<double>& values, size_t count) {
    out << name << " = [";
    for (size_t i = 0; i < count; ++i) {
        if (i > 0) out << (i % 10 == 0 ? ",\n    " : ", ");
        const double v = values[i];
        if (std::isfinite(v)) out << v;
        else out << (std::isnan(v) ? "float('nan')" : (v > 0 ? "float('inf')" : "float('-inf')"));
    }
    out << "]\n";
}
}  // namespace

std::string MatplotlibScript(Kind kind, const std::string& title, const std::vector<double>& x,
                             const std::vector<double>& y) {
    std::ostringstream s;
    s.precision(17);
    const bool scatter = kind == Kind::Scatter && !y.empty();
    size_t n = std::min(x.size(), kMaxValues);
    if (scatter) n = std::min(n, y.size());
    s << "import matplotlib.pyplot as plt\n";
    s << "import numpy as np\n\n";
    s << "# Data from the Table Viewer";
    const size_t total = scatter ? std::min(x.size(), y.size()) : x.size();
    if (n < total) s << ": the first " << n << " of " << total << " values";
    s << "\n";
    WriteList(s, "data", x, n);
    if (scatter) WriteList(s, "y_data", y, n);
    s << "\nplt.figure(figsize=(10, 6))\n";
    switch (kind) {
        case Kind::Histogram:
            s << "plt.hist(data, bins=30, edgecolor='black', alpha=0.7)\n";
            s << "plt.xlabel('Value')\nplt.ylabel('Frequency')\n";
            break;
        case Kind::Bar:
            s << "plt.bar(range(len(data)), data, alpha=0.7)\n";
            s << "plt.xlabel('Index')\nplt.ylabel('Value')\n";
            break;
        case Kind::Scatter:
            if (scatter) {
                s << "plt.scatter(data, y_data, alpha=0.7)\n";
                s << "plt.xlabel('X')\nplt.ylabel('Y')\n";
            } else {
                s << "plt.scatter(range(len(data)), data, alpha=0.7)\n";
            }
            break;
        case Kind::Box: s << "plt.boxplot(data)\n"; break;
        case Kind::Pie:
            s << "counts, bins = np.histogram(data, bins=8)\n";
            s << "labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(len(counts))]\n";
            s << "plt.pie(counts, labels=labels, autopct='%1.1f%%')\n";
            break;
        case Kind::Stairs: s << "plt.step(range(len(data)), data, where='mid')\n"; break;
        case Kind::Stem: s << "plt.stem(range(len(data)), data)\n"; break;
        case Kind::Area:
            s << "x = range(len(data))\nplt.fill_between(x, data, alpha=0.5)\nplt.plot(x, data)\n";
            break;
        case Kind::Line: s << "plt.plot(data)\n"; break;
    }
    s << "plt.title(" << PythonStringLiteral(title) << ")\n";
    s << "plt.tight_layout()\nplt.show()\n";
    return s.str();
}

}  // namespace cyxwiz::plotscript
