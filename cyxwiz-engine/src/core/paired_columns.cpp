#include "paired_columns.h"

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>

namespace cyxwiz::chartdata {

bool ParseFinite(const std::string& text, double* out) {
    size_t a = 0;
    size_t b = text.size();
    while (a < b && std::isspace(static_cast<unsigned char>(text[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(text[b - 1]))) --b;
    if (a == b) return false;
    const std::string cell = text.substr(a, b - a);
    char* end = nullptr;
    errno = 0;
    const double v = std::strtod(cell.c_str(), &end);
    if (end != cell.c_str() + cell.size() || errno == ERANGE || !std::isfinite(v)) return false;
    if (out) *out = v;
    return true;
}

void PairedNumbers(const std::vector<std::vector<std::string>>& rows, int x_column, int y_column,
                   std::vector<double>* xs, std::vector<double>* ys) {
    xs->clear();
    ys->clear();
    if (x_column < 0 || y_column < 0) return;
    xs->reserve(rows.size());
    ys->reserve(rows.size());
    for (const auto& row : rows) {
        if (x_column >= static_cast<int>(row.size()) || y_column >= static_cast<int>(row.size())) continue;
        double x = 0.0;
        double y = 0.0;
        if (!ParseFinite(row[static_cast<size_t>(x_column)], &x) || !ParseFinite(row[static_cast<size_t>(y_column)], &y)) continue;
        xs->push_back(x);
        ys->push_back(y);
    }
}

}  // namespace cyxwiz::chartdata
