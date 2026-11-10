#include "dense_contributions.h"

#include <arrayfire.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <stdexcept>

namespace cyxwiz::contributions {

Explanation Explain(const std::vector<float>& weight_row, const std::vector<float>& x, double bias, size_t top) {
    if (weight_row.size() != x.size()) throw std::runtime_error("the weights and the input differ in length");
    Explanation e;
    e.inputs = x.size();
    e.bias = bias;
    std::vector<Term> terms(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        terms[i] = {i, weight_row[i], x[i], weight_row[i] * x[i]};
        e.sum += static_cast<double>(weight_row[i]) * static_cast<double>(x[i]);
    }
    e.value = e.sum + e.bias;
    top = std::min(top, terms.size());
    std::partial_sort(terms.begin(), terms.begin() + static_cast<std::ptrdiff_t>(top), terms.end(),
                      [](const Term& a, const Term& b) {
                          const float fa = std::fabs(a.wx), fb = std::fabs(b.wx);
                          return fa > fb || (fa == fb && a.input < b.input);
                      });
    e.top.assign(terms.begin(), terms.begin() + static_cast<std::ptrdiff_t>(top));
    double shown = 0.0;
    for (const auto& t : e.top) shown += static_cast<double>(t.w) * static_cast<double>(t.x);
    e.rest = e.inputs - top;
    e.rest_sum = e.sum - shown;
    return e;
}

std::vector<float> MatrixRow(const Tensor& matrix, size_t row) {
    const auto& shape = matrix.Shape();
    if (shape.size() != 2 || row >= shape[0]) throw std::runtime_error("not a row of this matrix");
    // Row-major [rows, columns]: column-major (columns, rows); row r is column r.
    const af::array grid = af::moddims(matrix.GetRowMajorFlatArray().as(f32), static_cast<dim_t>(shape[1]),
                                       static_cast<dim_t>(shape[0]));
    const af::array one = grid(af::span, static_cast<int>(row));
    std::vector<float> values(shape[1]);
    if (!values.empty()) one.host(values.data());
    return values;
}

std::string InputPlace(size_t index, const std::vector<size_t>& source_shape) {
    if (source_shape.size() != 3) return {};
    const size_t h = source_shape[0], w = source_shape[1], c = source_shape[2];
    if (index >= h * w * c) return {};
    return "row " + std::to_string(index / (w * c)) + ", column " + std::to_string((index / c) % w) + ", channel " +
           std::to_string(index % c);
}

std::string Signed(double value) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%+.4f", value);
    return std::fabs(value) < 0.00005 ? std::string("0.0000") : std::string(buffer);
}

}  // namespace cyxwiz::contributions
